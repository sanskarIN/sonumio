"""Explicit global audio-history-conditioned original v-denoiser architecture.

Sonumio's previous clean-prefix prototype gives the denoiser clean local
samples inside its noisy future tensor. This research variant separately
encodes the ENTIRE preceding 8192-frame clip into a learnable context vector
and injects that context at every residual block, allowing distant future
frames to depend on the previous window beyond the local convolution span.

This is a new research architecture, not compatible with existing velocity
model weights/checkpoints. Future-only loss and deterministic v-DDIM sampling
must be trained together; a random model does not generate coherent sound.
Nothing in this module saves files, audio, weights or pretrained data.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.latent_audio import FRAMES
from research.longform_ddim import MAX_SEGMENTS, MAX_SIGNAL
from research.text_audio import (
    ByteTextTokenizer, TextEncoder, MAX_TOKENS,
)
from research.velocity_ddim import ddim_plan, velocity_ddim_transition
from research.velocity_prediction import velocity_target
from research.velocity_prefix_continuation import _prefix_frames, _anchor
from research.wave_diffusion import (
    SAMPLE_RATE, DiffusionSchedule, ConditionedResidual,
)
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-global-history-v-denoiser-v1"
INFERENCE_PROTOCOL = "sonumio-global-history-conditioned-ddim-v1"
SEQUENCE_PROTOCOL = "sonumio-global-history-conditioned-sequence-v1"
MAX_CONTEXT_FRAMES = FRAMES
MAX_BATCH = 8
TIME_CHANNELS = (16, 24, 32)
SOURCE_FRAMES = FRAMES


def _context_inputs(
    noisy: torch.Tensor, timestep: torch.Tensor,
    tokens: torch.Tensor, history: torch.Tensor,
    *, prefix_frames: int, check_identity: bool,
) -> None:
    """Validate before expensive neural work or any input transformation."""
    _prefix_frames(prefix_frames)
    if (not all(isinstance(x, torch.Tensor)
                for x in (noisy, timestep, tokens, history))):
        raise ValueError("History-conditioned input must use real tensors")
    if (noisy.ndim != 3 or noisy.shape[1:] != (1, FRAMES)
            or not 1 <= len(noisy) <= MAX_BATCH
            or history.shape != noisy.shape
            or timestep.shape != (len(noisy),)
            or tokens.shape != (len(noisy), MAX_TOKENS)
            or noisy.device.type != "cpu"
            or history.device.type != "cpu"
            or timestep.device.type != "cpu"
            or tokens.device.type != "cpu"
            or noisy.dtype != torch.float32
            or history.dtype != torch.float32
            or timestep.dtype != torch.long
            or tokens.dtype != torch.long):
        raise ValueError("Expected CPU float32 noisy/history [B,1,8192] and integer conditions")
    if (not torch.isfinite(noisy).all()
            or not torch.isfinite(history).all()
            or torch.max(torch.abs(noisy)).item() > MAX_SIGNAL
            or torch.max(torch.abs(history)).item() > MAX_SIGNAL):
        raise ValueError("Nonfinite or unbounded neural history/latent noise input")
    if check_identity and not torch.equal(
        noisy[..., :prefix_frames],
        history[..., -prefix_frames:],
    ):
        raise ValueError("Known noisy-input prefix must match historical audio exactly")


class HistoryEncoder(nn.Module):
    """Trainable multiscale history encoder with explicit energy statistics.

    The encoder compresses the whole prior 8192-frame clip into a vector.
    Its output conditions ALL current future frames. It does not claim to
    be a perceptual audio embedding or music/speech representation.
    """

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        if type(channels) is not int or channels not in TIME_CHANNELS:
            raise ValueError("History encoder channels must be 16, 24 or 32")
        self.channels = channels
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(8, 16, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(16, channels, kernel_size=9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=7, stride=4, padding=3),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(channels + 16 + 3, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, channels),
            nn.LayerNorm(channels),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if (not isinstance(history, torch.Tensor) or history.ndim != 3
                or history.shape[1:] != (1, FRAMES)
                or not 1 <= len(history) <= MAX_BATCH
                or history.dtype != torch.float32
                or history.device.type != "cpu"
                or not torch.isfinite(history).all()
                or torch.max(torch.abs(history)).item() > MAX_SIGNAL):
            raise ValueError("History encoder needs finite CPU waveform windows")
        values = history.reshape(len(history), 1, 16, FRAMES // 16)
        envelopes = values.double().square().mean(dim=-1).sqrt().to(torch.float32)
        global_mean = history.double().mean(dim=-1)
        global_rms = history.double().square().mean(dim=-1).sqrt()
        global_peak = history.abs().amax(dim=-1).double()
        basic = torch.cat((
            envelopes.reshape(len(history), 16),
            global_mean, global_rms, global_peak,
        ), dim=1).to(torch.float32)
        learned = self.pool(self.features(history)).squeeze(-1)
        combined = self.projection(torch.cat((learned, basic), dim=1))
        if (combined.shape != (len(history), self.channels)
                or not torch.isfinite(combined).all()):
            raise ValueError("Global history encoding is nonfinite or incompatible")
        return combined


class GlobalHistoryVelocityDenoiser(nn.Module):
    """Velocity-predicting raw denoiser with separate long-window history."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        if type(channels) is not int or channels not in TIME_CHANNELS:
            raise ValueError("Denoiser channels must be 16, 24 or 32")
        self.channels = channels
        self.history_encoder = HistoryEncoder(channels)
        self.text_encoder = TextEncoder(channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(2 * channels, channels * 2),
            nn.SiLU(), nn.Linear(channels * 2, channels),
        )
        self.input = nn.Conv1d(1, channels, 5, padding=2)
        self.residual = nn.ModuleList(
            ConditionedResidual(channels, d)
            for d in (1, 2, 4, 8, 16, 1)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv1d(channels, 1, 3, padding=1),
        )

    def _condition(
        self, tokens: torch.Tensor, timestep: torch.Tensor,
        history: torch.Tensor,
    ) -> torch.Tensor:
        half = self.channels // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half)
            / max(half - 1, 1)
        )
        angles = timestep.float()[:, None] * freqs[None]
        time = self.time_mlp(torch.cat(
            (angles.sin(), angles.cos()), dim=1
        ))
        text = self.text_encoder(tokens)
        audio = self.history_encoder(history)
        result = self.context_fusion(
            torch.cat((text + time, audio), dim=1)
        )
        if not torch.isfinite(result).all():
            raise ValueError("Conditioned neural embedding is nonfinite")
        return result

    def forward(
        self, noisy: torch.Tensor, timestep: torch.Tensor,
        tokens: torch.Tensor, history: torch.Tensor, *,
        prefix_frames: int = 1024,
    ) -> torch.Tensor:
        _context_inputs(
            noisy, timestep, tokens, history,
            prefix_frames=prefix_frames, check_identity=True,
        )
        if (not torch.all((timestep >= 0) & (timestep < 256))):
            raise ValueError("Invalid raw denoising timestep")
        condition = self._condition(tokens, timestep, history)
        hidden = self.input(noisy)
        for block in self.residual:
            hidden = block(hidden, condition)
        output = self.output(hidden)
        if output.shape != noisy.shape or not torch.isfinite(output).all():
            raise ValueError("Global-history denoiser produced invalid velocity")
        return output


def masked_history_velocity_loss(
    network: GlobalHistoryVelocityDenoiser,
    schedule: DiffusionSchedule, clean_future: torch.Tensor,
    tokens: torch.Tensor, prior_history: torch.Tensor, *,
    prefix_frames: int, generator: torch.Generator,
) -> torch.Tensor:
    """Train future v only on truly adjacent windows with clean known prefix.

    This requires prior_history[-P:] == clean_future[:P]. The loss is
    differentiable with respect to both neural text and history encoders,
    not the input audio. The local generator leaves global RNG unchanged.
    """
    _prefix_frames(prefix_frames)
    if (not isinstance(network, GlobalHistoryVelocityDenoiser)
            or not isinstance(schedule, DiffusionSchedule)
            or not isinstance(generator, torch.Generator)
            or not isinstance(clean_future, torch.Tensor)
            or not isinstance(prior_history, torch.Tensor)
            or not isinstance(tokens, torch.Tensor)
            or clean_future.shape != prior_history.shape
            or clean_future.ndim != 3 or clean_future.shape[1:] != (1, FRAMES)
            or tokens.shape != (len(clean_future), MAX_TOKENS)
            or clean_future.dtype != torch.float32
            or prior_history.dtype != torch.float32
            or clean_future.device.type != "cpu"
            or prior_history.device.type != "cpu"):
        raise ValueError("Invalid paired global-history raw v-training input")
    if (not torch.isfinite(clean_future).all()
            or not torch.isfinite(prior_history).all()
            or not torch.equal(
                clean_future[..., :prefix_frames],
                prior_history[..., -prefix_frames:],
            )):
        raise ValueError("History and future are not real contiguous audio")
    step = torch.randint(
        schedule.steps, (len(clean_future),),
        generator=generator, dtype=torch.long,
    )
    noise = torch.randn(
        clean_future.shape, generator=generator, dtype=torch.float32,
    )
    noisy = schedule.corrupt(clean_future, step, noise)
    anchored = _anchor(noisy, prior_history[..., -prefix_frames:])
    target = velocity_target(clean_future, noise, step, schedule)
    predicted = network(
        anchored, step, tokens, prior_history,
        prefix_frames=prefix_frames,
    )
    result = F.mse_loss(
        predicted[..., prefix_frames:], target[..., prefix_frames:],
    )
    if not torch.isfinite(result):
        raise ValueError("Nonfinite global-history future velocity loss")
    return result


@torch.no_grad()
def sample_global_history_ddim(
    network: GlobalHistoryVelocityDenoiser,
    schedule: DiffusionSchedule, caption: str,
    prior_history: torch.Tensor, *, prefix_frames: int,
    evaluations: int, seed: int = 0, guidance: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Sample one 8192-frame future clip conditioned on 8192 frames of history.

    Known overlap is copied exactly, not denoised or blended. This
    conditions on a full previous clip (not just the prefix), but the
    prototype still cannot claim useful sound quality or model maturity.
    """
    _prefix_frames(prefix_frames)
    plan = ddim_plan(schedule, evaluations=evaluations)
    if (not isinstance(network, GlobalHistoryVelocityDenoiser)
            or not isinstance(prior_history, torch.Tensor)
            or prior_history.shape != (FRAMES,)
            or prior_history.dtype != torch.float32
            or prior_history.device.type != "cpu"
            or prior_history.requires_grad
            or not torch.isfinite(prior_history).all()
            or torch.max(torch.abs(prior_history)).item() > MAX_SIGNAL
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(guidance) not in (float, int)
            or not math.isfinite(guidance) or not 0 <= guidance <= 4
            or type(caption) is not str or not caption.strip()):
        raise ValueError("Invalid bounded global history sampling settings")
    if next(network.parameters()).device.type != "cpu":
        raise ValueError("Global history model sampling is CPU-only")
    trained_hash = model_state_sha256(network)
    prior_mode = network.training
    prior_rng = torch.random.get_rng_state().clone()
    prefix = prior_history[-prefix_frames:].view(1, 1, -1)
    history_batch = prior_history.view(1, 1, FRAMES)
    tokens = ByteTextTokenizer().encode([caption])
    null_tokens = torch.zeros_like(tokens)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampled = torch.randn(
        (1, 1, FRAMES), generator=generator, dtype=torch.float32,
    )
    network.eval()
    try:
        for index, timestep in enumerate(plan.timesteps):
            sampled = _anchor(sampled, prefix)
            step = torch.tensor([timestep], dtype=torch.long)
            pred = network(
                sampled, step, tokens, history_batch,
                prefix_frames=prefix_frames,
            )
            if guidance != 1:
                uncond = network(
                    sampled, step, null_tokens, history_batch,
                    prefix_frames=prefix_frames,
                )
                pred = uncond + guidance * (pred - uncond)
            if not torch.isfinite(pred).all():
                raise ValueError("Nonfinite global-history predicted velocity")
            previous = plan.timesteps[index + 1] if index + 1 < len(
                plan.timesteps) else -1
            sampled = velocity_ddim_transition(
                sampled, pred, timestep, previous, schedule,
            )
        result = _anchor(sampled, prefix)[0, 0].detach()
        if (result.shape != (FRAMES,) or not torch.isfinite(result).all()
                or result.abs().max().item() > MAX_SIGNAL
                or not torch.equal(result[:prefix_frames], prefix[0, 0])):
            raise ValueError("Invalid or nonanchored global-history future")
        if (model_state_sha256(network) != trained_hash
                or not torch.equal(torch.random.get_rng_state(), prior_rng)):
            raise ValueError("Global-history inference mutated model or RNG")
        info = {
            "protocol": INFERENCE_PROTOCOL,
            "conditioning": "explicit_global_previous_8192_frame_encoder",
            "requires_matching_global_history_future_v_training": True,
            "sample_rate": SAMPLE_RATE,
            "prior_frames": FRAMES,
            "prefix_frames": prefix_frames,
            "sample_seed": seed,
            "timestep_evaluations": evaluations,
            "model_evaluations": plan.as_dict(guidance=guidance)[
                "model_evaluations"],
            "guidance": float(guidance),
            "known_prefix_exactly_preserved": True,
            "previous_audio_sha256": hashlib.sha256(
                prior_history.numpy().tobytes()
            ).hexdigest(),
            "generated_sha256": hashlib.sha256(
                result.numpy().tobytes()
            ).hexdigest(),
            "first_unknown_sample_jump": abs(float(
                result[prefix_frames] - result[prefix_frames - 1]
            )),
            "peak_before_clamp": float(result.abs().max()),
            "fraction_outside_unit_audio_range": float(
                (result.abs() > 1).float().mean()),
            "normalization_or_clipping_applied": False,
            "model_parameters_unchanged": True,
            "global_rng_unchanged": True,
            "audio_or_weights_saved": False,
            "publication_authorized": False,
            "listening_or_long_context_quality_measured": False,
        }
        return result, info
    finally:
        network.train(prior_mode)
        if not torch.equal(torch.random.get_rng_state(), prior_rng):
            torch.random.set_rng_state(prior_rng)


@torch.no_grad()
def extend_global_history_sequence(
    network: GlobalHistoryVelocityDenoiser,
    schedule: DiffusionSchedule, initial_clip: torch.Tensor,
    captions: tuple[str, ...] | str, *,
    segments: int, prefix_frames: int, evaluations: int,
    seed: int = 0, guidance: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Bounded continuation requiring a supplied first clip and future captions."""
    _prefix_frames(prefix_frames)
    if (type(segments) is not int or not 2 <= segments <= MAX_SEGMENTS
            or type(seed) is not int or not 0 <= seed <= 2**32 - segments
            or not isinstance(initial_clip, torch.Tensor)
            or initial_clip.shape != (FRAMES,)
            or initial_clip.dtype != torch.float32
            or initial_clip.device.type != "cpu"
            or initial_clip.requires_grad
            or not torch.isfinite(initial_clip).all()):
        raise ValueError("Invalid bounded initial history or segment count")
    if type(captions) is str and captions.strip():
        future_captions = (captions,) * (segments - 1)
    elif (type(captions) is tuple and len(captions) == segments - 1
          and all(type(x) is str and x.strip() for x in captions)):
        future_captions = captions
    else:
        raise ValueError("One caption per future segment or one shared caption required")
    output = initial_clip.clone()
    before_rng = torch.random.get_rng_state().clone()
    state = model_state_sha256(network)
    mode = network.training
    joins = []
    try:
        for index, caption in enumerate(future_captions, start=1):
            history = output[-FRAMES:].clone()
            future, trace = sample_global_history_ddim(
                network, schedule, caption, history,
                prefix_frames=prefix_frames,
                evaluations=evaluations, seed=seed + index,
                guidance=guidance,
            )
            output = torch.cat((output, future[prefix_frames:]))
            joins.append({
                "segment": index,
                "seed": seed + index,
                "future_caption": caption,
                "prior_clip_sha256": trace["previous_audio_sha256"],
                "full_generated_clip_sha256": trace["generated_sha256"],
                "first_new_sample_jump": trace["first_unknown_sample_jump"],
                "known_overlap_preserved": trace[
                    "known_prefix_exactly_preserved"],
            })
        expected = FRAMES + (segments - 1) * (FRAMES - prefix_frames)
        if (output.numel() != expected
                or not torch.isfinite(output).all()
                or model_state_sha256(network) != state
                or not torch.equal(torch.random.get_rng_state(), before_rng)):
            raise ValueError("Global history sequence state/provenance invalid")
        return output, {
            "protocol": SEQUENCE_PROTOCOL,
            "training_protocol": PROTOCOL,
            "source_initial_clip_supplied": True,
            "segments": segments, "prefix_frames": prefix_frames,
            "sample_rate": SAMPLE_RATE,
            "frames": output.numel(),
            "duration_seconds": output.numel() / SAMPLE_RATE,
            "continuation_seed_base": seed,
            "recording_sha256": hashlib.sha256(
                output.numpy().tobytes()
            ).hexdigest(),
            "joins": joins,
            "model_state_preserved": True,
            "rng_preserved": True,
            "model_weights_or_audio_written": False,
            "publication_authorized": False,
            "long_audio_quality_or_semantics_measured": False,
            "warning": (
                "Explicit full-history conditioning can transmit audio "
                "context into the future prediction, but this network "
                "is only an untrained/short-trained 8-kHz research model. "
                "It does not prove natural or semantically coherent audio."
            ),
        }
    finally:
        network.train(mode)
        if not torch.equal(torch.random.get_rng_state(), before_rng):
            torch.random.set_rng_state(before_rng)
