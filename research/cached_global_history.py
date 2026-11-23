"""Audited, ephemeral history-feature reuse for original Sonumio v-DDIM.

The full previous 8192-frame audio does not change across denoising steps.
Uncached inference unnecessarily runs the global HistoryEncoder at every
timestep, and twice per timestep for text CFG. An in-memory prepared
history tensor can be reused instead, while text/time residual
conditioning is *still recomputed at every denoiser evaluation*.

This is NOT a neural checkpoint, durable cache, user audio upload or
saved embedding. The feature is bound to exact source waveform bytes,
model parameter bytes and tensor bytes via SHA256 and fail-closed checks.
A caller must explicitly manage its lifetime. Neither cache nor audio
is written to disk and no speedup is promised without CPU profiling.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch

from research.global_history_denoiser import (
    GlobalHistoryVelocityDenoiser, FRAMES, MAX_SIGNAL,
)
from research.text_audio import ByteTextTokenizer
from research.velocity_ddim import (
    ddim_plan, velocity_ddim_transition,
)
from research.velocity_prefix_continuation import (
    _prefix_frames, _anchor,
)
from research.wave_diffusion import DiffusionSchedule, SAMPLE_RATE
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-audited-in-memory-history-feature-cache-v1"
INFERENCE_PROTOCOL = "sonumio-cached-global-history-ddim-parity-v1"


def _sha(tensor: torch.Tensor) -> str:
    if (not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.requires_grad
            or not torch.isfinite(tensor).all()):
        raise ValueError("Invalid finite CPU float32 history digest input")
    return hashlib.sha256(
        tensor.detach().contiguous().numpy().tobytes()
    ).hexdigest()


def _source(history: torch.Tensor) -> None:
    if (not isinstance(history, torch.Tensor)
            or history.shape != (FRAMES,)
            or history.device.type != "cpu"
            or history.dtype != torch.float32
            or history.requires_grad
            or not torch.isfinite(history).all()
            or history.abs().max().item() > MAX_SIGNAL):
        raise ValueError("Cached history source requires 8192 valid CPU samples")


def _model(model: GlobalHistoryVelocityDenoiser) -> None:
    if (not isinstance(model, GlobalHistoryVelocityDenoiser)
            or next(model.parameters()).device.type != "cpu"):
        raise ValueError("Require original CPU global-history velocity architecture")


@dataclass(frozen=True, slots=True)
class PreparedHistory:
    """Transient feature with tamper-evident audio/model/tensor provenance."""

    source_sha256: str
    model_sha256: str
    embedding_sha256: str
    embedding: torch.Tensor
    model_channels: int
    source_frames: int = FRAMES
    protocol: str = PROTOCOL

    def validate(
        self, model: GlobalHistoryVelocityDenoiser,
        prior_history: torch.Tensor,
    ) -> torch.Tensor:
        """Recheck all volatile values before each inference; never trust flags."""
        _model(model)
        _source(prior_history)
        if (self.protocol != PROTOCOL
                or self.source_frames != FRAMES
                or self.model_channels != model.channels
                or self.source_sha256 != _sha(prior_history)
                or self.model_sha256 != model_state_sha256(model)
                or not isinstance(self.embedding, torch.Tensor)
                or self.embedding.shape != (1, model.channels)
                or self.embedding.dtype != torch.float32
                or self.embedding.device.type != "cpu"
                or self.embedding.requires_grad
                or not torch.isfinite(self.embedding).all()
                or self.embedding_sha256 != _sha(self.embedding)):
            raise ValueError(
                "Stale, mutated, mismatched or improperly shaped history embedding"
            )
        return self.embedding

    def describe(self) -> dict:
        """No raw waveform or embedding arrays are included in metadata."""
        return {
            "protocol": self.protocol,
            "source_sha256": self.source_sha256,
            "model_sha256": self.model_sha256,
            "embedding_sha256": self.embedding_sha256,
            "model_channels": self.model_channels,
            "source_frames": self.source_frames,
            "embedding_dimension": self.model_channels,
            "audio_or_embeddings_exported": False,
            "model_weights_saved": False,
            "model_publication_authorized": False,
        }


@torch.no_grad()
def prepare_history(
    model: GlobalHistoryVelocityDenoiser,
    prior_history: torch.Tensor,
) -> PreparedHistory:
    """Encode full prior clip once, with model state and RNG preservation."""
    _model(model)
    _source(prior_history)
    prior_rng = torch.random.get_rng_state().clone()
    prior_mode = model.training
    before = model_state_sha256(model)
    model.eval()
    try:
        encoded = model.history_encoder(
            prior_history.view(1, 1, FRAMES),
        ).detach().clone()
        if (encoded.shape != (1, model.channels)
                or encoded.dtype != torch.float32
                or not torch.isfinite(encoded).all()
                or encoded.requires_grad):
            raise ValueError("Nonfinite or incompatible full-history feature")
        result = PreparedHistory(
            source_sha256=_sha(prior_history),
            model_sha256=before,
            embedding_sha256=_sha(encoded),
            embedding=encoded,
            model_channels=model.channels,
        )
        if (model_state_sha256(model) != before
                or not torch.equal(torch.random.get_rng_state(), prior_rng)):
            raise ValueError("History encoding unexpectedly mutated model or global RNG")
        result.validate(model, prior_history)
        return result
    finally:
        model.train(prior_mode)
        if not torch.equal(torch.random.get_rng_state(), prior_rng):
            torch.random.set_rng_state(prior_rng)


@torch.no_grad()
def sample_cached_global_history_ddim(
    model: GlobalHistoryVelocityDenoiser,
    schedule: DiffusionSchedule,
    caption: str,
    prior_history: torch.Tensor, *,
    prefix_frames: int, evaluations: int,
    seed: int = 0, guidance: float = 1.0,
    prepared_history: PreparedHistory | None = None,
) -> tuple[torch.Tensor, dict]:
    """Strict numerical parity with the uncached original v-DDIM sampler.

    Text and timestep embeddings are NOT cached: time and CFG language
    change each evaluation. Only history_encoder() output is reusable.
    Output remains raw and unclipped. A stale or forged cached feature
    is rejected before any generated noise or neural work occurs.
    """
    _model(model)
    _source(prior_history)
    _prefix_frames(prefix_frames)
    plan = ddim_plan(schedule, evaluations=evaluations)
    if (type(seed) is not int or not 0 <= seed < 2**32
            or type(guidance) not in (int, float)
            or not math.isfinite(guidance) or not 0 <= guidance <= 4
            or type(caption) is not str or not caption.strip()):
        raise ValueError("Invalid bounded cached history sampling options")
    if prepared_history is not None and not isinstance(
        prepared_history, PreparedHistory
    ):
        raise ValueError("History cache must be a validated PreparedHistory")
    original_rng = torch.random.get_rng_state().clone()
    original_mode = model.training
    original_model_sha = model_state_sha256(model)
    created_now = prepared_history is None
    cached = (
        prepare_history(model, prior_history)
        if created_now else prepared_history
    )
    context = cached.validate(model, prior_history)
    text = ByteTextTokenizer().encode([caption])
    null_text = torch.zeros_like(text)
    history_batch = prior_history.view(1, 1, FRAMES)
    prefix = prior_history[-prefix_frames:].view(1, 1, prefix_frames)
    local_rng = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(
        (1, 1, FRAMES), generator=local_rng, dtype=torch.float32,
    )
    model.eval()
    try:
        for index, step in enumerate(plan.timesteps):
            sample = _anchor(sample, prefix)
            timestep = torch.tensor([step], dtype=torch.long)
            # Same original network and operations, replacing only the
            # invariant call to the history encoder with its exact tensor.
            predicted = model(
                sample, timestep, text, history_batch,
                prefix_frames=prefix_frames,
                history_embedding=context,
            )
            if guidance != 1:
                null_prediction = model(
                    sample, timestep, null_text, history_batch,
                    prefix_frames=prefix_frames,
                    history_embedding=context,
                )
                predicted = (
                    null_prediction
                    + guidance * (predicted - null_prediction)
                )
            earlier = (
                plan.timesteps[index + 1]
                if index + 1 < plan.evaluations else -1
            )
            sample = velocity_ddim_transition(
                sample, predicted, step, earlier, schedule,
            )
        output = _anchor(sample, prefix)[0, 0].detach()
        if (output.shape != (FRAMES,)
                or output.dtype != torch.float32
                or not torch.isfinite(output).all()
                or output.abs().max().item() > MAX_SIGNAL
                or not torch.equal(output[:prefix_frames],
                                   prior_history[-prefix_frames:])):
            raise ValueError("Invalid cached v-DDIM waveform or known prefix")
        if (model_state_sha256(model) != original_model_sha
                or not torch.equal(torch.random.get_rng_state(), original_rng)):
            raise ValueError("Cached inference mutated network state or global RNG")
        cached.validate(model, prior_history)
        expected_naive = evaluations * (1 if guidance == 1 else 2)
        report = {
            "protocol": INFERENCE_PROTOCOL,
            "reuses_identical_full_history_embedding": True,
            "cache": cached.describe(),
            "cache_created_during_call": created_now,
            "history_encoder_evaluations_for_this_call": (
                1 if created_now else 0
            ),
            "expected_uncached_history_encoder_evaluations": expected_naive,
            "history_encoder_calls_avoided": expected_naive - (
                1 if created_now else 0
            ),
            "text_timestep_and_residual_network_still_evaluated_every_step": True,
            "timestep_evaluations": plan.evaluations,
            "total_denoiser_model_evaluations": expected_naive,
            "seed": seed, "guidance": float(guidance),
            "source_sha256": _sha(prior_history),
            "waveform_sha256": _sha(output),
            "sample_rate": SAMPLE_RATE, "frames": FRAMES,
            "prefix_frames": prefix_frames,
            "known_prefix_preserved_exactly": True,
            "preclamp_peak": float(output.abs().max()),
            "fraction_outside_unit_audio_range": float(
                (output.abs() > 1).float().mean()
            ),
            "normalization_or_clipping_applied": False,
            "neural_weights_and_rng_preserved": True,
            "audio_embeddings_or_model_weights_written": False,
            "model_publication_authorized": False,
            "latency_speedup_or_audio_quality_claim": False,
            "warning": (
                "History-feature reuse avoids redundant neural encoder calls; "
                "speedup requires measured CPU timing. Numeric sampler parity "
                "does not establish natural audio, prompt understanding, "
                "long-range speech/music coherence or model readiness."
            ),
        }
        return output, report
    finally:
        model.train(original_mode)
        if not torch.equal(torch.random.get_rng_state(), original_rng):
            torch.random.set_rng_state(original_rng)


@torch.no_grad()
def parity_check(
    model: GlobalHistoryVelocityDenoiser,
    schedule: DiffusionSchedule, caption: str,
    prior_history: torch.Tensor, *, prefix_frames: int,
    evaluations: int, seed: int, guidance: float,
) -> dict:
    """Execute both paths and demand bitwise output equivalence."""
    from research.global_history_denoiser import sample_global_history_ddim

    global_rng = torch.random.get_rng_state().clone()
    before = model_state_sha256(model)
    source, original = sample_global_history_ddim(
        model, schedule, caption, prior_history,
        prefix_frames=prefix_frames,
        evaluations=evaluations, seed=seed, guidance=guidance,
    )
    cached, report = sample_cached_global_history_ddim(
        model, schedule, caption, prior_history,
        prefix_frames=prefix_frames,
        evaluations=evaluations, seed=seed, guidance=guidance,
    )
    if (not torch.equal(source, cached)
            or model_state_sha256(model) != before
            or not torch.equal(torch.random.get_rng_state(), global_rng)):
        raise ValueError("History feature reuse changed the generated waveform")
    return {
        "protocol": INFERENCE_PROTOCOL,
        "uncached_sha256": _sha(source),
        "cached_sha256": _sha(cached),
        "exact_bitwise_sampler_parity": True,
        "history_encoder_calls_avoided": report["history_encoder_calls_avoided"],
        "guidance": float(guidance),
        "evaluations": evaluations,
        "seed": seed,
        "no_weights_embeddings_or_audio_exported": True,
        "perceptual_quality_or_real_world_latency_established": False,
    }
