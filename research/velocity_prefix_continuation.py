"""Prefix-anchored velocity diffusion for experimental audio continuation.

The *known prefix is clean and fixed at EVERY denoising timestep*, and only
unknown future samples are diffused. This is a distinct training/inference
protocol from ordinary full-clip velocity DDIM. A model must be trained with
`masked_future_velocity_loss` to expect this input distribution. Existing
unconditioned models can run a numerical smoke but cannot be claimed to
understand continuation.

Raw 8-kHz mono, fixed 8192-frame clips, in memory, no model/file release.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.latent_audio import FRAMES
from research.longform_ddim import (
    MAX_OVERLAP, MAX_SEGMENTS, MIN_OVERLAP, MAX_SIGNAL,
    _captions, _validate_overlaps, _validate_wave,
)
from research.text_audio import ByteTextTokenizer
from research.velocity_ddim import (
    ddim_plan, sample_velocity_ddim, velocity_ddim_transition,
)
from research.velocity_prediction import velocity_target
from research.wave_diffusion import DiffusionSchedule, SAMPLE_RATE
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-clean-prefix-velocity-ddim-research-v1"
SEQUENCE_PROTOCOL = "sonumio-clean-prefix-velocity-continuation-v1"


def _prefix_frames(value: int) -> None:
    if (type(value) is not int or value < MIN_OVERLAP
            or value > MAX_OVERLAP or value % 64 != 0):
        raise ValueError("Prefix must have 128..4096 frames in multiples of 64")


def _finite_raw(audio: torch.Tensor, *, batch: bool) -> None:
    if not isinstance(audio, torch.Tensor):
        raise ValueError("Requires a real raw waveform tensor")
    shape_ok = (audio.ndim == 3 and audio.shape[1:] == (1, FRAMES)
                and 1 <= len(audio) <= 8) if batch else (
                audio.ndim == 1 and audio.shape == (FRAMES,)
            )
    if (not shape_ok
            or audio.device.type != "cpu" or audio.dtype != torch.float32
            or not torch.isfinite(audio).all()
            or audio.detach().abs().max().item() > MAX_SIGNAL):
        raise ValueError("Requires finite bounded CPU float32 8192-frame mono")


def _known_prefix(prefix: torch.Tensor) -> None:
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 1:
        raise ValueError("Known prefix must be a mono frame vector")
    _prefix_frames(prefix.numel())
    if (prefix.dtype != torch.float32 or prefix.device.type != "cpu"
            or prefix.requires_grad or not torch.isfinite(prefix).all()
            or prefix.abs().max().item() > MAX_SIGNAL):
        raise ValueError("Known prefix must be finite detached CPU float32")


def _model_args(denoiser: nn.Module, schedule: DiffusionSchedule,
                *, guidance: float) -> None:
    if (not isinstance(denoiser, nn.Module)
            or not isinstance(schedule, DiffusionSchedule)
            or type(guidance) not in (float, int)
            or not math.isfinite(guidance) or not 0 <= guidance <= 4):
        raise ValueError("Invalid prefix-conditioned model, schedule or CFG")
    try:
        model_device = next(denoiser.parameters()).device
    except StopIteration as exc:
        raise ValueError("Prefix denoiser must have parameters") from exc
    if model_device.type != "cpu" or schedule.alpha_bars.device.type != "cpu":
        raise ValueError("Prefix research supports CPU models only")


def _anchor(noisy: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
    """Replace fixed known section without any in-place mutation."""
    return torch.cat((prefix, noisy[..., prefix.shape[-1]:]), dim=-1)


def masked_future_velocity_loss(
    denoiser: nn.Module, schedule: DiffusionSchedule,
    clean: torch.Tensor, tokens: torch.Tensor,
    *, prefix_frames: int, generator: torch.Generator,
) -> torch.Tensor:
    """Train the network to predict v on FUTURE frames only.

    The clean prefix is visible in x_t at every noise level. No loss is
    assigned to the prefix itself. This is the exact conditioning seen in
    prefix DDIM inference, unlike naively passing a denoiser a clean prefix
    only at inference time. The user must train suitable weights privately.
    """
    _prefix_frames(prefix_frames)
    _finite_raw(clean, batch=True)
    _model_args(denoiser, schedule, guidance=1.0)
    if (not isinstance(tokens, torch.Tensor) or tokens.ndim != 2
            or tokens.shape[0] != len(clean) or tokens.device.type != "cpu"
            or tokens.dtype != torch.long
            or not isinstance(generator, torch.Generator)):
        raise ValueError("Invalid bilingual tokens or local generator")
    timestep = torch.randint(
        schedule.steps, (len(clean),), generator=generator,
        dtype=torch.long,
    )
    epsilon = torch.randn(
        clean.shape, generator=generator, dtype=torch.float32,
    )
    noisy = schedule.corrupt(clean, timestep, epsilon)
    noisy = _anchor(noisy, clean[..., :prefix_frames])
    target = velocity_target(clean, epsilon, timestep, schedule)
    predicted = denoiser(noisy, timestep, tokens)
    if (predicted.shape != target.shape or predicted.dtype != torch.float32
            or not torch.isfinite(predicted).all()):
        raise ValueError("Invalid masked-future velocity model output")
    future_loss = F.mse_loss(
        predicted[..., prefix_frames:], target[..., prefix_frames:],
    )
    if not torch.isfinite(future_loss):
        raise ValueError("Nonfinite future-only velocity loss")
    return future_loss


@torch.no_grad()
def sample_prefix_velocity_ddim(
    denoiser: nn.Module, schedule: DiffusionSchedule,
    caption: str, prefix: torch.Tensor, *,
    evaluations: int, seed: int = 0, guidance: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Generate a clip anchored to an exact *clean* input prefix.

    Training recipe: masked_future_velocity_loss, not full-clip v-MSE.
    Prefix values are never predicted or blended: exact copies anchor
    every denoising step. Future frames are generated using deterministic
    DDIM without clipping or normalizing. Continuity remains unverified.
    """
    _known_prefix(prefix)
    plan = ddim_plan(schedule, evaluations=evaluations)
    _model_args(denoiser, schedule, guidance=guidance)
    if (type(seed) is not int or not 0 <= seed < 2**32
            or type(caption) is not str or not caption.strip()):
        raise ValueError("Invalid prefix DDIM seed or caption")
    model_hash = model_state_sha256(denoiser)
    prior_mode = denoiser.training
    prior_rng = torch.random.get_rng_state().clone()
    tokens = ByteTextTokenizer().encode([caption])
    unconditional = torch.zeros_like(tokens)
    reference = prefix.detach().clone().view(1, 1, -1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    current = torch.randn(
        (1, 1, FRAMES), generator=generator, dtype=torch.float32,
    )
    denoiser.eval()
    try:
        for i, timestep in enumerate(plan.timesteps):
            current = _anchor(current, reference)
            step = torch.tensor([timestep], dtype=torch.long)
            velocity = denoiser(current, step, tokens)
            if guidance != 1:
                null = denoiser(current, step, unconditional)
                velocity = null + guidance * (velocity - null)
            if (velocity.shape != current.shape
                    or velocity.dtype != torch.float32
                    or not torch.isfinite(velocity).all()):
                raise ValueError("Invalid prefix DDIM velocity prediction")
            previous = (
                plan.timesteps[i + 1] if i + 1 < plan.evaluations else -1
            )
            current = velocity_ddim_transition(
                current, velocity, timestep, previous, schedule,
            )
        result = _anchor(current, reference)[0, 0].detach()
        _validate_wave(result)
        if not torch.equal(result[:len(prefix)], prefix):
            raise ValueError("Known continuation prefix was not preserved")
        if model_hash != model_state_sha256(denoiser):
            raise ValueError("Prefix DDIM inference mutated model parameters")
        if not torch.equal(prior_rng, torch.random.get_rng_state()):
            raise ValueError("Prefix DDIM inference mutated global torch RNG")
        next_jump = abs(float(result[len(prefix)] - result[len(prefix) - 1]))
        report = {
            "protocol": PROTOCOL,
            "conditioning": "clean_prefix_anchored_at_every_ddim_timestep",
            "requires_matching_masked_future_training": True,
            "objective": "velocity_on_future_only",
            "schedule_profile": plan.profile,
            "timestep_evaluations": evaluations,
            "neural_evaluations": plan.as_dict(guidance=guidance)[
                "model_evaluations"],
            "prefix_frames": len(prefix), "seed": seed,
            "guidance": float(guidance), "sample_rate": SAMPLE_RATE,
            "exact_original_prefix_preserved": True,
            "prefix_sha256": hashlib.sha256(
                prefix.numpy().tobytes()).hexdigest(),
            "output_sha256": hashlib.sha256(
                result.numpy().tobytes()).hexdigest(),
            "first_future_sample_jump": next_jump,
            "unclipped_peak": float(result.abs().max()),
            "fraction_above_unit_range": float(
                (result.abs() > 1).float().mean()),
            "neural_model_state_preserved": True,
            "global_rng_preserved": True,
            "no_model_weights_or_audio_saved": True,
            "model_publication_authorized": False,
            "perceptual_or_semantic_continuity_evaluated": False,
        }
        return result, report
    finally:
        denoiser.train(prior_mode)
        # A failure must not leave global RNG state corrupted.
        if not torch.equal(torch.random.get_rng_state(), prior_rng):
            torch.random.set_rng_state(prior_rng)


@torch.no_grad()
def sample_continuation_sequence(
    denoiser: nn.Module, schedule: DiffusionSchedule,
    captions: str | tuple[str, ...], *,
    segments: int, prefix_frames: int, evaluations: int,
    seed: int = 0, guidance: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """One independent lead clip, then exactly anchored raw future samples."""
    _validate_overlaps(segments, prefix_frames)
    plan = ddim_plan(schedule, evaluations=evaluations)
    _model_args(denoiser, schedule, guidance=guidance)
    prompts = _captions(captions, segments)
    if type(seed) is not int or not 0 <= seed <= 2**32 - segments:
        raise ValueError("Invalid per-segment continuation seed range")
    source_rng = torch.random.get_rng_state().clone()
    model_hash = model_state_sha256(denoiser)
    mode = denoiser.training
    try:
        first = sample_velocity_ddim(
            denoiser, schedule, prompts[0], evaluations=evaluations,
            seed=seed, guidance=guidance,
        )
        _validate_wave(first)
        output = first.clone()
        joins = []
        for i, caption in enumerate(prompts[1:], start=1):
            prefix = output[-prefix_frames:].clone()
            current, diagnostic = sample_prefix_velocity_ddim(
                denoiser, schedule, caption, prefix,
                evaluations=plan.evaluations, seed=seed + i,
                guidance=guidance,
            )
            if diagnostic["prefix_sha256"] != hashlib.sha256(
                prefix.numpy().tobytes()).hexdigest():
                raise ValueError("Continuation conditioning provenance drift")
            join_at = output.numel()
            output = torch.cat((output, current[prefix_frames:]))
            joins.append({
                "segment": i, "join_at_frame": join_at,
                "known_prefix_sha256": diagnostic["prefix_sha256"],
                "exact_prefix_preserved": True,
                "first_unknown_sample_jump": diagnostic[
                    "first_future_sample_jump"],
                "neural_evaluations": diagnostic["neural_evaluations"],
                "seed": seed + i,
                "caption": caption,
            })
        expected_frames = FRAMES + (segments - 1) * (FRAMES - prefix_frames)
        if (output.numel() != expected_frames
                or not torch.isfinite(output).all()
                or model_hash != model_state_sha256(denoiser)
                or mode != denoiser.training
                or not torch.equal(source_rng, torch.random.get_rng_state())):
            raise ValueError("Invalid or unexpectedly mutable continuation result")
        return output, {
            "protocol": SEQUENCE_PROTOCOL,
            "training_requirement": "matched_clean_prefix_future_v_loss",
            "trained_long_context_model": False,
            "clip_count": segments, "prefix_frames": prefix_frames,
            "sample_rate": SAMPLE_RATE,
            "output_frames": output.numel(),
            "duration_seconds": output.numel() / SAMPLE_RATE,
            "lead_seed": seed, "joins": joins,
            "output_sha256": hashlib.sha256(
                output.numpy().tobytes()).hexdigest(),
            "model_state_preserved": True, "global_rng_preserved": True,
            "no_model_weights_or_audio_saved": True,
            "model_publication_authorized": False,
            "listening_quality_or_prompt_alignment_measured": False,
            "warning": (
                "Prefix-anchored diffusion constrains a clean history, "
                "but does not prove smooth future onset, scene coherence, "
                "speech, music, naturalness, or longer learned context. "
                "Requires separately trained masked-future v-model weights."
            ),
        }
    finally:
        denoiser.train(mode)
        if not torch.equal(source_rng, torch.random.get_rng_state()):
            torch.random.set_rng_state(source_rng)
