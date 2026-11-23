"""Bounded in-memory overlap synthesis of short Sonumio v-DDIM waveforms.

This supports *experimental longer numerical audio buffers* from the
existing 1.024-second, 8-kHz original raw/latent DDIM samplers. Independent
segments are connected with complementary raised-cosine crossfades so
overlaps preserve DC/constant-wave amplitude; they do NOT provide temporal
semantic consistency, high-fidelity speech, music, or new neural weights.

Only v-trained denoisers are valid. Nothing is clipped, saved or published.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch
from torch import nn

from research.latent_audio import FRAMES
from research.velocity_ddim import ddim_plan, sample_velocity_ddim
from research.wave_diffusion import DiffusionSchedule, SAMPLE_RATE
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-v-ddim-overlap-audio-research-v1"
MIN_OVERLAP = 128
MAX_OVERLAP = FRAMES // 2
MAX_SEGMENTS = 8
MAX_SIGNAL = 1.0e12


def _validate_overlaps(number: int, overlap_frames: int) -> None:
    if (type(number) is not int or not 2 <= number <= MAX_SEGMENTS
            or type(overlap_frames) is not int
            or not MIN_OVERLAP <= overlap_frames <= MAX_OVERLAP
            or overlap_frames % 64):
        raise ValueError(
            "Use 2..8 segments and 128..4096 overlap frames, multiple of 64"
        )


def _validate_wave(wave: torch.Tensor) -> None:
    if (not isinstance(wave, torch.Tensor)
            or wave.shape != (FRAMES,)
            or wave.device.type != "cpu"
            or wave.dtype != torch.float32 or wave.requires_grad
            or not torch.isfinite(wave).all()
            or torch.max(torch.abs(wave)).item() > MAX_SIGNAL):
        raise ValueError("Long-form source must be finite detached mono CPU float32")


@torch.no_grad()
def overlap_add(waves: Sequence[torch.Tensor], *, overlap_frames: int
                ) -> tuple[torch.Tensor, list[dict]]:
    """Raised-cosine overlap/add with exact length and explicit seam scores.

    The fade is complementary: (1-w) * previous + w * next. Equal input
    amplitudes remain equal, unlike equal-power mixing, which can amplify
    identical signals. This is a splice algorithm, not learned continuity.
    """
    if type(waves) not in (tuple, list):
        raise ValueError("Provide an explicit tuple/list of neural segments")
    _validate_overlaps(len(waves), overlap_frames)
    for wave in waves:
        _validate_wave(wave)

    positions = torch.arange(
        1, overlap_frames + 1, dtype=torch.float64
    )
    faded = (
        (1 - torch.cos(math.pi * positions / (overlap_frames + 1))) * 0.5
    ).to(torch.float32)
    inverse = 1 - faded
    output = waves[0].clone()
    diagnostics: list[dict] = []
    for number, incoming in enumerate(waves[1:], start=1):
        start = output.numel() - overlap_frames
        previous = output
        prior_tail = previous[start:]
        incoming_head = incoming[:overlap_frames]
        blended = inverse * prior_tail + faded * incoming_head
        output = torch.cat(
            (previous[:start], blended, incoming[overlap_frames:]), dim=0,
        )
        hard_jump = torch.abs(previous[-1] - incoming[0]).item()
        begin_jump = torch.abs(output[start] - output[start - 1]).item()
        end = start + overlap_frames
        finish_jump = torch.abs(output[end] - output[end - 1]).item()
        differences = torch.mean(torch.abs(prior_tail - incoming_head)).item()
        scores = (hard_jump, begin_jump, finish_jump, differences)
        if not all(math.isfinite(v) for v in scores):
            raise ValueError("Nonfinite overlap seam diagnostic")
        diagnostics.append({
            "transition": number,
            "blend_starts_at_frame": start,
            "blend_ends_before_frame": end,
            "hard_concatenation_sample_jump": hard_jump,
            "blend_start_sample_jump": begin_jump,
            "blend_end_sample_jump": finish_jump,
            "mean_overlap_waveform_difference": differences,
            "perceptual_click_or_semantic_quality_measured": False,
        })
    expected = FRAMES + (len(waves) - 1) * (FRAMES - overlap_frames)
    if (output.shape != (expected,) or output.dtype != torch.float32
            or not torch.isfinite(output).all()
            or torch.max(torch.abs(output)).item() > MAX_SIGNAL):
        raise ValueError("Invalid or unbounded overlapped waveform")
    return output, diagnostics


def _captions(captions: str | tuple[str, ...],
              segments: int) -> tuple[str, ...]:
    if isinstance(captions, str):
        supplied = (captions,) * segments
    elif type(captions) is tuple and len(captions) == segments:
        supplied = captions
    else:
        raise ValueError("Specify one caption or one per requested segment")
    if any(type(s) is not str or not s.strip() for s in supplied):
        raise ValueError("Long-form captions must be nonempty strings")
    return supplied


@torch.no_grad()
def sample_overlapped_velocity_ddim(
    denoiser: nn.Module, schedule: DiffusionSchedule,
    captions: str | tuple[str, ...], *,
    segments: int, overlap_frames: int = 1024, evaluations: int = 8,
    seed: int = 0, guidance: float = 1.0,
    latent: bool = False, codec: nn.Module | None = None,
) -> tuple[torch.Tensor, dict]:
    """Deterministic multiple clips, same v-trained model, one bounded buffer.

    Each segment uses a distinct explicit seed (seed, seed+1, ...). Independent
    generation DOES NOT preserve narrative/music continuity. Never substitute
    epsilon-trained weights or interpret final metrics as listening scores.
    """
    _validate_overlaps(segments, overlap_frames)
    plan = ddim_plan(schedule, evaluations=evaluations)
    prompts = _captions(captions, segments)
    if (type(seed) is not int or not 0 <= seed <= 2**32 - segments
            or type(guidance) not in (int, float)
            or not math.isfinite(guidance) or not 0 <= guidance <= 4
            or type(latent) is not bool or
            (latent and codec is None) or (not latent and codec is not None)
            or not isinstance(denoiser, nn.Module)
            or (codec is not None and not isinstance(codec, nn.Module))):
        raise ValueError("Invalid bounded long-form neural sampling parameters")

    before_rng = torch.random.get_rng_state().clone()
    model_hash = model_state_sha256(denoiser)
    codec_hash = model_state_sha256(codec) if codec is not None else None
    mode = denoiser.training
    codec_mode = codec.training if codec is not None else None
    sampled: list[torch.Tensor] = []
    metadata = []
    try:
        for index, prompt in enumerate(prompts):
            sampling_seed = seed + index
            result = sample_velocity_ddim(
                denoiser, schedule, prompt, seed=sampling_seed,
                evaluations=plan.evaluations, guidance=guidance,
                latent=latent, codec=codec,
                return_preactivation=latent,
            )
            if latent:
                wave, before_tanh = result
                if (not torch.isfinite(before_tanh).all()
                        or before_tanh.shape != (FRAMES,)):
                    raise ValueError("Invalid decoder pre-Tanh research activations")
            else:
                wave, before_tanh = result, None
            _validate_wave(wave)
            metadata.append({
                "index": index, "sampling_seed": sampling_seed,
                "caption": prompt,
                "short_clip_sha256": hashlib.sha256(
                    wave.detach().numpy().tobytes()
                ).hexdigest(),
                "prelimiter_signal_source": (
                    "latent_codec_decoder_before_tanh" if latent
                    else "raw_ddpm_before_final_clamp"
                ),
                **({"pre_tanh_saturation_fraction": float((
                    torch.abs(before_tanh) >= math.atanh(0.999)
                ).float().mean())} if latent else {}),
            })
            sampled.append(wave)
        output, joins = overlap_add(
            sampled, overlap_frames=overlap_frames,
        )
        if (model_hash != model_state_sha256(denoiser)
                or (codec is not None and
                    codec_hash != model_state_sha256(codec))):
            raise ValueError("Sampling changed the original neural model parameters")
        if (denoiser.training != mode or
                (codec is not None and codec.training != codec_mode)):
            raise ValueError("Long-form sampler did not preserve neural train/eval state")
        if not torch.equal(before_rng, torch.random.get_rng_state()):
            raise ValueError("Long-form sampler altered the global torch RNG")
        signal = output.detach()
        peak = float(signal.abs().max())
        rms = float(torch.sqrt(torch.mean(signal.double().square())))
        outside = float((signal.abs() > 1).float().mean())
        report = {
            "protocol": PROTOCOL,
            "sampler": "independent_v_ddim_clips_raised_cosine_overlap",
            "schedule_profile": plan.profile,
            "training_diffusion_steps": plan.training_steps,
            "model_evaluations_per_clip": plan.as_dict(
                guidance=guidance,
            )["model_evaluations"],
            "sample_rate": SAMPLE_RATE,
            "segments": segments, "overlap_frames": overlap_frames,
            "frames": signal.numel(),
            "duration_seconds": signal.numel() / SAMPLE_RATE,
            "seed": seed, "guidance": guidance, "latent": latent,
            "per_clip": metadata,
            "joins": joins,
            "output_sha256": hashlib.sha256(signal.numpy().tobytes()).hexdigest(),
            "unclipped_peak": peak, "unclipped_rms": rms,
            "fraction_above_unit_audio_range": outside,
            "limiter_or_normalization_applied": False,
            "model_state_preserved": True, "global_rng_preserved": True,
            "audio_files_or_model_weights_saved": False,
            "model_publication_authorized": False,
            "listening_or_multisegment_semantic_continuity_evaluated": False,
            "warning": (
                "Independent one-second v-DDIM clips plus crossfade are not "
                "trained long-context generation. A smooth splice cannot "
                "establish audible continuity, naturalness, speech, music, "
                "prompt adherence or any Sonumio model release readiness."
            ),
        }
        return signal, report
    finally:
        if not torch.equal(torch.random.get_rng_state(), before_rng):
            torch.random.set_rng_state(before_rng)
