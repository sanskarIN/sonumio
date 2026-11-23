"""Experimental deterministic DDIM inference for Sonumio v-trained denoisers.

This is a SEPARATE sampler from the original stochastic v-DDPM research.
It supports timestep respacing and paired same-noise comparisons on the
0.256/1.024-second synthetic-only model architectures, without modifying
training losses, legacy checkpoints, or existing sampling defaults.

Use ONLY weights trained to predict velocity (v), not epsilon. No weights,
audio files, streaming endpoints, or original Sonumio-1 models are produced.
Numerical sampler agreement is not perceptual or semantic quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from research.latent_audio import FRAMES, LATENT_CHANNELS, LATENT_FRAMES
from research.text_audio import ByteTextTokenizer
from research.velocity_prediction import decode_velocity
from research.wave_diffusion import DiffusionSchedule


@dataclass(frozen=True, slots=True)
class DDIMPlan:
    """Immutable inference budget and strictly descending DDPM timesteps.

    Both endpoints are retained: last trained noise level (T-1), and the
    earliest denoising level (0). Integer nearest-neighbor spacing is stable
    across Python versions and never builds a new diffusion schedule.
    """
    training_steps: int
    evaluations: int
    timesteps: tuple[int, ...]
    profile: str

    def __post_init__(self) -> None:
        if (type(self.training_steps) is not int
                or not 8 <= self.training_steps <= 256
                or type(self.evaluations) is not int
                or not 2 <= self.evaluations <= self.training_steps
                or type(self.timesteps) is not tuple
                or len(self.timesteps) != self.evaluations
                or any(type(t) is not int or not 0 <= t < self.training_steps
                       for t in self.timesteps)
                or self.timesteps[0] != self.training_steps - 1
                or self.timesteps[-1] != 0
                or any(a <= b for a, b in zip(
                    self.timesteps, self.timesteps[1:]))
                or self.profile not in DiffusionSchedule.PROFILES):
            raise ValueError("Invalid deterministic DDIM timestep plan")

    def as_dict(self, *, guidance: float = 1.0) -> dict:
        if (type(guidance) not in (int, float)
                or not math.isfinite(guidance) or not 0 <= guidance <= 4):
            raise ValueError("Invalid research classifier-free guidance")
        return {
            "protocol": "sonumio-velocity-ddim-plan-v1",
            "schedule_profile": self.profile,
            "training_steps": self.training_steps,
            "model_evaluations": self.evaluations * (1 if guidance == 1 else 2),
            "denoising_transitions": self.evaluations,
            "timesteps_descending": list(self.timesteps),
            "objective": "velocity",
            "sampler": "deterministic_ddim_eta_zero",
            "source_audio_or_model_weights_saved": False,
            "publication_authorized": False,
        }


def ddim_plan(schedule: DiffusionSchedule, *, evaluations: int) -> DDIMPlan:
    """Pick bounded integer timesteps without changing a trained noise schedule."""
    if (not isinstance(schedule, DiffusionSchedule)
            or type(schedule.steps) is not int
            or not 8 <= schedule.steps <= 256
            or type(evaluations) is not int
            or not 2 <= evaluations <= schedule.steps):
        raise ValueError("DDIM requires 2..T inference evaluations and T=8..256")
    denominator = evaluations - 1
    # Nearest integer, with exact endpoints and no floating rounding drift.
    selected = tuple(
        (index * (schedule.steps - 1) + denominator // 2) // denominator
        for index in range(evaluations - 1, -1, -1)
    )
    return DDIMPlan(schedule.steps, evaluations, selected, schedule.profile)


def velocity_ddim_transition(
    noisy: torch.Tensor, predicted_velocity: torch.Tensor,
    timestep: int, previous_timestep: int, schedule: DiffusionSchedule,
) -> torch.Tensor:
    """One deterministic DDIM eta=0 transition, WITHOUT amplitude clipping.

    x0 = sqrt(abar_t) * x_t - sqrt(1-abar_t) * v_t
    eps = sqrt(1-abar_t) * x_t + sqrt(abar_t) * v_t
    x_s = sqrt(abar_s) * x0 + sqrt(1-abar_s) * eps
    An endpoint previous_timestep=-1 means return the predicted clean x0.
    This is NOT the stochastic DDPM posterior and has no random noise.
    """
    if (type(timestep) is not int or type(previous_timestep) is not int
            or not 0 <= timestep < schedule.steps
            or not -1 <= previous_timestep < timestep):
        raise ValueError("DDIM reverse timestep order is invalid")
    if (not isinstance(noisy, torch.Tensor)
            or not isinstance(predicted_velocity, torch.Tensor)
            or noisy.ndim != 3 or len(noisy) < 1
            or noisy.device.type != "cpu"
            or schedule.alpha_bars.device.type != "cpu"):
        raise ValueError("DDIM transition needs bounded CPU audio tensors")
    indices = torch.full((len(noisy),), timestep, dtype=torch.long)
    x0, epsilon = decode_velocity(noisy, predicted_velocity, indices, schedule)
    if previous_timestep == -1:
        result = x0
    else:
        previous_alpha_bar = schedule.alpha_bars[previous_timestep]
        result = (
            previous_alpha_bar.sqrt() * x0
            + (1 - previous_alpha_bar).sqrt() * epsilon
        )
    if not torch.isfinite(result).all():
        raise ValueError("DDIM transition generated nonfinite values")
    return result


@torch.no_grad()
def sample_velocity_ddim(
    denoiser: nn.Module, schedule: DiffusionSchedule, caption: str,
    *, evaluations: int, seed: int = 0, guidance: float = 1.0,
    latent: bool = False, codec: nn.Module | None = None,
    return_preactivation: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """In-memory deterministic 8-kHz, 8192-frame v-trained inference.

    Different inference budgets share EXACTLY the same starting Gaussian
    tensor for a fixed seed; step skipping changes the trajectory. Guidance
    1 uses one v-model evaluation per selected timestep, other values use
    conditional and null predictions. Caller model modes and global RNG are
    restored; there is deliberately no output clamp or filesystem write.
    """
    plan = ddim_plan(schedule, evaluations=evaluations)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("DDIM seed must be unsigned 32-bit")
    if (type(guidance) not in (int, float) or not math.isfinite(guidance)
            or not 0 <= guidance <= 4):
        raise ValueError("DDIM guidance must be finite within [0,4]")
    if (type(latent) is not bool or type(return_preactivation) is not bool
            or (latent and codec is None) or (not latent and codec is not None)
            or (return_preactivation and not latent)):
        raise ValueError("DDIM latent decoding needs explicit matching codec")
    if not isinstance(denoiser, nn.Module):
        raise ValueError("DDIM denoiser must be a neural module")
    try:
        denoiser_device = next(denoiser.parameters()).device
        codec_device = next(codec.parameters()).device if codec is not None else None
    except StopIteration as exc:
        raise ValueError("DDIM requires parameterized neural modules") from exc
    if (denoiser_device.type != "cpu"
            or (codec_device is not None and codec_device.type != "cpu")
            or schedule.alpha_bars.device.type != "cpu"):
        raise ValueError("DDIM is CPU-only in the research prototype")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("DDIM needs a nonempty textual caption")
    tokens = ByteTextTokenizer().encode([caption])
    unconditioned = torch.zeros_like(tokens)
    shape = (1, LATENT_CHANNELS, LATENT_FRAMES) if latent else (1, 1, FRAMES)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    audio = torch.randn(shape, generator=generator, dtype=torch.float32)
    model_training = denoiser.training
    codec_training = codec.training if codec is not None else None
    denoiser.eval()
    if codec is not None:
        codec.eval()
    try:
        for i, timestep in enumerate(plan.timesteps):
            previous = (
                plan.timesteps[i + 1] if i + 1 < plan.evaluations else -1
            )
            step_tensor = torch.tensor([timestep], dtype=torch.long)
            velocity = denoiser(audio, step_tensor, tokens)
            if guidance != 1:
                null_velocity = denoiser(audio, step_tensor, unconditioned)
                velocity = null_velocity + guidance * (velocity - null_velocity)
            audio = velocity_ddim_transition(
                audio, velocity, timestep, previous, schedule
            )
        if codec is not None and return_preactivation:
            before, decoded = codec.decode_with_preactivation(audio)
            waveform, preactivation = decoded[0, 0], before[0, 0]
        elif codec is not None:
            waveform = codec.decode(audio)[0, 0]
        else:
            waveform = audio[0, 0]
        if (waveform.shape != (FRAMES,)
                or waveform.dtype != torch.float32
                or not torch.isfinite(waveform).all()):
            raise ValueError("DDIM output is not a finite mono 8192-frame waveform")
        output = waveform.detach().cpu()
        return ((output, preactivation.detach().cpu()) if return_preactivation
                else output)
    finally:
        denoiser.train(model_training)
        if codec is not None:
            codec.train(codec_training)
