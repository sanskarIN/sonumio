"""Stable, original velocity-prediction research objective for Sonumio DDPMs.

Independent from legacy epsilon-prediction experiments. v = sqrt(alpha_bar)
* eps - sqrt(1 - alpha_bar) * x0. The corresponding posterior uses an x0
estimate that avoids dividing epsilon errors by sqrt(alpha_t) at the near-
terminal cosine step. NOT evidence of perceptual fidelity or a model release.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from research.latent_audio import (
    FRAMES, LATENT_CHANNELS, LATENT_FRAMES, LatentResearchModel,
)
from research.text_audio import ByteTextTokenizer, TextWaveDenoiser
from research.wave_diffusion import DiffusionSchedule


def _validate_triplet(clean: torch.Tensor, noise: torch.Tensor,
                      timestep: torch.Tensor, schedule: DiffusionSchedule) -> None:
    if (clean.ndim != 3 or clean.shape != noise.shape or clean.dtype != torch.float32
            or noise.dtype != torch.float32 or timestep.shape != (len(clean),)
            or timestep.dtype != torch.long or not 0 <= timestep.min()
            or not (timestep < schedule.steps).all()
            or clean.device != noise.device or clean.device != timestep.device
            or not torch.isfinite(clean).all() or not torch.isfinite(noise).all()):
        raise ValueError('Velocity objective requires finite aligned float32 tensors and timesteps')


def velocity_target(clean: torch.Tensor, noise: torch.Tensor,
                    timestep: torch.Tensor, schedule: DiffusionSchedule) -> torch.Tensor:
    """v_t target; coefficients broadcast over channel and time axes."""
    _validate_triplet(clean, noise, timestep, schedule)
    abar = schedule.alpha_bars[timestep].view(-1, 1, 1)
    return abar.sqrt() * noise - (1 - abar).sqrt() * clean


def decode_velocity(noisy: torch.Tensor, v_pred: torch.Tensor,
                    timestep: torch.Tensor, schedule: DiffusionSchedule
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover x0 and epsilon without an unstable 1/sqrt(alpha) factor."""
    _validate_triplet(noisy, v_pred, timestep, schedule)
    abar = schedule.alpha_bars[timestep].view(-1, 1, 1)
    x0 = abar.sqrt() * noisy - (1 - abar).sqrt() * v_pred
    epsilon = (1 - abar).sqrt() * noisy + abar.sqrt() * v_pred
    return x0, epsilon


@torch.no_grad()
def posterior_v_step(noisy: torch.Tensor, v_pred: torch.Tensor, timestep: int,
                     schedule: DiffusionSchedule, *,
                     generator: torch.Generator) -> torch.Tensor:
    """DDPM posterior mean via predicted x0; variance zero at t=0."""
    if (type(timestep) is not int or not 0 <= timestep < schedule.steps
            or not isinstance(generator, torch.Generator)):
        raise ValueError('Invalid reverse diffusion timestep or RNG')
    indices = torch.full((len(noisy),), timestep, dtype=torch.long, device=noisy.device)
    x0, _ = decode_velocity(noisy, v_pred, indices, schedule)
    if timestep == 0:
        result = x0
    else:
        beta = schedule.betas[timestep]
        alpha = schedule.alphas[timestep]
        bar = schedule.alpha_bars[timestep]
        prev_bar = schedule.alpha_bars[timestep - 1]
        denominator = 1 - bar
        mean = (
            beta * prev_bar.sqrt() / denominator * x0
            + alpha.sqrt() * (1 - prev_bar) / denominator * noisy
        )
        posterior_variance = beta * (1 - prev_bar) / denominator
        result = mean + posterior_variance.sqrt() * torch.randn(
            noisy.shape, generator=generator, device=noisy.device, dtype=noisy.dtype
        )
    if not torch.isfinite(result).all():
        raise ValueError('Unbounded/nonfinite v-DDPM posterior')
    return result


@torch.no_grad()
def sample_velocity(denoiser: nn.Module, schedule: DiffusionSchedule,
                    caption: str, *, seed: int = 0, guidance: float = 1.0,
                    latent: bool = False, codec: nn.Module | None = None
                    ) -> torch.Tensor:
    """Private in-memory waveform: no clipping, saving, or checkpoint output.

    Use matching v-trained weights only. Never call with an epsilon-trained
    denoiser; the two targets and reverse updates are mathematically distinct.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Sampling seed must be unsigned 32-bit')
    if type(guidance) not in (int, float) or not math.isfinite(guidance) or not 0 <= guidance <= 4:
        raise ValueError('Guidance must be finite within [0, 4]')
    if type(latent) is not bool or (latent and codec is None) or (not latent and codec is not None):
        raise ValueError('Latent generation needs an explicit frozen decoder')
    if not 8 <= schedule.steps <= 256:
        raise ValueError('Research sampler needs 8..256 diffusion steps')
    if (schedule.betas.device.type != 'cpu'
            or next(denoiser.parameters()).device.type != 'cpu'
            or (codec is not None and next(codec.parameters()).device.type != 'cpu')):
        raise ValueError('Research v-DDPM sampler supports CPU only')
    generator = torch.Generator().manual_seed(seed)
    shape = (1, LATENT_CHANNELS, LATENT_FRAMES) if latent else (1, 1, FRAMES)
    audio = torch.randn(shape, generator=generator)
    tokens = ByteTextTokenizer().encode([caption])
    null = torch.zeros_like(tokens)
    was_training = denoiser.training
    decoder_was_training = codec.training if codec is not None else None
    denoiser.eval()
    if codec is not None:
        codec.eval()
    try:
        for index in range(schedule.steps - 1, -1, -1):
            timestep = torch.tensor([index], dtype=torch.long)
            v_pred = denoiser(audio, timestep, tokens)
            if guidance != 1:
                blank = denoiser(audio, timestep, null)
                v_pred = blank + guidance * (v_pred - blank)
            audio = posterior_v_step(
                audio, v_pred, index, schedule, generator=generator)
        result = codec.decode(audio)[0, 0] if codec is not None else audio[0, 0]
        if result.shape != (FRAMES,) or not torch.isfinite(result).all():
            raise ValueError('Invalid v-predicted generated waveform')
        return result.cpu()
    finally:
        denoiser.train(was_training)
        if codec is not None:
            codec.train(decoder_was_training)
