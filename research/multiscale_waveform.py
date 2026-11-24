"""Experimental multiscale byte-text-conditioned waveform velocity denoiser.

This is a separate research architecture, not Sonumio-1 or a replacement
for existing checkpoints. It preserves the raw 8 kHz mono sample domain.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from research.latent_audio import FRAMES
from research.text_audio import TextEncoder, MAX_TOKENS
from research.velocity_prediction import decode_velocity
from research.wave_diffusion import ConditionedResidual, DiffusionSchedule

PATCH = 8
CONDITION_CHANNELS = 32
WIDTHS = (24, 32, 48)


def pack_waveform(audio: torch.Tensor) -> torch.Tensor:
    """Losslessly reorganize adjacent samples into channels (no decimation)."""
    if (not isinstance(audio, torch.Tensor) or audio.ndim != 3
            or audio.shape[1:] != (1, FRAMES) or not 1 <= len(audio) <= 16
            or audio.dtype != torch.float32 or audio.device.type != "cpu"
            or not torch.isfinite(audio).all()):
        raise ValueError("Expected finite CPU mono waveform batch")
    return audio.reshape(len(audio), FRAMES // PATCH, PATCH).transpose(1, 2).contiguous()


def unpack_waveform(patches: torch.Tensor) -> torch.Tensor:
    if (not isinstance(patches, torch.Tensor) or patches.ndim != 3
            or patches.shape[1:] != (PATCH, FRAMES // PATCH)
            or not 1 <= len(patches) <= 16 or patches.dtype != torch.float32
            or patches.device.type != "cpu" or not torch.isfinite(patches).all()):
        raise ValueError("Expected finite CPU waveform patches")
    return patches.transpose(1, 2).reshape(len(patches), 1, FRAMES)


class ConditionedLevel(nn.Module):
    """Project shared time/text context to the width of a feature level."""

    def __init__(self, width: int, dilations: tuple[int, ...]):
        super().__init__()
        self.condition = nn.Linear(CONDITION_CHANNELS, width)
        self.blocks = nn.ModuleList(ConditionedResidual(width, d) for d in dilations)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        condition = self.condition(context)
        for block in self.blocks:
            x = block(x, condition)
        return x


class Downsample(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.conv = nn.Conv1d(incoming, outgoing, 8, stride=4, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleMerge(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.project = nn.Conv1d(incoming, outgoing, 3, padding=1)
        self.merge = nn.Conv1d(outgoing * 2, outgoing, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.project(F.interpolate(x, size=skip.shape[-1], mode="nearest"))
        return self.merge(torch.cat((x, skip), dim=1))


class GlobalContext(nn.Module):
    """Bounded self-attention at 64 temporal positions, not at audio rate."""

    def __init__(self, width: int):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, 64, width))
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0)
        self.feed = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                  nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != 64:
            raise ValueError("Unexpected bottleneck temporal layout")
        seq = value.transpose(1, 2)
        qkv = self.norm(seq + self.position)
        attended, _ = self.attention(qkv, qkv, qkv, need_weights=False)
        seq = seq + attended
        return (seq + self.feed(seq)).transpose(1, 2)


class MultiscaleWaveDenoiser(nn.Module):
    """Raw waveform U-Net with patch packing, skips and low-rate attention."""

    def __init__(self):
        super().__init__()
        a, b, c = WIDTHS
        self.text_encoder = TextEncoder(CONDITION_CHANNELS)
        self.time_mlp = nn.Sequential(
            nn.Linear(CONDITION_CHANNELS, CONDITION_CHANNELS * 2), nn.SiLU(),
            nn.Linear(CONDITION_CHANNELS * 2, CONDITION_CHANNELS),
        )
        self.context_merge = nn.Sequential(
            nn.Linear(CONDITION_CHANNELS * 2, CONDITION_CHANNELS), nn.SiLU(),
            nn.Linear(CONDITION_CHANNELS, CONDITION_CHANNELS),
        )
        self.input = nn.Conv1d(PATCH, a, 5, padding=2)
        self.high = ConditionedLevel(a, (1, 2))
        self.down_high = Downsample(a, b)
        self.mid = ConditionedLevel(b, (1, 2, 4))
        self.down_mid = Downsample(b, c)
        self.low = ConditionedLevel(c, (1, 2, 4))
        self.global_context = GlobalContext(c)
        self.low_out = ConditionedLevel(c, (1, 2))
        self.up_mid = UpsampleMerge(c, b)
        self.mid_out = ConditionedLevel(b, (1, 2))
        self.up_high = UpsampleMerge(b, a)
        self.high_out = ConditionedLevel(a, (1, 2))
        self.output = nn.Sequential(nn.GroupNorm(8, a), nn.SiLU(),
                                    nn.Conv1d(a, PATCH, 3, padding=1))

    def forward(self, audio: torch.Tensor, timestep: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        packed = pack_waveform(audio)
        if (timestep.shape != (len(audio),) or timestep.dtype != torch.long
                or timestep.device.type != "cpu" or (timestep < 0).any()
                or (timestep >= 256).any() or tokens.shape != (len(audio), MAX_TOKENS)
                or tokens.device.type != "cpu"):
            raise ValueError("Invalid time or caption batch")
        half = CONDITION_CHANNELS // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        angle = timestep.float()[:, None] * frequency[None]
        time = self.time_mlp(torch.cat((angle.sin(), angle.cos()), dim=1))
        text = self.text_encoder(tokens)
        context = self.context_merge(torch.cat((time, text), dim=1))
        high = self.high(self.input(packed), context)
        mid = self.mid(self.down_high(high), context)
        low = self.low(self.down_mid(mid), context)
        low = self.low_out(self.global_context(low), context)
        mid_out = self.mid_out(self.up_mid(low, mid), context)
        high_out = self.high_out(self.up_high(mid_out, high), context)
        return unpack_waveform(self.output(high_out))


def spectral_clean_loss(
    noisy: torch.Tensor, predicted_velocity: torch.Tensor, clean: torch.Tensor,
    timestep: torch.Tensor, schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Multi-resolution magnitude loss on the v-derived clean estimate.

    No phase reconstruction or perceptual model. Each row is weighted by
    (1-alpha_bar), so the auxiliary emphasizes the high-noise region where
    an estimate must infer more signal structure. The ordinary velocity
    objective remains separately computed and unweighted.
    """
    if clean.shape != noisy.shape or not torch.isfinite(clean).all():
        raise ValueError("Invalid clean spectral objective target")
    estimate, _ = decode_velocity(noisy, predicted_velocity, timestep, schedule)
    total = torch.zeros(len(clean), dtype=clean.dtype)
    for size in (128, 512, 1024):
        window = torch.hann_window(size, dtype=clean.dtype)
        target = torch.stft(clean[:, 0], size, hop_length=size // 4,
                            window=window, return_complex=True).abs()
        actual = torch.stft(estimate[:, 0], size, hop_length=size // 4,
                            window=window, return_complex=True).abs()
        total = total + (torch.log1p(actual) - torch.log1p(target)).abs().mean(dim=(1, 2)) / 3
    weights = 1 - schedule.alpha_bars[timestep]
    result = (weights * total).mean()
    if not torch.isfinite(result):
        raise ValueError("Nonfinite spectral clean objective")
    return result
