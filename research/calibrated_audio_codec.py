"""Experimental residual codec and train-only latent calibration.

The representation remains 16 channels by 128 frames for 8192 mono samples:
fourfold scalar compression, not 64-fold information compression. No bitstream,
quantization, entropy coding, pretrained weights or checkpoint format is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.latent_audio import FRAMES, LATENT_CHANNELS, LATENT_FRAMES

MIN_CHANNEL_STD = 1e-3


def _wave_batch(wave: torch.Tensor) -> None:
    if (not isinstance(wave, torch.Tensor) or wave.ndim != 3
            or not 1 <= len(wave) <= 128 or wave.shape[1:] != (1, FRAMES)
            or wave.dtype != torch.float32 or wave.device.type != "cpu"
            or not torch.isfinite(wave).all()):
        raise ValueError("Expected finite CPU float32 waveform batch")


def _latent_batch(latent: torch.Tensor) -> None:
    if (not isinstance(latent, torch.Tensor) or latent.ndim != 3
            or not 1 <= len(latent) <= 128
            or latent.shape[1:] != (LATENT_CHANNELS, LATENT_FRAMES)
            or latent.dtype != torch.float32 or latent.device.type != "cpu"
            or not torch.isfinite(latent).all()):
        raise ValueError("Expected finite CPU float32 latent batch")


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 0.5 * self.net(value)


class DownBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.net = nn.Sequential(
            ResidualUnit(incoming),
            nn.Conv1d(incoming, outgoing, 8, stride=4, padding=2),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class UpBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 7, padding=3),
            nn.SiLU(),
            ResidualUnit(outgoing, dilation=2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Explicit interpolation + stride-one convolution avoids the uneven
        # overlap mechanism of transposed convolution. It does not guarantee
        # an artifact-free reconstruction or adequate high-frequency detail.
        return self.net(F.interpolate(value, scale_factor=4, mode="nearest"))


class ResidualAudioCodec(nn.Module):
    """CPU research codec with the legacy latent shape and resize-conv decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, 7, padding=3), nn.SiLU(),
            DownBlock(8, 12), DownBlock(12, 16), DownBlock(16, LATENT_CHANNELS),
            ResidualUnit(LATENT_CHANNELS, dilation=2),
            nn.Conv1d(LATENT_CHANNELS, LATENT_CHANNELS, 1),
        )
        self.decoder = nn.Sequential(
            ResidualUnit(LATENT_CHANNELS, dilation=2),
            UpBlock(LATENT_CHANNELS, 16), UpBlock(16, 12), UpBlock(12, 8),
            nn.Conv1d(8, 1, 7, padding=3),
        )

    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        _wave_batch(wave)
        encoded = self.encoder(wave)
        _latent_batch(encoded)
        return encoded

    def decode_with_preactivation(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _latent_batch(latent)
        pre = self.decoder(latent)
        _wave_batch(pre)
        return pre, pre.tanh()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode_with_preactivation(latent)[1]

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(wave))


@dataclass(frozen=True)
class LatentCalibration:
    """Channelwise affine calibration fit on unique training recordings only."""

    mean: torch.Tensor
    scale: torch.Tensor
    std: torch.Tensor
    training_recordings: int

    @classmethod
    def fit(cls, training_latents: torch.Tensor) -> LatentCalibration:
        _latent_batch(training_latents)
        if len(training_latents) < 2 or training_latents.requires_grad:
            raise ValueError("Calibration needs at least two detached training recordings")
        data = training_latents.to(torch.float64)
        mean = data.mean(dim=(0, 2), keepdim=True).float()
        std = data.std(dim=(0, 2), correction=0, keepdim=True).float()
        scale = std.clamp_min(MIN_CHANNEL_STD)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("Invalid calibration moments")
        return cls(mean, scale, std, len(training_latents))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        _latent_batch(value)
        output = (value - self.mean) / self.scale
        _latent_batch(output)
        return output

    def restore(self, value: torch.Tensor) -> torch.Tensor:
        _latent_batch(value)
        output = value * self.scale + self.mean
        _latent_batch(output)
        return output

    def describe(self) -> dict:
        return {
            "training_recordings": self.training_recordings,
            "channel_mean": self.mean.flatten().tolist(),
            "channel_std": self.std.flatten().tolist(),
            "channel_scale": self.scale.flatten().tolist(),
            "minimum_scale": MIN_CHANNEL_STD,
            "floored_channels": int((self.std < MIN_CHANNEL_STD).sum()),
        }


class CalibratedCodec(nn.Module):
    """Adapter mapping standard-space diffusion latents to native codec space.

    Calibration buffers stay attached to the inference module. The fitted codec
    is frozen and never learns from validation data or generated waveforms.
    """

    def __init__(self, codec: nn.Module, calibration: LatentCalibration | None):
        super().__init__()
        self.codec = codec.eval().requires_grad_(False)
        self.register_buffer("offset", calibration.mean.clone() if calibration else
                             torch.zeros(1, LATENT_CHANNELS, 1))
        self.register_buffer("scale", calibration.scale.clone() if calibration else
                             torch.ones(1, LATENT_CHANNELS, 1))
        self.eval()

    def encode(self, wave: torch.Tensor) -> torch.Tensor:
        _wave_batch(wave)
        encoded = (self.codec.encode(wave) - self.offset) / self.scale
        _latent_batch(encoded)
        return encoded

    def native_latent(self, latent: torch.Tensor) -> torch.Tensor:
        _latent_batch(latent)
        result = latent * self.scale + self.offset
        _latent_batch(result)
        return result

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(self.native_latent(latent))

    def decode_with_preactivation(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.codec.decode_with_preactivation(self.native_latent(latent))


@torch.no_grad()
def latent_statistics(value: torch.Tensor) -> dict:
    _latent_batch(value)
    data = value.to(torch.float64)
    means = data.mean(dim=(0, 2))
    std = data.std(dim=(0, 2), correction=0)
    report = {
        "channel_mean": means.tolist(),
        "channel_std": std.tolist(),
        "maximum_absolute_channel_mean": float(means.abs().max()),
        "mean_absolute_channel_std_minus_one": float((std - 1).abs().mean()),
        "rms": float(data.square().mean().sqrt()),
        "absolute_value_over_three_fraction": float((data.abs() > 3).double().mean()),
    }
    if not all(math.isfinite(x) for x in means.tolist() + std.tolist()):
        raise ValueError("Nonfinite latent statistics")
    return report


@torch.no_grad()
def reconstruction_diagnostics(original: torch.Tensor, decoded: torch.Tensor) -> list[dict]:
    """Per-recording numerical fidelity, not a human listening metric."""
    _wave_batch(original)
    _wave_batch(decoded)
    if original.shape != decoded.shape:
        raise ValueError("Reconstruction batch shape mismatch")
    ref = original[:, 0].double()
    out = decoded[:, 0].double()
    error = out - ref
    reference_energy = ref.square().sum(dim=1)
    error_energy = error.square().sum(dim=1)
    # Relative error remains undefined for silence rather than reporting a
    # misleading perfect/infinite signal-to-noise score.
    ref_rms = ref.square().mean(dim=1).sqrt()
    out_rms = out.square().mean(dim=1).sqrt()
    spectral = torch.zeros(len(ref), dtype=torch.float64)
    for fft in (128, 512, 1024):
        window = torch.hann_window(fft, dtype=torch.float64)
        a = torch.stft(ref, fft, hop_length=fft // 4, window=window, return_complex=True).abs()
        b = torch.stft(out, fft, hop_length=fft // 4, window=window, return_complex=True).abs()
        spectral += (torch.log1p(a) - torch.log1p(b)).abs().mean(dim=(1, 2)) / 3
    rows = []
    for i in range(len(ref)):
        energy = float(reference_energy[i])
        dot = float((ref[i] * out[i]).sum())
        norm = float(ref[i].norm() * out[i].norm())
        rows.append({
            "mae": float(error[i].abs().mean()),
            "relative_squared_error": float(error_energy[i]) / energy if energy > 1e-12 else None,
            "reference_rms": float(ref_rms[i]),
            "decoded_rms": float(out_rms[i]),
            "rms_ratio": float(out_rms[i] / ref_rms[i]) if ref_rms[i] > 1e-8 else None,
            "waveform_cosine": max(-1.0, min(1.0, dot / norm)) if norm > 1e-12 else None,
            "multiresolution_log_magnitude_mae": float(spectral[i]),
            "decoded_clipping_fraction": float((out[i].abs() >= 0.999).double().mean()),
            "decoded_dc": float(out[i].mean()),
        })
    return rows
