"""Native 24 kHz mono codec research; not a deployed codec or model.

A lossless polyphase rearrangement precedes the learned encoder. It does
not discard the upper frequency bands as 8 kHz resampling would. The
learned bottleneck is lossy, and reconstruction quality must be measured.
"""
from __future__ import annotations

import math
import torch
from torch import nn

SAMPLE_RATE = 24_000
FRAMES = 12_288
LATENT_CHANNELS = 16
STRIDE = 192
LATENT_FRAMES = FRAMES // STRIDE
MAX_BATCH = 8
FFT_SIZES = (256, 512, 1024)
HIGHBAND_HZ = 4_500
EPS = 1e-8


def validate_wave(wave, *, normalized=True):
    if (not isinstance(wave, torch.Tensor) or wave.device.type != "cpu"
            or wave.dtype != torch.float32 or wave.ndim != 3
            or wave.shape[1:] != (1, FRAMES) or not 1 <= len(wave) <= MAX_BATCH
            or not torch.isfinite(wave).all()):
        raise ValueError("Expected finite CPU float32 24 kHz waveform batch")
    if normalized and wave.abs().max() > 1.0001:
        raise ValueError("Waveform exceeds normalized PCM range")
    if not normalized and wave.abs().max() > 16:
        raise ValueError("Waveform exceeds diagnostic amplitude bound")


def polyphase_pack(wave):
    validate_wave(wave, normalized=False)
    return wave.reshape(len(wave), -1, 3).transpose(1, 2).contiguous()


def polyphase_unpack(phases):
    if (not isinstance(phases, torch.Tensor) or phases.device.type != "cpu"
            or phases.dtype != torch.float32 or phases.ndim != 3
            or phases.shape[1:] != (3, FRAMES//3)
            or not 1 <= len(phases) <= MAX_BATCH or not torch.isfinite(phases).all()):
        raise ValueError("Invalid native codec polyphase tensor")
    return phases.transpose(1, 2).contiguous().reshape(len(phases), 1, FRAMES)


def validate_latent(latent):
    if (not isinstance(latent, torch.Tensor) or latent.device.type != "cpu"
            or latent.dtype != torch.float32 or latent.ndim != 3
            or latent.shape[1:] != (LATENT_CHANNELS, LATENT_FRAMES)
            or not 1 <= len(latent) <= MAX_BATCH or not torch.isfinite(latent).all()
            or latent.abs().max() > 1):
        raise ValueError("Invalid bounded native codec latent")


class Native24kCodec(nn.Module):
    """Stride-192 convolutional autoencoder, 125 latent steps per second."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 12, 7, padding=3), nn.SiLU(),
            nn.Conv1d(12, 16, 8, stride=4, padding=2), nn.SiLU(),
            nn.Conv1d(16, 24, 8, stride=4, padding=2), nn.SiLU(),
            nn.Conv1d(24, LATENT_CHANNELS, 8, stride=4, padding=2), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv1d(LATENT_CHANNELS, 24, 7, padding=3), nn.SiLU(),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv1d(24, 16, 7, padding=3), nn.SiLU(),
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv1d(16, 12, 7, padding=3), nn.SiLU(),
            nn.Conv1d(12, 3, 7, padding=3),
        )

    def encode(self, wave):
        validate_wave(wave)
        latent = self.encoder(polyphase_pack(wave))
        validate_latent(latent)
        return latent

    def decode_raw(self, latent):
        validate_latent(latent)
        pre = polyphase_unpack(self.decoder(latent))
        validate_wave(pre, normalized=False)
        return pre

    def forward(self, wave):
        return self.decode_raw(self.encode(wave)).tanh()


def _mag(wave, fft):
    window = torch.hann_window(fft, dtype=torch.float32)
    return torch.stft(wave[:, 0], fft, hop_length=fft//4, window=window,
                      center=False, return_complex=True).abs() / math.sqrt(fft)


def native_metrics(source, restored):
    """Per-example numeric diagnostics; never interprets a zero band as fidelity.

    High-band error is normalized by total source energy rather than by
    potentially zero high-band energy. Report source/output band fractions
    separately. All spectral metrics use the original 24 kHz clock.
    """
    validate_wave(source)
    validate_wave(restored, normalized=False)
    if source.shape != restored.shape:
        raise ValueError("Native reconstruction batch mismatch")
    diff = restored - source
    mae = diff.abs().mean((1, 2))
    mse = diff.square().mean((1, 2))
    spec_l1, convergence, high_error, source_fraction, restored_fraction = [], [], [], [], []
    for fft in FFT_SIZES:
        a, b = _mag(source, fft), _mag(restored, fft)
        high = torch.fft.rfftfreq(fft, d=1/SAMPLE_RATE) >= HIGHBAND_HZ
        # Parseval-consistent one-sided weighting for magnitude energy.
        weights = torch.ones(fft//2+1, dtype=torch.float32)
        weights[1:-1] = 2
        weights = weights.view(1, -1, 1)
        energy_a = (a.square()*weights).sum((1, 2))
        energy_b = (b.square()*weights).sum((1, 2))
        band_a = (a[:, high].square()*weights[:, high]).sum((1, 2))
        band_b = (b[:, high].square()*weights[:, high]).sum((1, 2))
        residual = (a-b).square()*weights
        spec_l1.append((torch.log1p(a)-torch.log1p(b)).abs().mean((1, 2)))
        # Smooth near-zero denominators and numerator to keep gradients finite.
        convergence.append((torch.sqrt(residual.sum((1, 2))+EPS)-math.sqrt(EPS))
                           / torch.sqrt(energy_a+EPS))
        high_error.append((torch.sqrt(residual[:, high].sum((1, 2))+EPS)-math.sqrt(EPS))
                          / torch.sqrt(energy_a+EPS))
        source_fraction.append(band_a/energy_a.clamp_min(EPS))
        restored_fraction.append(band_b/energy_b.clamp_min(EPS))
    metrics = {
        "waveform_mae": mae, "waveform_mse": mse,
        "log_spectral_mae": torch.stack(spec_l1).mean(0),
        "spectral_convergence": torch.stack(convergence).mean(0),
        "highband_magnitude_error": torch.stack(high_error).mean(0),
        "source_highband_fraction": torch.stack(source_fraction).mean(0),
        "restored_highband_fraction": torch.stack(restored_fraction).mean(0),
        "snr_db": 10*torch.log10((source.double().square().mean((1, 2))+1e-10)
                                /(diff.double().square().mean((1, 2))+1e-10)),
    }
    if any(not torch.isfinite(value).all() for value in metrics.values()):
        raise ValueError("Nonfinite native codec metric")
    return metrics


def native_loss(source, restored, highband_weight):
    if (type(highband_weight) not in (int, float) or not math.isfinite(highband_weight)
            or not 0 <= highband_weight <= 2):
        raise ValueError("Invalid high-band loss weight")
    values = native_metrics(source, restored)
    loss = (values["waveform_mae"] + .5*values["log_spectral_mae"]
            + .15*values["spectral_convergence"]
            + highband_weight*values["highband_magnitude_error"]).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite native codec objective")
    return loss
