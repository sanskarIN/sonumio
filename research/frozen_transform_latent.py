"""Frozen block-transform bridge to the existing continuous latent denoiser.

Only 16 coefficients per 64-sample block are supported, preserving the
16 x 128 latent layout. Unlike the quantized payload experiment, this module
uses continuous float32 coordinates; no bitstream or new model release.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from research.latent_audio import FRAMES, LATENT_CHANNELS, LATENT_FRAMES
from research.transform_audio_codec import TransformFit, dct_basis

METHODS = ("dct_low", "dct_ranked", "pca")
BLOCK = 64
SCALE_FLOOR = 1e-3


def _batch(value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if (not isinstance(value, torch.Tensor) or value.ndim != 3
            or not 1 <= len(value) <= 128 or value.shape[1:] != shape
            or value.dtype != torch.float32 or value.device.type != "cpu"
            or not torch.isfinite(value).all()):
        raise ValueError("Expected bounded finite CPU float32 transform batch")
    return value


class FrozenTransformCodec(nn.Module):
    """Fixed orthogonal projection and optional train-fitted latent scaling."""

    def __init__(self, fitted: TransformFit, method: str, training_waves: torch.Tensor,
                 *, calibrate: bool):
        super().__init__()
        if method not in METHODS or type(calibrate) is not bool:
            raise ValueError("Unsupported continuous transform latent configuration")
        _batch(training_waves, (1, FRAMES))
        if len(training_waves) < 2 or training_waves.requires_grad:
            raise ValueError("Calibration requires detached training recordings")
        if method == "pca":
            basis = fitted.pca_basis[:, :LATENT_CHANNELS]
            center = fitted.pca_center
        else:
            basis = dct_basis()[:, list(fitted.indices(method, LATENT_CHANNELS))]
            center = np.zeros(BLOCK)
        # The existing research DDIM sampler expects a parameterized decoder.
        # Register the fixed matrix as a non-trainable parameter; no optimizer
        # may update this projection. The remaining state is explicit buffers.
        self.basis = nn.Parameter(torch.tensor(basis.copy(), dtype=torch.float32), requires_grad=False)
        self.register_buffer("center", torch.tensor(center.copy(), dtype=torch.float32))
        self.method = method
        self.calibrate = calibrate
        with torch.no_grad():
            native = self._native_encode(training_waves).double()
            offset = native.mean(dim=(0, 2), keepdim=True).float()
            std = native.std(dim=(0, 2), correction=0, keepdim=True).float()
            scale = std.clamp_min(SCALE_FLOOR)
        self.register_buffer("training_mean", offset)
        self.register_buffer("training_std", std)
        self.register_buffer("offset", offset.clone() if calibrate else torch.zeros_like(offset))
        self.register_buffer("scale", scale.clone() if calibrate else torch.ones_like(scale))
        self.training_recordings = len(training_waves)
        self.eval()

    def _native_encode(self, waveform: torch.Tensor) -> torch.Tensor:
        wave = _batch(waveform, (1, FRAMES))
        blocks = wave.reshape(len(wave), LATENT_FRAMES, BLOCK) - self.center
        native = (blocks @ self.basis).transpose(1, 2).contiguous()
        return _batch(native, (LATENT_CHANNELS, LATENT_FRAMES))

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return _batch((self._native_encode(waveform) - self.offset) / self.scale,
                      (LATENT_CHANNELS, LATENT_FRAMES))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        value = _batch(latent, (LATENT_CHANNELS, LATENT_FRAMES))
        native = value * self.scale + self.offset
        blocks = native.transpose(1, 2) @ self.basis.T + self.center
        # Do not clip: reconstruction/generation metrics must see overrange
        # samples. The separate source probe consumes explicitly clipped PCM.
        return _batch(blocks.reshape(len(value), 1, FRAMES), (1, FRAMES))

    def decode_with_preactivation(self, latent: torch.Tensor):
        # No tanh or nonlinear output activation is used by this transform.
        wave = self.decode(latent)
        return wave, wave

    def describe(self) -> dict:
        return {
            "method": self.method, "calibrated": self.calibrate,
            "training_recordings": self.training_recordings,
            "shape": [LATENT_CHANNELS, LATENT_FRAMES],
            "training_channel_mean": self.training_mean.flatten().tolist(),
            "training_channel_std": self.training_std.flatten().tolist(),
            "applied_scale": self.scale.flatten().tolist(),
            "scale_floor": SCALE_FLOOR,
            "floored_channels": int((self.training_std < SCALE_FLOOR).sum()) if self.calibrate else 0,
            "state_sha256": state_sha256(self),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "quantized": False,
        }


def state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256(b"sonumio-frozen-transform-state-v1\0")
    for name, value in sorted(module.state_dict().items()):
        if value.dtype != torch.float32 or value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("Invalid research module state")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def latent_statistics(codec: FrozenTransformCodec, waves: torch.Tensor) -> dict:
    latents = torch.cat([codec.encode(batch) for batch in waves.split(8)]).double()
    means = latents.mean((0, 2))
    std = latents.std((0, 2), correction=0)
    return {
        "maximum_absolute_channel_mean": float(means.abs().max()),
        "mean_absolute_channel_std_minus_one": float((std - 1).abs().mean()),
        "rms": float(latents.square().mean().sqrt()),
        "absolute_value_over_three_fraction": float((latents.abs() > 3).double().mean()),
    }
