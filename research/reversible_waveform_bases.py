"""Full-information orthonormal patch coordinates for raw waveform diffusion.

No coefficient is removed or quantized. Noise and velocity targets remain
in the physical waveform domain. The coordinate choice only changes the
input and output representation of the existing multiscale network.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn

from research.multiscale_waveform import (
    PATCH, MultiscaleWaveDenoiser, pack_waveform, unpack_waveform,
)

BASES = ("identity", "dct", "haar")


def orthonormal_basis(name: str) -> torch.Tensor:
    """Rows are analysis vectors, ordered low-frequency/coarse first."""
    if type(name) is not str or name not in BASES:
        raise ValueError("Unknown reversible waveform basis")
    if name == "identity":
        return torch.eye(PATCH, dtype=torch.float32)
    if name == "dct":
        k = torch.arange(PATCH, dtype=torch.float64)[:, None]
        n = torch.arange(PATCH, dtype=torch.float64)[None, :]
        basis = (2 / PATCH) ** .5 * torch.cos(math.pi / PATCH * k * (n + .5))
        basis[0] /= 2 ** .5
    else:
        basis = torch.zeros(PATCH, PATCH, dtype=torch.float64)
        basis[0] = PATCH ** -.5
        row = 1
        for width in (PATCH, PATCH // 2, PATCH // 4):
            for offset in range(0, PATCH, width):
                basis[row, offset:offset + width // 2] = width ** -.5
                basis[row, offset + width // 2:offset + width] = -(width ** -.5)
                row += 1
        if row != PATCH:
            raise ValueError("Unexpected Haar basis layout")
    return basis.float()


def _validate_basis(basis: torch.Tensor):
    if (not isinstance(basis, torch.Tensor) or basis.shape != (PATCH, PATCH)
            or basis.dtype != torch.float32 or basis.device.type != "cpu"
            or not torch.isfinite(basis).all()
            or not torch.allclose(basis @ basis.T, torch.eye(PATCH), atol=2e-6, rtol=0)):
        raise ValueError("Expected finite orthonormal CPU patch basis")


def analysis_waveform(wave: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    _validate_basis(basis)
    packed = pack_waveform(wave)
    return unpack_waveform(basis @ packed)


def synthesis_waveform(coordinates: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    _validate_basis(basis)
    packed = pack_waveform(coordinates)
    return unpack_waveform(basis.T @ packed)


def basis_sha256(basis: torch.Tensor) -> str:
    _validate_basis(basis)
    return hashlib.sha256(b"sonumio-patch-basis-v1\0" + basis.contiguous().numpy().tobytes()).hexdigest()


class ReversibleBasisDenoiser(nn.Module):
    """Same trainable parameter layout for each orthonormal patch basis.

    The fixed basis is reconstructed from the named protocol, not learned.
    It is registered as a nonpersistent buffer; the study reports its own
    independent fingerprint rather than conflating it with learned weights.
    """

    def __init__(self, basis_name: str):
        super().__init__()
        basis = orthonormal_basis(basis_name)
        self.basis_name = basis_name
        self.register_buffer("basis", basis, persistent=False)
        self.core = MultiscaleWaveDenoiser()

    def forward(self, audio, timestep, tokens):
        coordinate_wave = analysis_waveform(audio, self.basis)
        estimate = self.core(coordinate_wave, timestep, tokens)
        return synthesis_waveform(estimate, self.basis)

    @torch.no_grad()
    def align_initial_function(self):
        """Rotate boundary convolutions to represent the original function.

        If y = Bx, input weights become W B^T and output weights become
        B V (including output bias). The intervening network is unchanged.
        This makes waveform-domain initial functions equal up to floating
        point roundoff, while subsequent coordinate-wise AdamW may differ.
        Call exactly once on an unrotated model before constructing optimizer.
        """
        if getattr(self, "_initial_function_aligned", False):
            raise ValueError("Initial function already aligned")
        incoming = self.core.input
        outgoing = self.core.output[-1]
        incoming.weight.copy_(torch.einsum("oik,ji->ojk", incoming.weight, self.basis))
        outgoing.weight.copy_(torch.einsum("ij,jok->iok", self.basis, outgoing.weight))
        outgoing.bias.copy_(self.basis @ outgoing.bias)
        self._initial_function_aligned = True

    def describe(self):
        return {
            "basis": self.basis_name, "basis_sha256": basis_sha256(self.basis),
            "patch_size": PATCH, "coefficients_per_patch": PATCH,
            "coefficient_count_preserved": True,
            "trainable_basis": False, "quantized": False,
            "initial_function_aligned": getattr(self, "_initial_function_aligned", False),
        }
