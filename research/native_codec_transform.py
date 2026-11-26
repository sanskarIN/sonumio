"""Block-transform initialized neural codec for native 24 kHz audio.

The architecture uses a trainable analysis/synthesis projection with
the same 16 x 64 latent shape as the convolutional baseline. No residual
waveform, hidden skip connection or side channel reaches the decoder.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from research.native_codec import (
    FRAMES, STRIDE, LATENT_CHANNELS, LATENT_FRAMES,
    validate_wave, validate_latent,
)

DCT_BINS = (*range(12), 72, 104, 144, 176)
INITIALIZATIONS = ("random_orthogonal", "frequency_stratified")
SCALE = math.sqrt(STRIDE)


def dct_basis(indices=DCT_BINS):
    if (type(indices) is not tuple or len(indices)!=LATENT_CHANNELS
            or len(set(indices))!=len(indices)
            or any(type(i) is not int or not 0<=i<STRIDE for i in indices)):
        raise ValueError("Invalid fixed DCT basis indices")
    n=torch.arange(STRIDE,dtype=torch.float64)
    k=torch.tensor(indices,dtype=torch.float64).reshape(-1,1)
    basis=torch.cos(math.pi*(n+.5)*k/STRIDE)*math.sqrt(2/STRIDE)
    basis[k[:,0]==0]/=math.sqrt(2)
    return basis.float()


def initial_basis(kind, seed):
    if kind not in INITIALIZATIONS or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("Invalid transform codec initialization")
    if kind=="frequency_stratified":
        return dct_basis()
    generator=torch.Generator(device="cpu").manual_seed(seed)
    raw=torch.randn(STRIDE,LATENT_CHANNELS,generator=generator,dtype=torch.float64)
    q,r=torch.linalg.qr(raw,mode="reduced")
    # Remove QR sign ambiguity before taking the transposed row basis.
    q=q*torch.where(torch.diagonal(r)<0,-1.,1.)
    return q.T.contiguous().float()


class NativeBlockCodec(nn.Module):
    """Trainable rank-16 block codec with explicit bounded latent transport."""

    def __init__(self, kind="frequency_stratified", seed=17):
        super().__init__()
        basis=initial_basis(kind,seed)
        self.analysis=nn.Parameter(basis.clone())
        self.synthesis=nn.Parameter(basis.T.contiguous().clone())

    def encode(self,wave):
        validate_wave(wave)
        blocks=wave[:,0].reshape(len(wave),LATENT_FRAMES,STRIDE)
        coefficients=blocks@self.analysis.T/SCALE
        if not torch.isfinite(coefficients).all():
            raise ValueError("Nonfinite block coefficients")
        latent=coefficients.clamp(-1,1).transpose(1,2).contiguous()
        validate_latent(latent)
        return latent

    def decode_raw(self,latent):
        validate_latent(latent)
        blocks=(latent.transpose(1,2)*SCALE)@self.synthesis.T
        result=blocks.reshape(len(latent),1,FRAMES)
        validate_wave(result,normalized=False)
        return result

    def forward(self,wave):
        return self.decode_raw(self.encode(wave)).tanh()


def projection_diagnostics(model):
    if not isinstance(model,NativeBlockCodec):
        raise ValueError("Expected native block codec")
    with torch.no_grad():
        analysis=model.analysis
        synthesis=model.synthesis
        if not torch.isfinite(analysis).all() or not torch.isfinite(synthesis).all():
            raise ValueError("Nonfinite transform state")
        identity=torch.eye(LATENT_CHANNELS,dtype=torch.float32)
        gram=analysis@analysis.T
        # These describe the learned linear operators before clipping/tanh.
        # They do not establish invertibility or perceptual fidelity.
        return {
            "analysis_gram_rmse":float((gram-identity).square().mean().sqrt()),
            "analysis_synthesis_rmse":float((analysis-synthesis.T).square().mean().sqrt()),
            "analysis_frobenius":float(analysis.norm()),
            "synthesis_frobenius":float(synthesis.norm()),
        }
