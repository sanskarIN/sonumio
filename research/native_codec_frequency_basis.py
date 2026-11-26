"""Training-only frequency-subspace initialization for the native block codec.

No validation or test samples are used to fit an analysis basis. Keep
low and high DCT subspaces separate so a fixed latent allocation cannot
be consumed entirely by low-frequency energy. This is a bounded linear
research model, not a production codec.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch

from research.native_codec import (
    FRAMES, STRIDE, LATENT_CHANNELS, LATENT_FRAMES, MAX_BATCH, SAMPLE_RATE,
    HIGHBAND_HZ, validate_wave,
)
from research.native_codec_transform import NativeBlockCodec, dct_basis

MAX_TRAIN_CLIPS=36
POWER_FLOOR=1e-6
SPLIT_BIN=math.ceil(HIGHBAND_HZ*2*STRIDE/SAMPLE_RATE)


@dataclass(frozen=True,slots=True)
class BasisPlan:
    high_rank:int=4

    def __post_init__(self):
        if type(self.high_rank) is not int or not 1<=self.high_rank<LATENT_CHANNELS:
            raise ValueError("Invalid high-frequency latent allocation")

    @property
    def low_rank(self):
        return LATENT_CHANNELS-self.high_rank


def full_dct():
    n=torch.arange(STRIDE,dtype=torch.float64)
    k=torch.arange(STRIDE,dtype=torch.float64).reshape(-1,1)
    basis=torch.cos(math.pi*(n+.5)*k/STRIDE)*math.sqrt(2/STRIDE)
    basis[0]/=math.sqrt(2)
    return basis


def _leading(covariance,rank):
    values,vectors=torch.linalg.eigh(covariance)
    if not torch.isfinite(values).all() or not torch.isfinite(vectors).all():
        raise ValueError("Invalid training covariance eigensystem")
    order=torch.arange(len(values)-1,len(values)-rank-1,-1)
    selected=vectors[:,order].T.contiguous()
    # Canonical signs; repeated eigenvalues may still have nonunique bases.
    pivot=selected.abs().argmax(dim=1)
    signs=torch.sign(selected[torch.arange(rank),pivot])
    selected=selected*signs[:,None]
    return selected,values[order].clamp_min(0)


@torch.no_grad()
def fit_frequency_basis(training,plan=None):
    """Fit equal-clip power-weighted uncentered second moments.

    Uncentered moments preserve DC without storing a separate mean vector
    or hidden decoder side information. Every source clip contributes
    equally after division by its total power (with a fixed floor).
    """
    plan=BasisPlan() if plan is None else plan
    if not isinstance(plan,BasisPlan):
        raise ValueError("Expected BasisPlan")
    if (not isinstance(training,torch.Tensor) or training.ndim!=3
            or training.shape[1:]!=(1,FRAMES)
            or not 1<=len(training)<=MAX_TRAIN_CLIPS):
        raise ValueError("Invalid bounded training panel")
    basis=full_dct()
    cov=torch.zeros(STRIDE,STRIDE,dtype=torch.float64)
    for source in training.split(MAX_BATCH):
        validate_wave(source)
        blocks=source[:,0].double().reshape(len(source),LATENT_FRAMES,STRIDE)
        power=blocks.square().mean((1,2)).clamp_min(POWER_FLOOR)
        coeff=(blocks@basis.T)/power.sqrt()[:,None,None]
        flat=coeff.reshape(-1,STRIDE)
        cov+=flat.T@flat
    cov/=len(training)*LATENT_FRAMES
    low_values=cov[:SPLIT_BIN,:SPLIT_BIN]
    high_values=cov[SPLIT_BIN:,SPLIT_BIN:]
    lo,le=_leading(low_values,plan.low_rank)
    hi,he=_leading(high_values,plan.high_rank)
    result=torch.cat((lo@basis[:SPLIT_BIN],hi@basis[SPLIT_BIN:]),dim=0).float()
    gram=result@result.T
    if not torch.allclose(gram,torch.eye(LATENT_CHANNELS),rtol=0,atol=3e-6):
        raise ValueError("Fitted frequency basis is not orthonormal")
    report={"training_clips":len(training),"low_rank":plan.low_rank,
            "high_rank":plan.high_rank,"split_dct_bin":SPLIT_BIN,
            "low_subspace_energy":float(torch.trace(low_values)),
            "high_subspace_energy":float(torch.trace(high_values)),
            "retained_low_energy":float(le.sum()),"retained_high_energy":float(he.sum()),
            "validation_used":False,"center_vector_stored":False}
    return result,report


def initialized_codec(basis):
    if (not isinstance(basis,torch.Tensor) or basis.shape!=(LATENT_CHANNELS,STRIDE)
            or basis.device.type!="cpu" or basis.dtype!=torch.float32
            or not torch.isfinite(basis).all()
            or not torch.allclose(basis@basis.T,torch.eye(LATENT_CHANNELS),rtol=0,atol=3e-6)):
        raise ValueError("Expected finite orthonormal training basis")
    model=NativeBlockCodec()
    with torch.no_grad():
        model.analysis.copy_(basis)
        model.synthesis.copy_(basis.T)
    return model


def anchored_penalty(model,anchor):
    """Preserve initial frequency geometry without freezing optimization."""
    if not isinstance(model,NativeBlockCodec):
        raise ValueError("Expected native block codec")
    if (not isinstance(anchor,torch.Tensor) or anchor.shape!=model.analysis.shape
            or anchor.dtype!=torch.float32 or anchor.device.type!="cpu"
            or anchor.requires_grad or not torch.isfinite(anchor).all()):
        raise ValueError("Invalid fixed codec anchor")
    analysis=model.analysis
    synthesis=model.synthesis
    # Row-wise sum, not element mean: one unit-norm row has energy one.
    distance=((analysis-anchor).square().sum(1).mean()
              +(synthesis.T-anchor).square().sum(1).mean())*.5
    gram=analysis@analysis.T
    identity=torch.eye(LATENT_CHANNELS,dtype=torch.float32)
    orthogonality=(gram-identity).square().sum(1).mean()
    agreement=(analysis-synthesis.T).square().sum(1).mean()
    result={"anchor_distance":distance,"orthogonality_error":orthogonality,
            "analysis_synthesis_disagreement":agreement}
    if any(not torch.isfinite(x) for x in result.values()):
        raise ValueError("Nonfinite transform regularizer")
    return result
