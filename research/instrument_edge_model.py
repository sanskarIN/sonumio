"""Absolute-duration edge parametrization for sustained pitched synthesis.

This experiment preserves the original model's trainable parameter shapes
and oscillator path. It only transforms the two learned envelope controls
from normalized duration fractions into log-scaled sample durations.
No reference-instrument-specific envelope constants are supplied.
"""
from __future__ import annotations

from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from research.instrument_harmonic_model import (
    HarmonicParameters, instrument_loss, _wave_pair, _relative_wave_mse, _spectral_error,
)
from research.instrument_long_model import WindowedHarmonicInstrument, MAX_NOTE_FRAMES
from research.instrument_note_data import MIN_FRAMES, MAX_BATCH

PROTOCOL="sonumio-absolute-instrument-edge-v1"
COORDINATES=("fraction","absolute")
OBJECTIVES=("waveform","envelope")
ATTACK_LIMITS=(16.,7200.)
RELEASE_LIMITS=(32.,12000.)
ENVELOPE_BLOCK=128
ENVELOPE_WEIGHT=.35


def absolute_edge_frames(fractions,frames,limits):
    """Map original sigmoid controls to a smooth, bounded log duration.

    The original controls encode u in [0,1] via .0005 + .499*u.
    We recover u, predict a duration in absolute samples and constrain
    it to less than half the note. Durations are differentiable, not
    rounded to integer samples.
    """
    if (not isinstance(fractions,torch.Tensor) or fractions.device.type!="cpu"
            or fractions.dtype!=torch.float32 or fractions.ndim!=1
            or not 1<=fractions.numel()<=MAX_BATCH
            or not bool(torch.isfinite(fractions).all())
            or bool(((fractions<.0004999)|(fractions>.4995001)).any())
            or type(frames) is not int or not MIN_FRAMES<=frames<=MAX_NOTE_FRAMES
            or limits not in (ATTACK_LIMITS,RELEASE_LIMITS)):
        raise ValueError("Invalid absolute edge controls")
    low,high=limits
    # Cap before exponentiation, avoiding saturated clamp gradients for
    # short notes. The domain remains physically feasible at every duration.
    ceiling=min(high,(frames-1)*.4995)
    unit=((fractions-.0005)/.499).clamp(0.,1.)
    return low*torch.exp(unit*math.log(ceiling/low))


class AbsoluteEdgeInstrument(WindowedHarmonicInstrument):
    """Identical parameter count, with edge control in absolute samples."""

    def predict_parameters(self,instruments,pitches,velocities,frames):
        prior=super().predict_parameters(instruments,pitches,velocities,frames)
        attack=absolute_edge_frames(prior.attack_fraction,frames,ATTACK_LIMITS)
        release=absolute_edge_frames(prior.release_fraction,frames,RELEASE_LIMITS)
        result=replace(prior,attack_fraction=attack/(frames-1),release_fraction=release/(frames-1))
        return result


def initialize_edges(width,seed,coordinate):
    if coordinate not in COORDINATES or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError("Invalid edge model initialization")
    model=AbsoluteEdgeInstrument if coordinate=="absolute" else WindowedHarmonicInstrument
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return model(width)


def _blocks(wave,block):
    if (not isinstance(wave,torch.Tensor) or wave.device.type!="cpu" or wave.dtype!=torch.float32
            or wave.ndim!=2 or not 1<=wave.shape[0]<=MAX_BATCH
            or not 1<=wave.shape[1]<=MAX_NOTE_FRAMES
            or type(block) is not int or not 16<=block<=1024
            or not bool(torch.isfinite(wave).all())):
        raise ValueError("Invalid envelope waveform or block")
    count=wave.shape[1]
    full,tail=divmod(count,block)
    if full:
        body=wave[:,:full*block].reshape(wave.shape[0],full,block).square().mean(dim=2)
    else:
        body=wave.new_empty((wave.shape[0],0))
    if tail:
        body=torch.cat((body,wave[:,full*block:].square().mean(dim=1,keepdim=True)),dim=1)
    weights=wave.new_full((full+int(bool(tail)),),float(block))
    if tail:
        weights[-1]=tail
    return body,weights/count


def envelope_error(predicted,target,block=ENVELOPE_BLOCK):
    """Duration-weighted compressed RMS-envelope error per example.

    No phase alignment, normalization to candidate peak, or arbitrary
    zero padding is applied. Quiet reference blocks use a fixed positive
    denominator floor so relative loss cannot explode near silence.
    """
    if (not isinstance(predicted,torch.Tensor) or not isinstance(target,torch.Tensor)
            or predicted.shape!=target.shape):
        raise ValueError("Invalid paired envelope waveform")
    actual,weights=_blocks(predicted,block)
    reference,other=_blocks(target,block)
    if not torch.equal(weights,other):
        raise ValueError("Envelope block geometry differs")
    a=torch.log1p(20.*torch.sqrt(actual+1e-10))
    b=torch.log1p(20.*torch.sqrt(reference+1e-10))
    error=((a-b).square()*weights).sum(dim=1)
    power=(b.square()*weights).sum(dim=1).clamp_min(.01)
    return error/power


def edge_training_loss(predicted,target,objective):
    if objective not in OBJECTIVES:
        raise ValueError("Unknown edge training objective")
    base=instrument_loss(predicted,target)
    if objective=="waveform":
        return base
    extra=envelope_error(predicted,target).mean()
    result=base+ENVELOPE_WEIGHT*extra
    if not bool(torch.isfinite(result)):
        raise ValueError("Nonfinite edge training objective")
    return result


@torch.no_grad()
def waveform_diagnostics(predicted,target):
    """Stable diagnostics for complete or near-silent note regions."""
    if (not isinstance(predicted,torch.Tensor) or not isinstance(target,torch.Tensor)
            or predicted.shape!=target.shape or predicted.ndim!=2
            or predicted.dtype!=torch.float32 or target.dtype!=torch.float32
            or predicted.device.type!="cpu" or target.device.type!="cpu"
            or not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all())):
        raise ValueError("Invalid edge diagnostic waveform pair")
    a=predicted.to(torch.float64)
    b=target.to(torch.float64)
    actual=a.square().mean(dim=1)
    reference=b.square().mean(dim=1)
    error=(a-b).square().mean(dim=1)
    return {
        "relative_wave_mse":error/reference.clamp_min(1e-8),
        "absolute_wave_mse":error,
        "rms_error":(actual.sqrt()-reference.sqrt()).abs(),
        "rms_ratio":(actual/reference.clamp_min(1e-8)).sqrt(),
        "envelope_error":envelope_error(predicted,target).to(torch.float64),
        "reference_rms":reference.sqrt(),
    }


@torch.no_grad()
def edge_parameters(model,instruments,pitches,velocities,frames):
    if not isinstance(model,WindowedHarmonicInstrument):
        raise ValueError("Expected windowed harmonic model")
    params=model.predict_parameters(instruments,pitches,velocities,frames)
    return {
        "attack_frames":params.attack_fraction*(frames-1),
        "release_frames":params.release_fraction*(frames-1),
    }
