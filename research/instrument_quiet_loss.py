"""Controlled training objectives for sustained notes near silence.

The baseline uses the existing relative waveform/spectral loss. Alternative
arms use a fixed reference-power floor and an optional absolute energy
term. Floors are prespecified, not fitted to held-out examples. No change
to the synthesis architecture or evaluation metrics is made.
"""
from __future__ import annotations

import torch

from research.instrument_harmonic_model import _wave_pair, _spectral_error, instrument_loss
from research.instrument_edge_model import envelope_error

PROTOCOL = "sonumio-quiet-region-objective-v1"
ARMS = ("baseline", "floor", "floor-envelope")
POWER_FLOOR = 1e-4
ENVELOPE_WEIGHT = .35


def loss_components(predicted, target, arm):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("Unknown quiet-region objective")
    predicted, target = _wave_pair(predicted, target)
    power = target.square().mean(dim=1)
    squared_error = (predicted-target).square().mean(dim=1)
    floor = 1e-8 if arm == "baseline" else POWER_FLOOR
    waveform = (squared_error/power.clamp_min(floor)).mean()
    spectral = _spectral_error(predicted, target).mean()
    envelope = envelope_error(predicted, target).mean()
    total = waveform + .25*spectral
    if arm == "floor-envelope":
        total = total + ENVELOPE_WEIGHT*envelope
    if not all(bool(torch.isfinite(v)) for v in (waveform,spectral,envelope,total)):
        raise ValueError("Nonfinite quiet-region objective")
    return total, {
        "waveform": waveform, "spectral": spectral, "envelope": envelope,
        "absolute_mse": squared_error.mean(), "reference_power": power.mean(),
        "floor_fraction": (power<floor).float().mean(),
    }


def relative_gradient_bound(target, floor):
    """Analytic multiplier for d(mean squared error)/d(predicted).

    The returned per-example reciprocal is not a bound on the full
    network gradient, which also depends on the model Jacobian.
    """
    if floor not in (1e-8, POWER_FLOOR):
        raise ValueError("Unsupported reference power floor")
    _, target = _wave_pair(target, target)
    return target.square().mean(dim=1).clamp_min(floor).reciprocal()


@torch.no_grad()
def energy_diagnostics(predicted,target):
    """Pairwise absolute and relative errors with both reference floors."""
    predicted,target=_wave_pair(predicted,target)
    power=target.double().square().mean(dim=1)
    squared=(predicted.double()-target.double()).square().mean(dim=1)
    actual=predicted.double().square().mean(dim=1)
    return {
        "absolute_mse": squared,
        "relative_mse": squared/power.clamp_min(1e-8),
        "floored_mse": squared/power.clamp_min(POWER_FLOOR),
        "reference_power": power,
        "predicted_power": actual,
        "rms_difference": (actual.sqrt()-power.sqrt()).abs(),
    }
