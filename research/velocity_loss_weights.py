"""Explicit timestep weighting for velocity prediction.

The Min-SNR formulation follows Hang et al., ICCV 2023. The optional
clean-MSE rule is a separate diagnostic derived from the exact velocity
to clean conversion. All policies preserve the sampling distribution.
"""
from __future__ import annotations

import math
import torch

POLICIES = ("uniform", "min_snr", "clean_mse")


def validate_policy(policy: str, gamma: float):
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("Unknown velocity weighting policy")
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or not 0.1 <= gamma <= 100:
        raise ValueError("Min-SNR gamma must be finite and in [0.1,100]")


def timestep_weights(alpha_bars: torch.Tensor, policy: str, gamma: float = 5.0) -> torch.Tensor:
    """Stable weights without dividing by small terminal signal powers.

    Min-SNR-v = min(SNR, gamma)/(SNR+1)
              = min(alpha_bar, gamma*(1-alpha_bar)).
    Clean-space MSE = (1-alpha_bar)*velocity MSE for fixed noisy input.
    No mean normalization: the published relative weights remain explicit.
    """
    validate_policy(policy, gamma)
    if (not isinstance(alpha_bars, torch.Tensor) or alpha_bars.ndim != 1
            or not 8 <= len(alpha_bars) <= 256 or alpha_bars.dtype != torch.float32
            or not torch.isfinite(alpha_bars).all()
            or not ((alpha_bars > 0) & (alpha_bars < 1)).all()
            or not (alpha_bars[:-1] > alpha_bars[1:]).all()):
        raise ValueError("Invalid diffusion signal power schedule")
    if policy == "uniform":
        return torch.ones_like(alpha_bars)
    if policy == "min_snr":
        return torch.minimum(alpha_bars, gamma * (1 - alpha_bars))
    return 1 - alpha_bars


def velocity_loss(prediction, target, timestep, weights):
    """Return per-example raw MSE and weighted MSE for logical batching."""
    if (prediction.shape != target.shape or prediction.ndim != 3
            or prediction.dtype != torch.float32 or target.dtype != torch.float32
            or len(prediction) == 0 or timestep.shape != (len(prediction),)
            or timestep.dtype != torch.long or weights.ndim != 1
            or not (prediction.device == target.device == timestep.device == weights.device)
            or not ((timestep >= 0) & (timestep < len(weights))).all()):
        raise ValueError("Incompatible velocity loss inputs")
    raw = (prediction - target).square().mean(dim=(1, 2))
    weighted = raw * weights[timestep]
    if not torch.isfinite(raw).all() or not torch.isfinite(weighted).all():
        raise ValueError("Nonfinite velocity loss")
    return raw, weighted


def describe_weights(weights: torch.Tensor) -> dict:
    values = weights.detach().cpu().double()
    if values.ndim != 1 or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid velocity weight diagnostics")
    return {
        "by_timestep": values.tolist(),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "terminal": float(values[-1]),
    }
