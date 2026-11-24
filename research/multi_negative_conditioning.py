"""Multi-negative matched-only margin for logical-batch velocity training.

Unlike a contrastive objective that can directly inflate wrong-caption error,
this objective uses the smallest detached wrong-caption error as a target
for improving the matched prediction. It is not a guarantee of grounding.
"""
from __future__ import annotations

import math

import torch

from research.counterfactual_conditioning import CaptionInterventions, per_example_mse

MAX_NEGATIVES = 4


def validate_margin(weight, margin, negatives):
    if (type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= 2
            or type(margin) not in (int, float) or not math.isfinite(margin) or not 0 < margin <= 0.5
            or type(negatives) is not int or not 1 <= negatives <= MAX_NEGATIVES):
        raise ValueError("Invalid multi-negative caption margin settings")


def negative_indices(table: CaptionInterventions, indices: torch.Tensor,
                     *, round_index: int, count: int) -> torch.Tensor:
    """Select distinct disjoint captions, preserving template and language."""
    if (not isinstance(table, CaptionInterventions) or type(count) is not int
            or not 1 <= count <= MAX_NEGATIVES or type(round_index) is not int or round_index < 0
            or indices.ndim != 1 or not len(indices) or indices.dtype != torch.long
            or indices.device.type != "cpu"
            or not ((indices >= 0) & (indices < len(table.candidates))).all()):
        raise ValueError("Invalid multi-negative caption selection")
    rows = []
    for index in indices.tolist():
        candidates = table.candidates[index]
        if len(candidates) < count:
            raise ValueError("Insufficient distinct disjoint negative captions")
        offset = (round_index * count) % len(candidates)
        rows.append([candidates[(offset + j) % len(candidates)] for j in range(count)])
    return torch.tensor(rows, dtype=torch.long)


def matched_margin_sum(matched: torch.Tensor, target: torch.Tensor,
                       negative_errors: torch.Tensor, *, margin: float) -> torch.Tensor:
    validate_margin(1.0, margin, 1)
    error = per_example_mse(matched, target)
    if (negative_errors.ndim != 2 or negative_errors.shape[0] != len(error)
            or not 1 <= negative_errors.shape[1] <= MAX_NEGATIVES
            or negative_errors.device != matched.device or negative_errors.dtype != torch.float32
            or not torch.isfinite(negative_errors).all() or (negative_errors < 0).any()):
        raise ValueError("Invalid detached negative error panel")
    hardest = negative_errors.detach().min(dim=1).values
    return torch.relu(error - (1 - margin) * hardest).sum()


def training_margin_sum(model, noisy, timestep, matched, target, keep, wrong_tokens, *, margin):
    """Sum only retained-caption margins, with no gradients through negatives."""
    if (keep.dtype != torch.bool or keep.shape != (len(noisy),)
            or wrong_tokens.ndim != 3 or wrong_tokens.shape[0] != len(noisy)
            or not 1 <= wrong_tokens.shape[1] <= MAX_NEGATIVES
            or wrong_tokens.dtype != torch.long
            or not (noisy.device == timestep.device == matched.device == target.device
                    == keep.device == wrong_tokens.device)):
        raise ValueError("Invalid multi-negative training batch")
    if not keep.any():
        return matched.new_zeros(())
    columns = []
    with torch.no_grad():
        for j in range(wrong_tokens.shape[1]):
            wrong = model(noisy[keep], timestep[keep], wrong_tokens[keep, j])
            columns.append(per_example_mse(wrong, target[keep]))
    return matched_margin_sum(matched[keep], target[keep], torch.stack(columns, dim=1), margin=margin)
