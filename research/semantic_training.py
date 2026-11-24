"""Training-only source semantics for exact first-party synthetic captions.

A small auxiliary head predicts the six multi-label source memberships from
the existing text encoder. This does not add a class lookup to inference,
and successful text classification cannot establish generated audio meaning.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from research.counterfactual_conditioning import CaptionInterventions
from research.wave_diffusion import CATEGORIES

MAX_SEMANTIC_WEIGHT = 2.0


def validate_semantic_weight(weight):
    if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 <= weight <= MAX_SEMANTIC_WEIGHT:
        raise ValueError("Semantic weight must be finite in [0,2]")


def synthetic_source_targets(tokens: torch.Tensor, seeds: tuple[int, ...]) -> torch.Tensor:
    """Bind known multi-label targets to the exact canonical token table."""
    table = CaptionInterventions.build(tokens, seeds)
    targets = torch.zeros((len(tokens), len(CATEGORIES)), dtype=torch.float32)
    for row, members in enumerate(table.members):
        targets[row, list(members)] = 1
    return targets


def validate_targets(targets, rows):
    if (not isinstance(targets, torch.Tensor) or targets.shape != (rows, len(CATEGORIES))
            or targets.dtype != torch.float32 or targets.device.type != "cpu"
            or targets.requires_grad or not ((targets == 0) | (targets == 1)).all()
            or not ((targets.sum(1) >= 1) & (targets.sum(1) <= 2)).all()):
        raise ValueError("Expected finite binary source targets with one or two members")


class SemanticHead(nn.Module):
    """Auxiliary linear projection, absent from the deployed denoiser."""

    def __init__(self, channels: int):
        super().__init__()
        if type(channels) is not int or channels < 8 or channels > 256 or channels % 8:
            raise ValueError("Invalid semantic head channel count")
        self.projection = nn.Linear(channels, len(CATEGORIES))

    def forward(self, features):
        if (features.ndim != 2 or features.shape[1] != self.projection.in_features
                or features.dtype != torch.float32 or not torch.isfinite(features).all()):
            raise ValueError("Invalid text features for auxiliary semantic head")
        return self.projection(features)


def semantic_loss(encoder, head, tokens, targets, keep):
    """Sum BCE across retained rows; caller divides by logical batch size.

    Dropped captions contribute exactly zero, including gradients to the
    auxiliary head. No source class is fed to the waveform denoiser.
    """
    if (tokens.ndim != 2 or targets.shape != (len(tokens), len(CATEGORIES))
            or targets.dtype != torch.float32 or keep.dtype != torch.bool
            or keep.shape != (len(tokens),)
            or not (tokens.device == targets.device == keep.device)
            or not ((targets == 0) | (targets == 1)).all()):
        raise ValueError("Invalid auxiliary semantic batch")
    if not keep.any():
        return targets.new_zeros(())
    logits = head(encoder(tokens[keep]))
    loss = F.binary_cross_entropy_with_logits(logits, targets[keep], reduction="none").mean(1).sum()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite semantic auxiliary loss")
    return loss


@torch.no_grad()
def semantic_diagnostics(encoder, head, tokens, targets):
    validate_targets(targets, len(tokens))
    device = next(head.parameters()).device
    encoder_mode, head_mode = encoder.training, head.training
    encoder.eval()
    head.eval()
    exact = 0
    correct_bits = 0
    true_positive = false_positive = false_negative = 0
    loss_sum = 0.0
    category_counts = torch.zeros(len(CATEGORIES), 3, dtype=torch.long)
    try:
        for offset in range(0, len(tokens), 8):
            caption = tokens[offset:offset+8].to(device)
            expected = targets[offset:offset+8].to(device).bool()
            logits = head(encoder(caption))
            loss_sum += float(F.binary_cross_entropy_with_logits(logits, expected.float(), reduction="sum"))
            predicted = logits >= 0
            exact += int((predicted == expected).all(1).sum())
            correct_bits += int((predicted == expected).sum())
            category_counts[:, 0] += (predicted & expected).sum(0).cpu()
            category_counts[:, 1] += (predicted & ~expected).sum(0).cpu()
            category_counts[:, 2] += (~predicted & expected).sum(0).cpu()
    finally:
        encoder.train(encoder_mode)
        head.train(head_mode)
    true_positive, false_positive, false_negative = map(int, category_counts.sum(0))
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "caption_rows": len(tokens),
        "exact_membership_matches": exact,
        "bit_accuracy": correct_bits / targets.numel(),
        "mean_binary_cross_entropy": loss_sum / targets.numel(),
        "micro_f1": 2 * true_positive / denominator if denominator else None,
        "by_category": {
            category: dict(zip(("true_positive", "false_positive", "false_negative"), map(int, counts), strict=True))
            for category, counts in zip(CATEGORIES, category_counts, strict=True)
        },
        "text_only_diagnostic": True,
        "generated_audio_semantics_established": False,
    }
