"""Controlled caption interventions for the tiny synthetic velocity trainer.

Negatives have disjoint source labels but preserve language, caption template,
recording seed and number of named sources. The relative-margin objective
never backpropagates through the wrong-caption prediction: the auxiliary term
cannot directly reward increasing the error of a wrong caption.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from research.text_audio import (
    ByteTextTokenizer, SYNTHETIC_COMBINATIONS, _synth_prompts,
)
from research.velocity_prediction import velocity_target


RELATIVE_MARGIN = 0.05
MAX_WEIGHT = 2.0


def validate_weight(weight: float) -> None:
    if (type(weight) not in (int, float) or not math.isfinite(weight)
            or not 0 <= weight <= MAX_WEIGHT):
        raise ValueError("Conditioning weight must be finite within [0, 2]")


@dataclass(frozen=True)
class CaptionInterventions:
    """Bounded index table tied to the exact synthetic corpus ordering."""

    members: tuple[tuple[int, ...], ...]
    seeds: tuple[int, ...]
    languages: tuple[str, ...]
    candidates: tuple[tuple[int, ...], ...]

    @classmethod
    def build(cls, tokens: torch.Tensor, seeds: tuple[int, ...]) -> CaptionInterventions:
        if (type(seeds) is not tuple or not 1 <= len(seeds) <= 32
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
                or len(seeds) != len(set(seeds))):
            raise ValueError("Invalid intervention recording seeds")
        records = [
            (members, seed, language)
            for members in SYNTHETIC_COMBINATIONS
            for seed in seeds
            for language in ("en", "hi")
        ]
        expected = ByteTextTokenizer().encode([
            _synth_prompts(members, language, seed) for members, seed, language in records
        ])
        if (tokens.device.type != "cpu" or tokens.dtype != torch.long
                or not torch.equal(tokens, expected)):
            raise ValueError("Caption order does not match the synthetic corpus")
        lookup = {record: index for index, record in enumerate(records)}
        candidates = tuple(
            tuple(lookup[other, seed, language] for other in SYNTHETIC_COMBINATIONS
                  if len(other) == len(members) and not set(other).intersection(members))
            for members, seed, language in records
        )
        if not all(candidates):
            raise ValueError("Every caption requires a disjoint negative")
        return cls(
            tuple(record[0] for record in records),
            tuple(record[1] for record in records),
            tuple(record[2] for record in records),
            candidates,
        )

    def wrong_indices(self, indices: torch.Tensor, *, round_index: int) -> torch.Tensor:
        if (type(round_index) is not int or round_index < 0
                or indices.device.type != "cpu" or indices.dtype != torch.long
                or indices.ndim != 1 or not len(indices)
                or not ((indices >= 0) & (indices < len(self.candidates))).all()):
            raise ValueError("Invalid counterfactual caption indices")
        # No RNG draws: the baseline and treatment use identical stochastic
        # corruption, minibatch, codec and classifier-free dropout streams.
        return torch.tensor([
            self.candidates[index][round_index % len(self.candidates[index])]
            for index in indices.tolist()
        ], dtype=torch.long)


def per_example_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if (prediction.shape != target.shape or prediction.ndim != 3
            or not len(prediction) or prediction.dtype != torch.float32
            or target.dtype != torch.float32 or prediction.device != target.device
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError("Expected finite aligned float32 velocity predictions")
    return (prediction - target).square().flatten(1).mean(dim=1)


def relative_caption_penalty(
    matched: torch.Tensor, wrong: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """One-sided relative margin, differentiating only the matched branch."""
    matched_error = per_example_mse(matched, target)
    wrong_error = per_example_mse(wrong.detach(), target.detach())
    return torch.relu(matched_error - (1.0 - RELATIVE_MARGIN) * wrong_error).mean()


def training_penalty(
    model: nn.Module, noisy: torch.Tensor, timestep: torch.Tensor,
    matched_prediction: torch.Tensor, target: torch.Tensor,
    keep: torch.Tensor, wrong_tokens: torch.Tensor,
) -> torch.Tensor:
    if (keep.shape != (len(noisy),) or keep.dtype != torch.bool
            or len(wrong_tokens) != len(noisy)):
        raise ValueError("Invalid counterfactual conditioning batch")
    # Dropped conditions train only the normal unconditional v-objective.
    if not keep.any():
        return matched_prediction.new_zeros(())
    with torch.no_grad():
        wrong_prediction = model(noisy[keep], timestep[keep], wrong_tokens[keep])
    return relative_caption_penalty(matched_prediction[keep], wrong_prediction, target[keep])


def _empty_summary() -> dict:
    return {
        "caption_cases": 0,
        "matched_mse": 0.0,
        "wrong_mse": 0.0,
        "null_mse": 0.0,
        "matched_beats_wrong": 0,
        "matched_beats_null": 0,
    }


def _finish(summary: dict) -> dict:
    count = summary["caption_cases"]
    if not count:
        raise ValueError("Empty held-out counterfactual stratum")
    for key in ("matched_mse", "wrong_mse", "null_mse"):
        summary[key] /= count
        if not math.isfinite(summary[key]):
            raise ValueError("Nonfinite held-out counterfactual metric")
    summary["wrong_minus_matched_mse"] = summary["wrong_mse"] - summary["matched_mse"]
    summary["null_minus_matched_mse"] = summary["null_mse"] - summary["matched_mse"]
    return summary


@torch.no_grad()
def heldout_interventions(
    model: nn.Module, source: torch.Tensor, tokens: torch.Tensor,
    table: CaptionInterventions, schedule, *, noise_seed: int,
) -> dict:
    """Score matched, disjoint and null captions on the exact same corruption.

    Each of three fixed noise levels scores all held-out captions. Bilingual
    captions share an underlying recording and the same noise; counts are
    caption cases, not independent samples or statistical confidence bounds.
    """
    if (model.training or type(noise_seed) is not int or not 0 <= noise_seed < 2**32
            or len(source) != len(tokens) or len(tokens) != len(table.candidates)
            or len(source) % 2 or schedule.steps < 3):
        raise ValueError("Invalid held-out intervention configuration")
    if not torch.equal(source[::2], source[1::2]):
        raise ValueError("Bilingual caption pairs must share the recording")
    rng = torch.Generator(device="cpu").manual_seed(noise_seed)
    levels = (0, schedule.steps // 2, schedule.steps - 1)
    report = {}
    for level_index, level in enumerate(levels):
        grouped = {
            f"{language}_{count}_source": _empty_summary()
            for language in ("en", "hi") for count in (1, 2)
        }
        for offset in range(0, len(source), 4):
            clean = source[offset:offset + 4]
            indices = torch.arange(offset, offset + len(clean))
            wrong_indices = table.wrong_indices(indices, round_index=level_index)
            noise = torch.randn(clean[::2].shape, generator=rng).repeat_interleave(2, dim=0)
            timestep = torch.full((len(clean),), level, dtype=torch.long)
            noisy = schedule.corrupt(clean, timestep, noise)
            target = velocity_target(clean, noise, timestep, schedule)
            matched = per_example_mse(model(noisy, timestep, tokens[indices]), target)
            wrong = per_example_mse(model(noisy, timestep, tokens[wrong_indices]), target)
            null = per_example_mse(model(noisy, timestep, torch.zeros_like(tokens[indices])), target)
            for local, index in enumerate(indices.tolist()):
                key = f"{table.languages[index]}_{len(table.members[index])}_source"
                row = grouped[key]
                a, b, c = float(matched[local]), float(wrong[local]), float(null[local])
                row["caption_cases"] += 1
                row["matched_mse"] += a
                row["wrong_mse"] += b
                row["null_mse"] += c
                row["matched_beats_wrong"] += int(a < b)
                row["matched_beats_null"] += int(a < c)
        report[str(level)] = {key: _finish(value) for key, value in grouped.items()}
    return {
        "by_timestep_and_caption_stratum": report,
        "unique_recordings": len(source) // 2,
        "caption_cases_per_timestep": len(source),
        "same_noise_for_bilingual_pairs": True,
        "same_corruption_for_all_caption_interventions": True,
        "independent_caption_cases": False,
    }
