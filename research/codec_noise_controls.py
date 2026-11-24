"""Codec-induced false acceptance controls for synthetic audio experiments.

A clean reference probe can still accept noise after a lossy codec changes
its spectrum. These panels are independent of source and probe training.
The measured rates are narrow artificial-noise diagnostics, not real-world
false-positive probability estimates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib

import numpy as np

from research.latent_audio import FRAMES
from research.open_set_source_probe import NOISE_FAMILIES, noise_wave
from research.wave_diffusion import CATEGORIES

NOISE_VALIDATION_SEEDS = (1921, 1922, 1923)
NOISE_CONFIRMATION_SEEDS = (1931, 1932, 1933, 1934)


@dataclass(frozen=True)
class NoisePanel:
    waves: np.ndarray
    families: tuple[str, ...]
    seeds: tuple[int, ...]
    hashes: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256("".join(self.hashes).encode("ascii")).hexdigest()


def build_noise_panel(seeds: tuple[int, ...]) -> NoisePanel:
    if (type(seeds) is not tuple or not 2 <= len(seeds) <= 8
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Invalid independent noise panel seeds")
    families = tuple(family for family in NOISE_FAMILIES for _ in seeds)
    row_seeds = seeds * len(NOISE_FAMILIES)
    waves = np.stack([noise_wave(family, seed)
                      for family, seed in zip(families, row_seeds, strict=True)])
    hashes = tuple(hashlib.sha256(wave.tobytes()).hexdigest() for wave in waves)
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate independent noise controls")
    waves.setflags(write=False)
    return NoisePanel(waves, families, row_seeds, hashes)


def noise_predictions(probe, waves: np.ndarray) -> tuple[str | None, ...]:
    """Return only accepted labels; rejected and silent rows abstain."""
    if (not isinstance(waves, np.ndarray) or waves.ndim != 2 or waves.shape[1] != FRAMES
            or not 1 <= len(waves) <= 56 or waves.dtype not in (np.float32, np.float64)
            or not np.isfinite(waves).all()):
        raise ValueError("Invalid noise waveform panel")
    result = []
    for wave in waves:
        decision = probe.detailed_score(np.clip(wave, -1, 1).astype(np.float32)).source
        if decision.accepted:
            if not decision.order or not 0 <= decision.order[0] < len(CATEGORIES):
                raise ValueError("Invalid accepted source decision")
            result.append(CATEGORIES[decision.order[0]])
        else:
            result.append(None)
    return tuple(result)


def compare_noise_predictions(panel: NoisePanel, original: tuple[str | None, ...],
                              reconstructed: tuple[str | None, ...]) -> dict:
    """Distinguish new false acceptance from pre-existing probe mistakes."""
    if (len(original) != len(panel.waves) or len(reconstructed) != len(panel.waves)
            or any(label is not None and label not in CATEGORIES
                   for label in original + reconstructed)):
        raise ValueError("Noise predictions do not align with panel")
    rows = []
    for family, seed, before, after in zip(
        panel.families, panel.seeds, original, reconstructed, strict=True,
    ):
        rows.append({
            "family": family, "seed": seed,
            "original_accepted_category": before,
            "reconstructed_accepted_category": after,
            "new_false_acceptance": before is None and after is not None,
            "removed_false_acceptance": before is not None and after is None,
            "changed_accepted_category": before is not None and after is not None and before != after,
        })

    def summarize(group):
        after_counts = Counter(row["reconstructed_accepted_category"] for row in group)
        return {
            "recordings": len(group),
            "original_accepted": sum(row["original_accepted_category"] is not None for row in group),
            "reconstructed_accepted": sum(row["reconstructed_accepted_category"] is not None for row in group),
            "new_false_acceptances": sum(row["new_false_acceptance"] for row in group),
            "removed_false_acceptances": sum(row["removed_false_acceptance"] for row in group),
            "changed_accepted_categories": sum(row["changed_accepted_category"] for row in group),
            "reconstructed_accepted_by_category": {name: after_counts[name] for name in CATEGORIES},
        }

    summary = summarize(rows)
    if summary["original_accepted"] + summary["new_false_acceptances"] - summary["removed_false_acceptances"] != summary["reconstructed_accepted"]:
        raise ValueError("Inconsistent false acceptance transitions")
    return {
        **summary,
        "fingerprint": panel.fingerprint,
        "by_family": {
            family: summarize([row for row in rows if row["family"] == family])
            for family in NOISE_FAMILIES
        },
        "cases": rows,
        "zero_original_and_reconstruction_acceptances": (
            summary["original_accepted"] == summary["reconstructed_accepted"] == 0
        ),
        "real_world_false_positive_rate_established": False,
    }
