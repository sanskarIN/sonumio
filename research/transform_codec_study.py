"""Fixed-panel rate/distortion and source-preservation study for transforms.

Train-only coefficient ranking/PCA, separate validation and probe holdout.
All results are numerical diagnostics; no audio, bitstreams or weights saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from statistics import mean

import numpy as np

from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, render_seeded_waveform
from research.open_set_source_probe import evaluate_probe, fit_probe, reference_panels
from research.generated_source_grounding import REFERENCE_SEEDS
from research.transform_audio_codec import BUDGETS, METHODS, TransformFit, reconstruction_metrics
from research.wave_diffusion import CATEGORIES, SAMPLE_RATE

PROTOCOL = "sonumio-transform-codec-ceiling-v1"
FIT_SEEDS = TRAIN_SEEDS + (2, 3, 4, 5)


def _panel(seeds):
    waves = np.stack([
        render_seeded_waveform((label,), seed)
        for label in range(len(CATEGORIES)) for seed in seeds
    ])
    labels = np.repeat(np.arange(len(CATEGORIES), dtype=np.int64), len(seeds))
    hashes = tuple(hashlib.sha256(wave.tobytes()).hexdigest() for wave in waves)
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate recordings within transform panel")
    return waves, labels, hashes


def _aggregate(rows):
    result = {"recordings": len(rows)}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = mean(values) if values else None
    return result


def _score_panel(fitted, method, budget, waves, labels, probe):
    coefficients = fitted.encode(waves, method, budget)
    decoded = fitted.decode(coefficients, method, budget)
    return score_decoded_panel(waves, decoded, labels, probe)


def score_decoded_panel(waves, decoded, labels, probe):
    """Shared pre-clipping numerical and post-clipping source evaluation."""
    if len(labels) != len(waves):
        raise ValueError("Recording and label count mismatch")
    metrics = reconstruction_metrics(waves, decoded)
    rows = []
    for i, label in enumerate(labels):
        target = int(label)
        original = probe.detailed_score(waves[i]).as_dict(target)
        reconstruction = probe.detailed_score(
            np.clip(decoded[i], -1, 1).astype(np.float32),
        ).as_dict(target)
        rows.append({
            "category": CATEGORIES[target],
            "original_correct_accepted": original["within_reference_radius"] and original["target_rank"] == 1,
            "reconstructed_correct_accepted": reconstruction["within_reference_radius"] and reconstruction["target_rank"] == 1,
            "original": original, "reconstructed": reconstruction,
            "metrics_before_clipping": metrics[i],
        })
    by_category = {}
    for category in CATEGORIES:
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "original_correct_accepted": sum(row["original_correct_accepted"] for row in selected),
            "reconstructed_correct_accepted": sum(row["reconstructed_correct_accepted"] for row in selected),
            "numerical": _aggregate([row["metrics_before_clipping"] for row in selected]),
        }
    return {
        "recordings": len(rows),
        "original_correct_accepted": sum(row["original_correct_accepted"] for row in rows),
        "reconstructed_correct_accepted": sum(row["reconstructed_correct_accepted"] for row in rows),
        "numerical": _aggregate(metrics),
        "by_category": by_category,
        "cases": rows,
    }


def run_study(*, budgets=BUDGETS, methods=METHODS):
    if (type(budgets) is not tuple or not budgets
            or any(type(x) is not int or x not in BUDGETS for x in budgets)
            or len(set(budgets)) != len(budgets)
            or type(methods) is not tuple or not methods
            or any(x not in METHODS for x in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError("Invalid transform study grid")
    fit_waves, fit_labels, fit_hashes = _panel(FIT_SEEDS)
    val_waves, val_labels, val_hashes = _panel(VALIDATION_SEEDS)
    sources, noises, reference_hashes = reference_panels()
    hashes = [digest for domain in reference_hashes.values() for split in domain.values() for digest in split]
    if len(set(fit_hashes + val_hashes + tuple(hashes))) != len(fit_hashes) + len(val_hashes) + len(hashes):
        raise ValueError("Exact waveform overlap across transform or reference splits")
    seed_groups = FIT_SEEDS + VALIDATION_SEEDS + tuple(
        seed for group in REFERENCE_SEEDS.values() for seed in group
    )
    if len(set(seed_groups)) != len(seed_groups):
        raise ValueError("Transform recording seeds overlap")
    fit = TransformFit.fit(fit_waves, fit_labels)
    probe = fit_probe(sources["fit"], sources["calibration"], noises["fit"], noises["calibration"])
    gate = evaluate_probe(probe, sources["holdout"], noises["holdout"])
    heldout = sources["holdout"].reshape(-1, FRAMES)
    heldout_labels = np.repeat(np.arange(len(CATEGORIES)), len(REFERENCE_SEEDS["holdout"]))
    rows = []
    for method in methods:
        for budget in budgets:
            validation = _score_panel(fit, method, budget, val_waves, val_labels, probe)
            reference = _score_panel(fit, method, budget, heldout, heldout_labels, probe)
            if budget == 64 and max(
                validation["numerical"]["mse"], reference["numerical"]["mse"]
            ) > 1e-12:
                raise ValueError("Full-coefficient transform failed reconstruction control")
            rows.append({
                **fit.describe(method, budget),
                "validation": validation, "probe_holdout": reference,
            })
    return {
        "protocol": PROTOCOL,
        "sample_rate": SAMPLE_RATE, "frames": FRAMES,
        "fit_recordings": len(fit_waves), "validation_recordings": len(val_waves),
        "reference_recordings": len(heldout),
        "fit_fingerprint": hashlib.sha256("".join(fit_hashes).encode("ascii")).hexdigest(),
        "validation_fingerprint": hashlib.sha256("".join(val_hashes).encode("ascii")).hexdigest(),
        "reference_gate_passed": gate["gate"]["passed"],
        "source_scores_descriptive_only": not gate["gate"]["passed"],
        "fit_only_basis_and_coordinate_selection": True,
        "exact_split_overlap": False,
        "quantized_bitrate_established": False,
        "model_readiness_established": False,
        "audio_or_weights_saved": False,
        "experiments": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    print(json.dumps(run_study(budgets=tuple(args.budgets), methods=tuple(args.methods)),
                     allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
