"""Fresh synthetic source/noise nuisance audit, no probe refitting."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from statistics import fmean

import numpy as np

from research.generated_source_grounding import REFERENCE_SEEDS
from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, render_seeded_waveform
from research.open_set_source_probe import NOISE_FAMILIES, NOISE_SEEDS, fit_probe, evaluate_probe, reference_panels, noise_wave
from research.probe_nuisance_transforms import CONTROLS, INVARIANCE, apply_transform, waveform_diagnostics
from research.wave_diffusion import CATEGORIES

PROTOCOL = "sonumio-frozen-probe-nuisance-audit-v1"
SOURCE_SEEDS = (1601, 1602, 1603, 1604)
CONTROL_SEEDS = (1961, 1962, 1963, 1964)


def _hash(wave):
    return hashlib.sha256(wave.tobytes()).hexdigest()


def _prediction(probe, wave):
    decision = probe.detailed_score(wave).source
    return {
        "closest": CATEGORIES[decision.order[0]] if decision.order else None,
        "accepted": bool(decision.accepted),
        "prediction": CATEGORIES[decision.order[0]] if decision.accepted else None,
    }


def _summarize(rows, *, source):
    if not rows:
        raise ValueError("Empty robustness panel")
    transitions = Counter()
    raw_changes = 0
    new_accepts = 0
    removed_accepts = 0
    changed_accepted = 0
    for row in rows:
        original, transformed = row["original"], row["transformed"]
        before, after = original["prediction"], transformed["prediction"]
        transitions[(before or "abstain", after or "abstain")] += 1
        raw_changes += int(original["closest"] != transformed["closest"])
        new_accepts += int(before is None and after is not None)
        removed_accepts += int(before is not None and after is None)
        changed_accepted += int(before is not None and after is not None and before != after)
    result = {
        "cases": len(rows),
        "unchanged_accepted_prediction": sum(row["original"]["prediction"] == row["transformed"]["prediction"] for row in rows),
        "raw_nearest_changes": raw_changes,
        "new_acceptances": new_accepts, "removed_acceptances": removed_accepts,
        "changed_accepted_category": changed_accepted,
        "original_accepted": sum(row["original"]["accepted"] for row in rows),
        "transformed_accepted": sum(row["transformed"]["accepted"] for row in rows),
        "transitions": [
            {"from": a, "to": b, "count": n} for (a,b),n in sorted(transitions.items())
        ],
        "mean_sample_mae": fmean(row["waveform"]["sample_mae"] for row in rows),
        "mean_transformed_rms": fmean(row["waveform"]["transformed_rms"] for row in rows),
        "silent_transformed_cases": sum(row["waveform"]["transformed_silent"] for row in rows),
    }
    if source:
        before = [row["original"]["prediction"] == row["label"] for row in rows]
        after = [row["transformed"]["prediction"] == row["label"] for row in rows]
        result.update({
            "original_correct_accepted": sum(before),
            "transformed_correct_accepted": sum(after),
            "correct_acceptances_lost": sum(a and not b for a,b in zip(before,after,strict=True)),
            "correct_acceptances_gained": sum(not a and b for a,b in zip(before,after,strict=True)),
        })
    else:
        result["new_false_acceptances"] = new_accepts
    return result


def _seed_check():
    previous = TRAIN_SEEDS + VALIDATION_SEEDS + tuple(
        seed for table in (REFERENCE_SEEDS, NOISE_SEEDS) for panel in table.values() for seed in panel
    )
    seeds = previous + SOURCE_SEEDS + CONTROL_SEEDS
    if len(set(seeds)) != len(seeds):
        raise ValueError("Overlapping reference, training or fresh audit seeds")


def run_audit(*, control_names=tuple(t.name for t in CONTROLS)):
    available = {t.name:t for t in CONTROLS}
    if (type(control_names) is not tuple or not control_names
            or any(type(n) is not str or n not in available for n in control_names)
            or len(set(control_names)) != len(control_names)):
        raise ValueError("Invalid bounded control selection")
    _seed_check()
    sources, noises, hashes = reference_panels()
    reference_hashes = {h for group in hashes.values() for panel in group.values() for h in panel}
    probe = fit_probe(sources["fit"], sources["calibration"], noises["fit"], noises["calibration"])
    reference = evaluate_probe(probe, sources["holdout"], noises["holdout"])
    fresh_sources = [(category, seed, render_seeded_waveform((i,), seed))
                     for i,category in enumerate(CATEGORIES) for seed in SOURCE_SEEDS]
    fresh_noise = [(family, seed, noise_wave(family, seed))
                   for family in NOISE_FAMILIES for seed in CONTROL_SEEDS]
    fresh_hashes = [_hash(wave) for _,_,wave in fresh_sources + fresh_noise]
    if len(set(fresh_hashes)) != len(fresh_hashes) or set(fresh_hashes) & reference_hashes:
        raise ValueError("Exact waveform overlap in fresh source/noise audit")
    panels = {}
    baseline = {(label,seed): _prediction(probe,wave)
                for label,seed,wave in fresh_sources + fresh_noise}
    for name in control_names:
        control = available[name]
        rows = {}
        for domain, recordings in (("source",fresh_sources),("noise",fresh_noise)):
            result = []
            for label,seed,wave in recordings:
                changed = apply_transform(wave,control)
                result.append({"label":label,"seed":seed,
                               "original":baseline[label,seed],
                               "transformed":_prediction(probe,changed),
                               "waveform":waveform_diagnostics(wave,changed)})
            labels = CATEGORIES if domain == "source" else NOISE_FAMILIES
            rows[domain] = {
                "aggregate": _summarize(result,source=domain=="source"),
                "by_label": {label:_summarize([r for r in result if r["label"]==label],source=domain=="source")
                             for label in labels},
                "cases": result,
            }
        # The scale/polarity controls test invariance of decisions, not
        # reclassification of the original input as correct by assumption.
        source, noise = rows["source"]["aggregate"], rows["noise"]["aggregate"]
        invariant = (
            source["unchanged_accepted_prediction"] == source["cases"]
            and noise["unchanged_accepted_prediction"] == noise["cases"]
            and not source["raw_nearest_changes"] and not noise["raw_nearest_changes"]
        )
        panels[name] = {
            "family": control.family, "parameter": control.value, **rows,
            "exact_decision_invariance": invariant if control.family==INVARIANCE else None,
            "stress_test_not_perceptual_equivalence": control.family!=INVARIANCE,
        }
    return {
        "protocol": PROTOCOL,
        "reference_gate": reference["gate"],
        "fresh_source_recordings": len(fresh_sources), "fresh_noise_recordings": len(fresh_noise),
        "fresh_panel_sha256": hashlib.sha256("".join(fresh_hashes).encode("ascii")).hexdigest(),
        "exact_reference_overlap": False, "training_or_calibration_changed": False,
        "control_results": panels, "controls_are_paired_not_independent_recordings": True,
        "perceptual_quality_established": False, "model_readiness_established": False,
        "audio_or_weights_saved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", nargs="+", choices=[t.name for t in CONTROLS])
    args = parser.parse_args()
    report = run_audit(**({"control_names":tuple(args.controls)} if args.controls else {}))
    print(json.dumps(report,allow_nan=False,indent=2))


if __name__ == "__main__":
    main()
