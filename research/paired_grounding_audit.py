"""Audit paired source-grounding reports without training or loading weights.

The exact random-label baseline conditions on the observed predictions and
abstentions. It is a descriptive within-panel test, not a population-level
confidence estimate or a real-world audio-quality certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import permutations, product
import json
from pathlib import Path
from typing import Any

SOURCE_PROTOCOL = "sonumio-generated-synthetic-source-grounding-v1"
PROTOCOL = "sonumio-paired-grounding-audit-v1"
CATEGORIES = ("rain", "thunder", "wind", "forest", "ocean", "ambient")
ARMS = ("raw", "latent")
LANGUAGES = ("en", "hi")
MAX_REPORT_BYTES = 2 * 1024 * 1024


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _object(value: Any, name: str) -> dict:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _array(value: Any, name: str) -> list:
    _require(isinstance(value, list), f"{name} must be an array")
    return value


def _seed(value: Any) -> int:
    _require(type(value) is int and 0 <= value < 2**32, "Invalid sampling seed")
    return value


def _decision(value: Any, target: str | None = None) -> tuple[str | None, bool]:
    row = _object(value, "decision")
    closest = row.get("closest")
    accepted = row.get("within_reference_radius")
    _require(closest is None or closest in CATEGORIES, "Invalid predicted category")
    _require(type(accepted) is bool and (closest is not None or not accepted),
             "Invalid acceptance state")
    if target is not None:
        rank = row.get("target_rank")
        _require(
            (closest is None and rank is None)
            or (closest is not None and type(rank) is int and 1 <= rank <= len(CATEGORIES)
                and (rank == 1) == (closest == target)),
            "Target rank contradicts predicted category",
        )
    return closest, accepted


def _noise_confounds(reference: dict) -> frozenset[str]:
    controls = _object(reference.get("noise_controls"), "noise controls")
    _require(set(controls) == {"white_8k", "filtered_to_8k"}, "Missing noise control family")
    confounds = set()
    for family, raw in controls.items():
        rows = _array(raw, family)
        _require(len(rows) == 3, "Expected three controls per noise family")
        seen = set()
        for raw_row in rows:
            row = _object(raw_row, "noise control")
            seed = _seed(row.get("seed"))
            _require(seed not in seen, "Duplicate noise control seed")
            seen.add(seed)
            closest, accepted = _decision(row)
            if accepted:
                confounds.add(closest)
    declared = _array(reference.get("categories_accepted_for_noise"), "noise confounds")
    _require(all(isinstance(x, str) and x in CATEGORIES for x in declared)
             and len(declared) == len(set(declared)) and set(declared) == confounds,
             "Declared noise confounds contradict controls")
    return frozenset(confounds)


def _exact_assignment(predictions: tuple[str | None, ...]) -> dict:
    """Enumerate all balanced one-to-one assignments of the six target labels."""
    _require(len(predictions) == len(CATEGORIES), "Incomplete prediction panel")
    observed = sum(p == t for p, t in zip(predictions, CATEGORIES, strict=True))
    distribution = Counter(
        sum(p == t for p, t in zip(predictions, labels, strict=True))
        for labels in permutations(CATEGORIES)
    )
    total = sum(distribution.values())
    tail = sum(count for score, count in distribution.items() if score >= observed)
    return {
        "observed": observed,
        "permutations": total,
        "tail_count": tail,
        "tail_fraction": tail / total,
        "expected_correct": sum(k * v for k, v in distribution.items()) / total,
        "distribution": {str(k): distribution[k] for k in range(len(CATEGORIES) + 1)},
    }


def _panel(rows: dict[str, dict], confounds: frozenset[str]) -> dict:
    matched, rotated, null = [], [], []
    safe_predictions = []
    raw_predictions = []
    null_decisions = set()
    for index, target in enumerate(CATEGORIES):
        row = rows[target]
        alternate = CATEGORIES[(index + 1) % len(CATEGORIES)]
        _require(row.get("rotated_label") == alternate, "Invalid rotated label")
        a = _decision(row.get("matched"), target)
        b = _decision(row.get("rotated"), alternate)
        c = _decision(row.get("null_guidance"), target)
        _require(a == b, "Rotated control changed the observed prediction")
        null_decisions.add(c)
        raw_predictions.append(a[0])
        safe_predictions.append(a[0] if a[1] and a[0] not in confounds else None)
        matched.append(a[1] and a[0] == target and a[0] not in confounds)
        rotated.append(b[1] and b[0] == alternate and b[0] not in confounds)
        null.append(c[1] and c[0] == target and c[0] not in confounds)
    _require(len(null_decisions) == 1, "Null prediction changed across captions")
    exact = _exact_assignment(tuple(safe_predictions))
    return {
        "forced_rank_one": _exact_assignment(tuple(raw_predictions)),
        "accepted_nonconfounded": exact,
        "accepted_coverage": sum(p is not None for p in safe_predictions),
        "distinct_accepted_categories": len(set(safe_predictions) - {None}),
        "matched_correct": sum(matched),
        "rotated_correct": sum(rotated),
        "null_correct": sum(null),
        "matched_minus_rotated": sum(matched) - sum(rotated),
        "matched_minus_null": sum(matched) - sum(null),
        "matched_only_vs_null": sum(a and not b for a, b in zip(matched, null)),
        "null_only_vs_matched": sum(b and not a for a, b in zip(matched, null)),
        "per_category": {
            target: {"matched": a, "rotated": b, "null": c}
            for target, a, b, c in zip(CATEGORIES, matched, rotated, null, strict=True)
        },
        # A conservative descriptive screen, never a model-readiness gate.
        "passes_descriptive_screen": (
            exact["tail_fraction"] <= 0.05 and sum(matched) > max(sum(rotated), sum(null))
        ),
    }


def audit_report(report: dict) -> dict:
    report = _object(report, "report")
    _require(report.get("protocol") == SOURCE_PROTOCOL, "Unsupported source report protocol")
    _require(report.get("weight_or_audio_files_saved") is False, "Unexpected saved artifacts")
    config = _object(report.get("configuration"), "configuration")
    _require(tuple(_array(config.get("categories"), "categories")) == CATEGORIES,
             "Unexpected category order")
    model_seed = _seed(config.get("model_seed"))
    seeds = tuple(_seed(seed) for seed in _array(config.get("sampling_seeds"), "sampling seeds"))
    _require(1 <= len(seeds) <= 4 and len(seeds) == len(set(seeds)), "Invalid seed panel")
    reference = _object(report.get("reference_probe"), "reference probe")
    _require(reference.get("overlap_with_generator_data") is False, "Reference overlap not excluded")
    confounds = _noise_confounds(reference)
    raw_cases = _array(report.get("cases"), "cases")
    raw_checks = _array(report.get("null_guidance_checks"), "null checks")
    result = {
        "protocol": PROTOCOL,
        "source_protocol": SOURCE_PROTOCOL,
        "model_seed": model_seed,
        "noise_confounded_categories": sorted(confounds),
        "panels": [],
        "interpretation": "Descriptive balanced-label assignments; dependent panels are not pooled.",
    }
    status = report.get("status")
    if status == "reference_gate_failed":
        _require(reference.get("supports_interpreting_generated_scores") is False
                 and not raw_cases and not raw_checks, "Inconsistent failed reference gate")
        return {**result, "status": "reference_gate_failed", "panels_passing": 0}
    _require(status == "scored" and reference.get("supports_interpreting_generated_scores") is True,
             "Source report was not successfully scored")
    expected_checks = set(product(ARMS, seeds))
    checks = set()
    _require(len(raw_checks) == len(expected_checks), "Incomplete null checks")
    for raw in raw_checks:
        row = _object(raw, "null check")
        key = (row.get("arm"), _seed(row.get("sampling_seed")))
        _require(key in expected_checks and key not in checks
                 and row.get("caption_invariant") is True, "Invalid null invariance check")
        checks.add(key)
    expected = set(product(ARMS, seeds, LANGUAGES))
    panels: dict[tuple, dict] = {key: {} for key in expected}
    _require(len(raw_cases) == len(expected) * len(CATEGORIES), "Incomplete or excess cases")
    for raw in raw_cases:
        row = _object(raw, "case")
        key = (row.get("arm"), _seed(row.get("sampling_seed")), row.get("language"))
        target = row.get("requested")
        _require(key in panels and target in CATEGORIES and target not in panels[key],
                 "Unknown or duplicate paired case")
        panels[key][target] = row
    for arm, seed, language in product(ARMS, seeds, LANGUAGES):
        rows = panels[arm, seed, language]
        _require(set(rows) == set(CATEGORIES), "Unbalanced paired case panel")
        result["panels"].append({
            "arm": arm, "sampling_seed": seed, "language": language,
            **_panel(rows, confounds),
        })
    # Both languages must use the same unconditional waveform/prediction.
    for arm, seed in product(ARMS, seeds):
        en = panels[arm, seed, "en"][CATEGORIES[0]]
        hi = panels[arm, seed, "hi"][CATEGORIES[0]]
        _require(_decision(en["null_guidance"]) == _decision(hi["null_guidance"]),
                 "Null prediction changed across languages")
    passing = sum(panel["passes_descriptive_screen"] for panel in result["panels"])
    return {
        **result, "status": "audited", "panels_passing": passing,
        "all_panels_pass_descriptive_screen": passing == len(result["panels"]),
        "model_readiness_established": False,
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def read_report(path: Path) -> dict:
    with path.open("rb") as handle:
        data = handle.read(MAX_REPORT_BYTES + 1)
    _require(len(data) <= MAX_REPORT_BYTES, "Report exceeds size limit")
    return _object(json.loads(data, parse_constant=_reject_constant), "report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        audit = audit_report(read_report(args.report))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Grounding audit failed: {exc}\n")
    print(json.dumps(audit, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
