"""Locate the exact first differing DSP stage on additional numerical seeds.

This is a research-only *diagnostic* for 5 seeds x 21 sound combinations x
2 ambient-sine methods. It reconstructs the separate portable DSP generator
and verifies output against the independent portable recipe on each host.
Records only keyed SHA digests of sound layers, RNG state, and resampling
stages: NEVER write audio, model weights, or an altered neural dataset.

"Additional numerical seed" is relative to the 84-case latent DSP grid;
seed 3 is also a text-audio training seed and is NOT a global model holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.ambient_sine_candidate import METHODS, candidate_drone
from research.complete_dsp_candidate import RAW_FRAMES, _crop
from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.fast_dsp_candidate import _fast_layer
from research.heldout_dsp_probe import HELDOUT_SEEDS, TEXT_AUDIO_TRAIN_OVERLAP
from research.portable_dsp_candidate import portable_raw
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES, _downsample_24k_to_8k
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-additional-seed-stage-trace-v1"
METHOD = METHODS[0]
GRID = tuple((seed, members) for seed in HELDOUT_SEEDS
             for members in SYNTHETIC_COMBINATIONS)
if len(GRID) != 105:
    raise RuntimeError("Expected 105 first-party numerical seed/source pairs")


def _sha(samples: np.ndarray) -> str:
    return digest_array(samples)["sha256"]


def _state(rng: np.random.Generator) -> str:
    """Hash the generator's internal bit state, never serialize its bytes."""
    return hashlib.sha256(
        json.dumps(rng.bit_generator.state, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def trace_one(seed: int, members: tuple[int, ...], method: str) -> dict:
    if ((seed, members) not in GRID or type(seed) is not int
            or type(members) is not tuple or method not in METHODS):
        raise ValueError("Invalid additional-seed stage identity")
    prompt = " and ".join(CATEGORIES[i] for i in members)
    kinds = dsp.recognized_sound_classes(prompt)
    t = np.arange(RAW_FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    mixed = np.zeros(RAW_FRAMES, dtype=np.float32)
    stages = {"time_grid": _sha(t), "initial_rng_state": _state(rng)}
    for kind in kinds:
        layer = (candidate_drone(t, method=method) if kind == "drone"
                 else _fast_layer(kind, rng, t))
        stages["layer_" + kind] = _sha(layer)
        stages["rng_after_" + kind] = _state(rng)
        mixed += layer
        stages["mixed_after_" + kind] = _sha(mixed)
    stages["pre_fade"] = _sha(mixed)
    edge = min(int(0.025 * dsp.SAMPLE_RATE), RAW_FRAMES // 2)
    if edge:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        stages["fade_ramp"] = _sha(ramp)
        mixed[:edge] *= ramp
        mixed[-edge:] *= ramp[::-1]
    stages["after_fade"] = _sha(mixed)
    peak = float(np.max(np.abs(mixed)))
    if peak > 0:
        mixed *= min(1.0, 0.92 / peak)
    stages["normalized_24k"] = _sha(mixed)
    canonical = portable_raw(members, seed, method)
    if not np.array_equal(mixed, canonical):
        raise ValueError("Stage reconstruction drifted from portable DSP renderer")
    down = _downsample_24k_to_8k(mixed)
    stages["resampled_8k"] = _sha(down)
    offset, crop = _crop(down, seed)
    stages["crop_8k"] = _sha(crop)
    return {
        "seed": seed, "sources": [CATEGORIES[i] for i in members],
        "method": method, "crop_offset": offset,
        "stage_order": list(stages), "stages": stages,
    }


def build_report() -> dict:
    rows = [trace_one(seed, members, method)
            for method in METHODS for seed, members in GRID]
    result = {
        "protocol": PROTOCOL,
        "records_per_method": 105,
        "rows": rows,
        "environment": _environment(),
        "text_audio_training_seed_overlap": list(TEXT_AUDIO_TRAIN_OVERLAP),
        "globally_heldout_for_neural_model_evaluation": False,
        "original_renderer_unchanged": True,
        "candidate_used_for_neural_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
    }
    validate_report(result)
    return result


def validate_report(report: dict) -> None:
    if (not isinstance(report, dict) or report.get("protocol") != PROTOCOL
            or report.get("records_per_method") != 105
            or report.get("text_audio_training_seed_overlap") != list(TEXT_AUDIO_TRAIN_OVERLAP)
            or report.get("globally_heldout_for_neural_model_evaluation") is not False
            or report.get("original_renderer_unchanged") is not True
            or report.get("candidate_used_for_neural_training") is not False
            or report.get("audio_files_saved") is not False
            or report.get("model_weights_saved") is not False):
        raise ValueError("Invalid numerical stage-provenance metadata")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 105 * len(METHODS):
        raise ValueError("Incomplete additional-seed trace")
    expected = {(method, seed, members) for method in METHODS for seed, members in GRID}
    seen = set()
    for row in rows:
        try:
            if not isinstance(row, dict):
                raise ValueError
            identity = (row["method"], row["seed"],
                        tuple(CATEGORIES.index(s) for s in row["sources"]))
            if identity in seen or identity not in expected or type(row["seed"]) is not int:
                raise ValueError
            seen.add(identity)
            stages, order = row["stages"], row["stage_order"]
            if (not isinstance(stages, dict) or not isinstance(order, list)
                    or len(order) != len(stages) or len(set(order)) != len(order)
                    or set(stages) != set(order) or
                    any(not _hash_ok(value) for value in stages.values())
                    or not {"time_grid", "initial_rng_state", "normalized_24k",
                            "resampled_8k", "crop_8k"}.issubset(stages)
                    or type(row["crop_offset"]) is not int
                    or not 400 <= row["crop_offset"] <= 1600):
                raise ValueError
        except (ValueError, TypeError, KeyError, IndexError):
            raise ValueError("Invalid stage SHA, execution order or identity") from None
    if seen != expected:
        raise ValueError("Missing additional source/seed combinations")


def compare_reports(first: dict, second: dict) -> dict:
    validate_report(first)
    validate_report(second)
    a = {(r["method"], r["seed"], tuple(r["sources"])): r
         for r in first["rows"]}
    b = {(r["method"], r["seed"], tuple(r["sources"])): r
         for r in second["rows"]}
    counts = {method: {} for method in METHODS}
    differences = []
    exact = 0
    for key, old in a.items():
        new = b[key]
        if old["stage_order"] != new["stage_order"]:
            raise ValueError("DSP stage execution order changed")
        changed = [name for name in old["stage_order"]
                   if old["stages"][name] != new["stages"][name]]
        method, seed, categories = key
        if changed:
            if len(differences) < 32:
                differences.append({
                    "method": method, "seed": seed, "sources": list(categories),
                    "first_divergent_stage": changed[0],
                    "all_differing_stages": changed,
                })
        else:
            exact += 1
        index = counts[method].setdefault(str(seed), {
            "cases": 0, "full_stage_exact": 0,
            "different_24k": 0, "different_8k_crop": 0,
            "first_stage_counts": {},
        })
        index["cases"] += 1
        index["full_stage_exact"] += not changed
        index["different_24k"] += (
            old["stages"]["normalized_24k"] != new["stages"]["normalized_24k"]
        )
        index["different_8k_crop"] += (
            old["stages"]["crop_8k"] != new["stages"]["crop_8k"]
        )
        if changed:
            index["first_stage_counts"][changed[0]] = (
                index["first_stage_counts"].get(changed[0], 0) + 1
            )
    return {
        "protocol": "sonumio-additional-seed-stage-comparison-v1",
        "records_per_method": 105,
        "fully_matching_method_records": exact,
        "total_method_records": len(a),
        "same_cpu_model": first["environment"]["cpu_model"]
                          == second["environment"]["cpu_model"],
        "numpy_versions": [
            first["environment"]["numpy"], second["environment"]["numpy"]
        ],
        "by_method_seed": counts,
        "first_differing_cases_capped_at_32": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare:
        first, second = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.compare
        ]
        data = compare_reports(first, second)
    else:
        data = build_report()
    payload = json.dumps(data, indent=2, sort_keys=True,
                         allow_nan=False, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
