"""Investigate original forest bird-chirp sine bytes across NumPy versions.

Uses the unchanged DSP generator as an *independent* bitwise oracle for
36 source/seed contexts. Records only SHA256 digests and first-party source
identities. This is a research-only stage inspection, NOT a replacement for
procedural generation or synthetic neural-training data.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from research.ambient_sine_candidate import _sin_float32_phase
from research.data_reproducibility import _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.fast_dsp_candidate import _fast_layer
from research.heldout_dsp_probe import HELDOUT_SEEDS
from research.latent_audio import TRAIN_SEEDS, VALIDATION_SEEDS
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-forest-sine-stage-trace-v1"
SEEDS = tuple(TRAIN_SEEDS) + tuple(VALIDATION_SEEDS) + HELDOUT_SEEDS
PREFIXES = ((), ("rain",), ("thunder",), ("wind",))
CHIRP_METHODS = ("original", "numpy_float64", "python_math_sin")
FRAMES = round(dsp.SAMPLE_RATE * 1.5)


def _digest(samples: np.ndarray) -> str:
    return digest_array(samples)["sha256"]


def _forest_variants(rng: np.random.Generator, t: np.ndarray) -> tuple[dict, dict]:
    n, sr = len(t), dsp.SAMPLE_RATE
    accum = {variant: np.zeros(n, dtype=np.float32) for variant in CHIRP_METHODS}
    chirps = []
    for i, position in enumerate(np.arange(0.35, t[-1], 0.7)):
        start = int(position * sr)
        length = min(int(0.19 * sr), n - start)
        if length <= 0:
            continue
        local_t = np.arange(length, dtype=np.float32) / sr
        freq = rng.uniform(900, 1900)
        phase = 2 * np.pi * (freq * local_t + 1800 * local_t ** 2)
        if phase.dtype != np.float32:
            raise ValueError("Unexpected original forest float32 phase dtype")
        window = np.sin(np.pi * np.linspace(0, 1, length)) ** 2
        old_sin = np.sin(phase)
        alt_double = _sin_float32_phase(phase, method="numpy_float64")
        alt_scalar = _sin_float32_phase(phase, method="python_math_sin")
        layers = {
            "original": (0.12 * old_sin * window).astype(np.float32),
            "numpy_float64": (0.12 * alt_double * window).astype(np.float32),
            "python_math_sin": (0.12 * alt_scalar * window).astype(np.float32),
        }
        hashes = {
            "local_time": _digest(local_t),
            "frequency": _digest(np.asarray([freq], dtype=np.float64)),
            "phase": _digest(phase),
            "window": _digest(window),
            "sine_original": _digest(old_sin),
            "sine_numpy_float64": _digest(alt_double),
            "sine_python_math_sin": _digest(alt_scalar),
        }
        for method, layer in layers.items():
            accum[method][start:start + length] += layer
            hashes["contribution_" + method] = _digest(layer)
            hashes["sum_" + method] = _digest(accum[method])
        chirps.append({"index": i, "start_frame": start, "frames": length,
                       "stage_order": list(hashes), "hashes": hashes})
    return accum, {"chirps": chirps, "chirp_count": len(chirps)}


def trace_one(seed: int, prefix: tuple[str, ...]) -> dict:
    if type(seed) is not int or seed not in SEEDS or prefix not in PREFIXES:
        raise ValueError("Invalid original forest source/seed context")
    rng = np.random.default_rng(seed)
    t = np.arange(FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    # Advance the SAME original RNG sequence that would precede the forest.
    for kind in prefix:
        _fast_layer(kind, rng, t)
    independent_rng = copy.deepcopy(rng)
    variants, trace = _forest_variants(rng, t)
    original = dsp._forest(independent_rng, t)
    if not np.array_equal(variants["original"], original):
        raise ValueError("Forest stage reconstruction changed original source bytes")
    if rng.bit_generator.state != independent_rng.bit_generator.state:
        raise ValueError("Forest stage reconstruction changed original RNG draws")
    hashes = {method: _digest(variant) for method, variant in variants.items()}
    errors = {}
    for method in CHIRP_METHODS[1:]:
        diff = variants[method].astype(np.float64) - original.astype(np.float64)
        errors[method] = {
            "max_absolute_error": float(np.max(np.abs(diff))),
            "different_float32_samples": int(np.count_nonzero(variants[method] != original)),
        }
    return {
        "seed": seed,
        "prefix": list(prefix),
        "original_rng_exact": True,
        "original_forest_exact": True,
        "chirps": trace["chirps"],
        "chirp_count": trace["chirp_count"],
        "layer_hashes": hashes,
        "candidate_errors": errors,
    }


def build_report() -> dict:
    records = [trace_one(seed, prefix)
               for seed in SEEDS for prefix in PREFIXES]
    result = {
        "protocol": PROTOCOL,
        "source_hz": dsp.SAMPLE_RATE,
        "source_frames": FRAMES,
        "records": records,
        "environment": _environment(),
        "original_generator_modified": False,
        "candidate_used_for_neural_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
        "globally_new_neural_evaluation_seeds": False,
    }
    validate_report(result)
    return result


def validate_report(data: dict) -> None:
    if (not isinstance(data, dict) or data.get("protocol") != PROTOCOL
            or data.get("source_hz") != dsp.SAMPLE_RATE
            or data.get("source_frames") != FRAMES
            or data.get("original_generator_modified") is not False
            or data.get("candidate_used_for_neural_training") is not False
            or data.get("audio_files_saved") is not False
            or data.get("model_weights_saved") is not False
            or data.get("globally_new_neural_evaluation_seeds") is not False):
        raise ValueError("Invalid first-party forest numerical research metadata")
    records = data.get("records")
    if not isinstance(records, list) or len(records) != len(SEEDS) * len(PREFIXES):
        raise ValueError("Incomplete original forest numerical context grid")
    observed = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid forest trace row")
        try:
            key = (record["seed"], tuple(record["prefix"]))
            if (type(record["seed"]) is not int
                    or key not in {(seed, prefix) for seed in SEEDS for prefix in PREFIXES}
                    or key in observed
                    or record.get("original_rng_exact") is not True
                    or record.get("original_forest_exact") is not True):
                raise ValueError
            observed.add(key)
            if (record.get("chirp_count") != 2
                    or not isinstance(record.get("chirps"), list)
                    or len(record["chirps"]) != 2):
                raise ValueError
            if set(record["layer_hashes"]) != set(CHIRP_METHODS):
                raise ValueError
            if any(not _hash_ok(x) for x in record["layer_hashes"].values()):
                raise ValueError
            if set(record["candidate_errors"]) != set(CHIRP_METHODS[1:]):
                raise ValueError
            for method in CHIRP_METHODS[1:]:
                metrics = record["candidate_errors"][method]
                maximum = metrics["max_absolute_error"]
                changed = metrics["different_float32_samples"]
                if (type(maximum) not in (int, float)
                        or not math.isfinite(maximum) or not 0 <= maximum < 2e-5
                        or type(changed) is not int or not 0 <= changed <= FRAMES):
                    raise ValueError
            for chirp in record["chirps"]:
                hashes, order = chirp["hashes"], chirp["stage_order"]
                if (not isinstance(hashes, dict) or not isinstance(order, list)
                        or set(order) != set(hashes) or len(order) != len(hashes)
                        or len(order) != len(set(order))
                        or any(not _hash_ok(x) for x in hashes.values())
                        or type(chirp.get("start_frame")) is not int
                        or type(chirp.get("frames")) is not int
                        or not 0 <= chirp["start_frame"] < FRAMES
                        or not 1 <= chirp["frames"] <= FRAMES
                        or type(chirp.get("index")) is not int
                        or chirp["index"] not in (0, 1)):
                    raise ValueError
        except (ValueError, KeyError, TypeError):
            raise ValueError("Invalid forest chirp fingerprint or identity") from None
    if len(observed) != len(SEEDS) * len(PREFIXES):
        raise ValueError("Incomplete forest source/seed coverage")


def compare_reports(old: dict, new: dict) -> dict:
    validate_report(old)
    validate_report(new)
    a = {(r["seed"], tuple(r["prefix"])): r for r in old["records"]}
    b = {(r["seed"], tuple(r["prefix"])): r for r in new["records"]}
    first_diffs = []
    counts = {}
    for key in a:
        before, after = a[key], b[key]
        diff = {
            method: before["layer_hashes"][method] != after["layer_hashes"][method]
            for method in CHIRP_METHODS
        }
        prefix_label = "+".join(key[1]) or "forest only"
        if any(diff.values()):
            row = {"seed": key[0], "prefix": prefix_label,
                   "different_layers": diff, "chirps": []}
            for left, right in zip(before["chirps"], after["chirps"]):
                if left["stage_order"] != right["stage_order"]:
                    raise ValueError("Forest stage execution order changed")
                stages = [name for name in left["stage_order"]
                          if left["hashes"][name] != right["hashes"][name]]
                row["chirps"].append({
                    "index": left["index"], "first_different_stage": stages[0] if stages else None,
                    "different_stages": stages,
                })
            first_diffs.append(row)
        counts.setdefault(str(key[0]), {"contexts": 0, "original_different": 0,
                                        "double_different": 0, "scalar_different": 0})
        item = counts[str(key[0])]
        item["contexts"] += 1
        item["original_different"] += diff["original"]
        item["double_different"] += diff["numpy_float64"]
        item["scalar_different"] += diff["python_math_sin"]
    return {
        "protocol": "sonumio-forest-stage-numpy-comparison-v1",
        "records": len(a),
        "same_cpu_model": old["environment"]["cpu_model"] == new["environment"]["cpu_model"],
        "numpy_versions": [old["environment"]["numpy"], new["environment"]["numpy"]],
        "by_seed": counts,
        "first_differences": first_diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare:
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.compare]
        value = compare_reports(reports[0], reports[1])
    else:
        value = build_report()
    encoded = json.dumps(value, sort_keys=True, indent=2,
                         allow_nan=False, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
