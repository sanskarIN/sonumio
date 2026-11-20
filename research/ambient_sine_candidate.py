"""Isolated reproducible-oscillator hypotheses for Sonumio's ambient source.

The original `dsp._drone` uses four float32 NumPy sine ufuncs. Its 110-Hz
and 220-Hz components differed between NumPy 1.26.4 and 2.4.6 on one CPU.
This module substitutes ONLY sine evaluation on the same float32 phase
arguments, and preserves original float32 weighting/mixing order.

Neither candidate is connected to DSP inference, neural training or public
model weights. The original drone is an unchanged numerical oracle.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from research.data_reproducibility import _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-ambient-sine-numerical-candidate-v1"
FRAMES = round(dsp.SAMPLE_RATE * 1.5)
METHODS = ("numpy_float64", "python_math_sin")


def _validate_time(t: np.ndarray) -> None:
    if (not isinstance(t, np.ndarray) or t.ndim != 1
            or t.dtype != np.float32 or not 1 <= len(t) <= 50000
            or not np.isfinite(t).all()):
        raise ValueError("Expected bounded finite float32 time positions")


def _sin_float32_phase(phase: np.ndarray, *, method: str) -> np.ndarray:
    if (not isinstance(phase, np.ndarray) or phase.ndim != 1
            or phase.dtype != np.float32 or not 1 <= len(phase) <= 50000
            or not np.isfinite(phase).all() or method not in METHODS):
        raise ValueError("Invalid bounded float32-phase sine request")
    if method == "numpy_float64":
        double = np.sin(phase.astype(np.float64))
    else:
        double = np.fromiter(
            (math.sin(float(x)) for x in phase),
            dtype=np.float64, count=phase.size,
        )
    result = double.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite numeric sine candidate")
    return result


def candidate_drone(t: np.ndarray, *, method: str = "numpy_float64") -> np.ndarray:
    """Preserve original DSP ordering; only the sine implementation differs."""
    _validate_time(t)
    if method not in METHODS:
        raise ValueError("Unknown ambient sine candidate")
    sin_110 = _sin_float32_phase(2 * np.pi * 110 * t, method=method)
    sin_164 = _sin_float32_phase(2 * np.pi * 164.81 * t, method=method)
    sin_220 = _sin_float32_phase(2 * np.pi * 220.0 * t, method=method)
    sin_lfo = _sin_float32_phase(2 * np.pi * 0.4 * t, method=method)
    wave = sin_110 + 0.55 * sin_164 + 0.25 * sin_220
    lfo = 0.38 + 0.08 * sin_lfo
    output = (wave * lfo).astype(np.float32)
    if not np.isfinite(output).all() or output.shape != t.shape:
        raise ValueError("Invalid ambient oscillator output")
    return output


def build_probe() -> dict:
    t = np.arange(FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    original = dsp._drone(t)
    if (original.shape != (FRAMES,) or original.dtype != np.float32
            or not np.isfinite(original).all()):
        raise ValueError("Original ambient reference is invalid")
    report = {}
    for method in METHODS:
        started = perf_counter()
        alternative = candidate_drone(t, method=method)
        elapsed = perf_counter() - started
        delta = alternative.astype(np.float64) - original.astype(np.float64)
        result = {
            "sha256": digest_array(alternative)["sha256"],
            "original_sha256": digest_array(original)["sha256"],
            "different_float32_values": int(np.count_nonzero(alternative != original)),
            "max_absolute_difference": float(np.max(np.abs(delta))),
            "rms_absolute_difference": float(np.sqrt(np.square(delta).mean())),
            "elapsed_seconds": float(elapsed),
        }
        report[method] = result
    value = {
        "protocol": PROTOCOL,
        "frames": FRAMES,
        "sample_rate": dsp.SAMPLE_RATE,
        "original_source_unchanged": True,
        "candidate_used_for_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
        "methods": report,
        "environment": _environment(),
    }
    validate_probe(value)
    return value


def validate_probe(value: dict) -> None:
    if (not isinstance(value, dict) or value.get("protocol") != PROTOCOL
            or value.get("frames") != FRAMES
            or value.get("sample_rate") != dsp.SAMPLE_RATE
            or value.get("original_source_unchanged") is not True
            or value.get("candidate_used_for_training") is not False
            or value.get("audio_files_saved") is not False
            or value.get("model_weights_saved") is not False):
        raise ValueError("Invalid ambient sine research metadata")
    methods = value.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(METHODS):
        raise ValueError("Incomplete ambient sine candidate comparison")
    for method in METHODS:
        entry = methods[method]
        if (not isinstance(entry, dict)
                or not _hash_ok(entry.get("sha256"))
                or not _hash_ok(entry.get("original_sha256"))):
            raise ValueError("Malformed candidate audio fingerprints")
        changed = entry.get("different_float32_values")
        maximum = entry.get("max_absolute_difference")
        rms = entry.get("rms_absolute_difference")
        elapsed = entry.get("elapsed_seconds")
        if (type(changed) is not int or not 0 <= changed <= FRAMES
                or type(maximum) not in (int, float)
                or type(rms) not in (int, float)
                or type(elapsed) not in (int, float)
                or not all(math.isfinite(x) for x in (maximum, rms, elapsed))
                or not 0 <= rms <= maximum <= 0.005
                or not elapsed > 0):
            raise ValueError("Invalid ambient numerical error or runtime")
    if (methods[METHODS[0]]["original_sha256"]
            != methods[METHODS[1]]["original_sha256"]):
        raise ValueError("Ambient methods used different original reference")


def compare_probes(first: dict, second: dict) -> dict:
    validate_probe(first)
    validate_probe(second)
    return {
        "protocol": "sonumio-ambient-sine-hypothesis-comparison-v1",
        "same_cpu_model": (
            first["environment"]["cpu_model"] ==
            second["environment"]["cpu_model"]
        ),
        "numpy_versions": [
            first["environment"]["numpy"], second["environment"]["numpy"]
        ],
        "same_original_bytes": (
            first["methods"][METHODS[0]]["original_sha256"]
            == second["methods"][METHODS[0]]["original_sha256"]
        ),
        "candidate_hashes_match": {
            name: first["methods"][name]["sha256"] ==
                  second["methods"][name]["sha256"] for name in METHODS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    result = (
        compare_probes(*[
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.compare
        ])
        if args.compare else build_probe()
    )
    serialized = json.dumps(result, indent=2, sort_keys=True,
                            ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
