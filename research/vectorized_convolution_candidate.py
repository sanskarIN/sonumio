"""Research-only, *unproven*, vectorized float64 DSP convolution candidate.

A faster algorithm to compare against the independent first-party scalar
math.fsum reference. DO NOT substitute this into training, original
procedural synthesis, public API, or published dataset recipes. Matching
float64 arithmetic alone does not guarantee identical float32 rounding on
different CPUs, NumPy versions or near rounding midpoints. Probe exact bytes.

No weights, waveforms or WAV files are serialized: JSON contains SHA256
digests and allowlisted numerical diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_convolution_probe import (
    FRAMES, SAMPLE_RATE, SOURCES, SEEDS, isolate_one,
)
from research.dsp_pipeline_trace import digest_array
from research.stable_convolution_candidate import (
    _positions, _source_inputs, selected_scalar_convolution,
)

PROTOCOL = "sonumio-vectorized-float64-convolution-candidate-v1"
# Verified scalar fsum full-output SHA aggregates from 4-host workflow
# #44060065472. These are hashes OF CONCATENATED PER-SEED HASHES,
# not individual SHA256s, not raw audio files and not neural weights.
SCALAR_SOURCE_REFERENCE = {
    "rain": "4b252e3bc9361b2a50cc5bdd0a65a12ac91f28de50504590270abd068fd81ff5",
    "thunder": "e2244e816126366e569ee03b0f9a70807adb1209811617933efdaf764fa0eba6",
    "wind": "a57ea85fe044f6b2f17ea66a9ca5c8232be099039c1c1e95becc2660aeeb1bf1",
    "water": "875fbf371c3282499838d142b909737821e36706d69eda6a3c5191c2a92a24ab",
}


def vectorized_float64_convolution(
    samples: np.ndarray, kernel: np.ndarray, *, mode: str
) -> np.ndarray:
    """Compute float64 np.convolve, with exactly one final float32 cast.

    NOT claimed universally reproducible. Numerical topology, rounding,
    backend implementations and host CPUs are controlled only by tests.
    Reuse the scalar validator to enforce bounded dimensions and floats.
    """
    selected_scalar_convolution(samples, kernel, (0,), mode=mode)
    a = np.asarray(samples, dtype=np.float64)
    b = np.asarray(kernel, dtype=np.float64)
    result = np.convolve(a, b, mode=mode).astype(np.float32)
    expected = len(samples) if mode == "same" else len(samples) + len(kernel) - 1
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ValueError("Invalid vectorized double-accumulation output")
    return result


def probe_one(source: str, seed: int) -> dict:
    # Validate exact original-seed inputs before measuring an alternative.
    original = isolate_one(source, seed)
    inputs, kernel, mode = _source_inputs(source, seed)
    if (digest_array(inputs)["sha256"] != original["stages"]["input"]["sha256"]
            or digest_array(kernel)["sha256"] != original["stages"]["kernel"]["sha256"]
            or original["mode"] != mode):
        raise ValueError("Source input or kernel changed from original DSP")
    started = perf_counter()
    output = vectorized_float64_convolution(inputs, kernel, mode=mode)
    vector_time = perf_counter() - started
    if not math.isfinite(vector_time) or vector_time <= 0:
        raise ValueError("Invalid vectorized convolution time")
    positions = _positions(FRAMES)
    reference = selected_scalar_convolution(inputs, kernel, positions, mode=mode)
    actual = output[np.asarray(positions)]
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    difference = float(delta.max())
    if not math.isfinite(difference) or difference > 2e-5:
        raise ValueError("Double-convolution candidate differs excessively")
    # Original full output is permitted to be CPU-dependent, but the
    # source input/kernel must match the original diagnostic exactly.
    return {
        "source": source,
        "seed": seed,
        "kernel_frames": int(kernel.size),
        "mode": mode,
        "convolution_frames": int(output.size),
        "input_sha256": original["stages"]["input"]["sha256"],
        "kernel_sha256": original["stages"]["kernel"]["sha256"],
        "original_numpy_legacy_crop_sha256": original["stages"]["convolution"]["sha256"],
        "vectorized_full_sha256": digest_array(output)["sha256"],
        "vectorized_legacy_crop_sha256": digest_array(output[:FRAMES])["sha256"],
        "reference_selected_sha256": digest_array(reference)["sha256"],
        "vectorized_selected_sha256": digest_array(actual)["sha256"],
        "reference_selected_frames": len(positions),
        "reference_selected_max_abs_error": difference,
        "reference_selected_different_values": int(np.count_nonzero(actual != reference)),
        "vectorized_seconds": float(vector_time),
    }


def _aggregate(rows: list[dict]) -> dict:
    result = {}
    for source in SOURCES:
        selected = [row for row in rows if row["source"] == source]
        digest = _digest(b"".join(
            bytes.fromhex(row["vectorized_full_sha256"]) for row in selected
        ))
        result[source] = {
            "cases": len(selected),
            "vectorized_full_sha256": digest,
            "matches_verified_scalar_full_reference": (
                digest == SCALAR_SOURCE_REFERENCE[source]
            ),
            "max_selected_absolute_error": max(
                row["reference_selected_max_abs_error"] for row in selected
            ),
            "changed_selected_float32_values": sum(
                row["reference_selected_different_values"] for row in selected
            ),
            "median_vectorized_seconds": float(np.median([
                row["vectorized_seconds"] for row in selected
            ])),
        }
    return result


def build_probe() -> dict:
    rows = [probe_one(source, seed) for source in SOURCES for seed in SEEDS]
    report = {
        "protocol": PROTOCOL,
        "source_frames": FRAMES,
        "sample_rate": SAMPLE_RATE,
        "reference": "verified scalar fsum source-convolution output hashes",
        "recordings": rows,
        "sources": _aggregate(rows),
        "environment": _environment(),
        "original_dsp_unchanged": True,
        "vectorized_candidate_for_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
        "reference_is_not_a_perceptual_quality_gate": True,
    }
    validate_probe(report)
    return report


def validate_probe(report: dict) -> None:
    if (not isinstance(report, dict)
            or report.get("protocol") != PROTOCOL
            or report.get("source_frames") != FRAMES
            or report.get("sample_rate") != SAMPLE_RATE
            or report.get("original_dsp_unchanged") is not True
            or report.get("vectorized_candidate_for_training") is not False
            or report.get("audio_files_saved") is not False
            or report.get("model_weights_saved") is not False
            or report.get("reference_is_not_a_perceptual_quality_gate") is not True):
        raise ValueError("Invalid vectorized study metadata")
    rows = report.get("recordings")
    expected = {(source, seed) for source in SOURCES for seed in SEEDS}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Incomplete vectorized source grid")
    seen = set()
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("Invalid vectorized row")
        identity = (r.get("source"), r.get("seed"))
        if identity not in expected or identity in seen:
            raise ValueError("Duplicate/invalid vectorized source identity")
        seen.add(identity)
        taps = {"rain": 180, "thunder": 140, "wind": 60, "water": 12}[identity[0]]
        mode = "full" if identity[0] == "rain" else "same"
        if (r.get("kernel_frames") != taps
                or r.get("mode") != mode
                or r.get("convolution_frames")
                != (FRAMES + taps - 1 if mode == "full" else FRAMES)
                or r.get("reference_selected_frames") != len(_positions(FRAMES))):
            raise ValueError("Invalid source convolution dimensions")
        for field in (
            "input_sha256", "kernel_sha256",
            "original_numpy_legacy_crop_sha256",
            "vectorized_full_sha256", "vectorized_legacy_crop_sha256",
            "reference_selected_sha256", "vectorized_selected_sha256",
        ):
            if not _hash_ok(r.get(field)):
                raise ValueError("Invalid vectorized provenance SHA")
        diff = r.get("reference_selected_max_abs_error")
        elapsed = r.get("vectorized_seconds")
        changed = r.get("reference_selected_different_values")
        if (type(diff) not in (int, float)
                or type(elapsed) not in (int, float)
                or not math.isfinite(diff) or not math.isfinite(elapsed)
                or not 0 <= diff <= 2e-5 or elapsed <= 0
                or type(changed) is not int
                or not 0 <= changed <= r["reference_selected_frames"]):
            raise ValueError("Invalid vectorized numerical diagnostics")
    if seen != expected or report.get("sources") != _aggregate(rows):
        raise ValueError("Invalid per-source vectorized digest aggregation")


def compare_probes(first: dict, second: dict) -> dict:
    validate_probe(first)
    validate_probe(second)
    one = {(r["source"], r["seed"]): r for r in first["recordings"]}
    two = {(r["source"], r["seed"]): r for r in second["recordings"]}
    return {
        "protocol": "sonumio-vectorized-double-candidate-comparison-v1",
        "cases": len(one),
        "identical_source_inputs_and_kernels": all(
            one[k][field] == two[k][field]
            for k in one for field in ("input_sha256", "kernel_sha256")
        ),
        "identical_vectorized_full_bytes": all(
            one[k]["vectorized_full_sha256"] == two[k]["vectorized_full_sha256"]
            for k in one
        ),
        "matches_scalar_reference_all_sources": (
            all(x["matches_verified_scalar_full_reference"]
                for x in first["sources"].values())
            and all(x["matches_verified_scalar_full_reference"]
                    for x in second["sources"].values())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    data = build_probe()
    encoded = json.dumps(data, sort_keys=True, indent=2,
                         allow_nan=False, ensure_ascii=False) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
