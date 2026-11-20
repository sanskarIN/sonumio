"""Full-frame CPU reproducibility experiment for a separate scalar DSP candidate.

This module does NOT change the production generator, original neural
training recordings, versioned synthetic recipe or model checkpoints.
It records only hashes and numerical metrics, never audio arrays or weights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_convolution_probe import (
    FRAMES, SAMPLE_RATE, SOURCES, SEEDS, isolate_one,
)
from research.dsp_pipeline_trace import digest_array
from research.stable_convolution_candidate import (
    _source_inputs, selected_scalar_convolution,
)

PROTOCOL = "sonumio-full-frame-scalar-convolution-candidate-v1"
BLOCK_SIZE = 1024


def full_scalar_convolution(
    samples: np.ndarray, kernel: np.ndarray, *, mode: str
) -> np.ndarray:
    """Evaluate *every* output using bounded scalar binary64 fsum blocks.

    This materializes the full convolution (including the full-mode tail)
    and rounds each final sample once to float32. Input limits and type
    validations are exactly those of selected_scalar_convolution.
    """
    selected_scalar_convolution(samples, kernel, (0,), mode=mode)
    length = len(samples) if mode == "same" else len(samples) + len(kernel) - 1
    output = np.empty(length, dtype=np.float32)
    for start in range(0, length, BLOCK_SIZE):
        end = min(length, start + BLOCK_SIZE)
        output[start:end] = selected_scalar_convolution(
            samples, kernel, tuple(range(start, end)), mode=mode
        )
    if not np.isfinite(output).all():
        raise ValueError("Non-finite scalar convolution output")
    return output


def probe_one_full(source: str, seed: int) -> dict:
    original = isolate_one(source, seed)
    samples, kernel, mode = _source_inputs(source, seed)
    if (
        original["mode"] != mode
        or original["stages"]["input"]["sha256"] != digest_array(samples)["sha256"]
        or original["stages"]["kernel"]["sha256"] != digest_array(kernel)["sha256"]
    ):
        raise ValueError("Original source input/kernel identity changed")
    original_result = np.convolve(samples, kernel, mode=mode)
    candidate = full_scalar_convolution(samples, kernel, mode=mode)
    if candidate.shape != original_result.shape:
        raise ValueError("Full-frame candidate output length differs")
    if (original["stages"]["convolution"]["sha256"]
            != digest_array(original_result[:FRAMES])["sha256"]):
        raise ValueError("Full-frame original NumPy output differs from source oracle")
    error = candidate.astype(np.float64) - original_result.astype(np.float64)
    max_abs = float(np.max(np.abs(error)))
    rms = float(np.sqrt(np.square(error).mean()))
    if not math.isfinite(max_abs) or max_abs > 2e-5:
        raise ValueError("Full-frame candidate numerical deviation exceeds bound")
    return {
        "source": source,
        "seed": seed,
        "mode": mode,
        "kernel_frames": int(kernel.size),
        "full_output_frames": int(candidate.size),
        "legacy_cropped_frames": FRAMES,
        "input_sha256": original["stages"]["input"]["sha256"],
        "kernel_sha256": original["stages"]["kernel"]["sha256"],
        "original_full_output_sha256": digest_array(original_result)["sha256"],
        "candidate_full_output_sha256": digest_array(candidate)["sha256"],
        "original_legacy_crop_sha256": original["stages"]["convolution"]["sha256"],
        "candidate_legacy_crop_sha256": digest_array(candidate[:FRAMES])["sha256"],
        "different_full_output_float32_values": int(np.count_nonzero(
            candidate != original_result
        )),
        "max_full_output_absolute_error": max_abs,
        "rms_full_output_absolute_error": rms,
    }


def _aggregate(records: list[dict]) -> dict:
    out = {}
    for source in SOURCES:
        group = [r for r in records if r["source"] == source]
        out[source] = {
            "cases": len(group),
            "candidate_full_sha256": _digest(b"".join(
                bytes.fromhex(r["candidate_full_output_sha256"]) for r in group
            )),
            "original_full_sha256": _digest(b"".join(
                bytes.fromhex(r["original_full_output_sha256"]) for r in group
            )),
            "candidate_legacy_crop_sha256": _digest(b"".join(
                bytes.fromhex(r["candidate_legacy_crop_sha256"]) for r in group
            )),
            "original_legacy_crop_sha256": _digest(b"".join(
                bytes.fromhex(r["original_legacy_crop_sha256"]) for r in group
            )),
            "max_full_output_absolute_error": max(
                r["max_full_output_absolute_error"] for r in group
            ),
            "different_full_output_float32_values": sum(
                r["different_full_output_float32_values"] for r in group
            ),
        }
    return out


def build_full_probe() -> dict:
    rows = [probe_one_full(source, seed) for source in SOURCES for seed in SEEDS]
    report = {
        "protocol": PROTOCOL,
        "sample_rate": SAMPLE_RATE,
        "original_dsp_unchanged": True,
        "scalar_candidate_only": True,
        "full_convolution_evaluated": True,
        "records": rows,
        "source_aggregate": _aggregate(rows),
        "environment": _environment(),
        "audio_files_saved": False,
        "model_weights_saved": False,
        "interpretation": (
            "Original-source convolutions only. Matching full source-convolution "
            "hashes do not establish full DSP waveform/crop or neural-training "
            "dataset identity, throughput, licensing or audible quality."
        ),
    }
    validate_full_probe(report)
    return report


def validate_full_probe(report: dict) -> None:
    if (
        not isinstance(report, dict)
        or report.get("protocol") != PROTOCOL
        or report.get("sample_rate") != SAMPLE_RATE
        or report.get("original_dsp_unchanged") is not True
        or report.get("scalar_candidate_only") is not True
        or report.get("full_convolution_evaluated") is not True
        or report.get("audio_files_saved") is not False
        or report.get("model_weights_saved") is not False
    ):
        raise ValueError("Invalid full-frame scalar study metadata")
    rows = report.get("records")
    expected = {(source, seed) for source in SOURCES for seed in SEEDS}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Incomplete full-frame source/seed grid")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid full-frame row")
        key = (row.get("source"), row.get("seed"))
        if key not in expected or key in seen:
            raise ValueError("Duplicate/invalid full-frame source/seed")
        seen.add(key)
        taps = {"rain": 180, "thunder": 140, "wind": 60, "water": 12}[key[0]]
        mode = "full" if key[0] == "rain" else "same"
        if (
            row.get("mode") != mode
            or row.get("kernel_frames") != taps
            or row.get("full_output_frames")
            != (FRAMES + taps - 1 if mode == "full" else FRAMES)
            or row.get("legacy_cropped_frames") != FRAMES
        ):
            raise ValueError("Unexpected full-frame output dimensions")
        for field in (
            "input_sha256", "kernel_sha256",
            "original_full_output_sha256", "candidate_full_output_sha256",
            "original_legacy_crop_sha256", "candidate_legacy_crop_sha256",
        ):
            if not _hash_ok(row.get(field)):
                raise ValueError("Malformed full-frame digest")
        max_abs = row.get("max_full_output_absolute_error")
        rms = row.get("rms_full_output_absolute_error")
        changed = row.get("different_full_output_float32_values")
        if (
            type(max_abs) not in (int, float)
            or type(rms) not in (int, float)
            or not math.isfinite(max_abs)
            or not math.isfinite(rms)
            or not 0 <= rms <= max_abs <= 2e-5
            or type(changed) is not int
            or not 0 <= changed <= row["full_output_frames"]
        ):
            raise ValueError("Invalid full-frame numerical metrics")
    if seen != expected or report.get("source_aggregate") != _aggregate(rows):
        raise ValueError("Inconsistent full-frame study aggregate")


def compare_full_probes(first: dict, second: dict) -> dict:
    validate_full_probe(first)
    validate_full_probe(second)
    a = {(r["source"], r["seed"]): r for r in first["records"]}
    b = {(r["source"], r["seed"]): r for r in second["records"]}
    return {
        "protocol": "sonumio-full-frame-scalar-candidate-comparison-v1",
        "cases": len(a),
        "same_inputs_and_kernels": all(
            a[k][field] == b[k][field]
            for k in a for field in ("input_sha256", "kernel_sha256")
        ),
        "candidate_full_hashes_equal": all(
            a[k]["candidate_full_output_sha256"]
            == b[k]["candidate_full_output_sha256"] for k in a
        ),
        "original_full_hashes_equal": all(
            a[k]["original_full_output_sha256"]
            == b[k]["original_full_output_sha256"] for k in a
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_full_probe()
    result = json.dumps(report, sort_keys=True, indent=2,
                        ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, end="")


if __name__ == "__main__":
    main()
