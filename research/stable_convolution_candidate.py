"""Research-only candidate for reproducible first-party DSP convolution.

No existing source renderer or training corpus calls this implementation.
The candidate computes products as Python binary64 scalars, uses math.fsum,
and rounds ONCE to float32. Cross-host byte identity is a hypothesis to test,
not a guarantee on all Python/NumPy/CPU combinations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.dsp_convolution_probe import (
    FRAMES, SAMPLE_RATE, SEEDS, SOURCES, isolate_one,
)
from research.dsp_pipeline_trace import digest_array
from research.data_reproducibility import _digest, _environment, _hash_ok

PROTOCOL = "sonumio-stable-convolution-candidate-v1"
SAMPLE_COUNT = 257
MAX_INPUT_FRAMES = 50000
MAX_KERNEL_FRAMES = 256
MAX_SELECTED_POSITIONS = 2048


def selected_scalar_convolution(
    samples: np.ndarray,
    kernel: np.ndarray,
    positions: tuple[int, ...],
    *,
    mode: str,
) -> np.ndarray:
    """Return selected convolution samples with specified scalar summation.

    Float32 inputs are promoted EXACTLY to Python float (binary64); each
    product is binary64 and math.fsum performs a compensated summation.
    Output is cast to float32 once. The original NumPy convolution may use
    different accumulation and rounding, so bitwise agreement is NOT
    required. This deliberately slow reference is research-only.
    """
    if (
        not isinstance(samples, np.ndarray)
        or not isinstance(kernel, np.ndarray)
        or samples.ndim != 1
        or kernel.ndim != 1
        or samples.dtype != np.float32
        or kernel.dtype != np.float32
        or not 1 <= samples.size <= MAX_INPUT_FRAMES
        or not 1 <= kernel.size <= min(samples.size, MAX_KERNEL_FRAMES)
        or not np.isfinite(samples).all()
        or not np.isfinite(kernel).all()
        or mode not in ("same", "full")
    ):
        raise ValueError("Expected bounded finite float32 convolution inputs")
    if (
        not isinstance(positions, tuple)
        or not 1 <= len(positions) <= MAX_SELECTED_POSITIONS
        or any(type(p) is not int for p in positions)
        or len(set(positions)) != len(positions)
    ):
        raise ValueError("Positions must be distinct Python integers")
    length = samples.size if mode == "same" else samples.size + kernel.size - 1
    if any(p < 0 or p >= length for p in positions):
        raise ValueError("Position out of convolution output bounds")

    # For a <= b kernels, np.convolve(a,b,'same') is centered at
    # floor((kernel_length - 1) / 2) in full-convolution coordinates.
    origin = (kernel.size - 1) // 2 if mode == "same" else 0
    values = np.empty(len(positions), dtype=np.float32)
    for row, position in enumerate(positions):
        index = position + origin
        left = max(0, index - (samples.size - 1))
        right = min(kernel.size - 1, index)
        estimate = math.fsum(
            float(samples[index - j]) * float(kernel[j])
            for j in range(left, right + 1)
        )
        if not math.isfinite(estimate):
            raise ValueError("Overflow during scalar convolution")
        values[row] = estimate
    if not np.isfinite(values).all():
        raise ValueError("Float32 output overflowed")
    return values


def _source_inputs(source: str, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Recreate original *inputs*, not a replacement sound generator."""
    if source not in SOURCES or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Unsupported source/seed for original corpus")
    rng = np.random.default_rng(seed)
    if source == "rain":
        # _rain consumes a full hiss draw BEFORE creating its impulses.
        rng.standard_normal(FRAMES).astype(np.float32)
        impulses = np.zeros(FRAMES, dtype=np.float32)
        indices = rng.integers(0, FRAMES, size=int(110 * FRAMES / SAMPLE_RATE))
        np.add.at(impulses, indices, rng.uniform(0.1, 0.6, size=len(indices)))
        kernel = np.exp(-np.arange(180) / 38).astype(np.float32)
        return impulses, kernel, "full"
    width = {"thunder": 140, "wind": 60, "water": 12}[source]
    noise = rng.standard_normal(FRAMES).astype(np.float32)
    kernel = np.ones(width, dtype=np.float32) / width
    return noise, kernel, "same"


def _positions(length: int) -> tuple[int, ...]:
    # Pure integer arithmetic ensures every run samples the same locations.
    points = {(i * (length - 1)) // (SAMPLE_COUNT - 1)
              for i in range(SAMPLE_COUNT)}
    points.update((0, 1, 11, 59, 139, 179, length // 2, length - 2, length - 1))
    return tuple(sorted(points))


def probe_one(source: str, seed: int) -> dict:
    original = isolate_one(source, seed)  # Unmodified DSP is correctness gate.
    samples, kernel, mode = _source_inputs(source, seed)
    if (
        original["mode"] != mode
        or original["stages"]["input"]["sha256"] != digest_array(samples)["sha256"]
        or original["stages"]["kernel"]["sha256"] != digest_array(kernel)["sha256"]
    ):
        raise ValueError("Candidate source inputs differ from original DSP inputs")
    positions = _positions(FRAMES)
    candidate = selected_scalar_convolution(samples, kernel, positions, mode=mode)
    actual = np.convolve(samples, kernel, mode=mode)[
        :FRAMES if mode == "full" else None
    ][np.asarray(positions)]
    # No unearned perceptual claim: this is a sample-wise numeric tolerance.
    deviation = np.abs(candidate.astype(np.float64) - actual.astype(np.float64))
    max_abs = float(np.max(deviation))
    if not math.isfinite(max_abs) or max_abs > 2e-5:
        raise ValueError("Candidate diverged too far from original convolution")
    return {
        "source": source,
        "seed": seed,
        "mode": mode,
        "kernel_frames": int(kernel.size),
        "sampled_positions": len(positions),
        "input_sha256": original["stages"]["input"]["sha256"],
        "kernel_sha256": original["stages"]["kernel"]["sha256"],
        "original_convolution_full_sha256": original["stages"]["convolution"]["sha256"],
        "original_selected_sha256": digest_array(actual)["sha256"],
        "candidate_selected_sha256": digest_array(candidate)["sha256"],
        "different_selected_float32_values": int(np.count_nonzero(actual != candidate)),
        "max_selected_absolute_error": max_abs,
        "rms_selected_absolute_error": float(np.sqrt(np.mean(deviation ** 2))),
    }


def _aggregate(rows: list[dict]) -> dict:
    return {
        source: {
            "cases": len(selected),
            "candidate_sha256": _digest(b"".join(
                bytes.fromhex(row["candidate_selected_sha256"]) for row in selected
            )),
            "original_selected_sha256": _digest(b"".join(
                bytes.fromhex(row["original_selected_sha256"]) for row in selected
            )),
            "original_full_sha256": _digest(b"".join(
                bytes.fromhex(row["original_convolution_full_sha256"])
                for row in selected
            )),
            "max_selected_absolute_error": max(
                row["max_selected_absolute_error"] for row in selected
            ),
            "different_selected_float32_values": sum(
                row["different_selected_float32_values"] for row in selected
            ),
        }
        for source in SOURCES
        for selected in [[row for row in rows if row["source"] == source]]
    }


def build_probe() -> dict:
    records = [probe_one(source, seed) for source in SOURCES for seed in SEEDS]
    result = {
        "protocol": PROTOCOL,
        "original_dsp_unchanged": True,
        "candidate_is_experimental_only": True,
        "sample_rate": SAMPLE_RATE,
        "full_frames_per_source": FRAMES,
        "records": records,
        "source_aggregate": _aggregate(records),
        "environment": _environment(),
        "audio_files_saved": False,
        "weights_saved": False,
        "caveat": (
            "Sampled numerical agreement does not establish full-waveform "
            "error bounds, perceptual quality, or universal cross-CPU identity."
        ),
    }
    validate_probe(result)
    return result


def validate_probe(report: dict) -> None:
    if (
        not isinstance(report, dict)
        or report.get("protocol") != PROTOCOL
        or report.get("original_dsp_unchanged") is not True
        or report.get("candidate_is_experimental_only") is not True
        or report.get("sample_rate") != SAMPLE_RATE
        or report.get("full_frames_per_source") != FRAMES
        or report.get("audio_files_saved") is not False
        or report.get("weights_saved") is not False
    ):
        raise ValueError("Invalid candidate study metadata")
    records = report.get("records")
    expected = {(source, seed) for source in SOURCES for seed in SEEDS}
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("Candidate source grid incomplete")
    seen = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Invalid candidate row")
        identity = (row.get("source"), row.get("seed"))
        if identity not in expected or identity in seen:
            raise ValueError("Duplicate or unexpected candidate source/seed")
        seen.add(identity)
        if (
            row.get("mode") != ("full" if identity[0] == "rain" else "same")
            or row.get("kernel_frames")
            != {"rain": 180, "thunder": 140, "wind": 60, "water": 12}[identity[0]]
            or row.get("sampled_positions") != len(_positions(FRAMES))
        ):
            raise ValueError("Invalid candidate dimensional metadata")
        for key in (
            "input_sha256", "kernel_sha256",
            "original_convolution_full_sha256",
            "original_selected_sha256", "candidate_selected_sha256",
        ):
            if not _hash_ok(row.get(key)):
                raise ValueError("Malformed study digest")
        error = row.get("max_selected_absolute_error")
        rms = row.get("rms_selected_absolute_error")
        count = row.get("different_selected_float32_values")
        if (
            type(error) not in (float, int)
            or type(rms) not in (float, int)
            or not math.isfinite(error)
            or not math.isfinite(rms)
            or not 0 <= rms <= error <= 2e-5
            or type(count) is not int
            or not 0 <= count <= row["sampled_positions"]
        ):
            raise ValueError("Invalid study numeric metrics")
    if seen != expected or report.get("source_aggregate") != _aggregate(records):
        raise ValueError("Inconsistent candidate study aggregate")


def compare_probes(first: dict, second: dict) -> dict:
    validate_probe(first)
    validate_probe(second)
    a = {(r["source"], r["seed"]): r for r in first["records"]}
    b = {(r["source"], r["seed"]): r for r in second["records"]}
    return {
        "protocol": "sonumio-stable-convolution-candidate-comparison-v1",
        "cases_compared": len(a),
        "same_input_and_kernel_hashes": all(
            a[k][field] == b[k][field]
            for k in a for field in ("input_sha256", "kernel_sha256")
        ),
        "candidate_sampled_hashes_equal": all(
            a[k]["candidate_selected_sha256"] == b[k]["candidate_selected_sha256"]
            for k in a
        ),
        "original_sampled_hashes_equal": all(
            a[k]["original_selected_sha256"] == b[k]["original_selected_sha256"]
            for k in a
        ),
        "not_a_proof_of_full_audio_reproducibility": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_probe()
    encoded = json.dumps(report, sort_keys=True, indent=2,
                         ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
