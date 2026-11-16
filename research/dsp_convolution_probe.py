"""Locate CPU-specific numerical differences inside the original DSP convolutions.

This is diagnostic-only. It uses the unchanged original rain/thunder/wind/water
source routines, recreates their inputs, kernels and convolutions and asserts
the rebuilt sound layer is byte-identical to the original private DSP
function. The output is only hashes and sampled math.fsum comparisons,
never WAV data, pretrained model weights or trained checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.latent_audio import TRAIN_SEEDS, VALIDATION_SEEDS
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-first-party-convolution-isolation-v1"
SOURCES = ("rain", "thunder", "wind", "water")
SEEDS = TRAIN_SEEDS + VALIDATION_SEEDS
SAMPLE_POSITIONS = (0, 1, 100, 500, 1000, 8000, 12000, 24000)
SAMPLE_RATE = dsp.SAMPLE_RATE
FRAMES = round(SAMPLE_RATE * 1.5)


def _reference_sample(inputs: np.ndarray, kernel: np.ndarray,
                      position: int, mode: str) -> float:
    # Reference uses scalar Python double products and math.fsum, not NumPy
    # convolve's vectorized reduction. This only samples 8 positions per case.
    origin = 0 if mode == "full" else (len(kernel) - 1) // 2
    index = position + origin
    return float(math.fsum(
        float(inputs[index - j]) * float(kernel[j])
        for j in range(len(kernel)) if 0 <= index - j < len(inputs)
    ))


def isolate_one(source: str, seed: int) -> dict:
    if source not in SOURCES or type(seed) is not int or seed not in SEEDS:
        raise ValueError("Unsupported first-party source or recording seed")
    n = FRAMES
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(seed)
    stages = {}
    if source == "rain":
        hiss = rng.standard_normal(n).astype(np.float32) * 0.055
        impulses = np.zeros(n, dtype=np.float32)
        indices = rng.integers(0, n, size=int(110 * len(t) / SAMPLE_RATE))
        np.add.at(impulses, indices, rng.uniform(0.1, 0.6, size=len(indices)))
        kernel = np.exp(-np.arange(180) / 38).astype(np.float32)
        stage_input = impulses
        mode = "full"
        convolution = np.convolve(stage_input, kernel, mode=mode)[:n]
        layer = hiss + 0.08 * convolution
        stages["rain_hiss"] = digest_array(hiss)
    else:
        width = {"thunder": 140, "wind": 60, "water": 12}[source]
        noise = rng.standard_normal(n).astype(np.float32)
        stage_input = noise
        kernel = np.ones(width, dtype=np.float32) / width
        mode = "same"
        convolution = np.convolve(stage_input, kernel, mode=mode)
        smooth = convolution.astype(np.float32)
        if source == "thunder":
            smooth /= max(float(np.max(np.abs(smooth))), 1e-6)
            center = 0.37 * t[-1] if n > 1 else 0.0
            decay = np.where(t >= center, np.exp(-(t - center) / 1.1), 0.0)
            attack = np.minimum((t - center) * 18, 1).clip(0, 1)
            layer = 0.50 * smooth * (attack * decay).astype(np.float32)
        elif source == "wind":
            gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * t)
            layer = 0.75 * smooth * gust.astype(np.float32)
        else:
            envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 0.24 * t - 0.9)) ** 2
            layer = smooth * envelope.astype(np.float32) * 1.8
    original = {
        "rain": dsp._rain, "thunder": dsp._thunder,
        "wind": dsp._wind, "water": dsp._water,
    }[source](np.random.default_rng(seed), t)
    if (not np.array_equal(original, layer)
            or original.dtype != layer.dtype
            or convolution.shape != (n,)):
        raise ValueError("Convolution diagnostic differs from original DSP layer")

    ref = np.asarray([
        _reference_sample(stage_input, kernel, pos, mode)
        for pos in SAMPLE_POSITIONS
    ], dtype=np.float64)
    actual = convolution[np.asarray(SAMPLE_POSITIONS)].astype(np.float64)
    max_reference_error = float(np.max(np.abs(actual - ref)))
    # The reference is not a bitwise identity target: the SIMD float32
    # convolution is allowed to round differently, but gross mismatch
    # would invalidate the selected index / kernel alignment.
    if not math.isfinite(max_reference_error) or max_reference_error > 2e-5:
        raise ValueError("Scalar convolution reference disagrees too strongly")
    stages.update({
        "input": digest_array(stage_input),
        "kernel": digest_array(kernel),
        "convolution": digest_array(convolution),
        "source_layer": digest_array(layer),
        "scalar_fsum_reference": digest_array(ref),
    })
    return {
        "source": source, "seed": seed, "kernel_frames": len(kernel),
        "mode": mode, "stages": stages,
        "max_selected_fsum_difference": max_reference_error,
    }


def _aggregate(records: list[dict]) -> dict:
    out = {}
    for source in SOURCES:
        rows = [r for r in records if r["source"] == source]
        labels = sorted({label for row in rows for label in row["stages"]})
        out[source] = {}
        for name in labels:
            out[source][name] = {
                "count": sum(name in row["stages"] for row in rows),
                "sha256": _digest(b"".join(
                    bytes.fromhex(row["stages"][name]["sha256"])
                    for row in rows if name in row["stages"]
                )),
                "bucket_0.001_sha256": _digest(b"".join(
                    bytes.fromhex(row["stages"][name]["bucket_0.001_sha256"])
                    for row in rows if name in row["stages"]
                )),
            }
    return out


def build_probe() -> dict:
    records = [isolate_one(source, seed) for source in SOURCES for seed in SEEDS]
    result = {
        "protocol": PROTOCOL,
        "original_source_routines_unchanged": True,
        "source_sample_rate": SAMPLE_RATE,
        "recordings": records,
        "source_stage_aggregate": _aggregate(records),
        "environment": _environment(),
        "audio_files_saved": False,
        "model_weights_saved": False,
        "interpretation": (
            "Exact source/kernel equality with different NumPy convolution "
            "SHA on another host implicates numerical convolution or its "
            "runtime, NOT proven CPU instructions or audible differences."
        ),
    }
    validate_probe(result)
    return result


def validate_probe(report: dict) -> None:
    if (not isinstance(report, dict) or report.get("protocol") != PROTOCOL
            or report.get("original_source_routines_unchanged") is not True
            or report.get("source_sample_rate") != 24000
            or report.get("audio_files_saved") is not False
            or report.get("model_weights_saved") is not False):
        raise ValueError("Invalid original convolution research report")
    rows = report.get("recordings")
    if not isinstance(rows, list) or len(rows) != len(SOURCES) * len(SEEDS):
        raise ValueError("Expected 16 source/seed convolution cases")
    observed = set()
    for row in rows:
        key = (row.get("source"), row.get("seed"))
        if key in observed or key not in {(s, seed) for s in SOURCES for seed in SEEDS}:
            raise ValueError("Unexpected or duplicate convolution case")
        observed.add(key)
        if row.get("kernel_frames") != {
            "rain": 180, "thunder": 140, "wind": 60, "water": 12,
        }[key[0]]:
            raise ValueError("Unexpected source kernel")
        if row.get("mode") != ("full" if key[0] == "rain" else "same"):
            raise ValueError("Unexpected convolution alignment mode")
        if not isinstance(row.get("stages"), dict):
            raise ValueError("Missing convolution stages")
        for name in ("input", "kernel", "convolution", "source_layer",
                     "scalar_fsum_reference"):
            if name not in row["stages"]:
                raise ValueError("Missing diagnostic processing stage")
            if not _hash_ok(row["stages"][name].get("sha256")):
                raise ValueError("Invalid processing hash")
        value = row.get("max_selected_fsum_difference")
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 2e-5:
            raise ValueError("Invalid scalar convolution error")
    if report.get("source_stage_aggregate") != _aggregate(rows):
        raise ValueError("Inconsistent convolution stage summary")


def compare_probes(a: dict, b: dict) -> dict:
    validate_probe(a)
    validate_probe(b)
    affected = {}
    for source in SOURCES:
        affected[source] = [
            name for name in a["source_stage_aggregate"][source]
            if a["source_stage_aggregate"][source][name]["sha256"]
               != b["source_stage_aggregate"][source][name]["sha256"]
        ]
    return {
        "protocol": "sonumio-dsp-convolution-cross-host-comparison-v1",
        "differing_stages_by_source": affected,
        "all_exact_stages_match": all(not value for value in affected.values()),
        "cause_is_proven": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    options = parser.parse_args()
    data = build_probe()
    reported = {
        key: value for key, value in data.items() if key != "recordings"
    } if options.summary_only else data
    encoded = json.dumps(reported, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
