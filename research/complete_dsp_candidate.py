"""Isolated alternate DSP recording recipe for numerical reproducibility research.

NOT used by Sonumio inference, neural training, public API, or model releases.
The original renderer, existing data fingerprints and all checkpoints remain
unchanged. This candidate replaces only source-level float32 convolutions
with the independently tested scalar fsum reference, leaving original
layer composition and FIR resampling intact to diagnose any remaining drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.full_convolution_candidate import full_scalar_convolution
from research.latent_audio import (
    FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, render_seeded_waveform,
)
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES, _downsample_24k_to_8k
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-isolated-complete-dsp-source-conv-candidate-v1"
CANDIDATE_RECIPE = "candidate-dsp-source-scalar-fsum-v1-not-for-training"
RAW_DURATION = 1.5
RAW_FRAMES = round(dsp.SAMPLE_RATE * RAW_DURATION)
# Smoke covers all 6 single sources and these 4 two-source categories. Both
# original data splits are represented with distinct recording seeds.
SMOKE_PAIRS = ((0, 1), (0, 5), (2, 3), (4, 5))
SMOKE_COMBINATIONS = tuple((i,) for i in range(len(CATEGORIES))) + SMOKE_PAIRS
SPLITS = (("train", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS))
MAX_EXACT_ERROR = 2e-5


def _candidate_smoothed(rng: np.random.Generator, n: int, taps: int) -> np.ndarray:
    noise = rng.standard_normal(n).astype(np.float32)
    kernel = np.ones(taps, dtype=np.float32) / taps
    return full_scalar_convolution(noise, kernel, mode="same")


def _candidate_layer(kind: str, rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """Mirror original source operations and RNG draw order exactly.

    Only the source-level np.convolve operation differs, not the input,
    kernel, seed, subsequent envelope or gain. Never patch global NumPy.
    """
    n = len(t)
    if kind == "rain":
        hiss = rng.standard_normal(n).astype(np.float32) * 0.055
        impulses = np.zeros(n, dtype=np.float32)
        indices = rng.integers(0, n, size=int(110 * len(t) / dsp.SAMPLE_RATE))
        np.add.at(impulses, indices, rng.uniform(0.1, 0.6, size=len(indices)))
        kernel = np.exp(-np.arange(180) / 38).astype(np.float32)
        drops = full_scalar_convolution(impulses, kernel, mode="full")[:n]
        return hiss + 0.08 * drops
    if kind == "thunder":
        rumble = _candidate_smoothed(rng, n, 140)
        rumble /= max(float(np.max(np.abs(rumble))), 1e-6)
        center = 0.37 * t[-1] if n > 1 else 0.0
        decay = np.where(t >= center, np.exp(-(t - center) / 1.1), 0.0)
        attack = np.minimum((t - center) * 18, 1).clip(0, 1)
        return 0.50 * rumble * (attack * decay).astype(np.float32)
    if kind == "wind":
        gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * t)
        return 0.75 * _candidate_smoothed(rng, n, 60) * gust.astype(np.float32)
    if kind == "water":
        envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 0.24 * t - 0.9)) ** 2
        return _candidate_smoothed(rng, n, 12) * envelope.astype(np.float32) * 1.8
    if kind == "forest":
        return dsp._forest(rng, t)
    if kind == "drone":
        return dsp._drone(t)
    raise ValueError("Unknown first-party source")


def candidate_raw(members: tuple[int, ...], seed: int) -> np.ndarray:
    """Research-only 1.5-second 24 kHz mono source-waveform composition."""
    if (members not in SYNTHETIC_COMBINATIONS or type(seed) is not int
            or not 0 <= seed < 2**32):
        raise ValueError("Invalid bounded source combination or seed")
    label = " and ".join(CATEGORIES[i] for i in members)
    kinds = dsp.recognized_sound_classes(label)
    t = np.arange(RAW_FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    output = np.zeros(RAW_FRAMES, dtype=np.float32)
    for kind in kinds:
        output += _candidate_layer(kind, rng, t)
    edge = min(int(0.025 * dsp.SAMPLE_RATE), RAW_FRAMES // 2)
    if edge:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        output[:edge] *= ramp
        output[-edge:] *= ramp[::-1]
    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output *= min(1.0, 0.92 / peak)
    if output.shape != (RAW_FRAMES,) or not np.isfinite(output).all():
        raise ValueError("Invalid candidate 24 kHz signal")
    return output


def _difference(first: np.ndarray, second: np.ndarray) -> dict:
    if (first.shape != second.shape or first.dtype != np.float32
            or second.dtype != np.float32 or not np.isfinite(first).all()
            or not np.isfinite(second).all()):
        raise ValueError("Only equal-shaped finite float32 arrays are comparable")
    delta = first.astype(np.float64) - second.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(np.square(delta)))),
        "different_float32_values": int(np.count_nonzero(first != second)),
    }


def _crop(resampled: np.ndarray, seed: int) -> tuple[int, np.ndarray]:
    offset = int(np.random.default_rng(seed).integers(400, 1601))
    cropped = resampled[offset:offset + FRAMES]
    if cropped.shape != (FRAMES,) or cropped.dtype != np.float32:
        raise ValueError("Candidate crop is incompatible with training-data shape")
    return offset, cropped.copy()


def trace_one(members: tuple[int, ...], seed: int, split: str) -> dict:
    if (split not in ("train", "validation")
            or type(seed) is not int
            or seed not in dict(SPLITS)[split]
            or members not in SYNTHETIC_COMBINATIONS):
        raise ValueError("Invalid source/seed/split combination")
    started = perf_counter()
    candidate_24 = candidate_raw(members, seed)
    original_24 = dsp.generate(" and ".join(CATEGORIES[i] for i in members),
                               RAW_DURATION, seed).samples
    candidate_8 = _downsample_24k_to_8k(candidate_24)
    original_8 = _downsample_24k_to_8k(original_24)
    offset, candidate_crop = _crop(candidate_8, seed)
    original_offset, original_crop = _crop(original_8, seed)
    if offset != original_offset:
        raise ValueError("Candidate changed recording crop window")
    oracle = render_seeded_waveform(members, seed)
    if not np.array_equal(original_crop, oracle):
        raise ValueError("Original train/eval source identity changed")
    raw_error = _difference(candidate_24, original_24)
    resampled_error = _difference(candidate_8, original_8)
    crop_error = _difference(candidate_crop, original_crop)
    if any(metrics["max_abs"] > MAX_EXACT_ERROR for metrics in
           (raw_error, resampled_error, crop_error)):
        raise ValueError("Candidate error exceeded research-only tolerance")
    return {
        "split": split,
        "sources": [CATEGORIES[i] for i in members],
        "seed": seed,
        "crop_offset": offset,
        "legacy_raw": digest_array(original_24)["sha256"],
        "candidate_raw": digest_array(candidate_24)["sha256"],
        "legacy_8k": digest_array(original_8)["sha256"],
        "candidate_8k": digest_array(candidate_8)["sha256"],
        "legacy_cropped": digest_array(original_crop)["sha256"],
        "candidate_cropped": digest_array(candidate_crop)["sha256"],
        "raw_error": raw_error,
        "resampled_error": resampled_error,
        "cropped_error": crop_error,
        "elapsed_seconds": float(perf_counter() - started),
    }


def _identities(scope: str) -> list[tuple[str, tuple[int, ...], int]]:
    if scope not in ("smoke", "full"):
        raise ValueError("Scope must be smoke or full")
    combinations = SMOKE_COMBINATIONS if scope == "smoke" else SYNTHETIC_COMBINATIONS
    return [(split, members, seed)
            for split, seeds in SPLITS
            for members in combinations for seed in seeds]


def _summarize(rows: list[dict]) -> dict:
    summary = {}
    for split, _ in SPLITS:
        selected = [r for r in rows if r["split"] == split]
        summary[split] = {
            "recordings": len(selected),
            "candidate_8k_dataset_sha256": _digest(b"".join(
                bytes.fromhex(row["candidate_cropped"]) for row in selected
            )),
            "original_8k_dataset_sha256": _digest(b"".join(
                bytes.fromhex(row["legacy_cropped"]) for row in selected
            )),
            "candidate_raw_sha256": _digest(b"".join(
                bytes.fromhex(row["candidate_raw"]) for row in selected
            )),
            "original_raw_sha256": _digest(b"".join(
                bytes.fromhex(row["legacy_raw"]) for row in selected
            )),
            "max_24k_absolute_error": max(r["raw_error"]["max_abs"] for r in selected),
            "max_crop_absolute_error": max(
                r["cropped_error"]["max_abs"] for r in selected
            ),
        }
    return summary


def build_report(scope: str = "smoke") -> dict:
    cases = _identities(scope)
    started = perf_counter()
    rows = [trace_one(members, seed, split) for split, members, seed in cases]
    report = {
        "protocol": PROTOCOL,
        "candidate_recipe": CANDIDATE_RECIPE,
        "scope": scope,
        "source_frames": RAW_FRAMES,
        "sample_rate_24k": dsp.SAMPLE_RATE,
        "sample_rate_8k": 8000,
        "cropped_frames_8k": FRAMES,
        "recordings": rows,
        "splits": _summarize(rows),
        "environment": _environment(),
        "elapsed_seconds": float(perf_counter() - started),
        "original_source_recipe_modified": False,
        "candidate_used_for_neural_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
    }
    validate_report(report)
    return report


def validate_report(report: dict) -> None:
    if (not isinstance(report, dict)
            or report.get("protocol") != PROTOCOL
            or report.get("candidate_recipe") != CANDIDATE_RECIPE
            or report.get("source_frames") != RAW_FRAMES
            or report.get("sample_rate_24k") != dsp.SAMPLE_RATE
            or report.get("sample_rate_8k") != 8000
            or report.get("cropped_frames_8k") != FRAMES
            or report.get("original_source_recipe_modified") is not False
            or report.get("candidate_used_for_neural_training") is not False
            or report.get("audio_files_saved") is not False
            or report.get("model_weights_saved") is not False):
        raise ValueError("Invalid candidate research metadata")
    expected = _identities(report.get("scope"))
    rows = report.get("recordings")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Wrong recording coverage")
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid candidate recording")
        try:
            ids = tuple(CATEGORIES.index(category) for category in row["sources"])
            key = (row["split"], ids, row["seed"])
        except (KeyError, ValueError, TypeError):
            raise ValueError("Invalid candidate recording identity") from None
        if key in observed or key not in expected:
            raise ValueError("Duplicate or unexpected recording")
        observed.add(key)
        if type(row.get("seed")) is not int or type(row.get("crop_offset")) is not int:
            raise ValueError("Invalid source seed or crop offset")
        if not 400 <= row["crop_offset"] <= 1600:
            raise ValueError("Invalid crop range")
        for name in ("legacy_raw", "candidate_raw", "legacy_8k",
                     "candidate_8k", "legacy_cropped", "candidate_cropped"):
            if not _hash_ok(row.get(name)):
                raise ValueError("Malformed waveform provenance digest")
        for name in ("raw_error", "resampled_error", "cropped_error"):
            metrics = row.get(name)
            if not isinstance(metrics, dict):
                raise ValueError("Missing waveform error metrics")
            peak = metrics.get("max_abs")
            rms = metrics.get("rms")
            count = metrics.get("different_float32_values")
            if (type(peak) not in (int, float) or type(rms) not in (int, float)
                    or not math.isfinite(peak) or not math.isfinite(rms)
                    or not 0 <= rms <= peak <= MAX_EXACT_ERROR
                    or type(count) is not int or not 0 <= count <=
                    (FRAMES if name == "cropped_error" else
                     RAW_FRAMES if name == "raw_error" else RAW_FRAMES // 3)):
                raise ValueError("Invalid waveform difference metrics")
        value = row.get("elapsed_seconds")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid timing measurement")
    if observed != set(expected) or report.get("splits") != _summarize(rows):
        raise ValueError("Inconsistent dataset source fingerprint summary")
    total = report.get("elapsed_seconds")
    if type(total) not in (int, float) or not math.isfinite(total) or total < 0:
        raise ValueError("Invalid complete-study timing")


def compare_reports(first: dict, second: dict) -> dict:
    validate_report(first)
    validate_report(second)
    if first["scope"] != second["scope"]:
        raise ValueError("Incompatible corpus scopes")
    def index(report: dict) -> dict:
        return {(r["split"], tuple(r["sources"]), r["seed"]): r
                for r in report["recordings"]}
    a, b = index(first), index(second)
    return {
        "protocol": "sonumio-complete-dsp-candidate-comparison-v1",
        "recordings": len(a),
        "candidate_24k_exact": all(
            a[k]["candidate_raw"] == b[k]["candidate_raw"] for k in a
        ),
        "candidate_8k_exact": all(
            a[k]["candidate_8k"] == b[k]["candidate_8k"] for k in a
        ),
        "candidate_crops_exact": all(
            a[k]["candidate_cropped"] == b[k]["candidate_cropped"] for k in a
        ),
        "original_crops_exact": all(
            a[k]["legacy_cropped"] == b[k]["legacy_cropped"] for k in a
        ),
        "same_recipe_and_scope": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.scope)
    payload = json.dumps(report, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
