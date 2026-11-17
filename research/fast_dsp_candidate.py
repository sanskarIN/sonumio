"""Experimental fast 24-kHz-to-8-kHz waveform corpus, never used for training.

A separate, vectorized float64 NumPy convolution candidate is injected into
independently reconstructed DSP layers, without patching the original audio
generator, original DSP trace, scalar fsum reference, or historical checkpoints.

Results are numerical JSON diagnostics and SHA256 fingerprints only. Even an
exact match with the scalar corpus does not constitute model-quality evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from research.complete_dsp_candidate import (
    FRAMES, RAW_DURATION, RAW_FRAMES, SPLITS, SMOKE_COMBINATIONS,
    _crop, _difference, _identities, MAX_EXACT_ERROR,
)
from research.candidate_reference import REFERENCE_SPLITS
from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.latent_audio import render_seeded_waveform
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.vectorized_convolution_candidate import vectorized_float64_convolution
from research.wave_diffusion import CATEGORIES, _downsample_24k_to_8k
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-fast-vectorized-double-dsp-candidate-v1"
FAST_RECIPE = "experimental-vectorized-f64-convolution-dsp-v1-not-for-training"


def _smoothed_fast(rng: np.random.Generator, n: int, taps: int) -> np.ndarray:
    noise = rng.standard_normal(n).astype(np.float32)
    kernel = np.ones(taps, dtype=np.float32) / taps
    return vectorized_float64_convolution(noise, kernel, mode="same")


def _fast_layer(kind: str, rng: np.random.Generator, t: np.ndarray) -> np.ndarray:
    """Reconstruct original RNG draws/envelope; swap ONLY convolution."""
    n = len(t)
    if kind == "rain":
        hiss = rng.standard_normal(n).astype(np.float32) * 0.055
        impulses = np.zeros(n, dtype=np.float32)
        indices = rng.integers(0, n, size=int(110 * len(t) / dsp.SAMPLE_RATE))
        np.add.at(impulses, indices, rng.uniform(0.1, 0.6, size=len(indices)))
        kernel = np.exp(-np.arange(180) / 38).astype(np.float32)
        drops = vectorized_float64_convolution(impulses, kernel, mode="full")[:n]
        return hiss + 0.08 * drops
    if kind == "thunder":
        rumble = _smoothed_fast(rng, n, 140)
        rumble /= max(float(np.max(np.abs(rumble))), 1e-6)
        center = 0.37 * t[-1] if n > 1 else 0.0
        decay = np.where(t >= center, np.exp(-(t - center) / 1.1), 0.0)
        attack = np.minimum((t - center) * 18, 1).clip(0, 1)
        return 0.50 * rumble * (attack * decay).astype(np.float32)
    if kind == "wind":
        gust = 0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * t)
        return 0.75 * _smoothed_fast(rng, n, 60) * gust.astype(np.float32)
    if kind == "water":
        envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 0.24 * t - 0.9)) ** 2
        return _smoothed_fast(rng, n, 12) * envelope.astype(np.float32) * 1.8
    if kind == "forest":
        return dsp._forest(rng, t)
    if kind == "drone":
        return dsp._drone(t)
    raise ValueError("Unknown original sound-layer category")


def render_fast_raw(members: tuple[int, ...], seed: int) -> np.ndarray:
    """Render one separate 36k-frame float32 synthetic research waveform."""
    if (type(members) is not tuple or members not in SYNTHETIC_COMBINATIONS
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("Invalid source combination or integer seed")
    prompt = " and ".join(CATEGORIES[i] for i in members)
    recognized = dsp.recognized_sound_classes(prompt)
    t = np.arange(RAW_FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    output = np.zeros(RAW_FRAMES, dtype=np.float32)
    for kind in recognized:
        output += _fast_layer(kind, rng, t)
    edge = min(int(0.025 * dsp.SAMPLE_RATE), len(output) // 2)
    if edge:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        output[:edge] *= ramp
        output[-edge:] *= ramp[::-1]
    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output *= min(1.0, 0.92 / peak)
    if (output.shape != (RAW_FRAMES,) or output.dtype != np.float32
            or not np.isfinite(output).all()):
        raise ValueError("Invalid complete candidate output")
    return output


def _case(members: tuple[int, ...], seed: int, split: str) -> dict:
    if (split not in dict(SPLITS) or type(seed) is not int
            or seed not in dict(SPLITS)[split]
            or type(members) is not tuple
            or members not in SYNTHETIC_COMBINATIONS):
        raise ValueError("Invalid candidate source/split/seed")
    start_fast = perf_counter()
    fast = render_fast_raw(members, seed)
    fast_seconds = perf_counter() - start_fast
    # Independent original render is a STRICT oracle for source semantics
    # and crop identity. Never compare candidate against itself.
    prompt = " and ".join(CATEGORIES[i] for i in members)
    original = dsp.generate(prompt, RAW_DURATION, seed).samples
    fast_8 = _downsample_24k_to_8k(fast)
    original_8 = _downsample_24k_to_8k(original)
    offset, fast_crop = _crop(fast_8, seed)
    original_offset, original_crop = _crop(original_8, seed)
    if offset != original_offset or not np.array_equal(
            original_crop, render_seeded_waveform(members, seed)):
        raise ValueError("Original train/validation recipe has drifted")
    raw_error = _difference(fast, original)
    crop_error = _difference(fast_crop, original_crop)
    if (raw_error["max_abs"] > MAX_EXACT_ERROR
            or crop_error["max_abs"] > MAX_EXACT_ERROR):
        raise ValueError("Fast alternative exceeds numerical error bound")
    if not math.isfinite(fast_seconds) or fast_seconds <= 0:
        raise ValueError("Invalid fast candidate runtime")
    return {
        "split": split,
        "sources": [CATEGORIES[i] for i in members],
        "seed": seed,
        "offset": offset,
        "fast_raw_sha256": digest_array(fast)["sha256"],
        "fast_8k_sha256": digest_array(fast_8)["sha256"],
        "fast_crop_sha256": digest_array(fast_crop)["sha256"],
        "original_raw_sha256": digest_array(original)["sha256"],
        "original_crop_sha256": digest_array(original_crop)["sha256"],
        "raw_error": raw_error,
        "crop_error": crop_error,
        "fast_24k_seconds": float(fast_seconds),
    }


def _aggregate(rows: list[dict]) -> dict:
    result = {}
    for split, _ in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        result[split] = {
            "recordings": len(subset),
            "fast_raw_sha256": _digest(b"".join(
                bytes.fromhex(r["fast_raw_sha256"]) for r in subset)),
            "fast_8k_crop_sha256": _digest(b"".join(
                bytes.fromhex(r["fast_crop_sha256"]) for r in subset)),
            "original_raw_sha256": _digest(b"".join(
                bytes.fromhex(r["original_raw_sha256"]) for r in subset)),
            "original_8k_crop_sha256": _digest(b"".join(
                bytes.fromhex(r["original_crop_sha256"]) for r in subset)),
            "max_24k_abs_error": max(x["raw_error"]["max_abs"] for x in subset),
            "max_8k_crop_abs_error": max(x["crop_error"]["max_abs"] for x in subset),
            "median_fast_24k_seconds": float(np.median([
                x["fast_24k_seconds"] for x in subset
            ])),
        }
    return result


def matches_scalar_reference(report: dict) -> bool | None:
    validate_report(report)
    if report["scope"] != "full":
        return None
    return all(
        report["splits"][split]["recordings"] == target["recordings"]
        and report["splits"][split]["fast_raw_sha256"] == target["candidate_raw_sha256"]
        and report["splits"][split]["fast_8k_crop_sha256"]
            == target["candidate_8k_dataset_sha256"]
        for split, target in REFERENCE_SPLITS.items()
    )


def build_report(scope: str = "smoke") -> dict:
    ids = _identities(scope)
    rows = [_case(members, seed, split) for split, members, seed in ids]
    report = {
        "protocol": PROTOCOL,
        "fast_recipe": FAST_RECIPE,
        "scope": scope,
        "source_frames": RAW_FRAMES,
        "crop_frames": FRAMES,
        "source_hz": dsp.SAMPLE_RATE,
        "crop_hz": 8000,
        "recordings": rows,
        "splits": _aggregate(rows),
        "environment": _environment(),
        "original_generator_unchanged": True,
        "candidate_used_for_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
    }
    validate_report(report)
    return report


def validate_report(report: dict) -> None:
    if (not isinstance(report, dict)
            or report.get("protocol") != PROTOCOL
            or report.get("fast_recipe") != FAST_RECIPE
            or report.get("source_frames") != RAW_FRAMES
            or report.get("crop_frames") != FRAMES
            or report.get("source_hz") != dsp.SAMPLE_RATE
            or report.get("crop_hz") != 8000
            or report.get("original_generator_unchanged") is not True
            or report.get("candidate_used_for_training") is not False
            or report.get("audio_files_saved") is not False
            or report.get("model_weights_saved") is not False):
        raise ValueError("Invalid fast candidate metadata")
    expected = set(_identities(report.get("scope")))
    rows = report.get("recordings")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Invalid fast recording grid")
    observed = set()
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("Invalid fast recording row")
        try:
            members = tuple(CATEGORIES.index(s) for s in r["sources"])
            identity = (r["split"], members, r["seed"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Malformed fast recording identity") from None
        if (identity in observed or identity not in expected
                or type(r["seed"]) is not int):
            raise ValueError("Invalid or duplicated fast recording identity")
        observed.add(identity)
        if type(r.get("offset")) is not int or not 400 <= r["offset"] <= 1600:
            raise ValueError("Invalid fast crop offset")
        for field in ("fast_raw_sha256", "fast_8k_sha256", "fast_crop_sha256",
                      "original_raw_sha256", "original_crop_sha256"):
            if not _hash_ok(r.get(field)):
                raise ValueError("Malformed fast output SHA")
        for field, limit in (("raw_error", RAW_FRAMES), ("crop_error", FRAMES)):
            metric = r.get(field)
            if not isinstance(metric, dict):
                raise ValueError("Missing fast error metric")
            peak, rms, count = (
                metric.get("max_abs"), metric.get("rms"),
                metric.get("different_float32_values"),
            )
            if (type(peak) not in (int, float) or type(rms) not in (int, float)
                    or not math.isfinite(peak) or not math.isfinite(rms)
                    or not 0 <= rms <= peak <= MAX_EXACT_ERROR
                    or type(count) is not int or not 0 <= count <= limit):
                raise ValueError("Invalid fast numerical metric")
        seconds = r.get("fast_24k_seconds")
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or seconds <= 0):
            raise ValueError("Invalid runtime metric")
    if observed != expected:
        raise ValueError("Incomplete fast source grid")
    for field in ("fast_crop_sha256", "original_crop_sha256"):
        across = {}
        for split, _ in SPLITS:
            hashes = [r[field] for r in rows if r["split"] == split]
            if len(hashes) != len(set(hashes)):
                raise ValueError("Duplicate waveform SHA in split " + split)
            across[split] = set(hashes)
        if across["train"] & across["validation"]:
            raise ValueError("Train/validation waveform SHA overlap")
    if report.get("splits") != _aggregate(rows):
        raise ValueError("Invalid fast aggregate hashes")


def compare_reports(first: dict, second: dict) -> dict:
    validate_report(first)
    validate_report(second)
    if first["scope"] != second["scope"]:
        raise ValueError("Incompatible fast experiment scopes")
    a = {(r["split"], tuple(r["sources"]), r["seed"]): r
         for r in first["recordings"]}
    b = {(r["split"], tuple(r["sources"]), r["seed"]): r
         for r in second["recordings"]}
    return {
        "protocol": "sonumio-fast-dsp-candidate-comparison-v1",
        "recordings": len(a),
        "identical_candidate_raw_sha": all(
            a[k]["fast_raw_sha256"] == b[k]["fast_raw_sha256"] for k in a
        ),
        "identical_candidate_cropped_sha": all(
            a[k]["fast_crop_sha256"] == b[k]["fast_crop_sha256"] for k in a
        ),
        "identical_original_raw_sha": all(
            a[k]["original_raw_sha256"] == b[k]["original_raw_sha256"] for k in a
        ),
        "matches_scalar_reference": (
            matches_scalar_reference(first)
            and matches_scalar_reference(second)
            if first["scope"] == "full" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.scope)
    payload = json.dumps(report, sort_keys=True, indent=2,
                         allow_nan=False, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
