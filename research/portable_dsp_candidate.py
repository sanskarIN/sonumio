"""Isolated version-portable ambient DSP hypothesis across all synthetic seeds.

RESEARCH ONLY: not used by procedural inference, historical neural training,
checkpoints, model releases, or the canonical 84-recording corpus.
This reconstructs the original DSP with a separate fast float64 convolution
and an explicitly chosen double-precision ambient sine evaluation.

Tests 21 first-party sound combinations × 9 distinct seeds = 189 waveforms
PER METHOD. Every report is fingerprints and numeric diagnostics ONLY.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from research.ambient_sine_candidate import METHODS, candidate_drone
from research.complete_dsp_candidate import (
    FRAMES, RAW_DURATION, RAW_FRAMES, MAX_EXACT_ERROR, _crop, _difference,
)
from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.fast_dsp_candidate import _fast_layer
from research.heldout_dsp_probe import HELDOUT_SEEDS
from research.latent_audio import TRAIN_SEEDS, VALIDATION_SEEDS, render_seeded_waveform
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES, _downsample_24k_to_8k
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-isolated-portable-ambient-complete-dsp-v1"
RECIPE = "research-f64-source-conv-plus-f64-ambient-sine-v1-no-training"
SPLITS = (
    ("train", tuple(TRAIN_SEEDS)),
    ("validation", tuple(VALIDATION_SEEDS)),
    ("heldout", HELDOUT_SEEDS),
)
CASES = tuple(
    (split, members, seed)
    for split, seeds in SPLITS for members in SYNTHETIC_COMBINATIONS for seed in seeds
)
if len(CASES) != 189 or len(set(CASES)) != 189:
    raise RuntimeError("DSP research seed/source grid is incomplete or overlapping")


def portable_raw(members: tuple[int, ...], seed: int, method: str) -> np.ndarray:
    """Render original source order with isolated alternative ambient sine."""
    if (type(members) is not tuple or members not in SYNTHETIC_COMBINATIONS
            or type(seed) is not int or not 0 <= seed < 2**32
            or method not in METHODS):
        raise ValueError("Invalid portable research sound/seed/method")
    prompt = " and ".join(CATEGORIES[i] for i in members)
    kinds = dsp.recognized_sound_classes(prompt)
    t = np.arange(RAW_FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    output = np.zeros(RAW_FRAMES, dtype=np.float32)
    for kind in kinds:
        output += (candidate_drone(t, method=method) if kind == "drone"
                   else _fast_layer(kind, rng, t))
    edge = min(int(0.025 * dsp.SAMPLE_RATE), RAW_FRAMES // 2)
    if edge:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        output[:edge] *= ramp
        output[-edge:] *= ramp[::-1]
    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output *= min(1.0, 0.92 / peak)
    if (output.shape != (RAW_FRAMES,) or output.dtype != np.float32
            or not np.isfinite(output).all()):
        raise ValueError("Invalid portable first-party DSP waveform")
    return output


def trace_one(split: str, members: tuple[int, ...], seed: int, method: str) -> dict:
    if (split, members, seed) not in CASES or method not in METHODS:
        raise ValueError("Unsupported portable DSP research case")
    start = perf_counter()
    new_raw = portable_raw(members, seed, method)
    render_seconds = perf_counter() - start
    prompt = " and ".join(CATEGORIES[i] for i in members)
    old_raw = dsp.generate(prompt, RAW_DURATION, seed).samples
    raw_error = _difference(new_raw, old_raw)
    new_8k = _downsample_24k_to_8k(new_raw)
    old_8k = _downsample_24k_to_8k(old_raw)
    offset, new_crop = _crop(new_8k, seed)
    legacy_offset, old_crop = _crop(old_8k, seed)
    if offset != legacy_offset:
        raise ValueError("Portable DSP changed original crop offset")
    if split != "heldout" and not np.array_equal(
        old_crop, render_seeded_waveform(members, seed)
    ):
        raise ValueError("Original training/validation source identity changed")
    crop_error = _difference(new_crop, old_crop)
    if (raw_error["max_abs"] > MAX_EXACT_ERROR
            or crop_error["max_abs"] > MAX_EXACT_ERROR):
        raise ValueError("Portable variant exceeds original DSP error threshold")
    if not math.isfinite(render_seconds) or render_seconds <= 0:
        raise ValueError("Invalid monotonic candidate runtime")
    return {
        "split": split, "sources": [CATEGORIES[i] for i in members],
        "seed": seed, "method": method, "crop_offset": offset,
        "portable_24k_sha256": digest_array(new_raw)["sha256"],
        "portable_8k_crop_sha256": digest_array(new_crop)["sha256"],
        "legacy_24k_sha256": digest_array(old_raw)["sha256"],
        "legacy_8k_crop_sha256": digest_array(old_crop)["sha256"],
        "raw_max_abs_error": raw_error["max_abs"],
        "crop_max_abs_error": crop_error["max_abs"],
        "render_24k_seconds": float(render_seconds),
    }


def _summary(rows: list[dict]) -> dict:
    out: dict = {}
    for method in METHODS:
        out[method] = {}
        for split, seeds in SPLITS:
            selected = [r for r in rows if r["method"] == method
                        and r["split"] == split]
            out[method][split] = {
                "recordings": len(selected),
                "portable_24k_sha256": _digest(b"".join(
                    bytes.fromhex(r["portable_24k_sha256"]) for r in selected)),
                "portable_8k_crop_sha256": _digest(b"".join(
                    bytes.fromhex(r["portable_8k_crop_sha256"]) for r in selected)),
                "legacy_8k_crop_sha256": _digest(b"".join(
                    bytes.fromhex(r["legacy_8k_crop_sha256"]) for r in selected)),
                "max_24k_absolute_error": max(
                    r["raw_max_abs_error"] for r in selected),
                "max_8k_crop_absolute_error": max(
                    r["crop_max_abs_error"] for r in selected),
                "median_24k_render_seconds": float(np.median([
                    r["render_24k_seconds"] for r in selected
                ])),
            }
    return out


def build_report() -> dict:
    rows = [
        trace_one(split, members, seed, method)
        for method in METHODS for split, members, seed in CASES
    ]
    report = {
        "protocol": PROTOCOL, "recipe": RECIPE,
        "source_hz": dsp.SAMPLE_RATE, "source_frames": RAW_FRAMES,
        "crop_hz": 8000, "crop_frames": FRAMES,
        "case_count_per_method": len(CASES),
        "recordings": rows, "splits": _summary(rows),
        "environment": _environment(),
        "original_renderer_modified": False,
        "original_neural_data_modified": False,
        "portable_candidate_used_for_training": False,
        "audio_files_saved": False, "model_weights_saved": False,
    }
    validate_report(report)
    return report


def validate_report(report: dict) -> None:
    if (not isinstance(report, dict)
            or report.get("protocol") != PROTOCOL
            or report.get("recipe") != RECIPE
            or report.get("source_hz") != dsp.SAMPLE_RATE
            or report.get("source_frames") != RAW_FRAMES
            or report.get("crop_hz") != 8000
            or report.get("crop_frames") != FRAMES
            or report.get("case_count_per_method") != 189
            or report.get("original_renderer_modified") is not False
            or report.get("original_neural_data_modified") is not False
            or report.get("portable_candidate_used_for_training") is not False
            or report.get("audio_files_saved") is not False
            or report.get("model_weights_saved") is not False):
        raise ValueError("Invalid portable numerical study metadata")
    rows = report.get("recordings")
    expected = {(method, split, members, seed)
                for method in METHODS for split, members, seed in CASES}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Invalid portable study case coverage")
    observed = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid portable row")
        try:
            sources = tuple(CATEGORIES.index(s) for s in row["sources"])
            identity = (row["method"], row["split"], sources, row["seed"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Invalid portable waveform identity") from None
        if (identity not in expected or identity in observed
                or type(row.get("seed")) is not int):
            raise ValueError("Duplicate or unsupported portable recording identity")
        observed.add(identity)
        if (type(row.get("crop_offset")) is not int or
                not 400 <= row["crop_offset"] <= 1600):
            raise ValueError("Invalid seed-specific crop offset")
        for name in ("portable_24k_sha256", "portable_8k_crop_sha256",
                     "legacy_24k_sha256", "legacy_8k_crop_sha256"):
            if not _hash_ok(row.get(name)):
                raise ValueError("Malformed portable provenance digest")
        for name in ("raw_max_abs_error", "crop_max_abs_error"):
            value = row.get(name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not 0 <= value <= MAX_EXACT_ERROR):
                raise ValueError("Invalid portable numerical difference")
        value = row.get("render_24k_seconds")
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value <= 0):
            raise ValueError("Invalid portable timing measurement")
    if observed != expected:
        raise ValueError("Incomplete portable waveform grid")
    # Repeatable ambient raw oscillators may be seed-invariant before crop.
    # All saved neural-data-shaped CROPS must be distinct across categories,
    # train, validation and previously unseen seeds, within each method.
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        for name in ("portable_8k_crop_sha256", "legacy_8k_crop_sha256"):
            values = [r[name] for r in subset]
            if len(set(values)) != len(values):
                raise ValueError("Duplicate exact cropped waveform across source seeds")
    if report.get("splits") != _summary(rows):
        raise ValueError("Inconsistent portable waveform aggregate fingerprints")


def compare_reports(first: dict, second: dict) -> dict:
    validate_report(first)
    validate_report(second)
    a = {(r["method"], r["split"], tuple(r["sources"]), r["seed"]): r
         for r in first["recordings"]}
    b = {(r["method"], r["split"], tuple(r["sources"]), r["seed"]): r
         for r in second["recordings"]}
    return {
        "protocol": "sonumio-portable-dsp-numerical-comparison-v1",
        "cases_per_method": len(CASES),
        "same_cpu_model": first["environment"]["cpu_model"]
                          == second["environment"]["cpu_model"],
        "numpy_versions": [
            first["environment"]["numpy"], second["environment"]["numpy"]
        ],
        "portable_24k_exact": {
            method: all(a[k]["portable_24k_sha256"]
                        == b[k]["portable_24k_sha256"]
                        for k in a if k[0] == method)
            for method in METHODS
        },
        "portable_final_crops_exact": {
            method: all(a[k]["portable_8k_crop_sha256"]
                        == b[k]["portable_8k_crop_sha256"]
                        for k in a if k[0] == method)
            for method in METHODS
        },
        "legacy_crops_exact": all(
            a[k]["legacy_8k_crop_sha256"] == b[k]["legacy_8k_crop_sha256"]
            for k in a
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare:
        first, second = [
            json.loads(p.read_text(encoding="utf-8")) for p in args.compare
        ]
        result = compare_reports(first, second)
    else:
        result = build_report()
    encoded = json.dumps(result, sort_keys=True, indent=2,
                         ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
