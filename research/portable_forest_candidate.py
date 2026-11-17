"""Second-generation, FOREST-aware candidate for DSP byte-portability research.

No original procedural layer, neural training data, checkpoint, or live service
imports this module. It changes TWO isolated numerical operations:
 - selected float64 source smoothing + ambient sine (existing portable study)
 - original forest chirp phase's float32 np.sin -> double-precision sine with
   the ORIGINAL phase computation, original random draws and original windows.

Both NumPy float64 and Python math.sin options are measured, not adopted.
No audio files, datasets, or neural weights are created or published.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from research.ambient_sine_candidate import METHODS, _sin_float32_phase, candidate_drone
from research.complete_dsp_candidate import (
    FRAMES, RAW_DURATION, RAW_FRAMES, MAX_EXACT_ERROR, _crop, _difference,
)
from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.fast_dsp_candidate import _fast_layer
from research.latent_audio import render_seeded_waveform
from research.portable_dsp_candidate import CASES, SPLITS, portable_raw
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES, _downsample_24k_to_8k
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-forest-aware-portable-dsp-research-v1"
RECIPE = "research-f64-conv-and-double-ambient-plus-double-forest-sine-no-training"


def candidate_forest(rng: np.random.Generator, t: np.ndarray,
                     method: str = "numpy_float64") -> np.ndarray:
    """Maintain the exact first-party forest RNG/phase/window with alternate sine."""
    if (not isinstance(rng, np.random.Generator)
            or not isinstance(t, np.ndarray) or t.dtype != np.float32
            or t.ndim != 1 or not 1 <= t.size <= 50000
            or not np.isfinite(t).all() or method not in METHODS):
        raise ValueError("Invalid original forest candidate input or method")
    n, sr = t.size, dsp.SAMPLE_RATE
    chirps = np.zeros(n, dtype=np.float32)
    for position in np.arange(0.35, t[-1], 0.7):
        start = int(position * sr)
        length = min(int(0.19 * sr), n - start)
        if length <= 0:
            continue
        local_t = np.arange(length, dtype=np.float32) / sr
        freq = rng.uniform(900, 1900)
        phase = 2 * np.pi * (freq * local_t + 1800 * local_t ** 2)
        if phase.dtype != np.float32:
            raise ValueError("Unexpected original phase dtype")
        window = np.sin(np.pi * np.linspace(0, 1, length)) ** 2
        sine = _sin_float32_phase(phase, method=method)
        chirps[start:start + length] += (0.12 * sine * window).astype(np.float32)
    if chirps.shape != t.shape or not np.isfinite(chirps).all():
        raise ValueError("Invalid forest candidate layer")
    return chirps


def forest_aware_raw(members: tuple[int, ...], seed: int,
                     method: str = "numpy_float64") -> np.ndarray:
    if (type(members) is not tuple or members not in SYNTHETIC_COMBINATIONS
            or type(seed) is not int or not 0 <= seed < 2**32
            or method not in METHODS):
        raise ValueError("Invalid source/seed for forest-aware candidate")
    prompt = " and ".join(CATEGORIES[i] for i in members)
    kinds = dsp.recognized_sound_classes(prompt)
    rng = np.random.default_rng(seed)
    t = np.arange(RAW_FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    output = np.zeros(RAW_FRAMES, dtype=np.float32)
    for kind in kinds:
        if kind == "forest":
            layer = candidate_forest(rng, t, method=method)
        elif kind == "drone":
            layer = candidate_drone(t, method=method)
        else:
            layer = _fast_layer(kind, rng, t)
        output += layer
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
        raise ValueError("Invalid forest-aware 24k output")
    return output


def _case(split: str, members: tuple[int, ...],
          seed: int, method: str) -> dict:
    if (split, members, seed) not in CASES or method not in METHODS:
        raise ValueError("Unsupported forest-aware research identity")
    started = perf_counter()
    candidate = forest_aware_raw(members, seed, method)
    seconds = perf_counter() - started
    original = dsp.generate(" and ".join(CATEGORIES[i] for i in members),
                            RAW_DURATION, seed).samples
    baseline = portable_raw(members, seed, method)
    candidate_8 = _downsample_24k_to_8k(candidate)
    original_8 = _downsample_24k_to_8k(original)
    offset, crop = _crop(candidate_8, seed)
    old_offset, old_crop = _crop(original_8, seed)
    if offset != old_offset:
        raise ValueError("Forest-aware candidate changed original crop offset")
    if split in ("train", "validation") and not np.array_equal(
        old_crop, render_seeded_waveform(members, seed)
    ):
        raise ValueError("Historical neural synthetic training oracle changed")
    numerical = _difference(candidate, original)
    crop_error = _difference(crop, old_crop)
    if numerical["max_abs"] > MAX_EXACT_ERROR or crop_error["max_abs"] > MAX_EXACT_ERROR:
        raise ValueError("Forest-aware candidate exceeds original sample tolerance")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid forest-aware candidate runtime")
    return {
        "split": split,
        "sources": [CATEGORIES[i] for i in members],
        "seed": seed, "method": method,
        "original_crop_offset": offset,
        "candidate_24k_sha256": digest_array(candidate)["sha256"],
        "candidate_8k_crop_sha256": digest_array(crop)["sha256"],
        "legacy_8k_crop_sha256": digest_array(old_crop)["sha256"],
        "previous_portable_24k_sha256": digest_array(baseline)["sha256"],
        "maximum_24k_absolute_error": numerical["max_abs"],
        "maximum_8k_crop_absolute_error": crop_error["max_abs"],
        "render_seconds_24k": float(seconds),
    }


def _summary(rows: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        result[method] = {}
        for split, _ in SPLITS:
            subset = [r for r in rows if r["method"] == method
                      and r["split"] == split]
            result[method][split] = {
                "recordings": len(subset),
                "candidate_24k_sha256": _digest(b"".join(
                    bytes.fromhex(r["candidate_24k_sha256"]) for r in subset)),
                "candidate_8k_crop_sha256": _digest(b"".join(
                    bytes.fromhex(r["candidate_8k_crop_sha256"]) for r in subset)),
                "previous_portable_24k_sha256": _digest(b"".join(
                    bytes.fromhex(r["previous_portable_24k_sha256"]) for r in subset)),
                "max_24k_abs_error": max(
                    r["maximum_24k_absolute_error"] for r in subset),
                "max_8k_crop_abs_error": max(
                    r["maximum_8k_crop_absolute_error"] for r in subset),
            }
    return result


def build_report() -> dict:
    rows = [_case(split, members, seed, method)
            for method in METHODS for split, members, seed in CASES]
    value = {
        "protocol": PROTOCOL,
        "recipe": RECIPE,
        "source_hz": dsp.SAMPLE_RATE, "source_frames": RAW_FRAMES,
        "crop_hz": 8000, "crop_frames": FRAMES,
        "case_count_per_method": len(CASES),
        "recordings": rows, "splits": _summary(rows),
        "environment": _environment(),
        "additional_seed_3_is_text_audio_training_seed": True,
        "original_generator_modified": False,
        "original_neural_data_modified": False,
        "candidate_used_for_neural_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
    }
    validate_report(value)
    return value


def validate_report(report: dict) -> None:
    if (not isinstance(report, dict) or report.get("protocol") != PROTOCOL
            or report.get("recipe") != RECIPE
            or report.get("source_hz") != dsp.SAMPLE_RATE
            or report.get("source_frames") != RAW_FRAMES
            or report.get("crop_hz") != 8000
            or report.get("crop_frames") != FRAMES
            or report.get("case_count_per_method") != 189
            or report.get("additional_seed_3_is_text_audio_training_seed") is not True
            or report.get("original_generator_modified") is not False
            or report.get("original_neural_data_modified") is not False
            or report.get("candidate_used_for_neural_training") is not False
            or report.get("audio_files_saved") is not False
            or report.get("model_weights_saved") is not False):
        raise ValueError("Invalid forest-aware DSP research metadata")
    rows = report.get("recordings")
    expected = {(method, split, members, seed)
                for method in METHODS for split, members, seed in CASES}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Incomplete forest-aware numerical grid")
    observed = set()
    for row in rows:
        try:
            if not isinstance(row, dict):
                raise ValueError
            identity = (row["method"], row["split"],
                        tuple(CATEGORIES.index(s) for s in row["sources"]),
                        row["seed"])
            if identity in observed or identity not in expected or type(row["seed"]) is not int:
                raise ValueError
            observed.add(identity)
            if type(row["original_crop_offset"]) is not int or not 400 <= row["original_crop_offset"] <= 1600:
                raise ValueError
            if any(not _hash_ok(row.get(name)) for name in (
                "candidate_24k_sha256", "candidate_8k_crop_sha256",
                "legacy_8k_crop_sha256", "previous_portable_24k_sha256"
            )):
                raise ValueError
            for name in ("maximum_24k_absolute_error",
                         "maximum_8k_crop_absolute_error"):
                value = row.get(name)
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or not 0 <= value <= MAX_EXACT_ERROR):
                    raise ValueError
            value = row.get("render_seconds_24k")
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise ValueError("Invalid forest-aware recording or numerical checksum") from None
    if observed != expected:
        raise ValueError("Incomplete first-party source/seed combinations")
    for method in METHODS:
        crop_hashes = [row["candidate_8k_crop_sha256"] for row in rows
                       if row["method"] == method]
        if len(set(crop_hashes)) != 189:
            raise ValueError("Duplicate forest-aware cropped waveform")
    if report.get("splits") != _summary(rows):
        raise ValueError("Inconsistent forest-aware SHA256 aggregate")


def compare_reports(first: dict, second: dict) -> dict:
    validate_report(first)
    validate_report(second)
    a = {(r["method"], r["split"], tuple(r["sources"]), r["seed"]): r
         for r in first["recordings"]}
    b = {(r["method"], r["split"], tuple(r["sources"]), r["seed"]): r
         for r in second["recordings"]}
    results = {}
    for method in METHODS:
        results[method] = {}
        for split, _ in SPLITS:
            keys = [key for key in a if key[0] == method and key[1] == split]
            raw = [key for key in keys if a[key]["candidate_24k_sha256"]
                   != b[key]["candidate_24k_sha256"]]
            crop = [key for key in keys if a[key]["candidate_8k_crop_sha256"]
                    != b[key]["candidate_8k_crop_sha256"]]
            results[method][split] = {
                "cases": len(keys),
                "different_24k": len(raw),
                "different_8k_crop": len(crop),
                "different_sources": [
                    {"seed": k[3], "sources": list(k[2])}
                    for k in keys if k in set(raw) | set(crop)
                ][:20],
            }
    return {
        "protocol": "sonumio-forest-aware-version-comparison-v1",
        "cases_per_method": len(CASES),
        "same_cpu_model": (first["environment"]["cpu_model"]
                           == second["environment"]["cpu_model"]),
        "numpy_versions": [first["environment"]["numpy"],
                           second["environment"]["numpy"]],
        "by_method_split": results,
        "all_candidate_bytes_match": all(
            r["different_24k"] == 0 and r["different_8k_crop"] == 0
            for x in results.values() for r in x.values()
        ),
        "previous_candidate_hashes_match": all(
            a[k]["previous_portable_24k_sha256"] ==
            b[k]["previous_portable_24k_sha256"] for k in a
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare:
        first, second = [json.loads(p.read_text(encoding="utf-8")) for p in args.compare]
        value = compare_reports(first, second)
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
