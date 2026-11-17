"""Paired three-way DSP benchmark: legacy, scalar fsum, and fast float64.

All timing is strictly in-memory 1.5-second 24-kHz sound generation.
Historical data and models are unaffected; no WAV or weight files saved.
Sample hashes and numerical differences are computed OUTSIDE the timers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.dsp_candidate import CASES, _output_hash, _summary
from research.complete_dsp_candidate import RAW_DURATION, RAW_FRAMES, _difference, candidate_raw
from research.data_reproducibility import _environment
from research.fast_dsp_candidate import render_fast_raw
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-paired-three-way-dsp-benchmark-v1"
VARIANTS = ("legacy", "scalar", "fast")


def _render_one(variant: str, members: tuple[int, ...], seed: int) -> tuple[float, np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError("Unknown 24-kHz benchmark variant")
    prompt = " and ".join(CATEGORIES[i] for i in members)
    start = perf_counter()
    if variant == "legacy":
        samples = dsp.generate(prompt, RAW_DURATION, seed).samples
    elif variant == "scalar":
        samples = candidate_raw(members, seed)
    else:
        samples = render_fast_raw(members, seed)
    elapsed = perf_counter() - start
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Invalid monotonic render duration")
    _output_hash(samples)  # Strict shape, finite, dtype and byte-validation
    return float(elapsed), samples


def benchmark_three_way(members: tuple[int, ...] = (0,), *, seed: int = 17,
                        repeats: int = 3, warmups: int = 1) -> dict:
    if (type(members) is not tuple or members not in SYNTHETIC_COMBINATIONS
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(repeats) is not int or not 1 <= repeats <= 9
            or type(warmups) is not int or not 0 <= warmups <= 3):
        raise ValueError("Invalid bounded three-way study configuration")
    for round_index in range(warmups):
        for variant in VARIANTS[round_index % 3:] + VARIANTS[:round_index % 3]:
            _render_one(variant, members, seed)

    measurements = {v: [] for v in VARIANTS}
    fingerprints = {v: set() for v in VARIANTS}
    comparisons = []
    for trial in range(repeats):
        order = VARIANTS[trial % 3:] + VARIANTS[:trial % 3]
        audio = {}
        for variant in order:
            elapsed, samples = _render_one(variant, members, seed)
            measurements[variant].append(elapsed)
            fingerprints[variant].add(_output_hash(samples))
            audio[variant] = samples
        comparisons.append({
            "fast_vs_scalar": _difference(audio["fast"], audio["scalar"]),
            "fast_vs_legacy": _difference(audio["fast"], audio["legacy"]),
            "scalar_vs_legacy": _difference(audio["scalar"], audio["legacy"]),
        })
    if any(len(fingerprints[v]) != 1 for v in VARIANTS):
        raise ValueError("Same-seed DSP bytes changed between same-host trials")

    summary = {v: {**_summary(measurements[v]),
                   "sha256": next(iter(fingerprints[v]))} for v in VARIANTS}
    def ratio(a: str, b: str) -> float:
        result = summary[a]["median_seconds"] / summary[b]["median_seconds"]
        if not math.isfinite(result):
            raise ValueError("Invalid benchmark latency ratio")
        return float(result)

    return {
        "protocol": PROTOCOL,
        "sources": [CATEGORIES[i] for i in members],
        "seed": seed,
        "repeats": repeats,
        "warmups_per_variant": warmups,
        "rotation": "three-variant cyclic position-balanced",
        "frames_24k": RAW_FRAMES,
        "duration_seconds": RAW_DURATION,
        "latency": summary,
        "fast_vs_legacy_median_latency_ratio": ratio("fast", "legacy"),
        "scalar_vs_legacy_median_latency_ratio": ratio("scalar", "legacy"),
        "scalar_over_fast_median_speedup": ratio("scalar", "fast"),
        "fast_equals_scalar_sha256": (
            summary["fast"]["sha256"] == summary["scalar"]["sha256"]
        ),
        "maximum_sample_error": {
            name: max(row[name]["max_abs"] for row in comparisons)
            for name in comparisons[0]
        },
        "environment": _environment(),
        "original_dsp_unchanged": True,
        "candidate_used_for_neural_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
        "interpretation": (
            "Local same-seed/host CPU timings only; in-memory 24-kHz DSP, "
            "not 8-kHz postprocessing, native-memory or neural inference. "
            "Reference agreement is per recorded seed and does not prove "
            "universal arithmetic reproducibility."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="rain")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark_three_way(
        CASES[args.case], seed=args.seed, repeats=args.repeats,
        warmups=args.warmups,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
