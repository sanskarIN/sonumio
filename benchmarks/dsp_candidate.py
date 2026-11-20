"""Bounded original-vs-candidate DSP performance research.

Not a neural model benchmark. Measures only in-memory 1.5-second source
rendering, excluding validation, waveform comparison, checkpoint I/O and WAV
encoding from timed intervals. No audio files, tensors, or weights are saved.

Examples:
  python -m benchmarks.dsp_candidate --case rain --repeats 3 --warmups 1
  python -m benchmarks.dsp_candidate --case rain-thunder --repeats 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from research.complete_dsp_candidate import (
    RAW_DURATION, RAW_FRAMES, _difference, candidate_raw,
)
from research.data_reproducibility import _environment
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-isolated-dsp-render-benchmark-v1"
CASES = {
    "rain": (0,),
    "thunder": (1,),
    "wind": (2,),
    "forest": (3,),
    "ocean": (4,),
    "ambient": (5,),
    "rain-thunder": (0, 1),
    "rain-ambient": (0, 5),
    "wind-forest": (2, 3),
    "ocean-ambient": (4, 5),
}


def _output_hash(waveform: np.ndarray) -> str:
    if (not isinstance(waveform, np.ndarray)
            or waveform.dtype != np.float32
            or waveform.shape != (RAW_FRAMES,)
            or not np.isfinite(waveform).all()):
        raise ValueError("Benchmark requires finite, fixed-length float32 output")
    return hashlib.sha256(np.asarray(waveform, dtype="<f4").tobytes()).hexdigest()


def _one_render(variant: str, members: tuple[int, ...], seed: int
                ) -> tuple[float, np.ndarray]:
    label = " and ".join(CATEGORIES[i] for i in members)
    # The timer excludes output checks, SHA, error metrics, and JSON writing.
    started = perf_counter()
    if variant == "legacy":
        samples = dsp.generate(label, RAW_DURATION, seed).samples
    elif variant == "candidate":
        samples = candidate_raw(members, seed)
    else:
        raise ValueError("Unknown rendering variant")
    elapsed = perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Invalid monotonic duration")
    _output_hash(samples)
    return elapsed, samples


def _summary(times: list[float]) -> dict:
    if not times or any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError("Non-finite or invalid benchmark timings")
    sorted_times = sorted(times)
    return {
        "median_seconds": float(statistics.median(sorted_times)),
        "minimum_seconds": float(sorted_times[0]),
        "maximum_seconds": float(sorted_times[-1]),
        "p95_seconds_nearest_rank": float(sorted_times[math.ceil(.95 * len(times)) - 1]),
        "median_real_time_factor": float(statistics.median(times) / RAW_DURATION),
    }


def benchmark_case(members: tuple[int, ...] = (0, 1), *,
                   seed: int = 17, repeats: int = 3, warmups: int = 1) -> dict:
    """Compare fixed DSP inputs with alternating trial order.

    Medians are local observations, not claims about all hardware. The
    experiment deliberately includes no neural inference or audio scoring.
    """
    if (type(members) is not tuple or members not in SYNTHETIC_COMBINATIONS
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(repeats) is not int or not 1 <= repeats <= 8
            or type(warmups) is not int or not 0 <= warmups <= 2):
        raise ValueError("Invalid bounded DSP benchmark configuration")

    for _ in range(warmups):
        for variant in ("legacy", "candidate"):
            _one_render(variant, members, seed)

    elapsed = {"legacy": [], "candidate": []}
    hashes = {"legacy": set(), "candidate": set()}
    errors = []
    for i in range(repeats):
        order = ("legacy", "candidate") if i % 2 == 0 else ("candidate", "legacy")
        round_output = {}
        for variant in order:
            seconds, samples = _one_render(variant, members, seed)
            elapsed[variant].append(seconds)
            hashes[variant].add(_output_hash(samples))
            round_output[variant] = samples
        errors.append(_difference(round_output["candidate"], round_output["legacy"]))
    # Repeated renderings with identical seeds must be bytewise identical on
    # the SAME host; cross-host hashes are evaluated in separate CI work.
    if any(len(hashes[variant]) != 1 for variant in hashes):
        raise ValueError("Same-host DSP output changed between benchmark repetitions")

    old = _summary(elapsed["legacy"])
    new = _summary(elapsed["candidate"])
    return {
        "protocol": PROTOCOL,
        "source_categories": [CATEGORIES[i] for i in members],
        "seed": seed,
        "repeats": repeats,
        "warmups_per_variant": warmups,
        "raw_seconds": RAW_DURATION,
        "raw_frames": RAW_FRAMES,
        "order_alternated": True,
        "legacy": {**old, "exact_sha256": next(iter(hashes["legacy"]))},
        "candidate": {**new, "exact_sha256": next(iter(hashes["candidate"]))},
        "candidate_to_legacy_median_latency_ratio": (
            new["median_seconds"] / old["median_seconds"]
        ),
        "max_absolute_sample_difference": max(x["max_abs"] for x in errors),
        "max_different_float32_values": max(
            x["different_float32_values"] for x in errors
        ),
        "environment": _environment(),
        "audio_files_saved": False,
        "model_weights_saved": False,
        "candidate_used_for_training": False,
        "caveat": (
            "Unpinned host timings, not reproducibility/quality proof. "
            "Native NumPy allocation memory and neural-training costs were "
            "not measured; latency excludes SHA/error and WAV encode."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="rain-thunder")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, help="Optional local JSON metrics only")
    args = parser.parse_args()
    report = benchmark_case(
        CASES[args.case], seed=args.seed,
        repeats=args.repeats, warmups=args.warmups,
    )
    encoded = json.dumps(report, sort_keys=True, indent=2,
                         allow_nan=False, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
