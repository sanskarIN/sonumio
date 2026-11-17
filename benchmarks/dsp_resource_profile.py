"""Isolated-process DSP memory and latency research (no neural generation).

Each variant runs in its own bounded subprocess. Timing measures ONLY the
1.5-second 24-kHz DSP render, after lazy imports/warmups. Peak RSS measures
the *whole process* including Python, NumPy, Torch research imports where
applicable, not just the renderer's native allocations. The result must not
be called a direct neural inference benchmark, portable byte proof, or
production memory requirement. No wave files or model weights are saved.

Example:
  python -m benchmarks.dsp_resource_profile --case thunder-forest --seed 41
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROTOCOL = "sonumio-dsp-isolated-process-resource-research-v1"
SAMPLE_RATE = 24000
DURATION_SECONDS = 1.5
RAW_FRAMES = 36000
SOURCES = ("rain", "thunder", "wind", "forest", "ocean", "ambient")
CASES = {
    "rain": (0,),
    "forest": (3,),
    "ambient": (5,),
    "rain-thunder": (0, 1),
    "thunder-forest": (1, 3),
    "wind-forest": (2, 3),
    "rain-ambient": (0, 5),
    "forest-ambient": (3, 5),
}
VARIANTS = (
    "original",
    "fast_float64",
    "portable_ambient_float64",
    "forest_aware_float64",
    "forest_aware_scalar",
)


def _validate_controls(case: str, seed: int,
                       repeats: int, warmups: int) -> tuple[int, ...]:
    if (not isinstance(case, str) or case not in CASES
            or type(seed) is not int or not 0 <= seed < 2**32
            or type(repeats) is not int or not 1 <= repeats <= 9
            or type(warmups) is not int or not 0 <= warmups <= 3):
        raise ValueError("Invalid bounded DSP resource research controls")
    return CASES[case]


def _renderer(variant: str):
    if variant == "original":
        from sonumio.engines.procedural import generate
        return lambda members, seed: generate(
            " and ".join(SOURCES[x] for x in members), DURATION_SECONDS, seed
        ).samples
    if variant == "fast_float64":
        from research.fast_dsp_candidate import render_fast_raw
        return lambda members, seed: render_fast_raw(members, seed)
    if variant == "portable_ambient_float64":
        from research.portable_dsp_candidate import portable_raw
        return lambda members, seed: portable_raw(members, seed, "numpy_float64")
    if variant == "forest_aware_float64":
        from research.portable_forest_candidate import forest_aware_raw
        return lambda members, seed: forest_aware_raw(members, seed, "numpy_float64")
    if variant == "forest_aware_scalar":
        from research.portable_forest_candidate import forest_aware_raw
        return lambda members, seed: forest_aware_raw(members, seed, "python_math_sin")
    raise ValueError("Unknown isolated DSP variant")


def _rss_highwater_kib() -> int | None:
    """Linux/macOS RUSAGE_SELF peak RSS; Windows reports unavailable, not zero."""
    if sys.platform not in ("linux", "darwin"):
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    # Linux reports KiB; macOS reports bytes.
    value = int(value / 1024) if sys.platform == "darwin" else int(value)
    return value if value > 0 else None


def _sample_audio(waveform: np.ndarray) -> str:
    if (not isinstance(waveform, np.ndarray)
            or waveform.dtype != np.float32
            or waveform.shape != (RAW_FRAMES,)
            or not np.isfinite(waveform).all()):
        raise ValueError("Expected finite 36000-frame float32 DSP waveform")
    return hashlib.sha256(np.asarray(waveform, dtype="<f4").tobytes()).hexdigest()


def _child_measure(variant: str, case: str, seed: int,
                   repeats: int, warmups: int) -> dict:
    members = _validate_controls(case, seed, repeats, warmups)
    if variant not in VARIANTS:
        raise ValueError("Unknown isolated DSP variant")
    render = _renderer(variant)
    highwater_before = _rss_highwater_kib()
    for _ in range(warmups):
        _sample_audio(render(members, seed))
    times = []
    digests = set()
    for _ in range(repeats):
        started = perf_counter()
        waveform = render(members, seed)
        elapsed = perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid monotonic DSP time")
        times.append(float(elapsed))
        digests.add(_sample_audio(waveform))
    if len(digests) != 1:
        raise ValueError("Identical seeded render changed within one process")
    highwater_after = _rss_highwater_kib()
    return {
        "variant": variant,
        "case": case,
        "seed": seed,
        "repeats": repeats,
        "warmups": warmups,
        "wave_sha256": next(iter(digests)),
        "median_render_seconds": float(statistics.median(times)),
        "minimum_render_seconds": min(times),
        "maximum_render_seconds": max(times),
        "p95_seconds_nearest_rank": sorted(times)[math.ceil(.95 * repeats) - 1],
        "total_process_peak_rss_kib": highwater_after,
        "peak_rss_after_imports_kib": highwater_before,
        "incremental_highwater_during_work_kib": (
            max(0, highwater_after - highwater_before)
            if highwater_after is not None and highwater_before is not None
            else None
        ),
        "memory_platform_support": (
            "rusage-linux-kib" if sys.platform == "linux"
            else "rusage-darwin-bytes" if sys.platform == "darwin"
            else "unavailable"
        ),
        "cpu_model": platform.processor()[:128],
        "numpy_version": np.__version__,
        "audio_files_saved": False,
        "model_weights_saved": False,
        "candidate_used_for_neural_training": False,
    }


def _run_child(variant: str, case: str, seed: int,
               repeats: int, warmups: int) -> dict:
    # Fixed executable, module, and allowlisted variant; shell=False.
    process = subprocess.run(
        [sys.executable, "-m", "benchmarks.dsp_resource_profile",
         "--child", "--variant", variant, "--case", case, "--seed", str(seed),
         "--repeats", str(repeats), "--warmups", str(warmups)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    result = json.loads(process.stdout)
    if (not isinstance(result, dict)
            or result.get("variant") != variant or result.get("case") != case
            or result.get("seed") != seed or result.get("repeats") != repeats
            or result.get("warmups") != warmups
            or result.get("audio_files_saved") is not False
            or result.get("model_weights_saved") is not False
            or result.get("candidate_used_for_neural_training") is not False
            or not isinstance(result.get("wave_sha256"), str)
            or len(result["wave_sha256"]) != 64):
        raise ValueError("Invalid isolated subprocess DSP measurement")
    return result


def profile_case(case: str = "thunder-forest", *, seed: int = 41,
                 repeats: int = 3, warmups: int = 1) -> dict:
    _validate_controls(case, seed, repeats, warmups)
    measurements = {
        variant: _run_child(variant, case, seed, repeats, warmups)
        for variant in VARIANTS
    }
    return {
        "protocol": PROTOCOL,
        "case": case, "source_categories": [
            SOURCES[x] for x in CASES[case]
        ],
        "seed": seed,
        "repeats": repeats,
        "warmups": warmups,
        "source_hz": SAMPLE_RATE,
        "source_frames": RAW_FRAMES,
        "variants": measurements,
        "original_generator_modified": False,
        "original_data_modified": False,
        "candidate_used_for_neural_training": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
        "limitation": (
            "Each method is measured in its own Python subprocess. Total peak "
            "RSS includes imports and interpreter memory; high-water increase "
            "is not peak DSP allocation; thread scheduling and caches differ. "
            "This is NOT neural inference, model quality, GPU, or throughput."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="thunder-forest")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--variant", choices=VARIANTS, default="original")
    parser.add_argument("--child", action="store_true",
                        help="Internal: one variant in this process")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.child:
        result = _child_measure(args.variant, args.case, args.seed,
                                args.repeats, args.warmups)
    else:
        result = profile_case(args.case, seed=args.seed,
                              repeats=args.repeats, warmups=args.warmups)
    encoded = json.dumps(result, sort_keys=True, indent=2,
                         allow_nan=False, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
