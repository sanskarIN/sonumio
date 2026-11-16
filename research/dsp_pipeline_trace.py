"""Trace the exact first-party Sonumio DSP synthesis pipeline without changing it.

The aim is to isolate which intermediate stage first differs across CPU
environments. This diagnostic calls the original procedural DSP functions,
reconstructs their existing stages, checks the final waveform against the
unmodified production DSP output, and stores only SHA256/metadata. NOT a
new synthetic data recipe, trained model, or audio-quality experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from research.data_reproducibility import (
    _digest, _environment, _hash_ok, QUANTIZATION_STEPS,
)
from research.latent_audio import (
    FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, render_seeded_waveform,
)
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-first-party-dsp-pipeline-trace-v1"
STAGES = (
    "time_grid",
    "mix_pre_fade",
    "mix_post_fade",
    "normalized_24k",
    "fir_kernel",
    "fir_filtered_24k",
    "downsampled_8k",
    "cropped_8k",
)
LAYERS = {
    "rain": dsp._rain,
    "thunder": dsp._thunder,
    "wind": dsp._wind,
    "forest": dsp._forest,
    "water": dsp._water,
}
KEY_PREFIXES = ("layer_", "mix_after_")


def digest_array(array: np.ndarray) -> dict:
    """Exact native-dtype SHA + diagnostic buckets, with no signal mutation."""
    if (not isinstance(array, np.ndarray) or array.ndim != 1
            or array.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(array).all() or len(array) < 2):
        raise ValueError("Expected finite 1-D float32/float64 intermediate")
    dtype = "<f4" if array.dtype == np.float32 else "<f8"
    data = np.asarray(array, dtype=dtype, order="C")
    result = {
        "dtype": "float32" if dtype == "<f4" else "float64",
        "frames": int(len(data)),
        "sha256": _digest(data.tobytes()),
    }
    # These hashes summarize amplitude buckets, NOT sample error bounds.
    values = data.astype(np.float64)
    for step in QUANTIZATION_STEPS:
        quant = np.rint(values / step)
        if float(np.max(np.abs(quant))) >= 2**30:
            raise ValueError("Intermediate exceeds bounded diagnostic range")
        result[f"bucket_{step:g}_sha256"] = _digest(quant.astype("<i4").tobytes())
    return result


def trace_one(members: tuple[int, ...], seed: int) -> dict:
    """Trace raw layer, mixing, fade, normalization, FIR and cropping stages."""
    if (members not in SYNTHETIC_COMBINATIONS
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("Invalid source combination or source seed")
    label = " and ".join(CATEGORIES[m] for m in members)
    recognized = dsp.recognized_sound_classes(label)
    length = round(dsp.SAMPLE_RATE * 1.5)
    t = np.arange(length, dtype=np.float32) / dsp.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    stages: dict[str, dict] = {"time_grid": digest_array(t)}
    output = np.zeros(length, dtype=np.float32)
    for kind in recognized:
        layer = dsp._drone(t) if kind == "drone" else LAYERS[kind](rng, t)
        stages["layer_" + kind] = digest_array(layer)
        output += layer
        stages["mix_after_" + kind] = digest_array(output)
    stages["mix_pre_fade"] = digest_array(output)

    edge = min(int(0.025 * dsp.SAMPLE_RATE), length // 2)
    if edge:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        output[:edge] *= ramp
        output[-edge:] *= ramp[::-1]
    stages["mix_post_fade"] = digest_array(output)
    peak = float(np.max(np.abs(output)))
    if peak > 0:
        output *= min(1.0, 0.92 / peak)
    stages["normalized_24k"] = digest_array(output)

    # Independent call to the unmodified official generator is a strict
    # correctness oracle. The stage reconstruction must be byte-identical.
    canonical = dsp.generate(label, 1.5, seed).samples
    if (canonical.dtype != np.float32 or canonical.shape != output.shape
            or not np.array_equal(canonical, output)):
        raise ValueError("Diagnostic diverged from original Sonumio DSP")
    taps = np.arange(-24, 25, dtype=np.float64)
    kernel = np.sinc(taps / 3) * np.hamming(len(taps))
    kernel /= kernel.sum()
    stages["fir_kernel"] = digest_array(kernel)
    filtered = np.convolve(output.astype(np.float64), kernel, mode="same")
    stages["fir_filtered_24k"] = digest_array(filtered)
    downsampled = filtered[::3].astype(np.float32)
    stages["downsampled_8k"] = digest_array(downsampled)
    offset = int(np.random.default_rng(seed).integers(400, 1601))
    cropped = downsampled[offset:offset + FRAMES]
    if cropped.shape != (FRAMES,):
        raise ValueError("Invalid crop dimensions")
    stages["cropped_8k"] = digest_array(cropped)
    original = render_seeded_waveform(members, seed)
    if not np.array_equal(cropped, original):
        raise ValueError("Diagnostic crop diverged from trained source audio")
    return {
        "sources": list(CATEGORIES[m] for m in members),
        "seed": seed, "offset": offset, "stages": stages,
    }


def _aggregate(records: list[dict]) -> dict:
    labels = sorted({stage for record in records for stage in record["stages"]})
    return {
        label: {
            "count": sum(label in record["stages"] for record in records),
            "sha256": _digest(b"".join(
                json.dumps({
                    "sources": record["sources"],
                    "seed": record["seed"],
                    "sha256": record["stages"][label]["sha256"],
                }, sort_keys=True, ensure_ascii=False).encode("utf-8")
                for record in records if label in record["stages"]
            )),
            **{
                f"bucket_{step:g}_sha256": _digest(b"".join(
                    bytes.fromhex(record["stages"][label][f"bucket_{step:g}_sha256"])
                    for record in records if label in record["stages"]
                ))
                for step in QUANTIZATION_STEPS
            },
        }
        for label in labels
    }


def build_trace() -> dict:
    records = [
        {"split": split, **trace_one(members, seed)}
        for split, seeds in (("train", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS))
        for members in SYNTHETIC_COMBINATIONS for seed in seeds
    ]
    all_stages = _aggregate(records)
    # Single-category cases help narrow differences to a DSP layer.
    per_single_source = {
        source: _aggregate([r for r in records if r["sources"] == [source]])
        for source in CATEGORIES
    }
    result = {
        "protocol": PROTOCOL,
        "source_recipe_unchanged": True,
        "matches_original_renderer_exactly": True,
        "sample_rate_hz": 8000, "frames": FRAMES,
        "records": records,
        "stage_aggregate": all_stages,
        "single_source_aggregate": per_single_source,
        "environment": _environment(),
        "audio_files_saved": False,
        "model_weights_saved": False,
    }
    validate_trace(result)
    return result


def validate_trace(value: dict) -> None:
    if (not isinstance(value, dict) or value.get("protocol") != PROTOCOL
            or value.get("source_recipe_unchanged") is not True
            or value.get("matches_original_renderer_exactly") is not True
            or value.get("sample_rate_hz") != 8000 or value.get("frames") != FRAMES
            or value.get("audio_files_saved") is not False
            or value.get("model_weights_saved") is not False):
        raise ValueError("Invalid or unsafe DSP pipeline trace")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 84:
        raise ValueError("Pipeline trace must cover 84 unique recordings")
    identities = set()
    for record in records:
        identity = (record.get("split"), tuple(record.get("sources", [])),
                    record.get("seed"))
        if identity in identities:
            raise ValueError("Duplicate trace record")
        identities.add(identity)
        stages = record.get("stages")
        if not isinstance(stages, dict) or not set(STAGES).issubset(stages):
            raise ValueError("Trace missing mandatory processing stage")
        if not any(x.startswith("layer_") for x in stages):
            raise ValueError("Trace missing source layer")
        for stage in stages.values():
            if (stage.get("dtype") not in ("float32", "float64")
                    or type(stage.get("frames")) is not int
                    or not 2 <= stage["frames"] <= 40000
                    or any(not _hash_ok(stage.get(key))
                           for key in ("sha256", "bucket_0.001_sha256",
                                       "bucket_0.01_sha256"))):
                raise ValueError("Invalid stage digest metadata")
    expected_ids = {
        (split, tuple(CATEGORIES[m] for m in members), seed)
        for split, seeds in (("train", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS))
        for members in SYNTHETIC_COMBINATIONS for seed in seeds
    }
    if identities != expected_ids:
        raise ValueError("Pipeline trace source grid incomplete")
    actual = _aggregate(records)
    if value.get("stage_aggregate") != actual:
        raise ValueError("Forged/inconsistent stage aggregate")
    for source in CATEGORIES:
        if value.get("single_source_aggregate", {}).get(source) != _aggregate(
            [r for r in records if r["sources"] == [source]]
        ):
            raise ValueError("Forged/inconsistent single-source aggregate")


def compare_traces(a: dict, b: dict) -> dict:
    """Classify first differing stages; do not infer cause from SHA alone."""
    validate_trace(a)
    validate_trace(b)
    left = {(r["split"], tuple(r["sources"]), r["seed"]): r for r in a["records"]}
    right = {(r["split"], tuple(r["sources"]), r["seed"]): r for r in b["records"]}
    if set(left) != set(right):
        raise ValueError("Cannot compare incompatible source grids")
    counts = {}
    first = []
    for key in sorted(left):
        first_difference = None
        for stage, info in left[key]["stages"].items():
            other = right[key]["stages"].get(stage)
            if other is None:
                raise ValueError("Missing corresponding pipeline stage")
            if info["sha256"] != other["sha256"]:
                if first_difference is None:
                    first_difference = stage
        if first_difference is not None:
            counts[first_difference] = counts.get(first_difference, 0) + 1
            if len(first) < 12:
                first.append({"split": key[0], "sources": list(key[1]),
                              "seed": key[2], "first_differing_stage": first_difference})
    return {
        "protocol": "sonumio-pipeline-trace-comparison-v1",
        "recordings_compared": len(left),
        "exact_stage_arrays_match": len(counts) == 0,
        "first_differing_stage_counts": counts,
        "examples": first,
        "cause_unproven": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true",
                        help="Log stage aggregates without per-record arrays")
    args = parser.parse_args()
    data = build_trace()
    report = {
        key: value for key, value in data.items() if key != "records"
    } if args.summary_only else data
    serialized = json.dumps(report, indent=2, sort_keys=True,
                            ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
