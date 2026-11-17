"""Locate the earliest cross-NumPy-version DSP byte divergence.

Reconstructs the unchanged procedural candidate stage-by-stage for selected
first-party seeds. No audio files, arrays or model weights leave the process.
Run twice in separate Python processes on one machine after changing ONLY
NumPy version to distinguish version differences from host differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.complete_dsp_candidate import (
    RAW_DURATION, RAW_FRAMES, _crop, MAX_EXACT_ERROR,
)
from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.fast_dsp_candidate import _fast_layer, render_fast_raw
from research.wave_diffusion import CATEGORIES, _downsample_24k_to_8k
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-numpy-stage-localization-v1"
SEEDS = (0, 1, 101, 102)
CANDIDATES = ((0,), (1,), (2,), (3,), (4,), (5,),
              (0, 1), (0, 5), (2, 3), (4, 5))
FRAMES_8K = 8192


def trace_one(members: tuple[int, ...], seed: int) -> dict:
    if (type(members) is not tuple or members not in CANDIDATES
            or type(seed) is not int or seed not in SEEDS):
        raise ValueError("Unsupported first-party NumPy stage identity")
    prompt = " and ".join(CATEGORIES[i] for i in members)
    kinds = dsp.recognized_sound_classes(prompt)
    t = np.arange(RAW_FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    rng = np.random.default_rng(seed)
    signal = np.zeros(RAW_FRAMES, dtype=np.float32)
    stages = {"time_grid": digest_array(t)["sha256"]}
    for kind in kinds:
        layer = _fast_layer(kind, rng, t)
        stages["layer_" + kind] = digest_array(layer)["sha256"]
        signal += layer
        stages["mix_after_" + kind] = digest_array(signal)["sha256"]
    stages["mix_pre_fade"] = digest_array(signal)["sha256"]
    edge = min(int(0.025 * dsp.SAMPLE_RATE), RAW_FRAMES // 2)
    if edge:
        ramp = np.linspace(0, 1, edge, dtype=np.float32)
        stages["fade_ramp"] = digest_array(ramp)["sha256"]
        signal[:edge] *= ramp
        signal[-edge:] *= ramp[::-1]
    stages["mix_post_fade"] = digest_array(signal)["sha256"]
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal *= min(1.0, 0.92 / peak)
    stages["normalized_24k"] = digest_array(signal)["sha256"]
    # Confirm that the stage reconstruction is bitwise identical to the
    # fast algorithm under the SAME NumPy version before using trace data.
    canonical = render_fast_raw(members, seed)
    if not np.array_equal(signal, canonical):
        raise ValueError("Stage probe differs from the actual fast DSP candidate")
    taps = np.arange(-24, 25, dtype=np.float64)
    kernel = np.sinc(taps / 3) * np.hamming(len(taps))
    kernel /= kernel.sum()
    stages["fir_kernel"] = digest_array(kernel)["sha256"]
    filtered = np.convolve(signal.astype(np.float64), kernel, mode="same")
    stages["fir_filtered_24k"] = digest_array(filtered)["sha256"]
    downsampled = filtered[::3].astype(np.float32)
    stages["downsampled_8k"] = digest_array(downsampled)["sha256"]
    original_down = _downsample_24k_to_8k(signal)
    if not np.array_equal(downsampled, original_down):
        raise ValueError("Manual FIR differs from original research resampler")
    offset, cropped = _crop(downsampled, seed)
    if cropped.shape != (FRAMES_8K,):
        raise ValueError("Invalid final crop")
    stages["cropped_8k"] = digest_array(cropped)["sha256"]
    return {
        "sources": [CATEGORIES[i] for i in members],
        "seed": seed,
        "crop_offset": offset,
        "stages": stages,
    }


def _aggregate(records: list[dict]) -> dict:
    stage_names = sorted({name for r in records for name in r["stages"]})
    return {name: {
        "recordings": sum(name in r["stages"] for r in records),
        "sha256": _digest(b"".join(
            bytes.fromhex(r["stages"][name])
            for r in records if name in r["stages"]
        )),
    } for name in stage_names}


def build_trace() -> dict:
    records = [trace_one(members, seed)
               for members in CANDIDATES for seed in SEEDS]
    result = {
        "protocol": PROTOCOL,
        "recordings": records,
        "stage_aggregate": _aggregate(records),
        "environment": _environment(),
        "source_recipe_unchanged": True,
        "candidate_not_used_for_training": True,
        "audio_files_saved": False,
        "weights_saved": False,
    }
    validate_trace(result)
    return result


def validate_trace(value: dict) -> None:
    if (not isinstance(value, dict) or value.get("protocol") != PROTOCOL
            or value.get("source_recipe_unchanged") is not True
            or value.get("candidate_not_used_for_training") is not True
            or value.get("audio_files_saved") is not False
            or value.get("weights_saved") is not False):
        raise ValueError("Invalid version-localization metadata")
    records = value.get("recordings")
    expected = {(tuple(CATEGORIES[i] for i in m), s)
                for m in CANDIDATES for s in SEEDS}
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("Missing source/seed trace grid")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid stage record")
        try:
            identity = (tuple(record["sources"]), record["seed"])
        except (KeyError, TypeError):
            raise ValueError("Invalid source identity") from None
        if (identity not in expected or identity in seen
                or type(record.get("seed")) is not int):
            raise ValueError("Duplicate/unexpected source trace")
        seen.add(identity)
        if type(record.get("crop_offset")) is not int or not 400 <= record["crop_offset"] <= 1600:
            raise ValueError("Invalid crop offset")
        stages = record.get("stages")
        if (not isinstance(stages, dict) or
                any(not _hash_ok(digest) for digest in stages.values()) or
                not {"time_grid", "mix_pre_fade", "mix_post_fade",
                     "normalized_24k", "fir_kernel", "fir_filtered_24k",
                     "downsampled_8k", "cropped_8k"}.issubset(stages)):
            raise ValueError("Invalid or incomplete stage hashes")
    if seen != expected or value.get("stage_aggregate") != _aggregate(records):
        raise ValueError("Inconsistent stage aggregate")


def compare_traces(first: dict, second: dict) -> dict:
    validate_trace(first)
    validate_trace(second)
    a = {(tuple(r["sources"]), r["seed"]): r for r in first["recordings"]}
    b = {(tuple(r["sources"]), r["seed"]): r for r in second["recordings"]}
    first_different = {}
    exact = 0
    for key, row in a.items():
        before, after = row["stages"], b[key]["stages"]
        if tuple(before) != tuple(after):
            raise ValueError("Stage execution order changed")
        mismatches = [name for name in before if before[name] != after[name]]
        if mismatches:
            first_different[f"{'/'.join(key[0])}@{key[1]}"] = mismatches[0]
        else:
            exact += 1
    common = sorted(set(first["stage_aggregate"]) & set(second["stage_aggregate"]))
    paired_differences = {
        name: sum(
            a[key]["stages"].get(name) != b[key]["stages"].get(name)
            for key in a
        )
        for name in common
    }
    aggregate_order_only = [
        name for name in common
        if paired_differences[name] == 0
        and first["stage_aggregate"][name]["sha256"]
        != second["stage_aggregate"][name]["sha256"]
    ]
    return {
        "protocol": "sonumio-numpy-stage-comparison-v1",
        "cases": len(a),
        "paired_differing_recordings_by_stage": paired_differences,
        "aggregate_order_only_differences": aggregate_order_only,
        "fully_identical_cases": exact,
        "first_divergent_stage_per_case": first_different,
        "matching_stages": [
            name for name in first["stage_aggregate"]
            if first["stage_aggregate"][name]["sha256"]
            == second["stage_aggregate"][name]["sha256"]
        ],
        "differing_stages": [
            name for name in first["stage_aggregate"]
            if first["stage_aggregate"][name]["sha256"]
            != second["stage_aggregate"][name]["sha256"]
        ],
        "first_numpy_version": first["environment"]["numpy"],
        "second_numpy_version": second["environment"]["numpy"],
        "same_cpu_model": first["environment"]["cpu_model"] == second["environment"]["cpu_model"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare:
        first, second = [json.loads(p.read_text(encoding="utf-8")) for p in args.compare]
        result = compare_traces(first, second)
    else:
        result = build_trace()
    encoded = json.dumps(result, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
