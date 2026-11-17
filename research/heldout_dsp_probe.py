"""Held-out-seed numerical reproducibility experiment for fast first-party DSP.

Five previously untested seeds × six individual source categories and fifteen
source pairs. Keep this distinct from the existing neural train/validation
recording seeds, original DSP generator and all model weights. This is NOT a
new training split or real-world dataset. No waveforms are serialized.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from research.complete_dsp_candidate import (
    FRAMES, RAW_DURATION, RAW_FRAMES, _crop, _difference, MAX_EXACT_ERROR,
    TRAIN_SEEDS, VALIDATION_SEEDS,
)
from research.data_reproducibility import _digest, _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from research.fast_dsp_candidate import render_fast_raw
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES, _downsample_24k_to_8k
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-fast-dsp-held-out-seed-fingerprint-v1"
HELDOUT_SEEDS = (3, 7, 17, 29, 41)
if set(HELDOUT_SEEDS) & (set(TRAIN_SEEDS) | set(VALIDATION_SEEDS)):
    raise RuntimeError("Held-out DSP seeds overlap train/validation source identities")


def _case(members: tuple[int, ...], seed: int) -> dict:
    if (type(members) is not tuple or members not in SYNTHETIC_COMBINATIONS
            or type(seed) is not int or seed not in HELDOUT_SEEDS):
        raise ValueError("Unsupported held-out DSP probe identity")
    prompt = " and ".join(CATEGORIES[i] for i in members)
    fast = render_fast_raw(members, seed)
    original = dsp.generate(prompt, RAW_DURATION, seed).samples
    raw_error = _difference(fast, original)
    fast_8 = _downsample_24k_to_8k(fast)
    old_8 = _downsample_24k_to_8k(original)
    offset, crop = _crop(fast_8, seed)
    old_offset, old_crop = _crop(old_8, seed)
    if offset != old_offset:
        raise ValueError("Held-out crop offset changed")
    crop_error = _difference(crop, old_crop)
    if (raw_error["max_abs"] > MAX_EXACT_ERROR
            or crop_error["max_abs"] > MAX_EXACT_ERROR):
        raise ValueError("Fast holdout deviation exceeded 2e-5")
    return {
        "source_categories": [CATEGORIES[i] for i in members],
        "seed": seed,
        "crop_offset": offset,
        "fast_raw_sha256": digest_array(fast)["sha256"],
        "fast_crop_sha256": digest_array(crop)["sha256"],
        "original_raw_sha256": digest_array(original)["sha256"],
        "original_crop_sha256": digest_array(old_crop)["sha256"],
        "raw_max_abs_error": raw_error["max_abs"],
        "crop_max_abs_error": crop_error["max_abs"],
        "raw_different_float32_values": raw_error["different_float32_values"],
        "crop_different_float32_values": crop_error["different_float32_values"],
    }


def _summary(records: list[dict]) -> dict:
    results = {}
    for seed in HELDOUT_SEEDS:
        current = [r for r in records if r["seed"] == seed]
        results[str(seed)] = {
            "recordings": len(current),
            "fast_24k_sha256": _digest(b"".join(
                bytes.fromhex(row["fast_raw_sha256"]) for row in current)),
            "fast_8k_crop_sha256": _digest(b"".join(
                bytes.fromhex(row["fast_crop_sha256"]) for row in current)),
            "original_24k_sha256": _digest(b"".join(
                bytes.fromhex(row["original_raw_sha256"]) for row in current)),
            "original_8k_crop_sha256": _digest(b"".join(
                bytes.fromhex(row["original_crop_sha256"]) for row in current)),
            "max_raw_abs_error": max(x["raw_max_abs_error"] for x in current),
            "max_crop_abs_error": max(x["crop_max_abs_error"] for x in current),
        }
    return results


def build_report() -> dict:
    recordings = [
        _case(members, seed)
        for seed in HELDOUT_SEEDS
        for members in SYNTHETIC_COMBINATIONS
    ]
    report = {
        "protocol": PROTOCOL,
        "seeds": list(HELDOUT_SEEDS),
        "source_hz": dsp.SAMPLE_RATE,
        "source_frames": RAW_FRAMES,
        "crop_hz": 8000,
        "crop_frames": FRAMES,
        "recordings": recordings,
        "by_seed": _summary(recordings),
        "environment": _environment(),
        "same_source_family_as_neural_experiments": True,
        "heldout_seeds_not_used_for_model_training": True,
        "candidate_used_for_neural_training": False,
        "original_generator_unchanged": True,
        "audio_files_saved": False,
        "model_weights_saved": False,
    }
    validate_report(report)
    return report


def validate_report(value: dict) -> None:
    if (not isinstance(value, dict)
            or value.get("protocol") != PROTOCOL
            or value.get("seeds") != list(HELDOUT_SEEDS)
            or value.get("source_hz") != dsp.SAMPLE_RATE
            or value.get("source_frames") != RAW_FRAMES
            or value.get("crop_hz") != 8000
            or value.get("crop_frames") != FRAMES
            or value.get("same_source_family_as_neural_experiments") is not True
            or value.get("heldout_seeds_not_used_for_model_training") is not True
            or value.get("candidate_used_for_neural_training") is not False
            or value.get("original_generator_unchanged") is not True
            or value.get("audio_files_saved") is not False
            or value.get("model_weights_saved") is not False):
        raise ValueError("Invalid held-out metadata or source identity")
    expected = {(seed, tuple(members))
                for seed in HELDOUT_SEEDS
                for members in SYNTHETIC_COMBINATIONS}
    rows = value.get("recordings")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("Invalid held-out case count")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed held-out case")
        try:
            members = tuple(CATEGORIES.index(label) for label
                            in row["source_categories"])
            key = (row["seed"], members)
        except (KeyError, ValueError, TypeError):
            raise ValueError("Malformed held-out identity") from None
        if (type(row.get("seed")) is not int or key not in expected
                or key in seen):
            raise ValueError("Duplicate or unexpected held-out source")
        seen.add(key)
        if (type(row.get("crop_offset")) is not int
                or not 400 <= row["crop_offset"] <= 1600):
            raise ValueError("Invalid held-out crop offset")
        for field in (
            "fast_raw_sha256", "fast_crop_sha256",
            "original_raw_sha256", "original_crop_sha256",
        ):
            if not _hash_ok(row.get(field)):
                raise ValueError("Invalid held-out SHA256")
        for name, limit in (("raw", RAW_FRAMES), ("crop", FRAMES)):
            error = row.get(name + "_max_abs_error")
            changed = row.get(name + "_different_float32_values")
            if (type(error) not in (int, float) or not math.isfinite(error)
                    or not 0 <= error <= MAX_EXACT_ERROR
                    or type(changed) is not int or not 0 <= changed <= limit):
                raise ValueError("Invalid held-out numerical error")
    if seen != expected or value.get("by_seed") != _summary(rows):
        raise ValueError("Inconsistent held-out source aggregate")

    # The ambient drone is seed-invariant before random crop, so raw SHA
    # uniqueness is NOT required; cropped waveforms must still be unique.
    for field in ("fast_crop_sha256", "original_crop_sha256"):
        hashes = [row[field] for row in rows]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Exact duplicate held-out cropped waveform")


def compare_reports(first: dict, second: dict) -> dict:
    validate_report(first)
    validate_report(second)
    a = {(r["seed"], tuple(r["source_categories"])): r
         for r in first["recordings"]}
    b = {(r["seed"], tuple(r["source_categories"])): r
         for r in second["recordings"]}
    return {
        "protocol": "sonumio-fast-heldout-numerical-comparison-v1",
        "distinct_recordings": len(a),
        "identical_fast_24k_hashes": all(
            a[k]["fast_raw_sha256"] == b[k]["fast_raw_sha256"] for k in a),
        "identical_fast_8k_crop_hashes": all(
            a[k]["fast_crop_sha256"] == b[k]["fast_crop_sha256"] for k in a),
        "identical_original_8k_crop_hashes": all(
            a[k]["original_crop_sha256"] == b[k]["original_crop_sha256"] for k in a),
    }



# Independently measured NumPy 2.4.6 four-host SHA aggregates, workflow
# https://github.com/sanskarIN/sonumio/actions/runs/44075765225
# Each seed aggregate hashes concatenated per-recording SHA256 digests.
HELDOUT_REFERENCE = {
    "3": {
        "fast_24k_sha256": "e1076d40c4d72e2b75bd3588f408c077d961d407d98c95c6c7c30d6f1893ef95",
        "fast_8k_crop_sha256": "aab15df63345413c2c5075803d81194413a1bee5a5994799ea277af101a8765b",
    },
    "7": {
        "fast_24k_sha256": "2840877f38f17e3ea0b39263cfa4533781473b2ef85688b1ee448962cdf8dda1",
        "fast_8k_crop_sha256": "68991ec70a90f369947d11b318fbcf43a71be78dd38a515d73d907b31ef3d806",
    },
    "17": {
        "fast_24k_sha256": "30df175004b4909e2cbb76971d6e0b0d3bb39179a45b3b2709355c680d4261b8",
        "fast_8k_crop_sha256": "5eed92f868e9a03fa64c461ffd51ea690caaaca36848c77a699000dcb144d897",
    },
    "29": {
        "fast_24k_sha256": "00776f177d1dfa0de268e89ca424f3f826f16128ccf57a1834f9b7f6b35393a9",
        "fast_8k_crop_sha256": "db8a189f7fb0ea536670d9eeb54cd75abefe27917b94a72cadcf98c30a89718f",
    },
    "41": {
        "fast_24k_sha256": "9f59d6cb67af7c2671e321fb75baa60ab968f114475977af89627af07d28858e",
        "fast_8k_crop_sha256": "a76d3a12854b6d59b90d084cb40e79223876deebfc099526b8ad0b37bf673726",
    },
}


def verify_heldout_reference(report: dict) -> dict:
    """Gate the full held-out corpus against independently verified bytes.

    Intentionally do NOT pin legacy NumPy float32 convolution digests,
    which differed across original CPU environments.
    """
    validate_report(report)
    for seed, hashes in HELDOUT_REFERENCE.items():
        observed = report["by_seed"].get(seed)
        if not isinstance(observed, dict) or observed.get("recordings") != 21:
            raise ValueError("Missing verified held-out seed " + seed)
        for name, reference in hashes.items():
            if observed.get(name) != reference:
                raise ValueError("Held-out 24k/8k reference drift: " + seed + "." + name)
    return {
        "protocol": "sonumio-fast-dsp-heldout-golden-v1",
        "recordings": 105,
        "matched_verified_sha256": True,
        "numpy_reference_version": "2.4.6",
        "reference_workflow": "https://github.com/sanskarIN/sonumio/actions/runs/44075765225",
        "old_model_data_unchanged": True,
        "model_weights_saved": False,
    }



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    data = json.dumps(report, indent=2, sort_keys=True,
                      allow_nan=False, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    print(data, end="")


if __name__ == "__main__":
    main()
