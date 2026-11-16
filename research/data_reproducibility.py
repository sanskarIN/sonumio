"""Audit exact and tolerance-quantized provenance of Sonumio's DSP research corpus.

No training weights, samples, WAVs, or external data are saved. This records
SHA256 digests of the actual float32 arrays that enter model training, not a
guess based on source seed or configuration. Quantized hashes are *diagnostic*,
NEVER proof of bitwise data identity, license validity, or sound fidelity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from research.latent_audio import (
    DATA_RECIPE, FRAMES, SAMPLE_RATE, TRAIN_SEEDS, VALIDATION_SEEDS,
    synthetic_bilingual_waves,
)
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES

PROTOCOL = "sonumio-dsp-audio-fingerprint-audit-v1"
# Both quantizers are for numerical change detection ONLY; no training arrays
# are quantized or altered by this script.
QUANTIZATION_STEPS = (0.001, 0.01)
MAX_MANIFEST_BYTES = 256 * 1024
MAX_MISMATCH_EXAMPLES = 12
_HEX = frozenset("0123456789abcdef")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_ok(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def waveform_digests(audio: np.ndarray) -> dict:
    """Hash byte-level float32 plus coarse bucketed representations.

    A matching bucketed hash does not guarantee a maximum sample deviation:
    large changes within a bucket (or changes in opposing directions) remain
    possible. Bucket hashes are never used to accept train/test leakage.
    """
    if (not isinstance(audio, np.ndarray)
            or audio.shape != (FRAMES,) or audio.dtype != np.dtype("float32")
            or not np.isfinite(audio).all()):
        raise ValueError("Expected one finite float32 8192-sample recording")
    # Canonical little endian representation prevents host byte order drift.
    exact = np.asarray(audio, dtype="<f4", order="C")
    result = {"exact_sha256": _digest(exact.tobytes())}
    values = audio.astype(np.float64)
    for level in QUANTIZATION_STEPS:
        bins = np.rint(values / level)
        if np.max(np.abs(bins)) > 2**30:
            raise ValueError("Audio cannot fit bounded signed int32 buckets")
        quantized = bins.astype("<i4")
        result[f"bucket_{level:g}_sha256"] = _digest(quantized.tobytes())
    result["rms"] = float(np.sqrt(np.square(values).mean()))
    result["peak"] = float(np.abs(values).max())
    return result


def _environment() -> dict:
    # No user environment dump or secrets; only allowlisted numerics metadata.
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
        "threads": {
            key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"
            )
        },
    }


def build_manifest() -> dict:
    """Inspect ALL 84 distinct-source/seed recordings before caption expansion."""
    entries = []
    splits = {}
    for split, seeds in (("train", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS)):
        dataset = synthetic_bilingual_waves(seeds)
        expected = len(SYNTHETIC_COMBINATIONS) * len(seeds)
        if dataset.examples != expected * 2 or dataset.waves.shape != (
            expected * 2, 1, FRAMES
        ):
            raise ValueError("Synthetic bilingual audio grid was not complete")
        source_hashes = set()
        descriptions = set()
        for group_i, members in enumerate(SYNTHETIC_COMBINATIONS):
            for seed_i, seed in enumerate(seeds):
                i = (group_i * len(seeds) + seed_i) * 2
                if not torch.equal(dataset.waves[i], dataset.waves[i + 1]):
                    raise ValueError("English/Hindi pair did not share exact recording")
                wave = dataset.waves[i, 0]
                if (wave.device.type != "cpu" or wave.requires_grad
                        or wave.dtype != torch.float32):
                    raise ValueError("Dataset waveform must be detached CPU float32")
                info = waveform_digests(wave.numpy())
                if info["exact_sha256"] in source_hashes:
                    raise ValueError("Duplicate recording within one data split")
                source_hashes.add(info["exact_sha256"])
                description = tuple(CATEGORIES[idx] for idx in members)
                if (description, seed) in descriptions:
                    raise ValueError("Repeated source identity in the dataset")
                descriptions.add((description, seed))
                entries.append({
                    "split": split,
                    "sources": list(description),
                    "seed": seed,
                    **info,
                })
        splits[split] = {
            "dataset_fingerprint": dataset.fingerprint,
            "recording_count": len(source_hashes),
            "caption_count": dataset.examples,
            "recording_sha256": _digest(
                b"".join(bytes.fromhex(x["exact_sha256"])
                         for x in entries if x["split"] == split)),
        }
    train = {e["exact_sha256"] for e in entries if e["split"] == "train"}
    validation = {e["exact_sha256"] for e in entries if e["split"] == "validation"}
    if train & validation:
        raise ValueError("Training and validation share exact waveform bytes")
    result = {
        "protocol": PROTOCOL,
        "data_recipe": DATA_RECIPE,
        "sample_rate": SAMPLE_RATE,
        "frames": FRAMES,
        "synthetic_only": True,
        "quantization_steps": list(QUANTIZATION_STEPS),
        "splits": splits,
        "entries": entries,
        "environment": _environment(),
        "no_audio_or_weights_saved": True,
    }
    validate_manifest(result)
    return result


def validate_manifest(value: dict) -> None:
    if (not isinstance(value, dict) or value.get("protocol") != PROTOCOL
            or value.get("data_recipe") != DATA_RECIPE
            or value.get("sample_rate") != 8000 or value.get("frames") != 8192
            or value.get("quantization_steps") != list(QUANTIZATION_STEPS)
            or value.get("synthetic_only") is not True
            or value.get("no_audio_or_weights_saved") is not True):
        raise ValueError("Incompatible or unsafe audio provenance manifest")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != 84:
        raise ValueError("Expected 84 unique source/recording-seed combinations")
    expected = set()
    for split, seeds in (("train", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS)):
        for members in SYNTHETIC_COMBINATIONS:
            for seed in seeds:
                expected.add((split, tuple(CATEGORIES[m] for m in members), seed))
    observed = set()
    hashes = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Manifest record is malformed")
        key = (entry.get("split"), tuple(entry.get("sources", ())), entry.get("seed"))
        if key not in expected or key in observed:
            raise ValueError("Manifest source grid contains missing/duplicate cases")
        observed.add(key)
        for name in ("exact_sha256", "bucket_0.001_sha256", "bucket_0.01_sha256"):
            if not _hash_ok(entry.get(name)):
                raise ValueError("Missing or invalid per-recording SHA256")
        if entry["exact_sha256"] in hashes:
            raise ValueError("Manifest contains train/validation byte duplicates")
        hashes.add(entry["exact_sha256"])
        for key in ("rms", "peak"):
            v = entry.get(key)
            if type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1.0001:
                raise ValueError("Invalid finite audio health statistic")
    if observed != expected:
        raise ValueError("Manifest omitted a source/recording case")
    splits = value.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation"}:
        raise ValueError("Manifest must contain train and validation")
    for split in ("train", "validation"):
        record = splits[split]
        if (record.get("recording_count") != 42
                or record.get("caption_count") != 84
                or not _hash_ok(record.get("dataset_fingerprint"))
                or not _hash_ok(record.get("recording_sha256"))):
            raise ValueError("Invalid split-level provenance")
        digest = _digest(b"".join(
            bytes.fromhex(entry["exact_sha256"])
            for entry in entries if entry["split"] == split
        ))
        if record["recording_sha256"] != digest:
            raise ValueError("Split-level byte digest contradicts recording hashes")


def compare_manifests(first: dict, second: dict) -> dict:
    """Compare two process outputs; do NOT label approximate buckets as identity."""
    validate_manifest(first)
    validate_manifest(second)
    if (first["data_recipe"] != second["data_recipe"]
            or first["quantization_steps"] != second["quantization_steps"]):
        raise ValueError("Cannot compare manifest entries from different recipes")
    one = {
        (e["split"], tuple(e["sources"]), e["seed"]): e
        for e in first["entries"]
    }
    two = {
        (e["split"], tuple(e["sources"]), e["seed"]): e
        for e in second["entries"]
    }
    if set(one) != set(two):
        raise ValueError("Comparisons require identical source grid")
    mismatched = []
    exact = True
    bucket_counts = {f"{x:g}": 0 for x in QUANTIZATION_STEPS}
    for key in sorted(one):
        x, y = one[key], two[key]
        difference = x["exact_sha256"] != y["exact_sha256"]
        if difference:
            exact = False
            if len(mismatched) < MAX_MISMATCH_EXAMPLES:
                mismatched.append({
                    "split": key[0], "sources": list(key[1]), "seed": key[2],
                    "first_sha256": x["exact_sha256"],
                    "second_sha256": y["exact_sha256"],
                })
        for step in QUANTIZATION_STEPS:
            if x[f"bucket_{step:g}_sha256"] != y[f"bucket_{step:g}_sha256"]:
                bucket_counts[f"{step:g}"] += 1
    same_fingerprints = all(
        first["splits"][split]["dataset_fingerprint"] ==
        second["splits"][split]["dataset_fingerprint"]
        for split in ("train", "validation")
    )
    if exact != same_fingerprints:
        raise ValueError("Dataset fingerprint conflicts with per-record byte comparison")
    return {
        "protocol": "sonumio-dsp-audio-manifest-comparison-v1",
        "recordings_compared": len(one),
        "exact_recording_bytes_equal": exact,
        "exact_dataset_fingerprints_equal": same_fingerprints,
        "quantized_bucket_mismatch_counts": bucket_counts,
        "illustrative_exact_mismatches": mismatched,
        "first_environment": first["environment"],
        "second_environment": second["environment"],
        "interpretation": (
            "Bucket hashes are diagnostics ONLY, not proof of byte identity "
            "or a bound on maximum sample error."
        ),
    }


def _read(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Invalid or oversized provenance manifest")
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write only JSON SHA/metrics; no WAV")
    parser.add_argument("--compare", type=Path, nargs=2,
                        metavar=("FIRST", "SECOND"))
    parser.add_argument("--strict-exact", action="store_true",
                        help="Exit failure when compared waveform bytes differ")
    options = parser.parse_args()
    if options.compare:
        result = compare_manifests(_read(options.compare[0]), _read(options.compare[1]))
    else:
        result = build_manifest()
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=False,
                         indent=2, allow_nan=False) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if options.strict_exact and options.compare and not result["exact_recording_bytes_equal"]:
        parser.exit(1, "Synthetic waveform byte fingerprints differ.\n")


if __name__ == "__main__":
    main()
