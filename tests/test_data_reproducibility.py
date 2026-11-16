"""Data provenance tests: exact bytes vs quantized buckets must never be confused."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.data_reproducibility import (
    PROTOCOL, _read, build_manifest, compare_manifests,
    validate_manifest, waveform_digests,
)
from research.latent_audio import FRAMES


@pytest.fixture(scope="module")
def manifest():
    return build_manifest()


def _digest_of_records(rows, split):
    return hashlib.sha256(b"".join(
        bytes.fromhex(record["exact_sha256"])
        for record in rows if record["split"] == split
    )).hexdigest()


def test_manifest_covers_all_recordings_not_twice_per_language(manifest):
    assert manifest["protocol"] == PROTOCOL
    assert manifest["data_recipe"]
    assert manifest["sample_rate"] == 8000 and manifest["frames"] == 8192
    assert len(manifest["entries"]) == 84
    assert manifest["no_audio_or_weights_saved"] is True
    assert manifest["synthetic_only"] is True
    for split in ("train", "validation"):
        group = manifest["splits"][split]
        assert group["recording_count"] == 42
        assert group["caption_count"] == 84
        assert len(group["dataset_fingerprint"]) == 64
        assert group["recording_sha256"] == _digest_of_records(
            manifest["entries"], split
        )
    assert len(set(e["exact_sha256"] for e in manifest["entries"])) == 84
    assert {x["seed"] for x in manifest["entries"] if x["split"] == "train"} == {0, 1}
    assert {x["seed"] for x in manifest["entries"] if x["split"] == "validation"} == {
        101, 102
    }
    assert json.dumps(manifest, allow_nan=False)
    validate_manifest(manifest)
    identical = compare_manifests(manifest, copy.deepcopy(manifest))
    assert identical["recordings_compared"] == 84
    assert identical["exact_recording_bytes_equal"]
    assert identical["exact_dataset_fingerprints_equal"]
    assert identical["quantized_bucket_mismatch_counts"] == {
        "0.001": 0, "0.01": 0
    }
    assert identical["illustrative_exact_mismatches"] == []


def test_waveform_digests_are_sensible_and_quantization_is_not_identity():
    silence = np.zeros(FRAMES, dtype=np.float32)
    tiny = silence.copy()
    tiny[12] = np.float32(0.00001)
    a, b = waveform_digests(silence), waveform_digests(tiny)
    assert a["exact_sha256"] != b["exact_sha256"]
    # A tiny change stays in both coarse buckets. Do not call this
    # "same recording" or mistake it for a max-error bound.
    assert a["bucket_0.001_sha256"] == b["bucket_0.001_sha256"]
    assert a["bucket_0.01_sha256"] == b["bucket_0.01_sha256"]
    bigger = silence.copy()
    bigger[12] = 0.2
    assert waveform_digests(bigger)["bucket_0.01_sha256"] != a["bucket_0.01_sha256"]
    for bad in (np.zeros(FRAMES-1, np.float32),
                np.zeros((1, FRAMES), np.float32),
                np.full(FRAMES, float("nan"), np.float32),
                np.zeros(FRAMES, dtype=np.int16)):
        with pytest.raises(ValueError):
            waveform_digests(bad)


def test_manifest_byte_drift_is_not_suppressed_by_tolerance_buckets(manifest):
    changed = copy.deepcopy(manifest)
    first = changed["entries"][0]
    first["exact_sha256"] = "1" * 64
    changed["splits"]["train"]["recording_sha256"] = _digest_of_records(
        changed["entries"], "train"
    )
    changed["splits"]["train"]["dataset_fingerprint"] = "2" * 64
    result = compare_manifests(manifest, changed)
    assert not result["exact_recording_bytes_equal"]
    assert not result["exact_dataset_fingerprints_equal"]
    assert result["quantized_bucket_mismatch_counts"] == {"0.001": 0, "0.01": 0}
    assert len(result["illustrative_exact_mismatches"]) == 1
    assert "not proof of byte identity" in result["interpretation"]


def test_manifest_rejects_invalid_duplicates_schema_and_bad_group_hash(manifest):
    bad = copy.deepcopy(manifest)
    bad["entries"][1]["seed"] = 0
    with pytest.raises(ValueError, match="missing/duplicate"):
        validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["entries"][0]["exact_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["entries"][0]["rms"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["splits"]["train"]["recording_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="contradicts"):
        validate_manifest(bad)
    bad = copy.deepcopy(manifest)
    bad["no_audio_or_weights_saved"] = False
    with pytest.raises(ValueError, match="unsafe"):
        validate_manifest(bad)


def test_compare_rejects_forged_or_incompatible_recipes(manifest):
    altered = copy.deepcopy(manifest)
    altered["data_recipe"] = "wrong-recipe"
    with pytest.raises(ValueError, match="unsafe"):
        compare_manifests(manifest, altered)
    altered = copy.deepcopy(manifest)
    altered["entries"][0]["split"] = "validation"
    with pytest.raises(ValueError):
        compare_manifests(manifest, altered)


def test_bounded_json_read_and_no_raw_audio_payload(tmp_path, manifest):
    manifest_path = tmp_path / "research-manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    assert _read(manifest_path)["splits"] == manifest["splits"]
    assert not any(key in manifest for key in ("waveform", "samples", "weights"))
    assert all("waveform" not in row and "samples" not in row
               for row in manifest["entries"])
    large = tmp_path / "oversized.json"
    large.write_bytes(b"x" * (256 * 1024 + 1))
    with pytest.raises(ValueError, match="oversized"):
        _read(large)
