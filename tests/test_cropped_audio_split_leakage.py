"""Different WAV recordings can still leak identical cropped neural inputs."""
from __future__ import annotations

import hashlib
import io
import json
import wave

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.audio_dataset import _cropped_clip_digest, load_reviewed_splits
from research.dataset_manifest import ManifestError


RATE = 24_000


def _wav(samples: np.ndarray) -> bytes:
    with io.BytesIO() as out:
        with wave.open(out, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(RATE)
            writer.writeframes(np.asarray(samples, dtype="<i2").tobytes())
        return out.getvalue()


def _manifest(tmp_path, training: bytes, validation: bytes):
    root = tmp_path / "audio"
    root.mkdir()
    records = []
    for i, (split, content) in enumerate(
        (("train", training), ("validation", validation))
    ):
        name = f"{split}.wav"
        (root / name).write_bytes(content)
        with wave.open(io.BytesIO(content), "rb") as recorder:
            duration = recorder.getnframes() / recorder.getframerate()
        records.append({
            "asset_id": f"asset-{i}",
            "source_url": "https://example.org/fixture-only",
            "local_path": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "caption": "rain" if i == 0 else "wind",
            "language": "en",
            "group_id": f"original-source-{i}",
            "split": split,
            "license": "CC0-1.0",
            "source_kind": "cc0",
            "rights_reviewed": True,
            "creator_permission": False,
            "voice_present": False,
            "sample_rate": RATE,
            "duration_seconds": duration,
        })
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path, root


def _source() -> np.ndarray:
    t = np.arange(RATE // 2, dtype=np.float64) / RATE
    return np.rint(8000 * np.sin(2 * np.pi * 321 * t)).astype("<i2")


def test_different_full_recordings_with_shared_cropped_input_are_rejected(tmp_path):
    original = _source()
    modified = original.copy()
    # The 8-kHz training crop uses a centered window near 2864..9136;
    # changing the last 800 source samples preserves that neural input.
    modified[-800:] = -modified[-800:]
    first, second = _wav(original), _wav(modified)
    assert first != second
    assert hashlib.sha256(first).digest() != hashlib.sha256(second).digest()
    manifest, root = _manifest(tmp_path, first, second)
    with pytest.raises(ManifestError, match=r"line 2: resampled model-input clip duplicates line 1"):
        load_reviewed_splits(manifest, root, clips_per_asset=1)


def test_different_cropped_inputs_are_accepted(tmp_path):
    original = _source()
    modified = original.copy()
    modified[4200:5400] = -modified[4200:5400]
    manifest, root = _manifest(tmp_path, _wav(original), _wav(modified))
    result, report = load_reviewed_splits(manifest, root, clips_per_asset=1)
    assert not np.array_equal(result["train"].clips, result["validation"].clips)
    assert report["distinct_model_input_clip_hashes"] == 2
    assert report["within_split_repeated_model_input_clips"] == 0
    assert report["cross_split_crop_duplicate_gate"] == "exact float32 8 kHz model-input PCM"
    assert report["near_duplicate_or_semantic_similarity_checked"] is False


def test_within_split_duplicate_crops_do_not_fake_recording_count(tmp_path):
    # Repeated selections from short files stay allowed within their split;
    # the loader reports this instead of pretending they're independent.
    original = _source()[:6144]  # exactly one complete 8-kHz crop
    altered = original.copy()
    altered[2500:3500] = -altered[2500:3500]
    manifest, root = _manifest(tmp_path, _wav(original), _wav(altered))
    loaded, report = load_reviewed_splits(manifest, root, clips_per_asset=3)
    assert loaded["train"].clips.shape == (3, 1, 2048)
    assert report["within_split_repeated_model_input_clips"] == 4
    assert report["distinct_model_input_clip_hashes"] == 2
    assert report["asset_counts"]["train"] == 1


def test_gain_and_dc_changed_model_input_cannot_cross_data_splits(tmp_path):
    # New bytes, new full decoded-audio digest and new exact resampled hashes:
    # the same aligned physical waveform was merely gained and DC-offset.
    original = _source()
    edited = np.rint(original.astype(np.float64) * 0.5 + 1000).astype("<i2")
    assert not np.array_equal(original, edited)
    manifest, root = _manifest(tmp_path, _wav(original), _wav(edited))
    with pytest.raises(ManifestError, match=r"line 2: aligned gain/DC/polarity-invariant model-input overlap"):
        load_reviewed_splits(manifest, root, clips_per_asset=1)


def test_distinct_model_inputs_report_narrow_aligned_overlap_audit(tmp_path):
    original = _source()
    modified = original.copy()
    modified[4200:5400] = -modified[4200:5400]
    manifest, root = _manifest(tmp_path, _wav(original), _wav(modified))
    loaded, report = load_reviewed_splits(manifest, root, clips_per_asset=1)
    assert report["aligned_overlap_indexed_clips"] == 2
    assert report["aligned_overlap_full_clip_comparisons"] >= 0
    assert "aligned centered abs(correlation)" in report["aligned_cross_split_overlap_gate"]
    # This does not establish absence of shifted/remixed/semantic duplicates.
    assert report["near_duplicate_or_semantic_similarity_checked"] is False
    assert loaded["train"].clips.shape[0] == 1



def test_crop_fingerprint_checks_exact_shape_dtype_and_nonfinite_inputs():
    good = np.zeros(2048, dtype=np.float32)
    assert _cropped_clip_digest(good) == _cropped_clip_digest(good.copy())
    bad = good.copy()
    bad[0] = np.nan
    for wrong in (np.zeros(200, dtype=np.float32),
                  np.zeros(2048, dtype=np.float64), bad):
        with pytest.raises(ManifestError, match="invalid resampled clip"):
            _cropped_clip_digest(wrong)
