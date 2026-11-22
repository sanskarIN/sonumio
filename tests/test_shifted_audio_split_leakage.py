"""Shifted-overlap checks on decoded, resampled *actual* 8 kHz model crops."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import wave

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.audio_dataset import load_reviewed_splits
from research.dataset_manifest import ManifestError


RATE = 8_000


def _wav(samples: np.ndarray) -> bytes:
    data = np.asarray(samples, dtype="<i2")
    out = BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes(data.tobytes())
    return out.getvalue()


def _dataset(tmp_path, train: np.ndarray, evaluation: np.ndarray):
    folder = tmp_path / "audio"
    folder.mkdir()
    rows = []
    for idx, (split, samples) in enumerate(
        (("train", train), ("validation", evaluation))
    ):
        name = f"{split}.wav"
        data = _wav(samples)
        (folder / name).write_bytes(data)
        rows.append({
            "asset_id": f"fixture-{idx}",
            "source_url": "https://example.org/fixtures-only",
            "local_path": name,
            "sha256": sha256(data).hexdigest(),
            "caption": "rain" if idx == 0 else "wind",
            "language": "en",
            "group_id": f"independent-declared-group-{idx}",
            "split": split,
            "license": "CC0-1.0",
            "source_kind": "cc0",
            "rights_reviewed": True,
            "creator_permission": False,
            "voice_present": False,
            "sample_rate": RATE,
            "duration_seconds": len(samples) / RATE,
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return manifest, folder


def _source(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        -10_000, 10_000, size=4096, dtype=np.int16
    )


@pytest.mark.parametrize("shift", [1, 97, 240, 512, 768, 900])
def test_distinct_wav_files_with_shifted_recording_are_rejected(tmp_path, shift):
    source = _source(11)
    shifted = np.roll(source, shift)
    assert sha256(_wav(source)).digest() != sha256(_wav(shifted)).digest()
    manifest, root = _dataset(tmp_path, source, shifted)
    with pytest.raises(
        ManifestError,
        match=rf"line 2: shifted .* offset {shift} frames",
    ):
        load_reviewed_splits(manifest, root, clips_per_asset=1)


def test_gain_dc_polarity_and_shift_cross_split_rejected(tmp_path):
    source = _source(17)
    edited = np.rint(-0.5 * np.roll(source, 103) + 500).astype(np.int16)
    manifest, root = _dataset(tmp_path, source, edited)
    with pytest.raises(ManifestError, match=r"line 2: shifted .* offset 103 frames"):
        load_reviewed_splits(manifest, root, clips_per_asset=1)


def test_unrelated_clips_are_accepted_with_narrow_audit_report(tmp_path):
    manifest, root = _dataset(tmp_path, _source(0), _source(1))
    loaded, report = load_reviewed_splits(manifest, root, clips_per_asset=1)
    assert loaded["train"].clips.shape == (1, 1, 2048)
    assert loaded["validation"].clips.shape == (1, 1, 2048)
    assert report["shifted_overlap_indexed_clips"] == 2
    assert report["shifted_overlap_indexed_anchors"] > 0
    assert report["shifted_overlap_full_clip_comparisons"] >= 0
    assert "1024 frames / 128 ms" in report["shifted_cross_split_overlap_gate"]
    assert report["shifted_overlap_postings_examined"] >= 0
    assert report["shifted_overlap_max_posting_visits_per_clip"] == 100_000
    assert report["near_duplicate_or_semantic_similarity_checked"] is False



def test_long_shift_with_reverse_polarity_rejected_in_actual_crop(tmp_path):
    original = _source(23)
    edited = np.rint(-0.5 * np.roll(original, 816) + 600).astype(np.int16)
    manifest, root = _dataset(tmp_path, original, edited)
    with pytest.raises(ManifestError, match=r"line 2: shifted .* offset 816 frames"):
        load_reviewed_splits(manifest, root, clips_per_asset=1)
