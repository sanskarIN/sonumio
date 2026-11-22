"""Regression checks for failure-safe API audio quota replacement."""
from __future__ import annotations

from io import BytesIO
import os
import uuid
import wave

import pytest

from sonumio import output_store
from sonumio.output_store import LocalAudioStore, OutputStoreError


def _wav(frames: int) -> bytes:
    stream = BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(b"\x01\x02" * frames)
    return stream.getvalue()


@pytest.mark.parametrize("failure_point", ["stage", "fsync", "link"])
def test_failed_replacement_does_not_evict_valid_preview(
    tmp_path, monkeypatch, failure_point,
):
    store = LocalAudioStore(
        tmp_path / "api", max_files=1,
        max_total_bytes=100, max_file_bytes=90,
    )
    original_audio = _wav(8)
    original_id = store.write(original_audio)

    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")

    if failure_point == "stage":
        monkeypatch.setattr(output_store.tempfile, "NamedTemporaryFile", fail)
    elif failure_point == "fsync":
        monkeypatch.setattr(output_store.os, "fsync", fail)
    else:
        monkeypatch.setattr(output_store.os, "link", fail)

    with pytest.raises(OSError, match="simulated storage failure"):
        store.write(_wav(10))
    assert store.read(original_id) == original_audio
    assert store.usage().files == 1
    assert store.usage().bytes == len(original_audio)
    assert not list(store.root.glob(".sonumio-*.tmp"))


def test_uuid_collision_exhaustion_does_not_evict_existing_preview(
    tmp_path, monkeypatch,
):
    store = LocalAudioStore(tmp_path / "api", max_files=1)
    old_audio = _wav(9)
    identifier = store.write(old_audio)
    monkeypatch.setattr(output_store, "uuid4", lambda: uuid.UUID(identifier))
    with pytest.raises(OutputStoreError, match="unique audio ID"):
        store.write(_wav(12))
    assert store.read(identifier) == old_audio
    assert not list(store.root.glob(".sonumio-*.tmp"))


def test_publish_new_preview_before_evicting_old_one(tmp_path, monkeypatch):
    store = LocalAudioStore(
        tmp_path / "api", max_files=1,
        max_total_bytes=100, max_file_bytes=90,
    )
    old_id = store.write(_wav(8))
    old_path = store.root / f"{old_id}.wav"
    original_link = os.link
    seen = []

    def inspect_link(source, destination):
        # The previous user-visible WAV must survive until the replacement
        # is successfully linked; no HTTP read can race this process lock.
        assert old_path.is_file()
        seen.append(destination)
        return original_link(source, destination)

    monkeypatch.setattr(output_store.os, "link", inspect_link)
    replacement = _wav(10)
    new_id = store.write(replacement)
    assert len(seen) == 1
    assert not old_path.exists()
    assert store.read(new_id) == replacement
    assert store.usage().files == 1
    assert store.usage().bytes == len(replacement)


def test_quota_eviction_error_rolls_back_unreported_new_preview(
    tmp_path, monkeypatch,
):
    store = LocalAudioStore(
        tmp_path / "api", max_files=1,
        max_total_bytes=100, max_file_bytes=90,
    )
    original_audio = _wav(8)
    old_id = store.write(original_audio)
    old_path = store.root / f"{old_id}.wav"
    original_unlink = type(old_path).unlink

    def fail_only_existing(self, *args, **kwargs):
        if self == old_path:
            raise OSError("simulated eviction failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(type(old_path), "unlink", fail_only_existing)
    with pytest.raises(OSError, match="simulated eviction failure"):
        store.write(_wav(10))

    # A newly linked preview must not silently survive a failed publication.
    assert store.read(old_id) == original_audio
    assert sorted(store.root.glob("*.wav")) == [old_path]
    assert not list(store.root.glob(".sonumio-*.tmp"))
