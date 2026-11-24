"""Reject post-open WAV tampering even if metadata was initially valid."""
from __future__ import annotations

from io import BytesIO
import os
import wave

import pytest

from sonumio import output_store
from sonumio.output_store import LocalAudioStore


def _wav(frames: int) -> bytes:
    stream = BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(b"\x02\x01" * frames)
    return stream.getvalue()


def test_after_read_path_swap_is_detected(tmp_path, monkeypatch):
    store = LocalAudioStore(tmp_path / "api")
    identifier = store.write(_wav(64))
    target = store.root / f"{identifier}.wav"
    original_fstat = os.fstat
    calls = 0

    def swap_on_post_read_check(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.unlink()
            target.write_bytes(_wav(8))
        return original_fstat(fd)

    monkeypatch.setattr(output_store.os, "fstat", swap_on_post_read_check)
    with pytest.raises(FileNotFoundError, match="Audio not found"):
        store.read(identifier)
    assert calls >= 2


def test_in_place_rewrite_during_read_is_detected(tmp_path, monkeypatch):
    store = LocalAudioStore(tmp_path / "api")
    identifier = store.write(_wav(64))
    target = store.root / f"{identifier}.wav"
    initial = target.stat()
    original_fstat = os.fstat
    calls = 0

    def mutate_on_post_read_check(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            payload = bytearray(target.read_bytes())
            payload[-1] ^= 1
            target.write_bytes(payload)
            os.utime(target, ns=(initial.st_atime_ns, initial.st_mtime_ns + 2_000_000_000))
        return original_fstat(fd)

    monkeypatch.setattr(output_store.os, "fstat", mutate_on_post_read_check)
    with pytest.raises(FileNotFoundError, match="Audio not found"):
        store.read(identifier)
    assert calls >= 2


def test_stable_audio_is_accepted_with_final_descriptor_check(tmp_path):
    store = LocalAudioStore(tmp_path / "api")
    audio = _wav(64)
    identifier = store.write(audio)
    assert store.read(identifier) == audio


def test_path_replacement_after_descriptor_read_never_serves_old_bytes(
    tmp_path, monkeypatch,
):
    """Path identity must still be checked after the fd has been closed."""
    store = LocalAudioStore(tmp_path / "api")
    expected = _wav(64)
    identifier = store.write(expected)
    target = store.root / f"{identifier}.wav"
    archived = store.root / "archived-original"
    path_type = type(target)
    real_lstat = path_type.lstat
    lookups = 0

    def replace_on_final_path_check(self, *args, **kwargs):
        nonlocal lookups
        if self == target:
            lookups += 1
            if lookups == 2:
                target.replace(archived)
                target.write_bytes(expected)  # same bytes, different inode
        return real_lstat(self, *args, **kwargs)

    monkeypatch.setattr(path_type, "lstat", replace_on_final_path_check)
    with pytest.raises(FileNotFoundError, match="Audio not found"):
        store.read(identifier)
    assert lookups >= 2
    assert archived.read_bytes() == expected
