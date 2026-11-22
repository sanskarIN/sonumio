"""Adversarial local-file replacement tests for preview reads.

The output store is a loopback-only, single-process service. These tests
simulate an external writer replacing a UUID WAV between metadata checking
and opening the actual file descriptor. The API must not return the bytes
from an unintended filesystem object.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from sonumio import output_store
from sonumio.output_store import LocalAudioStore


def _wav(frames: int) -> bytes:
    from io import BytesIO
    import wave

    result = BytesIO()
    with wave.open(result, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24_000)
        writer.writeframes(b"\x01\x02" * frames)
    return result.getvalue()


def _swap_on_open(monkeypatch, target: Path, change) -> None:
    original_open = os.open
    swapped = False

    def intercepted(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == target and not swapped:
            swapped = True
            change()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(output_store.os, "open", intercepted)


def test_replacement_regular_file_is_not_returned(tmp_path, monkeypatch):
    store = LocalAudioStore(tmp_path / "api")
    identifier = store.write(_wav(20))
    target = store.root / f"{identifier}.wav"
    untrusted_audio = _wav(5)

    def replace():
        target.unlink()
        target.write_bytes(untrusted_audio)

    _swap_on_open(monkeypatch, target, replace)
    with pytest.raises(FileNotFoundError, match="Audio not found"):
        store.read(identifier)
    assert target.read_bytes() == untrusted_audio


def test_link_swap_between_lstat_and_open_cannot_escape_store(
    tmp_path, monkeypatch,
):
    store = LocalAudioStore(tmp_path / "api")
    identifier = store.write(_wav(8))
    target = store.root / f"{identifier}.wav"
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"not sonumio output, never served")
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("Filesystem does not permit symlinks")
    finally:
        probe.unlink(missing_ok=True)

    def replace():
        target.unlink()
        target.symlink_to(outside)

    _swap_on_open(monkeypatch, target, replace)
    with pytest.raises(FileNotFoundError, match="Audio not found"):
        store.read(identifier)
    assert outside.read_bytes() == b"not sonumio output, never served"
    assert target.is_symlink()


def test_fd_read_keeps_original_bytes_when_path_changes_after_open(
    tmp_path, monkeypatch,
):
    store = LocalAudioStore(tmp_path / "api")
    expected_audio = _wav(13)
    identifier = store.write(expected_audio)
    target = store.root / f"{identifier}.wav"
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"private bytes")
    original_open = os.open

    def change_after_open(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if Path(path) == target:
            target.unlink()
            target.write_bytes(b"other audio file on this path")
        return fd

    monkeypatch.setattr(output_store.os, "open", change_after_open)
    assert store.read(identifier) == expected_audio
    assert outside.read_bytes() == b"private bytes"
    assert target.read_bytes() == b"other audio file on this path"


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
    reason="No nonblocking FIFO support on this platform",
)
def test_fifo_swap_is_rejected_without_blocking(tmp_path, monkeypatch):
    store = LocalAudioStore(tmp_path / "api")
    identifier = store.write(_wav(4))
    target = store.root / f"{identifier}.wav"

    def replace():
        target.unlink()
        os.mkfifo(target)

    _swap_on_open(monkeypatch, target, replace)
    with pytest.raises(FileNotFoundError, match="Audio not found"):
        store.read(identifier)
    assert target.exists()


def test_regular_uuid_named_directory_is_not_read_as_audio(tmp_path):
    store = LocalAudioStore(tmp_path / "api")
    identifier = store.write(_wav(3))
    target = store.root / f"{identifier}.wav"
    target.unlink()
    target.mkdir()
    with pytest.raises(FileNotFoundError, match="Audio not found"):
        store.read(identifier)
