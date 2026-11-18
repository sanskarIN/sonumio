"""Bounded file digest, race protection and provenance consistency tests."""
from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import pytest

import sonumio.audio_core.hashing as hashing
from sonumio.audio_core.errors import (
    AudioContractError, AudioIntegrityError, AudioLimitError,
)
from sonumio.audio_core.hashing import digest_file


def test_digest_file_uses_exact_bytes_and_rejects_bad_budget(tmp_path):
    payload = b"sonumio-local-provenance\x00" * 100
    path = tmp_path / "ordinary.bin"
    path.write_bytes(payload)
    assert digest_file(path) == hashlib.sha256(payload).hexdigest()
    assert digest_file(path, max_bytes=len(payload)) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(AudioLimitError, match="budget"):
        digest_file(path, max_bytes=len(payload) - 1)
    for bad in (True, 0, -1, 128 * 1024 * 1024 + 1, 2.5):
        with pytest.raises(AudioContractError, match="byte cap"):
            digest_file(path, max_bytes=bad)


def test_digest_file_rejects_directory_and_missing_files(tmp_path):
    with pytest.raises(AudioContractError, match="ordinary"):
        digest_file(tmp_path)
    with pytest.raises(AudioContractError, match="ordinary"):
        digest_file(tmp_path / "absent")


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX O_NOFOLLOW required")
def test_digest_file_rejects_symlink_swap_before_descriptor_open(tmp_path, monkeypatch):
    source = tmp_path / "allowed.bin"
    private = tmp_path / "private.bin"
    source.write_bytes(b"original")
    private.write_bytes(b"another-recording")
    original_open = os.open

    def swap(path, flags, *args, **kwargs):
        if str(path) == str(source):
            source.unlink()
            source.symlink_to(private)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(hashing.os, "open", swap)
    with pytest.raises(AudioContractError, match="ordinary"):
        digest_file(source)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFOs required")
def test_digest_file_rejects_fifo_without_waiting_for_writer(tmp_path):
    fifo = tmp_path / "file.fifo"
    os.mkfifo(fifo)
    with pytest.raises(AudioContractError, match="ordinary"):
        digest_file(fifo)


@pytest.mark.skipif(
    not (hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK")),
    reason="POSIX nonblocking open required",
)
def test_digest_file_rejects_raced_fifo_without_blocking(tmp_path, monkeypatch):
    source = tmp_path / "raced.bin"
    source.write_bytes(b"hello")
    original_open = os.open

    def swap(path, flags, *args, **kwargs):
        if str(path) == str(source):
            source.unlink()
            os.mkfifo(source)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(hashing.os, "open", swap)
    with pytest.raises(AudioContractError, match="descriptor"):
        digest_file(source)


def test_digest_file_rejects_midread_metadata_change(tmp_path, monkeypatch):
    source = tmp_path / "updated-during-digest.bin"
    source.write_bytes(b"stable-at-start")
    original_fstat = os.fstat
    count = 0

    def drift(fd):
        nonlocal count
        observed = original_fstat(fd)
        count += 1
        if count == 2:
            return SimpleNamespace(
                st_mode=observed.st_mode,
                st_size=observed.st_size,
                st_mtime_ns=observed.st_mtime_ns + 1,
                st_ctime_ns=observed.st_ctime_ns,
            )
        return observed

    monkeypatch.setattr(hashing.os, "fstat", drift)
    with pytest.raises(AudioIntegrityError, match="changed"):
        digest_file(source)
    assert count == 2


def test_digest_file_rejects_postopen_file_growth_past_budget(tmp_path, monkeypatch):
    path = tmp_path / "growing-file.bin"
    path.write_bytes(b"A" * 128)
    # Simulate the on-disk contents growing immediately after preflight.
    original_open = os.open
    def grow(path_arg, flags, *args, **kwargs):
        fd = original_open(path_arg, flags, *args, **kwargs)
        if str(path_arg) == str(path):
            with open(path, "ab") as writer:
                writer.write(b"B" * 32)
        return fd
    monkeypatch.setattr(hashing.os, "open", grow)
    with pytest.raises(AudioLimitError, match="budget"):
        digest_file(path, max_bytes=144)
