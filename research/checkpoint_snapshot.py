"""Bounded, single-snapshot loading for PRIVATE experimental neural checkpoints.

A checkpoint's SHA256, ZIP headers and numerical tensors must all come from
*the same bytes*. Reopening a pathname after its digest check permits a local
writer to substitute different bytes before NumPy loads the tensors.

This is defense in depth for local research artifacts, not a signature or a
substitute for trusted model provenance, audited audio rights, or sandboxing.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

MAX_WEIGHTS_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def read_regular_snapshot(path: str | Path, *, max_bytes: int) -> bytes:
    """Read one bounded regular-file identity, rejecting swaps and symlinks.

    Check the opened descriptor both before and after reading, then confirm
    the pathname still addresses that same identity. O_NONBLOCK avoids hanging
    if an adversary swaps an ordinary file for a FIFO on supported platforms.
    """
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_WEIGHTS_BYTES:
        raise ValueError("Invalid checkpoint snapshot size limit")
    filename = Path(path)
    try:
        if filename.parent.is_symlink():
            raise ValueError("Checkpoint parent directory must not be a symlink")
        expected = filename.lstat()
        if (not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1
                or not 0 < expected.st_size <= max_bytes):
            raise ValueError("Unsafe or oversized checkpoint file")
        flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                 | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0))
        fd = os.open(filename, flags)
        with os.fdopen(fd, "rb") as source:
            opened = os.fstat(source.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                    or _identity(opened) != _identity(expected)):
                raise ValueError("Checkpoint changed before reading")
            data = source.read(max_bytes + 1)
            after = os.fstat(source.fileno())
        if (len(data) != expected.st_size or len(data) > max_bytes
                or _identity(after) != _identity(opened)
                or _identity(filename.lstat()) != _identity(opened)):
            raise ValueError("Checkpoint changed during reading")
        return data
    except (OSError, OverflowError) as exc:
        raise ValueError("Checkpoint file cannot be safely read") from exc


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    output: dict = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate checkpoint manifest field")
        output[key] = value
    return output


def _no_nonfinite_constant(value: str) -> None:
    raise ValueError("Non-finite checkpoint manifest value")


def read_checkpoint_manifest(path: str | Path) -> dict:
    """Bounded, strict JSON manifest parsing without duplicate/NaN ambiguity."""
    blob = read_regular_snapshot(path, max_bytes=MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(
            blob.decode("utf-8"), object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Malformed checkpoint JSON manifest") from exc
    if type(manifest) is not dict:
        raise ValueError("Checkpoint manifest must be a JSON object")
    return manifest


def read_verified_weights(path: str | Path, expected_sha256: str,
                          *, max_bytes: int = MAX_WEIGHTS_BYTES) -> bytes:
    """Digest-check the snapshot exactly once; callers must never reopen path."""
    if type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("Invalid checkpoint SHA256 declaration")
    blob = read_regular_snapshot(path, max_bytes=max_bytes)
    if hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    return blob


def verified_npz_bytes(blob: bytes) -> io.BytesIO:
    """Return a fresh seekable handle over the already validated bytes.

    The caller should pass the same `blob` to validate_numeric_archive, then
    use `np.load(verified_npz_bytes(blob), allow_pickle=False)`; no new file IO.
    """
    if type(blob) is not bytes or not 0 < len(blob) <= MAX_WEIGHTS_BYTES:
        raise ValueError("Invalid bounded checkpoint archive snapshot")
    return io.BytesIO(blob)
