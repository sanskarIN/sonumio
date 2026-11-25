"""Bounded ordinary-file snapshots for local reviewed-corpus ingestion.

This is defense in depth for trusted local researchers. It is not a
hostile-filesystem sandbox, proof of recording rights, or a data downloader.
Expected content hashes remain the authority for approved audio bytes.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat

from sonumio._file_identity import same_path_descriptor, same_snapshot


class SnapshotError(ValueError):
    """An ordinary local file could not be read within the declared budget."""


def _check_root(path: Path, root: Path | None) -> None:
    if root is None:
        return
    if not path.resolve(strict=False).is_relative_to(root):
        raise SnapshotError("File resolves outside the configured audio root")


def read_local_snapshot(
    path: str | Path, *, max_bytes: int, root: Path | None = None,
) -> bytes:
    """Read exactly one bounded regular-file snapshot with identity checks.

    Refuse leaf symlinks and nonregular files before opening. Use
    O_NOFOLLOW/O_NONBLOCK where supported and compare path/descriptor
    snapshots with the project's platform-specific identity helper.
    Read at most max_bytes + 1, even if a file grows after preflight.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Snapshot byte cap must be a positive integer")
    source = Path(path)
    try:
        _check_root(source, root)
        before = source.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise SnapshotError("Expected ordinary local file, not a link or special file")
        if before.st_size > max_bytes:
            raise SnapshotError("Local file exceeds snapshot byte cap")
        flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                 | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        fd = os.open(source, flags)
        try:
            descriptor = os.fstat(fd)
            if not stat.S_ISREG(descriptor.st_mode) or not same_path_descriptor(before, descriptor):
                raise SnapshotError("Local file changed during open")
            if descriptor.st_size > max_bytes:
                raise SnapshotError("Local file exceeds snapshot byte cap")
            opened_path = source.lstat()
            _check_root(source, root)
            if not stat.S_ISREG(opened_path.st_mode) or not same_snapshot(before, opened_path):
                raise SnapshotError("Local file changed during open")
            # os.read avoids prefetch and permits explicit bounded loop reads
            # even on filesystems that return a short successful read.
            data = bytearray()
            while len(data) <= max_bytes:
                part = os.read(fd, min(1 << 20, max_bytes + 1 - len(data)))
                if not part:
                    break
                data.extend(part)
            if len(data) > max_bytes:
                raise SnapshotError("Local file exceeds snapshot byte cap")
            after = os.fstat(fd)
            at_path = source.lstat()
            _check_root(source, root)
            if (len(data) != descriptor.st_size
                    or not same_snapshot(descriptor, after)
                    or not stat.S_ISREG(at_path.st_mode)
                    or not same_snapshot(before, at_path)
                    or not same_path_descriptor(at_path, after)):
                raise SnapshotError("Local file changed during snapshot")
            return bytes(data)
        finally:
            os.close(fd)
    except (OSError, RuntimeError) as exc:
        raise SnapshotError("Cannot read ordinary local file snapshot") from exc
