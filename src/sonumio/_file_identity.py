"""Best-effort local file snapshot checks across path and descriptor APIs.

Windows os.fstat() may expose metadata with different precision or identity
fields from Path.lstat(). Avoid treating an unchanged local file as tampered,
while retaining exact path-to-path and descriptor-to-descriptor comparisons.

These checks protect ordinary local accidental swaps, not adversaries who can
change metadata and contents and then restore both between checks. An expected
SHA256 must be pinned separately when the source identity must be approved.
"""
from __future__ import annotations

import os


def same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare snapshots taken through the same metadata API."""
    return (
        left.st_dev, left.st_ino, left.st_size,
        left.st_mtime_ns, left.st_ctime_ns,
    ) == (
        right.st_dev, right.st_ino, right.st_size,
        right.st_mtime_ns, right.st_ctime_ns,
    )


def _windows_path_descriptor_matches(
    path_stat: os.stat_result, fd_stat: os.stat_result,
) -> bool:
    """Reconcile native Win32 path stat with CRT descriptor stat metadata."""
    return (path_stat.st_size == fd_stat.st_size
            and int(path_stat.st_mtime) == int(fd_stat.st_mtime))


def same_path_descriptor(path_stat: os.stat_result, fd_stat: os.stat_result) -> bool:
    """Compare independent path and descriptor metadata without Windows false alarms.

    On Windows, path-based stat normally uses native 100ns timestamps, while
    CRT descriptor-based fstat can report times rounded to whole seconds and
    may not report a matching inode/volume identifier. Both underlying
    snapshots are cross-checked again against their OWN API at EOF.
    """
    return (_windows_path_descriptor_matches(path_stat, fd_stat)
            if os.name == "nt" else same_snapshot(path_stat, fd_stat))


def same_path_identity(left: os.stat_result, right: os.stat_result) -> bool:
    """Check a stable path's reported file identity; zero IDs fall back to metadata."""
    if left.st_dev and right.st_dev and left.st_ino and right.st_ino:
        return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)
    return (left.st_size == right.st_size
            and left.st_mtime_ns == right.st_mtime_ns
            and left.st_ctime_ns == right.st_ctime_ns)
