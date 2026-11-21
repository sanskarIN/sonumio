"""Bounded, ephemeral WAV storage for Sonumio's loopback-only API.

Files live in a dedicated directory, separate from CLI exports. A local
server may restart without losing the ability to expire old API files.
This is not a multi-process or multi-user storage service.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import time
from typing import Callable
from uuid import uuid4


_ID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")
_TEMP_PREFIX = ".sonumio-"
_TEMP_SUFFIX = ".tmp"


class OutputStoreError(RuntimeError):
    """Output cannot be committed to local API storage."""


class OutputTooLargeError(OutputStoreError):
    """One WAV exceeds the configured size limit."""


@dataclass(frozen=True)
class StoreUsage:
    files: int
    bytes: int


class LocalAudioStore:
    """Thread-safe per-process WAV store with bounded size, count and age.

    Only files named as UUID WAVs inside root are managed. Reads return
    complete bytes under the same lock used for evictions, avoiding a TOCTOU
    race between a quota eviction and a file-backed HTTP response. Writes
    stage bytes in the same directory, then atomically link the completed
    file, never exposing partially written audio or clobbering an existing ID.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_files: int = 128,
        max_total_bytes: int = 256 * 1024 * 1024,
        max_file_bytes: int = 8 * 1024 * 1024,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
            raise ValueError("max_files must be a positive integer")
        for name, value in (("max_total_bytes", max_total_bytes),
                            ("max_file_bytes", max_file_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 44:
                raise ValueError(f"{name} must be an integer >= 44")
        if max_file_bytes > max_total_bytes:
            raise ValueError("max_file_bytes cannot exceed max_total_bytes")
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)) or not 1 <= ttl_seconds <= 604800:
            raise ValueError("ttl_seconds must be between 1 second and 7 days")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.root = Path(root)
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self.max_file_bytes = max_file_bytes
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._lock = threading.RLock()

    @staticmethod
    def valid_id(identifier: object) -> bool:
        return isinstance(identifier, str) and _ID.fullmatch(identifier) is not None

    def _directory(self) -> None:
        # A symlink at the API-owned boundary could redirect cleanup to a
        # caller's unrelated folder; refuse to work with such a directory.
        if self.root.is_symlink():
            raise OutputStoreError("API output directory must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise OutputStoreError("Invalid API output directory")

    def _scan(self, now: float) -> list[tuple[float, Path, int]]:
        """Expire owned files; return (mtime, path, size) oldest first."""
        result: list[tuple[float, Path, int]] = []
        for path in self.root.iterdir():
            if path.is_symlink():
                continue
            try:
                metadata = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            if path.name.startswith(_TEMP_PREFIX) and path.name.endswith(_TEMP_SUFFIX):
                if now - metadata.st_mtime >= self.ttl_seconds:
                    path.unlink(missing_ok=True)
                continue
            if path.suffix != ".wav" or not self.valid_id(path.stem):
                continue
            if (now - metadata.st_mtime >= self.ttl_seconds
                    or metadata.st_size > self.max_file_bytes):
                path.unlink(missing_ok=True)
                continue
            result.append((metadata.st_mtime, path, metadata.st_size))
        result.sort(key=lambda entry: (entry[0], entry[1].name))
        return result

    def _prune(self, now: float) -> list[tuple[float, Path, int]]:
        records = self._scan(now)
        total = sum(entry[2] for entry in records)
        while records and (len(records) > self.max_files or total > self.max_total_bytes):
            _, path, size = records.pop(0)
            path.unlink(missing_ok=True)
            total -= size
        return records

    def usage(self) -> StoreUsage:
        with self._lock:
            self._directory()
            records = self._prune(self._clock())
            return StoreUsage(len(records), sum(row[2] for row in records))

    def write(self, audio: bytes) -> str:
        if not isinstance(audio, bytes) or len(audio) < 44 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
            raise ValueError("Expected complete RIFF/WAVE bytes")
        if len(audio) > self.max_file_bytes or len(audio) > self.max_total_bytes:
            raise OutputTooLargeError("Generated WAV exceeds the local API output limit")
        with self._lock:
            self._directory()
            now = self._clock()
            records = self._prune(now)
            total = sum(row[2] for row in records)
            while records and (len(records) >= self.max_files
                               or total + len(audio) > self.max_total_bytes):
                _, old, size = records.pop(0)
                old.unlink(missing_ok=True)
                total -= size
            staged: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=self.root, prefix=_TEMP_PREFIX,
                    suffix=_TEMP_SUFFIX, delete=False,
                ) as handle:
                    staged = Path(handle.name)
                    handle.write(audio)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Record the managed lifetime before exposing the completed inode.
                os.utime(staged, (now, now))
                for _ in range(5):
                    identifier = str(uuid4())
                    destination = self.root / f"{identifier}.wav"
                    try:
                        # Link is atomic and fails rather than overwriting an
                        # existing file, unlike os.replace on UUID collision.
                        os.link(staged, destination)
                        return identifier
                    except FileExistsError:
                        continue
                raise OutputStoreError("Failed to allocate unique audio ID")
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)

    def read(self, identifier: str) -> bytes:
        if not self.valid_id(identifier):
            raise FileNotFoundError("Audio not found")
        with self._lock:
            self._directory()
            self._prune(self._clock())
            path = self.root / f"{identifier}.wav"
            if path.is_symlink():
                raise FileNotFoundError("Audio not found")
            try:
                with path.open("rb") as handle:
                    data = handle.read(self.max_file_bytes + 1)
            except (FileNotFoundError, IsADirectoryError) as exc:
                raise FileNotFoundError("Audio not found") from exc
            if len(data) > self.max_file_bytes:
                raise FileNotFoundError("Audio not found")
            return data

    def delete(self, identifier: object) -> bool:
        if not self.valid_id(identifier):
            return False
        with self._lock:
            self._directory()
            path = self.root / f"{identifier}.wav"
            if path.is_symlink():
                return False
            if not path.is_file():
                return False
            path.unlink()
            return True
