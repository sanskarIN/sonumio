"""Small, local-only crash-observation journal for Sonumio generation jobs.

This is a *metadata ledger*, not a durable work queue: interrupted jobs are
marked failed at restart and are NEVER replayed. Only fixed fields and a
validated audio ID/URL survive, never prompts, error traces, generated
waveform bytes, arbitrary renderer result dictionaries, or model weights.

One process must own the journal path. The local filesystem is trusted not to
have a hostile writer replacing its parent directory while the API runs.
"""
from __future__ import annotations

import errno
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

VERSION = 1
MAX_RECORDS = 512
MAX_BYTES = 256 * 1024
_ID = re.compile(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")
_KINDS = frozenset({"generation", "scene"})
_STATES = frozenset({"queued", "running", "cancelling",
                     "succeeded", "failed", "cancelled"})
_SAFE_ERRORS = frozenset({
    "Generation failed; see local server logs",
    "Generation failed; invalid renderer result",
    "Generation interrupted by local server restart",
})


class JobJournalError(RuntimeError):
    """Cannot safely read or update the local job-state ledger."""


def recoverable_audio_result(result: object) -> dict | None:
    """Retain only the local audio reference, not arbitrary renderer output."""
    if not isinstance(result, dict):
        return None
    identifier = result.get("id")
    url = result.get("audio_url")
    if (not isinstance(identifier, str)
            or _ID.fullmatch(identifier) is None
            or url != f"/v1/generations/{identifier}/audio"):
        return None
    return {"id": identifier, "audio_url": url}


def _valid_timestamp(value: object) -> bool:
    return value is None or (type(value) is str and len(value) <= 64)


def _valid_record(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    identifier = row.get("id")
    kind, status = row.get("kind"), row.get("status")
    epoch = row.get("finished_epoch")
    if (not isinstance(identifier, str) or _ID.fullmatch(identifier) is None
            or kind not in _KINDS or status not in _STATES
            or not all(_valid_timestamp(row.get(k)) for k in (
                "created_at", "started_at", "finished_at"))):
        return False
    if not isinstance(row.get("created_at"), str):
        return False
    if epoch is not None and (type(epoch) not in (int, float)
                              or not math.isfinite(epoch) or epoch < 0):
        return False
    if status in {"succeeded", "failed", "cancelled"}:
        if row.get("finished_at") is None or epoch is None:
            return False
    elif epoch is not None or row.get("finished_at") is not None:
        return False
    if status == "failed" and row.get("error") not in _SAFE_ERRORS:
        return False
    if status != "failed" and row.get("error") is not None:
        return False
    result = row.get("result")
    if result is not None:
        if status != "succeeded" or recoverable_audio_result(result) != result:
            return False
    return True


class LocalJobJournal:
    """Atomic, bounded JSON snapshots, owned by one loopback-only API process."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.name in ("", ".", ".."):
            raise ValueError("job journal must be a file path")

    def _directory(self) -> None:
        parent = self.path.parent
        if parent.is_symlink():
            raise JobJournalError("Job journal directory must not be a symlink")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise JobJournalError("Cannot create local job journal directory") from exc
        if parent.is_symlink() or not parent.is_dir():
            raise JobJournalError("Invalid local job journal directory")

    def load(self) -> list[dict]:
        """Read no more than the configured byte limit from one regular file."""
        self._directory()
        try:
            expected = self.path.lstat()
        except FileNotFoundError:
            return []
        if not stat.S_ISREG(expected.st_mode) or expected.st_size > MAX_BYTES:
            raise JobJournalError("Unsafe or oversized local job journal")

        flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                 | getattr(os, "O_CLOEXEC", 0)
                 | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0))
        try:
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return []
            raise JobJournalError("Cannot safely open local job journal") from exc

        try:
            with os.fdopen(descriptor, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if (not stat.S_ISREG(opened.st_mode)
                        or opened.st_nlink != 1
                        or (expected.st_dev, expected.st_ino, expected.st_size,
                            expected.st_mtime_ns, expected.st_ctime_ns)
                        != (opened.st_dev, opened.st_ino, opened.st_size,
                            opened.st_mtime_ns, opened.st_ctime_ns)):
                    raise JobJournalError("Local job journal changed while opening")
                raw = handle.read(MAX_BYTES + 1)
        except JobJournalError:
            raise
        except OSError as exc:
            raise JobJournalError("Cannot read local job journal") from exc
        if len(raw) > MAX_BYTES:
            raise JobJournalError("Local job journal exceeds bounded size")
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise JobJournalError("Malformed local job journal") from exc
        if (not isinstance(parsed, dict) or parsed.get("version") != VERSION
                or not isinstance(parsed.get("jobs"), list)
                or len(parsed["jobs"]) > MAX_RECORDS):
            raise JobJournalError("Invalid local job journal format")
        seen: set[str] = set()
        for item in parsed["jobs"]:
            if not _valid_record(item) or item["id"] in seen:
                raise JobJournalError("Invalid or duplicate job journal record")
            seen.add(item["id"])
        return parsed["jobs"]

    def save(self, rows: list[dict]) -> None:
        """Write complete metadata through an fsynced temp and atomic replace."""
        if (not isinstance(rows, list) or len(rows) > MAX_RECORDS
                or any(not _valid_record(row) for row in rows)
                or len({row["id"] for row in rows}) != len(rows)):
            raise JobJournalError("Invalid records for local job journal")
        self._directory()
        payload = json.dumps(
            {"version": VERSION, "jobs": rows}, ensure_ascii=True,
            allow_nan=False, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        if len(payload) > MAX_BYTES:
            raise JobJournalError("Local job journal exceeds bounded size")

        staged: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.path.parent, prefix=".sonumio-jobs-",
                suffix=".tmp", delete=False,
            ) as stream:
                staged = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # os.replace replaces a leaf symlink rather than writing through
            # it. Never rely on a prior existence check for the destination.
            os.replace(staged, self.path)
            staged = None
        except OSError as exc:
            raise JobJournalError("Cannot commit local job journal") from exc
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)
