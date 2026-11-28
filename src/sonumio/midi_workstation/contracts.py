"""Security boundaries and shared numerical contracts for symbolic audio projects.

This package is independent of trained models and external MIDI libraries.
The limits deliberately constrain hostile inputs before decoding or allocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any


class MidiWorkstationError(ValueError):
    """Invalid symbolic score, MIDI stream, or unsupported operation."""


class MidiLimitError(MidiWorkstationError):
    """Explicit resource budget exhausted before processing."""


class MidiFormatError(MidiWorkstationError):
    """A binary or interchange format violates its specification."""


class MidiIntegrityError(MidiWorkstationError):
    """A checksum, identity, or byte-snapshot invariant was violated."""


MAX_MIDI_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_TRACKS = 64
MAX_EVENTS = 120_000
MAX_EVENTS_PER_TRACK = 45_000
MAX_NOTES = 60_000
MAX_META_PAYLOAD = 256 * 1024
MAX_TICK = 0x0FFFFFFF
MAX_PPQ = 32767
MAX_TEMPO_CHANGES = 10_000
MAX_SCORE_TRACKS = 48
MAX_SCORE_NOTES = 48_000
MAX_TOTAL_NOTES = 80_000
MAX_OPERATION_ITEMS = 20_000
MAX_ANALYSIS_BINS = 4096
MAX_TEXT = 2048
MAX_LABEL = 256
MAX_FILENAME_BYTES = 256
MAX_RENDER_FRAMES = 120 * 192000
MAX_INTEGER = (1 << 63) - 1


def integer(value: object, name: str, minimum: int = 0, maximum: int = MAX_INTEGER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise MidiWorkstationError(f"{name}: expected integer in [{minimum}, {maximum}]")
    return value


def finite(value: object, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise MidiWorkstationError(f"{name}: expected finite number")
    if not minimum <= value <= maximum:
        raise MidiWorkstationError(f"{name}: outside [{minimum}, {maximum}]")
    return float(value)


def boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise MidiWorkstationError(f"{name}: expected boolean")
    return value


def text_value(value: object, name: str, maximum: int = MAX_LABEL) -> str:
    if type(value) is not str or len(value) > maximum:
        raise MidiWorkstationError(f"{name}: invalid bounded text")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise MidiWorkstationError(f"{name}: control characters are forbidden")
    return value


def raw_bytes(value: object, name: str, maximum: int = MAX_MIDI_BYTES) -> bytes:
    if type(value) is not bytes or len(value) > maximum:
        raise MidiWorkstationError(f"{name}: expected bounded bytes")
    return value


def digest(value: bytes) -> str:
    raw_bytes(value, "hash input")
    return hashlib.sha256(value).hexdigest()


def sha256(value: object, name: str = "SHA-256") -> str:
    if (type(value) is not str or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise MidiWorkstationError(f"{name}: expected lowercase SHA-256 hex")
    return value


def fraction(value: object, name: str, minimum: Fraction, maximum: Fraction) -> Fraction:
    if type(value) is not Fraction or not minimum <= value <= maximum:
        raise MidiWorkstationError(f"{name}: invalid rational value")
    return value


def checked_product(*counts: int, limit: int, name: str = "work") -> int:
    integer(limit, "work limit", 1)
    product = 1
    for count in counts:
        integer(count, name, 0)
        product *= count
        if product > limit:
            raise MidiLimitError(f"{name} exceeds {limit:,} unit limit")
    return product


def assert_keys(data: object, required: set[str], optional: set[str] | None = None,
                context: str = "object") -> dict[str, Any]:
    if type(data) is not dict or any(type(key) is not str for key in data):
        raise MidiFormatError(f"{context} must be a JSON object")
    keys = set(data)
    optional = set() if optional is None else optional
    if not required <= keys or keys - required - optional:
        raise MidiFormatError(f"{context} fields differ from the supported schema")
    return data


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise MidiFormatError(f"Duplicate JSON field: {key[:40]}")
        data[key] = value
    return data


def _reject_constant(value: str) -> None:
    raise MidiFormatError("Nonfinite JSON numbers are not supported")


def parse_json(raw: bytes, *, limit: int = MAX_JSON_BYTES) -> Any:
    raw_bytes(raw, "JSON", limit)
    if not raw:
        raise MidiFormatError("Empty JSON input")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        if isinstance(exc, MidiWorkstationError):
            raise
        raise MidiFormatError("Invalid strict UTF-8 JSON document") from exc


def canonical_json(value: Any, *, limit: int = MAX_JSON_BYTES) -> bytes:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (UnicodeError, ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise MidiFormatError("Cannot serialize strict JSON value") from exc
    raw_bytes(raw, "canonical JSON", limit)
    return raw


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    content: bytes
    sha256: str
    size: int

    def __post_init__(self) -> None:
        raw_bytes(self.content, "snapshot bytes")
        sha256(self.sha256)
        integer(self.size, "snapshot size", 0, MAX_MIDI_BYTES)
        if self.size != len(self.content) or digest(self.content) != self.sha256:
            raise MidiIntegrityError("Snapshot bytes differ from recorded digest")


def read_regular_file(path: str | Path, *, limit: int = MAX_MIDI_BYTES,
                      expected_sha256: str | None = None) -> FileSnapshot:
    """Open one bounded regular descriptor; refuse symlinks and changed snapshots.

    O_NOFOLLOW protects leaf symlinks where available. File identity is checked
    before and after reading. It is not a complete hostile-filesystem sandbox.
    """
    integer(limit, "file limit", 1, MAX_MIDI_BYTES)
    if expected_sha256 is not None:
        sha256(expected_sha256)
    path = Path(path)
    if path.is_symlink():
        raise MidiFormatError("Refusing symbolic-link input")
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise MidiFormatError("Unable to open input regular file") from exc
    with os.fdopen(fd, "rb") as file:
        first = os.fstat(file.fileno())
        if not stat.S_ISREG(first.st_mode):
            raise MidiFormatError("Input must be a regular file")
        if first.st_size > limit:
            raise MidiLimitError("Input file exceeds byte limit")
        content = file.read(limit + 1)
        last = os.fstat(file.fileno())
        if (first.st_dev, first.st_ino, first.st_size,
                first.st_mtime_ns, first.st_ctime_ns) != (
                last.st_dev, last.st_ino, last.st_size,
                last.st_mtime_ns, last.st_ctime_ns):
            raise MidiIntegrityError("Input file changed while reading")
    if len(content) > limit:
        raise MidiLimitError("Input file exceeds byte limit")
    actual = digest(content)
    if expected_sha256 is not None and actual != expected_sha256:
        raise MidiIntegrityError("Input SHA-256 does not match the pinned digest")
    return FileSnapshot(content, actual, len(content))
