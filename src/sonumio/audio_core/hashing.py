"""Stable local audio fingerprints and strict JSON provenance checks.

Hashes are for reproducibility and tamper detection, not proof that source
copyright, personal consent, or model training permission is valid.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

import numpy as np

from .contracts import AudioBuffer, check_audio, MAX_BYTES
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError

_SHA = re.compile(r"^[0-9a-f]{64}$")


def strict_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise AudioContractError("Expected lowercase 64-character SHA256")
    return value


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    data = {}
    for key, value in pairs:
        if key in data:
            raise AudioContractError("Duplicate JSON object property")
        data[key] = value
    return data


def _valid_json_data(value: Any, depth: int = 0) -> Any:
    """Convert only primitive JSON-safe values with bounded structural depth."""
    if depth > 16:
        raise AudioLimitError("JSON provenance nesting exceeds limit")
    if value is None or type(value) in (bool, str, int):
        if isinstance(value, str) and len(value) > 16384:
            raise AudioLimitError("JSON metadata string too long")
        if type(value) is int and abs(value) > 10**50:
            raise AudioLimitError("JSON metadata integer too large")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise AudioContractError("Nonfinite JSON metadata")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 10_000:
            raise AudioLimitError("JSON metadata list exceeds limit")
        return [_valid_json_data(x, depth + 1) for x in value]
    if type(value) is dict:
        if len(value) > 1_000:
            raise AudioLimitError("JSON metadata object exceeds limit")
        result = {}
        for key, val in value.items():
            if not isinstance(key, str):
                raise AudioContractError("JSON metadata keys must be strings")
            if key in ("__proto__", "__class__", "__dict__"):
                raise AudioContractError("Reserved JSON metadata key")
            result[key] = _valid_json_data(val, depth + 1)
        return result
    raise AudioContractError("Only primitive JSON metadata is allowed")


def canonical_json_bytes(metadata: Any, *, max_bytes: int = 2_000_000) -> bytes:
    if type(max_bytes) is not int or not 1 <= max_bytes <= 8_000_000:
        raise AudioContractError("Invalid JSON metadata budget")
    obj = _valid_json_data(metadata)
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                     sort_keys=True, allow_nan=False).encode("utf-8")
    if len(raw) > max_bytes:
        raise AudioLimitError("Canonical JSON exceeds configured byte budget")
    return raw


def parse_canonical_json(raw: bytes, *, max_bytes: int = 2_000_000) -> Any:
    if not isinstance(raw, bytes) or not 0 < len(raw) <= max_bytes:
        raise AudioContractError("Invalid bounded UTF8 JSON record")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys,
                         parse_constant=lambda x: (_ for _ in ()).throw(
                             AudioContractError("Nonfinite JSON constant")))
    except (ValueError, UnicodeError) as exc:
        raise AudioContractError("Malformed provenance JSON") from exc
    _valid_json_data(obj)
    return obj


def digest_audio(buffer: AudioBuffer | np.ndarray, *,
                 sample_rate: int | None = None,
                 include_geometry: bool = True) -> str:
    """Stable SHA256 over little-endian float32 PCM and optional clock geometry.

    Arrays must be channel-first; the digest intentionally distinguishes
    1-D vs 2-D sample geometry only through normalized mono topology.
    """
    if isinstance(buffer, AudioBuffer):
        samples, rate = buffer.samples, buffer.sample_rate
    else:
        if not isinstance(buffer, np.ndarray):
            raise AudioContractError("Expected AudioBuffer or NumPy array")
        samples = buffer.reshape(1, -1) if buffer.ndim == 1 else buffer
        check_audio(samples)
        rate = sample_rate
    if not isinstance(samples, np.ndarray) or samples.ndim != 2:
        raise AudioContractError("Audio hash requires channel-first frames")
    if include_geometry:
        from .contracts import check_rate
        check_rate(rate)
    h = hashlib.sha256()
    h.update(b"sonumio-audio-pcm-v1\x00" if include_geometry
             else b"sonumio-audio-samples-v1\x00")
    if include_geometry:
        h.update(int(rate).to_bytes(4, "little"))
        h.update(int(samples.shape[0]).to_bytes(2, "little"))
        h.update(int(samples.shape[1]).to_bytes(8, "little"))
    h.update(np.ascontiguousarray(samples, dtype="<f4").tobytes())
    return h.hexdigest()


def digest_file(path: str | Path, *, max_bytes: int = 128 * 1024 * 1024) -> str:
    """Hash one bounded regular file descriptor, not a potentially swapped path.

    A best-effort metadata consistency check detects many concurrent writes,
    but mtime/ctime checks do NOT establish a forensic immutable snapshot
    against hostile in-place modifications. Provenance still needs access
    controls, verified captured bytes and human authorization.
    """
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        raise AudioContractError("Invalid file digest byte cap")
    file = Path(path)
    if not file.is_file() or file.is_symlink():
        raise AudioContractError("Expected existing ordinary local file")
    # O_NOFOLLOW closes the symlink-swap gap where supported. O_NONBLOCK
    # avoids hanging if an attacker swaps a regular file path for a FIFO.
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0) |
             getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        fd = os.open(file, flags)
    except OSError as exc:
        raise AudioContractError("Cannot open ordinary local file") from exc
    h = hashlib.sha256()
    length = 0
    with os.fdopen(fd, "rb") as stream:
        original = os.fstat(stream.fileno())
        if not stat.S_ISREG(original.st_mode):
            raise AudioContractError("Expected ordinary local file descriptor")
        if original.st_size > max_bytes:
            raise AudioLimitError("File exceeds digest budget")
        while True:
            data = stream.read(min(1024 * 1024, max_bytes - length + 1))
            if not data:
                break
            length += len(data)
            if length > max_bytes:
                raise AudioLimitError("File grew beyond digest budget")
            h.update(data)
        final = os.fstat(stream.fileno())
        if (length != original.st_size or
                (original.st_size, original.st_mtime_ns, original.st_ctime_ns)
                != (final.st_size, final.st_mtime_ns, final.st_ctime_ns)):
            raise AudioIntegrityError(
                "File changed during SHA256 digest; retry with a stable snapshot"
            )
    return h.hexdigest()


def checksum_record(metadata: dict, *, payload_sha256: str) -> dict:
    """Return a standalone, canonical metadata record bound to a payload SHA.

    No cryptographic authentication or signed identity is implied by a hash.
    """
    strict_sha256(payload_sha256)
    if type(metadata) is not dict:
        raise AudioContractError("Expected primitive provenance object")
    if "record_sha256" in metadata or "payload_sha256" in metadata:
        raise AudioContractError("Reserved checksum fields")
    clean = _valid_json_data(metadata)
    body = dict(clean, payload_sha256=payload_sha256)
    body["record_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return body


def verify_record(record: dict, *, expected_payload: str | None = None) -> dict:
    if type(record) is not dict or "record_sha256" not in record:
        raise AudioIntegrityError("Missing provenance digest")
    body = dict(record)
    claimed = strict_sha256(body.pop("record_sha256"))
    payload = strict_sha256(body.get("payload_sha256"))
    if expected_payload is not None and payload != strict_sha256(expected_payload):
        raise AudioIntegrityError("Payload checksum does not match expected bytes")
    if hashlib.sha256(canonical_json_bytes(body)).hexdigest() != claimed:
        raise AudioIntegrityError("Provenance metadata was altered")
    return body
