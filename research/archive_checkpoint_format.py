"""Bounded raw numeric checkpoint format with no executable deserialization.

The complete file SHA256 is checked before decoding by the caller. Every
tensor descriptor must match a schema derived from a freshly constructed
model and optimizer. Untrusted shapes never control array allocations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np

from research.corpus_archive_format import (
    ArchiveError, canonical_json, parse_canonical, digest, require_sha256,
)

MAGIC = b"STCP0001"
HEADER_LENGTH = struct.Struct("<I")
MAX_HEADER_BYTES = 256 * 1024
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 4096
_DTYPES = {"f32": np.dtype("<f4"), "u8": np.dtype("uint8")}


@dataclass(frozen=True, slots=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self):
        if self.dtype not in _DTYPES or not isinstance(self.shape, tuple) or len(self.shape) > 8:
            raise ArchiveError("Unsupported checkpoint tensor specification")
        if any(type(x) is not int or x <= 0 for x in self.shape):
            raise ArchiveError("Invalid checkpoint tensor dimensions")
        if self.nbytes > MAX_CHECKPOINT_BYTES:
            raise ArchiveError("Checkpoint tensor exceeds byte budget")

    @property
    def wire_dtype(self) -> np.dtype:
        return _DTYPES[self.dtype]

    @property
    def native_dtype(self) -> np.dtype:
        return self.wire_dtype.newbyteorder("=")

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape)*self.wire_dtype.itemsize


def _validate_specs(specs: dict[str, TensorSpec]) -> None:
    if not isinstance(specs, dict) or not 1 <= len(specs) <= MAX_TENSORS:
        raise ArchiveError("Invalid checkpoint tensor schema")
    for name, spec in specs.items():
        if (type(name) is not str or not name or len(name) > 256
                or not isinstance(spec, TensorSpec)):
            raise ArchiveError("Invalid checkpoint tensor name or specification")
    if sum(spec.nbytes for spec in specs.values()) + MAX_HEADER_BYTES + 12 > MAX_CHECKPOINT_BYTES:
        raise ArchiveError("Checkpoint schema exceeds byte budget")


def maximum_encoded_bytes(specs: dict[str, TensorSpec]) -> int:
    _validate_specs(specs)
    return len(MAGIC) + HEADER_LENGTH.size + MAX_HEADER_BYTES + sum(
        spec.nbytes for spec in specs.values()
    )


def _exact_array(value: object, spec: TensorSpec) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.shape != spec.shape
            or value.dtype != spec.native_dtype):
        raise ArchiveError("Checkpoint tensor differs from expected shape or dtype")
    if spec.dtype == "f32" and not np.isfinite(value).all():
        raise ArchiveError("Checkpoint contains nonfinite tensor")
    return value


def encode_checkpoint(
    metadata: dict, arrays: dict[str, np.ndarray], specs: dict[str, TensorSpec],
) -> bytes:
    _validate_specs(specs)
    if not isinstance(metadata, dict) or not isinstance(arrays, dict) or set(arrays) != set(specs):
        raise ArchiveError("Invalid checkpoint metadata or tensor names")
    descriptors = []
    payloads = []
    offset = 0
    for name in sorted(specs):
        spec = specs[name]
        arr = _exact_array(arrays[name], spec)
        payload = np.asarray(arr, dtype=spec.wire_dtype, order="C").tobytes(order="C")
        if len(payload) != spec.nbytes:
            raise ArchiveError("Checkpoint tensor byte count mismatch")
        descriptors.append({
            "name": name, "dtype": spec.dtype, "shape": list(spec.shape),
            "offset": offset, "length": len(payload), "sha256": digest(payload),
        })
        offset += len(payload)
        payloads.append(payload)
    header = canonical_json({"metadata": metadata, "tensors": descriptors})
    if not 1 <= len(header) <= MAX_HEADER_BYTES:
        raise ArchiveError("Checkpoint header exceeds byte budget")
    result = b"".join([MAGIC, HEADER_LENGTH.pack(len(header)), header, *payloads])
    if len(result) > MAX_CHECKPOINT_BYTES:
        raise ArchiveError("Checkpoint exceeds file byte budget")
    return result


def _bounded_header(blob: bytes) -> tuple[dict, int]:
    if type(blob) is not bytes or not len(MAGIC)+HEADER_LENGTH.size < len(blob) <= MAX_CHECKPOINT_BYTES:
        raise ArchiveError("Invalid bounded checkpoint byte snapshot")
    if blob[:len(MAGIC)] != MAGIC:
        raise ArchiveError("Unsupported checkpoint magic")
    header_size, = HEADER_LENGTH.unpack_from(blob, len(MAGIC))
    if not 1 <= header_size <= MAX_HEADER_BYTES:
        raise ArchiveError("Invalid checkpoint header byte length")
    start = len(MAGIC)+HEADER_LENGTH.size
    stop = start+header_size
    if stop >= len(blob):
        raise ArchiveError("Truncated checkpoint header or payload")
    header = parse_canonical(blob[start:stop])
    if (not isinstance(header, dict) or set(header) != {"metadata", "tensors"}
            or not isinstance(header["metadata"], dict) or not isinstance(header["tensors"], list)
            or not 1 <= len(header["tensors"]) <= MAX_TENSORS):
        raise ArchiveError("Invalid checkpoint header fields")
    return header, stop


def read_checkpoint_metadata(blob: bytes) -> dict:
    """Read bounded canonical metadata only, not a verified tensor state."""
    return _bounded_header(blob)[0]["metadata"]


def decode_checkpoint(
    blob: bytes, specs: dict[str, TensorSpec],
) -> tuple[dict, dict[str, np.ndarray]]:
    _validate_specs(specs)
    header, payload_start = _bounded_header(blob)
    descriptors = header["tensors"]
    if len(descriptors) != len(specs):
        raise ArchiveError("Unexpected checkpoint tensor count")
    offset = 0
    arrays = {}
    for expected_name, descriptor in zip(sorted(specs), descriptors, strict=True):
        spec = specs[expected_name]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "name", "dtype", "shape", "offset", "length", "sha256",
        }:
            raise ArchiveError("Invalid checkpoint tensor descriptor fields")
        shape = descriptor["shape"]
        if (descriptor["name"] != expected_name or descriptor["dtype"] != spec.dtype
                or not isinstance(shape, list) or any(type(x) is not int for x in shape)
                or tuple(shape) != spec.shape
                or type(descriptor["offset"]) is not int or descriptor["offset"] != offset
                or type(descriptor["length"]) is not int or descriptor["length"] != spec.nbytes):
            raise ArchiveError("Checkpoint tensor descriptor differs from expected schema")
        checksum = require_sha256(descriptor["sha256"])
        first = payload_start+offset
        last = first+spec.nbytes
        if last > len(blob):
            raise ArchiveError("Truncated checkpoint tensor payload")
        # Slice bounded bytes; the read-only array cannot be used to alter
        # the snapshot and the shape comes only from the trusted schema.
        raw = blob[first:last]
        if digest(raw) != checksum:
            raise ArchiveError("Checkpoint tensor SHA256 mismatch")
        array = np.frombuffer(raw, dtype=spec.wire_dtype).reshape(spec.shape)
        if array.dtype != spec.native_dtype:
            array = np.frombuffer(array.astype(spec.native_dtype).tobytes(),
                                  dtype=spec.native_dtype).reshape(spec.shape)
        arrays[expected_name] = _exact_array(array, spec)
        offset += spec.nbytes
    if payload_start+offset != len(blob):
        raise ArchiveError("Checkpoint contains trailing or unindexed bytes")
    return header["metadata"], arrays
