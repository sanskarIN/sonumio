"""Bounded symmetric coefficient quantization and canonical bit packing.

A research payload has an out-of-band shape, bit width and fixed train-fitted
scale vector. This is not a self-describing audio container or entropy codec.
The lowest signed code is reserved, and nonzero padding is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

BIT_WIDTHS = (4, 8, 12, 16)
MAX_VALUES = 128 * 129 * 64
MAX_FRAMES = 129
MAX_CHANNELS = 64
MIN_RANGE = 1e-7


def _bits(bits: int) -> int:
    if type(bits) is not int or bits not in BIT_WIDTHS:
        raise ValueError("Supported coefficient bit widths are 4, 8, 12 and 16")
    return bits


def _shape(shape: tuple[int, int, int]) -> int:
    if (type(shape) is not tuple or len(shape) != 3
            or any(type(x) is not int or x < 1 for x in shape)
            or shape[0] > 128 or shape[1] > MAX_FRAMES or shape[2] > MAX_CHANNELS
            or math.prod(shape) > MAX_VALUES):
        raise ValueError("Invalid bounded coefficient tensor shape")
    return math.prod(shape)


def payload_size(shape: tuple[int, int, int], bits: int) -> int:
    return (_shape(shape) * _bits(bits) + 7) // 8


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    """Pack row-major signed codes, least-significant bit first in each byte."""
    bits = _bits(bits)
    if not isinstance(codes, np.ndarray) or codes.dtype != np.int16:
        raise ValueError("Expected int16 coefficient codes")
    count = _shape(codes.shape)
    bound = (1 << (bits - 1)) - 1
    if np.any(codes < -bound) or np.any(codes > bound):
        raise ValueError("Coefficient code out of range or reserved")
    unsigned = (codes.reshape(count).astype(np.int32) & ((1 << bits) - 1)).astype(np.uint16)
    positions = np.arange(bits, dtype=np.uint16)
    binary = ((unsigned[:, None] >> positions) & 1).astype(np.uint8)
    return np.packbits(binary.reshape(-1), bitorder="little").tobytes()


def unpack_codes(payload: bytes, shape: tuple[int, int, int], bits: int) -> np.ndarray:
    """Decode only exact-sized, canonical payloads with valid signed codes."""
    bits = _bits(bits)
    count = _shape(shape)
    if type(payload) is not bytes or len(payload) != payload_size(shape, bits):
        raise ValueError("Coefficient payload length does not match declared shape")
    binary = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")
    used = count * bits
    if binary[used:].any():
        raise ValueError("Nonzero coefficient payload padding")
    weights = (1 << np.arange(bits, dtype=np.int32))
    unsigned = binary[:used].reshape(count, bits).astype(np.int32) @ weights
    sign = 1 << (bits - 1)
    signed = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    if (signed == -sign).any():
        raise ValueError("Reserved minimum signed coefficient code")
    return signed.astype(np.int16).reshape(shape)


def _coefficients(values: np.ndarray) -> np.ndarray:
    if (not isinstance(values, np.ndarray) or values.dtype not in (np.float32, np.float64)
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite floating-point coefficients")
    _shape(values.shape)
    return values.astype(np.float64, copy=False)


@dataclass(frozen=True)
class CoefficientQuantizer:
    """Per-coordinate symmetric uniform scales fitted on training only."""

    ranges: np.ndarray
    bits: int
    training_recordings: int
    percentile: float

    @classmethod
    def fit(cls, coefficients: np.ndarray, bits: int, *, percentile: float = 100.0):
        values = _coefficients(coefficients)
        bits = _bits(bits)
        if (len(values) < 2 or type(percentile) not in (int, float)
                or not math.isfinite(percentile) or not 95 <= percentile <= 100):
            raise ValueError("Quantizer needs training recordings and percentile in [95,100]")
        ranges = np.maximum(np.percentile(np.abs(values), percentile, axis=(0, 1)), MIN_RANGE)
        ranges.setflags(write=False)
        return cls(ranges, bits, len(values), float(percentile))

    @property
    def bound(self) -> int:
        return (1 << (self.bits - 1)) - 1

    def _validate(self, coefficients: np.ndarray) -> np.ndarray:
        values = _coefficients(coefficients)
        if values.shape[-1] != len(self.ranges):
            raise ValueError("Quantizer coordinate count mismatch")
        return values

    def quantize(self, coefficients: np.ndarray) -> np.ndarray:
        values = self._validate(coefficients)
        # Clip before division to avoid overflowing for extreme finite inputs.
        normalized = np.clip(values, -self.ranges, self.ranges) / self.ranges
        return np.rint(normalized * self.bound).astype(np.int16)

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        if not isinstance(codes, np.ndarray) or codes.dtype != np.int16:
            raise ValueError("Expected int16 coefficient codes")
        _shape(codes.shape)
        if (codes.shape[-1] != len(self.ranges) or np.any(codes > self.bound)
                or np.any(codes < -self.bound)):
            raise ValueError("Invalid quantizer codes or coordinate count")
        return (codes.astype(np.float64) / self.bound) * self.ranges

    def encode(self, coefficients: np.ndarray) -> bytes:
        return pack_codes(self.quantize(coefficients), self.bits)

    def decode(self, payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
        return self.dequantize(unpack_codes(payload, shape, self.bits))

    def diagnose(self, original: np.ndarray, restored: np.ndarray) -> dict:
        values = self._validate(original)
        decoded = self._validate(restored)
        if values.shape != decoded.shape:
            raise ValueError("Quantization diagnostic shape mismatch")
        error = decoded - values
        return {
            "coefficient_mae": float(np.mean(np.abs(error))),
            "coefficient_mse": float(np.mean(error ** 2)),
            "saturated_fraction": float(np.mean(np.abs(values) > self.ranges)),
            "saturated_fraction_by_coordinate": np.mean(
                np.abs(values) > self.ranges, axis=(0, 1),
            ).tolist(),
            "zero_code_fraction": float(np.mean(decoded == 0)),
        }

    def describe(self) -> dict:
        return {
            "bits_per_coefficient": self.bits,
            "training_recordings": self.training_recordings,
            "range_percentile": self.percentile,
            "coordinate_ranges": self.ranges.tolist(),
            "minimum_range": MIN_RANGE,
            "rounding": "nearest_ties_to_even",
            "reserved_code": -(1 << (self.bits - 1)),
            "payload_bit_order": "little",
            "self_describing_container": False,
        }
