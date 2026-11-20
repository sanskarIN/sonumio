"""Deterministic PCM quantization, optional TPDF dither and decoded samples.

Digital dither is a quantization option, not a watermark or noise-removal
algorithm. No external RNG state is accessed: explicit seeds are mandatory
when dither=True so outputs can be reproduced in tests and exports.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .contracts import check_audio, as_channel_first, finite_float
from .errors import AudioContractError, AudioFormatError


@dataclass(frozen=True, slots=True)
class QuantizationSettings:
    bits: int = 16
    dither: bool = False
    seed: int | None = None
    headroom_db: float = 0.0

    def __post_init__(self) -> None:
        if type(self.bits) is not int or self.bits not in (8, 16, 24, 32):
            raise AudioFormatError("PCM bit depth must be 8, 16, 24 or 32")
        if type(self.dither) is not bool:
            raise AudioContractError("dither must be boolean")
        if self.dither and (type(self.seed) is not int or
                            not 0 <= self.seed < 2**32):
            raise AudioContractError("TPDF dither requires explicit 32-bit seed")
        if self.seed is not None and (
            type(self.seed) is not int or not 0 <= self.seed < 2**32
        ):
            raise AudioContractError("Invalid deterministic dither seed")
        finite_float(self.headroom_db, "headroom_db", minimum=-60, maximum=0)


def _integer_range(bits: int) -> tuple[int, int]:
    if type(bits) is not int or bits not in (8, 16, 24, 32):
        raise AudioFormatError("Invalid PCM integer bit depth")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def quantize_pcm(samples: np.ndarray, settings: QuantizationSettings
                 ) -> np.ndarray:
    """Return int64 container of signed PCM values; 8-bit offset is elsewhere.

    Peak < +1 maps to the positive max, while -1 maps to exact signed min.
    Dither is drawn in quantization-step units *before* round-to-even.
    """
    if not isinstance(settings, QuantizationSettings):
        raise AudioContractError("Expected QuantizationSettings")
    source = as_channel_first(check_audio(samples), float64=True)
    gain = 10 ** (settings.headroom_db / 20)
    source = source * gain
    scale = 1 << (settings.bits - 1)
    low, high = _integer_range(settings.bits)
    x = np.clip(source, -1, 1) * scale
    if settings.dither:
        rng = np.random.default_rng(settings.seed)
        x += rng.random(x.shape) - rng.random(x.shape)
    return np.clip(np.rint(x), low, high).astype(np.int64)


def dequantize_pcm(integers: np.ndarray, *, bits: int) -> np.ndarray:
    low, high = _integer_range(bits)
    if not isinstance(integers, np.ndarray) or integers.ndim not in (1, 2):
        raise AudioContractError("Expected 1-D or 2-D integer PCM")
    if integers.dtype.kind not in ("i", "u"):
        raise AudioContractError("PCM dequantization requires integer samples")
    if integers.size == 0 or np.any(integers < low) or np.any(integers > high):
        raise AudioContractError("Integer sample outside selected PCM bit depth")
    return (integers.astype(np.float64) / (1 << (bits - 1))).astype(np.float32)


def pack_pcm(integers: np.ndarray, *, bits: int) -> bytes:
    """Pack channel-FIRST integers into interleaved little-endian PCM bytes."""
    low, high = _integer_range(bits)
    if not isinstance(integers, np.ndarray) or integers.ndim not in (1, 2):
        raise AudioContractError("Expected signed PCM integer array")
    if integers.dtype.kind not in ("i", "u") or integers.size == 0:
        raise AudioContractError("Expected nonempty integer PCM samples")
    if np.any(integers < low) or np.any(integers > high):
        raise AudioContractError("Integer value outside PCM bit depth")
    interleaved = integers if integers.ndim == 1 else integers.T
    values = interleaved.reshape(-1).astype(np.int64, copy=False)
    if bits == 8:
        return (values + 128).astype(np.uint8).tobytes()
    if bits == 16:
        return values.astype("<i2").tobytes()
    if bits == 32:
        return values.astype("<i4").tobytes()
    # Exact two's-complement 24-bit little-endian, including negatives.
    unsigned = np.bitwise_and(values, 0xFFFFFF).astype(np.uint32)
    packed = np.empty((len(unsigned), 3), dtype=np.uint8)
    packed[:, 0] = unsigned & 0xFF
    packed[:, 1] = (unsigned >> 8) & 0xFF
    packed[:, 2] = (unsigned >> 16) & 0xFF
    return packed.tobytes()


def unpack_pcm(data: bytes, *, bits: int, channels: int) -> np.ndarray:
    """Decode little-endian PCM bytes into a channel-first int64 array."""
    _integer_range(bits)
    if type(channels) is not int or not 1 <= channels <= 8:
        raise AudioContractError("Expected 1..8 channels")
    if not isinstance(data, bytes) or not data:
        raise AudioContractError("Expected nonempty PCM bytes")
    step = (bits // 8) * channels
    if len(data) % step:
        raise AudioFormatError("PCM payload is not an integral number of frames")
    if bits == 8:
        values = np.frombuffer(data, dtype=np.uint8).astype(np.int64) - 128
    elif bits == 16:
        values = np.frombuffer(data, dtype="<i2").astype(np.int64)
    elif bits == 32:
        values = np.frombuffer(data, dtype="<i4").astype(np.int64)
    else:
        triplets = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        unsigned = (
            triplets[:, 0].astype(np.int64)
            | (triplets[:, 1].astype(np.int64) << 8)
            | (triplets[:, 2].astype(np.int64) << 16)
        )
        values = np.where(unsigned >= (1 << 23), unsigned - (1 << 24), unsigned)
    return values.reshape(-1, channels).T.copy()


def encode_pcm_bytes(samples: np.ndarray, *, settings: QuantizationSettings
                     ) -> bytes:
    return pack_pcm(quantize_pcm(samples, settings), bits=settings.bits)


def decode_pcm_bytes(data: bytes, *, bits: int, channels: int) -> np.ndarray:
    return dequantize_pcm(unpack_pcm(data, bits=bits, channels=channels), bits=bits)
