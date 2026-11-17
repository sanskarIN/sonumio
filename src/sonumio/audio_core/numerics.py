"""Conservative floating DSP arithmetic and channel-preserving mixing.

Signal computations use float64 intermediates where practical, validate
finite arguments and return explicitly typed float32 PCM. A decibel
quantity is metadata only: it is NOT an EBU-R128 integrated-LUFS estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .contracts import check_audio, finite_float, as_channel_first, MAX_BYTES
from .errors import AudioContractError, AudioLimitError


def db_to_gain(decibels: float) -> float:
    """Amplitude gain from conventional 20 log10 decibels (bounded)."""
    value = finite_float(decibels, "decibels", minimum=-300, maximum=100)
    result = 10.0 ** (value / 20.0)
    if not math.isfinite(result):
        raise AudioContractError("Decibel gain overflow")
    return result


def gain_to_db(gain: float, *, floor_db: float = -300) -> float:
    gain = finite_float(gain, "gain", minimum=0, maximum=1e6)
    floor_db = finite_float(floor_db, "floor_db", minimum=-400, maximum=0)
    if gain == 0:
        return floor_db
    return max(floor_db, 20 * math.log10(gain))


def dbfs(level: float, *, floor_db: float = -300) -> float:
    """Peak/RMS amplitude-to-dBFS, not perceived loudness or LUFS."""
    return gain_to_db(level, floor_db=floor_db)


@dataclass(frozen=True, slots=True)
class Decibel:
    """Bounded signed decibel change, usable for gain-staging automation."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", finite_float(
            self.value, "decibel", minimum=-300, maximum=100
        ))

    @property
    def linear(self) -> float:
        return db_to_gain(self.value)

    def __add__(self, other: "Decibel") -> "Decibel":
        if not isinstance(other, Decibel):
            return NotImplemented
        return Decibel(self.value + other.value)

    def __sub__(self, other: "Decibel") -> "Decibel":
        if not isinstance(other, Decibel):
            return NotImplemented
        return Decibel(self.value - other.value)


def stable_peak(samples: np.ndarray, *, per_channel: bool = False):
    data = as_channel_first(check_audio(samples), float64=True)
    peaks = np.max(np.abs(data), axis=1)
    return peaks if per_channel else float(np.max(peaks))


def stable_rms(samples: np.ndarray, *, per_channel: bool = False):
    data = as_channel_first(check_audio(samples), float64=True)
    levels = np.sqrt(np.mean(np.square(data), axis=1))
    return levels if per_channel else float(np.sqrt(np.mean(np.square(levels))))


def finite_clamp(value: float, minimum: float, maximum: float) -> float:
    lower = finite_float(minimum, "minimum")
    upper = finite_float(maximum, "maximum")
    if lower > upper:
        raise AudioContractError("Clamp minimum cannot exceed maximum")
    return min(max(finite_float(value, "value"), lower), upper)


def saturate(samples: np.ndarray, *, threshold: float = 1.0,
             mode: str = "hard") -> np.ndarray:
    """Hard clip or continuously differentiable tanh soft saturation.

    This is an effect, not an imperceptible limiter; saturation changes
    timbre and clipping metadata should be inspected afterwards.
    """
    bound = finite_float(threshold, "threshold", minimum=1e-5, maximum=100)
    if mode not in ("hard", "tanh", "cubic"):
        raise AudioContractError("Unknown saturation mode")
    data = as_channel_first(check_audio(samples), float64=True)
    if mode == "hard":
        processed = np.clip(data, -bound, bound)
    elif mode == "tanh":
        processed = bound * np.tanh(data / bound)
    else:
        x = np.clip(data / bound, -1, 1)
        processed = bound * (1.5 * x - 0.5 * x ** 3)
    return np.asarray(processed, dtype=np.float32)


def linear_ramp(frames: int, start: float = 0.0, stop: float = 1.0,
                *, endpoint: bool = True) -> np.ndarray:
    if type(frames) is not int or not 1 <= frames <= 5_000_000:
        raise AudioContractError("Invalid linear ramp frame budget")
    a = finite_float(start, "ramp start")
    b = finite_float(stop, "ramp stop")
    return np.linspace(a, b, frames, endpoint=endpoint, dtype=np.float64)


def equal_power_ramps(frames: int, *, endpoint: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Cosine/sine equal-power crossfade; perfect for uncorrelated sources.

    Crossfading identical correlated signals can raise mid-fade amplitude;
    a linear crossfade may be preferable for aligned duplicates.
    """
    if type(frames) is not int or not 2 <= frames <= 5_000_000:
        raise AudioContractError("Expected two or more fade frames")
    angles = linear_ramp(frames, 0, math.pi / 2, endpoint=endpoint)
    return np.cos(angles), np.sin(angles)


def mix_accumulate(inputs: Sequence[np.ndarray],
                   *, gains_db: Sequence[float] | None = None,
                   normalize: str = "none",
                   cap_frames: int | None = None) -> np.ndarray:
    """Mix channel-aligned arrays with float64 accumulation.

    Inputs must agree on channel topology; shorter inputs are zero-padded
    to the longest one. No hidden peak normalization is applied unless
    explicitly requested. This operation never modifies its inputs.
    """
    if not isinstance(inputs, (list, tuple)) or not 1 <= len(inputs) <= 64:
        raise AudioContractError("Expected 1..64 source audio arrays")
    converted = [as_channel_first(check_audio(x), float64=True) for x in inputs]
    channels = converted[0].shape[0]
    if any(x.shape[0] != channels for x in converted):
        raise AudioContractError("All sources need equal channel counts")
    max_frames = max(x.shape[1] for x in converted)
    if cap_frames is not None:
        if type(cap_frames) is not int or cap_frames < 1:
            raise AudioContractError("Invalid cap_frames")
        max_frames = min(max_frames, cap_frames)
    if channels * max_frames * 16 > MAX_BYTES:
        raise AudioLimitError("Mix working buffer exceeds memory cap")
    if gains_db is None:
        gains_db = [0.0] * len(converted)
    if not isinstance(gains_db, (list, tuple)) or len(gains_db) != len(converted):
        raise AudioContractError("Expected one gain per source")
    if normalize not in ("none", "peak", "equal_power"):
        raise AudioContractError("Unknown explicit mix normalization")
    result = np.zeros((channels, max_frames), dtype=np.float64)
    for array, gain in zip(converted, gains_db):
        length = min(array.shape[1], max_frames)
        result[:, :length] += array[:, :length] * db_to_gain(gain)
    if normalize == "equal_power":
        result /= math.sqrt(len(converted))
    elif normalize == "peak":
        high = float(np.max(np.abs(result)))
        if high > 1:
            result /= high
    return result.astype(np.float32)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation of equal-shape PCM, with double intermediates."""
    left, right = check_audio(a), check_audio(b)
    if left.shape != right.shape:
        raise AudioContractError("Interpolation shape mismatch")
    position = finite_float(t, "interpolation progress", minimum=0, maximum=1)
    return (left.astype(np.float64) * (1 - position)
            + right.astype(np.float64) * position).astype(np.float32)


def mid_side_encode(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = as_channel_first(check_audio(stereo), float64=True)
    if array.shape[0] != 2:
        raise AudioContractError("Mid/side encoding requires exactly two channels")
    mid = (array[0] + array[1]) / 2
    side = (array[0] - array[1]) / 2
    return mid.astype(np.float32), side.astype(np.float32)


def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    left, right = check_audio(mid), check_audio(side)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise AudioContractError("Mid and side must be equal-size mono vectors")
    a, b = left.astype(np.float64), right.astype(np.float64)
    return np.stack((a + b, a - b)).astype(np.float32)


def balance_stereo(stereo: np.ndarray, balance: float) -> np.ndarray:
    """Equal-power stereo balance, with -1 full left and +1 full right."""
    data = as_channel_first(check_audio(stereo), float64=True)
    if data.shape[0] != 2:
        raise AudioContractError("Stereo balancing requires two channels")
    value = finite_float(balance, "balance", minimum=-1, maximum=1)
    if value >= 0:
        gains = (math.cos(value * math.pi / 2), 1)
    else:
        gains = (1, math.cos(-value * math.pi / 2))
    return (data * np.asarray(gains)[:, None]).astype(np.float32)
