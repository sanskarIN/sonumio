"""Explicit spectral and transition windows, overlap and COLA checks."""
from __future__ import annotations
from functools import lru_cache
import math
from typing import Iterable

import numpy as np

from .contracts import finite_float, check_audio, MAX_BYTES
from .errors import AudioContractError, AudioLimitError

MAX_WINDOW = 262_144


def _length(frames: int) -> int:
    if type(frames) is not int or not 1 <= frames <= MAX_WINDOW:
        raise AudioContractError("Invalid spectral window length")
    return frames


def _grid(frames: int, periodic: bool) -> np.ndarray:
    _length(frames)
    if type(periodic) is not bool:
        raise AudioContractError("periodic must be boolean")
    if frames == 1:
        return np.zeros(1, dtype=np.float64)
    denominator = frames if periodic else frames - 1
    return (2 * math.pi / denominator) * np.arange(frames, dtype=np.float64)


def hann(frames: int, *, periodic: bool = False) -> np.ndarray:
    """Periodic for FFT analysis, symmetric for signal fades."""
    if frames == 1:
        _length(frames)
        return np.ones(1, dtype=np.float64)
    angle = _grid(frames, periodic)
    return 0.5 - 0.5 * np.cos(angle)


def hamming(frames: int, *, periodic: bool = False) -> np.ndarray:
    if frames == 1:
        _length(frames)
        return np.ones(1, dtype=np.float64)
    return 0.54 - 0.46 * np.cos(_grid(frames, periodic))


def blackman(frames: int, *, periodic: bool = False) -> np.ndarray:
    if frames == 1:
        _length(frames)
        return np.ones(1, dtype=np.float64)
    angle = _grid(frames, periodic)
    return 0.42 - 0.5 * np.cos(angle) + 0.08 * np.cos(2 * angle)


def tukey(frames: int, *, alpha: float = 0.5,
          periodic: bool = False) -> np.ndarray:
    _length(frames)
    alpha = finite_float(alpha, "Tukey alpha", minimum=0, maximum=1)
    if frames == 1:
        return np.ones(1, dtype=np.float64)
    if alpha == 0:
        return np.ones(frames, dtype=np.float64)
    if alpha == 1:
        return hann(frames, periodic=periodic)
    denom = frames if periodic else frames - 1
    x = np.arange(frames, dtype=np.float64) / denom
    y = np.ones(frames, dtype=np.float64)
    lower, upper = x < alpha / 2, x > 1 - alpha / 2
    y[lower] = 0.5 * (1 + np.cos(math.pi * (2 * x[lower] / alpha - 1)))
    y[upper] = 0.5 * (1 + np.cos(math.pi * (2 * x[upper] / alpha - 2 / alpha + 1)))
    return y


def sqrt_hann(frames: int, *, periodic: bool = False) -> np.ndarray:
    return np.sqrt(np.maximum(0, hann(frames, periodic=periodic)))


@lru_cache(maxsize=128)
def _cached_window(name: str, frames: int, periodic: bool, alpha: float) -> bytes:
    if name == "rectangular":
        data = np.ones(_length(frames), dtype=np.float64)
    elif name == "hann":
        data = hann(frames, periodic=periodic)
    elif name == "hamming":
        data = hamming(frames, periodic=periodic)
    elif name == "blackman":
        data = blackman(frames, periodic=periodic)
    elif name == "sqrt_hann":
        data = sqrt_hann(frames, periodic=periodic)
    elif name == "tukey":
        data = tukey(frames, alpha=alpha, periodic=periodic)
    else:
        raise AudioContractError("Unknown spectral window")
    return data.astype("<f8", copy=False).tobytes()


def window_bank(name: str, frames: int, *,
                periodic: bool = False, alpha: float = 0.5) -> np.ndarray:
    """Return an independent mutable copy; cached backing is immutable bytes."""
    _length(frames)
    a = finite_float(alpha, "window alpha", minimum=0, maximum=1)
    raw = _cached_window(name, frames, periodic, a)
    return np.frombuffer(raw, dtype="<f8").copy()


def envelope(frames: int, *, fade_in: int = 0, fade_out: int = 0,
             curve: str = "cosine") -> np.ndarray:
    """A bounded mono envelope; zero fades mean no modification."""
    if type(frames) is not int or not 1 <= frames <= 5_000_000:
        raise AudioContractError("Invalid envelope length")
    if (type(fade_in) is not int or type(fade_out) is not int
            or fade_in < 0 or fade_out < 0 or fade_in + fade_out > frames):
        raise AudioContractError("Fades overlap or extend beyond audio")
    if curve not in ("linear", "cosine", "smoothstep", "equal_power"):
        raise AudioContractError("Unknown fade envelope shape")
    curve_in = np.ones(frames, dtype=np.float64)
    if fade_in:
        u = np.linspace(0, 1, fade_in, dtype=np.float64)
        if curve == "linear":
            curve_in[:fade_in] = u
        elif curve == "cosine":
            curve_in[:fade_in] = np.sin(math.pi / 2 * u) ** 2
        elif curve == "smoothstep":
            curve_in[:fade_in] = u * u * (3 - 2 * u)
        else:
            curve_in[:fade_in] = np.sin(math.pi / 2 * u)
    if fade_out:
        u = np.linspace(0, 1, fade_out, dtype=np.float64)
        if curve == "linear":
            curve_in[-fade_out:] *= 1 - u
        elif curve == "cosine":
            curve_in[-fade_out:] *= np.cos(math.pi / 2 * u) ** 2
        elif curve == "smoothstep":
            u = 1 - u
            curve_in[-fade_out:] *= u * u * (3 - 2 * u)
        else:
            curve_in[-fade_out:] *= np.cos(math.pi / 2 * u)
    # Exact zero endpoints are important for rendering/hash stability: cosine
    # half-pi otherwise leaves a tiny floating-point residual at the edge.
    if fade_in:
        curve_in[0] = 0.0
    if fade_out:
        curve_in[-1] = 0.0
    return curve_in


def overlap_envelope(window: np.ndarray, hops: int, *,
                     periods: int = 6, square: bool = False) -> np.ndarray:
    """Return overlap-sum normalization for an interior periodic segment."""
    if (not isinstance(window, np.ndarray) or window.ndim != 1
            or window.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(window).all() or len(window) == 0
            or type(hops) is not int or not 1 <= hops <= len(window)
            or type(periods) is not int or not 3 <= periods <= 20):
        raise AudioContractError("Invalid spectral overlap geometry")
    span = len(window) + periods * hops
    if span * 8 > MAX_BYTES:
        raise AudioLimitError("Window overlap exceeds memory budget")
    output = np.zeros(span, dtype=np.float64)
    w = np.asarray(window, dtype=np.float64)
    if square:
        w = w ** 2
    for i in range(periods + 1):
        start = i * hops
        output[start:start + len(w)] += w
    return output[hops * 2:hops * (periods - 1)]


def validate_cola(window: np.ndarray, hop: int, *,
                  tolerance: float = 1e-7,
                  square: bool = False) -> bool:
    """Check constant overlap-add in a fully covered interior region."""
    tol = finite_float(tolerance, "COLA tolerance", minimum=0, maximum=0.1)
    interior = overlap_envelope(window, hop, square=square)
    if interior.size == 0:
        raise AudioContractError("Insufficient overlapping windows to check")
    return bool(np.max(interior) - np.min(interior) <= tol * max(
        1.0, float(np.max(np.abs(interior)))
    ))


def weighted_overlap_add(
    frames: Iterable[np.ndarray], *,
    hop: int, window: np.ndarray,
    output_frames: int | None = None,
    regularization: float = 1e-10,
) -> np.ndarray:
    """Invert windowed frames with normalization by sum(window²).

    Frames here are ALREADY windowed analysis frames. A window is applied
    a second time during synthesis, then divided by squared-window sum.
    This supports unmodified STFT->ISTFT with nonzero interior overlap,
    but Hann boundary zeros must be padded/cropped by the caller.
    """
    if (not isinstance(window, np.ndarray) or window.ndim != 1
            or len(window) < 2 or not np.isfinite(window).all()
            or type(hop) is not int or not 1 <= hop <= len(window)):
        raise AudioContractError("Invalid overlap-add window or hop")
    reg = finite_float(regularization, "regularization", minimum=0, maximum=.01)
    blocks = list(frames)
    if not blocks or len(blocks) > 100_000:
        raise AudioContractError("Expected bounded nonempty frame sequence")
    n = len(window)
    if any(not isinstance(b, np.ndarray) or b.shape != (n,)
           or b.dtype not in (np.float32, np.float64)
           or not np.isfinite(b).all() for b in blocks):
        raise AudioContractError("All weighted frames need equal finite shape")
    length = (len(blocks) - 1) * hop + n
    if output_frames is not None:
        if type(output_frames) is not int or not 1 <= output_frames <= length:
            raise AudioContractError("Invalid overlap-add output limit")
    if length * 16 > MAX_BYTES:
        raise AudioLimitError("Weighted overlap-add memory budget exceeded")
    output = np.zeros(length, dtype=np.float64)
    weight = np.zeros(length, dtype=np.float64)
    w = np.asarray(window, dtype=np.float64)
    for i, block in enumerate(blocks):
        start = i * hop
        output[start:start+n] += block.astype(np.float64) * w
        weight[start:start+n] += w * w
    mask = weight > reg
    output[mask] /= weight[mask]
    output[~mask] = 0
    if output_frames is not None:
        output = output[:output_frames]
    return output.astype(np.float32)
