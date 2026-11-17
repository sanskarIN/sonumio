"""Deterministic offline biquad/FIR filtering with explicit stability limits.

Biquad coefficients use standard RBJ cookbook analog-prototype formulas,
with finite Q, Nyquist and gain validation. A direct-form transposed-II
implementation preserves state across chunks without accessing system audio.
This is reference DSP and not a learning-based denoiser or de-reverberator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .contracts import AudioBuffer, AudioSpec, as_channel_first, check_rate, MAX_BYTES
from .errors import AudioContractError, AudioLimitError
from .windows import window_bank


@dataclass(frozen=True, slots=True)
class Biquad:
    """Normalized a0=1 coefficients of a stable real second-order IIR."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    def __post_init__(self) -> None:
        values = (self.b0, self.b1, self.b2, self.a1, self.a2)
        if any(type(v) not in (float, int) or not math.isfinite(v)
               for v in values):
            raise AudioContractError("Biquad coefficients must be finite")
        # Jury stability for z² + a1*z + a2 (strict inside unit circle).
        if (abs(self.a2) >= 1 or
                1 + self.a1 + self.a2 <= 0 or
                1 - self.a1 + self.a2 <= 0):
            raise AudioContractError("Unstable digital biquad denominator")

    @property
    def numerator(self) -> tuple[float, float, float]:
        return self.b0, self.b1, self.b2

    @property
    def denominator(self) -> tuple[float, float, float]:
        return 1.0, self.a1, self.a2


@dataclass(frozen=True, slots=True)
class FilterSpec:
    sample_rate: int
    cutoff_hz: float
    kind: str
    q: float = 0.7071067811865476
    gain_db: float = 0.0

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        if self.kind not in (
            "lowpass", "highpass", "bandpass", "notch",
            "allpass", "peaking", "lowshelf", "highshelf"
        ):
            raise AudioContractError("Unknown RBJ biquad filter kind")
        for name, value, lo, hi in (
            ("cutoff_hz", self.cutoff_hz, 1.0, self.sample_rate / 2 - 1),
            ("q", self.q, 0.05, 50.0),
            ("gain_db", self.gain_db, -48.0, 48.0),
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise AudioContractError(name + " must be finite")
            if not lo <= value <= hi:
                raise AudioContractError(name + " outside supported range")


def design_biquad(spec: FilterSpec) -> Biquad:
    if not isinstance(spec, FilterSpec):
        raise AudioContractError("Expected FilterSpec")
    w0 = 2 * math.pi * spec.cutoff_hz / spec.sample_rate
    cosine, sine = math.cos(w0), math.sin(w0)
    alpha = sine / (2 * spec.q)
    amplitude = 10 ** (spec.gain_db / 40)
    kind = spec.kind
    if kind == "lowpass":
        b0, b1, b2 = (1 - cosine) / 2, 1 - cosine, (1 - cosine) / 2
        a0, a1, a2 = 1 + alpha, -2 * cosine, 1 - alpha
    elif kind == "highpass":
        b0, b1, b2 = (1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2
        a0, a1, a2 = 1 + alpha, -2 * cosine, 1 - alpha
    elif kind == "bandpass":
        b0, b1, b2 = alpha, 0, -alpha
        a0, a1, a2 = 1 + alpha, -2 * cosine, 1 - alpha
    elif kind == "notch":
        b0, b1, b2 = 1, -2 * cosine, 1
        a0, a1, a2 = 1 + alpha, -2 * cosine, 1 - alpha
    elif kind == "allpass":
        b0, b1, b2 = 1 - alpha, -2 * cosine, 1 + alpha
        a0, a1, a2 = 1 + alpha, -2 * cosine, 1 - alpha
    elif kind == "peaking":
        b0, b1, b2 = 1 + alpha * amplitude, -2 * cosine, 1 - alpha * amplitude
        a0, a1, a2 = 1 + alpha / amplitude, -2 * cosine, 1 - alpha / amplitude
    else:
        # Shelf formula from the Audio EQ Cookbook.
        shelf_alpha = sine / 2 * math.sqrt(
            max(0, (amplitude + 1 / amplitude)
                * (1 / spec.q - 1) + 2)
        )
        sqrt_a = math.sqrt(amplitude)
        if kind == "lowshelf":
            b0 = amplitude * ((amplitude + 1) - (amplitude - 1) * cosine
                              + 2 * sqrt_a * shelf_alpha)
            b1 = 2 * amplitude * ((amplitude - 1) - (amplitude + 1) * cosine)
            b2 = amplitude * ((amplitude + 1) - (amplitude - 1) * cosine
                              - 2 * sqrt_a * shelf_alpha)
            a0 = (amplitude + 1) + (amplitude - 1) * cosine + 2 * sqrt_a * shelf_alpha
            a1 = -2 * ((amplitude - 1) + (amplitude + 1) * cosine)
            a2 = (amplitude + 1) + (amplitude - 1) * cosine - 2 * sqrt_a * shelf_alpha
        else:
            b0 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine
                              + 2 * sqrt_a * shelf_alpha)
            b1 = -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cosine)
            b2 = amplitude * ((amplitude + 1) + (amplitude - 1) * cosine
                              - 2 * sqrt_a * shelf_alpha)
            a0 = (amplitude + 1) - (amplitude - 1) * cosine + 2 * sqrt_a * shelf_alpha
            a1 = 2 * ((amplitude - 1) - (amplitude + 1) * cosine)
            a2 = (amplitude + 1) - (amplitude - 1) * cosine - 2 * sqrt_a * shelf_alpha
    if not math.isfinite(a0) or abs(a0) < 1e-12:
        raise AudioContractError("Biquad normalization denominator underflow")
    return Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


class BiquadState:
    """Stateful causal transposed direct-form-II filter with manual reset."""

    def __init__(self, coefficients: Biquad, channels: int):
        if not isinstance(coefficients, Biquad):
            raise AudioContractError("Expected normalized Biquad coefficients")
        if type(channels) is not int or not 1 <= channels <= 8:
            raise AudioContractError("Invalid filter channel count")
        self.coefficients = coefficients
        self.channels = channels
        self.delay1 = np.zeros(channels, dtype=np.float64)
        self.delay2 = np.zeros(channels, dtype=np.float64)

    def reset(self) -> None:
        self.delay1.fill(0)
        self.delay2.fill(0)

    def process(self, data: np.ndarray) -> np.ndarray:
        source = as_channel_first(data, float64=True)
        if source.shape[0] != self.channels:
            raise AudioContractError("Filter channel topology mismatch")
        out = np.empty(source.shape, dtype=np.float32)
        c = self.coefficients
        for frame in range(source.shape[1]):
            x = source[:, frame]
            y = c.b0 * x + self.delay1
            self.delay1 = c.b1 * x - c.a1 * y + self.delay2
            self.delay2 = c.b2 * x - c.a2 * y
            if not (np.isfinite(y).all() and np.isfinite(self.delay1).all()
                    and np.isfinite(self.delay2).all()):
                raise AudioContractError("Digital filter diverged")
            out[:, frame] = y.astype(np.float32)
        return out


def biquad_filter(audio: AudioBuffer, spec: FilterSpec, *,
                  passes: int = 1) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer) or not isinstance(spec, FilterSpec):
        raise AudioContractError("Expected AudioBuffer and FilterSpec")
    if spec.sample_rate != audio.sample_rate:
        raise AudioContractError("Filter and sound sample clocks differ")
    if type(passes) is not int or not 1 <= passes <= 8:
        raise AudioContractError("Filter passes must be 1..8")
    coeffs = design_biquad(spec)
    samples = audio.mutable_copy()
    for _ in range(passes):
        samples = BiquadState(coeffs, audio.channels).process(samples)
    return audio.copy_with(samples, source_note="biquad " + spec.kind)


def design_fir_lowpass(sample_rate: int, cutoff_hz: float, *,
                       taps: int = 129, window: str = "blackman") -> np.ndarray:
    check_rate(sample_rate)
    if (type(cutoff_hz) not in (int, float) or not math.isfinite(cutoff_hz)
            or not 1 <= cutoff_hz < sample_rate / 2):
        raise AudioContractError("FIR cutoff outside Nyquist")
    if type(taps) is not int or not 5 <= taps <= 4095 or taps % 2 == 0:
        raise AudioContractError("FIR filter requires odd 5..4095 taps")
    if window not in ("hann", "hamming", "blackman"):
        raise AudioContractError("Unknown FIR window")
    grid = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
    fc = cutoff_hz / sample_rate
    kernel = 2 * fc * np.sinc(2 * fc * grid)
    kernel *= window_bank(window, taps, periodic=False)
    total = float(kernel.sum())
    if not math.isfinite(total) or abs(total) < 1e-12:
        raise AudioContractError("FIR design has vanishing DC gain")
    return kernel / total


def design_fir_highpass(sample_rate: int, cutoff_hz: float, *,
                        taps: int = 129, window: str = "blackman") -> np.ndarray:
    kernel = -design_fir_lowpass(sample_rate, cutoff_hz,
                                 taps=taps, window=window)
    kernel[(taps - 1) // 2] += 1
    return kernel


def fir_filter(samples: np.ndarray, taps: np.ndarray, *,
               mode: str = "same") -> np.ndarray:
    data = as_channel_first(samples, float64=True)
    if (not isinstance(taps, np.ndarray) or taps.ndim != 1
            or not 1 <= len(taps) <= 4095 or not np.isfinite(taps).all()):
        raise AudioContractError("FIR taps must be a finite 1-D vector")
    if mode not in ("same", "full", "valid"):
        raise AudioContractError("Unknown FIR convolution mode")
    if mode == "valid" and len(taps) > data.shape[1]:
        raise AudioContractError("FIR valid convolution requires taps <= source")
    out_frames = (
        data.shape[1] + len(taps) - 1 if mode == "full" else
        data.shape[1] - len(taps) + 1 if mode == "valid" else data.shape[1]
    )
    if out_frames * data.shape[0] * 8 > MAX_BYTES:
        raise AudioLimitError("FIR convolution output too large")
    kernel = taps.astype(np.float64)
    result = np.stack([
        np.convolve(channel, kernel, mode=mode) for channel in data
    ])
    if not np.isfinite(result).all():
        raise AudioContractError("FIR output became nonfinite")
    return result.astype(np.float32)


def dc_block(samples: np.ndarray, *, pole: float = 0.995) -> np.ndarray:
    """Causal highpass y[n]=x[n]-x[n-1]+pole*y[n-1]."""
    if type(pole) not in (float, int) or not 0 <= pole < 1:
        raise AudioContractError("DC blocker pole must be in [0,1)")
    data = as_channel_first(samples, float64=True)
    out = np.empty(data.shape, dtype=np.float32)
    previous_x = np.zeros(data.shape[0], dtype=np.float64)
    previous_y = np.zeros(data.shape[0], dtype=np.float64)
    for n in range(data.shape[1]):
        x = data[:, n]
        y = x - previous_x + pole * previous_y
        out[:, n] = y.astype(np.float32)
        previous_x, previous_y = x, y
    return out
