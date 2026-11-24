"""Deterministic nuisance and degradation controls for an 8 kHz source probe.

Controls never change the probe fit or calibration data. Some operations
(polarity, bounded gain) preserve source identity by construction; others
(time shifts, filtering, quantization) are stressors, not guaranteed
perceptual equivalences. This module deliberately does not label any
transformed waveform as a new training example.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from research.latent_audio import FRAMES

MAX_INPUT_PEAK = 1.0001
INVARIANCE = "scale_polarity"
TEMPORAL = "temporal_stress"
DISTORTION = "distortion_stress"


@dataclass(frozen=True)
class Transform:
    name: str
    kind: str
    value: float
    family: str

    def __post_init__(self):
        if type(self.name) is not str or not self.name or len(self.name) > 64:
            raise ValueError("Invalid control identifier")
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("Expected finite transform parameter")
        if self.kind == "gain":
            valid = self.family == INVARIANCE and .125 <= self.value <= 1.0
        elif self.kind == "polarity":
            valid = self.family == INVARIANCE and self.value == -1
        elif self.kind == "circular_shift":
            valid = (self.family == TEMPORAL and type(self.value) is int
                     and 1 <= abs(self.value) <= FRAMES // 4)
        elif self.kind == "quantize":
            valid = (self.family == DISTORTION and type(self.value) is int
                     and 4 <= self.value <= 16)
        elif self.kind == "lowpass":
            valid = self.family == DISTORTION and 250 <= self.value <= 3500
        elif self.kind == "clip":
            valid = self.family == DISTORTION and .1 <= self.value <= .95
        else:
            valid = False
        if not valid:
            raise ValueError("Unsupported bounded control transform")


CONTROLS = (
    Transform("gain_half", "gain", .5, INVARIANCE),
    Transform("gain_quarter", "gain", .25, INVARIANCE),
    Transform("polarity_flip", "polarity", -1, INVARIANCE),
    Transform("shift_128", "circular_shift", 128, TEMPORAL),
    Transform("shift_1024", "circular_shift", 1024, TEMPORAL),
    Transform("shift_minus_512", "circular_shift", -512, TEMPORAL),
    Transform("quantize_16bit", "quantize", 16, DISTORTION),
    Transform("quantize_8bit", "quantize", 8, DISTORTION),
    Transform("quantize_4bit", "quantize", 4, DISTORTION),
    Transform("lowpass_3000hz", "lowpass", 3000, DISTORTION),
    Transform("lowpass_1500hz", "lowpass", 1500, DISTORTION),
    Transform("clip_half", "clip", .5, DISTORTION),
)


def validate_wave(wave):
    if (not isinstance(wave, np.ndarray) or wave.shape != (FRAMES,)
            or wave.dtype != np.float32 or not np.isfinite(wave).all()
            or np.max(np.abs(wave)) > MAX_INPUT_PEAK):
        raise ValueError("Expected bounded finite float32 mono waveform")


def apply_transform(wave: np.ndarray, transform: Transform) -> np.ndarray:
    """Return a new float32 buffer; never mutate or normalize the source."""
    validate_wave(wave)
    if not isinstance(transform, Transform):
        raise ValueError("Expected validated nuisance control")
    value = transform.value
    signal = wave.astype(np.float64)
    if transform.kind == "gain":
        result = signal * value
    elif transform.kind == "polarity":
        result = -signal
    elif transform.kind == "circular_shift":
        result = np.roll(signal, value)
    elif transform.kind == "quantize":
        # Symmetric signed grid; do not peak normalize or apply dither.
        bound = (1 << (value - 1)) - 1
        result = np.rint(np.clip(signal, -1, 1) * bound) / bound
    elif transform.kind == "lowpass":
        # Smooth linear-phase windowed-sinc FIR, fixed odd 129 taps.
        # Reflection padding avoids inserting silence at the clip boundary.
        taps = np.arange(-64, 65, dtype=np.float64)
        frequency = value / 8000
        kernel = 2 * frequency * np.sinc(2 * frequency * taps) * np.hamming(len(taps))
        kernel /= kernel.sum()
        result = np.convolve(np.pad(signal, (64, 64), mode="reflect"), kernel, mode="valid")
        # Explicit output limiting: no hidden amplitude renormalization.
        result = np.clip(result, -1, 1)
    else:
        result = np.clip(signal, -value, value)
    result = np.ascontiguousarray(result, dtype=np.float32)
    validate_wave(result)
    return result


def waveform_diagnostics(original: np.ndarray, changed: np.ndarray) -> dict:
    validate_wave(original)
    validate_wave(changed)
    a, b = original.astype(np.float64), changed.astype(np.float64)
    power = np.mean(a ** 2)
    other = np.mean(b ** 2)
    error = np.mean((a - b) ** 2)
    denominator = np.sqrt(power * other)
    return {
        "original_rms": float(np.sqrt(power)),
        "transformed_rms": float(np.sqrt(other)),
        "sample_mae": float(np.mean(np.abs(a - b))),
        "sample_rmse": float(np.sqrt(error)),
        "relative_squared_error": float(error / power) if power > 1e-15 else None,
        "waveform_cosine": float(np.mean(a * b) / denominator) if denominator > 1e-15 else None,
        "exactly_equal": bool(np.array_equal(original, changed)),
        "transformed_silent": bool(other <= 1e-15),
    }
