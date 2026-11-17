"""Explicit offline STFT spectral repair and analysis operations.

These are interpretable CPU algorithms, NOT a learned audio denoiser,
commercial mastering processor, speech separator or automatic content filter.
Operations remain opt-in, constrained by the existing finite AudioBuffer/STFT
contracts, and must not replace training data without separate evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .contracts import AudioBuffer, finite_float, MAX_BYTES
from .errors import AudioContractError, AudioLimitError
from .stft import Spectrogram, StftConfig, stft, istft, complex_mask


@dataclass(frozen=True, slots=True)
class SpectralGateOptions:
    """Frequency-domain amplitude gate relative to a declared noise profile."""

    threshold_ratio: float = 2.0
    floor_gain: float = 0.05
    exponent: float = 1.0
    temporal_smooth: int = 1
    frequency_smooth: int = 1

    def __post_init__(self):
        finite_float(self.threshold_ratio, "noise threshold ratio",
                     minimum=.1, maximum=100)
        finite_float(self.floor_gain, "suppression floor",
                     minimum=0, maximum=1)
        finite_float(self.exponent, "mask exponent", minimum=.25, maximum=4)
        for name, n in (
            ("temporal_smooth", self.temporal_smooth),
            ("frequency_smooth", self.frequency_smooth),
        ):
            if type(n) is not int or not 1 <= n <= 31 or n % 2 == 0:
                raise AudioContractError(
                    name + " must be odd and between 1 and 31"
                )


def _validate_spectrum(spec: Spectrogram) -> Spectrogram:
    if not isinstance(spec, Spectrogram):
        raise AudioContractError("Expected finite Spectrogram")
    return spec


def estimate_noise_profile(spectrum: Spectrogram, *,
                           start_frame: int = 0, stop_frame: int = 8,
                           statistic: str = "median") -> np.ndarray:
    """Estimate per-channel/per-bin noise from EXPLICIT selected quiet frames.

    This function does not guess which frames contain only background noise.
    User/researcher must select an appropriate source region and visually or
    aurally review the result before interpreting candidate quality.
    """
    spec = _validate_spectrum(spectrum)
    if (type(start_frame) is not int or type(stop_frame) is not int
            or not 0 <= start_frame < stop_frame <= spec.frames):
        raise AudioContractError("Noise profile frame range is outside STFT")
    if statistic not in ("median", "mean", "maximum"):
        raise AudioContractError("Unsupported noise-profile estimator")
    magnitude = spec.magnitude()[:, :, start_frame:stop_frame]
    if statistic == "median":
        profile = np.median(magnitude, axis=2)
    elif statistic == "mean":
        profile = np.mean(magnitude, axis=2)
    else:
        profile = np.max(magnitude, axis=2)
    if not np.isfinite(profile).all() or np.any(profile < 0):
        raise AudioContractError("Nonfinite spectral noise profile")
    return np.asarray(profile, dtype=np.float32)


def _median_axis(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """A small odd-window median operator with bounded memory copies."""
    if type(radius) is not int or not 0 <= radius <= 15:
        raise AudioContractError("Median radius exceeds reference DSP bound")
    if radius == 0:
        return values.copy()
    if values.size * 8 * (2 * radius + 1) > MAX_BYTES:
        raise AudioLimitError("Median filter scratch exceeds DSP memory limit")
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    result = np.empty(values.shape, dtype=np.float64)
    # Avoid a large (window × bins × frames) materialized tensor.
    for i in range(values.shape[axis]):
        selection = [slice(None)] * values.ndim
        selection[axis] = slice(i, i + 2 * radius + 1)
        dest = [slice(None)] * values.ndim
        dest[axis] = i
        result[tuple(dest)] = np.median(padded[tuple(selection)], axis=axis)
    return result


def spectral_gate_mask(spec: Spectrogram, profile: np.ndarray, *,
                       options: SpectralGateOptions = SpectralGateOptions()
                       ) -> np.ndarray:
    """Real-valued 0..1 suppression mask, never interpreted as neural output."""
    s = _validate_spectrum(spec)
    if not isinstance(options, SpectralGateOptions):
        raise AudioContractError("Expected SpectralGateOptions")
    if (not isinstance(profile, np.ndarray)
            or profile.shape != s.spectrum.shape[:2]
            or profile.dtype not in (np.float32, np.float64)
            or not np.isfinite(profile).all()
            or np.any(profile < 0)):
        raise AudioContractError("Noise profile must match channel/bin geometry")
    mag = s.magnitude()
    baseline = options.threshold_ratio * profile.astype(np.float64)
    active = (mag - baseline[:, :, None]) / np.maximum(mag, 1e-12)
    ratio = np.clip(active, 0, 1) ** options.exponent
    mask = options.floor_gain + (1 - options.floor_gain) * ratio
    if options.temporal_smooth > 1:
        mask = _median_axis(mask, options.temporal_smooth // 2, axis=2)
    if options.frequency_smooth > 1:
        mask = _median_axis(mask, options.frequency_smooth // 2, axis=1)
    if np.any((mask < 0) | (mask > 1)) or not np.isfinite(mask).all():
        raise AudioContractError("Spectral gate produced invalid gain mask")
    return mask.astype(np.float32)


def spectral_gate(audio: AudioBuffer, *,
                  noise_start_seconds: float = 0,
                  noise_stop_seconds: float = .05,
                  config: StftConfig | None = None,
                  options: SpectralGateOptions = SpectralGateOptions()
                  ) -> AudioBuffer:
    """Reduce background spectrum, with explicit reference interval.

    The selected reference region is presumed to contain only noise.
    This assumption is NOT validated automatically.
    """
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected canonical AudioBuffer")
    beginning = finite_float(noise_start_seconds, "noise_start_seconds",
                             minimum=0, maximum=audio.seconds)
    end = finite_float(noise_stop_seconds, "noise_stop_seconds",
                       minimum=0, maximum=audio.seconds)
    if end <= beginning:
        raise AudioContractError("Noise interval must have positive duration")
    cfg = config if config is not None else StftConfig()
    if not isinstance(cfg, StftConfig):
        raise AudioContractError("Invalid STFT analysis configuration")
    spec = stft(audio, cfg)
    # Convert sample-frame selection to STFT-frame selection.
    first = round(beginning * audio.sample_rate / cfg.hop_length)
    last = round(end * audio.sample_rate / cfg.hop_length)
    first = min(first, spec.frames - 1)
    last = min(max(first + 1, last), spec.frames)
    profile = estimate_noise_profile(spec, start_frame=first,
                                     stop_frame=last)
    mask = spectral_gate_mask(spec, profile, options=options)
    modified = complex_mask(spec, mask)
    return istft(modified)


@dataclass(frozen=True, slots=True)
class SpectralBand:
    """One explicitly selected absolute Hz control point of a log-gain curve."""

    frequency_hz: float
    gain_db: float

    def __post_init__(self):
        finite_float(self.frequency_hz, "band frequency", minimum=0,
                     maximum=96000)
        finite_float(self.gain_db, "spectral band gain", minimum=-60,
                     maximum=24)


def spectral_eq_mask(spec: Spectrogram, bands: Sequence[SpectralBand]
                     ) -> np.ndarray:
    """Interpolate dB gain points linearly on the actual bin-frequency axis."""
    s = _validate_spectrum(spec)
    if not isinstance(bands, (list, tuple)) or not 2 <= len(bands) <= 32:
        raise AudioContractError("EQ requires 2..32 ordered frequency bands")
    if any(not isinstance(x, SpectralBand) for x in bands):
        raise AudioContractError("Expected SpectralBand control points")
    freq = [x.frequency_hz for x in bands]
    if freq != sorted(set(freq)) or freq[-1] > s.sample_rate / 2:
        raise AudioContractError("EQ control points must be distinct and below Nyquist")
    dbs = np.interp(s.frequencies, freq, [x.gain_db for x in bands])
    gain = np.power(10, dbs / 20)
    if not np.isfinite(gain).all() or np.max(gain) > 16:
        raise AudioContractError("Spectral EQ gain exceeds bounded limit")
    return gain.astype(np.float32)


def spectral_eq(audio: AudioBuffer, bands: Sequence[SpectralBand], *,
                config: StftConfig | None = None) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    spectrum = stft(audio, config if config is not None else StftConfig())
    gains = spectral_eq_mask(spectrum, bands)
    out = complex_mask(spectrum, gains[None, :, None], maximum_gain=16)
    return istft(out)


def spectral_tilt(audio: AudioBuffer, *,
                  db_per_octave: float = -3,
                  pivot_hz: float = 1000,
                  config: StftConfig | None = None) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    slope = finite_float(db_per_octave, "spectral tilt dB/octave",
                         minimum=-12, maximum=12)
    pivot = finite_float(pivot_hz, "spectral tilt pivot",
                         minimum=10, maximum=audio.sample_rate/2)
    spec = stft(audio, config if config is not None else StftConfig())
    ratio = np.maximum(spec.frequencies, 10.0) / pivot
    gain_db = np.clip(slope * np.log2(ratio), -24, 24)
    values = 10 ** (gain_db / 20)
    return istft(complex_mask(spec, values[None, :, None], maximum_gain=16))


def median_harmonic_percussive_masks(spec: Spectrogram, *,
                                     time_filter: int = 9,
                                     frequency_filter: int = 9,
                                     mask_power: float = 2) -> tuple[np.ndarray, np.ndarray]:
    """Median-filter HPSS-style soft masks, not complete source separation.

    Temporal median estimates sustained narrow-bin components; frequency
    median estimates transient broadband components. These are heuristic
    spectral masks and cannot isolate singers or instruments semantically.
    """
    s = _validate_spectrum(spec)
    if (type(time_filter) is not int or not 1 <= time_filter <= 31
            or time_filter % 2 == 0 or type(frequency_filter) is not int
            or not 1 <= frequency_filter <= 31
            or frequency_filter % 2 == 0):
        raise AudioContractError("HPSS median filters must be odd sizes <=31")
    power = finite_float(mask_power, "HPSS mask power", minimum=.5, maximum=8)
    magnitude = s.magnitude().astype(np.float64)
    if magnitude.size * 8 * max(time_filter, frequency_filter) > MAX_BYTES:
        raise AudioLimitError("HPSS median scratch exceeds bounded memory")
    harmonic = _median_axis(magnitude, time_filter // 2, axis=2)
    percussive = _median_axis(magnitude, frequency_filter // 2, axis=1)
    harmonic = np.maximum(harmonic, 0) ** power
    percussive = np.maximum(percussive, 0) ** power
    denominator = harmonic + percussive
    h_mask = np.divide(harmonic, denominator,
                       out=np.full_like(harmonic, .5),
                       where=denominator > 1e-20)
    p_mask = 1 - h_mask
    return h_mask.astype(np.float32), p_mask.astype(np.float32)


def heuristic_harmonic_percussive(audio: AudioBuffer, *,
                                  config: StftConfig | None = None,
                                  time_filter: int = 9,
                                  frequency_filter: int = 9
                                  ) -> tuple[AudioBuffer, AudioBuffer]:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    spec = stft(audio, config if config is not None else StftConfig())
    h, p = median_harmonic_percussive_masks(
        spec, time_filter=time_filter, frequency_filter=frequency_filter
    )
    sustained = istft(complex_mask(spec, h))
    transients = istft(complex_mask(spec, p))
    return sustained, transients


def spectral_operation_report(audio: AudioBuffer,
                              processed: AudioBuffer,
                              operation: str) -> dict:
    if (not isinstance(audio, AudioBuffer) or not isinstance(processed, AudioBuffer)
            or not audio.spec.same_shape(processed.spec)
            or not isinstance(operation, str) or not operation):
        raise AudioContractError("Spectral result/input geometry mismatch")
    from .numerics import stable_peak, stable_rms
    return {
        "operation": operation,
        "sample_rate": audio.sample_rate,
        "frames": audio.frames,
        "channels": audio.channels,
        "input_peak": stable_peak(audio.samples),
        "output_peak": stable_peak(processed.samples),
        "input_rms": stable_rms(audio.samples),
        "output_rms": stable_rms(processed.samples),
        "neural_model_inference": False,
        "perceptual_enhancement_claimed": False,
        "generator_or_training_data_modified": False,
    }
