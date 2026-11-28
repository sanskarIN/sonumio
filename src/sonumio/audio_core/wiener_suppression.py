"""Bounded profile-based Wiener-style spectral suppression.

This offline CPU reference uses explicitly selected noise-only frames to
estimate a spectral power profile, then applies a smooth gain approximating
a regularized Wiener filter. It is not a learned speech denoiser and cannot
guarantee perceptual enhancement or source separation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError
from .stft import Spectrogram, StftConfig, stft, istft, complex_mask
from .spectral_effects import estimate_noise_profile

MAX_WIENER_BYTES = 192 * 1024 * 1024
MAX_SMOOTHING_WORK = 100_000_000


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class WienerSettings:
    n_fft: int = 1024
    hop_frames: int = 256
    noise_start_seconds: float = 0.
    noise_stop_seconds: float = .1
    noise_statistic: str = "median"
    oversubtraction: float = 1.
    minimum_gain: float = .05
    smoothing_time: int = 3
    smoothing_frequency: int = 3
    channel_linked_profile: bool = False
    wet: float = 1.

    def __post_init__(self):
        n = _integer(self.n_fft, "n_fft", 64, 8192)
        _integer(self.hop_frames, "hop_frames", 16, n // 2)
        if self.noise_statistic not in ("median", "mean", "maximum"):
            raise AudioContractError("Invalid noise profile statistic")
        start = finite_float(self.noise_start_seconds, "noise_start_seconds",
                             minimum=0., maximum=120.)
        stop = finite_float(self.noise_stop_seconds, "noise_stop_seconds",
                            minimum=0., maximum=120.)
        if stop <= start:
            raise AudioContractError("Noise profile interval must increase")
        finite_float(self.oversubtraction, "oversubtraction",
                     minimum=.1, maximum=8.)
        finite_float(self.minimum_gain, "minimum_gain", minimum=0., maximum=1.)
        finite_float(self.wet, "wet", minimum=0., maximum=1.)
        for name, value in (
            ("smoothing_time", self.smoothing_time),
            ("smoothing_frequency", self.smoothing_frequency),
        ):
            width = _integer(value, name, 1, 15)
            if not width % 2:
                raise AudioContractError("Wiener smoothing windows must be odd")
        if type(self.channel_linked_profile) is not bool:
            raise AudioContractError("Wiener channel linkage must be boolean")


@dataclass(frozen=True, slots=True)
class NoiseProfile:
    """Immutable per-channel/per-bin estimated noise power."""

    power: np.ndarray
    sample_rate: int
    n_fft: int
    source_start_frame: int
    source_stop_frame: int

    def __post_init__(self):
        data = self.power
        if (not isinstance(data, np.ndarray) or data.ndim != 2
                or not 1 <= data.shape[0] <= 8
                or data.shape[1] != self.n_fft // 2 + 1
                or data.dtype != np.float64
                or not np.isfinite(data).all() or np.any(data < 0)
                or type(self.source_start_frame) is not int
                or type(self.source_stop_frame) is not int
                or not 0 <= self.source_start_frame < self.source_stop_frame):
            raise AudioContractError("Invalid finite noise power profile")
        copy = np.frombuffer(data.tobytes(), dtype=np.float64)
        object.__setattr__(self, "power", copy.reshape(data.shape))


@dataclass(frozen=True, slots=True)
class WienerDiagnostics:
    sample_rate: int
    channels: int
    frames: int
    n_fft: int
    hop_frames: int
    noise_profile_frames: tuple[int, int]
    average_gain: float
    minimum_applied_gain: float
    maximum_applied_gain: float
    noise_profile_total_power: float
    input_peak: float
    output_peak: float
    input_rms: float
    output_rms: float
    protocol: str = "sonumio-regularized-wiener-v1"

    def as_dict(self):
        result = asdict(self)
        result["model_inference"] = False
        result["noise_only_region_automatically_verified"] = False
        result["perceptual_quality_established"] = False
        result["sample_clock_changed"] = False
        return result


@dataclass(frozen=True, slots=True)
class WienerResult:
    audio: AudioBuffer
    profile: NoiseProfile
    report: WienerDiagnostics


def _profile_interval(spec: Spectrogram, options: WienerSettings):
    if not isinstance(spec, Spectrogram) or not isinstance(options, WienerSettings):
        raise AudioContractError("Expected Spectrogram and WienerSettings")
    if options.noise_stop_seconds > spec.original_frames / spec.sample_rate:
        raise AudioContractError("Noise reference interval exceeds source duration")
    first = round(options.noise_start_seconds * spec.sample_rate /
                  spec.config.hop_length)
    last = round(options.noise_stop_seconds * spec.sample_rate /
                 spec.config.hop_length)
    if first >= spec.frames:
        raise AudioContractError("Noise profile start is outside STFT frames")
    last = min(spec.frames, max(first + 1, last))
    return first, last


def estimate_wiener_profile(
    spec: Spectrogram,
    options: WienerSettings = WienerSettings(),
) -> NoiseProfile:
    """Explicit noise-only reference, optional channel-linked power envelope."""
    first, last = _profile_interval(spec, options)
    magnitudes = estimate_noise_profile(
        spec, start_frame=first, stop_frame=last,
        statistic=options.noise_statistic,
    )
    values = magnitudes.astype(np.float64) ** 2
    if options.channel_linked_profile:
        linked = np.max(values, axis=0)
        values[:] = linked[None, :]
    return NoiseProfile(
        values, spec.sample_rate, spec.config.n_fft, first, last,
    )


def _smooth(array: np.ndarray, length: int, axis: int) -> np.ndarray:
    if length == 1:
        return array
    if array.size * length > MAX_SMOOTHING_WORK:
        raise AudioLimitError("Wiener smoothing exceeds work budget")
    radius = length // 2
    pad = [(0, 0)] * array.ndim
    pad[axis] = (radius, radius)
    values = np.pad(array, pad, mode="edge")
    result = np.zeros_like(array, dtype=np.float64)
    for offset in range(length):
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(offset, offset + array.shape[axis])
        result += values[tuple(slices)]
    result /= length
    return result


def wiener_gain(
    spec: Spectrogram,
    profile: NoiseProfile,
    options: WienerSettings = WienerSettings(),
) -> np.ndarray:
    """Calculate bounded real amplitude gains, preserving complex phase."""
    if (not isinstance(spec, Spectrogram) or not isinstance(profile, NoiseProfile)
            or not isinstance(options, WienerSettings)):
        raise AudioContractError("Expected spectrum, profile and WienerSettings")
    if (profile.power.shape != spec.spectrum.shape[:2]
            or profile.sample_rate != spec.sample_rate
            or profile.n_fft != spec.config.n_fft):
        raise AudioContractError("Wiener noise profile geometry mismatch")
    channels, bins, frames = spec.spectrum.shape
    required = channels * bins * frames * 8 * 7
    if required > MAX_WIENER_BYTES:
        raise AudioLimitError("Wiener working mask exceeds memory budget")
    observed = np.abs(spec.spectrum) ** 2
    background = profile.power[:, :, None] * options.oversubtraction
    # Nonnegative power subtraction is the familiar Wiener-like
    # suppression approximation, not an optimal stationary estimator.
    numerator = np.maximum(observed - background, 0.)
    ratio = np.divide(
        numerator, np.maximum(observed, 1e-20),
        out=np.zeros_like(numerator),
    )
    gain = np.sqrt(np.clip(ratio, 0, 1))
    gain = _smooth(gain, options.smoothing_time, axis=2)
    gain = _smooth(gain, options.smoothing_frequency, axis=1)
    gain = np.clip(gain, options.minimum_gain, 1.)
    if not np.isfinite(gain).all():
        raise AudioContractError("Wiener mask contains nonfinite values")
    return gain


def suppress_wiener(
    audio: AudioBuffer,
    options: WienerSettings = WienerSettings(),
) -> WienerResult:
    """Profile noise, apply bounded spectral mask, mix dry/wet if requested."""
    if not isinstance(audio, AudioBuffer) or not isinstance(options, WienerSettings):
        raise AudioContractError("Expected AudioBuffer and WienerSettings")
    cfg = StftConfig(
        n_fft=options.n_fft, hop_length=options.hop_frames,
        window="hann", center=True, pad_end=True,
    )
    spec = stft(audio, cfg)
    profile = estimate_wiener_profile(spec, options)
    gain = wiener_gain(spec, profile, options)
    filtered = istft(complex_mask(spec, gain))
    if options.wet == 0:
        out = audio
    elif options.wet == 1:
        out = filtered
    else:
        mixture = (
            audio.samples.astype(np.float64) * (1 - options.wet)
            + filtered.samples.astype(np.float64) * options.wet
        )
        out = audio.copy_with(
            mixture.astype(np.float32), source_note="mixed Wiener suppression",
        )
    def _rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    report = WienerDiagnostics(
        audio.sample_rate, audio.channels, audio.frames,
        cfg.n_fft, cfg.hop_length,
        (profile.source_start_frame, profile.source_stop_frame),
        float(np.mean(gain)), float(np.min(gain)), float(np.max(gain)),
        float(np.sum(profile.power)),
        float(np.max(np.abs(audio.samples))),
        float(np.max(np.abs(out.samples))),
        _rms(audio.samples), _rms(out.samples),
    )
    return WienerResult(out, profile, report)
