"""Bounded difference-function F0 analysis for monophonic audio.

Implements an offline YIN-style cumulative-mean normalized difference
(CMNDF), deterministic local trough selection and optional parabolic lag
refinement. This is a reference *pitch hypothesis* estimator, not trained
polyphonic transcription, note labeling or singer identity recognition.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError

MAX_PITCH_FRAMES = 16_384
MAX_PITCH_PRODUCTS = 200_000_000
MAX_PITCH_WORK_BYTES = 192 * 1024 * 1024
EPSILON = 1e-14


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class PitchSettings:
    frame_frames: int = 1024
    hop_frames: int = 256
    minimum_hz: float = 65.
    maximum_hz: float = 1400.
    yin_threshold: float = .15
    fallback_threshold: float = .3
    silence_dbfs: float = -60.
    interpolate_lag: bool = True

    def __post_init__(self):
        size = _integer(self.frame_frames, "pitch frame", 128, 4096)
        _integer(self.hop_frames, "pitch hop", 32, size)
        minimum = finite_float(self.minimum_hz, "minimum_hz",
                               minimum=20., maximum=2000.)
        maximum = finite_float(self.maximum_hz, "maximum_hz",
                               minimum=20., maximum=8000.)
        if maximum <= minimum:
            raise AudioContractError("Pitch frequency interval must increase")
        threshold = finite_float(self.yin_threshold, "YIN threshold",
                                 minimum=.01, maximum=.8)
        fallback = finite_float(self.fallback_threshold, "fallback threshold",
                                minimum=.01, maximum=.95)
        if fallback < threshold:
            raise AudioContractError("Pitch fallback cannot be below YIN threshold")
        finite_float(self.silence_dbfs, "silence threshold",
                     minimum=-120., maximum=-10.)
        if type(self.interpolate_lag) is not bool:
            raise AudioContractError("Pitch lag interpolation flag must be boolean")


@dataclass(frozen=True, slots=True)
class PitchFrame:
    center_frame: int
    seconds: float
    frequency_hz: float | None
    confidence: float
    selected_channel: int
    rms: float

    @property
    def voiced(self) -> bool:
        return self.frequency_hz is not None


@dataclass(frozen=True, slots=True)
class PitchSummary:
    sample_rate: int
    source_frames: int
    channels: int
    analysis_frames: int
    voiced_frames: int
    voiced_fraction: float
    median_pitch_hz: float | None
    minimum_pitch_hz: float | None
    maximum_pitch_hz: float | None
    median_confidence: float
    protocol: str = "sonumio-yin-monophonic-v1"

    def as_dict(self):
        value = asdict(self)
        value["model_inference"] = False
        value["polyphonic_transcription_claimed"] = False
        value["perceptual_quality_established"] = False
        return value


@dataclass(frozen=True, slots=True)
class PitchTrack:
    frames: tuple[PitchFrame, ...]
    settings: PitchSettings
    summary: PitchSummary

    def as_dict(self):
        return {
            "protocol": self.summary.protocol,
            "settings": asdict(self.settings),
            "summary": self.summary.as_dict(),
            "frames": [asdict(v) | {"voiced": v.voiced} for v in self.frames],
            "model_inference": False,
        }


@dataclass(frozen=True, slots=True)
class PitchGeometry:
    sample_rate: int
    channels: int
    input_frames: int
    frame: int
    hop: int
    minimum_hz: float
    maximum_hz: float

    def __post_init__(self):
        from .contracts import check_rate
        check_rate(self.sample_rate)
        _integer(self.channels, "pitch channels", 1, 8)
        _integer(self.input_frames, "pitch input frames", 1, 120 * self.sample_rate)
        _integer(self.frame, "pitch analysis window", 128, 4096)
        _integer(self.hop, "pitch hop", 32, self.frame)
        if self.minimum_lag > self.maximum_lag:
            raise AudioContractError("Pitch interval has no resolvable lags")
        if self.estimated_products > MAX_PITCH_PRODUCTS:
            raise AudioLimitError("Pitch estimation exceeds difference work budget")
        if self.estimated_work_bytes > MAX_PITCH_WORK_BYTES:
            raise AudioLimitError("Pitch analysis exceeds bounded work memory")
        if self.output_frames > MAX_PITCH_FRAMES:
            raise AudioLimitError("Pitch track contains too many analysis frames")

    @property
    def output_frames(self):
        return 1 + (self.input_frames - 1) // self.hop

    @property
    def minimum_lag(self):
        return max(2, math.ceil(self.sample_rate / self.maximum_hz))

    @property
    def maximum_lag(self):
        return math.floor(self.sample_rate / self.minimum_hz)

    @property
    def estimated_products(self):
        return (self.output_frames * self.frame *
                (self.maximum_lag - self.minimum_lag + 1))

    @property
    def estimated_work_bytes(self):
        # One zero-padded source, one channel/window, cross-correlation,
        # squared-energy cumsum and CMNDF scratch, with conservative margin.
        return 8 * (
            self.channels * (self.input_frames + self.frame + 2 * self.maximum_lag)
            + self.output_frames * 12
            + 16 * (self.frame + self.maximum_lag)
        ) * 3


def _difference(segment: np.ndarray, frame: int, maximum_lag: int) -> np.ndarray:
    """Compute YIN difference d[tau] using valid cross correlation and energy.

    For all tau, d[tau] = ||x[0:W] - x[tau:tau+W]||².
    A fixed W is important: otherwise the difference function is biased
    toward larger lags through fewer comparisons.
    """
    if len(segment) < frame + maximum_lag:
        raise AudioContractError("Pitch difference segment is incomplete")
    reference = segment[:frame]
    correlation = np.correlate(segment, reference, mode="valid")
    if len(correlation) < maximum_lag + 1:
        raise AudioContractError("Pitch lag cross-correlation incomplete")
    squares = segment * segment
    cumulative = np.concatenate(([0.], np.cumsum(squares, dtype=np.float64)))
    offsets = np.arange(maximum_lag + 1)
    shifted = cumulative[frame + offsets] - cumulative[offsets]
    energy = float(np.dot(reference, reference))
    difference = np.maximum(energy + shifted - 2 * correlation[:len(offsets)], 0.)
    difference[0] = 0.
    return difference


def _cmndf(difference: np.ndarray) -> np.ndarray:
    if (not isinstance(difference, np.ndarray) or difference.ndim != 1
            or len(difference) < 3 or difference.dtype not in (np.float32, np.float64)
            or not np.isfinite(difference).all()
            or np.any(difference < 0)):
        raise AudioContractError("Invalid pitch difference curve")
    cumulative = np.cumsum(difference, dtype=np.float64)
    cumulative[0] = 0.
    normalized = np.ones_like(difference, dtype=np.float64)
    lags = np.arange(1, len(difference))
    normalized[1:] = np.divide(
        difference[1:] * lags,
        cumulative[1:],
        out=np.ones(len(lags), dtype=np.float64),
        where=cumulative[1:] > EPSILON,
    )
    return np.clip(normalized, 0., 10.)


def _select_lag(
    curve: np.ndarray, minimum: int, maximum: int,
    threshold: float, fallback: float,
) -> int | None:
    if minimum > maximum or maximum >= len(curve):
        raise AudioContractError("Pitch lag selection outside difference curve")
    accepted = None
    for tau in range(minimum, maximum + 1):
        if curve[tau] <= threshold:
            # Descend to first trough once the YIN threshold is crossed.
            while tau < maximum and curve[tau + 1] <= curve[tau]:
                tau += 1
            accepted = tau
            break
    if accepted is not None:
        return accepted
    values = curve[minimum:maximum + 1]
    index = int(np.argmin(values)) + minimum
    return index if float(curve[index]) <= fallback else None


def _refine_lag(curve: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(curve) - 1:
        return float(lag)
    y0, y1, y2 = (float(curve[lag - 1]), float(curve[lag]),
                  float(curve[lag + 1]))
    denominator = y0 - 2 * y1 + y2
    if denominator <= EPSILON:
        return float(lag)
    shift = .5 * (y0 - y2) / denominator
    return lag + min(.5, max(-.5, shift))


def _window(audio: AudioBuffer, index: int, geometry: PitchGeometry):
    """Choose strongest-RMS channel instead of downmixing opposite phase."""
    position = index * geometry.hop
    start = position - geometry.frame // 2
    end = start + geometry.frame + geometry.maximum_lag
    # Noncausal offline context, zero-padded in a predictable way.
    data = np.zeros(
        (audio.channels, geometry.frame + geometry.maximum_lag),
        dtype=np.float64,
    )
    lo, hi = max(start, 0), min(end, audio.frames)
    if lo < hi:
        data[:, lo - start:hi - start] = audio.samples[:, lo:hi]
    rms = np.sqrt(np.mean(data[:, :geometry.frame] ** 2, axis=1))
    channel = int(np.argmax(rms))
    return data[channel], channel, float(rms[channel])


def track_pitch(
    audio: AudioBuffer,
    settings: PitchSettings = PitchSettings(),
) -> PitchTrack:
    """Return independent monophonic pitch hypotheses at fixed hop centers."""
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, PitchSettings):
        raise AudioContractError("Expected AudioBuffer and PitchSettings")
    geometry = PitchGeometry(
        audio.sample_rate, audio.channels, audio.frames,
        settings.frame_frames, settings.hop_frames,
        settings.minimum_hz, settings.maximum_hz,
    )
    floor = 10 ** (settings.silence_dbfs / 20)
    events = []
    for index in range(geometry.output_frames):
        data, channel, rms = _window(audio, index, geometry)
        frequency = None
        confidence = 0.
        if rms >= floor:
            difference = _difference(data, geometry.frame, geometry.maximum_lag)
            normalized = _cmndf(difference)
            lag = _select_lag(
                normalized, geometry.minimum_lag, geometry.maximum_lag,
                settings.yin_threshold, settings.fallback_threshold,
            )
            if lag is not None:
                estimate = (_refine_lag(normalized, lag)
                            if settings.interpolate_lag else float(lag))
                candidate = geometry.sample_rate / estimate
                if settings.minimum_hz <= candidate <= settings.maximum_hz:
                    frequency = float(candidate)
                    confidence = max(0., min(1., 1. - float(normalized[lag])))
        center = index * geometry.hop
        events.append(PitchFrame(
            center, center / audio.sample_rate, frequency,
            confidence, channel, rms,
        ))
    voiced = [event for event in events if event.voiced]
    freqs = [event.frequency_hz for event in voiced]
    summary = PitchSummary(
        audio.sample_rate, audio.frames, audio.channels, len(events),
        len(voiced), len(voiced) / len(events),
        float(np.median(freqs)) if freqs else None,
        float(min(freqs)) if freqs else None,
        float(max(freqs)) if freqs else None,
        float(np.median([event.confidence for event in voiced])) if voiced else 0.,
    )
    return PitchTrack(tuple(events), settings, summary)
