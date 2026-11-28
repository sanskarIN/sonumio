"""Interpretable transient-onset and tempo-hypothesis analysis.

A bounded spectral-flux novelty curve and deterministic peak picking provide
sample-clock onset candidates. A normalized lag correlation supplies tempo
hypotheses, not a definitive beat or meter transcription. Silent, irregular
or short signals may yield no tempo. No trained model is used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError
from .features import spectral_flux
from .stft import StftConfig, stft

MAX_ANALYSIS_FRAMES = 32_768
MAX_ONSETS = 2048
MAX_CORRELATION_PAIRS = 12_000_000


def _integer(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise AudioContractError(f"{name} must be integer in [{lo}, {hi}]")
    return value


@dataclass(frozen=True, slots=True)
class RhythmSettings:
    n_fft: int = 512
    hop_frames: int = 128
    smoothing_frames: int = 3
    min_spacing_seconds: float = .08
    min_strength: float = .15
    prominence: float = .08
    neighborhood_frames: int = 7
    min_bpm: float = 50.
    max_bpm: float = 220.
    max_candidates: int = 5

    def __post_init__(self):
        size = _integer(self.n_fft, "n_fft", 64, 4096)
        _integer(self.hop_frames, "hop_frames", 16, size // 2)
        s = _integer(self.smoothing_frames, "smoothing_frames", 1, 15)
        if s % 2 == 0:
            raise AudioContractError("Rhythm smoothing window must be odd")
        neighborhood = _integer(self.neighborhood_frames, "neighborhood_frames", 3, 63)
        if neighborhood % 2 == 0:
            raise AudioContractError("Rhythm neighborhood must be odd")
        _integer(self.max_candidates, "max_candidates", 1, 12)
        for name, value, low, high in (
            ("min_spacing_seconds", self.min_spacing_seconds, .01, 2.),
            ("min_strength", self.min_strength, 0., 1.),
            ("prominence", self.prominence, 0., 1.),
            ("min_bpm", self.min_bpm, 30., 300.),
            ("max_bpm", self.max_bpm, 30., 300.),
        ):
            finite_float(value, name, minimum=low, maximum=high)
        if self.min_bpm >= self.max_bpm:
            raise AudioContractError("Tempo bounds must be strictly increasing")


@dataclass(frozen=True, slots=True)
class OnsetEvent:
    frame: int
    seconds: float
    strength: float


@dataclass(frozen=True, slots=True)
class TempoCandidate:
    bpm: float
    score: float
    lag_frames: int
    interbeat_frames: float


@dataclass(frozen=True, slots=True, eq=False)
class RhythmAnalysis:
    """Immutable normalized novelty curve and bounded structured hypotheses."""

    envelope: np.ndarray
    sample_rate: int
    source_frames: int
    settings: RhythmSettings
    onsets: tuple[OnsetEvent, ...]
    tempos: tuple[TempoCandidate, ...]
    beat_frames: tuple[int, ...]

    def __post_init__(self):
        values = self.envelope
        if (not isinstance(values, np.ndarray) or values.ndim != 1
                or values.size < 1 or values.size > MAX_ANALYSIS_FRAMES
                or values.dtype != np.float64 or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(values > 1 + 1e-10)):
            raise AudioContractError("Rhythm envelope must be finite and normalized")
        if (not isinstance(self.settings, RhythmSettings)
                or type(self.sample_rate) is not int
                or type(self.source_frames) is not int
                or self.source_frames < 1):
            raise AudioContractError("Invalid rhythm metadata")
        if (not isinstance(self.onsets, tuple) or len(self.onsets) > MAX_ONSETS
                or any(not isinstance(x, OnsetEvent) for x in self.onsets)
                or not isinstance(self.tempos, tuple)
                or any(not isinstance(x, TempoCandidate) for x in self.tempos)
                or not isinstance(self.beat_frames, tuple)
                or any(type(x) is not int or not 0 <= x < self.source_frames
                       for x in self.beat_frames)):
            raise AudioContractError("Invalid rhythm analysis events")
        frozen = np.frombuffer(values.tobytes(), dtype=np.float64)
        object.__setattr__(self, "envelope", frozen)

    def as_dict(self) -> dict:
        return {
            "protocol": "sonumio-rhythm-hypotheses-v1",
            "sample_rate": self.sample_rate,
            "source_frames": self.source_frames,
            "analysis_hop_frames": self.settings.hop_frames,
            "analysis_n_fft": self.settings.n_fft,
            "envelope_frames": len(self.envelope),
            "onsets": [asdict(v) for v in self.onsets],
            "tempos": [asdict(v) for v in self.tempos],
            "beat_frames": list(self.beat_frames),
            "model_inference": False,
            "meter_inferred": False,
            "tempo_is_hypothesis": True,
            "perceptual_quality_established": False,
        }


def _preflight(audio: AudioBuffer, settings: RhythmSettings) -> int:
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, RhythmSettings):
        raise AudioContractError("Expected AudioBuffer and RhythmSettings")
    frames = (audio.frames + settings.n_fft) // settings.hop_frames + 2
    if frames > MAX_ANALYSIS_FRAMES:
        raise AudioLimitError("Rhythm analysis exceeds frame-count limit")
    # STFT preflights its own memory; this one also bounds novelty and
    # temporal-correlation loops independently of spectral allocations.
    lags = math.ceil(60 * audio.sample_rate / settings.hop_frames / settings.min_bpm)
    if frames * min(frames, lags) > MAX_CORRELATION_PAIRS:
        raise AudioLimitError("Rhythm tempo correlation exceeds work limit")
    return frames


def novelty_envelope(
    audio: AudioBuffer,
    settings: RhythmSettings = RhythmSettings(),
) -> np.ndarray:
    """Bounded positive spectral-flux novelty on the STFT hop clock."""
    _preflight(audio, settings)
    cfg = StftConfig(
        n_fft=settings.n_fft, hop_length=settings.hop_frames,
        window="hann", center=True, pad_end=True,
    )
    spectrum = stft(audio, cfg)
    flux = np.mean(spectral_flux(spectrum), axis=0).astype(np.float64)
    if settings.smoothing_frames > 1:
        width = settings.smoothing_frames
        # Edge padding avoids suppressing the beginning of an analysis.
        padded = np.pad(flux, (width // 2, width // 2), mode="edge")
        flux = np.convolve(padded, np.ones(width) / width, mode="valid")
    baseline = float(np.median(flux))
    flux = np.maximum(flux - baseline, 0.)
    peak = float(np.max(flux))
    if peak:
        flux /= peak
    if not np.isfinite(flux).all():
        raise AudioContractError("Spectral onset envelope contains nonfinite values")
    return flux


def pick_onsets(
    envelope: np.ndarray, sample_rate: int, source_frames: int,
    settings: RhythmSettings = RhythmSettings(),
) -> tuple[OnsetEvent, ...]:
    """Greedy peak picking with deterministic strength/time tie-breaking."""
    from .contracts import check_rate
    check_rate(sample_rate)
    if (not isinstance(envelope, np.ndarray) or envelope.ndim != 1
            or envelope.dtype not in (np.float32, np.float64)
            or not 3 <= len(envelope) <= MAX_ANALYSIS_FRAMES
            or not np.isfinite(envelope).all()
            or np.any(envelope < 0) or np.any(envelope > 1 + 1e-10)
            or type(source_frames) is not int or source_frames < 1
            or not isinstance(settings, RhythmSettings)):
        raise AudioContractError("Invalid bounded onset envelope")
    half = settings.neighborhood_frames // 2
    minimum_gap = max(1, round(
        settings.min_spacing_seconds * sample_rate / settings.hop_frames
    ))
    candidate = []
    for i in range(1, len(envelope) - 1):
        value = float(envelope[i])
        if value < settings.min_strength:
            continue
        # Prefer the earliest hop on equal-height plateaus.
        if value <= envelope[i - 1] or value < envelope[i + 1]:
            continue
        frame = i * settings.hop_frames
        if frame >= source_frames:
            continue
        a, b = max(0, i - half), min(len(envelope), i + half + 1)
        local = float(np.median(envelope[a:b]))
        if value - local >= settings.prominence:
            candidate.append((i, value))
    candidate.sort(key=lambda pair: (-pair[1], pair[0]))
    accepted = []
    for index, strength in candidate:
        if all(abs(index - old) >= minimum_gap for old, _ in accepted):
            accepted.append((index, strength))
            if len(accepted) > MAX_ONSETS:
                raise AudioLimitError("Rhythm analysis contains too many onsets")
    accepted.sort()
    return tuple(
        OnsetEvent(
            index * settings.hop_frames,
            index * settings.hop_frames / sample_rate,
            strength,
        )
        for index, strength in accepted
    )


def estimate_tempo(
    envelope: np.ndarray, sample_rate: int,
    onsets: tuple[OnsetEvent, ...],
    settings: RhythmSettings = RhythmSettings(),
) -> tuple[TempoCandidate, ...]:
    """Return normalized lag-correlation peaks within explicit BPM bounds."""
    from .contracts import check_rate
    check_rate(sample_rate)
    if (not isinstance(settings, RhythmSettings)
            or not isinstance(envelope, np.ndarray)
            or envelope.ndim != 1 or envelope.dtype not in (np.float32, np.float64)
            or not np.isfinite(envelope).all()
            or not isinstance(onsets, tuple)
            or any(not isinstance(x, OnsetEvent) for x in onsets)):
        raise AudioContractError("Invalid tempo candidate input")
    if len(onsets) < 3 or float(np.max(envelope)) <= 0:
        return ()
    low = max(1, math.ceil(60 * sample_rate /
                           (settings.hop_frames * settings.max_bpm)))
    high = math.floor(60 * sample_rate /
                      (settings.hop_frames * settings.min_bpm))
    high = min(high, len(envelope) - 2)
    if low > high:
        return ()
    if (high - low + 1) * len(envelope) > MAX_CORRELATION_PAIRS:
        raise AudioLimitError("Tempo autocorrelation exceeds work limit")
    centered = envelope.astype(np.float64) - float(np.mean(envelope))
    scores = []
    for lag in range(low, high + 1):
        left, right = centered[:-lag], centered[lag:]
        denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
        value = 0. if denominator <= 1e-15 else float(np.dot(left, right) / denominator)
        # A negligible deterministic prior breaks octave ties in favor of
        # directly repeated beats, while retaining observable correlation.
        value = max(0., value)
        scores.append((lag, value))
    if not scores or max(v for _, v in scores) <= .05:
        return ()
    # Local maxima only: avoid five nearly identical BPM estimates.
    local = []
    for j, item in enumerate(scores):
        strength = item[1]
        before = scores[j - 1][1] if j else -1
        after = scores[j + 1][1] if j + 1 < len(scores) else -1
        if strength >= before and strength >= after and strength > .05:
            local.append(item)
    local.sort(key=lambda pair: (-pair[1], pair[0]))
    selected = []
    for lag, score in local:
        bpm = 60 * sample_rate / (settings.hop_frames * lag)
        if any(abs(bpm - x.bpm) < 3 for x in selected):
            continue
        selected.append(TempoCandidate(
            bpm, score, lag, 60 * sample_rate / bpm,
        ))
        if len(selected) == settings.max_candidates:
            break
    return tuple(selected)


def _beat_grid(
    onsets: tuple[OnsetEvent, ...],
    best: TempoCandidate | None,
    source_frames: int,
) -> tuple[int, ...]:
    if best is None or not onsets:
        return ()
    period = best.interbeat_frames
    # Use the earliest onset as the conservative beat-phase anchor.
    anchor = onsets[0].frame
    before = math.ceil(anchor / period)
    after = math.ceil((source_frames - anchor) / period)
    positions = {
        round(anchor + k * period)
        for k in range(-before, after + 1)
        if 0 <= round(anchor + k * period) < source_frames
    }
    return tuple(sorted(positions))


def analyze_rhythm(
    audio: AudioBuffer,
    settings: RhythmSettings = RhythmSettings(),
) -> RhythmAnalysis:
    """Produce onset/tempo hypotheses without modifying or moving the audio."""
    novelty = novelty_envelope(audio, settings)
    onsets = pick_onsets(novelty, audio.sample_rate, audio.frames, settings)
    tempo = estimate_tempo(novelty, audio.sample_rate, onsets, settings)
    grid = _beat_grid(onsets, tempo[0] if tempo else None, audio.frames)
    return RhythmAnalysis(
        novelty, audio.sample_rate, audio.frames, settings, onsets, tempo, grid,
    )
