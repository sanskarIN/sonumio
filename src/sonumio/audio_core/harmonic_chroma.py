"""Deterministic pitch-class spectra and triad hypotheses for local audio.

Maps valid STFT bins to pitch classes in equal temperament and scores
interpretable major/minor templates. This is not a neural chord recognizer,
key detector, transcription system or calibrated music-theory truth.
Harmonics, inversions and polyphonic instruments create ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError
from .stft import Spectrogram, StftConfig, stft

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITIES = ("major", "minor")
MAX_FRAMES = 12000
MAX_CHROMA_PRODUCTS = 64_000_000
MAX_CHROMA_BYTES = 192 * 1024 * 1024


def _integer(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise AudioContractError(f"{name} must be integer in [{lo}, {hi}]")
    return value


@dataclass(frozen=True, slots=True)
class ChromaSettings:
    n_fft: int = 2048
    hop_frames: int = 512
    lowest_hz: float = 65.
    highest_hz: float = 4000.
    reference_a_hz: float = 440.
    magnitude_power: float = 1.
    minimum_rms: float = .002
    minimum_triad_energy: float = .07
    minimum_chord_score: float = .55
    minimum_score_gap: float = .015
    minimum_segment_frames: int = 2

    def __post_init__(self):
        n = _integer(self.n_fft, "n_fft", 256, 8192)
        _integer(self.hop_frames, "hop_frames", 32, n // 2)
        low = finite_float(self.lowest_hz, "lowest_hz", minimum=20., maximum=4000.)
        high = finite_float(self.highest_hz, "highest_hz", minimum=20., maximum=96000.)
        if high <= low:
            raise AudioContractError("Chroma frequency bounds must increase")
        finite_float(self.reference_a_hz, "reference_a_hz", minimum=400., maximum=480.)
        finite_float(self.magnitude_power, "magnitude_power", minimum=.5, maximum=2.)
        finite_float(self.minimum_rms, "minimum_rms", minimum=0., maximum=1.)
        finite_float(self.minimum_triad_energy, "minimum_triad_energy",
                     minimum=0., maximum=.3)
        finite_float(self.minimum_chord_score, "minimum_chord_score",
                     minimum=.3, maximum=1.)
        finite_float(self.minimum_score_gap, "minimum_score_gap",
                     minimum=0., maximum=.5)
        _integer(self.minimum_segment_frames, "minimum_segment_frames", 1, 256)


@dataclass(frozen=True, slots=True)
class ChordHypothesis:
    root: str
    quality: str
    score: float
    score_gap: float

    @property
    def symbol(self) -> str:
        return self.root if self.quality == "major" else self.root + "m"


@dataclass(frozen=True, slots=True)
class TonalSegment:
    start_frame: int
    stop_frame: int
    symbol: str
    mean_score: float


@dataclass(frozen=True, slots=True, eq=False)
class TonalAnalysis:
    """Read-only pitch class matrix [12, analysis_frames]."""

    chroma: np.ndarray
    sample_rate: int
    source_frames: int
    hop_frames: int
    hypotheses: tuple[ChordHypothesis | None, ...]
    segments: tuple[TonalSegment, ...]
    protocol: str = "sonumio-chroma-triad-hypotheses-v1"

    def __post_init__(self):
        data = self.chroma
        if (not isinstance(data, np.ndarray) or data.ndim != 2
                or data.shape[0] != 12 or not 1 <= data.shape[1] <= MAX_FRAMES
                or data.dtype != np.float64 or not np.isfinite(data).all()
                or np.any(data < 0) or np.any(data > 1 + 1e-10)
                or not isinstance(self.hypotheses, tuple)
                or len(self.hypotheses) != data.shape[1]
                or not isinstance(self.segments, tuple)
                or any(not isinstance(x, TonalSegment) for x in self.segments)):
            raise AudioContractError("Invalid bounded tonal analysis metadata")
        if (type(self.sample_rate) is not int or type(self.source_frames) is not int
                or self.sample_rate <= 0 or self.source_frames <= 0
                or type(self.hop_frames) is not int or self.hop_frames <= 0):
            raise AudioContractError("Invalid tonal sample clock")
        frozen = np.frombuffer(data.tobytes(), dtype=np.float64)
        object.__setattr__(self, "chroma", frozen.reshape(data.shape))

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "sample_rate": self.sample_rate,
            "source_frames": self.source_frames,
            "hop_frames": self.hop_frames,
            "analysis_frames": self.chroma.shape[1],
            "overall_chroma": self.mean_chroma().tolist(),
            "segments": [asdict(s) for s in self.segments],
            "chord_hypotheses": [
                None if h is None else {**asdict(h), "symbol": h.symbol}
                for h in self.hypotheses
            ],
            "model_inference": False,
            "chord_is_hypothesis": True,
            "key_signature_inferred": False,
            "perceptual_quality_established": False,
        }

    def mean_chroma(self):
        profile = np.mean(self.chroma, axis=1)
        total = float(np.sum(profile))
        return profile / total if total > 0 else profile


def _geometry(audio: AudioBuffer, settings: ChromaSettings):
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, ChromaSettings):
        raise AudioContractError("Expected AudioBuffer and ChromaSettings")
    if settings.highest_hz > audio.sample_rate / 2:
        raise AudioContractError("Chroma highest_hz exceeds Nyquist")
    count = 1 + (audio.frames + settings.n_fft - 1) // settings.hop_frames
    bins = settings.n_fft // 2 + 1
    if count > MAX_FRAMES or bins * count * audio.channels > MAX_CHROMA_PRODUCTS:
        raise AudioLimitError("Chroma spectral analysis exceeds geometry budget")
    # Spectral engine enforces its own allocation limits; this also bounds
    # chroma and audio scratch during bin folding and template scoring.
    estimate = (
        count * bins * audio.channels * 32
        + count * 12 * 8 * 12
        + audio.frames * audio.channels * 8
    )
    if estimate > MAX_CHROMA_BYTES:
        raise AudioLimitError("Chroma analysis exceeds work memory")
    return count


def _pitch_mapping(frequencies: np.ndarray, options: ChromaSettings):
    if (not isinstance(frequencies, np.ndarray) or frequencies.ndim != 1
            or not np.isfinite(frequencies).all()
            or np.any(frequencies < 0)):
        raise AudioContractError("Expected nonnegative FFT bin centers")
    valid = (frequencies >= options.lowest_hz) & (
        frequencies <= options.highest_hz
    )
    if not np.any(valid):
        raise AudioContractError("No FFT bins fall within chroma bounds")
    indices = np.flatnonzero(valid)
    semitones = 69 + 12 * np.log2(frequencies[indices] / options.reference_a_hz)
    # Fractional semitone weights provide gradual tuning tolerance instead
    # of assigning a bin abruptly to a discrete note class.
    base = np.floor(semitones)
    upper_fraction = semitones - base
    lower_class = base.astype(np.int64) % 12
    upper_class = (lower_class + 1) % 12
    return indices, lower_class, upper_class, upper_fraction


def chroma_from_spectrum(
    spectrum: Spectrogram, settings: ChromaSettings = ChromaSettings(),
) -> np.ndarray:
    """Fold channel-separated magnitude into bounded normalized pitch classes."""
    if not isinstance(spectrum, Spectrogram) or not isinstance(settings, ChromaSettings):
        raise AudioContractError("Expected Spectrogram and ChromaSettings")
    if settings.highest_hz > spectrum.sample_rate / 2:
        raise AudioContractError("Chroma frequency exceeds source Nyquist")
    indices, left, right, fraction = _pitch_mapping(spectrum.frequencies, settings)
    magnitude = np.abs(spectrum.spectrum[:, indices, :])
    # Aggregate magnitudes rather than waveforms so opposite-polarity stereo
    # channels cannot cancel the pitch-class evidence.
    energy = np.mean(magnitude, axis=0).astype(np.float64)
    if settings.magnitude_power != 1:
        energy **= settings.magnitude_power
    classes = np.zeros((12, spectrum.frames), dtype=np.float64)
    for index in range(len(indices)):
        classes[left[index]] += energy[index] * (1 - fraction[index])
        classes[right[index]] += energy[index] * fraction[index]
    total = np.sum(classes, axis=0)
    np.divide(classes, total[None, :], out=classes,
              where=total[None, :] > 1e-15)
    if not np.isfinite(classes).all():
        raise AudioContractError("Chroma extraction generated nonfinite energies")
    return classes


def chord_templates() -> tuple[tuple[str, str, np.ndarray], ...]:
    """Deterministic 12-root major/minor triad membership masks."""
    result = []
    for root in range(12):
        for quality, third in (("major", 4), ("minor", 3)):
            mask = np.zeros(12, dtype=np.float64)
            mask[[root, (root + third) % 12, (root + 7) % 12]] = 1.
            result.append((NAMES[root], quality, mask))
    return tuple(result)


def score_triad(
    vector: np.ndarray, settings: ChromaSettings = ChromaSettings(),
) -> ChordHypothesis | None:
    """Return a triad only if all member classes have useful evidence."""
    if (not isinstance(vector, np.ndarray) or vector.shape != (12,)
            or vector.dtype not in (np.float32, np.float64)
            or not np.isfinite(vector).all() or np.any(vector < 0)):
        raise AudioContractError("Expected finite nonnegative pitch-class vector")
    if not isinstance(settings, ChromaSettings):
        raise AudioContractError("Expected ChromaSettings")
    total = float(np.sum(vector))
    if total < 1e-12:
        return None
    values = vector / total
    templates = chord_templates()
    candidates = []
    for root, quality, mask in templates:
        members = values[mask == 1]
        if float(np.min(members)) < settings.minimum_triad_energy:
            continue
        score = float(np.sum(members))
        candidates.append((score, root, quality))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], NAMES.index(x[1]), x[2]))
    best, root, quality = candidates[0]
    second = candidates[1][0] if len(candidates) > 1 else 0.
    gap = best - second
    if best < settings.minimum_chord_score or gap < settings.minimum_score_gap:
        return None
    return ChordHypothesis(root, quality, best, gap)


def _segments(
    hypotheses: tuple[ChordHypothesis | None, ...],
    hop: int, frames: int, minimum_frames: int,
) -> tuple[TonalSegment, ...]:
    events = []
    i = 0
    while i < len(hypotheses):
        entry = hypotheses[i]
        j = i + 1
        while j < len(hypotheses) and (
            (entry is None and hypotheses[j] is None) or (
                entry is not None and hypotheses[j] is not None
                and entry.symbol == hypotheses[j].symbol
            )
        ):
            j += 1
        start = min(i * hop, frames)
        stop = min(j * hop, frames)
        if entry is not None and j - i >= minimum_frames and stop > start:
            score = float(np.mean([
                x.score for x in hypotheses[i:j] if x is not None
            ]))
            events.append(TonalSegment(start, stop, entry.symbol, score))
        i = j
    return tuple(events)


def analyze_chroma(
    audio: AudioBuffer,
    settings: ChromaSettings = ChromaSettings(),
) -> TonalAnalysis:
    """Report sample-clock triad hypotheses without modifying input audio."""
    _geometry(audio, settings)
    cfg = StftConfig(
        n_fft=settings.n_fft, hop_length=settings.hop_frames,
        window="hann", center=True, pad_end=True,
    )
    spec = stft(audio, cfg)
    classes = chroma_from_spectrum(spec, settings)
    # RMS energy is determined from the original audio, not normalized
    # chroma, otherwise extremely quiet input may appear confident.
    candidates = []
    half = settings.n_fft // 2
    for index in range(spec.frames):
        center = index * settings.hop_frames
        start = max(0, center - half)
        stop = min(audio.frames, center + half)
        if stop <= start:
            candidates.append(None)
            continue
        data = audio.samples[:, start:stop].astype(np.float64)
        rms = float(np.sqrt(np.mean(data * data)))
        if rms < settings.minimum_rms:
            candidates.append(None)
        else:
            candidates.append(score_triad(classes[:, index], settings))
    events = tuple(candidates)
    segments = _segments(
        events, settings.hop_frames, audio.frames,
        settings.minimum_segment_frames,
    )
    return TonalAnalysis(classes, audio.sample_rate, audio.frames,
                         settings.hop_frames, events, segments)
