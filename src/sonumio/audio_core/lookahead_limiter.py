"""Linked-channel offline lookahead sample-peak limiting.

Unlike hard clipping, each frame receives one common scalar gain.
The backward envelope anticipates the strongest future attenuation
with a bounded linear attack slope. The forward envelope smooths
release without ever exceeding the per-frame safe gain. This processor
does not certify reconstructed analog or intersample peaks.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np

from .contracts import AudioBuffer, check_rate, finite_float
from .errors import AudioContractError, AudioLimitError, AudioProcessingError

MAX_LIMITER_FRAMES = 4_000_000
MAX_LIMITER_BYTES = 128*1024*1024


@dataclass(frozen=True, slots=True)
class LookaheadSettings:
    ceiling_dbfs: float = -1.
    lookahead_ms: int = 5
    release_ms: int = 80

    def __post_init__(self):
        finite_float(self.ceiling_dbfs, "sample peak ceiling", minimum=-40, maximum=0)
        if type(self.lookahead_ms) is not int or not 0 <= self.lookahead_ms <= 100:
            raise AudioContractError("Lookahead must be integer milliseconds in [0,100]")
        if type(self.release_ms) is not int or not 0 <= self.release_ms <= 2000:
            raise AudioContractError("Release must be integer milliseconds in [0,2000]")

    @property
    def ceiling(self):
        # The canonical output uses float32; choose a representable cap
        # no higher than the requested linear amplitude.
        value = 10**(self.ceiling_dbfs/20)
        rounded = np.float32(value)
        if float(rounded) > value:
            rounded = np.nextafter(rounded, np.float32(0))
        return float(rounded)

    def lookahead_frames(self, rate):
        check_rate(rate)
        return (self.lookahead_ms*rate+500)//1000


@dataclass(frozen=True, slots=True)
class LimiterReport:
    sample_rate: int
    frames: int
    channels: int
    settings: LookaheadSettings
    lookahead_frames: int
    source_peak: float
    output_peak: float
    minimum_gain: float
    maximum_reduction_db: float
    mean_reduction_db: float
    attenuated_frames: int
    protocol: str = "sonumio-linked-lookahead-sample-limiter-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "sample_peak_ceiling": self.settings.ceiling,
            "channel_linked": True,
            "hard_clipping_applied": False,
            "certified_true_peak": False,
            "model_inference": False,
        }


@dataclass(frozen=True, slots=True)
class LimiterResult:
    audio: AudioBuffer
    report: LimiterReport


def limiter_envelope(peaks, sample_rate, settings=LookaheadSettings()):
    """Return one safe gain per frame for an already measured peak vector.

    The reverse pass enforces a maximum gain drop over the selected
    lookahead time, using the strongest required reduction to set the
    slope. Smaller events can be anticipated over shorter intervals.
    The causal release pass may extend attenuation but cannot relax it
    beyond the backward bound. The current sample always has priority.
    """
    check_rate(sample_rate)
    if not isinstance(settings, LookaheadSettings):
        raise AudioContractError("Expected LookaheadSettings")
    if (not isinstance(peaks, np.ndarray) or peaks.ndim != 1 or not len(peaks)
            or peaks.dtype not in (np.float32, np.float64)
            or not np.isfinite(peaks).all() or np.any(peaks < 0)):
        raise AudioContractError("Expected finite nonnegative frame peaks")
    if len(peaks) > MAX_LIMITER_FRAMES or len(peaks)*32 > MAX_LIMITER_BYTES:
        raise AudioLimitError("Limiter envelope exceeds working budget")
    peak = peaks.astype(np.float64, copy=False)
    if float(peak.max()) <= settings.ceiling:
        return np.ones(len(peak), dtype=np.float64)
    required = np.ones(len(peak), dtype=np.float64)
    over = peak > settings.ceiling
    required[over] = settings.ceiling/peak[over]
    # For a maximum reduction R and H lookahead frames, the linear
    # attack can start at most H frames before its strongest event.
    lookahead = settings.lookahead_frames(sample_rate)
    if lookahead:
        slope = (1.-float(required.min()))/lookahead
        previous = 1.
        for i in range(len(required)-1, -1, -1):
            previous = min(required[i], previous+slope)
            required[i] = previous
    if settings.release_ms:
        coefficient = math.exp(-1/(sample_rate*settings.release_ms/1000))
        previous = 1.
        for i in range(len(required)):
            recovered = 1-(1-previous)*coefficient
            previous = min(required[i], recovered)
            required[i] = previous
    if (not np.isfinite(required).all() or float(required.min()) <= 0
            or float(required.max()) > 1.):
        raise AudioProcessingError("Invalid limiter gain envelope")
    return required


def limit_audio(audio, settings=LookaheadSettings()):
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, LookaheadSettings):
        raise AudioContractError("Expected AudioBuffer and LookaheadSettings")
    estimate = audio.frames*(8*audio.channels*2+40)
    if audio.frames > MAX_LIMITER_FRAMES or estimate > MAX_LIMITER_BYTES:
        raise AudioLimitError("Offline limiter exceeds working-memory budget")
    source = audio.samples.astype(np.float64)
    peaks = np.max(np.abs(source), axis=0)
    source_peak = float(peaks.max())
    gain = limiter_envelope(peaks, audio.sample_rate, settings)
    output = (source*gain).astype(np.float32)
    if not np.isfinite(output).all():
        raise AudioProcessingError("Limiter output is nonfinite")
    output_peak = float(np.max(np.abs(output)))
    if output_peak > settings.ceiling:
        # The cap is a float32 value; a correctly rounded multiplication
        # cannot exceed it when the exact product is within half an ULP.
        # Never silently clip in the event of a numeric implementation bug.
        raise AudioProcessingError("Sample peak limiter violated its ceiling")
    minimum = float(gain.min())
    db = -20*np.log10(gain)
    result = audio.copy_with(output, source_note="linked offline lookahead limiter")
    return LimiterResult(result, LimiterReport(
        audio.sample_rate, audio.frames, audio.channels, settings,
        settings.lookahead_frames(audio.sample_rate), source_peak, output_peak,
        minimum, -20*math.log10(minimum), float(np.mean(db)),
        int(np.count_nonzero(gain < 1.)),
    ))
