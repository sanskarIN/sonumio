"""GCC-PHAT time-delay hypotheses for explicitly selected PCM channels.

The reported lag is positive when the observed signal arrives later than
the reference. An optional integer sample-clock shift can compensate a
reported lag without resampling. Correlation evidence is not a calibrated
measure of synchronization certainty in reverberant or unrelated audio.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .contracts import AudioBuffer, AudioLayout, check_rate, finite_float
from .errors import AudioContractError, AudioLimitError


MAX_ALIGNMENT_FRAMES = 500_000
MAX_ALIGNMENT_FFT = 2_097_152
MAX_ALIGNMENT_BYTES = 128 * 1024 * 1024
MAX_DELAY_FRAMES = 100_000
MAX_REFINEMENT = .5


def _int(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class AlignmentSettings:
    maximum_lag_frames: int = 8000
    reference_channel: int = 0
    observed_channel: int = 0
    remove_dc: bool = True
    taper: bool = True
    phat_exponent: float = 1.
    minimum_signal_rms: float = 1e-5
    minimum_peak_score: float = 0.
    refinement: bool = True

    def __post_init__(self):
        _int(self.maximum_lag_frames, "maximum_lag_frames", 1, MAX_DELAY_FRAMES)
        _int(self.reference_channel, "reference_channel", 0, 7)
        _int(self.observed_channel, "observed_channel", 0, 7)
        for key in ("remove_dc", "taper", "refinement"):
            if type(getattr(self, key)) is not bool:
                raise AudioContractError(f"{key} must be boolean")
        finite_float(self.phat_exponent, "phat_exponent", minimum=0., maximum=1.)
        finite_float(self.minimum_signal_rms, "minimum_signal_rms",
                     minimum=0., maximum=1.)
        finite_float(self.minimum_peak_score, "minimum_peak_score",
                     minimum=0., maximum=1.)


@dataclass(frozen=True, slots=True)
class AlignmentGeometry:
    reference_frames: int
    observed_frames: int
    maximum_lag_frames: int
    reference_channels: int
    observed_channels: int

    def __post_init__(self):
        for name, number in (
            ("reference_frames", self.reference_frames),
            ("observed_frames", self.observed_frames),
        ):
            _int(number, name, 8, MAX_ALIGNMENT_FRAMES)
        _int(self.maximum_lag_frames, "maximum_lag_frames", 1, MAX_DELAY_FRAMES)
        _int(self.reference_channels, "reference_channels", 1, 8)
        _int(self.observed_channels, "observed_channels", 1, 8)
        if self.maximum_lag_frames >= min(self.reference_frames, self.observed_frames):
            raise AudioContractError("Lag bound must be smaller than both signals")
        if self.fft_length > MAX_ALIGNMENT_FFT:
            raise AudioLimitError("GCC-PHAT FFT geometry exceeds limit")
        if self.estimated_bytes > MAX_ALIGNMENT_BYTES:
            raise AudioLimitError("GCC-PHAT correlation workspace exceeds budget")

    @property
    def fft_length(self):
        raw = self.reference_frames + self.observed_frames - 1
        return 1 << (raw - 1).bit_length()

    @property
    def estimated_bytes(self):
        n = self.fft_length
        # Time signals, two complex spectra, whitened cross-spectrum,
        # correlation and scaled magnitude scratch.
        return 12 * (self.reference_frames + self.observed_frames) + n * 96


@dataclass(frozen=True, slots=True)
class AlignmentHypothesis:
    lag_frames: int
    fractional_lag_frames: float
    seconds: float
    peak_score: float
    secondary_peak_score: float
    peak_ratio: float
    ambiguous: bool


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    sample_rate: int
    reference_frames: int
    observed_frames: int
    reference_channel: int
    observed_channel: int
    reference_rms: float
    observed_rms: float
    maximum_lag_frames: int
    phat_exponent: float
    hypothesis: AlignmentHypothesis | None
    protocol: str = "sonumio-gcc-phat-alignment-v1"

    def as_dict(self):
        data = asdict(self)
        data["model_inference"] = False
        data["clock_drift_estimated"] = False
        data["physical_propagation_delay_certified"] = False
        data["automatic_resampling"] = False
        return data


def _parabolic(values: np.ndarray, peak_index: int) -> float:
    if peak_index <= 0 or peak_index >= len(values) - 1:
        return 0.
    left, center, right = map(float, values[peak_index - 1:peak_index + 2])
    curvature = left - 2 * center + right
    if curvature >= -1e-16:
        return 0.
    delta = .5 * (left - right) / curvature
    return max(-MAX_REFINEMENT, min(MAX_REFINEMENT, delta))


def _preflight(reference, observed, options):
    if (not isinstance(reference, AudioBuffer)
            or not isinstance(observed, AudioBuffer)
            or not isinstance(options, AlignmentSettings)):
        raise AudioContractError("Expected PCM references and AlignmentSettings")
    if reference.sample_rate != observed.sample_rate:
        raise AudioContractError("GCC-PHAT input sample clocks differ")
    if options.reference_channel >= reference.channels:
        raise AudioContractError("Reference channel is outside PCM topology")
    if options.observed_channel >= observed.channels:
        raise AudioContractError("Observed channel is outside PCM topology")
    return AlignmentGeometry(
        reference.frames, observed.frames, options.maximum_lag_frames,
        reference.channels, observed.channels,
    )


def estimate_delay(
    reference: AudioBuffer,
    observed: AudioBuffer,
    options: AlignmentSettings = AlignmentSettings(),
) -> AlignmentReport:
    """Return signed relative lag with an explicit nonconfidence condition."""
    g = _preflight(reference, observed, options)
    x = reference.samples[options.reference_channel].astype(np.float64)
    y = observed.samples[options.observed_channel].astype(np.float64)
    if options.remove_dc:
        x -= float(np.mean(x))
        y -= float(np.mean(y))
    xrms = math.sqrt(float(np.mean(x * x)))
    yrms = math.sqrt(float(np.mean(y * y)))
    if xrms <= options.minimum_signal_rms or yrms <= options.minimum_signal_rms:
        return AlignmentReport(
            reference.sample_rate, reference.frames, observed.frames,
            options.reference_channel, options.observed_channel,
            xrms, yrms, options.maximum_lag_frames, options.phat_exponent, None,
        )
    if options.taper:
        x *= np.hanning(len(x))
        y *= np.hanning(len(y))
    X = np.fft.rfft(x, n=g.fft_length)
    Y = np.fft.rfft(y, n=g.fft_length)
    cross = Y * np.conj(X)
    magnitude = np.abs(cross)
    # PHAT whitens amplitude spectra and retains phase-delay evidence.
    if options.phat_exponent:
        cross /= np.maximum(magnitude, 1e-14) ** options.phat_exponent
    correlation = np.fft.irfft(cross, n=g.fft_length)
    bound = options.maximum_lag_frames
    lags = np.arange(-bound, bound + 1)
    score = correlation[lags % g.fft_length]
    magnitude_score = np.abs(score)
    if not np.isfinite(magnitude_score).all():
        raise AudioContractError("GCC-PHAT produced nonfinite correlation")
    index = int(np.argmax(magnitude_score))
    dominant = float(magnitude_score[index])
    if dominant <= 1e-15:
        hypothesis = None
    else:
        # Scale-invariant peak strength, not a calibrated correctness score.
        norm = float(np.sum(magnitude_score * magnitude_score))
        strength = dominant / max(math.sqrt(norm), 1e-15)
        if strength < options.minimum_peak_score:
            hypothesis = None
        else:
            secondary = magnitude_score.copy()
            secondary[max(0, index - 3):min(len(secondary), index + 4)] = 0
            runner_up = float(np.max(secondary))
            ratio = dominant / max(runner_up, 1e-12)
            correction = (
                _parabolic(magnitude_score, index)
                if options.refinement else 0.
            )
            integer_lag = int(lags[index])
            fractional = integer_lag + correction
            hypothesis = AlignmentHypothesis(
                integer_lag, fractional, fractional / reference.sample_rate,
                strength, runner_up / max(math.sqrt(norm), 1e-15),
                ratio, ratio < 1.2,
            )
    return AlignmentReport(
        reference.sample_rate, reference.frames, observed.frames,
        options.reference_channel, options.observed_channel,
        xrms, yrms, options.maximum_lag_frames, options.phat_exponent,
        hypothesis,
    )


def shift_integer_frames(
    audio: AudioBuffer,
    shift_frames: int,
) -> AudioBuffer:
    """Shift a signal in-place on its declared clock without changing input.

    Positive shift adds leading zero samples. Negative shift discards
    leading samples and appends silence. The result retains the exact
    original number of frames, channel layout and sample rate.
    """
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    _int(shift_frames, "shift_frames", -MAX_DELAY_FRAMES, MAX_DELAY_FRAMES)
    values = np.zeros(audio.samples.shape, dtype=np.float32)
    frames = audio.frames
    if abs(shift_frames) < frames:
        if shift_frames >= 0:
            values[:, shift_frames:] = audio.samples[:, :frames - shift_frames]
        else:
            advance = -shift_frames
            values[:, :frames - advance] = audio.samples[:, advance:]
    return audio.copy_with(values, source_note="explicit sample-clock integer shift")


def align_observed_to_reference(
    observed: AudioBuffer, report: AlignmentReport,
) -> AudioBuffer:
    """Apply the *integer* lag hypothesis, without fractional resampling."""
    if not isinstance(observed, AudioBuffer) or not isinstance(report, AlignmentReport):
        raise AudioContractError("Expected PCM and AlignmentReport")
    if observed.sample_rate != report.sample_rate or (
        observed.frames != report.observed_frames
    ):
        raise AudioContractError("Alignment report does not match observed PCM")
    if report.hypothesis is None:
        raise AudioContractError("No reliable alignment hypothesis available")
    return shift_integer_frames(observed, -report.hypothesis.lag_frames)
