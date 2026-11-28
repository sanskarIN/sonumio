"""Deterministic excitation sweep and bounded linear impulse estimation.

A log-sine sweep can probe a known linear time-invariant system. Its
recorded response is deconvolved by an explicitly regularized frequency
domain least squares estimate. This is a research reference: without
calibration, controlled recording geometry and noise analysis, results
must not be represented as measured acoustic ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, AudioLayout, AudioSpec, check_rate, finite_float, make_buffer
from .errors import AudioContractError, AudioLimitError
from .matrix_convolution import ImpulseMatrix

MAX_SWEEP_SECONDS = 12.
MAX_CAPTURE_SECONDS = 20.
MAX_IR_TAPS = 65_536
MAX_WORK_BYTES = 192 * 1024 * 1024
MAX_FFT_POINTS = 8_000_000


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class SweepPlan:
    sample_rate: int = 24000
    frames: int = 24000
    start_hz: float = 40.
    end_hz: float = 8000.
    amplitude: float = .5
    fade_frames: int = 128

    def __post_init__(self):
        check_rate(self.sample_rate)
        _integer(self.frames, "sweep frames", 256,
                 int(MAX_SWEEP_SECONDS * self.sample_rate))
        lower = finite_float(self.start_hz, "sweep start frequency",
                             minimum=10., maximum=self.sample_rate / 2 - 10)
        upper = finite_float(self.end_hz, "sweep end frequency",
                             minimum=10., maximum=self.sample_rate / 2 - 10)
        if upper <= lower:
            raise AudioContractError("Sweep frequency interval must increase")
        finite_float(self.amplitude, "sweep amplitude", minimum=.01, maximum=1.)
        _integer(self.fade_frames, "sweep fade", 0, self.frames // 4)
        if self.frames < 2 * self.fade_frames + 32:
            raise AudioContractError("Sweep is too short for declared fades")


@dataclass(frozen=True, slots=True)
class IdentificationSettings:
    taps: int = 1024
    regularization: float = 1e-10
    latency_frames: int = 0
    block_frames: int = 1024
    minimum_excitation_rms: float = .01

    def __post_init__(self):
        _integer(self.taps, "impulse taps", 1, MAX_IR_TAPS)
        finite_float(self.regularization, "regularization",
                     minimum=1e-14, maximum=.1)
        _integer(self.latency_frames, "latency_frames", 0, MAX_IR_TAPS)
        _integer(self.block_frames, "block_frames", 64, 16384)
        if self.block_frames & (self.block_frames - 1):
            raise AudioContractError("Impulse FFT block must be power of two")
        finite_float(self.minimum_excitation_rms, "excitation RMS",
                     minimum=1e-6, maximum=1.)


@dataclass(frozen=True, slots=True)
class IdentificationGeometry:
    channels: int
    excitation_frames: int
    recording_frames: int
    taps: int

    def __post_init__(self):
        _integer(self.channels, "capture channels", 1, 8)
        _integer(self.excitation_frames, "excitation frames", 1, 12 * 192000)
        _integer(self.recording_frames, "recording frames", 1, 20 * 192000)
        _integer(self.taps, "estimated taps", 1, MAX_IR_TAPS)
        if self.recording_frames < self.excitation_frames:
            raise AudioContractError("Recorded response shorter than excitation")
        if self.estimated_bytes > MAX_WORK_BYTES:
            raise AudioLimitError("Impulse identification exceeds bounded work memory")
        if self.fft_size * (2 * self.channels + 3) > MAX_FFT_POINTS:
            raise AudioLimitError("Impulse identification exceeds FFT work budget")

    @property
    def fft_size(self) -> int:
        return 1 << (self.excitation_frames + self.recording_frames - 2).bit_length()

    @property
    def bins(self) -> int:
        return self.fft_size // 2 + 1

    @property
    def estimated_bytes(self) -> int:
        # Full recording and excitation spectra, complex products, real
        # impulse scratch, immutable result and temporary float conversions.
        return (
            self.channels * (self.recording_frames * 24 + self.taps * 32)
            + 16 * self.bins * (2 * self.channels + 4)
            + 8 * self.fft_size * (self.channels + 2)
        ) * 2


@dataclass(frozen=True, slots=True)
class ChannelResponse:
    channel: int
    peak_tap: int
    peak_amplitude: float
    rms: float
    energy: float
    fit_rms: float
    relative_fit_rms: float


@dataclass(frozen=True, slots=True)
class IdentificationReport:
    sample_rate: int
    excitation_frames: int
    recorded_frames: int
    latency_frames: int
    taps: int
    channels: int
    regularization: float
    fft_size: int
    channels_report: tuple[ChannelResponse, ...]
    protocol: str = "sonumio-regularized-linear-ir-v1"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["model_inference"] = False
        result["measured_acoustics_certified"] = False
        result["nonlinear_harmonic_rejection_claimed"] = False
        result["perceptual_quality_established"] = False
        return result


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    impulse: ImpulseMatrix
    report: IdentificationReport


def render_log_sweep(plan: SweepPlan) -> AudioBuffer:
    """Original logarithmic chirp on the exact source sample clock."""
    if not isinstance(plan, SweepPlan):
        raise AudioContractError("Expected SweepPlan")
    sample = np.arange(plan.frames, dtype=np.float64)
    time = sample / plan.sample_rate
    duration = plan.frames / plan.sample_rate
    speed = math.log(plan.end_hz / plan.start_hz) / duration
    phase = (
        2 * math.pi * plan.start_hz
        * np.expm1(speed * time) / speed
    )
    output = plan.amplitude * np.sin(phase)
    fade = plan.fade_frames
    if fade:
        ramp = np.sin(
            .5 * np.pi * (np.arange(1, fade + 1) / (fade + 1))
        ) ** 2
        output[:fade] *= ramp
        output[-fade:] *= ramp[::-1]
    if not np.isfinite(output).all() or float(np.max(np.abs(output))) > 1.00001:
        raise AudioContractError("Invalid bounded logarithmic sweep output")
    return make_buffer(output.astype(np.float32), plan.sample_rate,
                       source_note="original log-sine calibration stimulus")


def _check_audio(
    excitation: AudioBuffer, recording: AudioBuffer,
    settings: IdentificationSettings,
) -> IdentificationGeometry:
    if (not isinstance(excitation, AudioBuffer)
            or not isinstance(recording, AudioBuffer)
            or not isinstance(settings, IdentificationSettings)):
        raise AudioContractError("Expected PCM source, capture and identification settings")
    if excitation.channels != 1:
        raise AudioContractError("Sweep excitation must contain exactly one channel")
    if recording.sample_rate != excitation.sample_rate:
        raise AudioContractError("Sweep recording and source clocks differ")
    if recording.frames > int(MAX_CAPTURE_SECONDS * recording.sample_rate):
        raise AudioLimitError("Sweep capture exceeds duration cap")
    if settings.latency_frames + settings.taps > recording.frames:
        raise AudioContractError("Requested impulse and latency exceed recording")
    return IdentificationGeometry(
        recording.channels, excitation.frames, recording.frames,
        settings.taps,
    )


def _fit_error(
    excitation: np.ndarray, observation: np.ndarray,
    impulse: np.ndarray, delay: int,
) -> tuple[float, float]:
    prediction = np.convolve(excitation, impulse, mode="full")
    expected = np.zeros_like(observation)
    upper = min(len(expected), delay + len(prediction))
    if upper > delay:
        expected[delay:upper] = prediction[:upper - delay]
    residual = expected - observation
    rms = float(np.sqrt(np.mean(residual * residual)))
    baseline = float(np.sqrt(np.mean(observation * observation)))
    return rms, rms / max(baseline, 1e-12)


def identify_linear_response(
    excitation: AudioBuffer,
    recording: AudioBuffer,
    settings: IdentificationSettings = IdentificationSettings(),
) -> IdentificationResult:
    """Bounded Tikhonov spectral deconvolution with explicit latency.

    For each capture output Y and declared sweep X, estimate
    H = conj(X)*Y/(|X|² + lambda*max(|X|²)), then inverse FFT and extract
    the requested finite impulse interval. No implicit gain normalization.
    """
    g = _check_audio(excitation, recording, settings)
    x = excitation.samples[0].astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < settings.minimum_excitation_rms:
        raise AudioContractError("Sweep excitation energy too low for identification")
    spectrum = np.fft.rfft(x, n=g.fft_size)
    power = np.abs(spectrum) ** 2
    maximum = float(np.max(power))
    if maximum < 1e-15:
        raise AudioContractError("Sweep has no spectral identification energy")
    inverse = np.conj(spectrum) / (
        power + settings.regularization * maximum
    )
    output = np.empty((g.channels, 1, settings.taps), dtype=np.float64)
    diagnostics = []
    for channel in range(g.channels):
        observed = recording.samples[channel].astype(np.float64)
        measured = np.fft.rfft(observed, n=g.fft_size)
        response = np.fft.irfft(measured * inverse, n=g.fft_size)
        start = settings.latency_frames
        impulse = response[start:start + settings.taps]
        if not np.isfinite(impulse).all():
            raise AudioContractError("Identification generated nonfinite coefficients")
        if float(np.max(np.abs(impulse))) > 1_000_000:
            raise AudioLimitError("Identified impulse gain exceeds bounded limit")
        output[channel, 0] = impulse
        peak = int(np.argmax(np.abs(impulse)))
        error, relative = _fit_error(x, observed, impulse, settings.latency_frames)
        diagnostics.append(ChannelResponse(
            channel, peak, float(impulse[peak]),
            float(np.sqrt(np.mean(impulse * impulse))),
            float(np.sum(impulse * impulse)), error, relative,
        ))
    matrix = ImpulseMatrix(
        output, excitation.sample_rate, block_frames=settings.block_frames,
    )
    report = IdentificationReport(
        excitation.sample_rate, excitation.frames,
        recording.frames, settings.latency_frames, settings.taps,
        recording.channels, settings.regularization, g.fft_size,
        tuple(diagnostics),
    )
    return IdentificationResult(matrix, report)
