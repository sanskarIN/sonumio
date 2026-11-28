"""Phase-aligned finite-impulse-response crossover and multiband dynamics.

The crossover is a complementary, linear-phase FIR bank. Its kernels sum
to a centered unit impulse, so summing all bands reconstructs the input
(subject to floating-point error) after group-delay compensation. This
offline reference is not a zero-latency, true-peak or broadcast master.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, AudioSpec, check_rate, finite_float
from .dynamics import CompressorConfig, compress
from .errors import AudioContractError, AudioLimitError
from .filters import design_fir_lowpass

MAX_CROSSOVERS = 5
MAX_CROSSOVER_TAPS = 4095
MAX_WORK_BYTES = 192 * 1024 * 1024
MAX_FFT_POINTS = 32_000_000


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise AudioContractError(f"{name} must be integer in [{lower}, {upper}]")
    return value


@dataclass(frozen=True, slots=True)
class CrossoverPlan:
    """Sample-clock pinned, strictly increasing FIR crossover frequencies."""

    sample_rate: int
    cutoffs_hz: tuple[float, ...] = (180., 2500.)
    taps: int = 257
    window: str = "blackman"

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        if (type(self.cutoffs_hz) is not tuple
                or not 1 <= len(self.cutoffs_hz) <= MAX_CROSSOVERS):
            raise AudioContractError("Crossover requires 1..5 cutoffs")
        values = tuple(
            finite_float(v, "crossover frequency", minimum=1.,
                         maximum=self.sample_rate / 2 - 1)
            for v in self.cutoffs_hz
        )
        if any(b <= a for a, b in zip(values, values[1:])):
            raise AudioContractError("Crossover frequencies must strictly increase")
        object.__setattr__(self, "cutoffs_hz", values)
        _integer(self.taps, "crossover taps", 5, MAX_CROSSOVER_TAPS)
        if not self.taps % 2:
            raise AudioContractError("Crossover taps must be odd for exact group delay")
        if self.window not in ("hann", "hamming", "blackman"):
            raise AudioContractError("Unsupported crossover window")

    @property
    def bands(self) -> int:
        return len(self.cutoffs_hz) + 1

    @property
    def delay_frames(self) -> int:
        return (self.taps - 1) // 2

    def kernels(self) -> np.ndarray:
        """Return detached, immutable float64 [band, tap] complementary kernels."""
        lows = [
            design_fir_lowpass(
                self.sample_rate, hz, taps=self.taps, window=self.window,
            )
            for hz in self.cutoffs_hz
        ]
        impulse = np.zeros(self.taps, dtype=np.float64)
        impulse[self.delay_frames] = 1.
        result = [lows[0]]
        result.extend(b - a for a, b in zip(lows, lows[1:]))
        result.append(impulse - lows[-1])
        kernels = np.stack(result)
        # Guarantee complementarity and keep the internal design immutable.
        if not np.allclose(kernels.sum(axis=0), impulse, rtol=0, atol=1e-12):
            raise AudioContractError("Crossover kernels do not reconstruct unity")
        frozen = np.frombuffer(kernels.tobytes(order="C"), dtype=np.float64)
        return frozen.reshape(kernels.shape)


@dataclass(frozen=True, slots=True)
class CrossoverGeometry:
    channels: int
    frames: int
    bands: int
    taps: int

    def __post_init__(self):
        _integer(self.channels, "channels", 1, 8)
        _integer(self.frames, "frames", 1, 120 * 192000)
        _integer(self.bands, "bands", 2, 6)
        _integer(self.taps, "taps", 5, MAX_CROSSOVER_TAPS)
        if self.estimated_work_bytes > MAX_WORK_BYTES:
            raise AudioLimitError("Crossover FFT and audio exceed working memory")
        if self.fft_points > MAX_FFT_POINTS:
            raise AudioLimitError("Crossover exceeds FFT work limit")

    @property
    def fft_size(self):
        return 1 << (self.frames + self.taps - 2).bit_length()

    @property
    def bins(self):
        return self.fft_size // 2 + 1

    @property
    def fft_points(self):
        return (self.channels + self.bands * (self.channels + 1)) * self.fft_size

    @property
    def estimated_work_bytes(self):
        # A conservative bound: shared input spectra, kernel FFT, complex
        # products, inverse FFT, all band PCM, immutable AudioBuffer copies,
        # recombination and temporary conversion arrays.
        complex_bytes = 16 * self.bins * (2 * self.channels + 2)
        time_bytes = (
            8 * self.fft_size * self.channels
            + 32 * self.frames * self.channels * self.bands
            + 24 * self.frames * self.channels
        )
        return 2 * (complex_bytes + time_bytes)


@dataclass(frozen=True, slots=True)
class BandLevel:
    lower_hz: float
    upper_hz: float
    input_rms: float
    output_rms: float
    input_peak: float
    output_peak: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MultibandReport:
    sample_rate: int
    frames: int
    channels: int
    cutoffs_hz: tuple[float, ...]
    group_delay_frames: int
    bands: tuple[BandLevel, ...]
    wet: float
    output_gain_db: float
    input_peak: float
    output_peak: float
    output_rms: float
    protocol: str = "sonumio-fir-multiband-v1"

    def as_dict(self) -> dict:
        result = asdict(self)
        result["cutoffs_hz"] = list(self.cutoffs_hz)
        result["model_inference"] = False
        result["perceptual_quality_established"] = False
        result["automatic_clipping_prevention"] = False
        return result


@dataclass(frozen=True, slots=True)
class MultibandResult:
    audio: AudioBuffer
    report: MultibandReport


def split_bands(audio: AudioBuffer, plan: CrossoverPlan) -> tuple[AudioBuffer, ...]:
    """Split into delay-compensated, same-length complementary frequency bands.

    Boundary treatment is zero extension. The convolution delay is removed
    by reading a centered interval of the full linear convolution.
    This is noncausal at signal boundaries and should not be used as a
    streaming callback without an explicit latency policy.
    """
    if not isinstance(audio, AudioBuffer) or not isinstance(plan, CrossoverPlan):
        raise AudioContractError("Expected AudioBuffer and CrossoverPlan")
    if audio.sample_rate != plan.sample_rate:
        raise AudioContractError("Crossover and audio sample clocks differ")
    geometry = CrossoverGeometry(audio.channels, audio.frames, plan.bands, plan.taps)
    kernels = plan.kernels()
    spectra = np.fft.rfft(audio.samples, n=geometry.fft_size, axis=-1)
    start = plan.delay_frames
    stop = start + audio.frames
    bands = []
    for kernel in kernels:
        response = np.fft.rfft(kernel, n=geometry.fft_size)
        shaped = np.fft.irfft(spectra * response[None, :],
                             n=geometry.fft_size, axis=-1)
        segment = shaped[:, start:stop]
        if not np.isfinite(segment).all():
            raise AudioContractError("Crossover produced nonfinite samples")
        if float(np.max(np.abs(segment))) > float(np.finfo(np.float32).max):
            raise AudioLimitError("Crossover output exceeds float32 range")
        bands.append(audio.copy_with(
            segment.astype(np.float32), source_note="linear phase crossover",
        ))
    return tuple(bands)


def recombine_bands(
    bands: tuple[AudioBuffer, ...], *,
    source_note: str = "recombined crossover bands",
) -> AudioBuffer:
    """Accumulate same-geometry bands without implicit normalization."""
    if type(bands) is not tuple or not 2 <= len(bands) <= 6:
        raise AudioContractError("Expected 2..6 crossover bands")
    if any(not isinstance(b, AudioBuffer) for b in bands):
        raise AudioContractError("Expected AudioBuffer crossover bands")
    spec = bands[0].spec
    if any(b.spec != spec for b in bands):
        raise AudioContractError("Band geometry, layouts and sample clocks differ")
    out = np.zeros(bands[0].samples.shape, dtype=np.float64)
    for b in bands:
        out += b.samples
    if float(np.max(np.abs(out))) > float(np.finfo(np.float32).max):
        raise AudioLimitError("Recombined audio exceeds float32 range")
    return bands[0].copy_with(out.astype(np.float32), source_note=source_note)


def multiband_compress(
    audio: AudioBuffer,
    plan: CrossoverPlan,
    compressors: tuple[CompressorConfig, ...],
    *,
    wet: float = 1.,
    output_gain_db: float = 0.,
) -> MultibandResult:
    """Compress separately in each frequency band, then recombine.

    Output is never silently clipped, limited or normalized. Channel linkage
    follows each CompressorConfig. wet=0 is an exact identity bypass.
    """
    if not isinstance(audio, AudioBuffer) or not isinstance(plan, CrossoverPlan):
        raise AudioContractError("Expected AudioBuffer and CrossoverPlan")
    if (type(compressors) is not tuple or len(compressors) != plan.bands
            or any(not isinstance(x, CompressorConfig) for x in compressors)):
        raise AudioContractError("Expected one CompressorConfig per band")
    mix = finite_float(wet, "multiband wet", minimum=0, maximum=1)
    gain_db = finite_float(output_gain_db, "multiband output gain",
                           minimum=-60, maximum=24)
    if audio.sample_rate != plan.sample_rate:
        raise AudioContractError("Multiband audio clock differs from plan")
    # Also reject oversized work for bypass, preserving a consistent
    # configuration budget regardless of parameter values.
    CrossoverGeometry(audio.channels, audio.frames, plan.bands, plan.taps)
    originals = split_bands(audio, plan)
    processed = tuple(compress(b, c) for b, c in zip(originals, compressors))
    ranges = (0., *plan.cutoffs_hz, audio.sample_rate / 2)
    levels = tuple(
        BandLevel(
            ranges[index], ranges[index + 1],
            float(np.sqrt(np.mean(before.samples.astype(np.float64) ** 2))),
            float(np.sqrt(np.mean(after.samples.astype(np.float64) ** 2))),
            float(np.max(np.abs(before.samples))),
            float(np.max(np.abs(after.samples))),
        )
        for index, (before, after) in enumerate(zip(originals, processed))
    )
    if mix == 0 and gain_db == 0:
        result = audio
    else:
        output = np.zeros(audio.samples.shape, dtype=np.float64)
        for band in processed:
            output += band.samples.astype(np.float64)
        if mix != 1:
            output = mix * output + (1 - mix) * audio.samples
        if gain_db:
            output *= 10 ** (gain_db / 20)
        if not np.isfinite(output).all():
            raise AudioContractError("Multiband dynamics produced nonfinite samples")
        if float(np.max(np.abs(output))) > float(np.finfo(np.float32).max):
            raise AudioLimitError("Multiband dynamics exceeds float32 range")
        result = audio.copy_with(output.astype(np.float32),
                                 source_note="offline FIR multiband compressor")
    report = MultibandReport(
        audio.sample_rate, audio.frames, audio.channels,
        plan.cutoffs_hz, plan.delay_frames, levels, mix, gain_db,
        float(np.max(np.abs(audio.samples))),
        float(np.max(np.abs(result.samples))),
        float(np.sqrt(np.mean(result.samples.astype(np.float64) ** 2))),
    )
    return MultibandResult(result, report)
