"""Bounded median-filter harmonic/percussive STFT decomposition.

Harmonic magnitudes remain locally stable over time, while percussive
magnitudes spread over neighboring frequency bins. Independent median
filters estimate these traits; complementary soft masks partition the
original complex STFT. This is an interpretable CPU reference, not
trained source separation or guaranteed recovery of real instruments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError
from .stft import StftConfig, Spectrogram, stft, istft, complex_mask


MAX_MEDIAN_INPUT = 64 * 1024 * 1024
MAX_MEDIAN_WORK = 192 * 1024 * 1024
MAX_MEDIAN_PRODUCTS = 150_000_000
MAX_MASK_BINS = 32_000_000


def _int(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise AudioContractError(f"{name} must be integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class HpssSettings:
    """DSP and filtering geometry, with explicit partition sharpness."""

    n_fft: int = 1024
    hop_frames: int = 256
    harmonic_median_frames: int = 17
    percussive_median_bins: int = 17
    mask_power: float = 2.
    margin_harmonic: float = 1.
    margin_percussive: float = 1.
    minimum_magnitude: float = 1e-10

    def __post_init__(self):
        size = _int(self.n_fft, "n_fft", 64, 8192)
        _int(self.hop_frames, "hop_frames", 16, size // 2)
        for name in ("harmonic_median_frames", "percussive_median_bins"):
            value = _int(getattr(self, name), name, 3, 101)
            if not value % 2:
                raise AudioContractError(name + " must be odd")
        finite_float(self.mask_power, "mask_power", minimum=1., maximum=8.)
        finite_float(self.margin_harmonic, "margin_harmonic", minimum=.1, maximum=10.)
        finite_float(self.margin_percussive, "margin_percussive", minimum=.1, maximum=10.)
        finite_float(self.minimum_magnitude, "minimum_magnitude",
                     minimum=1e-14, maximum=.001)


@dataclass(frozen=True, slots=True)
class HpssGeometry:
    channels: int
    bins: int
    frames: int
    harmonic_width: int
    percussive_width: int

    def __post_init__(self):
        _int(self.channels, "channels", 1, 8)
        _int(self.bins, "frequency bins", 33, 4097)
        _int(self.frames, "STFT frames", 1, 100000)
        _int(self.harmonic_width, "harmonic median", 3, 101)
        _int(self.percussive_width, "percussive median", 3, 101)
        if self.samples > MAX_MASK_BINS:
            raise AudioLimitError("HPSS spectrum exceeds bounded mask geometry")
        if self.spectral_bytes > MAX_MEDIAN_INPUT:
            raise AudioLimitError("HPSS input spectrum exceeds budget")
        if self.estimated_median_products > MAX_MEDIAN_PRODUCTS:
            raise AudioLimitError("HPSS median work exceeds operation cap")
        if self.estimated_work_bytes > MAX_MEDIAN_WORK:
            raise AudioLimitError("HPSS intermediate arrays exceed memory limit")

    @property
    def samples(self):
        return self.channels * self.bins * self.frames

    @property
    def spectral_bytes(self):
        return self.samples * 16

    @property
    def estimated_median_products(self):
        return self.samples * (self.harmonic_width + self.percussive_width)

    @property
    def estimated_work_bytes(self):
        # Sliding-window median uses transient sorted/copy scratch, alongside
        # magnitudes, two median maps, mask, two result spectra and ISTFT.
        return (
            self.samples * (self.harmonic_width + self.percussive_width) * 8
            + self.samples * 16 * 5
        )


@dataclass(frozen=True, slots=True)
class SeparationStats:
    channels: int
    frames: int
    sample_rate: int
    analysis_frames: int
    fft_bins: int
    harmonic_median_frames: int
    percussive_median_bins: int
    harmonic_energy_fraction: float
    percussive_energy_fraction: float
    signal_peak: float
    harmonic_peak: float
    percussive_peak: float
    reconstruction_peak_error: float
    reconstruction_rms_error: float
    protocol: str = "sonumio-median-hpss-v1"

    def as_dict(self):
        value = asdict(self)
        value["model_inference"] = False
        value["perfect_instrument_isolation_claimed"] = False
        value["perceptual_quality_established"] = False
        value["source_recording_modified"] = False
        return value


@dataclass(frozen=True, slots=True)
class SeparationResult:
    harmonic: AudioBuffer
    percussive: AudioBuffer
    stats: SeparationStats


def _preflight(spec: Spectrogram, settings: HpssSettings) -> HpssGeometry:
    if not isinstance(spec, Spectrogram) or not isinstance(settings, HpssSettings):
        raise AudioContractError("Expected Spectrogram and HpssSettings")
    if spec.config.n_fft != settings.n_fft or (
        spec.config.hop_length != settings.hop_frames
    ):
        raise AudioContractError("Spectrum clock differs from HPSS configuration")
    return HpssGeometry(
        spec.channels, spec.bins, spec.frames,
        settings.harmonic_median_frames,
        settings.percussive_median_bins,
    )


def _median_axis(matrix: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Edge-extended bounded local median, no mutation of FFT magnitudes."""
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 3
            or matrix.dtype != np.float64
            or not np.isfinite(matrix).all() or np.any(matrix < 0)
            or axis not in (1, 2)):
        raise AudioContractError("Invalid nonnegative HPSS magnitude array")
    _int(width, "median width", 3, 101)
    if not width % 2:
        raise AudioContractError("Median width must be odd")
    if matrix.size * width * 8 > MAX_MEDIAN_WORK:
        raise AudioLimitError("HPSS local median exceeds memory budget")
    pad = [(0, 0)] * 3
    pad[axis] = (width // 2, width // 2)
    padded = np.pad(matrix, pad, mode="edge")
    windows = sliding_window_view(padded, width, axis=axis)
    return np.median(windows, axis=-1)


def _mask_from_estimates(
    harmonic: np.ndarray, percussive: np.ndarray,
    options: HpssSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Masks sum to one even for silent bins, so no lost residual is implied."""
    if (not isinstance(options, HpssSettings)
            or not isinstance(harmonic, np.ndarray)
            or not isinstance(percussive, np.ndarray)
            or harmonic.shape != percussive.shape
            or harmonic.ndim != 3
            or harmonic.dtype != np.float64
            or percussive.dtype != np.float64
            or not np.isfinite(harmonic).all()
            or not np.isfinite(percussive).all()
            or np.any(harmonic < 0)
            or np.any(percussive < 0)):
        raise AudioContractError("HPSS estimates must be matching finite power maps")
    # Margins bias the competition but masks remain complementary.
    high = (harmonic / options.margin_harmonic) ** options.mask_power
    low = (percussive / options.margin_percussive) ** options.mask_power
    denominator = high + low
    weight = np.divide(
        high, denominator, out=np.full_like(high, .5),
        where=denominator > options.minimum_magnitude ** options.mask_power,
    )
    weight = np.clip(weight, 0., 1.)
    other = 1. - weight
    if not np.isfinite(weight).all():
        raise AudioContractError("HPSS mask computation produced nonfinite gains")
    return weight, other


def separate_spectrogram(
    spectrum: Spectrogram,
    settings: HpssSettings = HpssSettings(),
) -> tuple[Spectrogram, Spectrogram, np.ndarray]:
    """Construct complementary complex spectra and detached harmonic gains."""
    _preflight(spectrum, settings)
    magnitude = np.abs(spectrum.spectrum)
    harmonic = _median_axis(
        magnitude, settings.harmonic_median_frames, axis=2,
    )
    percussive = _median_axis(
        magnitude, settings.percussive_median_bins, axis=1,
    )
    harmonic_mask, percussive_mask = _mask_from_estimates(
        harmonic, percussive, settings,
    )
    a = complex_mask(spectrum, harmonic_mask)
    b = complex_mask(spectrum, percussive_mask)
    # The public gain map is not a writable alias of internal scratch.
    mask = np.frombuffer(harmonic_mask.tobytes(), dtype=np.float64)
    return a, b, mask.reshape(harmonic_mask.shape)


def split_harmonic_percussive(
    audio: AudioBuffer,
    settings: HpssSettings = HpssSettings(),
) -> SeparationResult:
    """Return two immutable, same-clock components with diagnostic errors."""
    if not isinstance(audio, AudioBuffer) or not isinstance(settings, HpssSettings):
        raise AudioContractError("Expected AudioBuffer and HpssSettings")
    cfg = StftConfig(
        n_fft=settings.n_fft, hop_length=settings.hop_frames,
        window="hann", center=True, pad_end=True,
    )
    spectrum = stft(audio, cfg)
    harmonic_spec, percussive_spec, mask = separate_spectrogram(spectrum, settings)
    harmonic = audio.copy_with(istft(harmonic_spec).samples,
                               source_note="median HPSS harmonic estimate")
    percussive = audio.copy_with(istft(percussive_spec).samples,
                                 source_note="median HPSS percussive estimate")
    reconstruction = (
        harmonic.samples.astype(np.float64)
        + percussive.samples.astype(np.float64)
    )
    diff = reconstruction - audio.samples
    h_energy = float(np.sum(harmonic.samples.astype(np.float64) ** 2))
    p_energy = float(np.sum(percussive.samples.astype(np.float64) ** 2))
    denominator = max(h_energy + p_energy, 1e-30)
    stats = SeparationStats(
        audio.channels, audio.frames, audio.sample_rate, spectrum.frames,
        spectrum.bins, settings.harmonic_median_frames,
        settings.percussive_median_bins, h_energy / denominator,
        p_energy / denominator,
        float(np.max(np.abs(audio.samples))),
        float(np.max(np.abs(harmonic.samples))),
        float(np.max(np.abs(percussive.samples))),
        float(np.max(np.abs(diff))),
        float(np.sqrt(np.mean(diff * diff))),
    )
    return SeparationResult(harmonic, percussive, stats)
