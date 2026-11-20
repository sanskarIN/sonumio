"""Invertible channel-first short-time Fourier transforms.

This implementation is a small CPU reference for analysis and effects.
It is not advertised as an EBU loudness meter or learned audio codec.
Center-padding allows Hann endpoints to be reconstructed without losing
first/last samples; weighted overlap-add explicitly normalizes energy.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .contracts import AudioBuffer, AudioSpec, as_channel_first, check_audio, MAX_BYTES
from .errors import AudioContractError, AudioLimitError
from .windows import window_bank


@dataclass(frozen=True, slots=True)
class StftConfig:
    n_fft: int = 1024
    hop_length: int = 256
    window: str = "hann"
    center: bool = True
    pad_end: bool = True

    def __post_init__(self):
        if type(self.n_fft) is not int or not 32 <= self.n_fft <= 8192:
            raise AudioContractError("STFT FFT must be between 32 and 8192")
        if type(self.hop_length) is not int or not 1 <= self.hop_length <= self.n_fft:
            raise AudioContractError("STFT hop must be between 1 and FFT size")
        if self.window not in ("hann", "hamming", "blackman", "rectangular", "sqrt_hann"):
            raise AudioContractError("Invalid STFT analysis window")
        if type(self.center) is not bool or type(self.pad_end) is not bool:
            raise AudioContractError("STFT boundary flags must be booleans")


@dataclass(frozen=True, slots=True, eq=False)
class Spectrogram:
    """Readonly complex spectrum shaped channels × bins × time_frames."""

    spectrum: np.ndarray
    sample_rate: int
    original_frames: int
    config: StftConfig

    def __post_init__(self):
        from .contracts import check_rate
        check_rate(self.sample_rate)
        if not isinstance(self.config, StftConfig):
            raise AudioContractError("Expected StftConfig")
        data = self.spectrum
        if (not isinstance(data, np.ndarray) or data.ndim != 3
                or data.shape[0] < 1 or data.shape[0] > 8
                or data.shape[1] != self.config.n_fft // 2 + 1
                or data.shape[2] < 1 or data.dtype.kind != "c"):
            raise AudioContractError("Invalid complex STFT geometry")
        if (type(self.original_frames) is not int
                or self.original_frames < 1
                or not np.isfinite(data).all()):
            raise AudioContractError("Invalid nonfinite STFT metadata")
        if data.nbytes > MAX_BYTES:
            raise AudioLimitError("STFT exceeds bounded in-memory allocation")
        copy = np.array(data, dtype=np.complex128, copy=True)
        frozen = np.frombuffer(copy.tobytes(order="C"), dtype=np.complex128)
        object.__setattr__(self, "spectrum", frozen.reshape(copy.shape))

    @property
    def channels(self) -> int:
        return self.spectrum.shape[0]

    @property
    def bins(self) -> int:
        return self.spectrum.shape[1]

    @property
    def frames(self) -> int:
        return self.spectrum.shape[2]

    @property
    def frequencies(self) -> np.ndarray:
        return np.fft.rfftfreq(self.config.n_fft, 1 / self.sample_rate)

    @property
    def times(self) -> np.ndarray:
        frame_positions = np.arange(self.frames) * self.config.hop_length
        if self.config.center:
            return frame_positions / self.sample_rate
        return (frame_positions + self.config.n_fft / 2) / self.sample_rate

    def magnitude(self) -> np.ndarray:
        return np.abs(self.spectrum)

    def power(self) -> np.ndarray:
        return np.abs(self.spectrum) ** 2

    def phase(self) -> np.ndarray:
        return np.angle(self.spectrum)

    def with_spectrum(self, values: np.ndarray) -> "Spectrogram":
        return Spectrogram(values, self.sample_rate,
                           self.original_frames, self.config)


def _padded_signal(data: np.ndarray, cfg: StftConfig):
    n_fft = cfg.n_fft
    if cfg.center:
        pad = n_fft // 2
        data = np.pad(data, ((0, 0), (pad, pad)), mode="constant")
    if len(data[0]) < n_fft:
        if not cfg.pad_end:
            raise AudioContractError("Audio shorter than FFT and pad_end disabled")
        data = np.pad(data, ((0, 0), (0, n_fft - data.shape[1])),
                      mode="constant")
    if cfg.pad_end:
        leftover = (data.shape[1] - n_fft) % cfg.hop_length
        if leftover:
            data = np.pad(data, ((0, 0), (0, cfg.hop_length - leftover)),
                          mode="constant")
    return data


def stft(buffer: AudioBuffer, config: StftConfig | None = None) -> Spectrogram:
    if not isinstance(buffer, AudioBuffer):
        raise AudioContractError("STFT requires canonical AudioBuffer")
    cfg = StftConfig() if config is None else config
    if not isinstance(cfg, StftConfig):
        raise AudioContractError("Invalid STFT configuration")
    data = _padded_signal(buffer.mutable_copy(dtype="float64"), cfg)
    n = 1 + (data.shape[1] - cfg.n_fft) // cfg.hop_length
    bins = cfg.n_fft // 2 + 1
    if buffer.channels * bins * n * 16 > MAX_BYTES:
        raise AudioLimitError("Spectral working array exceeds memory budget")
    win = window_bank(cfg.window, cfg.n_fft, periodic=True)
    spec = np.empty((buffer.channels, bins, n), dtype=np.complex128)
    # Deliberate finite batches; no as_strided views of untrusted memory.
    for frame in range(n):
        offset = frame * cfg.hop_length
        segment = data[:, offset:offset + cfg.n_fft] * win
        spec[:, :, frame] = np.fft.rfft(segment, n=cfg.n_fft, axis=1)
    return Spectrogram(spec, buffer.sample_rate, buffer.frames, cfg)


def istft(spectrogram: Spectrogram, *, length: int | None = None,
          epsilon: float = 1e-10) -> AudioBuffer:
    """Inverse STFT with explicit squared-window overlap normalization.

    Reconstructs exactly specified original duration for centered,
    pad_end=True transforms to numerical float32 tolerance.
    """
    if not isinstance(spectrogram, Spectrogram):
        raise AudioContractError("Expected validated Spectrogram")
    cfg = spectrogram.config
    if type(length) is not int and length is not None:
        raise AudioContractError("Expected integral requested ISTFT length")
    if length is not None and length <= 0:
        raise AudioContractError("ISTFT length must be positive")
    if type(epsilon) not in (float, int) or not math.isfinite(
        epsilon
    ) or not 0 <= epsilon <= 0.01:
        raise AudioContractError("Invalid spectral inverse regularization")
    n = cfg.n_fft + (spectrogram.frames - 1) * cfg.hop_length
    if n * spectrogram.channels * 16 > MAX_BYTES:
        raise AudioLimitError("ISTFT working allocation exceeds budget")
    result = np.zeros((spectrogram.channels, n), dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    win = window_bank(cfg.window, cfg.n_fft, periodic=True)
    for frame in range(spectrogram.frames):
        offset = frame * cfg.hop_length
        segment = np.fft.irfft(spectrogram.spectrum[:, :, frame],
                               n=cfg.n_fft, axis=1)
        result[:, offset:offset + cfg.n_fft] += segment * win
        weights[offset:offset + cfg.n_fft] += win ** 2
    mask = weights > epsilon
    result[:, mask] /= weights[mask]
    result[:, ~mask] = 0
    if cfg.center:
        pad = cfg.n_fft // 2
        if result.shape[1] <= 2 * pad:
            raise AudioContractError("STFT too short for symmetric unpadding")
        result = result[:, pad:-pad]
    desired = spectrogram.original_frames if length is None else length
    if desired > result.shape[1]:
        if not cfg.pad_end:
            raise AudioContractError("Cannot recover STFT frames not analyzed")
        result = np.pad(result, ((0, 0), (0, desired - result.shape[1])),
                        mode="constant")
    result = result[:, :desired]
    if not np.isfinite(result).all():
        raise AudioContractError("ISTFT produced nonfinite samples")
    from .contracts import make_buffer
    return make_buffer(result.astype(np.float32), spectrogram.sample_rate,
                       source_note="inverse bounded STFT")


def complex_mask(spectrogram: Spectrogram, mask: np.ndarray, *,
                 maximum_gain: float = 8.0) -> Spectrogram:
    if not isinstance(spectrogram, Spectrogram) or not isinstance(mask, np.ndarray):
        raise AudioContractError("Expected spectrum and finite complex/real mask")
    if mask.shape not in (
        spectrogram.spectrum.shape,
        spectrogram.spectrum.shape[1:],
    ):
        raise AudioContractError("Spectral mask shape mismatch")
    if not np.isfinite(mask).all():
        raise AudioContractError("Nonfinite spectral mask")
    if not math.isfinite(maximum_gain) or not 0 < maximum_gain <= 64:
        raise AudioContractError("Invalid spectral gain cap")
    if np.max(np.abs(mask)) > maximum_gain:
        raise AudioContractError("Spectral mask exceeds gain cap")
    return spectrogram.with_spectrum(spectrogram.spectrum * mask)


def spectrogram_report(s: Spectrogram) -> dict:
    if not isinstance(s, Spectrogram):
        raise AudioContractError("Expected Spectrogram")
    spectrum = s.spectrum
    return {
        "sample_rate": s.sample_rate,
        "source_frames": s.original_frames,
        "analysis_frames": s.frames,
        "channels": s.channels,
        "bins": s.bins,
        "n_fft": s.config.n_fft,
        "hop_length": s.config.hop_length,
        "max_magnitude": float(np.max(np.abs(spectrum))),
        "fft_window": s.config.window,
        "centered": s.config.center,
        "model_inference": False,
    }
