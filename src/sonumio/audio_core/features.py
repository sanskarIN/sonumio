"""Interpretable CPU audio diagnostics for reference signal comparisons.

These are deterministic acoustic features, NOT CLAP/Whisper embeddings,
human listening quality scores, speaker recognition, LUFS or generative
model metrics. Avoid using them as a replacement for perceptual evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .contracts import AudioBuffer, finite_float, check_rate
from .errors import AudioContractError
from .numerics import dbfs, stable_peak, stable_rms
from .stft import Spectrogram, StftConfig, stft


def _spectrum(value: Spectrogram) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(value, Spectrogram):
        raise AudioContractError("Expected bounded Spectrogram input")
    return value.magnitude(), value.frequencies


def spectral_centroid(spec: Spectrogram) -> np.ndarray:
    magnitude, frequencies = _spectrum(spec)
    weight = magnitude.sum(axis=1)
    center = np.sum(magnitude * frequencies[None, :, None],
                    axis=1) / np.maximum(weight, 1e-20)
    return center.astype(np.float32)


def spectral_bandwidth(spec: Spectrogram, *, order: float = 2) -> np.ndarray:
    ord_value = finite_float(order, "bandwidth order", minimum=1, maximum=10)
    magnitude, hz = _spectrum(spec)
    centroid = spectral_centroid(spec).astype(np.float64)
    deviation = np.abs(hz[None, :, None] - centroid[:, None, :]) ** ord_value
    moment = (magnitude * deviation).sum(axis=1) / np.maximum(
        magnitude.sum(axis=1), 1e-20
    )
    return np.power(moment, 1 / ord_value).astype(np.float32)


def spectral_rolloff(spec: Spectrogram, *, fraction: float = 0.85) -> np.ndarray:
    rate = finite_float(fraction, "rolloff energy fraction", minimum=0, maximum=1)
    magnitude, hz = _spectrum(spec)
    energy = magnitude ** 2
    cdf = np.cumsum(energy, axis=1)
    threshold = cdf[:, -1:, :] * rate
    index = np.argmax(cdf >= threshold, axis=1)
    return hz[index].astype(np.float32)


def spectral_flatness(spec: Spectrogram, *, epsilon: float = 1e-12
                      ) -> np.ndarray:
    floor = finite_float(epsilon, "flatness floor", minimum=1e-20, maximum=1e-3)
    magnitude, _ = _spectrum(spec)
    safe = magnitude + floor
    geometric = np.exp(np.mean(np.log(safe), axis=1))
    arithmetic = np.mean(safe, axis=1)
    return np.clip(geometric / arithmetic, 0, 1).astype(np.float32)


def spectral_flux(spec: Spectrogram, *, positive_only: bool = True
                  ) -> np.ndarray:
    if type(positive_only) is not bool:
        raise AudioContractError("Expected boolean positive_only")
    magnitude, _ = _spectrum(spec)
    diff = np.diff(magnitude, axis=2, prepend=magnitude[:, :, :1])
    if positive_only:
        diff = np.maximum(0, diff)
    return np.sqrt(np.mean(diff ** 2, axis=1)).astype(np.float32)


def frame_energy(spec: Spectrogram) -> np.ndarray:
    magnitude, _ = _spectrum(spec)
    return np.mean(magnitude ** 2, axis=1).astype(np.float32)


def zero_crossing_rate(samples: np.ndarray, *, per_channel: bool = True
                       ) -> np.ndarray | float:
    from .contracts import as_channel_first, check_audio
    x = as_channel_first(check_audio(samples), float64=True)
    if x.shape[1] < 2:
        values = np.zeros(x.shape[0], dtype=np.float32)
    else:
        # A zero is neither a positive nor a negative signal excursion.
        signs = np.sign(x)
        signs[signs == 0] = np.nan
        changed = (
            np.isfinite(signs[:, 1:]) & np.isfinite(signs[:, :-1])
            & (signs[:, 1:] != signs[:, :-1])
        )
        values = np.mean(changed, axis=1).astype(np.float32)
    return values if per_channel else float(np.mean(values))


def crest_factor_db(samples: np.ndarray) -> float:
    peak, rms = stable_peak(samples), stable_rms(samples)
    if rms == 0:
        return 0.0
    return 20 * math.log10(peak / rms)


def silence_fraction(samples: np.ndarray, *, floor_dbfs: float = -60
                     ) -> float:
    floor = finite_float(floor_dbfs, "silence floor dBFS",
                         minimum=-180, maximum=0)
    from .contracts import as_channel_first, check_audio
    x = as_channel_first(check_audio(samples), float64=True)
    threshold = 10 ** (floor / 20)
    return float(np.mean(np.max(np.abs(x), axis=0) < threshold))


def clipping_fraction(samples: np.ndarray, *, threshold: float = 1
                      ) -> float:
    bound = finite_float(threshold, "clip threshold", minimum=0.01, maximum=100)
    from .contracts import check_audio
    return float(np.mean(np.abs(check_audio(samples)) >= bound))


@dataclass(frozen=True, slots=True)
class FeatureSummary:
    channels: int
    sample_rate: int
    seconds: float
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    zero_crossing_rate: float
    silence_fraction: float
    clipping_fraction: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    spectral_flux: float

    def as_dict(self) -> dict[str, float | int]:
        from dataclasses import asdict
        return asdict(self)


def summarize_audio(buffer: AudioBuffer, *,
                    config: StftConfig | None = None,
                    silence_floor_dbfs: float = -60) -> FeatureSummary:
    if not isinstance(buffer, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer for quality features")
    spectrum = stft(buffer, config or StftConfig())
    peak, rms = stable_peak(buffer.samples), stable_rms(buffer.samples)
    return FeatureSummary(
        channels=buffer.channels,
        sample_rate=buffer.sample_rate,
        seconds=buffer.seconds,
        peak_dbfs=dbfs(peak),
        rms_dbfs=dbfs(rms),
        crest_db=crest_factor_db(buffer.samples),
        zero_crossing_rate=float(zero_crossing_rate(buffer.samples,
                                                    per_channel=False)),
        silence_fraction=silence_fraction(buffer.samples,
                                          floor_dbfs=silence_floor_dbfs),
        clipping_fraction=clipping_fraction(buffer.samples),
        spectral_centroid_hz=float(np.mean(spectral_centroid(spectrum))),
        spectral_bandwidth_hz=float(np.mean(spectral_bandwidth(spectrum))),
        spectral_rolloff_hz=float(np.mean(spectral_rolloff(spectrum))),
        spectral_flatness=float(np.mean(spectral_flatness(spectrum))),
        spectral_flux=float(np.mean(spectral_flux(spectrum))),
    )


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """HTK mel-frequency mapping, not a mel spectrogram implementation."""
    x = np.asarray(mel, dtype=np.float64)
    if not np.isfinite(x).all():
        raise AudioContractError("Invalid mel frequency")
    return 700 * (np.power(10, x / 2595) - 1)


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    x = np.asarray(hz, dtype=np.float64)
    if not np.isfinite(x).all() or np.any(x < 0):
        raise AudioContractError("Hertz values must be nonnegative")
    return 2595 * np.log10(1 + x / 700)


def triangular_mel_bank(sample_rate: int, fft_length: int, *,
                        bands: int = 40, fmin: float = 0,
                        fmax: float | None = None) -> np.ndarray:
    check_rate(sample_rate)
    if (type(fft_length) is not int or not 32 <= fft_length <= 8192
            or type(bands) is not int or not 2 <= bands <= 256):
        raise AudioContractError("Unsupported mel filterbank geometry")
    low = finite_float(fmin, "mel fmin", minimum=0)
    high = sample_rate / 2 if fmax is None else finite_float(
        fmax, "mel fmax", minimum=0
    )
    if high <= low or high > sample_rate / 2:
        raise AudioContractError("Mel upper boundary must be within Nyquist")
    hz = mel_to_hz(np.linspace(float(hz_to_mel(low)), float(hz_to_mel(high)),
                                   bands + 2))
    freqs = np.fft.rfftfreq(fft_length, 1 / sample_rate)
    output = np.zeros((bands, len(freqs)), dtype=np.float64)
    for index in range(bands):
        left, center, right = hz[index:index + 3]
        if center <= left or right <= center:
            raise AudioContractError("Collapsed mel triangle")
        up = (freqs - left) / (center - left)
        down = (right - freqs) / (right - center)
        output[index] = np.maximum(0, np.minimum(up, down))
    return output.astype(np.float32)


def mel_spectrogram(spec: Spectrogram, *, bands: int = 40,
                    fmin: float = 0,
                    fmax: float | None = None,
                    power: float = 2.0) -> np.ndarray:
    exponent = finite_float(power, "mel spectrum power", minimum=1, maximum=3)
    bank = triangular_mel_bank(spec.sample_rate, spec.config.n_fft,
                               bands=bands, fmin=fmin, fmax=fmax)
    source = np.abs(spec.spectrum) ** exponent
    return np.einsum("bf,cft->cbt", bank, source, optimize=True).astype(np.float32)
