"""Frequency-selective de-esser with bounded causal envelope gain control.

Sibilance-like high-frequency energy is isolated with the shared STFT.
A causal RMS compressor then attenuates that band only. This is an
acoustic-band heuristic, not a speech-phoneme recognizer, learned voice
enhancer, or certified perceptual-quality improvement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError
from .stft import StftConfig, stft, istft, complex_mask

MAX_DEESS_CONTROL_SAMPLES = 4_000_000
MAX_DEESS_WORK_BYTES = 192 * 1024 * 1024


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class DeEsserSettings:
    n_fft: int = 1024
    hop_frames: int = 256
    band_low_hz: float = 3500.
    band_high_hz: float = 7800.
    threshold_dbfs: float = -28.
    ratio: float = 4.
    attack_seconds: float = .001
    release_seconds: float = .05
    maximum_reduction_db: float = 18.
    wet: float = 1.
    link_channels: bool = True

    def __post_init__(self):
        size = _integer(self.n_fft, "n_fft", 256, 8192)
        _integer(self.hop_frames, "hop_frames", 32, size // 2)
        low = finite_float(self.band_low_hz, "band_low_hz",
                           minimum=100., maximum=20000.)
        high = finite_float(self.band_high_hz, "band_high_hz",
                            minimum=100., maximum=20000.)
        if high <= low:
            raise AudioContractError("De-esser frequency band must increase")
        finite_float(self.threshold_dbfs, "threshold_dbfs",
                     minimum=-100., maximum=0.)
        finite_float(self.ratio, "ratio", minimum=1., maximum=30.)
        finite_float(self.attack_seconds, "attack_seconds",
                     minimum=0., maximum=2.)
        finite_float(self.release_seconds, "release_seconds",
                     minimum=0., maximum=4.)
        finite_float(self.maximum_reduction_db, "maximum_reduction_db",
                     minimum=0., maximum=48.)
        finite_float(self.wet, "wet", minimum=0., maximum=1.)
        if type(self.link_channels) is not bool:
            raise AudioContractError("De-esser channel linkage must be boolean")


@dataclass(frozen=True, slots=True)
class DeEsserReport:
    sample_rate: int
    channels: int
    frames: int
    band_low_hz: float
    band_high_hz: float
    threshold_dbfs: float
    ratio: float
    linked_channels: bool
    wet: float
    affected_frames: int
    minimum_gain: float
    mean_gain: float
    input_peak: float
    output_peak: float
    input_band_rms: float
    output_band_rms_estimate: float
    protocol: str = "sonumio-sibilance-band-dynamics-v1"

    def as_dict(self):
        data = asdict(self)
        data["model_inference"] = False
        data["speech_recognition_used"] = False
        data["sibilant_phoneme_detected"] = False
        data["perceptual_quality_established"] = False
        data["automatic_clipping_prevention"] = False
        return data


@dataclass(frozen=True, slots=True)
class DeEsserResult:
    audio: AudioBuffer
    report: DeEsserReport


def _preflight(audio: AudioBuffer, options: DeEsserSettings):
    if not isinstance(audio, AudioBuffer) or not isinstance(options, DeEsserSettings):
        raise AudioContractError("Expected AudioBuffer and DeEsserSettings")
    if options.band_high_hz > audio.sample_rate / 2:
        raise AudioContractError("De-esser upper band exceeds source Nyquist")
    if audio.frames * audio.channels > MAX_DEESS_CONTROL_SAMPLES:
        raise AudioLimitError("De-esser detector exceeds CPU work budget")
    if audio.frames * audio.channels * 8 * 9 > MAX_DEESS_WORK_BYTES:
        raise AudioLimitError("De-esser PCM exceeds bounded memory")


def isolate_sibilance(
    audio: AudioBuffer, options: DeEsserSettings = DeEsserSettings(),
) -> AudioBuffer:
    """Extract the selected spectral band into an immutable source-clock PCM."""
    _preflight(audio, options)
    cfg = StftConfig(
        n_fft=options.n_fft, hop_length=options.hop_frames,
        window="hann", center=True, pad_end=True,
    )
    spec = stft(audio, cfg)
    frequencies = spec.frequencies
    active = (
        (frequencies >= options.band_low_hz)
        & (frequencies <= options.band_high_hz)
    )
    if not np.any(active):
        raise AudioContractError("No FFT bins inside selected sibilance band")
    matrix = np.broadcast_to(
        active.reshape(1, -1, 1).astype(np.float64),
        spec.spectrum.shape,
    )
    isolated = istft(complex_mask(spec, matrix))
    if isolated.frames != audio.frames or isolated.channels != audio.channels:
        raise AudioContractError("Sibilance extraction changed PCM geometry")
    return audio.copy_with(
        isolated.samples, source_note="isolated high-frequency band",
    )


def deesser_gain(
    band: AudioBuffer,
    options: DeEsserSettings = DeEsserSettings(),
) -> np.ndarray:
    """Compute bounded causal gain envelopes on separated band PCM.

    Linked channels use a maximum-power detector to avoid inconsistent
    left/right gain jumps, not an unrequested waveform downmix.
    """
    _preflight(band, options)
    values = band.samples.astype(np.float64)
    channels, frames = values.shape
    output = np.empty((channels, frames), dtype=np.float64)
    attack = (
        math.exp(-1 / (options.attack_seconds * band.sample_rate))
        if options.attack_seconds else 0.
    )
    release = (
        math.exp(-1 / (options.release_seconds * band.sample_rate))
        if options.release_seconds else 0.
    )
    state = np.zeros(1 if options.link_channels else channels, dtype=np.float64)
    expansion = 1 - 1 / options.ratio
    for frame in range(frames):
        power = values[:, frame] ** 2
        if options.link_channels:
            power = np.array([np.max(power)], dtype=np.float64)
        attack_phase = power > state
        coefficient = np.where(attack_phase, attack, release)
        state = coefficient * state + (1 - coefficient) * power
        level_db = 10 * np.log10(np.maximum(state, 1e-24))
        over = np.maximum(level_db - options.threshold_dbfs, 0.)
        reduction = np.minimum(
            over * expansion, options.maximum_reduction_db,
        )
        gain = 10 ** (-reduction / 20)
        output[:, frame] = gain[0] if options.link_channels else gain
    if not np.isfinite(output).all() or np.any(output < 0) or np.any(output > 1):
        raise AudioContractError("De-esser gain calculation produced invalid values")
    return output


def deess(
    audio: AudioBuffer,
    options: DeEsserSettings = DeEsserSettings(),
) -> DeEsserResult:
    """Subtract bounded attenuated high-frequency energy from original PCM."""
    _preflight(audio, options)
    band = isolate_sibilance(audio, options)
    gains = deesser_gain(band, options)
    if options.wet == 0:
        output = audio
    else:
        attenuation = band.samples.astype(np.float64) * (
            1 - gains
        ) * options.wet
        restored = audio.samples.astype(np.float64) - attenuation
        if not np.isfinite(restored).all():
            raise AudioContractError("De-esser produced nonfinite output")
        if float(np.max(np.abs(restored))) > float(np.finfo(np.float32).max):
            raise AudioLimitError("De-esser output exceeds float32 range")
        output = audio.copy_with(
            restored.astype(np.float32),
            source_note="explicit sibilance-band dynamics",
        )

    def rms(samples):
        values = samples.astype(np.float64)
        return float(np.sqrt(np.mean(values * values)))

    report = DeEsserReport(
        audio.sample_rate, audio.channels, audio.frames,
        options.band_low_hz, options.band_high_hz,
        options.threshold_dbfs, options.ratio,
        options.link_channels, options.wet,
        int(np.count_nonzero(np.any(gains < 1 - 1e-9, axis=0))),
        float(np.min(gains)), float(np.mean(gains)),
        float(np.max(np.abs(audio.samples))),
        float(np.max(np.abs(output.samples))),
        rms(band.samples),
        rms(band.samples.astype(np.float64) * gains),
    )
    return DeEsserResult(output, report)
