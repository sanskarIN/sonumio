"""Envelope followers, peak normalization, soft-knee dynamics and gates.

These bounded offline processors are deliberately transparent and do not
claim calibrated LUFS loudness, broadcast compliance or transparent mastering.
Time constants are sample-clock aware; a shared detector links all channels
and preserves the spatial image rather than pumping L/R independently.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from .contracts import AudioBuffer, finite_float, as_channel_first
from .errors import AudioContractError
from .numerics import db_to_gain, stable_peak, stable_rms


def _smoothing(seconds: float, sample_rate: int) -> float:
    time = finite_float(seconds, "smoothing time", minimum=0, maximum=10)
    if time == 0:
        return 0
    return math.exp(-1.0 / (time * sample_rate))


@dataclass(frozen=True, slots=True)
class CompressorConfig:
    threshold_dbfs: float = -18
    ratio: float = 4
    attack_seconds: float = 0.01
    release_seconds: float = 0.1
    knee_db: float = 6
    makeup_db: float = 0
    link_channels: bool = True

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("threshold_dbfs", self.threshold_dbfs, -96, 0),
            ("ratio", self.ratio, 1, 40),
            ("attack_seconds", self.attack_seconds, 0, 2),
            ("release_seconds", self.release_seconds, 0, 10),
            ("knee_db", self.knee_db, 0, 24),
            ("makeup_db", self.makeup_db, -24, 36),
        ):
            finite_float(value, name, minimum=lower, maximum=upper)
        if type(self.link_channels) is not bool:
            raise AudioContractError("Compressor channel linkage must be boolean")


def envelope_follower(samples: np.ndarray, sample_rate: int, *,
                      attack_seconds: float = 0.01,
                      release_seconds: float = 0.1,
                      detector: str = "peak",
                      link_channels: bool = True) -> np.ndarray:
    from .contracts import check_rate
    check_rate(sample_rate)
    if detector not in ("peak", "rms"):
        raise AudioContractError("Envelope detector must be peak or rms")
    if type(link_channels) is not bool:
        raise AudioContractError("Invalid envelope linkage")
    source = as_channel_first(samples, float64=True)
    attack = _smoothing(attack_seconds, sample_rate)
    release = _smoothing(release_seconds, sample_rate)
    if link_channels:
        if detector == "peak":
            values = np.max(np.abs(source), axis=0)[None, :]
        else:
            values = np.sqrt(np.mean(source ** 2, axis=0))[None, :]
    else:
        values = np.abs(source)
    out = np.empty(values.shape, dtype=np.float64)
    prev = np.zeros(values.shape[0], dtype=np.float64)
    for n in range(values.shape[1]):
        current = values[:, n]
        coeff = np.where(current > prev, attack, release)
        prev = coeff * prev + (1 - coeff) * current
        out[:, n] = prev
    return out.astype(np.float32)


def _compression_db(level_db: np.ndarray, config: CompressorConfig) -> np.ndarray:
    threshold, ratio, knee = (
        config.threshold_dbfs, config.ratio, config.knee_db
    )
    above = level_db - threshold
    if knee == 0:
        return np.maximum(above, 0) * (1 / ratio - 1)
    lower = above <= -knee / 2
    upper = above >= knee / 2
    soft = (1 / ratio - 1) * ((above + knee / 2) ** 2) / (2 * knee)
    hard = (1 / ratio - 1) * above
    return np.where(lower, 0, np.where(upper, hard, soft))


def compress(audio: AudioBuffer, config: CompressorConfig = CompressorConfig()
             ) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer) or not isinstance(config, CompressorConfig):
        raise AudioContractError("Expected AudioBuffer and CompressorConfig")
    source = audio.mutable_copy(dtype="float64")
    envelope = envelope_follower(
        audio.samples, audio.sample_rate, attack_seconds=config.attack_seconds,
        release_seconds=config.release_seconds, link_channels=config.link_channels
    ).astype(np.float64)
    level_db = 20 * np.log10(np.maximum(envelope, 1e-12))
    change_db = _compression_db(level_db, config) + config.makeup_db
    gain = np.power(10, change_db / 20)
    if gain.shape[0] == 1 and audio.channels > 1:
        gain = np.repeat(gain, audio.channels, axis=0)
    output = (source * gain).astype(np.float32)
    if not np.isfinite(output).all():
        raise AudioContractError("Compressor produced nonfinite samples")
    return audio.copy_with(output, source_note="offline compressor")


@dataclass(frozen=True, slots=True)
class GateConfig:
    open_threshold_dbfs: float = -42
    close_threshold_dbfs: float = -50
    attack_seconds: float = 0.005
    release_seconds: float = 0.08
    hold_seconds: float = 0.02
    floor_db: float = -90

    def __post_init__(self) -> None:
        for name, value, low, high in (
            ("open_threshold_dbfs", self.open_threshold_dbfs, -120, 0),
            ("close_threshold_dbfs", self.close_threshold_dbfs, -120, 0),
            ("attack_seconds", self.attack_seconds, 0, 2),
            ("release_seconds", self.release_seconds, 0, 10),
            ("hold_seconds", self.hold_seconds, 0, 5),
            ("floor_db", self.floor_db, -120, 0),
        ):
            finite_float(value, name, minimum=low, maximum=high)
        if self.close_threshold_dbfs > self.open_threshold_dbfs:
            raise AudioContractError("Gate close threshold exceeds open threshold")


def noise_gate(audio: AudioBuffer, config: GateConfig = GateConfig()) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer) or not isinstance(config, GateConfig):
        raise AudioContractError("Expected AudioBuffer and GateConfig")
    data = audio.mutable_copy(dtype="float64")
    envelope = envelope_follower(
        audio.samples, audio.sample_rate,
        attack_seconds=max(.0001, config.attack_seconds),
        release_seconds=config.release_seconds
    )[0].astype(np.float64)
    open_value, close_value = (
        db_to_gain(config.open_threshold_dbfs),
        db_to_gain(config.close_threshold_dbfs),
    )
    hold = round(config.hold_seconds * audio.sample_rate)
    open_state, remain = False, 0
    target = np.empty(len(envelope), dtype=np.float64)
    for i, level in enumerate(envelope):
        if level >= open_value:
            open_state, remain = True, hold
        elif level < close_value:
            if remain:
                remain -= 1
            else:
                open_state = False
        target[i] = 1.0 if open_state else db_to_gain(config.floor_db)
    # Apply separate smooth attack/release to reduce gate clicks.
    attack = _smoothing(config.attack_seconds, audio.sample_rate)
    release = _smoothing(config.release_seconds, audio.sample_rate)
    gain = np.empty_like(target)
    previous = target[0]
    for i, desired in enumerate(target):
        coeff = attack if desired > previous else release
        previous = coeff * previous + (1 - coeff) * desired
        gain[i] = previous
    return audio.copy_with((data * gain[None, :]).astype(np.float32),
                           source_note="hysteretic noise gate")


def peak_normalize(audio: AudioBuffer, *, target_dbfs: float = -1,
                   allow_boost: bool = True) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer) or type(allow_boost) is not bool:
        raise AudioContractError("Expected AudioBuffer and boolean boost")
    db = finite_float(target_dbfs, "target peak dBFS", minimum=-80, maximum=0)
    original = stable_peak(audio.samples)
    if original == 0:
        return audio.copy_with(audio.samples, source_note="silent peak normalization")
    target = db_to_gain(db)
    scale = target / original
    if not allow_boost:
        scale = min(1, scale)
    return audio.copy_with((audio.samples.astype(np.float64) * scale).astype(np.float32),
                           source_note="explicit peak-normalized audio")


def rms_normalize(audio: AudioBuffer, *, target_dbfs: float = -20,
                  max_gain_db: float = 24) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    db = finite_float(target_dbfs, "target RMS dBFS", minimum=-80, maximum=0)
    limit = finite_float(max_gain_db, "max_gain_db", minimum=0, maximum=48)
    old = stable_rms(audio.samples)
    if old < 1e-12:
        return audio.copy_with(audio.samples, source_note="silent RMS normalization")
    gain = min(db_to_gain(limit), db_to_gain(db) / old)
    return audio.copy_with((audio.samples.astype(np.float64) * gain).astype(np.float32),
                           source_note="explicit RMS-normalized audio")


def peak_limit(audio: AudioBuffer, *, ceiling_dbfs: float = -0.5,
               release_seconds: float = 0.05) -> AudioBuffer:
    """Predictive offline sample peak limiter (not oversampled true-peak)."""
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    ceiling = db_to_gain(finite_float(ceiling_dbfs, "limiter ceiling",
                                     minimum=-40, maximum=0))
    release = _smoothing(release_seconds, audio.sample_rate)
    samples = audio.mutable_copy(dtype="float64")
    peaks = np.max(np.abs(samples), axis=0)
    required = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-20))
    # Look backward for future peaks: conservative offline attenuation.
    envelope = np.ones(audio.frames, dtype=np.float64)
    next_gain = 1.0
    for i in range(audio.frames - 1, -1, -1):
        if required[i] < next_gain:
            next_gain = required[i]
        else:
            next_gain = release * next_gain + (1 - release) * required[i]
        envelope[i] = min(next_gain, required[i])
    output = samples * envelope[None, :]
    np.clip(output, -ceiling, ceiling, out=output)
    return audio.copy_with(output.astype(np.float32),
                           source_note="offline sample peak limited")


def dynamic_range_report(audio: AudioBuffer) -> dict[str, float]:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    peak = stable_peak(audio.samples)
    rms = stable_rms(audio.samples)
    if peak <= 1e-12:
        crest = 0.0
    else:
        crest = 20 * math.log10(peak / max(rms, 1e-12))
    absolute = np.max(np.abs(audio.samples.astype(np.float64)), axis=0)
    return {
        "peak": peak,
        "rms": rms,
        "crest_db": crest,
        "fraction_above_0_dbfs": float(np.mean(absolute >= 1)),
        "fraction_above_minus_1_dbfs": float(np.mean(absolute >= db_to_gain(-1))),
        "noise_floor_estimating_method": "none",
    }
