"""Reproducible offline modulation, delay, convolution and reverb effects.

These are signal transformations, not generative AI models. Each effect uses
bounded feedback, explicit wet/dry control and finite output checks. The
stereo image is preserved through shared causal delay behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .contracts import AudioBuffer, MAX_BYTES, finite_float
from .errors import AudioContractError, AudioLimitError
from .filters import fir_filter


def _wet_dry(wet: float) -> tuple[float, float]:
    mix = finite_float(wet, "wet level", minimum=0, maximum=1)
    return 1 - mix, mix


def _effect_result(audio: AudioBuffer, values: np.ndarray, name: str
                   ) -> AudioBuffer:
    if (not isinstance(values, np.ndarray) or values.shape != audio.samples.shape
            or not np.isfinite(values).all()):
        raise AudioContractError("Effect " + name + " produced invalid audio")
    return audio.copy_with(values.astype(np.float32), source_note=name)


@dataclass(frozen=True, slots=True)
class DelayConfig:
    delay_seconds: float = 0.24
    feedback: float = 0.25
    wet: float = 0.25
    taps: int = 1
    ping_pong: bool = False

    def __post_init__(self) -> None:
        finite_float(self.delay_seconds, "delay seconds", minimum=.001, maximum=4)
        finite_float(self.feedback, "feedback", minimum=0, maximum=.95)
        _wet_dry(self.wet)
        if type(self.taps) is not int or not 1 <= self.taps <= 16:
            raise AudioContractError("Delay taps must be 1..16")
        if type(self.ping_pong) is not bool:
            raise AudioContractError("ping_pong must be boolean")


def delay(audio: AudioBuffer, config: DelayConfig = DelayConfig()
          ) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer) or not isinstance(config, DelayConfig):
        raise AudioContractError("Expected AudioBuffer and DelayConfig")
    delay_frames = max(1, round(audio.sample_rate * config.delay_seconds))
    if delay_frames > 4 * audio.sample_rate:
        raise AudioLimitError("Delay exceeds configured maximum")
    samples = audio.mutable_copy(dtype="float64")
    taps = min(config.taps, (audio.frames - 1) // delay_frames)
    wet_out = np.zeros_like(samples)
    if taps <= 0 or config.wet == 0:
        return _effect_result(audio, samples, "dry no-op delay")
    for tap in range(1, taps + 1):
        offset = tap * delay_frames
        amplitude = config.feedback ** (tap - 1)
        if config.ping_pong and audio.channels == 2 and tap % 2:
            wet_out[0, offset:] += samples[1, :-offset] * amplitude
            wet_out[1, offset:] += samples[0, :-offset] * amplitude
        else:
            wet_out[:, offset:] += samples[:, :-offset] * amplitude
    dry, wet = _wet_dry(config.wet)
    return _effect_result(audio, dry * samples + wet * wet_out, "multi-tap delay")


@dataclass(frozen=True, slots=True)
class ModulationConfig:
    rate_hz: float = 1
    depth: float = .5
    wet: float = .5
    waveform: str = "sine"
    phase: float = 0

    def __post_init__(self) -> None:
        finite_float(self.rate_hz, "modulation rate", minimum=.01, maximum=30)
        finite_float(self.depth, "modulation depth", minimum=0, maximum=1)
        _wet_dry(self.wet)
        finite_float(self.phase, "modulation phase", minimum=-100, maximum=100)
        if self.waveform not in ("sine", "triangle", "square", "saw"):
            raise AudioContractError("Unsupported modulation waveform")


def _oscillator(t: np.ndarray, config: ModulationConfig) -> np.ndarray:
    phase = config.rate_hz * t + config.phase / (2 * math.pi)
    if config.waveform == "sine":
        return np.sin(2 * math.pi * phase)
    if config.waveform == "triangle":
        return 4 * np.abs(phase - np.floor(phase + .5)) - 1
    if config.waveform == "square":
        return np.where(np.sin(2 * math.pi * phase) >= 0, 1., -1.)
    return 2 * (phase - np.floor(phase + .5))


def tremolo(audio: AudioBuffer, config: ModulationConfig = ModulationConfig()
            ) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer) or not isinstance(config, ModulationConfig):
        raise AudioContractError("Expected AudioBuffer and ModulationConfig")
    t = np.arange(audio.frames, dtype=np.float64) / audio.sample_rate
    lfo = _oscillator(t, config)
    amplitude = 1 - config.depth / 2 + (config.depth / 2) * lfo
    dry, wet = _wet_dry(config.wet)
    return _effect_result(
        audio,
        audio.samples.astype(np.float64) * (dry + wet * amplitude[None, :]),
        "tremolo",
    )


def ring_modulate(audio: AudioBuffer, config: ModulationConfig
                  ) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer) or not isinstance(config, ModulationConfig):
        raise AudioContractError("Expected AudioBuffer and ModulationConfig")
    t = np.arange(audio.frames, dtype=np.float64) / audio.sample_rate
    oscillator = _oscillator(t, config)
    dry, wet = _wet_dry(config.wet)
    return _effect_result(
        audio, audio.samples.astype(np.float64)
        * (dry + wet * oscillator[None, :] * config.depth), "ring modulation"
    )


def chorus(audio: AudioBuffer, *, rate_hz: float = 0.55,
           depth_ms: float = 2, base_delay_ms: float = 12,
           wet: float = .35, voices: int = 2) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    rate = finite_float(rate_hz, "chorus rate", minimum=.01, maximum=10)
    depth = finite_float(depth_ms, "chorus depth ms", minimum=0, maximum=20)
    base = finite_float(base_delay_ms, "chorus base delay ms",
                        minimum=1, maximum=60)
    if type(voices) is not int or not 1 <= voices <= 8:
        raise AudioContractError("Chorus voices must be 1..8")
    dry, wet_value = _wet_dry(wet)
    original = audio.mutable_copy(dtype="float64")
    n = audio.frames
    clock = np.arange(n, dtype=np.float64)
    total = np.zeros_like(original)
    for voice in range(voices):
        modulation = (depth / 1000 * audio.sample_rate
                      * np.sin(2 * math.pi * rate * clock / audio.sample_rate
                               + 2 * math.pi * voice / voices))
        delayed = (base / 1000 * audio.sample_rate) + modulation
        positions = clock - delayed
        left = np.floor(positions).astype(np.int64)
        fraction = positions - left
        safe_left = np.clip(left, 0, n - 1)
        safe_right = np.clip(left + 1, 0, n - 1)
        valid = positions >= 0
        for channel in range(audio.channels):
            result = ((1 - fraction) * original[channel, safe_left]
                      + fraction * original[channel, safe_right])
            total[channel] += np.where(valid, result, 0)
    total /= voices
    return _effect_result(audio, dry * original + wet_value * total, "chorus")


def flanger(audio: AudioBuffer, *, rate_hz: float = .3,
            depth_ms: float = 3, base_delay_ms: float = 3,
            feedback: float = 0, wet: float = .5) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    rate = finite_float(rate_hz, "flanger rate", minimum=.01, maximum=5)
    depth = finite_float(depth_ms, "flanger depth", minimum=0, maximum=10)
    base = finite_float(base_delay_ms, "flanger base delay", minimum=0.1, maximum=20)
    feedback = finite_float(feedback, "flanger feedback", minimum=-.85, maximum=.85)
    dry, wet_value = _wet_dry(wet)
    original = audio.mutable_copy(dtype="float64")
    out = np.empty_like(original)
    effected = np.zeros_like(original)
    for frame in range(audio.frames):
        delay_frames = (base + depth * math.sin(
            2 * math.pi * rate * frame / audio.sample_rate)) / 1000 * audio.sample_rate
        previous = frame - max(1, round(delay_frames))
        if previous >= 0:
            effected[:, frame] = (
                original[:, previous] + feedback * effected[:, previous]
            )
        out[:, frame] = dry * original[:, frame] + wet_value * effected[:, frame]
    return _effect_result(audio, out, "flanger")


def convolution_reverb(audio: AudioBuffer, impulse: np.ndarray, *,
                       wet: float = .35, normalization: str = "peak",
                       truncate_tail: bool = True) -> AudioBuffer:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    if (not isinstance(impulse, np.ndarray) or impulse.ndim != 1
            or impulse.dtype not in (np.float32, np.float64)
            or not 1 <= len(impulse) <= 4095 or not np.isfinite(impulse).all()):
        raise AudioContractError("Expected finite mono 1..4095-sample impulse")
    if normalization not in ("none", "peak", "energy"):
        raise AudioContractError("Invalid reverb impulse normalization")
    dry, wet_value = _wet_dry(wet)
    kernel = impulse.astype(np.float64)
    if normalization == "peak":
        maximum = float(np.max(np.abs(kernel)))
        if maximum > 0:
            kernel /= maximum
    elif normalization == "energy":
        norm = float(np.sqrt(np.sum(kernel ** 2)))
        if norm > 0:
            kernel /= norm
    output = fir_filter(audio.samples, kernel, mode="full")
    if truncate_tail:
        output = output[:, :audio.frames]
        original = audio.samples.astype(np.float64)
    else:
        if (output.shape[1] * audio.channels * 8 > MAX_BYTES
                or output.shape[1] > audio.sample_rate * 120):
            raise AudioLimitError("Reverb tail extends beyond memory bound")
        original = np.zeros_like(output, dtype=np.float64)
        original[:, :audio.frames] = audio.samples
    combined = dry * original + wet_value * output.astype(np.float64)
    if not truncate_tail:
        from .contracts import make_buffer
        return make_buffer(combined.astype(np.float32), audio.sample_rate,
                           layout=audio.spec.layout,
                           source_note="convolution reverb extended tail")
    return _effect_result(audio, combined, "convolution reverb")


def effect_report(input_audio: AudioBuffer, output_audio: AudioBuffer,
                  effect_name: str) -> dict:
    if (not isinstance(input_audio, AudioBuffer)
            or not isinstance(output_audio, AudioBuffer)
            or input_audio.sample_rate != output_audio.sample_rate
            or input_audio.channels != output_audio.channels
            or not isinstance(effect_name, str) or not effect_name):
        raise AudioContractError("Invalid effect report source/output")
    delta = output_audio.frames - input_audio.frames
    from .numerics import stable_peak, stable_rms
    return {
        "effect": effect_name,
        "input_peak": stable_peak(input_audio.samples),
        "output_peak": stable_peak(output_audio.samples),
        "input_rms": stable_rms(input_audio.samples),
        "output_rms": stable_rms(output_audio.samples),
        "output_sample_rate": output_audio.sample_rate,
        "new_tail_frames": delta,
        "original_generator_unchanged": True,
        "model_inference": False,
    }
