"""Local, deterministic multi-track sound-scene composer.

Only the bundled procedural engine is supported. Each layer is rendered,
placed on a shared 24 kHz timeline, independently faded and gain-adjusted,
then globally peak-limited before export. It does not edit uploaded audio,
clone voices, or imply neural text-to-audio generation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from time import perf_counter
from typing import Sequence
import numpy as np

from .audio import (
    SAMPLE_RATE, AudioResult, encode_wav, encode_wav_channels,
    metrics, metrics_channels,
)
from .errors import GenerationError
from .service import Generated, generate_audio

MAX_SCENE_SECONDS = 30.0
MAX_LAYERS = 8


@dataclass(frozen=True)
class SceneLayer:
    prompt: str
    start_seconds: float = 0.0
    duration_seconds: float = 3.0
    gain_db: float = 0.0
    fade_in_seconds: float = 0.02
    fade_out_seconds: float = 0.05
    seed: int = 0
    pan: float = 0.0  # -1 fully left; 0 centered; +1 fully right (stereo opt-in)
    pan_end: float | None = None  # Optional pan sweep over this layer's frames


def _valid_number(value: float, minimum: float, maximum: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise GenerationError(f"{field} must be a finite number")
    if not minimum <= value <= maximum:
        raise GenerationError(f"{field} must be between {minimum} and {maximum}")
    return float(value)


def _validate_layer(layer: SceneLayer, total: float) -> None:
    _valid_number(layer.start_seconds, 0, MAX_SCENE_SECONDS, "start_seconds")
    _valid_number(layer.duration_seconds, 0.5, 15, "duration_seconds")
    _valid_number(layer.gain_db, -60, 12, "gain_db")
    _valid_number(layer.pan, -1, 1, "pan")
    if layer.pan_end is not None:
        _valid_number(layer.pan_end, -1, 1, "pan_end")
    _valid_number(layer.fade_in_seconds, 0, 10, "fade_in_seconds")
    _valid_number(layer.fade_out_seconds, 0, 10, "fade_out_seconds")
    if layer.fade_in_seconds + layer.fade_out_seconds > layer.duration_seconds + 1e-8:
        raise GenerationError("fade-in plus fade-out must not exceed layer duration")
    if layer.start_seconds + layer.duration_seconds > total + 1e-7:
        raise GenerationError("scene layer extends beyond total scene duration")
    if isinstance(layer.seed, bool) or not isinstance(layer.seed, int) or not 0 <= layer.seed < 2**32:
        raise GenerationError("seed must be an integer in [0, 2**32)")


def validate_scene_plan(layers: Sequence[SceneLayer], duration_seconds: float) -> None:
    """Reject invalid scene requests before placing expensive work on the queue."""
    total = _valid_number(duration_seconds, 0.5, MAX_SCENE_SECONDS, "scene duration_seconds")
    if not 1 <= len(layers) <= MAX_LAYERS:
        raise GenerationError(f"A scene must contain 1..{MAX_LAYERS} layers")
    for layer in layers:
        if not isinstance(layer, SceneLayer):
            raise GenerationError("Expected SceneLayer elements")
        _validate_layer(layer, total)
        from .validation import validate_request
        validate_request(layer.prompt, layer.duration_seconds)


def _pan_sweep_gains(start_pan: float, end_pan: float,
                     frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power stereo gains at exact frame positions, including endpoints.

    The first rendered sample uses start_pan and the last uses end_pan. A
    one-frame interval uses the start position. Exact +/-1 endpoints zero the
    opposite channel despite floating-point trigonometric roundoff.
    """
    if frames < 1:
        raise ValueError("Pan sweep requires at least one sample")
    positions = np.linspace(start_pan, end_pan, frames, dtype=np.float64)
    angle = (positions + 1.0) * (math.pi / 4.0)
    left = np.cos(angle)
    right = np.sin(angle)
    left[positions == 1.0] = 0.0
    right[positions == -1.0] = 0.0
    return left, right


def compose_scene(layers: Sequence[SceneLayer], duration_seconds: float) -> Generated:
    validate_scene_plan(layers, duration_seconds)
    total = float(duration_seconds)
    start = perf_counter()
    frames = round(total * SAMPLE_RATE)
    # A move away from center requires stereo even when its start is centered.
    stereo = any(layer.pan != 0.0 or (
        layer.pan_end is not None and layer.pan_end != 0.0
    ) for layer in layers)
    mix = (np.zeros((2, frames), dtype=np.float64) if stereo
           else np.zeros(frames, dtype=np.float64))
    for layer in layers:
        # No arbitrary third-party execution or unreviewed model routing.
        audio = generate_audio(layer.prompt, layer.duration_seconds,
                               model_id="procedural-sfx-v0", seed=layer.seed)
        samples = audio.result.samples.astype(np.float64).copy()
        count = len(samples)
        attack = round(layer.fade_in_seconds * SAMPLE_RATE)
        release = round(layer.fade_out_seconds * SAMPLE_RATE)
        if attack:
            samples[:attack] *= np.sin(np.linspace(0, np.pi / 2, attack, endpoint=True)) ** 2
        if release:
            samples[-release:] *= np.cos(np.linspace(0, np.pi / 2, release, endpoint=True)) ** 2
        samples *= 10 ** (layer.gain_db / 20)
        position = round(layer.start_seconds * SAMPLE_RATE)
        if position + count > frames:
            # Floating point rounding might produce a one-sample discrepancy.
            if position + count - frames > 1:
                raise GenerationError("Layer exceeds output sample range")
            samples = samples[:frames - position]
        if stereo:
            end_pan = layer.pan if layer.pan_end is None else layer.pan_end
            if end_pan != layer.pan:
                # Frame-indexed motion is independent of scene position,
                # layer overlap, rendering seeds and output quantization.
                left_gain, right_gain = _pan_sweep_gains(
                    layer.pan, end_pan, len(samples),
                )
            else:
                # Keep existing constant-pan arithmetic byte-identical.
                # Exact endpoint branches avoid small opposite-channel leaks.
                if layer.pan == -1.0:
                    left_gain, right_gain = 1.0, 0.0
                elif layer.pan == 1.0:
                    left_gain, right_gain = 0.0, 1.0
                else:
                    angle = (layer.pan + 1.0) * math.pi / 4.0
                    left_gain, right_gain = math.cos(angle), math.sin(angle)
            mix[0, position:position + len(samples)] += samples * left_gain
            mix[1, position:position + len(samples)] += samples * right_gain
        else:
            # Preserve exact v0.0.1 mono behavior when no pan is requested.
            mix[position:position + len(samples)] += samples
    pre_limit_peak = float(np.max(np.abs(mix)))
    gain = min(1.0, 0.95 / pre_limit_peak) if pre_limit_peak else 1.0
    mix = (mix * gain).astype(np.float32)
    output = AudioResult(samples=mix, sample_rate=SAMPLE_RATE, engine_id="procedural-scene-v0", seed=0)
    information = metrics_channels(mix) if stereo else metrics(mix)
    elapsed = perf_counter() - start
    information.update({
        "generation_wall_seconds": round(elapsed, 6),
        "real_time_factor": round(elapsed / total, 6),
        "pre_limit_peak": round(pre_limit_peak, 6),
        "master_gain": round(gain, 6),
        "layer_count": len(layers),
    })
    wav_bytes = (encode_wav_channels(mix, SAMPLE_RATE) if stereo
                 else encode_wav(mix, SAMPLE_RATE))
    return Generated(result=output, wav_bytes=wav_bytes, metrics=information)
