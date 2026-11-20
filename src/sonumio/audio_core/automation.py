"""Bounded sample-accurate keyframe automation and tempo/frame conversion.

Automation curves are immutable, validated and deterministic. They are
designed for offline controls, not uncontrolled executable expressions.
A curve can modulate amplitude or a parameter, but cannot open files or
invoke a model. All breakpoints belong to one fixed integer sample clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Sequence
import numpy as np

from .contracts import AudioBuffer, check_rate, finite_float, MAX_BYTES
from .errors import AudioContractError, AudioLimitError


@dataclass(frozen=True, order=True, slots=True)
class Keyframe:
    frame: int
    value: float
    interpolation: str = "linear"

    def __post_init__(self):
        if type(self.frame) is not int or self.frame < 0:
            raise AudioContractError("Automation frame must be nonnegative integer")
        finite_float(self.value, "automation value", minimum=-1e6, maximum=1e6)
        if self.interpolation not in (
            "linear", "step", "smoothstep", "cosine", "exponential"
        ):
            raise AudioContractError("Unknown keyframe interpolation law")


@dataclass(frozen=True, slots=True)
class AutomationCurve:
    """Immutable ascending keyframe timeline with explicit outside behavior."""

    keys: tuple[Keyframe, ...]
    before: str = "hold"
    after: str = "hold"

    def __post_init__(self):
        if not isinstance(self.keys, tuple) or not 1 <= len(self.keys) <= 4096:
            raise AudioContractError("Automation requires 1..4096 keys")
        if any(not isinstance(x, Keyframe) for x in self.keys):
            raise AudioContractError("Expected Keyframe objects")
        positions = [x.frame for x in self.keys]
        if positions != sorted(set(positions)):
            raise AudioContractError("Keyframe frames must be strictly increasing")
        if self.before not in ("hold", "zero") or self.after not in ("hold", "zero"):
            raise AudioContractError("Invalid automation boundary mode")

    @property
    def start(self) -> int:
        return self.keys[0].frame

    @property
    def stop(self) -> int:
        return self.keys[-1].frame

    def value_at(self, frame: int) -> float:
        if type(frame) is not int or frame < 0:
            raise AudioContractError("Expected nonnegative keyframe query")
        if frame < self.start:
            return self.keys[0].value if self.before == "hold" else 0.0
        if frame > self.stop:
            return self.keys[-1].value if self.after == "hold" else 0.0
        if frame == self.stop or len(self.keys) == 1:
            return self.keys[-1].value
        # Binary search avoids scanning every frame for thousands of points.
        import bisect
        positions = [k.frame for k in self.keys]
        left = bisect.bisect_right(positions, frame) - 1
        a, b = self.keys[left:left+2]
        u = (frame - a.frame) / (b.frame - a.frame)
        return float(_blend(a.value, b.value, u, a.interpolation))

    def render(self, frames: int, *, start: int = 0) -> np.ndarray:
        """Create float32 automation values for a bounded output time span."""
        if (type(frames) is not int or frames <= 0 or frames > 10_000_000
                or type(start) is not int or start < 0):
            raise AudioContractError("Invalid automation render span")
        if frames * 8 > MAX_BYTES:
            raise AudioLimitError("Automation render exceeds memory cap")
        indices = np.arange(start, start + frames, dtype=np.int64)
        result = np.zeros(frames, dtype=np.float64)
        if self.before == "hold":
            result[indices < self.start] = self.keys[0].value
        if self.after == "hold":
            result[indices > self.stop] = self.keys[-1].value
        for a, b in zip(self.keys[:-1], self.keys[1:]):
            span = (indices >= a.frame) & (indices < b.frame)
            if not np.any(span):
                continue
            u = (indices[span] - a.frame) / (b.frame - a.frame)
            result[span] = _blend(a.value, b.value, u, a.interpolation)
        result[indices == self.stop] = self.keys[-1].value
        if len(self.keys) == 1:
            result[indices == self.start] = self.keys[0].value
        return result.astype(np.float32)

    def shifted(self, frames: int) -> "AutomationCurve":
        if type(frames) is not int or self.start + frames < 0:
            raise AudioContractError("Automation shift moves before origin")
        return AutomationCurve(tuple(
            Keyframe(k.frame + frames, k.value, k.interpolation)
            for k in self.keys
        ), self.before, self.after)

    def scaled(self, scale: float, offset: float = 0) -> "AutomationCurve":
        gain = finite_float(scale, "automation scale", minimum=-1000, maximum=1000)
        bias = finite_float(offset, "automation offset", minimum=-1e6, maximum=1e6)
        return AutomationCurve(tuple(
            Keyframe(k.frame, gain * k.value + bias, k.interpolation)
            for k in self.keys
        ), self.before, self.after)

    def clipped(self, minimum: float, maximum: float) -> "AutomationCurve":
        low = finite_float(minimum, "automation minimum")
        high = finite_float(maximum, "automation maximum")
        if low > high:
            raise AudioContractError("Invalid automation clipping limits")
        return AutomationCurve(tuple(
            Keyframe(k.frame, min(max(k.value, low), high), k.interpolation)
            for k in self.keys
        ), self.before, self.after)


def _blend(a: float, b: float, t, law: str):
    if law == "step":
        u = np.zeros_like(t, dtype=np.float64)
    elif law == "smoothstep":
        u = t * t * (3 - 2 * t)
    elif law == "cosine":
        u = (1 - np.cos(math.pi * t)) / 2
    elif law == "exponential":
        if a == 0 or b == 0 or (a < 0) != (b < 0):
            raise AudioContractError("Exponential automation requires same-sign nonzero endpoints")
        return a * np.power(b / a, t)
    else:
        u = t
    return a + (b - a) * u


def apply_gain_automation(audio: AudioBuffer, curve: AutomationCurve,
                          *, gain_unit: str = "linear") -> AudioBuffer:
    if not isinstance(audio, AudioBuffer) or not isinstance(curve, AutomationCurve):
        raise AudioContractError("Expected AudioBuffer and AutomationCurve")
    if gain_unit not in ("linear", "db"):
        raise AudioContractError("Automation gain unit must be linear or db")
    values = curve.render(audio.frames).astype(np.float64)
    if gain_unit == "db":
        if np.any((values < -120) | (values > 36)):
            raise AudioContractError("Automated dB range exceeds safe bounds")
        values = np.power(10, values / 20)
    elif np.any(np.abs(values) > 100):
        raise AudioContractError("Automated amplitude gain exceeds limit")
    output = audio.samples.astype(np.float64) * values[None, :]
    if not np.isfinite(output).all():
        raise AudioContractError("Gain automation produced nonfinite PCM")
    return audio.copy_with(output.astype(np.float32), source_note="sample-exact automation")


@dataclass(frozen=True, slots=True)
class Tempo:
    """Fixed BPM beat clock; not a mutable DAW timeline."""

    sample_rate: int
    bpm: Fraction
    numerator: int = 4
    denominator: int = 4

    def __post_init__(self):
        check_rate(self.sample_rate)
        if not isinstance(self.bpm, Fraction) or not 20 <= self.bpm <= 300:
            raise AudioContractError("Tempo BPM must be Fraction between 20 and 300")
        if type(self.numerator) is not int or not 1 <= self.numerator <= 16:
            raise AudioContractError("Time signature numerator must be 1..16")
        if type(self.denominator) is not int or self.denominator not in (1, 2, 4, 8, 16):
            raise AudioContractError("Unsupported time signature denominator")

    @classmethod
    def from_bpm(cls, rate: int, bpm: float, *, numerator: int = 4,
                 denominator: int = 4) -> "Tempo":
        value = finite_float(bpm, "BPM", minimum=20, maximum=300)
        return cls(rate, Fraction(str(value)), numerator, denominator)

    def beat_to_frame(self, beats: int | Fraction) -> int:
        if type(beats) is int:
            target = Fraction(beats)
        elif isinstance(beats, Fraction):
            target = beats
        else:
            raise AudioContractError("Beat position must be an exact Fraction/int")
        if target < 0:
            raise AudioContractError("Beat position cannot be negative")
        value = target * Fraction(60 * self.sample_rate, 1) / self.bpm
        return round(value)

    def bar_to_frame(self, bars: int | Fraction) -> int:
        beat_per_bar = Fraction(self.numerator * 4, self.denominator)
        return self.beat_to_frame(beat_per_bar * bars)

    def beat_grid(self, beats: int, *, start_beat: int = 0
                  ) -> tuple[int, ...]:
        if (type(beats) is not int or not 0 <= beats <= 10_000
                or type(start_beat) is not int or start_beat < 0):
            raise AudioContractError("Invalid bounded beat grid")
        return tuple(self.beat_to_frame(k)
                     for k in range(start_beat, start_beat + beats))


def envelope_adsr(frames: int, *, attack: int, decay: int,
                  sustain: float, release: int) -> np.ndarray:
    """Generate an offline sample-accurate ADSR envelope of fixed length."""
    if (type(frames) is not int or not 1 <= frames <= 10_000_000
            or any(type(v) is not int or v < 0 for v in (attack, decay, release))
            or attack + decay + release > frames):
        raise AudioContractError("ADSR phases must fit bounded clip frames")
    level = finite_float(sustain, "ADSR sustain", minimum=0, maximum=1)
    values = np.ones(frames, dtype=np.float64) * level
    pos = 0
    if attack:
        values[pos:pos + attack] = np.linspace(0, 1, attack, dtype=np.float64)
        pos += attack
    if decay:
        values[pos:pos + decay] = np.linspace(1, level, decay, dtype=np.float64)
    if release:
        values[-release:] = np.linspace(level, 0, release, dtype=np.float64)
    return values.astype(np.float32)
