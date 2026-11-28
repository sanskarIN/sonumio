"""Bounded non-destructive audio slicing and sample-clock trigger rendering.

Source AudioBuffer data are never modified. Slice boundaries are explicit
integer frames; each trigger can play forwards/reversed or at a declared
rate using bounded linear interpolation. This is basic resampling, not
pitch-preserving time stretch or perceptually anti-aliased synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, AudioLayout, AudioSpec, finite_float, make_buffer
from .errors import AudioContractError, AudioLimitError

MAX_SLICES = 1024
MAX_TRIGGERS = 2048
MAX_ACTIVE_VOICES = 64
MAX_RENDERED_SOURCE_SAMPLES = 35_000_000
MAX_SAMPLER_WORK_BYTES = 192 * 1024 * 1024


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class SampleSlice:
    """Half-open interval on one source AudioBuffer sample clock."""

    start_frame: int
    stop_frame: int
    label: str = ""

    def __post_init__(self):
        _integer(self.start_frame, "slice start", 0, 120 * 192000)
        _integer(self.stop_frame, "slice stop", 1, 120 * 192000)
        if self.stop_frame <= self.start_frame:
            raise AudioContractError("Slice stop must exceed start")
        if (type(self.label) is not str or len(self.label) > 80
                or any(ord(ch) < 32 for ch in self.label)):
            raise AudioContractError("Invalid slice label")

    @property
    def frames(self):
        return self.stop_frame - self.start_frame


@dataclass(frozen=True, slots=True)
class SamplerTrigger:
    start_frame: int
    slice_index: int
    velocity: float = 1.
    rate: float = 1.
    reverse: bool = False
    pan: float = 0.
    fade_frames: int = 8

    def __post_init__(self):
        _integer(self.start_frame, "trigger start", 0, 120 * 192000)
        _integer(self.slice_index, "slice index", 0, MAX_SLICES - 1)
        finite_float(self.velocity, "trigger velocity", minimum=0., maximum=1.)
        finite_float(self.rate, "trigger rate", minimum=.125, maximum=4.)
        finite_float(self.pan, "trigger pan", minimum=-1., maximum=1.)
        _integer(self.fade_frames, "trigger fade", 0, 8192)
        if type(self.reverse) is not bool:
            raise AudioContractError("Trigger reverse must be boolean")


@dataclass(frozen=True, slots=True)
class SamplerSettings:
    frames: int
    channels: int = 2
    gain: float = .5

    def __post_init__(self):
        _integer(self.frames, "output frames", 1, 120 * 192000)
        _integer(self.channels, "output channels", 1, 2)
        finite_float(self.gain, "sampler gain", minimum=0., maximum=1.)


@dataclass(frozen=True, slots=True)
class SamplerEvent:
    start_frame: int
    stop_frame: int
    slice_index: int
    source_start_frame: int
    source_stop_frame: int
    playback_rate: float
    reversed: bool
    velocity: float


@dataclass(frozen=True, slots=True)
class SamplerReport:
    sample_rate: int
    source_frames: int
    output_frames: int
    source_channels: int
    output_channels: int
    slice_count: int
    trigger_count: int
    peak_polyphony: int
    output_peak: float
    output_rms: float
    events: tuple[SamplerEvent, ...]
    protocol: str = "sonumio-nondestructive-sampler-v1"

    def as_dict(self):
        result = asdict(self)
        result["source_mutated"] = False
        result["model_inference"] = False
        result["pitch_preservation_claimed"] = False
        result["anti_aliasing_guaranteed"] = False
        return result


@dataclass(frozen=True, slots=True)
class SamplerResult:
    audio: AudioBuffer
    report: SamplerReport


def sample_slices_from_onsets(
    source: AudioBuffer, onset_frames: tuple[int, ...],
    *,
    include_lead: bool = True,
    min_frames: int = 1,
) -> tuple[SampleSlice, ...]:
    """Derive half-open regions between inspected integer onset times."""
    if not isinstance(source, AudioBuffer) or type(onset_frames) is not tuple:
        raise AudioContractError("Expected PCM source and tuple of onset frames")
    if type(include_lead) is not bool:
        raise AudioContractError("include_lead must be boolean")
    _integer(min_frames, "minimum slice frames", 1, source.frames)
    if len(onset_frames) > MAX_SLICES:
        raise AudioLimitError("Too many onsets for bounded sampler")
    if (any(type(n) is not int or not 0 < n < source.frames
            for n in onset_frames)
            or list(onset_frames) != sorted(set(onset_frames))):
        raise AudioContractError("Onsets must be strictly increasing interior frame positions")
    boundaries = (0, *onset_frames, source.frames)
    result = []
    for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        if i == 0 and not include_lead:
            continue
        if b - a >= min_frames:
            result.append(SampleSlice(a, b, label=f"slice-{len(result) + 1}"))
    if len(result) > MAX_SLICES:
        raise AudioLimitError("Too many materialized onset regions")
    return tuple(result)


def _output_duration(span: SampleSlice, trigger: SamplerTrigger) -> int:
    duration = math.ceil(span.frames / trigger.rate)
    if duration < 1:
        raise AudioContractError("Sampler trigger output duration is empty")
    return duration


def _preflight(
    source: AudioBuffer,
    slices: tuple[SampleSlice, ...],
    triggers: tuple[SamplerTrigger, ...],
    settings: SamplerSettings,
):
    if not isinstance(source, AudioBuffer) or not isinstance(settings, SamplerSettings):
        raise AudioContractError("Expected source AudioBuffer and SamplerSettings")
    if (type(slices) is not tuple or not 1 <= len(slices) <= MAX_SLICES
            or type(triggers) is not tuple or len(triggers) > MAX_TRIGGERS):
        raise AudioContractError("Expected bounded slice and trigger tuples")
    if (any(not isinstance(x, SampleSlice) for x in slices)
            or any(not isinstance(x, SamplerTrigger) for x in triggers)):
        raise AudioContractError("Invalid sampler slice or event")
    for sl in slices:
        if sl.stop_frame > source.frames:
            raise AudioContractError("Sampler slice exceeds source frames")
    if settings.channels != source.channels and not (
        source.channels == 1 and settings.channels == 2
    ):
        raise AudioContractError("Sampler topology supports native channels or mono-to-stereo")
    if settings.channels == 1 and any(t.pan != 0 for t in triggers):
        raise AudioContractError("Mono output cannot pan")
    if source.channels == 2 and any(t.pan != 0 for t in triggers):
        raise AudioContractError("Stereo source pan is not implicitly remapped")
    output_spec = AudioSpec(source.sample_rate,
                            AudioLayout.for_channels(settings.channels),
                            settings.frames)
    intervals = []
    work = 0
    for trigger in triggers:
        if trigger.slice_index >= len(slices):
            raise AudioContractError("Sampler trigger refers to nonexistent slice")
        sl = slices[trigger.slice_index]
        duration = _output_duration(sl, trigger)
        stop = trigger.start_frame + duration
        if stop > settings.frames:
            raise AudioContractError("Sampler voice extends past declared output")
        work += duration
        intervals.append((trigger.start_frame, stop))
    if work * settings.channels > MAX_RENDERED_SOURCE_SAMPLES:
        raise AudioLimitError("Sampler rendering exceeds total event work")
    if output_spec.byte_count * 4 + work * 24 > MAX_SAMPLER_WORK_BYTES:
        raise AudioLimitError("Sampler exceeds source mixing memory budget")
    events = sorted(
        [entry for start, stop in intervals
         for entry in ((start, +1), (stop, -1))],
        key=lambda pair: (pair[0], pair[1]),
    )
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
        if peak > MAX_ACTIVE_VOICES:
            raise AudioLimitError("Sampler exceeds active voice limit")
    return peak, intervals


def _render_slice(
    source: AudioBuffer,
    span: SampleSlice,
    trigger: SamplerTrigger,
) -> np.ndarray:
    """Bounded, floating fractional-frame playback of one immutable slice."""
    frames = _output_duration(span, trigger)
    local = np.arange(frames, dtype=np.float64) * trigger.rate
    local = np.clip(local, 0, span.frames - 1)
    if trigger.reverse:
        local = span.frames - 1 - local
    absolute = span.start_frame + local
    i = np.floor(absolute).astype(np.intp)
    j = np.minimum(i + 1, span.stop_frame - 1)
    fraction = absolute - i
    first = source.samples[:, i].astype(np.float64)
    second = source.samples[:, j].astype(np.float64)
    values = first * (1 - fraction[None, :]) + second * fraction[None, :]
    fade = min(trigger.fade_frames, frames // 2)
    if fade:
        ramp = np.arange(1, fade + 1, dtype=np.float64) / (fade + 1)
        values[:, :fade] *= ramp[None, :]
        values[:, -fade:] *= ramp[::-1][None, :]
    values *= trigger.velocity
    return values


def render_sampler(
    source: AudioBuffer,
    slices: tuple[SampleSlice, ...],
    triggers: tuple[SamplerTrigger, ...],
    settings: SamplerSettings,
) -> SamplerResult:
    """Render non-destructive clip triggers into finite float32 PCM."""
    maximum, intervals = _preflight(source, slices, triggers, settings)
    output = np.zeros((settings.channels, settings.frames), dtype=np.float64)
    events = []
    for trigger, (start, stop) in zip(triggers, intervals):
        span = slices[trigger.slice_index]
        audio = _render_slice(source, span, trigger) * settings.gain
        if source.channels == 1 and settings.channels == 2:
            angle = (trigger.pan + 1) * math.pi / 4
            output[0, start:stop] += audio[0] * math.cos(angle)
            output[1, start:stop] += audio[0] * math.sin(angle)
        else:
            output[:, start:stop] += audio
        events.append(SamplerEvent(
            start, stop, trigger.slice_index,
            span.start_frame, span.stop_frame,
            trigger.rate, trigger.reverse, trigger.velocity,
        ))
    if not np.isfinite(output).all():
        raise AudioContractError("Sampler mix produced nonfinite PCM")
    if float(np.max(np.abs(output))) > float(np.finfo(np.float32).max):
        raise AudioLimitError("Sampler output exceeds finite float32 range")
    result = make_buffer(
        output.astype(np.float32), source.sample_rate,
        layout=AudioLayout.for_channels(settings.channels),
        source_note="non-destructive sample-clock slicer",
    )
    report = SamplerReport(
        source.sample_rate, source.frames, result.frames,
        source.channels, result.channels, len(slices),
        len(triggers), maximum, float(np.max(np.abs(result.samples))),
        float(np.sqrt(np.mean(result.samples.astype(np.float64) ** 2))),
        tuple(events),
    )
    return SamplerResult(result, report)
