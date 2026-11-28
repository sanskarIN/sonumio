"""Deterministic damped delay-line plucked-string reference synthesis.

Karplus-Strong-type excitation of a short feedback delay models a
decaying string-like tone. The one-pole averaging loop is approximate:
pitch can be detuned by integer delay rounding and depends on damping,
and the output is not a physical calibrated instrument simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, AudioLayout, AudioSpec, check_rate, finite_float, make_buffer
from .errors import AudioContractError, AudioLimitError

MAX_PLUCK_NOTES = 1024
MAX_PLUCK_POLYPHONY = 32
MAX_TOTAL_PLUCK_SAMPLES = 24_000_000
MAX_PLUCK_BYTES = 192 * 1024 * 1024


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class StringVoice:
    damping: float = .996
    brightness: float = .5
    pluck_position: float = .22
    attack_frames: int = 8
    release_frames: int = 64
    excitation_smoothing: int = 3

    def __post_init__(self):
        finite_float(self.damping, "string damping", minimum=.80, maximum=.99999)
        finite_float(self.brightness, "string brightness", minimum=0., maximum=1.)
        finite_float(self.pluck_position, "pluck position", minimum=.02, maximum=.48)
        _integer(self.attack_frames, "attack_frames", 0, 8192)
        _integer(self.release_frames, "release_frames", 0, 8192)
        _integer(self.excitation_smoothing, "excitation smoothing", 1, 15)


@dataclass(frozen=True, slots=True)
class PluckNote:
    start_frame: int
    duration_frames: int
    midi_note: int
    velocity: float = 1.
    pan: float = 0.
    seed: int = 0

    def __post_init__(self):
        _integer(self.start_frame, "note start", 0, 120 * 192000)
        _integer(self.duration_frames, "note duration", 1, 120 * 192000)
        _integer(self.midi_note, "MIDI note", 0, 127)
        finite_float(self.velocity, "velocity", minimum=0., maximum=1.)
        finite_float(self.pan, "pan", minimum=-1., maximum=1.)
        _integer(self.seed, "excitation seed", 0, 2**32 - 1)

    @property
    def frequency_hz(self):
        return 440 * 2 ** ((self.midi_note - 69) / 12)


@dataclass(frozen=True, slots=True)
class PluckedStringPlan:
    sample_rate: int = 24000
    frames: int = 24000
    channels: int = 2
    gain: float = .2
    voice: StringVoice = StringVoice()

    def __post_init__(self):
        check_rate(self.sample_rate)
        _integer(self.frames, "output frames", 1, 120 * self.sample_rate)
        _integer(self.channels, "channels", 1, 2)
        finite_float(self.gain, "plucked string gain", minimum=0., maximum=1.)
        if not isinstance(self.voice, StringVoice):
            raise AudioContractError("Expected StringVoice")
        AudioSpec(self.sample_rate, AudioLayout.for_channels(self.channels), self.frames)


@dataclass(frozen=True, slots=True)
class PluckStats:
    start_frame: int
    stop_frame: int
    midi_note: int
    nominal_frequency_hz: float
    approximate_loop_frequency_hz: float
    loop_samples: int
    velocity: float


@dataclass(frozen=True, slots=True)
class PluckedStringReport:
    sample_rate: int
    frames: int
    channels: int
    notes: int
    maximum_polyphony: int
    peak: float
    rms: float
    stats: tuple[PluckStats, ...]
    protocol: str = "sonumio-karplus-strong-reference-v1"

    def as_dict(self):
        value = asdict(self)
        value["model_inference"] = False
        value["physical_instrument_accuracy_claimed"] = False
        value["pitch_is_exact"] = False
        value["automatic_normalization"] = False
        return value


@dataclass(frozen=True, slots=True)
class PluckedStringResult:
    audio: AudioBuffer
    report: PluckedStringReport


def loop_length(plan: PluckedStringPlan, note: PluckNote) -> int:
    if not isinstance(plan, PluckedStringPlan) or not isinstance(note, PluckNote):
        raise AudioContractError("Expected PluckedStringPlan and PluckNote")
    # The loop interpolation term contributes a small phase delay.
    fraction = (1 - plan.voice.brightness) * .5
    length = round(plan.sample_rate / note.frequency_hz - fraction)
    if not 4 <= length <= 16000:
        raise AudioContractError("String note outside stable feedback loop range")
    return length


def _excite(length: int, voice: StringVoice, seed: int) -> np.ndarray:
    # NumPy PCG64 uses a deterministic bitstream for the given seed.
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1., 1., length)
    if voice.excitation_smoothing > 1:
        smooth = voice.excitation_smoothing
        padded = np.pad(noise, (smooth - 1, 0), mode="wrap")
        noise = np.convolve(padded, np.ones(smooth) / smooth, mode="valid")
    shift = max(1, round(length * voice.pluck_position))
    # Displacement-like pluck excitation: spatial comb creates overtone
    # notches that depend explicitly on the pick position.
    excitation = noise - np.roll(noise, shift)
    peak = float(np.max(np.abs(excitation)))
    if peak:
        excitation /= peak
    return excitation


def render_string_voice(
    note: PluckNote,
    sample_rate: int,
    voice: StringVoice,
) -> tuple[np.ndarray, float, int]:
    """Generate exactly one damped note, including bounded release samples."""
    check_rate(sample_rate)
    if not isinstance(note, PluckNote) or not isinstance(voice, StringVoice):
        raise AudioContractError("Expected note event and string voice")
    length = round(sample_rate / note.frequency_hz - (1 - voice.brightness) / 2)
    if not 4 <= length <= 16000:
        raise AudioContractError("String note outside stable loop range")
    frames = note.duration_frames + voice.release_frames
    if frames > MAX_TOTAL_PLUCK_SAMPLES:
        raise AudioLimitError("Single pluck exceeds maximum rendered samples")
    ring = _excite(length, voice, note.seed)
    output = np.empty(frames, dtype=np.float64)
    pointer = 0
    previous = 0.
    for n in range(frames):
        current = float(ring[pointer])
        output[n] = current
        filtered = voice.brightness * current + (
            1 - voice.brightness
        ) * (current + previous) * .5
        ring[pointer] = voice.damping * filtered
        previous = current
        pointer = (pointer + 1) % length
    if voice.attack_frames:
        steps = min(voice.attack_frames, frames)
        output[:steps] *= np.arange(1, steps + 1) / steps
    if voice.release_frames:
        output[-voice.release_frames:] *= (
            1 - np.arange(1, voice.release_frames + 1) / voice.release_frames
        )
    output *= note.velocity
    if not np.isfinite(output).all():
        raise AudioContractError("String feedback loop produced nonfinite audio")
    loop_frequency = sample_rate / (length + (1 - voice.brightness) / 2)
    return output, loop_frequency, length


def _polyphony(notes: tuple[PluckNote, ...], release: int):
    events = []
    total = 0
    for note in notes:
        stop = note.start_frame + note.duration_frames + release
        total += note.duration_frames + release
        events.append((note.start_frame, +1))
        events.append((stop, -1))
    if total > MAX_TOTAL_PLUCK_SAMPLES:
        raise AudioLimitError("Plucked string arrangement exceeds work budget")
    events.sort(key=lambda item: (item[0], item[1]))
    active = peak = 0
    for _, direction in events:
        active += direction
        peak = max(peak, active)
        if peak > MAX_PLUCK_POLYPHONY:
            raise AudioLimitError("Too many simultaneously sounding string voices")
    return peak, total


def render_plucked_strings(
    plan: PluckedStringPlan,
    notes: tuple[PluckNote, ...],
) -> PluckedStringResult:
    if not isinstance(plan, PluckedStringPlan):
        raise AudioContractError("Expected PluckedStringPlan")
    if (type(notes) is not tuple or len(notes) > MAX_PLUCK_NOTES
            or any(not isinstance(n, PluckNote) for n in notes)):
        raise AudioContractError("Expected bounded tuple of PluckNote events")
    if plan.channels == 1 and any(note.pan != 0 for note in notes):
        raise AudioContractError("Mono string output cannot pan")
    for note in notes:
        loop_length(plan, note)
        if note.start_frame + note.duration_frames + plan.voice.release_frames > plan.frames:
            raise AudioContractError("Plucked note extends past declared duration")
    maximum, work = _polyphony(notes, plan.voice.release_frames)
    estimated = (
        plan.frames * plan.channels * 16 + work * 16
        + MAX_PLUCK_POLYPHONY * 16000 * 8
    )
    if estimated > MAX_PLUCK_BYTES:
        raise AudioLimitError("Plucked string working memory exceeds budget")
    output = np.zeros((plan.channels, plan.frames), dtype=np.float64)
    events = []
    for note in notes:
        waveform, approximate_frequency, delay = render_string_voice(
            note, plan.sample_rate, plan.voice,
        )
        waveform *= plan.gain
        start, stop = note.start_frame, note.start_frame + len(waveform)
        if plan.channels == 1:
            output[0, start:stop] += waveform
        else:
            angle = (note.pan + 1) * math.pi / 4
            output[0, start:stop] += waveform * math.cos(angle)
            output[1, start:stop] += waveform * math.sin(angle)
        events.append(PluckStats(
            start, stop, note.midi_note, note.frequency_hz,
            approximate_frequency, delay, note.velocity,
        ))
    if not np.isfinite(output).all():
        raise AudioContractError("String arrangement produced nonfinite audio")
    if float(np.max(np.abs(output))) > float(np.finfo(np.float32).max):
        raise AudioLimitError("String arrangement exceeds finite PCM range")
    audio = make_buffer(
        output.astype(np.float32), plan.sample_rate,
        source_note="original damped plucked string reference",
    )
    report = PluckedStringReport(
        plan.sample_rate, plan.frames, plan.channels, len(notes), maximum,
        float(np.max(np.abs(audio.samples))),
        float(np.sqrt(np.mean(audio.samples.astype(np.float64) ** 2))),
        tuple(events),
    )
    return PluckedStringResult(audio, report)
