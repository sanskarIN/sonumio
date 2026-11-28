"""Deterministic Fourier-limited wavetable voice synthesis reference.

Each note builds or reuses a finite-harmonic table with harmonics strictly
below its Nyquist frequency. Exact sample-clock note and release envelopes,
bounded polyphony and stereo equal-power pan are explicit. Linear table
interpolation is not an alias-free oscillator and the result is not a
trained music generator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import AudioBuffer, AudioLayout, AudioSpec, check_rate, finite_float, make_buffer
from .errors import AudioContractError, AudioLimitError

WAVEFORMS = ("sine", "saw", "square", "triangle")
MAX_NOTES = 2048
MAX_POLYPHONY = 64
MAX_RENDER_SAMPLES = 40_000_000
MAX_TABLE_WORK = 10_000_000
MAX_WORK_BYTES = 192 * 1024 * 1024


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class VoiceEnvelope:
    attack_frames: int = 32
    decay_frames: int = 128
    sustain_level: float = .7
    release_frames: int = 512

    def __post_init__(self):
        for name, v in (
            ("attack_frames", self.attack_frames),
            ("decay_frames", self.decay_frames),
            ("release_frames", self.release_frames),
        ):
            _integer(v, name, 0, 1_000_000)
        finite_float(self.sustain_level, "sustain_level", minimum=0., maximum=1.)


@dataclass(frozen=True, slots=True)
class WavetablePlan:
    sample_rate: int = 24000
    frames: int = 24000
    channels: int = 2
    waveform: str = "saw"
    table_size: int = 2048
    max_harmonics: int = 128
    master_gain: float = .2
    envelope: VoiceEnvelope = VoiceEnvelope()

    def __post_init__(self):
        check_rate(self.sample_rate)
        _integer(self.frames, "output frames", 1, 120 * self.sample_rate)
        _integer(self.channels, "output channels", 1, 2)
        if self.waveform not in WAVEFORMS:
            raise AudioContractError("Unknown finite Fourier waveform")
        size = _integer(self.table_size, "table_size", 256, 8192)
        if size & (size - 1):
            raise AudioContractError("Wavetable size must be power of two")
        _integer(self.max_harmonics, "max_harmonics", 1, min(256, size // 2 - 1))
        finite_float(self.master_gain, "master_gain", minimum=0., maximum=1.)
        if not isinstance(self.envelope, VoiceEnvelope):
            raise AudioContractError("Expected VoiceEnvelope")
        AudioSpec(self.sample_rate, AudioLayout.for_channels(self.channels), self.frames)


@dataclass(frozen=True, slots=True)
class WavetableNote:
    start_frame: int
    gate_frames: int
    midi_note: int
    velocity: float = 1.
    pan: float = 0.
    initial_phase: float = 0.

    def __post_init__(self):
        _integer(self.start_frame, "note start", 0, 120 * 192000)
        _integer(self.gate_frames, "note gate", 1, 120 * 192000)
        _integer(self.midi_note, "MIDI note", 0, 127)
        finite_float(self.velocity, "note velocity", minimum=0., maximum=1.)
        finite_float(self.pan, "note pan", minimum=-1., maximum=1.)
        finite_float(self.initial_phase, "initial phase", minimum=0., maximum=1.)

    @property
    def frequency_hz(self):
        return 440. * 2 ** ((self.midi_note - 69) / 12)


@dataclass(frozen=True, slots=True)
class NoteRenderStats:
    start_frame: int
    stop_frame: int
    midi_note: int
    frequency_hz: float
    harmonics: int
    velocity: float


@dataclass(frozen=True, slots=True)
class WavetableReport:
    sample_rate: int
    frames: int
    channels: int
    waveform: str
    table_size: int
    notes: int
    maximum_polyphony: int
    distinct_tables: int
    stats: tuple[NoteRenderStats, ...]
    peak: float
    rms: float
    protocol: str = "sonumio-finite-fourier-wavetable-v1"

    def as_dict(self):
        out = asdict(self)
        out["model_inference"] = False
        out["harmonic_alias_free_guaranteed"] = False
        out["audio_clipped_or_normalized"] = False
        out["original_mathematical_waveforms"] = True
        return out


@dataclass(frozen=True, slots=True)
class WavetableResult:
    audio: AudioBuffer
    report: WavetableReport


def _harmonic_amplitude(waveform: str, harmonic: int) -> float:
    if waveform == "sine":
        return 1. if harmonic == 1 else 0.
    if waveform == "saw":
        return 2 / (math.pi * harmonic) * (1 if harmonic % 2 else -1)
    if not harmonic % 2:
        return 0.
    if waveform == "square":
        return 4 / (math.pi * harmonic)
    if waveform == "triangle":
        return 8 / (math.pi * math.pi * harmonic * harmonic) * (
            1 if harmonic % 4 == 1 else -1
        )
    raise AudioContractError("Unknown mathematical waveform")


def build_wavetable(
    waveform: str, table_size: int, harmonics: int,
) -> np.ndarray:
    """Immutable floating waveform with exact zero DC Fourier coefficients."""
    if waveform not in WAVEFORMS:
        raise AudioContractError("Unknown Fourier waveform")
    size = _integer(table_size, "table_size", 256, 8192)
    if size & (size - 1):
        raise AudioContractError("Wavetable must be power of two")
    count = _integer(harmonics, "harmonics", 1, min(256, size // 2 - 1))
    if size * count > MAX_TABLE_WORK:
        raise AudioLimitError("Wavetable Fourier construction exceeds limit")
    phases = (2 * np.pi / size) * np.arange(size, dtype=np.float64)
    output = np.zeros(size, dtype=np.float64)
    for harmonic in range(1, count + 1):
        gain = _harmonic_amplitude(waveform, harmonic)
        if gain:
            output += gain * np.sin(harmonic * phases)
    if not np.isfinite(output).all():
        raise AudioContractError("Fourier table contains nonfinite values")
    peak = float(np.max(np.abs(output)))
    if peak:
        output /= peak
    return np.frombuffer(output.tobytes(), dtype=np.float64)


def permitted_harmonics(plan: WavetablePlan, note: WavetableNote) -> int:
    if not isinstance(plan, WavetablePlan) or not isinstance(note, WavetableNote):
        raise AudioContractError("Expected synthesis plan and MIDI note")
    # Small margin below Nyquist accounts for linear interpolation's
    # small, nonzero sampling images. This does not eliminate all aliasing.
    maximum = math.floor((plan.sample_rate / 2 - 1) / note.frequency_hz)
    if maximum < 1:
        raise AudioContractError("Note fundamental exceeds synthesizer Nyquist")
    return 1 if plan.waveform == "sine" else min(maximum, plan.max_harmonics)


def _envelope(length: int, gate: int, options: VoiceEnvelope) -> np.ndarray:
    if type(length) is not int or length != gate + options.release_frames:
        raise AudioContractError("Invalid note envelope duration")
    t = np.arange(gate, dtype=np.float64)
    level = np.full(gate, options.sustain_level, dtype=np.float64)
    attack, decay = options.attack_frames, options.decay_frames
    if attack:
        a = t < attack
        level[a] = (t[a] + 1) / attack
    if decay:
        d = (t >= attack) & (t < attack + decay)
        level[d] = 1 - (1 - options.sustain_level) * (
            (t[d] - attack + 1) / decay
        )
    if options.release_frames:
        start = float(level[-1])
        release = start * (1 - (np.arange(1, options.release_frames + 1)
                               / options.release_frames))
        level = np.concatenate((level, release))
    if not np.isfinite(level).all():
        raise AudioContractError("Note envelope contains nonfinite levels")
    return np.clip(level, 0., 1.)


def _oscillator(
    table: np.ndarray, frames: int, frequency_hz: float,
    sample_rate: int, initial_phase: float,
) -> np.ndarray:
    size = len(table)
    n = np.arange(frames, dtype=np.float64)
    phase = (initial_phase + n * frequency_hz / sample_rate) % 1.
    position = phase * size
    left = np.floor(position).astype(np.intp)
    frac = position - left
    right = (left + 1) % size
    return table[left] * (1 - frac) + table[right] * frac


def _polyphony(notes: tuple[WavetableNote, ...], release: int) -> int:
    events = []
    for note in notes:
        events.append((note.start_frame, +1))
        events.append((note.start_frame + note.gate_frames + release, -1))
    # Note-off at a frame occurs before note-on at the same frame.
    events.sort(key=lambda x: (x[0], x[1]))
    current = maximum = 0
    for _, delta in events:
        current += delta
        maximum = max(maximum, current)
        if maximum > MAX_POLYPHONY:
            raise AudioLimitError("Wavetable arrangement exceeds polyphony bound")
    return maximum


def _preflight(plan: WavetablePlan, notes: tuple[WavetableNote, ...]):
    if not isinstance(plan, WavetablePlan):
        raise AudioContractError("Expected WavetablePlan")
    if type(notes) is not tuple or len(notes) > MAX_NOTES:
        raise AudioContractError("Expected a tuple of at most 2048 notes")
    if any(not isinstance(note, WavetableNote) for note in notes):
        raise AudioContractError("Expected WavetableNote events")
    if plan.channels == 1 and any(note.pan != 0 for note in notes):
        raise AudioContractError("Mono synthesizer cannot pan note events")
    count = _polyphony(notes, plan.envelope.release_frames)
    total = sum(note.gate_frames + plan.envelope.release_frames for note in notes)
    if total > MAX_RENDER_SAMPLES:
        raise AudioLimitError("Wavetable arrangement exceeds total note samples")
    table_sizes = set()
    for note in notes:
        stop = note.start_frame + note.gate_frames + plan.envelope.release_frames
        if stop > plan.frames:
            raise AudioContractError("Note release extends outside requested output")
        table_sizes.add(permitted_harmonics(plan, note))
    if sum(plan.table_size * h for h in table_sizes) > MAX_TABLE_WORK:
        raise AudioLimitError("Wavetable table bank exceeds Fourier work budget")
    if (plan.frames * plan.channels * 16
            + total * 40 + len(table_sizes) * plan.table_size * 8) > MAX_WORK_BYTES:
        raise AudioLimitError("Wavetable synthesis exceeds bounded work memory")
    return count, tuple(sorted(table_sizes))


def render_wavetable(
    plan: WavetablePlan,
    notes: tuple[WavetableNote, ...],
) -> WavetableResult:
    """Render original mathematical waveforms with exact sample-clock notes."""
    maximum, unique_harmonics = _preflight(plan, notes)
    cache = {
        harmonic: build_wavetable(plan.waveform, plan.table_size, harmonic)
        for harmonic in unique_harmonics
    }
    output = np.zeros((plan.channels, plan.frames), dtype=np.float64)
    stats = []
    for note in notes:
        h = permitted_harmonics(plan, note)
        frames = note.gate_frames + plan.envelope.release_frames
        oscillator = _oscillator(
            cache[h], frames, note.frequency_hz,
            plan.sample_rate, note.initial_phase,
        )
        env = _envelope(frames, note.gate_frames, plan.envelope)
        values = oscillator * env * note.velocity * plan.master_gain
        start = note.start_frame
        end = start + frames
        if plan.channels == 1:
            output[0, start:end] += values
        else:
            angle = (note.pan + 1) * math.pi / 4
            output[0, start:end] += math.cos(angle) * values
            output[1, start:end] += math.sin(angle) * values
        stats.append(NoteRenderStats(
            start, end, note.midi_note, note.frequency_hz, h, note.velocity,
        ))
    if not np.isfinite(output).all():
        raise AudioContractError("Wavetable synthesizer produced nonfinite audio")
    if float(np.max(np.abs(output))) > float(np.finfo(np.float32).max):
        raise AudioLimitError("Wavetable output exceeds finite float32 bounds")
    audio = make_buffer(output.astype(np.float32), plan.sample_rate,
                        source_note="original Fourier wavetable synthesis")
    report = WavetableReport(
        plan.sample_rate, plan.frames, plan.channels, plan.waveform,
        plan.table_size, len(notes), maximum, len(cache), tuple(stats),
        float(np.max(np.abs(audio.samples))),
        float(np.sqrt(np.mean(audio.samples.astype(np.float64) ** 2))),
    )
    return WavetableResult(audio, report)
