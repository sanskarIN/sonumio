"""Original, explicitly parameterized rhythmic-pattern composition engine.

This generates no copyrighted melody: notes are entirely supplied as symbolic
pattern cells by the caller. Repetition, accents, steps, density variation and
track routing are bounded deterministic operations. An event's random chance
is hash-derived from its own stable identity, not mutable global RNG state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import (
    MAX_SCORE_NOTES, MAX_SCORE_TRACKS, MAX_TICK, MidiLimitError,
    MidiWorkstationError, boolean, integer, text_value,
)
from .score import Score, ScoreNote, ScoreTrack, note_sort_key
from .tempo import MeterPoint, TempoPoint

MAX_PATTERN_BARS = 64
MAX_PATTERN_CELLS = 1024
MAX_PATTERN_STEPS = 512
MAX_PATTERN_PASSES = 256
MAX_PATTERN_SEED = 2**64 - 1


@dataclass(frozen=True, slots=True)
class PatternCell:
    step: int
    pitch: int
    duration_steps: int = 1
    velocity: int = 100
    probability_percent: int = 100
    accent_percent: int = 100
    microshift_ticks: int = 0
    pan: int = 64
    label: str = ""

    def __post_init__(self):
        integer(self.step, "pattern step", 0, MAX_PATTERN_STEPS - 1)
        integer(self.pitch, "pattern pitch", 0, 127)
        integer(self.duration_steps, "note step duration", 1, MAX_PATTERN_STEPS)
        integer(self.velocity, "pattern velocity", 1, 127)
        integer(self.probability_percent, "note probability", 0, 100)
        integer(self.accent_percent, "accent percentage", 1, 200)
        integer(self.microshift_ticks, "microtiming", -2048, 2048)
        integer(self.pan, "note pan", 0, 127)
        text_value(self.label, "pattern label", 64)


@dataclass(frozen=True, slots=True)
class PatternVoice:
    id: str
    name: str
    channel: int
    program: int
    cells: tuple[PatternCell, ...]
    repeats_every_bars: int = 1
    rotate_steps_per_bar: int = 0
    bank_msb: int = 0
    bank_lsb: int = 0
    muted: bool = False
    solo: bool = False

    def __post_init__(self):
        from .score import NOTE_ID_RE
        if type(self.id) is not str or not NOTE_ID_RE.fullmatch(self.id):
            raise MidiWorkstationError("Pattern voice needs a stable ASCII ID")
        text_value(self.name, "voice name", 256)
        integer(self.channel, "voice MIDI channel", 0, 15)
        integer(self.program, "voice MIDI program", 0, 127)
        integer(self.bank_msb, "voice bank MSB", 0, 127)
        integer(self.bank_lsb, "voice bank LSB", 0, 127)
        integer(self.repeats_every_bars, "voice repeat bars", 1, 16)
        integer(self.rotate_steps_per_bar, "voice rotation", -MAX_PATTERN_STEPS,
                MAX_PATTERN_STEPS)
        boolean(self.muted, "voice muted")
        boolean(self.solo, "voice solo")
        if (type(self.cells) is not tuple or len(self.cells) > MAX_PATTERN_CELLS
                or any(not isinstance(cell, PatternCell) for cell in self.cells)):
            raise MidiWorkstationError("Pattern cells must form a bounded typed tuple")
        # Cells at one step can form chords, but duplicate step/pitch would
        # otherwise create an ambiguous repeated note on one MIDI channel.
        identities = [(cell.step, cell.pitch) for cell in self.cells]
        if len(identities) != len(set(identities)):
            raise MidiWorkstationError("Duplicate step/pitch in one pattern voice")
        if identities != sorted(identities):
            raise MidiWorkstationError("Pattern cells must be sorted by step and pitch")


@dataclass(frozen=True, slots=True)
class PatternPlan:
    ppqn: int = 480
    bpm: int = 120
    bars: int = 8
    numerator: int = 4
    denominator: int = 4
    steps_per_quarter: int = 4
    seed: int = 0
    title: str = "Original pattern composition"
    force_first_pass: bool = False

    def __post_init__(self):
        integer(self.ppqn, "pattern PPQN", 24, 32767)
        integer(self.bpm, "pattern BPM", 30, 300)
        integer(self.bars, "pattern bars", 1, MAX_PATTERN_BARS)
        integer(self.numerator, "pattern meter numerator", 1, 12)
        if type(self.denominator) is not int or self.denominator not in (2, 4, 8, 16):
            raise MidiWorkstationError("Pattern denominator must be 2,4,8,16")
        integer(self.steps_per_quarter, "steps per quarter", 1, 32)
        integer(self.seed, "pattern seed", 0, MAX_PATTERN_SEED)
        text_value(self.title, "pattern title", 256)
        boolean(self.force_first_pass, "first pass guarantee")
        if (self.ppqn % self.steps_per_quarter
                or (self.ppqn * self.numerator * 4) % self.denominator
                or (self.numerator * self.steps_per_quarter * 4) % self.denominator):
            raise MidiWorkstationError("Meter subdivisions must map to integer ticks")
        if self.end_ticks > MAX_TICK:
            raise MidiLimitError("Pattern arrangement exceeds MIDI tick bound")
        if self.steps_per_bar * self.bars > MAX_PATTERN_PASSES * MAX_PATTERN_STEPS:
            raise MidiLimitError("Pattern repetition exceeds timeline budget")

    @property
    def steps_per_bar(self):
        return self.numerator * self.steps_per_quarter * 4 // self.denominator

    @property
    def ticks_per_step(self):
        return self.ppqn // self.steps_per_quarter

    @property
    def ticks_per_bar(self):
        return self.steps_per_bar * self.ticks_per_step

    @property
    def end_ticks(self):
        return self.bars * self.ticks_per_bar


@dataclass(frozen=True, slots=True)
class PatternReport:
    source_cells: int
    score_notes: int
    excluded_by_probability: int
    clipped_notes: int
    tracks: int
    bars: int
    ppqn: int
    seed: int
    score_digest: str
    protocol: str = "sonumio-original-pattern-render-v1"

    def as_dict(self):
        return {**asdict(self), "trained_model_used": False,
                "source_recordings_used": False,
                "source_ownership_verified": False,
                "deterministic": True}


def _hash_percent(seed: int, voice_id: str, bar: int, cell: int):
    token = f"{seed}/{voice_id}/{bar}/{cell}".encode("utf-8")
    # Use rejection-free division of 256-bit space into a 0..99 bucket.
    return int.from_bytes(sha256(token).digest(), "big") % 100


def _accent(value: int, percentage: int):
    return min(127, max(1, (value * percentage + 50) // 100))


def _check_plan(plan, voices):
    if not isinstance(plan, PatternPlan):
        raise MidiWorkstationError("Expected PatternPlan")
    if (type(voices) is not tuple or not 1 <= len(voices) <= MAX_SCORE_TRACKS
            or any(not isinstance(v, PatternVoice) for v in voices)):
        raise MidiWorkstationError("Expected 1..48 original pattern voices")
    identities = [v.id for v in voices]
    if len(identities) != len(set(identities)):
        raise MidiWorkstationError("Duplicate pattern voice ID")
    programs = {}
    for voice in voices:
        patch = (voice.program, voice.bank_msb, voice.bank_lsb)
        previous = programs.setdefault(voice.channel, patch)
        if previous != patch:
            raise MidiWorkstationError("Shared MIDI channel has incompatible patches")
        if voice.repeats_every_bars * plan.steps_per_bar > MAX_PATTERN_STEPS:
            raise MidiWorkstationError("Pattern repeat length exceeds maximum steps")
        for cell in voice.cells:
            if cell.step >= voice.repeats_every_bars * plan.steps_per_bar:
                raise MidiWorkstationError("Pattern cell exceeds its repeat cycle")
    worst = sum(
        len(voice.cells) * ((plan.bars + voice.repeats_every_bars - 1)
                           // voice.repeats_every_bars)
        for voice in voices
    )
    if worst > MAX_SCORE_NOTES:
        raise MidiLimitError("Pattern render would exceed symbolic note budget")
    return worst


def render_patterns(plan: PatternPlan, voices: tuple[PatternVoice, ...]):
    """Compile independently constructed symbolic note patterns into Score.

    Each cell ID and repeat index establishes stable deterministic output.
    The caller controls every pitch; the engine only handles score structure.
    """
    _check_plan(plan, voices)
    excluded = clipped = count = 0
    tracks = []
    for voice in voices:
        notes = []
        cycle_steps = voice.repeats_every_bars * plan.steps_per_bar
        for repeat_bar in range(0, plan.bars, voice.repeats_every_bars):
            repeat_index = repeat_bar // voice.repeats_every_bars
            for cell_index, cell in enumerate(voice.cells):
                if (cell.probability_percent == 0
                        or (cell.probability_percent < 100
                            and not (plan.force_first_pass and repeat_index == 0)
                            and _hash_percent(plan.seed, voice.id, repeat_index,
                                              cell_index) >= cell.probability_percent)):
                    excluded += 1
                    continue
                rotated = (cell.step + repeat_index * voice.rotate_steps_per_bar) % cycle_steps
                start = repeat_bar * plan.ticks_per_bar + rotated * plan.ticks_per_step
                start = max(0, min(plan.end_ticks - 1, start + cell.microshift_ticks))
                duration = cell.duration_steps * plan.ticks_per_step
                if duration > plan.end_ticks - start:
                    clipped += 1
                    duration = plan.end_ticks - start
                note_id = f"{voice.id}-{repeat_index:03d}-{cell_index:04d}"
                notes.append(ScoreNote(
                    note_id, start, duration, cell.pitch,
                    _accent(cell.velocity, cell.accent_percent), 0, cell.pan,
                ))
                count += 1
        tracks.append(ScoreTrack(
            voice.id, voice.name, voice.channel, voice.program,
            voice.bank_msb, voice.bank_lsb,
            tuple(sorted(notes, key=note_sort_key)), voice.muted, voice.solo,
        ))
    tempo = max(1, round(60_000_000 / plan.bpm))
    score = Score(plan.title, plan.ppqn, tuple(tracks),
                  (TempoPoint(0, tempo),),
                  (MeterPoint(0, plan.numerator, plan.denominator),),
                  plan.end_ticks)
    report = PatternReport(sum(len(v.cells) for v in voices), count,
                           excluded, clipped, len(voices), plan.bars,
                           plan.ppqn, plan.seed, score.sha256)
    return score, report


def pulse_pattern(pitches: tuple[int, ...], *, every_steps: int = 4,
                  total_steps: int = 16, velocity: int = 92,
                  duration_steps: int = 2) -> tuple[PatternCell, ...]:
    """Helper for original scale/arpeggio experiments with explicit notes."""
    if type(pitches) is not tuple or not 1 <= len(pitches) <= 32:
        raise MidiWorkstationError("Pulse helper requires 1..32 explicit pitches")
    for pitch in pitches:
        integer(pitch, "pulse MIDI pitch", 0, 127)
    integer(every_steps, "pulse step interval", 1, 128)
    integer(total_steps, "pulse length", 1, MAX_PATTERN_STEPS)
    integer(velocity, "pulse velocity", 1, 127)
    integer(duration_steps, "pulse note length", 1, MAX_PATTERN_STEPS)
    result = [PatternCell(step, pitches[index % len(pitches)],
                          duration_steps, velocity)
              for index, step in enumerate(range(0, total_steps, every_steps))]
    return tuple(result)
