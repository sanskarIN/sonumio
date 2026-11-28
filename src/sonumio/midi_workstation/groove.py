"""Original rhythmic groove templates, extraction and repeatable clip placement.

Templates contain small deviations from a declared pulse grid and velocity
ratios, not licensed samples, note recordings or externally loaded patterns.
Musical results are mathematical suggestions, not quality-certified music.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from statistics import median
from typing import Iterable

from .contracts import (
    MAX_SCORE_NOTES, MAX_TICK, MidiLimitError, MidiWorkstationError,
    finite, integer,
)
from .score import Score, ScoreNote, ScoreTrack, note_sort_key

MAX_GROOVE_SLOTS = 256
MAX_GROOVE_TEMPLATES = 48
MAX_CLIP_REPETITIONS = 256


@dataclass(frozen=True, slots=True)
class GrooveSlot:
    index: int
    offset_ticks: int = 0
    velocity_percent: int = 100
    length_percent: int = 100

    def __post_init__(self):
        integer(self.index, "groove step", 0, MAX_GROOVE_SLOTS - 1)
        integer(self.offset_ticks, "groove timing deviation", -1024, 1024)
        integer(self.velocity_percent, "groove velocity", 1, 300)
        integer(self.length_percent, "groove articulation", 1, 300)


@dataclass(frozen=True, slots=True)
class GrooveTemplate:
    step_ticks: int
    steps: tuple[GrooveSlot, ...]
    label: str = "Original groove"

    def __post_init__(self):
        integer(self.step_ticks, "groove subdivision", 1, MAX_TICK)
        if (type(self.steps) is not tuple or not 1 <= len(self.steps) <= MAX_GROOVE_SLOTS
                or any(not isinstance(s, GrooveSlot) for s in self.steps)):
            raise MidiWorkstationError("Groove needs bounded typed slots")
        indexes = [item.index for item in self.steps]
        if indexes != list(range(len(self.steps))):
            raise MidiWorkstationError("Groove slot indexes must be contiguous 0..N-1")
        if type(self.label) is not str or len(self.label) > 80:
            raise MidiWorkstationError("Invalid groove template label")
        # Offsets may not invert adjacent nominal pulse ordering, because
        # that would silently reorder notes in an otherwise bounded score.
        for previous, next_ in zip(self.steps, self.steps[1:]):
            if previous.offset_ticks - next_.offset_ticks >= self.step_ticks:
                raise MidiWorkstationError("Groove offsets invert neighboring pulses")

    @property
    def period_ticks(self) -> int:
        return self.step_ticks * len(self.steps)

    def slot_at(self, tick: int) -> GrooveSlot:
        integer(tick, "groove lookup tick", 0, MAX_TICK)
        return self.steps[(tick // self.step_ticks) % len(self.steps)]

    @classmethod
    def straight(cls, *, step_ticks: int = 120, slots: int = 16) -> "GrooveTemplate":
        integer(slots, "groove slot count", 1, MAX_GROOVE_SLOTS)
        return cls(step_ticks, tuple(GrooveSlot(i) for i in range(slots)))

    @classmethod
    def swing(cls, *, step_ticks: int = 120, slots: int = 16,
              swing_percent: int = 30) -> "GrooveTemplate":
        integer(swing_percent, "swing percent", 0, 75)
        delay = (step_ticks * swing_percent + 50) // 100
        return cls(step_ticks, tuple(
            GrooveSlot(i, delay if i % 2 else 0) for i in range(slots)
        ), "Mathematical alternating swing")


@dataclass(frozen=True, slots=True)
class GrooveApplySettings:
    timing_strength: int = 100
    velocity_strength: int = 100
    length_strength: int = 100
    preserve_duration: bool = False
    prevent_note_inversion: bool = True

    def __post_init__(self):
        for key in ("timing_strength", "velocity_strength", "length_strength"):
            integer(getattr(self, key), key, 0, 100)
        if type(self.preserve_duration) is not bool or type(self.prevent_note_inversion) is not bool:
            raise MidiWorkstationError("Groove option flags must be bool")


def _percent(value: int, percent: int, strength: int) -> int:
    delta = (value * (percent - 100) * strength)
    if delta < 0:
        delta = -((-delta + 5000) // 10000)
    else:
        delta = (delta + 5000) // 10000
    return value + delta


def apply_groove_note(note: ScoreNote, template: GrooveTemplate,
                      settings: GrooveApplySettings,
                      *, score_end: int) -> ScoreNote:
    if (not isinstance(note, ScoreNote) or not isinstance(template, GrooveTemplate)
            or not isinstance(settings, GrooveApplySettings)):
        raise MidiWorkstationError("Expected groove note, template and settings")
    integer(score_end, "score duration", 1, MAX_TICK)
    # The pulse index is computed on the unmodified source timeline, never
    # the output of a preceding edit, so operations are deterministic.
    slot = template.slot_at(note.start)
    # Signed offset multiplied directly; _percent is only for magnitudes.
    shift = (abs(slot.offset_ticks) * settings.timing_strength + 50) // 100 * (
        -1 if slot.offset_ticks < 0 else 1
    )
    start = max(0, min(score_end - 1, note.start + shift))
    velocity = max(1, min(127, _percent(
        note.velocity, slot.velocity_percent, settings.velocity_strength)))
    length = (note.duration if settings.preserve_duration else max(1, _percent(
        note.duration, slot.length_percent, settings.length_strength)))
    length = min(length, score_end - start)
    return replace(note, start=start, duration=length, velocity=velocity)


def apply_groove(score: Score, template: GrooveTemplate,
                 options: GrooveApplySettings = GrooveApplySettings(),
                 *, tracks: tuple[str, ...] | None = None) -> tuple[Score, dict]:
    if (not isinstance(score, Score) or not isinstance(template, GrooveTemplate)
            or not isinstance(options, GrooveApplySettings)):
        raise MidiWorkstationError("Expected Score, groove and options")
    if tracks is not None and (type(tracks) is not tuple or
                               not set(tracks) <= {t.id for t in score.tracks}):
        raise MidiWorkstationError("Invalid groove track selection")
    selected = {t.id for t in score.tracks} if tracks is None else set(tracks)
    changed = 0
    transformed = []
    for track in score.tracks:
        if track.id not in selected:
            transformed.append(track)
            continue
        notes = []
        for note in track.notes:
            adjusted = apply_groove_note(note, template, options, score_end=score.end)
            changed += int(adjusted != note)
            notes.append(adjusted)
        if options.prevent_note_inversion:
            # For each pitch, compare temporal note ordering before and after
            # applying swing; overlapping chords are not collapsed.
            original = sorted(track.notes, key=lambda n: (n.pitch, n.start, n.id))
            adjusted = sorted(notes, key=lambda n: (n.pitch, n.start, n.id))
            if [n.id for n in original] != [n.id for n in adjusted]:
                raise MidiWorkstationError("Groove timing would invert same-pitch notes")
        transformed.append(track.replace_notes(tuple(notes)))
    result = score.with_tracks(tuple(transformed))
    return result, {
        "protocol": "sonumio-symbolic-groove-apply-v1",
        "source_sha256": score.sha256, "result_sha256": result.sha256,
        "template_label": template.label, "template_steps": len(template.steps),
        "changed_notes": changed, "source_mutated": False,
        "model_inference": False,
    }


@dataclass(frozen=True, slots=True)
class GrooveObservation:
    slot: int
    count: int
    timing_offset_median: int
    velocity_median: int
    duration_median: int


def extract_groove(track: ScoreTrack, *, step_ticks: int,
                   slots: int = 16, maximum_deviation: int = 60) -> tuple[GrooveTemplate, tuple[GrooveObservation, ...]]:
    if not isinstance(track, ScoreTrack):
        raise MidiWorkstationError("Expected ScoreTrack")
    integer(step_ticks, "groove extraction grid", 1, MAX_TICK)
    integer(slots, "groove extraction slots", 1, MAX_GROOVE_SLOTS)
    integer(maximum_deviation, "groove deviation", 0, step_ticks // 2)
    offsets: dict[int, list[int]] = {index: [] for index in range(slots)}
    velocities: dict[int, list[int]] = {index: [] for index in range(slots)}
    lengths: dict[int, list[int]] = {index: [] for index in range(slots)}
    for note in track.notes:
        nominal = (note.start + step_ticks // 2) // step_ticks
        delta = note.start - nominal * step_ticks
        if abs(delta) > maximum_deviation:
            continue
        slot = nominal % slots
        offsets[slot].append(delta)
        velocities[slot].append(note.velocity)
        lengths[slot].append(note.duration)
    observations = []
    typical_velocity = int(median([n.velocity for n in track.notes])) if track.notes else 100
    typical_length = int(median([n.duration for n in track.notes])) if track.notes else step_ticks
    typical_velocity = max(1, typical_velocity)
    typical_length = max(1, typical_length)
    entries = []
    for slot in range(slots):
        count = len(offsets[slot])
        timing = int(median(offsets[slot])) if count else 0
        vel = int(median(velocities[slot])) if count else typical_velocity
        length = int(median(lengths[slot])) if count else typical_length
        entry = GrooveSlot(
            slot, timing,
            max(1, min(300, (vel * 100 + typical_velocity // 2) // typical_velocity)),
            max(1, min(300, (length * 100 + typical_length // 2) // typical_length)),
        )
        entries.append(entry)
        observations.append(GrooveObservation(slot, count, timing, vel, length))
    return GrooveTemplate(step_ticks, tuple(entries), "Observed symbolic groove"), tuple(observations)


@dataclass(frozen=True, slots=True)
class Clip:
    source_track_id: str
    start_tick: int
    stop_tick: int
    repeats: int = 1
    offset_tick: int = 0
    transpose: int = 0
    velocity_percent: int = 100
    id_suffix: str = "clip"

    def __post_init__(self):
        if type(self.source_track_id) is not str or not self.source_track_id:
            raise MidiWorkstationError("Invalid clip source track")
        integer(self.start_tick, "clip start", 0, MAX_TICK)
        integer(self.stop_tick, "clip stop", self.start_tick + 1, MAX_TICK)
        integer(self.repeats, "clip repetitions", 1, MAX_CLIP_REPETITIONS)
        integer(self.offset_tick, "clip destination", 0, MAX_TICK)
        integer(self.transpose, "clip transposition", -127, 127)
        integer(self.velocity_percent, "clip velocity percent", 1, 300)
        if type(self.id_suffix) is not str or not 1 <= len(self.id_suffix) <= 16:
            raise MidiWorkstationError("Invalid clip ID suffix")

    @property
    def length(self) -> int:
        return self.stop_tick - self.start_tick


def render_clip(score: Score, clip: Clip, *, target_track_id: str) -> tuple[Score, dict]:
    if not isinstance(score, Score) or not isinstance(clip, Clip):
        raise MidiWorkstationError("Expected Score and Clip")
    source = score.track_by_id(clip.source_track_id)
    target = score.track_by_id(target_track_id)
    selected = [n for n in source.notes if clip.start_tick <= n.start
                and n.stop <= clip.stop_tick]
    if len(selected) * clip.repeats + score.note_count > MAX_SCORE_NOTES:
        raise MidiLimitError("Clip duplication exceeds note count")
    generated = []
    for repeat in range(clip.repeats):
        for note in selected:
            position = clip.offset_tick + repeat * clip.length + note.start - clip.start_tick
            duration = note.duration
            if position + duration > score.end:
                raise MidiWorkstationError("Rendered clip extends beyond score end")
            pitch = integer(note.pitch + clip.transpose, "clip note pitch", 0, 127)
            velocity = max(1, min(127, (note.velocity * clip.velocity_percent + 50) // 100))
            generated.append(ScoreNote(
                f"{note.id}_{clip.id_suffix}{repeat}", position, duration,
                pitch, velocity, note.release_velocity, note.pan,
            ))
    replacement = target.replace_notes(target.notes + tuple(generated))
    tracks = tuple(replacement if t.id == target_track_id else t for t in score.tracks)
    result = score.with_tracks(tracks)
    return result, {
        "protocol": "sonumio-symbolic-clip-render-v1",
        "source_sha256": score.sha256, "result_sha256": result.sha256,
        "created_notes": len(generated), "clip_repetitions": clip.repeats,
        "mutated_source": False, "external_samples_read": False,
    }
