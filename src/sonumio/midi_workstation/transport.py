"""Deterministic symbolic-to-audio sample-clock transport and scheduling.

Tempo changes are integrated exactly as rational microseconds before the
single explicit frame-rounding step. No dependence on Python wall clocks,
thread scheduling, model inference or operating-system audio devices.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Iterator

from .contracts import (
    MAX_RENDER_FRAMES, MAX_SCORE_NOTES, MAX_TICK,
    MidiLimitError, MidiWorkstationError, integer,
)
from .score import Score, ScoreNote


@dataclass(frozen=True, slots=True)
class TransportSettings:
    sample_rate: int = 24000
    rounding: str = "nearest"
    minimum_note_frames: int = 1
    tail_frames: int = 0
    include_muted: bool = False
    allow_collision_adjustment: bool = True
    maximum_output_frames: int = MAX_RENDER_FRAMES

    def __post_init__(self):
        integer(self.sample_rate, "transport sample rate", 8000, 192000)
        if self.rounding not in ("floor", "nearest", "ceil"):
            raise MidiWorkstationError("Unknown frame-rounding policy")
        integer(self.minimum_note_frames, "minimum rendered note frames", 1, 4096)
        integer(self.tail_frames, "render tail", 0, 120 * self.sample_rate)
        integer(self.maximum_output_frames, "transport budget", 1, MAX_RENDER_FRAMES)
        if type(self.include_muted) is not bool or type(self.allow_collision_adjustment) is not bool:
            raise MidiWorkstationError("Transport policy flags must be boolean")


@dataclass(frozen=True, slots=True)
class ScheduledNote:
    track_id: str
    note_id: str
    channel: int
    program: int
    bank_msb: int
    bank_lsb: int
    pitch: int
    velocity: int
    pan: int
    onset_tick: int
    release_tick: int
    start_frame: int
    stop_frame: int

    def __post_init__(self):
        if type(self.track_id) is not str or type(self.note_id) is not str:
            raise MidiWorkstationError("Scheduled note requires stable identifiers")
        for name, value, maximum in (
            ("channel", self.channel, 15), ("program", self.program, 127),
            ("bank_msb", self.bank_msb, 127), ("bank_lsb", self.bank_lsb, 127),
            ("pitch", self.pitch, 127), ("velocity", self.velocity, 127),
            ("pan", self.pan, 127), ("onset_tick", self.onset_tick, MAX_TICK),
            ("release_tick", self.release_tick, MAX_TICK),
            ("start_frame", self.start_frame, MAX_RENDER_FRAMES),
            ("stop_frame", self.stop_frame, MAX_RENDER_FRAMES),
        ):
            integer(value, name, 0, maximum)
        if not 0 < self.velocity or not self.onset_tick < self.release_tick:
            raise MidiWorkstationError("Invalid scheduled note velocity or tick interval")
        if not self.start_frame < self.stop_frame:
            raise MidiWorkstationError("Scheduled note requires positive audio duration")

    @property
    def frames(self) -> int:
        return self.stop_frame - self.start_frame

    @property
    def velocity_fraction(self) -> Fraction:
        return Fraction(self.velocity, 127)

    @property
    def pan_signed(self) -> Fraction:
        # Center MIDI pan 64 maps exactly to zero; endpoints map to ±1.
        return (Fraction(self.pan - 64, 64) if self.pan <= 64
                else Fraction(self.pan - 64, 63))


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    frame: int
    kind: str
    track_id: str
    note_id: str
    pitch: int
    channel: int

    def __post_init__(self):
        integer(self.frame, "schedule event frame", 0, MAX_RENDER_FRAMES)
        if self.kind not in ("note-on", "note-off"):
            raise MidiWorkstationError("Unknown schedule event")
        integer(self.pitch, "event pitch", 0, 127)
        integer(self.channel, "event channel", 0, 15)


def event_order(event: ScheduledEvent) -> tuple:
    return (event.frame, 0 if event.kind == "note-off" else 1,
            event.track_id, event.note_id)


@dataclass(frozen=True, slots=True)
class Schedule:
    settings: TransportSettings
    source_sha256: str
    output_frames: int
    notes: tuple[ScheduledNote, ...]
    events: tuple[ScheduledEvent, ...]
    collision_adjustments: int
    timeline_end_frame: int
    protocol: str = "sonumio-symbolic-sample-schedule-v1"

    def __post_init__(self):
        if not isinstance(self.settings, TransportSettings):
            raise MidiWorkstationError("Schedule requires transport configuration")
        integer(self.output_frames, "schedule frames", 1, self.settings.maximum_output_frames)
        integer(self.timeline_end_frame, "timeline final frame", 0, self.output_frames)
        integer(self.collision_adjustments, "collision count", 0)
        if type(self.notes) is not tuple or len(self.notes) > MAX_SCORE_NOTES:
            raise MidiLimitError("Too many scheduled notes")
        if len(self.events) != len(self.notes) * 2:
            raise MidiWorkstationError("Every scheduled note requires on and off event")
        if list(self.events) != sorted(self.events, key=event_order):
            raise MidiWorkstationError("Transport events are not in deterministic clock order")

    @property
    def total_note_frames(self) -> int:
        return sum(note.frames for note in self.notes)

    @property
    def voices(self) -> tuple[int, ...]:
        return tuple(sorted({note.channel for note in self.notes}))

    def events_between(self, start_frame: int, stop_frame: int) -> tuple[ScheduledEvent, ...]:
        integer(start_frame, "transport frame start", 0, self.output_frames)
        integer(stop_frame, "transport frame stop", start_frame, self.output_frames)
        frames = [event.frame for event in self.events]
        a = bisect_left(frames, start_frame)
        b = bisect_left(frames, stop_frame)
        return self.events[a:b]

    def active_at(self, frame: int) -> tuple[ScheduledNote, ...]:
        integer(frame, "transport seek frame", 0, self.output_frames)
        return tuple(note for note in self.notes
                     if note.start_frame <= frame < note.stop_frame)

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "sample_rate": self.settings.sample_rate,
            "output_frames": self.output_frames,
            "timeline_end_frame": self.timeline_end_frame,
            "notes": len(self.notes),
            "events": len(self.events),
            "collision_adjustments": self.collision_adjustments,
            "channels": list(self.voices),
            "source_score_sha256": self.source_sha256,
            "model_inference": False,
            "audio_rendered": False,
            "source_mutated": False,
        }


def build_schedule(score: Score,
                   settings: TransportSettings = TransportSettings()) -> Schedule:
    if not isinstance(score, Score) or not isinstance(settings, TransportSettings):
        raise MidiWorkstationError("Expected Score and TransportSettings")
    tempo = score.tempo_map
    end_frame = tempo.ticks_to_frame(score.end, settings.sample_rate,
                                     rounding=settings.rounding)
    total_frames = end_frame + settings.tail_frames
    if not 1 <= total_frames <= settings.maximum_output_frames:
        raise MidiLimitError("Sample-clock score exceeds declared output duration")
    selected = {track.id for track in score.tracks} if settings.include_muted else set(score.active_track_ids)
    notes = []
    adjusted = 0
    for track in score.tracks:
        if track.id not in selected:
            continue
        for note in track.notes:
            first = tempo.ticks_to_frame(note.start, settings.sample_rate,
                                         rounding=settings.rounding)
            last = tempo.ticks_to_frame(note.stop, settings.sample_rate,
                                        rounding=settings.rounding)
            if last - first < settings.minimum_note_frames:
                if not settings.allow_collision_adjustment:
                    raise MidiWorkstationError("Quantized audio clock collapsed note")
                last = first + settings.minimum_note_frames
                adjusted += 1
            if last > end_frame:
                raise MidiWorkstationError("Scheduled note does not fit timeline")
            notes.append(ScheduledNote(
                track.id, note.id, track.channel, track.program,
                track.bank_msb, track.bank_lsb, note.pitch, note.velocity,
                note.pan, note.start, note.stop, first, last,
            ))
    notes.sort(key=lambda n: (n.start_frame, n.track_id, n.pitch, n.note_id))
    events = []
    for note in notes:
        events.extend((
            ScheduledEvent(note.start_frame, "note-on", note.track_id,
                           note.note_id, note.pitch, note.channel),
            ScheduledEvent(note.stop_frame, "note-off", note.track_id,
                           note.note_id, note.pitch, note.channel),
        ))
    events.sort(key=event_order)
    return Schedule(settings, score.sha256, total_frames, tuple(notes),
                    tuple(events), adjusted, end_frame)


@dataclass(frozen=True, slots=True)
class TransportCursor:
    schedule: Schedule
    frame: int = 0

    def __post_init__(self):
        if not isinstance(self.schedule, Schedule):
            raise MidiWorkstationError("Transport cursor requires Schedule")
        integer(self.frame, "cursor frame", 0, self.schedule.output_frames)

    @property
    def at_end(self) -> bool:
        return self.frame == self.schedule.output_frames

    def seek(self, frame: int) -> "TransportCursor":
        integer(frame, "seek frame", 0, self.schedule.output_frames)
        return TransportCursor(self.schedule, frame)

    def advance(self, frames: int) -> tuple["TransportCursor", tuple[ScheduledEvent, ...]]:
        integer(frames, "cursor advance frames", 0, self.schedule.output_frames)
        destination = min(self.frame + frames, self.schedule.output_frames)
        return self.seek(destination), self.schedule.events_between(self.frame, destination)

    def remaining(self) -> int:
        return self.schedule.output_frames - self.frame
