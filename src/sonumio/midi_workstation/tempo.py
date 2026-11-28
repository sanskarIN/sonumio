"""Exact rational tick/time maps for tempo and SMPTE MIDI divisions.

Tempo segments accumulate integer microsecond-numerators without drift.
No audio sample-rate conversion occurs in this module. Format-2 MIDI has
independent track sequences and requires selecting the applicable track.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .contracts import (
    MAX_TEMPO_CHANGES, MAX_TICK, MAX_PPQ, MidiFormatError, MidiLimitError,
    MidiWorkstationError, integer,
)
from .model import (
    META_KEY, META_METER, META_TEMPO,
    MidiEvent, MidiFile, TimeDivision,
)

DEFAULT_TEMPO = 500_000


@dataclass(frozen=True, slots=True)
class TempoPoint:
    tick: int
    microseconds_per_quarter: int

    def __post_init__(self):
        integer(self.tick, "tempo tick", 0, MAX_TICK)
        integer(self.microseconds_per_quarter, "microseconds per quarter", 1, 0xFFFFFF)

    @property
    def bpm(self) -> Fraction:
        return Fraction(60_000_000, self.microseconds_per_quarter)


@dataclass(frozen=True, slots=True)
class TempoSegment:
    start_tick: int
    start_microseconds: Fraction
    microseconds_per_quarter: int

    def __post_init__(self):
        integer(self.start_tick, "tempo segment tick", 0, MAX_TICK)
        if type(self.start_microseconds) is not Fraction or self.start_microseconds < 0:
            raise MidiWorkstationError("Invalid tempo segment start duration")
        integer(self.microseconds_per_quarter, "tempo", 1, 0xFFFFFF)


@dataclass(frozen=True, slots=True)
class TempoMap:
    division: TimeDivision
    points: tuple[TempoPoint, ...]

    def __post_init__(self):
        if not isinstance(self.division, TimeDivision):
            raise MidiWorkstationError("Tempo map needs a division")
        if (type(self.points) is not tuple
                or not 1 <= len(self.points) <= MAX_TEMPO_CHANGES
                or any(not isinstance(point, TempoPoint) for point in self.points)):
            raise MidiWorkstationError("Invalid bounded tempo point tuple")
        if self.points[0].tick != 0:
            raise MidiFormatError("Tempo map must begin at tick zero")
        ticks = [point.tick for point in self.points]
        if ticks != sorted(set(ticks)):
            raise MidiFormatError("Tempo map points must be strictly increasing")
        if self.division.smpte and self.points != (TempoPoint(0, DEFAULT_TEMPO),):
            # SMPTE clock is independent of tempo: allow MIDI tempo meta in
            # document but disallow replacing this map's wall-clock speed.
            raise MidiFormatError("SMPTE timing may not use tempo acceleration")

    @classmethod
    def constant(cls, ppqn: int = 480, tempo: int = DEFAULT_TEMPO) -> "TempoMap":
        return cls(TimeDivision(ppqn), (TempoPoint(0, tempo),))

    @classmethod
    def from_midi(cls, midi: MidiFile, *, track: int | None = None) -> "TempoMap":
        if not isinstance(midi, MidiFile):
            raise MidiWorkstationError("Expected MidiFile")
        if midi.format == 2 and track is None:
            raise MidiWorkstationError("Format-2 MIDI requires explicit tempo track")
        if track is not None:
            integer(track, "tempo track", 0, len(midi.tracks) - 1)
        relevant = (midi.tracks[track],) if track is not None else midi.tracks
        changes: dict[int, int] = {}
        for selected in relevant:
            for event in selected.events:
                if not event.is_tempo:
                    continue
                value = int.from_bytes(event.data, "big")
                previous = changes.get(event.tick)
                if previous is not None and previous != value:
                    raise MidiFormatError("Conflicting MIDI tempo events at identical tick")
                changes[event.tick] = value
        if midi.division.smpte:
            return cls(midi.division, (TempoPoint(0, DEFAULT_TEMPO),))
        changes.setdefault(0, DEFAULT_TEMPO)
        points = tuple(TempoPoint(tick, value) for tick, value in sorted(changes.items()))
        return cls(midi.division, points)

    @property
    def segments(self) -> tuple[TempoSegment, ...]:
        if self.division.smpte:
            return (TempoSegment(0, Fraction(0), DEFAULT_TEMPO),)
        previous_tick = 0
        elapsed = Fraction(0)
        result: list[TempoSegment] = []
        prior = self.points[0].microseconds_per_quarter
        for point in self.points:
            elapsed += Fraction((point.tick - previous_tick) * prior, self.division.ppqn)
            result.append(TempoSegment(point.tick, elapsed, point.microseconds_per_quarter))
            previous_tick, prior = point.tick, point.microseconds_per_quarter
        return tuple(result)

    def tick_to_microseconds(self, tick: int) -> Fraction:
        integer(tick, "MIDI time tick", 0, MAX_TICK)
        if self.division.smpte:
            return Fraction(1_000_000 * tick, 1) / self.division.ticks_per_second
        starts = [point.tick for point in self.points]
        selected = bisect_right(starts, tick) - 1
        point = self.points[selected]
        segment = self.segments[selected]
        return (segment.start_microseconds
                + Fraction((tick - point.tick) * point.microseconds_per_quarter,
                           self.division.ppqn))

    def microseconds_to_tick(self, us: Fraction, *, rounding: str = "floor") -> int:
        if type(us) is not Fraction or us < 0:
            raise MidiWorkstationError("MIDI microseconds must be nonnegative Fraction")
        if rounding not in ("floor", "nearest", "ceil"):
            raise MidiWorkstationError("Unknown tick rounding policy")
        if self.division.smpte:
            exact = us * self.division.ticks_per_second / 1_000_000
        else:
            segments = self.segments
            elapsed = [segment.start_microseconds for segment in segments]
            index = bisect_right(elapsed, us) - 1
            selected = segments[max(0, index)]
            exact = (selected.start_tick + (us - selected.start_microseconds)
                     * self.division.ppqn / selected.microseconds_per_quarter)
        if rounding == "floor":
            answer = exact.numerator // exact.denominator
        elif rounding == "ceil":
            answer = -(-exact.numerator // exact.denominator)
        else:
            answer = (2 * exact.numerator + exact.denominator) // (2 * exact.denominator)
        return integer(answer, "converted tick", 0, MAX_TICK)

    def tick_to_seconds(self, tick: int) -> Fraction:
        return self.tick_to_microseconds(tick) / 1_000_000

    def ticks_to_frame(self, tick: int, sample_rate: int,
                       *, rounding: str = "nearest") -> int:
        integer(sample_rate, "audio sample rate", 8000, 192000)
        us = self.tick_to_microseconds(tick)
        return self._round(us * sample_rate / 1_000_000, rounding)

    def frame_to_tick(self, frame: int, sample_rate: int,
                      *, rounding: str = "nearest") -> int:
        integer(frame, "audio frame", 0)
        integer(sample_rate, "audio sample rate", 8000, 192000)
        return self.microseconds_to_tick(Fraction(frame * 1_000_000, sample_rate),
                                         rounding=rounding)

    @staticmethod
    def _round(value: Fraction, policy: str) -> int:
        if type(value) is not Fraction or value < 0:
            raise MidiWorkstationError("Expected positive rational time")
        if policy == "floor":
            return value.numerator // value.denominator
        if policy == "ceil":
            return -(-value.numerator // value.denominator)
        if policy == "nearest":
            return (2 * value.numerator + value.denominator) // (2 * value.denominator)
        raise MidiWorkstationError("Unknown rounding policy")

    def duration(self, start: int, stop: int) -> Fraction:
        integer(start, "duration start tick", 0, MAX_TICK)
        integer(stop, "duration stop tick", start, MAX_TICK)
        return self.tick_to_microseconds(stop) - self.tick_to_microseconds(start)

    def tempo_at(self, tick: int) -> TempoPoint:
        integer(tick, "tempo lookup tick", 0, MAX_TICK)
        index = bisect_right([p.tick for p in self.points], tick) - 1
        return self.points[index]

    def insert(self, tick: int, microseconds_per_quarter: int,
               *, replace: bool = False) -> "TempoMap":
        if self.division.smpte:
            raise MidiWorkstationError("Cannot edit tempo speed on SMPTE clock")
        change = TempoPoint(tick, microseconds_per_quarter)
        points = {point.tick: point for point in self.points}
        if tick in points and not replace:
            raise MidiWorkstationError("Tempo point already exists at tick")
        points[tick] = change
        return TempoMap(self.division, tuple(sorted(points.values(), key=lambda p: p.tick)))

    def remove(self, tick: int) -> "TempoMap":
        integer(tick, "tempo removal tick", 1, MAX_TICK)
        selected = tuple(point for point in self.points if point.tick != tick)
        if len(selected) == len(self.points):
            raise MidiWorkstationError("Tempo point does not exist")
        return TempoMap(self.division, selected)

    def to_midi_events(self) -> tuple[MidiEvent, ...]:
        if self.division.smpte:
            return ()
        return tuple(MidiEvent.meta(p.tick, META_TEMPO,
                                    p.microseconds_per_quarter.to_bytes(3, "big"))
                     for p in self.points)


@dataclass(frozen=True, slots=True)
class MeterPoint:
    tick: int
    numerator: int
    denominator: int
    clocks_per_metronome: int = 24
    thirtyseconds_per_quarter: int = 8

    def __post_init__(self):
        integer(self.tick, "meter tick", 0, MAX_TICK)
        integer(self.numerator, "meter numerator", 1, 255)
        if type(self.denominator) is not int or self.denominator not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise MidiWorkstationError("Meter denominator must be power of two <= 128")
        integer(self.clocks_per_metronome, "metronome clocks", 1, 255)
        integer(self.thirtyseconds_per_quarter, "32nds per quarter", 1, 255)

    @property
    def beat_length_quarters(self) -> Fraction:
        return Fraction(4, self.denominator)

    def measure_ticks(self, ppqn: int) -> Fraction:
        integer(ppqn, "PPQN", 1, MAX_PPQ)
        return Fraction(self.numerator * ppqn * 4, self.denominator)

    def to_midi(self) -> MidiEvent:
        return MidiEvent.meta(self.tick, META_METER,
                              bytes((self.numerator, self.denominator.bit_length() - 1,
                                     self.clocks_per_metronome,
                                     self.thirtyseconds_per_quarter)))


@dataclass(frozen=True, slots=True)
class MusicalPosition:
    """1-based bar number plus zero-based rational beat location."""
    bar: int
    beat: Fraction
    meter: MeterPoint

    def __post_init__(self):
        integer(self.bar, "bar", 1)
        if type(self.beat) is not Fraction or self.beat < 0:
            raise MidiWorkstationError("Invalid beat within bar")
        if not isinstance(self.meter, MeterPoint) or self.beat >= self.meter.numerator:
            raise MidiWorkstationError("Beat exceeds declared meter")


@dataclass(frozen=True, slots=True)
class MeterMap:
    division: TimeDivision
    points: tuple[MeterPoint, ...]

    def __post_init__(self):
        if not isinstance(self.division, TimeDivision) or self.division.smpte:
            raise MidiWorkstationError("Meter coordinates require PPQN")
        if type(self.points) is not tuple or not self.points or self.points[0].tick != 0:
            raise MidiWorkstationError("Meter timeline must start at tick zero")
        starts = [p.tick for p in self.points]
        if starts != sorted(set(starts)):
            raise MidiWorkstationError("Meter timeline must have unique increasing starts")
        if len(self.points) > MAX_TEMPO_CHANGES:
            raise MidiLimitError("Meter timeline exceeds bounded events")

    @classmethod
    def from_midi(cls, midi: MidiFile, *, track: int | None = None) -> "MeterMap":
        if not isinstance(midi, MidiFile) or midi.division.smpte:
            raise MidiWorkstationError("Meter extraction requires PPQN MIDI")
        if midi.format == 2 and track is None:
            raise MidiWorkstationError("Format-2 meter requires explicit sequence")
        if track is not None:
            integer(track, "meter track", 0, len(midi.tracks) - 1)
        selected = midi.tracks if track is None else (midi.tracks[track],)
        found: dict[int, MeterPoint] = {}
        for seq in selected:
            for event in seq.events:
                if not event.is_meter:
                    continue
                d = event.data
                value = MeterPoint(event.tick, d[0], 1 << d[1], d[2], d[3])
                if event.tick in found and found[event.tick] != value:
                    raise MidiFormatError("Conflicting simultaneous time-signatures")
                found[event.tick] = value
        found.setdefault(0, MeterPoint(0, 4, 4))
        return cls(midi.division, tuple(sorted(found.values(), key=lambda x: x.tick)))

    def meter_at(self, tick: int) -> MeterPoint:
        integer(tick, "meter tick", 0, MAX_TICK)
        position = bisect_right([p.tick for p in self.points], tick) - 1
        return self.points[position]

    def position(self, tick: int) -> MusicalPosition:
        integer(tick, "meter position tick", 0, MAX_TICK)
        starts = [point.tick for point in self.points]
        index = bisect_right(starts, tick) - 1
        bar = 1
        for i in range(index):
            point = self.points[i]
            next_tick = self.points[i + 1].tick
            fractional_measures = Fraction(next_tick - point.tick, 1) / point.measure_ticks(self.division.ppqn)
            # A meter change establishes a fresh bar even mid-bar.
            bar += (fractional_measures.numerator + fractional_measures.denominator - 1) // fractional_measures.denominator
        point = self.points[index]
        elapsed = Fraction(tick - point.tick, 1)
        span = point.measure_ticks(self.division.ppqn)
        passed = elapsed // span
        within = elapsed - passed * span
        beat = within / Fraction(self.division.ppqn * 4, point.denominator)
        return MusicalPosition(bar + passed, beat, point)

    def to_midi_events(self) -> tuple[MidiEvent, ...]:
        return tuple(point.to_midi() for point in self.points)


def tempo_summary(midi: MidiFile, *, track: int | None = None) -> dict:
    tempo = TempoMap.from_midi(midi, track=track)
    return {
        "protocol": "sonumio-midi-tempo-v1",
        "division": {"ppqn": midi.division.ppqn,
                     "smpte_fps_code": midi.division.smpte_fps_code,
                     "ticks_per_frame": midi.division.ticks_per_frame},
        "points": [{"tick": p.tick, "microseconds_per_quarter": p.microseconds_per_quarter,
                    "bpm": float(p.bpm)} for p in tempo.points],
        "duration_microseconds": int(tempo.tick_to_microseconds(midi.end)),
        "clock_drift_compensation": False,
    }
