"""Strict MIDI continuous-controller and pitch-bend lanes.

MIDI CC and pitch bends are channel state, not audio waveforms. The default
values and interpolation schemes are explicit. The module never applies
unknown controller semantics or silently invents instrument behavior.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Iterable

from .contracts import MAX_TICK, MAX_EVENTS, MidiFormatError, MidiLimitError, MidiWorkstationError, integer
from .model import MidiEvent, MidiFile

MAX_LANE_POINTS = 30000
MAX_AUTOMATION_SAMPLES = 30000


@dataclass(frozen=True, slots=True)
class ControlPoint:
    tick: int
    value: int

    def __post_init__(self):
        integer(self.tick, "controller tick", 0, MAX_TICK)
        integer(self.value, "controller value", 0, 127)


@dataclass(frozen=True, slots=True)
class ControllerLane:
    channel: int
    controller: int
    points: tuple[ControlPoint, ...]
    initial_value: int = 0

    def __post_init__(self):
        integer(self.channel, "controller channel", 0, 15)
        integer(self.controller, "controller number", 0, 127)
        integer(self.initial_value, "initial controller value", 0, 127)
        if type(self.points) is not tuple or len(self.points) > MAX_LANE_POINTS:
            raise MidiWorkstationError("Invalid bounded CC point tuple")
        if any(not isinstance(point, ControlPoint) for point in self.points):
            raise MidiWorkstationError("Controller lane contains invalid point")
        positions = [point.tick for point in self.points]
        if positions != sorted(set(positions)):
            raise MidiWorkstationError("Controller points must be unique and ascending")

    @property
    def last_tick(self) -> int:
        return self.points[-1].tick if self.points else 0

    def value_at(self, tick: int, *, mode: str = "hold") -> Fraction:
        integer(tick, "CC interpolation tick", 0, MAX_TICK)
        if mode not in ("hold", "linear"):
            raise MidiWorkstationError("Unknown controller interpolation mode")
        ticks = [point.tick for point in self.points]
        index = bisect_right(ticks, tick) - 1
        if index < 0:
            return Fraction(self.initial_value)
        left = self.points[index]
        if mode == "hold" or index + 1 == len(self.points):
            return Fraction(left.value)
        right = self.points[index + 1]
        t = Fraction(tick - left.tick, right.tick - left.tick)
        return Fraction(left.value) * (1 - t) + Fraction(right.value) * t

    def insert(self, point: ControlPoint, *, replace: bool = False) -> "ControllerLane":
        if not isinstance(point, ControlPoint) or type(replace) is not bool:
            raise MidiWorkstationError("Expected CC point and replace policy")
        values = {p.tick: p for p in self.points}
        if point.tick in values and not replace:
            raise MidiWorkstationError("CC point already exists")
        values[point.tick] = point
        return ControllerLane(self.channel, self.controller,
                              tuple(sorted(values.values(), key=lambda p: p.tick)),
                              self.initial_value)

    def remove(self, tick: int) -> "ControllerLane":
        integer(tick, "CC removal tick", 0, MAX_TICK)
        values = tuple(p for p in self.points if p.tick != tick)
        if len(values) == len(self.points):
            raise MidiWorkstationError("CC point does not exist")
        return ControllerLane(self.channel, self.controller, values, self.initial_value)

    def to_midi(self) -> tuple[MidiEvent, ...]:
        return tuple(MidiEvent.channel(p.tick, 0xB, self.channel,
                                       self.controller, p.value)
                     for p in self.points)

    def resample(self, *, start: int, stop: int, step: int,
                 mode: str = "linear") -> "ControllerLane":
        integer(start, "CC sample start", 0, MAX_TICK)
        integer(stop, "CC sample stop", start + 1, MAX_TICK)
        integer(step, "CC sample step", 1, MAX_TICK)
        count = (stop - start - 1) // step + 2
        if count > MAX_AUTOMATION_SAMPLES:
            raise MidiLimitError("CC interpolation exceeds point budget")
        positions = list(range(start, stop, step)) + [stop]
        samples = []
        for tick in positions:
            exact = self.value_at(tick, mode=mode)
            rounded = (2 * exact.numerator + exact.denominator) // (2 * exact.denominator)
            samples.append(ControlPoint(tick, max(0, min(127, rounded))))
        return ControllerLane(self.channel, self.controller,
                              tuple(samples), self.initial_value)

    def simplify(self, *, maximum_error: int = 1) -> "ControllerLane":
        """Reduce linear samples, keeping exact endpoints and error bound."""
        integer(maximum_error, "CC reduction tolerance", 0, 127)
        if len(self.points) <= 2:
            return self
        keep = {0, len(self.points) - 1}
        stack = [(0, len(self.points) - 1)]
        while stack:
            left, right = stack.pop()
            if right <= left + 1:
                continue
            a, b = self.points[left], self.points[right]
            worst = 0.
            location = -1
            for i in range(left + 1, right):
                current = self.points[i]
                predicted = Fraction(a.value) + Fraction(
                    (current.tick - a.tick) * (b.value - a.value),
                    b.tick - a.tick,
                )
                residual = abs(Fraction(current.value) - predicted)
                if residual > worst:
                    worst = residual
                    location = i
            if worst > maximum_error and location >= 0:
                keep.add(location)
                stack.extend(((left, location), (location, right)))
        result = tuple(self.points[index] for index in sorted(keep))
        return ControllerLane(self.channel, self.controller, result,
                              self.initial_value)

    def summary(self) -> dict:
        return {
            "protocol": "sonumio-midi-controller-lane-v1",
            "channel": self.channel, "controller": self.controller,
            "points": len(self.points), "end_tick": self.last_tick,
            "minimum": min((p.value for p in self.points), default=self.initial_value),
            "maximum": max((p.value for p in self.points), default=self.initial_value),
            "initial_value": self.initial_value,
            "interpolation_applied_implicitly": False,
            "model_inference": False,
        }


@dataclass(frozen=True, slots=True)
class PitchBendPoint:
    tick: int
    value: int

    def __post_init__(self):
        integer(self.tick, "pitch bend tick", 0, MAX_TICK)
        integer(self.value, "pitch bend value", 0, 16383)

    @property
    def centered(self) -> int:
        return self.value - 8192


@dataclass(frozen=True, slots=True)
class PitchBendLane:
    channel: int
    points: tuple[PitchBendPoint, ...]
    initial_value: int = 8192

    def __post_init__(self):
        integer(self.channel, "bend channel", 0, 15)
        integer(self.initial_value, "initial bend", 0, 16383)
        if type(self.points) is not tuple or len(self.points) > MAX_LANE_POINTS:
            raise MidiWorkstationError("Invalid bounded pitch bend tuple")
        if any(not isinstance(p, PitchBendPoint) for p in self.points):
            raise MidiWorkstationError("Invalid pitch bend observation")
        if [p.tick for p in self.points] != sorted(set(p.tick for p in self.points)):
            raise MidiWorkstationError("Pitch bend ticks must be increasing and unique")

    def value_at(self, tick: int) -> int:
        integer(tick, "bend lookup tick", 0, MAX_TICK)
        index = bisect_right([p.tick for p in self.points], tick) - 1
        return self.points[index].value if index >= 0 else self.initial_value

    def semitones_at(self, tick: int, *, range_semitones: int = 2,
                     range_cents: int = 0) -> Fraction:
        integer(range_semitones, "bend range semitones", 0, 96)
        integer(range_cents, "bend range cents", 0, 99)
        centered = self.value_at(tick) - 8192
        total = Fraction(range_semitones * 100 + range_cents, 100)
        denominator = 8192 if centered < 0 else 8191
        return Fraction(centered, denominator) * total

    def to_midi(self) -> tuple[MidiEvent, ...]:
        return tuple(MidiEvent.channel(point.tick, 0xE, self.channel,
                                       point.value & 127, point.value >> 7)
                     for point in self.points)

    def summary(self) -> dict:
        return {
            "protocol": "sonumio-midi-pitch-bend-lane-v1",
            "channel": self.channel, "points": len(self.points),
            "minimum": min((p.value for p in self.points), default=self.initial_value),
            "maximum": max((p.value for p in self.points), default=self.initial_value),
            "default_range_assumed": False,
            "model_inference": False,
        }


def extract_controllers(midi: MidiFile, *, track: int,
                        channel: int) -> tuple[ControllerLane, ...]:
    if not isinstance(midi, MidiFile):
        raise MidiWorkstationError("Expected MidiFile")
    integer(track, "CC source track", 0, len(midi.tracks) - 1)
    integer(channel, "CC source channel", 0, 15)
    by_number: dict[int, dict[int, int]] = {}
    for event in midi.tracks[track].events:
        if event.kind == 0xB and event.channel_number == channel:
            number, value = event.data
            # At same tick the last message wins, following MIDI channel order.
            by_number.setdefault(number, {})[event.tick] = value
    return tuple(ControllerLane(channel, number, tuple(
        ControlPoint(tick, value) for tick, value in sorted(changes.items())
    )) for number, changes in sorted(by_number.items()))


def extract_pitch_bends(midi: MidiFile, *, track: int,
                        channel: int) -> PitchBendLane:
    if not isinstance(midi, MidiFile):
        raise MidiWorkstationError("Expected MidiFile")
    integer(track, "bend source track", 0, len(midi.tracks) - 1)
    integer(channel, "bend source channel", 0, 15)
    points = {}
    for event in midi.tracks[track].events:
        if event.kind == 0xE and event.channel_number == channel:
            points[event.tick] = PitchBendPoint(
                event.tick, event.data[0] | event.data[1] << 7,
            )
    return PitchBendLane(channel, tuple(points[t] for t in sorted(points)))


@dataclass(frozen=True, slots=True)
class BendSensitivity:
    tick: int
    channel: int
    semitones: int
    cents: int

    def __post_init__(self):
        integer(self.tick, "RPN time", 0, MAX_TICK)
        integer(self.channel, "RPN channel", 0, 15)
        integer(self.semitones, "RPN semitones", 0, 96)
        integer(self.cents, "RPN cents", 0, 99)


def extract_bend_sensitivity(midi: MidiFile, *, track: int,
                             channel: int) -> tuple[BendSensitivity, ...]:
    """Only recognize RPN 0/0 pitch bend sensitivity, not arbitrary RPNs."""
    integer(track, "RPN source track", 0, len(midi.tracks) - 1)
    integer(channel, "RPN source channel", 0, 15)
    msb = lsb = 127
    semitones, cents = 2, 0
    result = [BendSensitivity(0, channel, semitones, cents)]
    for event in midi.tracks[track].events:
        if event.kind != 0xB or event.channel_number != channel:
            continue
        number, value = event.data
        if number == 101:
            msb = value
        elif number == 100:
            lsb = value
        elif number in (6, 38) and msb == 0 and lsb == 0:
            if number == 6:
                semitones = min(value, 96)
            else:
                cents = min(value, 99)
            observation = BendSensitivity(event.tick, channel, semitones, cents)
            if result[-1].tick == event.tick:
                result[-1] = observation
            else:
                result.append(observation)
    return tuple(result)


def inspect_controllers(midi: MidiFile, *, track: int, channel: int) -> dict:
    lanes = extract_controllers(midi, track=track, channel=channel)
    bends = extract_pitch_bends(midi, track=track, channel=channel)
    sensitivity = extract_bend_sensitivity(midi, track=track, channel=channel)
    return {
        "protocol": "sonumio-midi-performance-controls-v1",
        "track": track, "channel": channel,
        "controller_lanes": [lane.summary() for lane in lanes],
        "pitch_bend": bends.summary(),
        "bend_sensitivity": [asdict(item) for item in sensitivity],
        "rpn_support": "pitch-bend sensitivity only; other RPN/NRPN are not interpreted",
        "channel_controls_applied_to_audio": False,
        "model_inference": False,
    }
