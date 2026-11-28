"""Lossless, typed Standard MIDI File event model with strict invariants.

Raw event payloads remain immutable so editors can operate without silently
interpreting controllers, SysEx, tuning messages, or bank selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterator

from .byteio import data_byte, nibble, signed_8
from .contracts import (
    MAX_EVENTS, MAX_EVENTS_PER_TRACK, MAX_LABEL, MAX_META_PAYLOAD, MAX_MIDI_BYTES,
    MAX_PPQ, MAX_TICK, MAX_TRACKS, MidiFormatError, MidiLimitError,
    MidiWorkstationError, digest, integer, raw_bytes, text_value,
)

CHANNEL_LENGTHS = {0x8: 2, 0x9: 2, 0xA: 2, 0xB: 2,
                   0xC: 1, 0xD: 1, 0xE: 2}
META_EOT = 0x2F
META_TEMPO = 0x51
META_METER = 0x58
META_KEY = 0x59
META_TRACK_NAME = 0x03
META_LYRICS = 0x05
META_MARKER = 0x06
META_TEXT = 0x01
META_COPYRIGHT = 0x02
META_INSTRUMENT_NAME = 0x04
META_CUE = 0x07


def channel_data_length(status: int) -> int:
    integer(status, "channel status", 0x80, 0xEF)
    return CHANNEL_LENGTHS[status >> 4]


@dataclass(frozen=True, slots=True)
class TimeDivision:
    """Either pulses/quarter-note or SMPTE frames and subframe ticks.

    SMPTE -29 uses 30000/1001 drop-frame timing. File metadata does not
    indicate timestamp origin or wall-clock synchronization.
    """
    ppqn: int | None = 480
    smpte_fps_code: int | None = None
    ticks_per_frame: int | None = None

    def __post_init__(self):
        if self.ppqn is not None:
            integer(self.ppqn, "PPQN", 1, MAX_PPQ)
            if self.smpte_fps_code is not None or self.ticks_per_frame is not None:
                raise MidiWorkstationError("PPQN and SMPTE divisions are exclusive")
        else:
            if self.smpte_fps_code not in (-24, -25, -29, -30):
                raise MidiWorkstationError("Unsupported SMPTE frame-code")
            integer(self.ticks_per_frame, "SMPTE ticks per frame", 1, 255)

    @property
    def smpte(self) -> bool:
        return self.ppqn is None

    @property
    def frame_rate(self) -> Fraction | None:
        if self.smpte_fps_code is None:
            return None
        if self.smpte_fps_code == -29:
            return Fraction(30000, 1001)
        return Fraction(-self.smpte_fps_code, 1)

    @property
    def ticks_per_second(self) -> Fraction | None:
        rate = self.frame_rate
        return None if rate is None else rate * self.ticks_per_frame

    def word(self) -> int:
        if self.ppqn is not None:
            return self.ppqn
        return ((self.smpte_fps_code & 255) << 8) | self.ticks_per_frame

    @classmethod
    def from_word(cls, word: int) -> "TimeDivision":
        integer(word, "MIDI division", 0, 65535)
        if word & 0x8000:
            code = signed_8(word >> 8)
            return cls(None, code, word & 255)
        if not word:
            raise MidiFormatError("Zero PPQN is invalid")
        return cls(word)


@dataclass(frozen=True, slots=True)
class MidiEvent:
    tick: int
    status: int
    data: bytes
    meta_type: int | None = None

    def __post_init__(self):
        integer(self.tick, "MIDI event tick", 0, MAX_TICK)
        integer(self.status, "MIDI event status", 0x80, 0xFF)
        raw_bytes(self.data, "MIDI event payload", MAX_META_PAYLOAD)
        if 0x80 <= self.status <= 0xEF:
            if self.meta_type is not None:
                raise MidiWorkstationError("Channel message cannot have meta type")
            if len(self.data) != channel_data_length(self.status):
                raise MidiFormatError("MIDI channel message has invalid data length")
            if any(value > 127 for value in self.data):
                raise MidiFormatError("Channel data bytes must be below 128")
        elif self.status == 0xFF:
            integer(self.meta_type, "meta event type", 0, 127)
            validate_meta(self.meta_type, self.data)
        elif self.status not in (0xF0, 0xF7):
            raise MidiFormatError("Unsupported SMF system event")
        elif self.meta_type is not None:
            raise MidiFormatError("SysEx messages cannot contain meta types")

    @classmethod
    def channel(cls, tick: int, kind: int, channel: int,
                *data: int) -> "MidiEvent":
        integer(kind, "channel message family", 0x8, 0xE)
        nibble(channel)
        return cls(tick, (kind << 4) | channel, bytes(data))

    @classmethod
    def meta(cls, tick: int, kind: int, payload: bytes) -> "MidiEvent":
        return cls(tick, 0xFF, payload, kind)

    @classmethod
    def sysex(cls, tick: int, status: int, payload: bytes) -> "MidiEvent":
        if status not in (0xF0, 0xF7):
            raise MidiFormatError("Expected SysEx status")
        return cls(tick, status, payload)

    @property
    def channel_number(self) -> int | None:
        return self.status & 0x0F if self.status <= 0xEF else None

    @property
    def kind(self) -> int | None:
        return self.status >> 4 if self.status <= 0xEF else None

    @property
    def is_note_on(self) -> bool:
        return self.kind == 0x9 and self.data[1] != 0

    @property
    def is_note_off(self) -> bool:
        return self.kind == 0x8 or self.kind == 0x9 and self.data[1] == 0

    @property
    def is_eot(self) -> bool:
        return self.status == 0xFF and self.meta_type == META_EOT

    @property
    def is_tempo(self) -> bool:
        return self.status == 0xFF and self.meta_type == META_TEMPO

    @property
    def is_meter(self) -> bool:
        return self.status == 0xFF and self.meta_type == META_METER

    @property
    def is_key(self) -> bool:
        return self.status == 0xFF and self.meta_type == META_KEY

    @property
    def is_channel(self) -> bool:
        return self.status <= 0xEF

    @property
    def is_sysex(self) -> bool:
        return self.status in (0xF0, 0xF7)

    def with_tick(self, tick: int) -> "MidiEvent":
        return MidiEvent(tick, self.status, self.data, self.meta_type)


def validate_meta(kind: int, data: bytes) -> None:
    integer(kind, "meta event type", 0, 127)
    raw_bytes(data, "meta payload", MAX_META_PAYLOAD)
    if kind == META_EOT and data:
        raise MidiFormatError("End of track meta event must be empty")
    if kind == META_TEMPO and (len(data) != 3 or not int.from_bytes(data, "big")):
        raise MidiFormatError("Tempo meta event must have positive 3-byte value")
    if kind == META_METER and (len(data) != 4 or not data[0]
                               or data[1] > 7 or not data[2] or not data[3]):
        raise MidiFormatError("Unsupported meter meta event payload")
    if kind == META_KEY:
        if len(data) != 2 or not -7 <= signed_8(data[0]) <= 7 or data[1] > 1:
            raise MidiFormatError("Invalid key signature meta event")
    if kind == 0x20 and (len(data) != 1 or data[0] > 15):
        raise MidiFormatError("Invalid MIDI channel-prefix meta event")
    if kind == 0x21 and len(data) != 1:
        raise MidiFormatError("Invalid MIDI port-prefix meta event")
    if kind == 0x54 and len(data) != 5:
        raise MidiFormatError("Invalid SMPTE offset meta event")
    if kind == 0x00 and len(data) != 2:
        raise MidiFormatError("MIDI sequence number must contain 2 bytes")


@dataclass(frozen=True, slots=True)
class MidiTrack:
    events: tuple[MidiEvent, ...]

    def __post_init__(self):
        if type(self.events) is not tuple or not 1 <= len(self.events) <= MAX_EVENTS_PER_TRACK:
            raise MidiWorkstationError("Track must have 1..45,000 MIDI events")
        if any(not isinstance(event, MidiEvent) for event in self.events):
            raise MidiWorkstationError("Track must contain typed MIDI events")
        ticks = [event.tick for event in self.events]
        if ticks != sorted(ticks):
            raise MidiFormatError("Track absolute event ticks must be sorted")
        if not self.events[-1].is_eot or sum(event.is_eot for event in self.events) != 1:
            raise MidiFormatError("Track must terminate with exactly one end marker")

    @property
    def end(self) -> int:
        return self.events[-1].tick

    @property
    def note_events(self) -> tuple[MidiEvent, ...]:
        return tuple(e for e in self.events if e.is_note_on or e.is_note_off)

    @property
    def name(self) -> str | None:
        for event in self.events:
            if event.status == 0xFF and event.meta_type == META_TRACK_NAME:
                return event.data.decode("utf-8", "replace")[:MAX_LABEL]
        return None

    @property
    def controllers(self) -> tuple[MidiEvent, ...]:
        return tuple(e for e in self.events if e.kind == 0xB)

    @property
    def programs(self) -> tuple[MidiEvent, ...]:
        return tuple(e for e in self.events if e.kind == 0xC)

    def by_channel(self, channel: int) -> tuple[MidiEvent, ...]:
        nibble(channel)
        return tuple(e for e in self.events if e.channel_number == channel)

    def replace(self, events: tuple[MidiEvent, ...]) -> "MidiTrack":
        return MidiTrack(events)

    @classmethod
    def from_events(cls, events: tuple[MidiEvent, ...], *, end: int | None = None) -> "MidiTrack":
        if type(events) is not tuple or any(not isinstance(e, MidiEvent) for e in events):
            raise MidiWorkstationError("Expected tuple of MidiEvent")
        if any(e.is_eot for e in events):
            raise MidiWorkstationError("Use MidiTrack directly when EOT is supplied")
        if end is None:
            end = max((event.tick for event in events), default=0)
        integer(end, "track end", 0, MAX_TICK)
        if any(event.tick > end for event in events):
            raise MidiWorkstationError("Track end precedes an event")
        # Honor the caller's order within equal timestamps, which is important
        # for program changes, controllers, note retriggers and sustain states.
        ordered = tuple(sorted(events, key=lambda e: e.tick))
        return cls(ordered + (MidiEvent.meta(end, META_EOT, b""),))


@dataclass(frozen=True, slots=True)
class MidiFile:
    format: int
    division: TimeDivision
    tracks: tuple[MidiTrack, ...]

    def __post_init__(self):
        integer(self.format, "MIDI file format", 0, 2)
        if not isinstance(self.division, TimeDivision):
            raise MidiWorkstationError("MIDI file requires TimeDivision")
        if type(self.tracks) is not tuple or not 1 <= len(self.tracks) <= MAX_TRACKS:
            raise MidiWorkstationError("MIDI file must contain bounded tracks")
        if any(not isinstance(track, MidiTrack) for track in self.tracks):
            raise MidiWorkstationError("MIDI file tracks must be MidiTrack")
        if self.format == 0 and len(self.tracks) != 1:
            raise MidiFormatError("Format-0 MIDI must have exactly one track")
        if sum(len(track.events) for track in self.tracks) > MAX_EVENTS:
            raise MidiLimitError("MIDI file exceeds total event budget")

    @property
    def end(self) -> int:
        return max(track.end for track in self.tracks)

    @property
    def events(self) -> tuple[MidiEvent, ...]:
        return tuple(e for track in self.tracks for e in track.events)

    @property
    def tempo_events(self) -> tuple[MidiEvent, ...]:
        return tuple(e for e in self.events if e.is_tempo)

    @property
    def meter_events(self) -> tuple[MidiEvent, ...]:
        return tuple(e for e in self.events if e.is_meter)

    @property
    def key_events(self) -> tuple[MidiEvent, ...]:
        return tuple(e for e in self.events if e.is_key)

    def track(self, index: int) -> MidiTrack:
        integer(index, "track index", 0, len(self.tracks) - 1)
        return self.tracks[index]

    def each_channel(self) -> Iterator[tuple[int, MidiEvent]]:
        for index, track in enumerate(self.tracks):
            for event in track.events:
                if event.is_channel:
                    yield index, event

    def with_tracks(self, tracks: tuple[MidiTrack, ...]) -> "MidiFile":
        return MidiFile(self.format, self.division, tracks)

    @classmethod
    def empty(cls, ppqn: int = 480) -> "MidiFile":
        return cls(0, TimeDivision(ppqn),
                   (MidiTrack.from_events((), end=0),))


@dataclass(frozen=True, slots=True)
class MidiSnapshot:
    midi: MidiFile
    sha256: str
    raw_bytes: int

    def __post_init__(self):
        if not isinstance(self.midi, MidiFile):
            raise MidiWorkstationError("Snapshot must reference parsed MIDI")
        if type(self.sha256) is not str or len(self.sha256) != 64:
            raise MidiWorkstationError("Invalid MIDI snapshot digest")
        integer(self.raw_bytes, "MIDI byte length", 14, MAX_MIDI_BYTES)
