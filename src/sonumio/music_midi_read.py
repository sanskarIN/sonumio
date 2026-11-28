"""Bounded Standard MIDI File parsing for local interchange.

Supports format 0/1 PPQ files, channel running status, meta and SysEx
events. Inspection preserves absolute tick geometry. Unsupported
performance controls are recorded rather than silently applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

from .errors import GenerationError
from .music_midi import MAX_MIDI_BYTES

PROTOCOL = "sonumio-midi-inspection-v1"
MAX_TRACKS = 32
MAX_MESSAGES = 32768
MAX_NOTES = 8192
MAX_TICK = 0x0fffffff
MAX_META_BYTES = 65536
MAX_TEXT_BYTES = 256


class _Reader:
    def __init__(self, data):
        self.data = memoryview(data)
        self.pos = 0

    @property
    def remaining(self):
        return len(self.data) - self.pos

    def take(self, count):
        if type(count) is not int or not 0 <= count <= self.remaining:
            raise GenerationError("Truncated MIDI message or chunk")
        start = self.pos
        self.pos += count
        return self.data[start:self.pos].tobytes()

    def byte(self):
        return self.take(1)[0]

    def uint(self, count):
        return int.from_bytes(self.take(count),"big")

    def vlq(self):
        result = 0
        for _ in range(4):
            value = self.byte()
            result = (result << 7) | (value & 0x7f)
            if not value & 0x80:
                return result
        raise GenerationError("MIDI variable-length integer exceeds four bytes")


@dataclass(frozen=True, slots=True)
class MidiNote:
    track: int
    channel: int
    pitch: int
    velocity: int
    start: int
    stop: int
    program: int


@dataclass(frozen=True, slots=True)
class MetaEvent:
    tick: int
    kind: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class ChannelEvent:
    tick: int
    status: int
    data: tuple[int,...]

    @property
    def channel(self):
        return self.status & 0xf

    @property
    def kind(self):
        return self.status & 0xf0


@dataclass(frozen=True, slots=True)
class MidiTrack:
    index: int
    end: int
    name: str | None
    notes: tuple[MidiNote,...]
    meta: tuple[MetaEvent,...]
    channels: tuple[ChannelEvent,...]
    sysex_count: int
    message_count: int


@dataclass(frozen=True, slots=True)
class MidiDocument:
    format: int
    ppq: int
    tracks: tuple[MidiTrack,...]
    source_sha256: str
    source_bytes: int

    @property
    def notes(self):
        return tuple(note for track in self.tracks for note in track.notes)

    @property
    def end(self):
        return max(track.end for track in self.tracks)


def _ascii_name(data):
    if len(data) > MAX_TEXT_BYTES:
        raise GenerationError("MIDI track name exceeds text budget")
    # MIDI text encodings are not standardized. Decode for display only;
    # raw bytes are retained in the meta ledger. Never interpret as markup.
    return data.decode("utf-8",errors="replace")


def _validate_meta(kind,payload):
    if kind == 0x2f and payload:
        raise GenerationError("MIDI end-of-track must have zero length")
    if kind == 0x51 and (len(payload)!=3 or int.from_bytes(payload,"big")==0):
        raise GenerationError("Invalid MIDI tempo")
    if kind == 0x58 and (len(payload)!=4 or payload[0]==0 or payload[1]>7):
        raise GenerationError("Invalid MIDI time signature")
    if kind == 0x59 and (len(payload)!=2 or not -7<=int.from_bytes(payload[:1],"big",signed=True)<=7
                         or payload[1] not in (0,1)):
        raise GenerationError("Invalid MIDI key signature")
    if kind == 0x20 and (len(payload)!=1 or payload[0]>15):
        raise GenerationError("Invalid MIDI channel prefix")
    if kind == 0x21 and len(payload)!=1:
        raise GenerationError("Invalid MIDI port meta event")


def _parse_track(data,index,message_budget,note_budget):
    reader = _Reader(data)
    tick = 0
    running = None
    active = {}
    programs = [0]*16
    notes = []
    meta = []
    channel_events = []
    name = None
    sysex = 0
    messages = 0
    ended = False
    while reader.remaining:
        messages += 1
        if messages > message_budget:
            raise GenerationError("MIDI message budget exceeded")
        tick += reader.vlq()
        if tick > MAX_TICK:
            raise GenerationError("MIDI absolute tick budget exceeded")
        first = reader.byte()
        initial = None
        if first < 0x80:
            if running is None:
                raise GenerationError("MIDI running status has no channel message")
            status = running
            initial = first
        else:
            status = first
        if status == 0xff:
            running = None
            kind = reader.byte()
            if kind > 0x7f:
                raise GenerationError("Invalid MIDI meta type")
            length = reader.vlq()
            if length > MAX_META_BYTES:
                raise GenerationError("MIDI meta event exceeds byte budget")
            payload = reader.take(length)
            _validate_meta(kind,payload)
            meta.append(MetaEvent(tick,kind,payload))
            if kind == 3 and name is None:
                name = _ascii_name(payload)
            if kind == 0x2f:
                if reader.remaining:
                    raise GenerationError("Trailing bytes after MIDI end-of-track")
                ended = True
                break
        elif status in (0xf0,0xf7):
            running = None
            length = reader.vlq()
            if length > MAX_META_BYTES:
                raise GenerationError("MIDI SysEx exceeds byte budget")
            reader.take(length)
            sysex += 1
        elif 0x80 <= status <= 0xef:
            running = status
            kind,channel = status & 0xf0,status & 0xf
            length = 1 if kind in (0xc0,0xd0) else 2
            values = ([] if initial is None else [initial])
            values.extend(reader.take(length-len(values)))
            if any(value > 127 for value in values):
                raise GenerationError("MIDI channel data contains a status byte")
            event = ChannelEvent(tick,status,tuple(values))
            channel_events.append(event)
            if kind == 0xc0:
                programs[channel] = values[0]
            if kind == 0x90 and values[1] > 0:
                key=(channel,values[0])
                if key in active:
                    raise GenerationError("Overlapping same-channel MIDI note is ambiguous")
                if len(active)+len(notes)>=note_budget:
                    raise GenerationError("MIDI note budget exceeded")
                active[key]=(tick,values[1],programs[channel])
            elif kind in (0x80,0x90):
                key=(channel,values[0])
                if key not in active:
                    raise GenerationError("MIDI note-off has no matching note-on")
                start,velocity,program=active.pop(key)
                if start>=tick:
                    raise GenerationError("MIDI note duration must be positive")
                notes.append(MidiNote(index,channel,values[0],velocity,start,tick,program))
        else:
            raise GenerationError("Unsupported system status in Standard MIDI File")
    if not ended or active:
        raise GenerationError("MIDI track missing end marker or has unclosed notes")
    notes.sort(key=lambda note:(note.start,note.channel,note.pitch,note.stop,note.velocity))
    return MidiTrack(index,tick,name,tuple(notes),tuple(meta),tuple(channel_events),sysex,messages)


def decode_midi(data):
    if type(data) is not bytes or not 14<=len(data)<=MAX_MIDI_BYTES:
        raise GenerationError("Invalid bounded MIDI byte count")
    reader = _Reader(data)
    if reader.take(4)!=b"MThd" or reader.uint(4)!=6:
        raise GenerationError("Expected canonical Standard MIDI File header")
    kind,count,division = struct.unpack(">HHH",reader.take(6))
    if kind not in (0,1) or not 1<=count<=MAX_TRACKS or (kind==0 and count!=1):
        raise GenerationError("Unsupported MIDI format or track count")
    if not 1<=division<=0x7fff:
        raise GenerationError("SMPTE MIDI timing is not supported")
    tracks=[]
    messages=MAX_MESSAGES
    notes=MAX_NOTES
    for index in range(count):
        if reader.take(4)!=b"MTrk":
            raise GenerationError("Expected MIDI track chunk")
        length=reader.uint(4)
        if not 1<=length<=reader.remaining:
            raise GenerationError("Invalid MIDI track chunk size")
        track=_parse_track(reader.take(length),index,messages,notes)
        messages-=track.message_count
        notes-=len(track.notes)
        tracks.append(track)
    if reader.remaining:
        raise GenerationError("Unexpected trailing MIDI chunks")
    return MidiDocument(kind,division,tuple(tracks),hashlib.sha256(data).hexdigest(),len(data))


def _tempo_map(document):
    """Resolve tempo events from all synchronous tracks, rejecting conflicts."""
    changes={0:500000}
    seen=set()
    for track in document.tracks:
        for meta in track.meta:
            if meta.kind!=0x51:
                continue
            tempo=int.from_bytes(meta.payload,"big")
            if meta.tick in seen and changes[meta.tick]!=tempo:
                raise GenerationError("Conflicting simultaneous MIDI tempo events")
            seen.add(meta.tick)
            changes[meta.tick]=tempo
    return tuple(sorted(changes.items()))


def duration_microseconds(document):
    if not isinstance(document,MidiDocument):
        raise GenerationError("Expected MidiDocument")
    tempo=_tempo_map(document)
    numerator=0
    cursor=0
    active=500000
    for tick,new in tempo:
        if tick>document.end:
            break
        numerator+=(tick-cursor)*active
        cursor=tick
        active=new
    numerator+=(document.end-cursor)*active
    # Round to nearest microsecond only once at the complete endpoint.
    return (2*numerator+document.ppq)//(2*document.ppq)


def inspect_midi(document):
    if not isinstance(document,MidiDocument):
        raise GenerationError("Expected MidiDocument")
    tempos=_tempo_map(document)
    track_reports=[]
    for track in document.tracks:
        controls={}
        for event in track.channels:
            if event.kind not in (0x80,0x90,0xc0):
                key=f"0x{event.kind:02x}"
                controls[key]=controls.get(key,0)+1
        signatures=[{"tick":event.tick,"numerator":event.payload[0],"denominator":2**event.payload[1]}
                    for event in track.meta if event.kind==0x58]
        track_reports.append({"index":track.index,"name":track.name,"end_tick":track.end,
            "notes":len(track.notes),"channels":sorted({event.channel for event in track.channels}),
            "pitch_min":min((note.pitch for note in track.notes),default=None),
            "pitch_max":max((note.pitch for note in track.notes),default=None),
            "programs":sorted({note.program for note in track.notes}),
            "time_signatures":signatures,"unapplied_channel_events":controls,
            "sysex_events":track.sysex_count,"messages":track.message_count})
    return {"protocol":PROTOCOL,"format":document.format,"ppq":document.ppq,
            "source_sha256":document.source_sha256,"source_bytes":document.source_bytes,
            "tracks":track_reports,"note_count":len(document.notes),"end_tick":document.end,
            "duration_microseconds":duration_microseconds(document),
            "tempo_map":[{"tick":tick,"microseconds_per_beat":value} for tick,value in tempos],
            "note_timing":"physical key-on/key-off; performance controllers not applied",
            "audio_rendered":False}


def read_midi(path,*,expected_sha256=None):
    if expected_sha256 is not None and (type(expected_sha256) is not str or len(expected_sha256)!=64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise GenerationError("Expected lowercase MIDI SHA256")
    from pathlib import Path
    with Path(path).open("rb") as stream:
        data=stream.read(MAX_MIDI_BYTES+1)
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest()!=expected_sha256:
        raise GenerationError("MIDI source SHA256 mismatch")
    return decode_midi(data)
