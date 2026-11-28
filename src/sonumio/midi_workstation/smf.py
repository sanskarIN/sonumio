"""Full bounded read/write support for Standard MIDI File types 0, 1, 2.

Running status is resolved during decoding and can be emitted canonically by
encoding. Tracks, metadata and opaque SysEx payloads are retained without
inventing semantic effects. Strict format parsing resists partial chunks,
unsupported inline system messages, trailing bytes and resource exhaustion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .byteio import Cursor, be16, chunk, encode_vlq
from .contracts import (
    MAX_EVENTS, MAX_EVENTS_PER_TRACK, MAX_META_PAYLOAD,
    MAX_MIDI_BYTES, MAX_TICK, MAX_TRACKS,
    MidiFormatError, MidiLimitError, MidiWorkstationError, MidiIntegrityError,
    digest, integer, raw_bytes, read_regular_file,
)
from .model import (
    META_EOT, MidiEvent, MidiFile, MidiSnapshot, MidiTrack,
    TimeDivision, channel_data_length,
)


def _decode_track(data: bytes, remaining: int) -> MidiTrack:
    stream = Cursor.from_bytes(data)
    tick = 0
    running_status: int | None = None
    events: list[MidiEvent] = []
    ended = False
    while stream.remaining:
        if len(events) >= remaining or len(events) >= MAX_EVENTS_PER_TRACK:
            raise MidiLimitError("Track exceeds MIDI event resource budget")
        delta = stream.vlq()
        tick += delta
        if tick > MAX_TICK:
            raise MidiLimitError("MIDI absolute timestamp exceeds bound")
        first = stream.byte()
        first_data: bytes | None = None
        if first < 0x80:
            if running_status is None:
                raise MidiFormatError("Channel running status not initialized")
            first_data = bytes((first,))
            status = running_status
        else:
            status = first
        if status == 0xFF:
            running_status = None
            kind = stream.byte()
            if kind > 127:
                raise MidiFormatError("Invalid meta event type")
            length = stream.vlq()
            if length > MAX_META_PAYLOAD:
                raise MidiLimitError("Oversized metadata payload")
            event = MidiEvent.meta(tick, kind, stream.take(length))
            events.append(event)
            if event.is_eot:
                if stream.remaining:
                    raise MidiFormatError("Data found after end-of-track marker")
                ended = True
                break
            continue
        if status in (0xF0, 0xF7):
            running_status = None
            length = stream.vlq()
            if length > MAX_META_PAYLOAD:
                raise MidiLimitError("Oversized SysEx payload")
            events.append(MidiEvent.sysex(tick, status, stream.take(length)))
            continue
        if not 0x80 <= status <= 0xEF:
            raise MidiFormatError("Unsupported system status in SMF track")
        length = channel_data_length(status)
        prefix = b"" if first_data is None else first_data
        payload = prefix + stream.take(length - len(prefix))
        events.append(MidiEvent(tick, status, payload))
        running_status = status
    if not ended:
        raise MidiFormatError("Track is missing required end-of-track marker")
    return MidiTrack(tuple(events))


def parse_smf(data: bytes) -> MidiSnapshot:
    raw_bytes(data, "Standard MIDI File", MAX_MIDI_BYTES)
    if len(data) < 14:
        raise MidiFormatError("MIDI file shorter than 14-byte header")
    stream = Cursor.from_bytes(data)
    stream.expect(b"MThd")
    length = stream.uint32()
    if length != 6:
        raise MidiFormatError("SMF header must contain exactly six data bytes")
    format_code = stream.uint16()
    count = stream.uint16()
    division = TimeDivision.from_word(stream.uint16())
    integer(format_code, "MIDI format", 0, 2)
    integer(count, "MIDI track count", 1, MAX_TRACKS)
    if format_code == 0 and count != 1:
        raise MidiFormatError("Format-0 MIDI must contain a single track")
    tracks: list[MidiTrack] = []
    remaining = MAX_EVENTS
    for index in range(count):
        stream.expect(b"MTrk")
        track_length = stream.uint32()
        if not 4 <= track_length <= stream.remaining:
            raise MidiFormatError("Invalid MIDI track chunk length")
        track = _decode_track(stream.take(track_length), remaining)
        tracks.append(track)
        remaining -= len(track.events)
    stream.exact_end()
    return MidiSnapshot(MidiFile(format_code, division, tuple(tracks)),
                        digest(data), len(data))


def _encode_track(track: MidiTrack, *, running_status: bool) -> bytes:
    output = bytearray()
    previous = 0
    active_status: int | None = None
    for event in track.events:
        delta = event.tick - previous
        if delta < 0:
            raise MidiFormatError("Track events were not monotonic")
        output.extend(encode_vlq(delta))
        previous = event.tick
        if event.is_channel:
            if not running_status or active_status != event.status:
                output.append(event.status)
            output.extend(event.data)
            active_status = event.status
        elif event.status == 0xFF:
            output.extend((0xFF, event.meta_type))
            output.extend(encode_vlq(len(event.data)))
            output.extend(event.data)
            active_status = None
        elif event.is_sysex:
            output.append(event.status)
            output.extend(encode_vlq(len(event.data)))
            output.extend(event.data)
            active_status = None
        else:
            raise MidiFormatError("Unsupported MIDI event type during export")
        if len(output) + 8 > MAX_MIDI_BYTES:
            raise MidiLimitError("MIDI track encoding exceeds byte budget")
    return chunk(b"MTrk", bytes(output))


def encode_smf(midi: MidiFile, *, running_status: bool = True) -> bytes:
    if not isinstance(midi, MidiFile):
        raise MidiWorkstationError("Expected typed MidiFile")
    if type(running_status) is not bool:
        raise MidiWorkstationError("running_status must be boolean")
    header = chunk(b"MThd", be16(midi.format) + be16(len(midi.tracks))
                   + be16(midi.division.word()))
    output = bytearray(header)
    for track in midi.tracks:
        encoded = _encode_track(track, running_status=running_status)
        if len(output) + len(encoded) > MAX_MIDI_BYTES:
            raise MidiLimitError("Encoded MIDI exceeds file size limit")
        output.extend(encoded)
    return bytes(output)


def canonicalize_smf(data: bytes, *, running_status: bool = True) -> bytes:
    return encode_smf(parse_smf(data).midi, running_status=running_status)


def read_smf(path: str | Path, *, expected_sha256: str | None = None) -> MidiSnapshot:
    snapshot = read_regular_file(path, expected_sha256=expected_sha256)
    parsed = parse_smf(snapshot.content)
    if parsed.sha256 != snapshot.sha256:
        raise MidiFormatError("File integrity changed during MIDI parsing")
    return parsed


def validate_roundtrip(data: bytes, *, running_status: bool = True) -> dict:
    before = parse_smf(data)
    recoded = encode_smf(before.midi, running_status=running_status)
    after = parse_smf(recoded)
    if before.midi != after.midi:
        raise MidiIntegrityError("MIDI decode/encode failed semantic roundtrip")
    return {
        "protocol": "sonumio-midi-roundtrip-v1",
        "source_sha256": before.sha256,
        "canonical_sha256": after.sha256,
        "source_bytes": len(data),
        "canonical_bytes": len(recoded),
        "formats_match": True,
        "track_count": len(before.midi.tracks),
        "event_count": len(before.midi.events),
        "semantic_roundtrip": True,
        "model_inference": False,
    }
