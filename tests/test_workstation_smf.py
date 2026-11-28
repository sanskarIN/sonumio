"""Binary MIDI boundaries, semantic round-trips and malformed stream rejection."""
import os
import random
import struct
from itertools import product

import pytest

from sonumio.midi_workstation.byteio import (
    Cursor, be16, be32, chunk, decode_vlq, encode_vlq, signed_8,
)
from sonumio.midi_workstation.contracts import (
    MAX_TICK, MidiFormatError, MidiIntegrityError, MidiLimitError,
    MidiWorkstationError,
)
from sonumio.midi_workstation.model import (
    MidiEvent, MidiFile, MidiTrack, TimeDivision,
)
from sonumio.midi_workstation.smf import (
    canonicalize_smf, encode_smf, parse_smf, read_smf, validate_roundtrip,
)


def voice_track(*, channel=0, pitch=60, length=480, repeats=1):
    events = [MidiEvent.meta(0, 3, b"Original melody"),
              MidiEvent.channel(0, 0xC, channel, 24)]
    for i in range(repeats):
        start = i * length
        events.append(MidiEvent.channel(start, 0x9, channel, pitch, 100))
        events.append(MidiEvent.channel(start + length, 0x8, channel, pitch, 0))
    return MidiTrack.from_events(tuple(events), end=length * repeats)


def example(format=0, division=TimeDivision(480)):
    t = voice_track()
    if format == 0:
        return MidiFile(0, division, (t,))
    conductor = MidiTrack.from_events((
        MidiEvent.meta(0, 0x51, b"\x07\xa1\x20"),
        MidiEvent.meta(0, 0x58, b"\x04\x02\x18\x08"),
        MidiEvent.meta(0, 6, b"Verse"),
    ), end=480)
    return MidiFile(format, division, (conductor, t))


@pytest.mark.parametrize("value", [0, 1, 2, 0x7F, 0x80, 0x81, 0x3FFF,
                                     0x4000, 0x1FFFFF, 0x200000, MAX_TICK])
def test_vlq_boundaries(value):
    encoded = encode_vlq(value)
    assert 1 <= len(encoded) <= 4
    assert decode_vlq(encoded) == value
    assert encoded[-1] < 0x80
    assert all(byte & 0x80 for byte in encoded[:-1])


@pytest.mark.parametrize("value", [-1, MAX_TICK + 1, 2.5, True, "16", None])
def test_vlq_rejects_invalid_values(value):
    with pytest.raises(MidiWorkstationError):
        encode_vlq(value)


@pytest.mark.parametrize("invalid", [b"", b"\x80", b"\x80\x80\x80\x80",
                                      b"\x81\x80\x80\x80\x00", b"\x00\x00"])
def test_vlq_decode_rejects_invalid_length(invalid):
    with pytest.raises(MidiWorkstationError):
        decode_vlq(invalid)


@pytest.mark.parametrize("word", [0x60, 0x7FFF, 0xE818, 0xE719, 0xE31E, 0xE21E])
def test_time_division_binary_roundtrip(word):
    division = TimeDivision.from_word(word)
    assert division.word() == word


def test_smpte_minus_29_uses_fractional_drop_frame_rate():
    division = TimeDivision(None, -29, 80)
    assert str(division.frame_rate) == "30000/1001"
    assert division.ticks_per_second.numerator == 2_400_000
    assert division.ticks_per_second.denominator == 1001


@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("running", [True, False])
def test_binary_roundtrip_covers_all_smf_formats(kind, running):
    source = example(kind)
    raw = encode_smf(source, running_status=running)
    decoded = parse_smf(raw)
    assert decoded.midi == source
    assert decoded.source_bytes == len(raw) if hasattr(decoded, 'source_bytes') else decoded.raw_bytes == len(raw)
    assert len(decoded.sha256) == 64
    assert encode_smf(decoded.midi, running_status=running) == raw
    receipt = validate_roundtrip(raw, running_status=running)
    assert receipt["semantic_roundtrip"] is True
    assert receipt["track_count"] == len(source.tracks)


@pytest.mark.parametrize("status,data", [
    (0x80, b"\x3c\x00"), (0x90, b"\x3c\x5a"), (0xA0, b"\x3c\x40"),
    (0xB0, b"\x40\x7f"), (0xC0, b"\x23"), (0xD0, b"\x2b"),
    (0xE0, b"\x00\x40"),
])
def test_all_channel_status_classes_roundtrip(status, data):
    event = MidiEvent(0, status, data)
    track = MidiTrack.from_events((event,), end=10)
    raw = encode_smf(MidiFile(0, TimeDivision(), (track,)))
    assert parse_smf(raw).midi.tracks[0].events[0] == event


@pytest.mark.parametrize("status", [0xF0, 0xF7])
def test_opaque_sysex_payload_is_preserved_without_executing(status):
    payload = b"\x7e\x7f\x09\x01\x00" * 5
    stream = MidiFile(0, TimeDivision(), (
        MidiTrack.from_events((MidiEvent.sysex(0, status, payload),), end=10),
    ))
    before = encode_smf(stream)
    after = parse_smf(before).midi
    assert after == stream
    assert encode_smf(after) == before


def test_running_status_works_across_multiple_equal_status_messages():
    events = tuple(MidiEvent.channel(i * 10, 0x9, 0, 60 + i, 90)
                   for i in range(10))
    midi = MidiFile(0, TimeDivision(), (MidiTrack.from_events(events, end=100),))
    compact = encode_smf(midi, running_status=True)
    verbose = encode_smf(midi, running_status=False)
    assert len(compact) < len(verbose)
    assert parse_smf(compact).midi == parse_smf(verbose).midi == midi


def test_running_status_is_reset_after_meta_and_sysex():
    # Following meta or sysex, an unprefixed data byte is prohibited.
    track = (b"\x00\x90\x3c\x7f"
             b"\x00\xff\x01\x01a"
             b"\x00\x3e\x7f"
             b"\x00\xff\x2f\x00")
    header = b"MThd" + be32(6) + be16(0) + be16(1) + be16(480)
    with pytest.raises(MidiFormatError):
        parse_smf(header + chunk(b"MTrk", track))
    track = (b"\x00\x90\x3c\x7f"
             b"\x00\xf0\x01\x7e"
             b"\x00\x3e\x7f"
             b"\x00\xff\x2f\x00")
    with pytest.raises(MidiFormatError):
        parse_smf(header + chunk(b"MTrk", track))


@pytest.mark.parametrize("invalid", [
    b"", b"MThd", b"MThd\x00\x00\x00\x06",
    b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x01\xe0",
    b"MThd\x00\x00\x00\x07\x00\x00\x00\x01\x01\xe0\x00",
])
def test_invalid_headers_fail_closed(invalid):
    with pytest.raises(MidiWorkstationError):
        parse_smf(invalid)


def test_parser_rejects_trailing_file_bytes():
    raw = encode_smf(example())
    with pytest.raises(MidiFormatError):
        parse_smf(raw + b"unapproved")


def test_parser_rejects_trailing_bytes_after_end_track():
    raw = encode_smf(example())
    # Replace final track tail with a forged extra byte and update its length.
    prefix, tail = raw[:-4], raw[-4:]
    tampered = bytearray(raw)
    track_start = tampered.index(b"MTrk")
    length = int.from_bytes(tampered[track_start + 4:track_start + 8], "big")
    tampered[track_start + 4:track_start + 8] = (length + 1).to_bytes(4, "big")
    tampered.extend(b"\x00")
    with pytest.raises(MidiFormatError):
        parse_smf(bytes(tampered))


def test_parser_rejects_chunk_length_overflow():
    raw = bytearray(encode_smf(example()))
    i = raw.index(b"MTrk") + 4
    raw[i:i + 4] = b"\xff\xff\xff\xff"
    with pytest.raises(MidiFormatError):
        parse_smf(bytes(raw))


@pytest.mark.parametrize("status", [0xF1, 0xF2, 0xF3, 0xF8, 0xFE])
def test_unexpected_system_bytes_rejected_in_track(status):
    midi = b"MThd" + be32(6) + be16(0) + be16(1) + be16(480)
    payload = bytes((0, status)) + b"\x00\xff\x2f\x00"
    with pytest.raises(MidiFormatError):
        parse_smf(midi + chunk(b"MTrk", payload))


def test_end_of_track_must_be_empty_and_last():
    with pytest.raises(MidiFormatError):
        MidiEvent.meta(0, 0x2F, b"x")
    with pytest.raises(MidiFormatError):
        MidiTrack((MidiEvent.meta(0, 0x2F, b""), MidiEvent.meta(0, 0x2F, b"")))


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x00\x00\x00", b"\xff\xff\xff\x00"])
def test_tempo_meta_requires_exactly_three_positive_bytes(payload):
    with pytest.raises(MidiFormatError):
        MidiEvent.meta(0, 0x51, payload)


def test_unknown_meta_bytes_are_preserved_without_evaluation():
    special = MidiEvent.meta(32, 0x7E, b"<script>alert(1)</script>")
    track = MidiTrack.from_events((special,), end=64)
    raw = encode_smf(MidiFile(0, TimeDivision(), (track,)))
    assert parse_smf(raw).midi.tracks[0].events[0] == special


def test_randomized_semantic_roundtrip_with_safe_budgets():
    rng = random.Random(8422)
    for _ in range(50):
        time = 0
        events = []
        for index in range(rng.randint(5, 120)):
            time += rng.randint(0, 500)
            channel = rng.randrange(0, 16)
            kind = rng.choice((0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE))
            data = tuple(rng.randint(0, 127) for _ in range(1 if kind in (0xC, 0xD) else 2))
            events.append(MidiEvent.channel(time, kind, channel, *data))
            if rng.random() < .2:
                events.append(MidiEvent.meta(time, 1, b"test"))
        source = MidiFile(0, TimeDivision(rng.randint(96, 960)),
                          (MidiTrack.from_events(tuple(events), end=time + 2),))
        decoded = parse_smf(encode_smf(source)).midi
        assert decoded == source


def test_file_integrity_and_leaf_symlink_rejection(tmp_path):
    path = tmp_path / "composition.mid"
    raw = encode_smf(example())
    path.write_bytes(raw)
    snapshot = read_smf(path, expected_sha256=__import__("hashlib").sha256(raw).hexdigest())
    assert snapshot.midi == example()
    with pytest.raises(MidiIntegrityError):
        read_smf(path, expected_sha256="0" * 64)
    alias = tmp_path / "alias.mid"
    try:
        alias.symlink_to(path)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with pytest.raises(MidiFormatError):
        read_smf(alias)
