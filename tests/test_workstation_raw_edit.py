"""MIDI event surgery: preserve arbitrary source metadata with safe typed edits."""
import random

import pytest

from sonumio.midi_workstation.contracts import (
    MidiIntegrityError, MidiWorkstationError,
)
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision
from sonumio.midi_workstation.raw_edit import MidiTransformSettings, transform_midi
from sonumio.midi_workstation.smf import encode_smf, parse_smf
from sonumio.midi_workstation.contracts import digest


def note(tick, channel, pitch, velocity=90):
    return MidiEvent.channel(tick, 0x9, channel, pitch, velocity)


def off(tick, channel, pitch, velocity=45):
    return MidiEvent.channel(tick, 0x8, channel, pitch, velocity)


def song(*, drums=False, meta=True):
    events = []
    if meta:
        events.extend((
            MidiEvent.meta(0, 0x03, b"Custom original test"),
            MidiEvent.meta(0, 0x59, b"\x01\x00"),
            MidiEvent.meta(0, 0x51, b"\x07\xa1\x20"),
            MidiEvent.sysex(0, 0xF0, b"\x7e\x7f\x09\x01"),
            MidiEvent.meta(0, 0x7D, b"unknown, preserved"),
        ))
    events.extend((
        MidiEvent.channel(0, 0xC, 0, 41),
        MidiEvent.channel(0, 0xB, 0, 64, 127),
        note(10, 0, 60, 80),
        MidiEvent.channel(12, 0xA, 0, 60, 75),
        off(80, 0, 60, 17),
        MidiEvent.channel(95, 0xB, 0, 64, 0),
        note(100, 0, 64, 111),
        off(180, 0, 64, 0),
    ))
    if drums:
        events += [note(200, 9, 36, 100), off(230, 9, 36, 50)]
    return MidiFile(0, TimeDivision(), (MidiTrack.from_events(tuple(events), end=480),))


def edit(midi=None, **kwargs):
    return transform_midi(song() if midi is None else midi, MidiTransformSettings(**kwargs))


def test_identity_is_semantically_exact_and_preserves_all_events():
    source = song(drums=True)
    out, report = edit(source)
    assert out == source
    assert encode_smf(out) == encode_smf(source)
    assert report.result_sha256 == report.source_sha256
    assert report.changed_pitch_events == 0
    assert report.changed_velocity_events == 0
    assert report.untouched_percussion_events == 2
    assert report.as_dict()["opaque_meta_sysex_preserved"] is True


def test_transposes_on_off_and_polyphonic_pressure_together():
    original = song(drums=True)
    out, report = edit(original, semitones=5)
    played = [(e.kind, e.data[0]) for e in out.events if e.kind in (8, 9, 10)]
    assert played == [(9, 65), (10, 65), (8, 65), (9, 69), (8, 69),
                      (9, 36), (8, 36)]
    assert report.changed_pitch_events == 5
    assert report.untouched_percussion_events == 2
    assert original.events[7].data[0] == 60


@pytest.mark.parametrize("semitones", [-48, -12, -1, 0, 1, 12, 48])
def test_pitch_bounds_and_atomic_rejection(semitones):
    original = song()
    before = encode_smf(original)
    if any(not 0 <= x + semitones <= 127 for x in (60, 64)):
        with pytest.raises(MidiWorkstationError, match="outside"):
            edit(original, semitones=semitones)
    else:
        transformed, report = edit(original, semitones=semitones)
        assert transformed.tracks[0].events[7].data[0] == 60 + semitones
        assert report.changed_pitch_events == (0 if semitones == 0 else 5)
    assert encode_smf(original) == before


def test_release_velocity_never_changed_and_zero_velocity_key_on_preserved():
    events = (note(0, 0, 60, 120),
              MidiEvent.channel(20, 0x9, 0, 60, 0),
              note(30, 0, 62, 40), off(50, 0, 62, 91))
    midi = MidiFile(0, TimeDivision(), (MidiTrack.from_events(events, end=70),))
    out, r = edit(midi, velocity_numerator=2, velocity_offset=-3)
    assert [event.data for event in out.events if event.kind == 9] == [
        bytes((60, 127)), bytes((60, 0)), bytes((62, 77))]
    assert out.tracks[0].events[3].data == bytes((62, 91))
    assert r.clamped_velocities == 1
    assert r.selected_note_ons == 2
    assert r.selected_note_offs == 2


@pytest.mark.parametrize("num,den,offset,expected", [
    (1, 2, 0, 40), (1, 3, 0, 27), (2, 3, 0, 53),
    (5, 2, 0, 127), (1, 1, -126, 1), (1, 1, 47, 127),
    (1, 1, 0, 80),
])
def test_velocity_rational_rounding_and_clamping(num, den, offset, expected):
    out, _ = edit(velocity_numerator=num, velocity_denominator=den,
                  velocity_offset=offset)
    first = next(e for e in out.events if e.is_note_on)
    assert first.data[1] == expected


@pytest.mark.parametrize("value", [True, None, 1.0, "1", -1, 1025])
def test_velocity_numerator_rejects_invalid(value):
    with pytest.raises(MidiWorkstationError):
        MidiTransformSettings(velocity_numerator=value)


@pytest.mark.parametrize("value", [True, None, 1.0, "1", 0, 1025])
def test_velocity_denominator_rejects_invalid(value):
    with pytest.raises(MidiWorkstationError):
        MidiTransformSettings(velocity_denominator=value)


@pytest.mark.parametrize("value", [-50, 49, 3.4, True])
def test_invalid_transposition_rejected(value):
    with pytest.raises(MidiWorkstationError):
        MidiTransformSettings(semitones=value)


@pytest.mark.parametrize("selection", [(), (1, 0), (1, 1), (True,), (-1,), (64,), [0]])
def test_bad_track_selection_rejected(selection):
    with pytest.raises(MidiWorkstationError):
        MidiTransformSettings(tracks=selection)


@pytest.mark.parametrize("selection", [(), (1, 0), (1, 1), (True,), (-1,), (16,), [0]])
def test_bad_channel_selection_rejected(selection):
    with pytest.raises(MidiWorkstationError):
        MidiTransformSettings(channels=selection)


def test_selective_channel_transform_does_not_touch_other_channels():
    out, r = edit(song(drums=True), semitones=3, channels=(9,), include_percussion=True)
    left = [(e.kind, e.data[0]) for e in out.events if e.kind in (8, 9)]
    assert left == [(9, 60), (8, 60), (9, 64), (8, 64), (9, 39), (8, 39)]
    assert r.changed_pitch_events == 2


def test_selective_track_does_not_touch_unselected_track():
    t0 = MidiTrack.from_events((note(0, 0, 60), off(100, 0, 60)), end=200)
    t1 = MidiTrack.from_events((note(0, 1, 65), off(100, 1, 65)), end=200)
    midi = MidiFile(1, TimeDivision(), (t0, t1))
    out, _ = edit(midi, tracks=(1,), semitones=2)
    assert [e.data[0] for e in out.tracks[0].note_events] == [60, 60]
    assert [e.data[0] for e in out.tracks[1].note_events] == [67, 67]


def test_out_of_bounds_track_selection_raises_even_if_other_valid():
    with pytest.raises(MidiWorkstationError, match="absent"):
        edit(tracks=(1,), semitones=2)


def test_signature_not_implicitly_changed():
    source = song()
    out, r = edit(source, semitones=2)
    assert next(e.data for e in out.events if e.is_key) == b"\x01\x00"
    assert r.key_signatures_preserved_without_transposition == 1


@pytest.mark.parametrize("original,transpose,expected", [
    (0, 2, 2), (1, 2, 3), (-1, 2, 1), (0, 6, 6),
    (0, -6, 6), (3, -2, 1), (7, 1, 2),
])
def test_explicit_signature_shift_selects_minimal_accidental_spelling(original, transpose, expected):
    key = MidiEvent.meta(0, 0x59, bytes((original & 255, 0)))
    midi = MidiFile(0, TimeDivision(), (MidiTrack.from_events((key,), end=10),))
    out, report = edit(midi, semitones=transpose, transpose_key_signatures=True)
    signature = next(e for e in out.events if e.is_key)
    root_before = (7 * original) % 12
    root_after = (7 * int.from_bytes(signature.data[:1], "big", signed=True)) % 12
    assert root_after == (root_before + transpose) % 12
    assert report.changed_key_signatures == 1
    assert int.from_bytes(signature.data[:1], "big", signed=True) == expected


def test_filtered_key_signature_shift_refused():
    for option in (dict(tracks=(0,)), dict(channels=(0,)),
                   dict(pitch_low=1), dict(pitch_high=126)):
        with pytest.raises(MidiWorkstationError, match="whole-score"):
            MidiTransformSettings(transpose_key_signatures=True, **option)


def test_preserves_controller_and_sysex_opaque_payload_even_if_pitch_changes():
    source = song(meta=True)
    out, _ = edit(source, semitones=-2, velocity_numerator=4,
                  velocity_denominator=5)
    old = [e for e in source.events if e.kind not in (8, 9, 10)]
    new = [e for e in out.events if e.kind not in (8, 9, 10)]
    assert old == new
    assert [e.tick for e in out.events] == [e.tick for e in source.events]
    assert parse_smf(encode_smf(out)).midi == out


def test_digest_pin_rejects_wrong_source_without_mutating():
    source = song()
    before = encode_smf(source)
    with pytest.raises(MidiIntegrityError):
        transform_midi(source, MidiTransformSettings(semitones=2),
                       expected_source_sha256="0" * 64)
    assert encode_smf(source) == before
    assert transform_midi(source, expected_source_sha256=digest(before))[0] == source


def test_transform_detects_new_overlapping_same_pitch_collision():
    midi = MidiFile(0, TimeDivision(), (MidiTrack.from_events((
        note(0, 0, 60), note(10, 0, 62), off(100, 0, 60), off(110, 0, 62),
    ), end=150),))
    with pytest.raises(MidiWorkstationError, match="overlapping"):
        edit(midi, semitones=2, pitch_low=60, pitch_high=60)
    out, report = edit(midi, semitones=2, pitch_low=60, pitch_high=60,
                       reject_new_collisions=False)
    assert report.collisions_before == 0
    assert report.collisions_after == 1
    assert len(out.events) == 5


def test_existing_same_pitch_overlap_does_not_fail_if_not_increased():
    midi = MidiFile(0, TimeDivision(), (MidiTrack.from_events((
        note(0, 0, 60), note(10, 0, 60), off(100, 0, 60), off(110, 0, 60),
    ), end=150),))
    out, report = edit(midi, semitones=1)
    assert report.collisions_before == report.collisions_after == 1
    assert [e.data[0] for e in out.tracks[0].note_events] == [61, 61, 61, 61]


def test_randomized_mutation_preserves_separately_typed_nonpitch_events():
    randomizer = random.Random(20261128)
    for sample in range(180):
        starts = sorted(randomizer.sample(range(0, 4000), 14))
        events = []
        for index, tick in enumerate(starts):
            pitch = randomizer.randint(40, 80)
            events.extend((note(tick, index % 4, pitch, randomizer.randint(1, 127)),
                           off(tick + 3, index % 4, pitch, 0)))
            events.append(MidiEvent.meta(tick + 1, 0x7D, bytes((sample % 256, index))))
        midi = MidiFile(0, TimeDivision(), (
            MidiTrack.from_events(tuple(events), end=4500),
        ))
        original_bytes = encode_smf(midi)
        semitones = randomizer.randint(-12, 12)
        out, report = edit(midi, semitones=semitones,
                           velocity_numerator=randomizer.randint(1, 4),
                           velocity_denominator=randomizer.randint(1, 4))
        assert parse_smf(encode_smf(out)).midi == out
        assert midi == parse_smf(original_bytes).midi
        assert report.changed_pitch_events == (0 if not semitones else 28)
        assert [(event.tick, event.status, event.data) for event in midi.events
                if event.status == 0xFF] == [
                    (event.tick, event.status, event.data) for event in out.events
                    if event.status == 0xFF]


def test_cross_track_shared_channel_collision_is_detected():
    left = MidiTrack.from_events((note(0, 0, 60), off(100, 0, 60)), end=150)
    right = MidiTrack.from_events((note(10, 0, 62), off(110, 0, 62)), end=150)
    midi = MidiFile(1, TimeDivision(), (left, right))
    with pytest.raises(MidiWorkstationError, match="overlapping"):
        edit(midi, semitones=2, tracks=(0,))


def test_format_two_channels_are_independent_sequences():
    left = MidiTrack.from_events((note(0, 0, 60), off(100, 0, 60)), end=150)
    right = MidiTrack.from_events((note(10, 0, 62), off(110, 0, 62)), end=150)
    midi = MidiFile(2, TimeDivision(), (left, right))
    out, report = edit(midi, semitones=2, tracks=(0,))
    assert report.collisions_before == report.collisions_after == 0
    assert out.tracks[0].note_events[0].data[0] == 62
    assert out.tracks[1].note_events[0].data[0] == 62


def test_existing_overlap_identity_is_preserved_across_uniform_transposition():
    midi = MidiFile(1, TimeDivision(), (
        MidiTrack.from_events((note(0, 0, 60), off(100, 0, 60)), end=160),
        MidiTrack.from_events((note(20, 0, 60), off(120, 0, 60)), end=160),
    ))
    out, report = edit(midi, semitones=3)
    assert report.collisions_before == report.collisions_after == 1
    assert out.tracks[0].note_events[0].data[0] == 63


def test_new_collision_is_detected_even_when_total_collision_count_unchanged():
    source = MidiTrack.from_events((
        note(0, 0, 60), note(10, 0, 60),
        note(20, 1, 62), note(30, 1, 64),
        off(50, 0, 60), off(60, 0, 60),
        off(70, 1, 62), off(80, 1, 64),
    ), end=100)
    midi = MidiFile(0, TimeDivision(), (source,))
    # Move key 62 onto key 64: the original same-pitch overlap on
    # channel 0 is unchanged but the new channel-1 overlap must fail.
    with pytest.raises(MidiWorkstationError, match="overlapping"):
        edit(midi, semitones=2, pitch_low=62, pitch_high=62)
