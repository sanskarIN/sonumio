"""Immutable high-level musical score contracts and opt-in MIDI interoperability."""
import json
from dataclasses import replace

import pytest

from sonumio.midi_workstation.contracts import MidiFormatError, MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision
from sonumio.midi_workstation.performance import PerformanceSettings
from sonumio.midi_workstation.score import (
    Score, ScoreNote, ScoreTrack, from_midi, to_midi, note_sort_key,
)
from sonumio.midi_workstation.tempo import MeterPoint, TempoPoint
from sonumio.midi_workstation.smf import encode_smf, parse_smf


def note(id="n1", start=0, duration=240, pitch=60, velocity=90):
    return ScoreNote(id, start, duration, pitch, velocity)


def track(id="keys", channel=0, program=0, notes=(), name="Original keys"):
    return ScoreTrack(id, name, channel, program, 0, 0,
                      tuple(sorted(notes, key=note_sort_key)))


def song():
    return Score("Original composition", 480, (
        track("keys", 0, 0, (
            note("a", 0, 480, 60),
            note("b", 480, 480, 64),
            note("c", 960, 480, 67),
        )),
        track("bass", 1, 32, (
            note("bass1", 0, 960, 36),
            note("bass2", 960, 960, 43),
        )),
    ), (TempoPoint(0, 500000), TempoPoint(960, 420000)),
        (MeterPoint(0, 4, 4),), 1920)


def test_empty_score_has_musically_defined_clock_and_no_notes():
    s = Score.empty()
    assert s.ppqn == 480
    assert s.end == 1920
    assert s.note_count == 0
    assert s.active_track_ids == ("track-1",)
    assert s.duration_microseconds == 2000000
    assert len(s.sha256) == 64


def test_immutable_multitrack_score_summary():
    s = song()
    assert s.note_count == 5
    assert len(s.tracks) == 2
    assert s.track_by_id("bass").program == 32
    assert s.note_by_id("keys", "b").pitch == 64
    assert s.tempo_map.tempo_at(1000).microseconds_per_quarter == 420000
    assert s.duration_microseconds == 1840000
    assert s.description()["trained_models_used"] is False


def test_score_midi_roundtrip_preserves_pitch_timing_and_programs():
    original = song()
    midi = to_midi(original)
    assert midi.format == 1
    assert len(midi.tracks) == 3
    assert midi.division.ppqn == 480
    encoded = encode_smf(midi)
    decoded = parse_smf(encoded).midi
    assert decoded == midi
    restored, report = from_midi(decoded)
    assert restored.note_count == original.note_count
    assert restored.ppqn == original.ppqn
    assert restored.end == original.end
    assert restored.tempo == original.tempo
    assert restored.meter == original.meter
    assert report["performance_controller_messages_not_represented"] is True
    for original_track in original.tracks:
        same = next(t for t in restored.tracks if t.channel == original_track.channel)
        assert same.program == original_track.program
        actual = [(n.start, n.duration, n.pitch, n.velocity) for n in same.notes]
        expected = [(n.start, n.duration, n.pitch, n.velocity) for n in original_track.notes]
        assert actual == expected


def test_midi_export_has_notes_off_before_retrigger_on():
    s = Score("New music", 480, (
        track("keys", notes=(note("one", 0, 120), note("two", 120, 120))),
    ), (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 300)
    midi = to_midi(s)
    events = midi.tracks[1].events
    same_tick = [e for e in events if e.tick == 120]
    assert [event.kind for event in same_tick] == [0x8, 0x9]


def test_note_velocity_and_pan_do_not_silently_change_exported_geometry():
    s = Score.empty()
    n = ScoreNote("velocity", 4, 120, 70, velocity=127,
                  release_velocity=50, pan=16)
    t = s.tracks[0].replace_notes((n,))
    updated = s.with_tracks((t,))
    midi = to_midi(updated)
    key_on = next(e for e in midi.tracks[1].events if e.kind == 0x9)
    key_off = next(e for e in midi.tracks[1].events if e.kind == 0x8)
    assert key_on.data == bytes((70, 127))
    assert key_off.data == bytes((70, 50))


def test_export_can_exclude_muted_tracks_without_mutating_score():
    s = song()
    muted = replace(s.tracks[1], muted=True)
    updated = s.with_tracks((s.tracks[0], muted))
    assert len(to_midi(updated, include_muted=True).tracks) == 3
    assert len(to_midi(updated, include_muted=False).tracks) == 2
    assert len(updated.active_notes) == 3
    assert len(updated.notes) == 5


def test_solo_takes_precedence_over_unmuted_tracks():
    s = song()
    tracks = (replace(s.tracks[0], solo=False), replace(s.tracks[1], solo=True))
    updated = s.with_tracks(tracks)
    assert updated.active_track_ids == ("bass",)
    assert len(updated.active_notes) == 2


def test_score_revision_digest_is_content_sensitive():
    s = song()
    base = s.sha256
    altered = s.with_tracks(s.tracks)
    assert altered.revision == s.revision + 1
    assert altered.sha256 != base
    assert s.sha256 == base


def test_score_track_order_is_significant_for_presentation():
    s = song()
    reordered = s.with_tracks(s.tracks[::-1])
    assert reordered.sha256 != s.sha256
    assert reordered.note_count == s.note_count


def test_duplicate_track_ids_not_allowed():
    s = song()
    with pytest.raises(MidiWorkstationError):
        s.with_tracks((s.tracks[0], s.tracks[0]))


def test_duplicate_note_ids_not_allowed():
    with pytest.raises(MidiWorkstationError):
        track(notes=(note("id", 0), note("id", 240)))


def test_note_reordering_requires_canonical_sorting():
    with pytest.raises(MidiWorkstationError):
        ScoreTrack("a", "A", 0, 0, 0, 0,
                   (note("later", 480), note("earlier", 0)))


def test_shared_channels_with_different_patches_rejected():
    with pytest.raises(MidiWorkstationError):
        Score("Conflict", 480,
              (track("a", 0, 2), track("b", 0, 3)),
              (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 500)


def test_shared_channel_with_same_patch_can_be_layered():
    s = Score("Layered", 480,
              (track("a", 0, 2), track("b", 0, 2)),
              (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 500)
    assert len(s.tracks) == 2


def test_score_notes_cannot_extend_past_global_end():
    with pytest.raises(MidiWorkstationError):
        Score("Overflow", 480, (track(notes=(note("long", 490, 40),)),),
              (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 500)


def test_note_transpose_shift_and_resize_validate_midi_range():
    n = note("n", 10, 100, 60)
    assert n.transpose(12).pitch == 72
    assert n.shift(20).start == 30
    assert n.resized(20).stop == 30
    assert n.with_velocity(120).velocity == 120
    with pytest.raises(MidiWorkstationError):
        n.transpose(100)
    with pytest.raises(MidiWorkstationError):
        n.shift(-20)
    with pytest.raises(MidiWorkstationError):
        n.resized(0)


@pytest.mark.parametrize("bad", [
    dict(id="not valid"), dict(id=""), dict(start=-1),
    dict(duration=0), dict(pitch=-1), dict(pitch=128),
    dict(velocity=0), dict(velocity=128), dict(pan=128),
    dict(release_velocity=-1),
])
def test_invalid_note_rejected(bad):
    payload = dict(id="n", start=0, duration=100, pitch=60)
    payload.update(bad)
    with pytest.raises(MidiWorkstationError):
        ScoreNote(**payload)


def test_smpte_midi_must_not_be_guessed_into_quarter_note_grid():
    file = MidiFile(0, TimeDivision(None, -30, 80), (
        MidiTrack.from_events((
            MidiEvent.channel(0, 0x9, 0, 60, 100),
            MidiEvent.channel(100, 0x8, 0, 60, 0),
        ), end=100),
    ))
    with pytest.raises(MidiWorkstationError):
        from_midi(file)


def test_import_rejects_unrepresentable_program_changes():
    file = MidiFile(0, TimeDivision(480), (
        MidiTrack.from_events((
            MidiEvent.channel(0, 0xC, 0, 1),
            MidiEvent.channel(0, 0x9, 0, 60, 100),
            MidiEvent.channel(100, 0x8, 0, 60, 0),
            MidiEvent.channel(100, 0xC, 0, 2),
            MidiEvent.channel(100, 0x9, 0, 64, 100),
            MidiEvent.channel(200, 0x8, 0, 64, 0),
        ), end=200),
    ))
    with pytest.raises(MidiWorkstationError):
        from_midi(file)


def test_midi_import_discloses_sustain_semantic_change():
    midi = MidiFile(0, TimeDivision(480), (
        MidiTrack.from_events((
            MidiEvent.channel(0, 0xB, 0, 64, 127),
            MidiEvent.channel(0, 0x9, 0, 60, 100),
            MidiEvent.channel(100, 0x8, 0, 60, 0),
            MidiEvent.channel(200, 0xB, 0, 64, 0),
        ), end=240),
    ))
    physical, ledger = from_midi(midi, audible_duration=False)
    audible, ledger2 = from_midi(midi, audible_duration=True)
    assert physical.notes[0].duration == 100
    assert audible.notes[0].duration == 200
    assert ledger2["original_sustain_applied"] is True
    assert ledger["original_sustain_applied"] is False


def test_export_and_source_sound_not_confused_with_models():
    s = song()
    midi = to_midi(s)
    assert not any(e.is_sysex for e in midi.events)
    assert len(midi.tempo_events) == 2
    assert len(midi.meter_events) == 1
    assert s.description()["trained_models_used"] is False
