"""Original-track provenance survives format-1 shared channel state."""

import pytest

from sonumio.midi_workstation.contracts import MidiFormatError, MidiWorkstationError
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision
from sonumio.midi_workstation.performance import (
    PerformanceSettings, reconstruct_performance,
)


def event(tick, kind, channel, *values):
    return MidiEvent.channel(tick, kind, channel, *values)


def format_one(*track_events, end=480):
    return MidiFile(1, TimeDivision(480), tuple(
        MidiTrack.from_events(tuple(events), end=end)
        for events in track_events
    ))


def shared(source, *, dangling="reject"):
    return reconstruct_performance(source, PerformanceSettings(
        isolate_tracks=False, dangling_note_on=dangling,
    ))


def test_cross_track_note_off_preserves_original_note_track():
    source = format_one(
        (event(0, 0x9, 0, 60, 101),),
        (event(120, 0x8, 0, 60, 32),),
    )
    with pytest.raises(MidiFormatError):
        reconstruct_performance(source)
    note = shared(source).notes[0]
    assert note.track == 0
    assert note.key_release == 120
    assert note.release_velocity == 32


def test_note_on_second_track_released_from_first_keeps_second_provenance():
    source = format_one(
        (event(100, 0x8, 0, 64, 0),),
        (event(0, 0x9, 0, 64, 87),),
    )
    note = shared(source).notes[0]
    assert note.track == 1
    assert note.velocity == 87
    assert note.sound_stop == 100


def test_shared_sustain_release_does_not_misattribute_note():
    source = format_one(
        (event(0, 0xB, 0, 64, 127),
         event(200, 0xB, 0, 64, 0)),
        (event(20, 0x9, 0, 67, 110),
         event(80, 0x8, 0, 67, 15)),
    )
    note = shared(source).notes[0]
    assert note.track == 1
    assert note.key_release == 80
    assert note.sound_stop == 200
    assert note.sustained is True


def test_all_notes_off_from_other_track_preserves_source():
    source = format_one(
        (event(90, 0xB, 0, 123, 0),),
        (event(10, 0x9, 0, 60, 100),),
    )
    result = shared(source)
    assert result.notes[0].track == 1
    assert result.notes[0].key_release == 90
    assert result.diagnostics.forced_closures == 1


def test_dangling_shared_note_retains_own_track():
    source = format_one(
        (event(0, 0xB, 0, 7, 90),),
        (event(80, 0x9, 0, 69, 100),),
        end=240,
    )
    result = shared(source, dangling="close")
    assert result.notes[0].track == 1
    assert result.notes[0].sound_stop == 240


def test_cross_track_program_change_is_a_snapshot_not_track_change():
    source = format_one(
        (event(0, 0xC, 3, 49),
         event(0, 0xB, 3, 0, 2),
         event(0, 0xB, 3, 32, 4)),
        (event(20, 0x9, 3, 72, 93),
         event(100, 0x8, 3, 72, 0)),
    )
    note = shared(source).notes[0]
    assert (note.track, note.channel, note.program) == (1, 3, 49)
    assert (note.bank_msb, note.bank_lsb) == (2, 4)


def test_track_isolation_keeps_note_offs_separate():
    source = format_one(
        (event(0, 0x9, 0, 60, 100), event(120, 0x8, 0, 60, 0)),
        (event(20, 0x9, 0, 64, 100), event(140, 0x8, 0, 64, 0)),
    )
    independent = reconstruct_performance(source)
    combined = shared(source)
    assert {(n.track, n.pitch) for n in independent.notes} == {
        (0, 60), (1, 64),
    }
    assert [(n.track, n.pitch) for n in combined.notes] == [
        (0, 60), (1, 64),
    ]


def test_format_two_still_rejects_shared_channel_state():
    source = MidiFile(2, TimeDivision(480), (
        MidiTrack.from_events((event(0, 0x9, 0, 60, 100),
                               event(100, 0x8, 0, 60, 0)), end=100),
        MidiTrack.from_events((event(0, 0x9, 0, 64, 100),
                               event(100, 0x8, 0, 64, 0)), end=100),
    ))
    with pytest.raises(MidiWorkstationError):
        shared(source)
