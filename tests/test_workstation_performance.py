"""Controller-aware MIDI performance pairing, sustain and explicit loss policy."""
import pytest

from sonumio.midi_workstation.contracts import (
    MidiFormatError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.model import MidiFile, MidiEvent, MidiTrack, TimeDivision
from sonumio.midi_workstation.performance import (
    Performance, PerformanceSettings, NoteSpan,
    reconstruct_performance, performance_report,
)


def event(tick, kind, channel, *payload):
    return MidiEvent.channel(tick, kind, channel, *payload)


def score(*events, end=480, format=0):
    track = MidiTrack.from_events(tuple(sorted(events, key=lambda e: e.tick)), end=end)
    return MidiFile(format, TimeDivision(480), (track,))


def played(*events, settings=PerformanceSettings(), end=480):
    return reconstruct_performance(score(*events, end=end), settings)


def test_physical_key_on_off_note_reconstructed():
    performance = played(event(20, 0x9, 0, 60, 101), event(80, 0x8, 0, 60, 50))
    assert len(performance.notes) == 1
    note = performance.notes[0]
    assert note.start == 20
    assert note.key_release == note.sound_stop == 80
    assert note.key_duration == note.audible_duration == 60
    assert note.velocity == 101
    assert note.release_velocity == 50
    assert note.program == 0
    assert note.sustained is False


def test_zero_velocity_note_on_behaves_like_note_off():
    result = played(event(0, 0x9, 1, 64, 100), event(100, 0x9, 1, 64, 0))
    assert len(result.notes) == 1
    assert result.notes[0].start == 0
    assert result.notes[0].sound_stop == 100


@pytest.mark.parametrize("mode,expected", [("fifo", [(0, 100), (20, 200)]),
                                                ("lifo", [(0, 200), (20, 100)])])
def test_overlapping_same_pitch_pairs_by_declared_policy(mode, expected):
    result = played(
        event(0, 0x9, 0, 60, 90), event(20, 0x9, 0, 60, 80),
        event(100, 0x8, 0, 60, 0), event(200, 0x8, 0, 60, 0),
        settings=PerformanceSettings(same_pitch=mode),
    )
    assert [(n.start, n.key_release) for n in result.notes] == expected
    assert result.diagnostics.overlapping_onsets == 1


def test_same_pitch_overlap_can_be_rejected():
    with pytest.raises(MidiFormatError):
        played(event(0, 0x9, 0, 60, 90), event(20, 0x9, 0, 60, 90),
               settings=PerformanceSettings(same_pitch="reject"))


def test_sustain_holds_note_after_physical_release():
    result = played(
        event(0, 0xB, 0, 64, 127),
        event(10, 0x9, 0, 60, 100),
        event(100, 0x8, 0, 60, 32),
        event(200, 0xB, 0, 64, 0),
    )
    note = result.notes[0]
    assert note.key_release == 100
    assert note.sound_stop == 200
    assert note.sustained
    assert note.release_velocity == 32
    assert result.diagnostics.sustain_releases == 1


def test_sustain_policy_can_ignore_pedal():
    notes = played(
        event(0, 0xB, 0, 64, 127),
        event(10, 0x9, 0, 60, 100),
        event(100, 0x8, 0, 60, 0),
        event(200, 0xB, 0, 64, 0),
        settings=PerformanceSettings(sustain=False),
    ).notes
    assert notes[0].sound_stop == 100
    assert not notes[0].sustained


def test_multiple_sustained_notes_release_simultaneously():
    performance = played(
        event(0, 0xB, 0, 64, 120),
        event(0, 0x9, 0, 60, 100),
        event(10, 0x9, 0, 64, 90),
        event(50, 0x8, 0, 60, 0),
        event(80, 0x8, 0, 64, 0),
        event(120, 0xB, 0, 64, 63),
    )
    assert len(performance.notes) == 2
    assert all(note.sound_stop == 120 for note in performance.notes)
    assert performance.diagnostics.sustain_releases == 2


def test_program_and_bank_snapshot_at_note_on():
    result = played(
        event(0, 0xB, 0, 0, 2),
        event(0, 0xB, 0, 32, 4),
        event(0, 0xC, 0, 49),
        event(20, 0x9, 0, 72, 100),
        event(80, 0xC, 0, 9),
        event(100, 0x8, 0, 72, 0),
    )
    note = result.notes[0]
    assert note.program == 49
    assert note.bank_msb == 2
    assert note.bank_lsb == 4
    assert note.bank == 260


@pytest.mark.parametrize("controller", [120, 123])
def test_all_notes_off_controllers_close_active_notes(controller):
    result = played(
        event(10, 0x9, 0, 60, 100),
        event(20, 0x9, 0, 64, 100),
        event(100, 0xB, 0, controller, 0),
    )
    assert len(result.notes) == 2
    assert all(n.sound_stop == 100 for n in result.notes)
    assert result.diagnostics.forced_closures == 2


def test_reset_all_controllers_releases_sustained_notes():
    result = played(
        event(0, 0xB, 0, 64, 127),
        event(10, 0x9, 0, 60, 100),
        event(70, 0x8, 0, 60, 0),
        event(100, 0xB, 0, 121, 0),
    )
    assert result.notes[0].sound_stop == 100


def test_unmatched_note_off_is_rejected_by_default():
    with pytest.raises(MidiFormatError):
        played(event(20, 0x8, 0, 60, 0))


def test_unmatched_note_off_can_be_ignored():
    result = played(event(20, 0x8, 0, 60, 0),
                    settings=PerformanceSettings(unmatched_note_off="ignore"))
    assert result.notes == ()
    assert result.diagnostics.ignored_note_offs == 1


def test_unmatched_note_on_requires_explicit_close_policy():
    with pytest.raises(MidiFormatError):
        played(event(20, 0x9, 0, 60, 100))
    result = played(event(20, 0x9, 0, 60, 100),
                    settings=PerformanceSettings(dangling_note_on="close"))
    assert result.notes[0].sound_stop == 480
    assert result.diagnostics.auto_closed_at_end


def test_sustain_hanging_at_eot_can_be_closed():
    result = played(
        event(0, 0xB, 0, 64, 127),
        event(10, 0x9, 0, 60, 100),
        event(20, 0x8, 0, 60, 0),
        settings=PerformanceSettings(dangling_note_on="close"),
    )
    assert result.notes[0].sound_stop == 480
    assert result.notes[0].key_release == 20


def test_zero_duration_note_excluded_unless_explicit():
    result = played(event(20, 0x9, 0, 60, 100),
                    event(20, 0x8, 0, 60, 0))
    assert len(result.notes) == 0
    assert result.diagnostics.zero_duration_omissions == 1
    included = played(event(20, 0x9, 0, 60, 100),
                      event(20, 0x8, 0, 60, 0),
                      settings=PerformanceSettings(count_zero_duration=True))
    assert len(included.notes) == 1
    assert included.notes[0].sound_stop == included.notes[0].start


def test_two_channels_at_same_pitch_do_not_interfere():
    result = played(
        event(0, 0x9, 0, 60, 90), event(10, 0x9, 1, 60, 100),
        event(60, 0x8, 0, 60, 0), event(70, 0x8, 1, 60, 0),
    )
    assert len(result.notes) == 2
    assert [(note.channel, note.key_release) for note in result.notes] == [(0, 60), (1, 70)]


def test_format_two_requires_isolated_performance_tracks():
    one = MidiTrack.from_events((event(0, 0x9, 0, 60, 100),
                                 event(100, 0x8, 0, 60, 0)), end=100)
    two = MidiTrack.from_events((event(0, 0x9, 0, 64, 100),
                                 event(120, 0x8, 0, 64, 0)), end=120)
    source = MidiFile(2, TimeDivision(), (one, two))
    with pytest.raises(MidiWorkstationError):
        reconstruct_performance(source, PerformanceSettings(isolate_tracks=False))
    result = reconstruct_performance(source)
    assert len(result.notes) == 2
    assert {n.track for n in result.notes} == {0, 1}


def test_performance_filter_keeps_original_diagnostics():
    result = played(
        event(0, 0x9, 0, 60, 100), event(100, 0x8, 0, 60, 0),
        event(0, 0x9, 1, 67, 100), event(120, 0x8, 1, 67, 0),
    )
    filtered = result.filter(channel=1)
    assert len(filtered.notes) == 1
    assert filtered.notes[0].pitch == 67
    assert filtered.diagnostics == result.diagnostics


def test_inspection_reports_declared_semantic_limits():
    result = played(event(0, 0x9, 0, 60, 100), event(100, 0x8, 0, 60, 0))
    report = performance_report(result)
    assert report["audio_rendered"] is False
    assert report["physical_key_release_preserved"] is True
    assert report["notes"] == 1
    assert report["diagnostics"]["model_inference"] is False


@pytest.mark.parametrize("bad", [
    dict(same_pitch="maybe"), dict(unmatched_note_off="repair"),
    dict(dangling_note_on="clip"), dict(sustain=1),
    dict(isolate_tracks=0), dict(count_zero_duration=1),
])
def test_invalid_performance_policies_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        PerformanceSettings(**bad)
