"""Exact rational tempo-to-audio frame scheduling and deterministic transport."""
from fractions import Fraction
import json
import pytest

from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.tempo import TempoPoint, MeterPoint
from sonumio.midi_workstation.transport import (
    TransportSettings, ScheduledNote, ScheduledEvent, Schedule,
    TransportCursor, build_schedule, event_order,
)


def song():
    return Score("Transport test", 480, (
        ScoreTrack("lead", "Lead", 0, 80, 0, 0, (
            ScoreNote("one", 0, 480, 60, 100),
            ScoreNote("two", 480, 480, 64, 110),
            ScoreNote("three", 960, 480, 67, 120),
        )),
        ScoreTrack("bass", "Bass", 1, 32, 0, 0, (
            ScoreNote("bass1", 0, 960, 36, 80),
        )),
    ), (TempoPoint(0, 500000), TempoPoint(960, 250000)),
        (MeterPoint(0, 4, 4),), 1920)


def test_frames_are_integrated_across_exact_tempo_changes():
    s = song()
    plan = build_schedule(s, TransportSettings(sample_rate=48000))
    assert plan.timeline_end_frame == 72000
    assert plan.output_frames == 72000
    assert len(plan.notes) == 4
    lead = [n for n in plan.notes if n.track_id == "lead"]
    assert [(n.start_frame, n.stop_frame) for n in lead] == [
        (0, 24000), (24000, 48000), (48000, 60000),
    ]
    assert plan.source_sha256 == s.sha256


def test_tail_is_explicit_and_not_counted_as_musical_timeline():
    result = build_schedule(song(), TransportSettings(sample_rate=8000,
                                                       tail_frames=1000))
    assert result.timeline_end_frame == 12000
    assert result.output_frames == 13000
    assert result.events_between(12000, 13000) == ()


def test_note_on_and_off_appear_in_stable_frame_order():
    result = build_schedule(song())
    assert list(result.events) == sorted(result.events, key=event_order)
    assert len(result.events) == 8
    at_zero = result.events_between(0, 1)
    assert len(at_zero) == 2
    assert all(e.kind == "note-on" for e in at_zero)
    at_boundary = result.events_between(12000, 12001)
    assert [e.kind for e in at_boundary] == ["note-off", "note-on"]


def test_active_notes_at_sample_frame_half_open():
    result = build_schedule(song(), TransportSettings(sample_rate=48000))
    assert len(result.active_at(0)) == 2
    assert len(result.active_at(24000)) == 2
    assert len(result.active_at(59999)) == 1
    assert len(result.active_at(72000)) == 0


def test_midi_pan_maps_center_and_endpoints_exactly():
    s = song()
    lead = s.track_by_id("lead")
    modified = lead.replace_notes((
        ScoreNote("left", 0, 120, 60, pan=0),
        ScoreNote("center", 120, 120, 60, pan=64),
        ScoreNote("right", 240, 120, 60, pan=127),
    ))
    updated = s.with_tracks((modified, s.track_by_id("bass")))
    schedule = build_schedule(updated)
    pans = {item.note_id: item.pan_signed for item in schedule.notes}
    assert pans["left"] == Fraction(-1)
    assert pans["center"] == Fraction(0)
    assert pans["right"] == Fraction(1)


def test_transport_cursor_seek_and_advance_do_not_mutate():
    schedule = build_schedule(song())
    cursor = TransportCursor(schedule)
    assert cursor.frame == 0
    assert not cursor.at_end
    moved, event_batch = cursor.advance(12000)
    assert moved.frame == 12000
    assert cursor.frame == 0
    assert len(event_batch) > 0
    end = moved.seek(schedule.output_frames)
    assert end.at_end
    assert end.remaining() == 0


def test_transport_chunk_size_does_not_change_emitted_event_sequence():
    schedule = build_schedule(song())
    baseline = list(schedule.events)
    for chunk in (1, 7, 64, 133, 1024, 20000):
        cursor = TransportCursor(schedule)
        combined = []
        while not cursor.at_end:
            cursor, events = cursor.advance(chunk)
            combined.extend(events)
        assert combined == baseline


def test_silent_score_has_duration_but_no_events():
    score = Score.empty()
    schedule = build_schedule(score)
    assert schedule.notes == ()
    assert schedule.events == ()
    assert schedule.output_frames > 0
    assert schedule.total_note_frames == 0


def test_score_solo_and_mute_selection_are_honored():
    from dataclasses import replace
    s = song()
    bass = replace(s.track_by_id("bass"), muted=True)
    changed = s.with_tracks((s.track_by_id("lead"), bass))
    default = build_schedule(changed)
    included = build_schedule(changed, TransportSettings(include_muted=True))
    assert len(default.notes) == 3
    assert len(included.notes) == 4


def test_minimum_positive_frame_rule_can_adjust_short_notes():
    tiny = Score("Tiny", 32767, (
        ScoreTrack("a", "A", 0, 0, 0, 0, (
            ScoreNote("short", 0, 1, 60),
        )),
    ), (TempoPoint(0, 10000),), (MeterPoint(0, 4, 4),), 32767)
    adjusted = build_schedule(tiny, TransportSettings(sample_rate=8000))
    assert adjusted.collision_adjustments == 1
    assert adjusted.notes[0].frames == 1
    with pytest.raises(MidiWorkstationError):
        build_schedule(tiny, TransportSettings(sample_rate=8000,
                                               allow_collision_adjustment=False))


def test_duration_budget_fails_before_audio_rendering():
    s = song()
    with pytest.raises(MidiLimitError):
        build_schedule(s, TransportSettings(sample_rate=48000,
                                            maximum_output_frames=1000))


def test_report_is_json_compatible_and_no_model_claim():
    result = build_schedule(song())
    payload = json.dumps(result.as_dict(), sort_keys=True, allow_nan=False)
    assert result.source_sha256 in payload
    assert result.as_dict()["model_inference"] is False
    assert result.as_dict()["audio_rendered"] is False


@pytest.mark.parametrize("bad", [
    dict(sample_rate=7000), dict(sample_rate=True),
    dict(rounding="stochastic"), dict(minimum_note_frames=0),
    dict(tail_frames=-1), dict(include_muted=1),
    dict(allow_collision_adjustment=1), dict(maximum_output_frames=0),
])
def test_invalid_transport_settings_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        TransportSettings(**bad)


def test_invalid_schedule_event_and_note_model_rejected():
    with pytest.raises(MidiWorkstationError):
        ScheduledEvent(0, "mute", "a", "n", 60, 0)
    with pytest.raises(MidiWorkstationError):
        ScheduledNote("a", "n", 0, 0, 0, 0, 60, 100, 64,
                      0, 10, 20, 20)
