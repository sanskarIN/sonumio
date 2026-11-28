"""Immutable region transforms, bounded partial notes, and tempo-aware clips."""
from dataclasses import replace

import pytest

from sonumio.midi_workstation.contracts import MidiWorkstationError
from sonumio.midi_workstation.regions import (
    TickRegion, reverse_region, transpose_region, stretch_region,
    repeat_region, crop_score, concatenate_scores,
)
from sonumio.midi_workstation.score import Score, ScoreNote
from sonumio.midi_workstation.tempo import TempoPoint, MeterPoint


def score():
    base = Score.empty(end=3840)
    track = base.tracks[0].replace_notes((
        ScoreNote("a", 100, 100, 60),
        ScoreNote("b", 320, 120, 64),
        ScoreNote("c", 620, 150, 67),
        ScoreNote("d", 1120, 150, 72),
    ))
    return base.with_tracks((track,))


def test_reverse_region_preserves_duration_pitch_and_velocity():
    original = score()
    result, report = reverse_region(original, TickRegion(0, 960))
    assert [n.start for n in result.notes] == [190, 520, 760, 1120]
    assert [n.pitch for n in result.notes] == [67, 64, 60, 72]
    assert sorted(n.duration for n in result.notes) == sorted(n.duration for n in original.notes)
    assert report.selected == 3
    assert original.notes[0].start == 100


def test_double_reversal_restores_original_music():
    original = score()
    once, _ = reverse_region(original, TickRegion(0, 960))
    twice, _ = reverse_region(once, TickRegion(0, 960))
    assert twice.notes == original.notes
    assert twice.sha256 != original.sha256  # Revision counters are distinct.


def test_transpose_region_leaves_outside_events_alone():
    source = score()
    output, report = transpose_region(source, TickRegion(0, 960), 12)
    assert [n.pitch for n in output.notes] == [72, 76, 79, 72]
    assert report.selected == 3
    assert source.notes[0].pitch == 60


def test_stretch_region_uses_separately_rounded_endpoints():
    source = score()
    output, report = stretch_region(source, TickRegion(0, 960), 1, 2)
    assert [(n.start, n.duration) for n in output.notes[:3]] == [
        (50, 50), (160, 60), (310, 75),
    ]
    assert output.notes[3] == source.notes[3]
    assert report.selected == 3


def test_stretch_region_rejects_overflow():
    source = score()
    with pytest.raises(MidiWorkstationError):
        stretch_region(source, TickRegion(0, 3840), 20, 1)


def test_boundary_crossing_requires_explicit_policy():
    source = score()
    region = TickRegion(350, 600)
    with pytest.raises(MidiWorkstationError):
        reverse_region(source, region)
    with pytest.raises(MidiWorkstationError):
        transpose_region(source, region, 2)
    with pytest.raises(MidiWorkstationError):
        stretch_region(source, region, 2, 1)
    result, report = transpose_region(source, region, 2, overlapping="preserve")
    assert result.notes == source.notes
    assert report.selected == 0


def test_repeat_region_appends_stable_namespaced_notes():
    source = score()
    result, report = repeat_region(source, TickRegion(0, 960), 2)
    assert result.note_count == 10
    assert report.created == 6
    assert result.note_by_id("track-1", "a-r1").start == 1060
    assert result.note_by_id("track-1", "a-r2").start == 2020
    assert source.note_count == 4


def test_repeated_note_id_collision_fails_closed():
    original = score()
    track = original.tracks[0].replace_notes(original.notes + (
        ScoreNote("a-r1", 1700, 30, 60),
    ))
    source = original.with_tracks((track,))
    with pytest.raises(MidiWorkstationError):
        repeat_region(source, TickRegion(0, 960), 1)


def test_repeat_requires_room_for_all_copies():
    with pytest.raises(MidiWorkstationError):
        repeat_region(score(), TickRegion(0, 1920), 2)


def test_crop_with_trimmed_boundaries_and_shifted_clock():
    source = score()
    crop, receipt = crop_score(source, TickRegion(150, 700))
    assert crop.end == 550
    assert crop.note_count == 3
    assert [(n.start, n.duration) for n in crop.notes] == [
        (0, 50), (170, 120), (470, 80),
    ]
    assert crop.tempo[0].tick == 0
    assert receipt.trimmed == 2
    assert source.note_count == 4


def test_crop_rejects_overlapping_boundaries_if_requested():
    with pytest.raises(MidiWorkstationError):
        crop_score(score(), TickRegion(150, 700), crossing="reject")


def test_crop_preserves_active_tempo_and_meter():
    base = score()
    source = replace(base,
        tempo=(TempoPoint(0, 500000), TempoPoint(960, 400000),
               TempoPoint(2200, 600000)),
        meter=(MeterPoint(0, 4, 4), MeterPoint(960, 3, 4)))
    crop, _ = crop_score(source, TickRegion(1200, 3200))
    assert crop.tempo[0] == TempoPoint(0, 400000)
    assert crop.tempo[1] == TempoPoint(1000, 600000)
    assert crop.meter == (MeterPoint(0, 3, 4),)


def test_concatenate_rebases_notes_and_restores_tempo_context():
    source = score()
    first, _ = crop_score(source, TickRegion(0, 960))
    second, _ = crop_score(source, TickRegion(960, 1920))
    assembled, report = concatenate_scores((first, second), title="Two clips")
    assert assembled.end == 1920
    assert assembled.note_count == 4
    assert assembled.title == "Two clips"
    assert assembled.notes[-1].id.startswith("p1-")
    assert report["source_unchanged"] is True
    assert first.notes[0].id == "a"


def test_concatenate_multiple_same_track_parts():
    original = score()
    excerpt, _ = crop_score(original, TickRegion(0, 960))
    assembled, report = concatenate_scores((excerpt, excerpt, excerpt))
    assert assembled.end == 2880
    assert assembled.note_count == 9
    assert len({n.id for n in assembled.notes}) == 9
    assert report["part_count"] == 3


def test_concatenate_rejects_mismatched_track_channel():
    base = score()
    different = Score.empty(end=960)
    different = replace(different, tracks=(replace(different.tracks[0], channel=3),))
    with pytest.raises(MidiWorkstationError):
        concatenate_scores((base, different))


def test_nonpositive_and_invalid_regions_fail_closed():
    with pytest.raises(MidiWorkstationError):
        TickRegion(100, 100)
    with pytest.raises(MidiWorkstationError):
        TickRegion(-1, 100)
    with pytest.raises(MidiWorkstationError):
        TickRegion(100, True)


def test_unknown_track_rejected_for_all_region_edit_operations():
    source = score()
    for operation in (
        lambda: transpose_region(source, TickRegion(0, 960), 5, tracks=("missing",)),
        lambda: reverse_region(source, TickRegion(0, 960), tracks=("missing",)),
        lambda: stretch_region(source, TickRegion(0, 960), 2, 1, tracks=("missing",)),
        lambda: crop_score(source, TickRegion(0, 960), tracks=("missing",)),
    ):
        with pytest.raises(MidiWorkstationError):
            operation()


def test_empty_selected_track_crop_is_rejected():
    with pytest.raises(MidiWorkstationError):
        crop_score(score(), TickRegion(0, 960), tracks=())


def test_changed_score_receipts_are_json_compatible():
    import json
    source = score()
    next_score, report = transpose_region(source, TickRegion(0, 960), 3)
    json.dumps(report.as_dict(), allow_nan=False)
    assert report.as_dict()["remote_action_performed"] is False
    assert report.input_sha256 == source.sha256
    assert report.output_sha256 == next_score.sha256
