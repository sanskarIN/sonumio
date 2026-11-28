"""Groove curves, symbolic extraction, loop duplication and timing boundaries."""
import json
import pytest

from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.groove import (
    GrooveSlot, GrooveTemplate, GrooveApplySettings, GrooveObservation,
    Clip, apply_groove_note, apply_groove, extract_groove, render_clip,
)
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack, note_sort_key
from sonumio.midi_workstation.tempo import TempoPoint, MeterPoint


def score():
    instruments = (
        ScoreTrack("drums", "Original drums", 9, 0, 0, 0, (
            ScoreNote("kick1", 0, 50, 36, 100),
            ScoreNote("hat1", 120, 50, 42, 100),
            ScoreNote("snare1", 240, 50, 38, 100),
            ScoreNote("hat2", 360, 50, 42, 100),
            ScoreNote("kick2", 480, 50, 36, 100),
            ScoreNote("hat3", 600, 50, 42, 100),
            ScoreNote("snare2", 720, 50, 38, 100),
            ScoreNote("hat4", 840, 50, 42, 100),
        )),
        ScoreTrack("piano", "Keys", 0, 0, 0, 0, (
            ScoreNote("p1", 0, 480, 60, 80),
            ScoreNote("p2", 480, 480, 64, 80),
        )),
    )
    return Score("Procedural groove", 480, instruments,
                 (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 2400)


def test_straight_groove_has_no_timing_or_velocity_changes():
    template = GrooveTemplate.straight()
    result, report = apply_groove(score(), template)
    assert all(a.notes == b.notes for a, b in zip(result.tracks, score().tracks))
    assert report["changed_notes"] == 0
    assert report["source_mutated"] is False


def test_swing_delays_alternate_subdivisions():
    template = GrooveTemplate.swing(step_ticks=120, slots=16, swing_percent=50)
    assert template.steps[0].offset_ticks == 0
    assert template.steps[1].offset_ticks == 60
    assert template.steps[2].offset_ticks == 0
    assert template.steps[3].offset_ticks == 60
    assert template.period_ticks == 1920


def test_groove_adjusts_only_selected_tracks():
    source = score()
    swing = GrooveTemplate.swing(step_ticks=120, swing_percent=25)
    output, report = apply_groove(source, swing, tracks=("drums",))
    assert output.track_by_id("piano") == source.track_by_id("piano")
    assert output.track_by_id("drums").notes[1].start == 150
    assert report["changed_notes"] >= 4
    assert source.track_by_id("drums").notes[1].start == 120


def test_velocity_and_articulation_are_bounded_integer_percentages():
    template = GrooveTemplate(120, (
        GrooveSlot(0, 0, 150, 200),
        GrooveSlot(1, 10, 50, 50),
    ))
    options = GrooveApplySettings()
    first = apply_groove_note(ScoreNote("n", 0, 100, 60, 80),
                              template, options, score_end=1000)
    second = apply_groove_note(ScoreNote("m", 120, 100, 60, 80),
                               template, options, score_end=1000)
    assert first.velocity == 120
    assert first.duration == 200
    assert second.velocity == 40
    assert second.duration == 50
    assert second.start == 130


def test_zero_strength_has_no_output_change():
    template = GrooveTemplate.swing(step_ticks=120, swing_percent=50)
    options = GrooveApplySettings(0, 0, 0)
    out, report = apply_groove(score(), template, options)
    assert all(a.notes == b.notes for a, b in zip(out.tracks, score().tracks))
    assert report["changed_notes"] == 0


def test_duration_preservation_ignores_articulation_slots():
    template = GrooveTemplate(120, (GrooveSlot(0, 0, 100, 300),))
    source = ScoreNote("n", 0, 100, 60)
    new = apply_groove_note(source, template,
                            GrooveApplySettings(preserve_duration=True),
                            score_end=500)
    assert new.duration == source.duration


def test_observed_groove_template_extracts_median_offsets():
    source = score()
    template, observations = extract_groove(source.track_by_id("drums"),
                                            step_ticks=120, slots=8)
    assert len(observations) == 8
    assert all(isinstance(ob, GrooveObservation) for ob in observations)
    assert template.steps[0].velocity_percent == 100
    assert all(slot.offset_ticks == 0 for slot in template.steps)
    assert sum(ob.count for ob in observations) == len(source.track_by_id("drums").notes)


def test_extraction_ignores_notes_far_from_candidate_grid():
    source = score()
    shifted = source.track_by_id("drums").replace_notes((
        ScoreNote("out", 55, 50, 36),
    ))
    template, observations = extract_groove(shifted, step_ticks=120,
                                            maximum_deviation=20)
    assert sum(item.count for item in observations) == 0
    assert all(slot.offset_ticks == 0 for slot in template.steps)


def test_clip_renders_repeated_original_pattern_in_target_track():
    source = score()
    clip = Clip("drums", 0, 480, repeats=3,
                offset_tick=1000, transpose=0, velocity_percent=50)
    output, report = render_clip(source, clip, target_track_id="drums")
    assert report["created_notes"] == 12
    assert len(output.track_by_id("drums").notes) == 20
    assert output.note_by_id("drums", "kick1_clip0").start == 1000
    assert output.note_by_id("drums", "kick1_clip1").start == 1480
    assert output.note_by_id("drums", "hat1_clip2").velocity == 50
    assert len(source.track_by_id("drums").notes) == 8


def test_clip_can_render_into_another_track():
    source = score()
    clip = Clip("drums", 0, 480, repeats=1, offset_tick=1400, transpose=12)
    output, _ = render_clip(source, clip, target_track_id="piano")
    assert output.note_by_id("piano", "kick1_clip0").pitch == 48
    assert output.track_by_id("drums") == source.track_by_id("drums")


def test_clip_selection_excludes_notes_overlapping_source_boundary():
    source = score()
    clip = Clip("piano", 100, 600, repeats=1, offset_tick=1300)
    output, report = render_clip(source, clip, target_track_id="piano")
    assert report["created_notes"] == 0
    assert output.track_by_id("piano").notes == source.track_by_id("piano").notes


def test_clip_end_beyond_score_fails_before_modification():
    source = score()
    clip = Clip("drums", 0, 960, repeats=3, offset_tick=1800)
    with pytest.raises(MidiWorkstationError):
        render_clip(source, clip, target_track_id="drums")


def test_clip_duplicate_note_id_collision_fails():
    source = score()
    clip = Clip("drums", 0, 480, repeats=1, offset_tick=1000)
    output, _ = render_clip(source, clip, target_track_id="drums")
    with pytest.raises(MidiWorkstationError):
        render_clip(output, clip, target_track_id="drums")


@pytest.mark.parametrize("bad", [
    dict(index=-1), dict(index=256), dict(offset_ticks=1025),
    dict(velocity_percent=0), dict(velocity_percent=301),
    dict(length_percent=0), dict(length_percent=301),
])
def test_invalid_groove_slot_rejected(bad):
    values = dict(index=0)
    values.update(bad)
    with pytest.raises(MidiWorkstationError):
        GrooveSlot(**values)


def test_groove_requires_canonical_contiguous_indices():
    with pytest.raises(MidiWorkstationError):
        GrooveTemplate(120, (GrooveSlot(0), GrooveSlot(2)))
    with pytest.raises(MidiWorkstationError):
        GrooveTemplate(120, (GrooveSlot(0, 100), GrooveSlot(1, -100)))


@pytest.mark.parametrize("bad", [
    dict(timing_strength=-1), dict(timing_strength=101),
    dict(velocity_strength=200), dict(length_strength=-1),
    dict(preserve_duration=1), dict(prevent_note_inversion=1),
])
def test_invalid_groove_apply_options_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        GrooveApplySettings(**bad)


@pytest.mark.parametrize("bad", [
    dict(source_track_id=""), dict(start_tick=-1),
    dict(stop_tick=0), dict(repeats=0), dict(repeats=257),
    dict(offset_tick=-1), dict(transpose=-128),
    dict(velocity_percent=301), dict(id_suffix=""),
])
def test_invalid_clip_geometry_rejected(bad):
    values = dict(source_track_id="drums", start_tick=0, stop_tick=480)
    values.update(bad)
    with pytest.raises(MidiWorkstationError):
        Clip(**values)


def test_reports_serialize_without_claiming_trained_quality():
    import json
    source = score()
    applied, report = apply_groove(source, GrooveTemplate.straight())
    json.dumps(report, allow_nan=False)
    assert report["model_inference"] is False
    assert report["source_mutated"] is False
