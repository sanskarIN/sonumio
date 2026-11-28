"""Piano-roll viewport, rectangles, source-digest gestures and safe clipboard."""
from dataclasses import replace
import json

import pytest

from sonumio.midi_workstation.contracts import (
    MidiIntegrityError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.pianoroll import (
    RollViewport, RollGesture, view_piano_roll, rectangular_selection,
    apply_roll_gesture, copy_roll_notes, paste_roll_notes,
)
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack


def score():
    base = Score.empty(end=2880)
    notes = (ScoreNote("a", 120, 240, 60),
             ScoreNote("b", 240, 480, 64),
             ScoreNote("c", 840, 240, 67),
             ScoreNote("d", 1500, 240, 72))
    return base.with_tracks((base.tracks[0].replace_notes(notes),))


def test_viewport_exposes_clipped_edges_and_source_sha():
    source = score()
    view = view_piano_roll(source, RollViewport(300, 900))
    assert view.visible_note_count == 3
    assert view.source_note_count == 4
    assert [box.note_id for box in view.boxes] == ["a", "b", "c"]
    assert view.boxes[0].hidden_left is True
    assert view.boxes[1].hidden_left is True
    assert view.boxes[-1].hidden_right is True
    assert view.source_sha256 == source.sha256
    assert view.as_dict()["no_pixel_coordinates_inferred"] is True
    json.dumps(view.as_dict(), allow_nan=False)


def test_pitch_range_filter_is_inclusive():
    source = score()
    view = view_piano_roll(source, RollViewport(0, 2880, 64, 67))
    assert [box.pitch for box in view.boxes] == [64, 67]


def test_viewport_is_invariant_to_selected_note_order():
    source = score()
    first = view_piano_roll(source, RollViewport(0, 2880),
                            selection=(("track-1", "a"), ("track-1", "c")))
    second = view_piano_roll(source, RollViewport(0, 2880),
                             selection=(("track-1", "c"), ("track-1", "a")))
    assert first == second
    assert [box.note_id for box in first.boxes if box.selected] == ["a", "c"]


def test_rectangular_selection_modes_are_explicit():
    source = score()
    view = view_piano_roll(source, RollViewport(0, 2880))
    overlaps = rectangular_selection(view, first_tick=300, last_tick=900,
                                     lowest_pitch=60, highest_pitch=72)
    contained = rectangular_selection(view, first_tick=300, last_tick=900,
                                      lowest_pitch=60, highest_pitch=72,
                                      mode="contained")
    onsets = rectangular_selection(view, first_tick=300, last_tick=900,
                                   lowest_pitch=60, highest_pitch=72,
                                   mode="onset")
    assert {id for _, id in overlaps} == {"a", "b", "c"}
    assert contained == ()
    assert onsets == (("track-1", "c"),)


def test_transposition_and_drag_are_atomic_and_bound_by_sha():
    source = score()
    gesture = RollGesture(
        source.sha256, (("track-1", "a"), ("track-1", "b")),
        shift_ticks=120, transpose_semitones=2, velocity_delta=-5,
    )
    updated, report = apply_roll_gesture(source, gesture)
    assert [(n.start, n.pitch, n.velocity) for n in updated.notes[:2]] == [
        (240, 62, 95), (360, 66, 95),
    ]
    assert source.notes[0].start == 120
    assert report.changed_notes == 2
    assert report.as_dict()["source_mutated"] is False
    with pytest.raises(MidiIntegrityError):
        apply_roll_gesture(updated, gesture)


def test_resize_edges_preserves_positive_duration():
    source = score()
    gesture = RollGesture(source.sha256, (("track-1", "a"),),
                          resize_left_ticks=60, resize_right_ticks=90)
    changed, _ = apply_roll_gesture(source, gesture)
    assert changed.notes[0].start == 180
    assert changed.notes[0].duration == 270


def test_piano_roll_rejects_invalid_resize_and_overflow():
    source = score()
    for updates in (
        {"resize_left_ticks": 500},
        {"shift_ticks": -300},
        {"shift_ticks": 2850},
        {"transpose_semitones": 100},
        {"velocity_delta": 100},
    ):
        gesture = RollGesture(source.sha256, (("track-1", "a"),), **updates)
        with pytest.raises(MidiWorkstationError):
            apply_roll_gesture(source, gesture)


def test_clamped_velocity_allows_saturating_gesture():
    source = score()
    gesture = RollGesture(source.sha256, (("track-1", "a"),),
                          velocity_delta=100, clamp_velocity=True)
    result, _ = apply_roll_gesture(source, gesture)
    assert result.notes[0].velocity == 127


def test_unknown_selected_note_fails_before_any_score_change():
    source = score()
    gesture = RollGesture(source.sha256, (("track-1", "missing"),))
    with pytest.raises(MidiWorkstationError):
        apply_roll_gesture(source, gesture)
    assert source.note_count == 4


def test_copy_paste_tracks_stable_relative_note_intervals():
    source = score()
    clipboard = copy_roll_notes(source, (("track-1", "a"), ("track-1", "b")))
    assert clipboard.origin_tick == 120
    assert clipboard.origin_pitch == 60
    updated, report = paste_roll_notes(source, clipboard,
                                       destination_tick=1700,
                                       destination_pitch=72)
    assert updated.note_count == 6
    assert updated.note_by_id("track-1", "a-paste").start == 1700
    assert updated.note_by_id("track-1", "a-paste").pitch == 72
    assert updated.note_by_id("track-1", "b-paste").start == 1820
    assert updated.note_by_id("track-1", "b-paste").pitch == 76
    assert report["created"] == 2
    assert source.note_count == 4


def test_copy_paste_id_collision_is_not_silently_resolved():
    source = score()
    clipboard = copy_roll_notes(source, (("track-1", "a"),))
    first, _ = paste_roll_notes(source, clipboard, destination_tick=1000,
                                destination_pitch=60)
    with pytest.raises(MidiWorkstationError):
        paste_roll_notes(first, clipboard, destination_tick=1400,
                         destination_pitch=65)


def test_copy_paste_bounds_reject_overflow_without_source_change():
    source = score()
    clipboard = copy_roll_notes(source, (("track-1", "b"),))
    with pytest.raises(MidiWorkstationError):
        paste_roll_notes(source, clipboard, destination_tick=2800,
                         destination_pitch=80)
    larger = copy_roll_notes(source, (("track-1", "b"), ("track-1", "c")))
    with pytest.raises(MidiWorkstationError):
        paste_roll_notes(source, larger, destination_tick=2000,
                         destination_pitch=127)


def test_empty_and_duplicate_clipboard_selections_are_rejected():
    source = score()
    with pytest.raises(MidiLimitError):
        copy_roll_notes(source, ())
    with pytest.raises(MidiWorkstationError):
        copy_roll_notes(source, (("track-1", "a"), ("track-1", "a")))
    with pytest.raises(MidiWorkstationError):
        copy_roll_notes(source, (("track-1", "missing"),))


def test_viewport_does_not_materialize_unbounded_note_collection():
    source = score()
    with pytest.raises(MidiLimitError):
        view_piano_roll(source, RollViewport(0, 2880, maximum_visible=1))


def test_track_filters_and_selection_are_validated():
    source = score()
    with pytest.raises(MidiWorkstationError):
        view_piano_roll(source, RollViewport(0, 2880,
                                             track_ids=("missing",)))
    with pytest.raises(MidiWorkstationError):
        view_piano_roll(source, RollViewport(0, 2880),
                        selection=(("not-a-track", "a"),))
    muted = replace(source.tracks[0], muted=True)
    source = source.with_tracks((muted,))
    view = view_piano_roll(source, RollViewport(0, 2880,
                                               include_muted=False))
    assert not view.boxes


@pytest.mark.parametrize("options", [
    dict(first_tick=-1, last_tick=100),
    dict(first_tick=100, last_tick=100),
    dict(first_tick=100, last_tick=200, lowest_pitch=90, highest_pitch=60),
    dict(first_tick=100, last_tick=200, maximum_visible=0),
    dict(first_tick=100, last_tick=200, include_muted=1),
    dict(first_tick=100, last_tick=200, track_ids=("a", "a")),
])
def test_invalid_viewport_contracts(options):
    with pytest.raises(MidiWorkstationError):
        RollViewport(**options)


def test_roll_digest_prevents_stale_drag_replay():
    original = score()
    gesture = RollGesture(original.sha256, (("track-1", "b"),), shift_ticks=20)
    first, _ = apply_roll_gesture(original, gesture)
    with pytest.raises(MidiIntegrityError):
        apply_roll_gesture(first, gesture)
    assert first.note_by_id("track-1", "b").start == 260
