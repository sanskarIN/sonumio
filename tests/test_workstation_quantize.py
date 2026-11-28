"""Sample-clock-independent symbolic timing correction and controlled variation."""
import random

import pytest

from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.quantize import (
    QuantizeGrid, QuantizeReport, HumanizeSettings,
    quantize_note, quantize_score, humanize_score,
)
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack, note_sort_key
from sonumio.midi_workstation.tempo import TempoPoint, MeterPoint


def song():
    a = (
        ScoreNote("a1", 7, 117, 60, 95),
        ScoreNote("a2", 123, 116, 64, 90),
        ScoreNote("a3", 241, 123, 67, 97),
        ScoreNote("a4", 365, 115, 72, 102),
        ScoreNote("a5", 481, 110, 60, 107),
    )
    b = (
        ScoreNote("b1", 0, 240, 36, 80),
        ScoreNote("b2", 240, 240, 40, 81),
    )
    return Score("Symbolic timing", 480, (
        ScoreTrack("keys", "Keys", 0, 0, 0, 0, a),
        ScoreTrack("bass", "Bass", 1, 32, 0, 0, b),
    ), (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 1920)


@pytest.mark.parametrize("tick,expected", [
    (0, 0), (1, 0), (59, 0), (60, 0), (61, 120),
    (118, 120), (120, 120), (179, 120), (181, 240),
    (240, 240), (359, 360), (480, 480),
])
def test_nearest_straight_grid_anchor(tick, expected):
    assert QuantizeGrid(120).anchor(tick) == expected


@pytest.mark.parametrize("step", [24, 60, 120, 240, 480])
def test_zero_swing_grid_is_uniform(step):
    grid = QuantizeGrid(step)
    values = [grid.tick_at_index(i) for i in range(30)]
    assert values == [i * step for i in range(30)]


def test_swing_delays_only_odd_grid_positions():
    straight = QuantizeGrid(120)
    swing = QuantizeGrid(120, swing_percent=50)
    assert straight.tick_at_index(1) == 120
    assert swing.tick_at_index(1) == 150
    assert swing.tick_at_index(2) == 240
    assert swing.tick_at_index(3) == 390


def test_strength_zero_is_bitwise_original_note():
    source = ScoreNote("n", 113, 70, 60, 90)
    assert quantize_note(source, QuantizeGrid(120, strength_percent=0)) == source


def test_strength_half_interpolates_quantized_timing():
    source = ScoreNote("n", 110, 70, 60, 90)
    adjusted = quantize_note(source, QuantizeGrid(120, strength_percent=50))
    assert adjusted.start == 115
    assert adjusted.duration == source.duration


def test_release_quantization_recomputes_safely():
    source = ScoreNote("n", 20, 150, 60, 90)
    adjusted = quantize_note(source, QuantizeGrid(120, quantize_releases=True))
    assert adjusted.start == 0
    assert adjusted.stop == 120


def test_quantization_preserves_score_and_selected_tracks():
    source = song()
    before = source.sha256
    result, report = quantize_score(source, QuantizeGrid(120), tracks=("keys",))
    assert result.track_by_id("bass") == source.track_by_id("bass")
    assert result.track_by_id("keys").notes[0].start == 0
    assert report.changed_notes > 0
    assert report.changed_onsets > 0
    assert report.original_sha256 == before
    assert source.sha256 == before
    assert result.revision == source.revision + 1
    assert report.as_dict()["model_inference"] is False


def test_quantize_completely_aligned_score_is_stable_except_revision():
    source = song()
    once, _ = quantize_score(source, QuantizeGrid(120))
    twice, report = quantize_score(once, QuantizeGrid(120))
    assert [t.notes for t in once.tracks] == [t.notes for t in twice.tracks]
    assert report.changed_notes == 0
    assert twice.revision == once.revision + 1


def test_extreme_grid_can_reject_unfittable_note():
    note = ScoreNote("late", 9, 1, 60)
    with pytest.raises(MidiWorkstationError):
        quantize_note(note, QuantizeGrid(120, offset_ticks=119, min_note_ticks=2), score_end=10)


def test_quantization_cannot_move_note_past_score_end():
    source = ScoreNote("late", 1919, 1, 60)
    with pytest.raises(MidiWorkstationError):
        quantize_note(source, QuantizeGrid(120), score_end=1920)


def test_humanization_is_deterministic_by_seed_and_note_id():
    source = song()
    cfg = HumanizeSettings(timing_ticks=9, velocity_units=20,
                           duration_ticks=12, seed=777)
    a, report_a = humanize_score(source, cfg)
    b, report_b = humanize_score(source, cfg)
    assert a == b
    assert report_a == report_b
    assert a.sha256 != source.sha256
    assert source.notes[0].start == 7
    assert report_a["model_inference"] is False


def test_different_seeds_can_produce_independent_expression():
    source = song()
    a, _ = humanize_score(source, HumanizeSettings(seed=100))
    b, _ = humanize_score(source, HumanizeSettings(seed=101))
    assert a.tracks != b.tracks


def test_zero_humanization_leaves_note_data_unchanged():
    source = song()
    output, report = humanize_score(source, HumanizeSettings(
        timing_ticks=0, velocity_units=0, duration_ticks=0,
    ))
    assert [t.notes for t in output.tracks] == [t.notes for t in source.tracks]
    assert report["changed"] == 0


def test_humanize_keeps_simultaneous_chord_attacks():
    s = song()
    notes = (
        ScoreNote("c1", 200, 90, 60),
        ScoreNote("c2", 200, 90, 64),
        ScoreNote("c3", 200, 90, 67),
    )
    track = s.track_by_id("keys").replace_notes(notes)
    chord = s.with_tracks((track, s.track_by_id("bass")))
    output, _ = humanize_score(chord, HumanizeSettings(
        timing_ticks=10, preserve_simultaneous_chords=True, seed=1234,
    ))
    assert len({n.start for n in output.track_by_id("keys").notes}) == 1


def test_humanize_velocity_never_goes_outside_midi_range():
    source = song()
    for seed in range(20):
        out, _ = humanize_score(source, HumanizeSettings(
            timing_ticks=80, velocity_units=64, duration_ticks=50, seed=seed,
        ))
        assert all(1 <= n.velocity <= 127 for n in out.notes)
        assert all(1 <= n.duration and n.stop <= source.end for n in out.notes)


@pytest.mark.parametrize("bad", [
    dict(step_ticks=0), dict(step_ticks=True),
    dict(step_ticks=120, offset_ticks=120),
    dict(step_ticks=120, swing_percent=80),
    dict(step_ticks=120, strength_percent=101),
    dict(step_ticks=120, quantize_releases=1),
    dict(step_ticks=120, min_note_ticks=0),
])
def test_invalid_quantize_grid_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        QuantizeGrid(**bad)


@pytest.mark.parametrize("bad", [
    dict(timing_ticks=-1), dict(timing_ticks=5000),
    dict(velocity_units=65), dict(duration_ticks=-1),
    dict(seed=-1), dict(seed=True),
    dict(preserve_simultaneous_chords=1),
    dict(minimum_duration=0),
])
def test_invalid_humanization_settings_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        HumanizeSettings(**bad)


def test_invalid_selected_track_is_rejected_before_processing():
    source = song()
    with pytest.raises(MidiWorkstationError):
        quantize_score(source, QuantizeGrid(120), tracks=("missing",))
    with pytest.raises(MidiWorkstationError):
        humanize_score(source, HumanizeSettings(), tracks=("missing",))
