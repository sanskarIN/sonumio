"""Quantitative symbolic score measurements, note density and polyphony."""
import json
from dataclasses import replace

import pytest

from sonumio.midi_workstation.analysis import (
    PolyphonyPoint, PolyphonyStats, TrackMetrics, ScoreMetrics, TimeWindow,
    QuantizationStats, polyphony_timeline, quantization_statistics,
    analyze_windows, analyze_score, measure_track, note_density,
    score_inspection,
)
from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.tempo import TempoPoint, MeterPoint


def song():
    return Score("Original analysis", 480, (
        ScoreTrack("keys", "Keys", 0, 0, 0, 0, (
            ScoreNote("c", 0, 480, 60, 100),
            ScoreNote("e", 0, 480, 64, 90),
            ScoreNote("g", 0, 480, 67, 95),
            ScoreNote("d", 480, 240, 62, 92),
            ScoreNote("f", 720, 240, 65, 90),
        )),
        ScoreTrack("bass", "Bass", 1, 32, 0, 0, (
            ScoreNote("bass1", 0, 960, 36, 80),
        )),
    ), (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 960)


def test_constant_polyphony_sums_occupied_note_ticks():
    notes = (ScoreNote("a", 0, 100, 60), ScoreNote("b", 0, 100, 64))
    result = polyphony_timeline(notes, end=200)
    assert result.peak_voices == 2
    assert result.mean_voices == 1.
    assert result.voice_ticks == 200
    assert result.occupied_ticks == 100
    assert result.occupied_fraction == .5


def test_half_open_boundaries_do_not_count_phantom_overlap():
    notes = (ScoreNote("one", 0, 100, 60),
             ScoreNote("two", 100, 100, 60))
    result = polyphony_timeline(notes, end=200, include_changes=True)
    assert result.peak_voices == 1
    assert result.mean_voices == 1
    assert result.changes[-1].active_notes == 0


def test_empty_polyphony_is_finite_and_deterministic():
    result = polyphony_timeline((), end=100)
    assert result.peak_voices == 0
    assert result.voice_ticks == 0
    assert result.occupied_fraction == 0
    assert result.changes == ()


def test_multi_track_score_metrics_reflect_chord_polyphony():
    s = song()
    report = analyze_score(s)
    assert isinstance(report, ScoreMetrics)
    assert report.note_count == 6
    assert report.active_note_count == 6
    assert report.global_peak_polyphony == 4
    assert len(report.tracks) == 2
    assert report.tracks[0].peak_polyphony == 3
    assert report.tracks[1].peak_polyphony == 1
    assert report.duration_microseconds == 1000000
    assert report.pitch_classes[0] == 2
    assert report.as_dict()["musical_quality_established"] is False


def test_muted_track_is_excluded_until_explicitly_requested():
    s = song()
    muted = replace(s.tracks[1], muted=True)
    score = s.with_tracks((s.tracks[0], muted))
    standard = analyze_score(score)
    inclusive = analyze_score(score, include_muted=True)
    assert standard.active_note_count == 5
    assert inclusive.active_note_count == 6
    assert inclusive.global_peak_polyphony > standard.global_peak_polyphony


def test_window_chord_hypothesis_is_explicit_not_certified():
    s = song()
    windows = analyze_windows(s, window_ticks=480)
    assert len(windows) == 2
    assert windows[0].start == 0
    assert windows[0].stop == 480
    assert windows[0].notes_started == 4
    assert windows[0].notes_active == 4
    assert windows[0].unique_pitch_classes == 3
    assert windows[0].best_chord is not None
    assert windows[0].chord_score is not None
    assert windows[1].notes_started == 2


def test_high_density_windows_are_bounded():
    s = song()
    with pytest.raises(MidiLimitError):
        analyze_windows(replace(s, end=9600), window_ticks=1)


def test_note_density_sum_equals_note_count():
    s = song()
    for bins in (1, 2, 4, 10, 32, 64, 500):
        counts = note_density(s, bins=bins)
        assert len(counts) == bins
        assert sum(counts) == s.note_count
        assert all(x >= 0 for x in counts)


def test_track_metrics_include_empty_track_safely():
    empty = ScoreTrack("empty", "Empty", 2, 0, 0, 0, ())
    stats = measure_track(empty, score_end=1920)
    assert stats.pitch_min is None
    assert stats.pitch_max is None
    assert stats.median_velocity is None
    assert stats.notes == 0
    assert stats.coverage == 0


def test_quantization_error_metrics_are_observable():
    notes = (
        ScoreNote("a", 5, 115, 60),
        ScoreNote("b", 130, 125, 64),
        ScoreNote("c", 240, 120, 67),
    )
    result = quantization_statistics(notes, step_ticks=120)
    assert isinstance(result, QuantizationStats)
    assert result.note_count == 3
    assert result.aligned_onsets == 1
    assert result.max_absolute_onset_error == 10
    assert result.mean_absolute_onset_error == pytest.approx(5.)


def test_same_pitch_overlap_counts_pairs():
    s = Score("Overlaps", 480, (
        ScoreTrack("one", "One", 0, 0, 0, 0, (
            ScoreNote("a", 0, 200, 60),
            ScoreNote("b", 100, 200, 60),
            ScoreNote("c", 150, 100, 60),
        )),
    ), (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 500)
    assert analyze_score(s).overlapping_same_pitch_pairs == 3


def test_summary_is_stable_and_json_compatible():
    s = song()
    a = score_inspection(s, window_ticks=240)
    b = score_inspection(s, window_ticks=240)
    assert a == b
    encoded = json.dumps(a, sort_keys=True, allow_nan=False)
    assert s.sha256 in encoded
    assert a["model_inference"] is False
    assert a["subjective_quality_scored"] is False


@pytest.mark.parametrize("bad", [
    (0, False), (960, 1), (True, False),
])
def test_invalid_polyphony_timeline_arguments_rejected(bad):
    end, include = bad
    with pytest.raises(MidiWorkstationError):
        polyphony_timeline((), end=end, include_changes=include)


def test_unknown_metric_track_selection_rejected():
    with pytest.raises(MidiWorkstationError):
        analyze_windows(song(), track_ids=("unknown",))
    with pytest.raises(MidiWorkstationError):
        note_density(song(), bins=0)


def test_source_score_digest_does_not_change_after_reporting():
    s = song()
    before = s.sha256
    for _ in range(3):
        analyze_score(s)
        analyze_windows(s)
        score_inspection(s)
    assert s.sha256 == before
