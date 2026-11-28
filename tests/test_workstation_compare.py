"""Bounded exact score-pitch comparison and timing alignment diagnostics."""
from dataclasses import replace
import json

import pytest

from sonumio.midi_workstation.compare import CompareSettings, compare_scores
from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack


def make_score(notes=None, channel=0, name="track-1"):
    if notes is None:
        notes = (ScoreNote("a", 100, 80, 60),
                 ScoreNote("b", 300, 100, 64),
                 ScoreNote("c", 500, 100, 67))
    base = Score.empty(end=1920)
    track = ScoreTrack(name, name, channel, 0, 0, 0, notes)
    return replace(base, tracks=(track,))


def test_identical_scores_have_zero_error_and_no_missing_notes():
    a = make_score()
    result = compare_scores(a, a)
    assert result.matched_notes == 3
    assert result.missing_notes == 0
    assert result.additional_notes == 0
    assert result.mean_onset_error_ticks == 0
    assert result.maximum_release_error_ticks == 0
    assert result.mean_velocity_error == 0
    assert result.source_sha256 == a.sha256
    assert result.target_sha256 == a.sha256
    assert result.as_dict()["quality_calibrated"] is False


def test_note_shift_and_velocity_changes_report_signed_error():
    base = make_score()
    shifted = tuple(replace(note, start=note.start+20,
                            velocity=note.velocity+10)
                    for note in base.notes)
    target = make_score(shifted)
    result = compare_scores(base, target)
    assert result.matched_notes == 3
    assert result.mean_onset_error_ticks == 20
    assert result.maximum_onset_error_ticks == 20
    assert result.mean_release_error_ticks == 20
    assert result.mean_velocity_error == 10
    assert all(x.onset_difference == 20 for x in result.matches)
    assert all(x.velocity_difference == 10 for x in result.matches)


def test_missing_and_extra_note_counters():
    a = make_score()
    b = make_score((a.notes[0], a.notes[2],
                    ScoreNote("extra", 1300, 100, 72)))
    result = compare_scores(a, b)
    assert result.matched_notes == 2
    assert result.missing_notes == 1
    assert result.additional_notes == 1


def test_zero_tolerance_reports_only_exact_timing():
    a = make_score()
    b = make_score(tuple(replace(n, start=n.start+1) for n in a.notes))
    strict = CompareSettings(onset_tolerance_ticks=0, release_tolerance_ticks=0)
    report = compare_scores(a, b, strict)
    assert report.matched_notes == 0
    assert report.missing_notes == 3
    assert report.additional_notes == 3


def test_pitch_class_matching_can_report_octave_differences():
    a = make_score()
    b = make_score(tuple(replace(n, pitch=n.pitch+12) for n in a.notes))
    exact = compare_scores(a, b)
    classes = compare_scores(a, b, CompareSettings(pitch_mode="class"))
    assert exact.matched_notes == 0
    assert classes.matched_notes == 3
    assert classes.pitch_class_matches == 3
    assert classes.exact_pitch_matches == 0
    assert all(note.matching_pitch_class for note in classes.matches)


def test_route_matching_requires_same_track_by_default():
    original = make_score(name="melody")
    renamed = make_score(name="lead")
    by_track = compare_scores(original, renamed)
    by_channel = compare_scores(original, renamed,
                                CompareSettings(route_mode="channel"))
    assert by_track.matched_notes == 0
    assert by_channel.matched_notes == 3


def test_source_unchanged_and_deterministic_matching():
    source = make_score()
    target = make_score(tuple(replace(n, start=n.start+30) for n in source.notes))
    before = source.sha256
    a = compare_scores(source, target)
    b = compare_scores(source, target)
    assert a == b
    assert source.sha256 == before
    json.dumps(a.as_dict(), allow_nan=False)
    assert a.as_dict()["model_inference"] is False


def test_score_ppqn_mismatch_is_explicit_error():
    a = make_score()
    b = replace(a, ppqn=960)
    with pytest.raises(MidiWorkstationError):
        compare_scores(a, b)


def test_out_of_tolerance_note_does_not_pair_with_unrelated_pulse():
    a = make_score((ScoreNote("n", 100, 60, 60),))
    b = make_score((ScoreNote("n", 1600, 60, 60),))
    result = compare_scores(a, b)
    assert result.matched_notes == 0
    assert result.missing_notes == 1
    assert result.additional_notes == 1


def test_one_note_can_only_be_matched_once():
    a = make_score((ScoreNote("a", 100, 100, 60),))
    b = make_score((ScoreNote("b", 101, 100, 60),
                    ScoreNote("c", 105, 100, 60)))
    result = compare_scores(a, b)
    assert result.matched_notes == 1
    assert result.additional_notes == 1
    assert result.matches[0].target_id in ("b", "c")


def test_dynamic_programming_resolves_interleaved_repeated_notes():
    original = make_score(tuple(
        ScoreNote(f"note{i}", i * 100, 40, 60) for i in range(12)
    ))
    later = make_score(tuple(
        ScoreNote(f"other{i}", i * 100 + 20, 40, 60) for i in range(12)
    ))
    result = compare_scores(original, later)
    assert result.matched_notes == 12
    assert len({m.target_id for m in result.matches}) == 12
    assert result.mean_onset_error_ticks == 20


def test_large_same_pitch_group_is_preflight_rejected():
    base = Score.empty(end=4000)
    notes = tuple(ScoreNote(f"n{i}", i*2, 1, 60) for i in range(1500))
    original = replace(base, tracks=(base.tracks[0].replace_notes(notes),))
    with pytest.raises(MidiLimitError):
        compare_scores(original, original)


def test_percussion_octave_matching_rejected_in_channel_mode():
    source = make_score(channel=9)
    with pytest.raises(MidiWorkstationError):
        compare_scores(source, source,
                       CompareSettings(route_mode="channel", pitch_mode="class"))


@pytest.mark.parametrize("bad", [
    dict(onset_tolerance_ticks=-1), dict(onset_tolerance_ticks=10000),
    dict(release_tolerance_ticks=-1), dict(maximum_matches=0),
    dict(pitch_mode="fuzzy"), dict(route_mode="unknown"),
    dict(include_muted=1),
])
def test_invalid_compare_options_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        CompareSettings(**bad)


def test_invalid_inputs_fail_without_network_or_mutation():
    original = make_score()
    with pytest.raises(MidiWorkstationError):
        compare_scores(None, original)
    with pytest.raises(MidiWorkstationError):
        compare_scores(original, None)
    with pytest.raises(MidiWorkstationError):
        compare_scores(original, original, None)


def test_match_report_cap_does_not_silently_truncate():
    source = make_score()
    with pytest.raises(MidiLimitError):
        compare_scores(source, source, CompareSettings(maximum_matches=2))


def test_score_without_notes_has_empty_evidence():
    source = make_score(())
    result = compare_scores(source, source)
    assert result.matched_notes == 0
    assert result.source_notes == 0
    assert result.target_notes == 0
    assert result.mean_onset_error_ticks == 0
    assert not result.matches
