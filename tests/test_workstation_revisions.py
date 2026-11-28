"""Conflict-aware symbolic score revisions and three-way merge tests."""
from dataclasses import replace

import pytest

from sonumio.midi_workstation.contracts import MidiIntegrityError, MidiWorkstationError
from sonumio.midi_workstation.revisions import diff_scores, merge_scores
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack


def example():
    note = ScoreNote("n1", 10, 50, 60)
    track = ScoreTrack("a", "Piano", 0, 0, 0, 0, (note,))
    return replace(Score.empty(), tracks=(track,))


def modified(score, field, value):
    return replace(score, **{field: value}, revision=score.revision + 1)


def with_note(score, note):
    track = score.tracks[0].replace_notes((note,))
    return score.with_tracks((track,))


def with_second(score, note):
    track = score.tracks[0].replace_notes(score.tracks[0].notes + (note,))
    return score.with_tracks((track,))


def test_identical_diff_is_empty_and_nonmutating():
    old = example()
    result = diff_scores(old, old)
    assert not result.changes
    assert result.from_digest == old.sha256
    assert result.to_digest == old.sha256
    assert result.as_dict()["remote_writes"] is False


@pytest.mark.parametrize("field,value", [
    ("title", "Song B"), ("end", 2400),
])
def test_top_level_changes_have_lexical_paths(field, value):
    base = example()
    change = modified(base, field, value)
    diff = diff_scores(base, change)
    assert [item.location for item in diff.changes] == [field]
    assert diff.notes_updated == 0


def test_added_note_and_deleted_note_are_differentiated():
    base = example()
    added = with_second(base, ScoreNote("n2", 80, 30, 72))
    result = diff_scores(base, added)
    assert result.notes_added == 1
    assert result.notes_removed == 0
    assert any("notes/n2" in item.location for item in result.changes)
    gone = with_note(added, added.tracks[0].notes[-1])
    result = diff_scores(added, gone)
    assert result.notes_removed == 1
    assert result.notes_added == 0


def test_change_existing_note_counts_as_update():
    base = example()
    after = with_note(base, replace(base.notes[0], velocity=91))
    report = diff_scores(base, after)
    assert report.notes_updated == 1
    assert report.notes_added == 0
    assert report.changes[0].new["velocity"] == 91


def test_independent_note_changes_merge_without_conflict():
    base = with_second(example(), ScoreNote("n2", 100, 20, 65))
    ours = base.with_tracks((base.tracks[0].replace_notes((replace(base.notes[0], velocity=70), base.notes[1])),))
    theirs = base.with_tracks((base.tracks[0].replace_notes((base.notes[0], replace(base.notes[1], velocity=40))),))
    result = merge_scores(base, ours, theirs)
    assert result.succeeded
    assert result.score.note_by_id("a", "n1").velocity == 70
    assert result.score.note_by_id("a", "n2").velocity == 40
    assert result.result_digest == result.score.sha256
    assert base.notes[0].velocity == 100


def test_same_note_identical_edits_are_shared():
    base = example()
    next_score = with_note(base, replace(base.notes[0], pitch=66))
    result = merge_scores(base, next_score, next_score)
    assert result.succeeded
    assert result.shared_changes >= 1
    assert result.score.notes[0].pitch == 66


@pytest.mark.parametrize("a,b", [(65, 67), (64, 60), (70, 77)])
def test_same_note_divergent_edits_conflict(a, b):
    base = example()
    ours = with_note(base, replace(base.notes[0], pitch=a))
    theirs = with_note(base, replace(base.notes[0], pitch=b))
    result = merge_scores(base, ours, theirs)
    if a == 60 or b == 60:
        assert result.succeeded
    else:
        assert not result.succeeded
        assert result.score is None
        assert any("notes/n1" in c.path for c in result.conflicts)


def test_independent_additions_and_top_level_metadata_merge():
    base = example()
    ours = with_second(base, ScoreNote("new", 90, 40, 72))
    theirs = modified(base, "title", "Updated title")
    result = merge_scores(base, ours, theirs)
    assert result.succeeded
    assert result.score.title == "Updated title"
    assert result.score.note_count == 2
    assert result.score.revision == 2


def test_conflicting_new_note_ids_fails_closed():
    base = example()
    ours = with_second(base, ScoreNote("n2", 100, 20, 72))
    theirs = with_second(base, ScoreNote("n2", 100, 20, 66))
    result = merge_scores(base, ours, theirs)
    assert result.score is None
    assert result.conflicts[0].path.endswith("notes/n2")


def test_conflicting_score_clock_is_not_implicitly_rescaled():
    base = example()
    ours = modified(base, "ppqn", 960)
    theirs = modified(base, "ppqn", 720)
    result = merge_scores(base, ours, theirs)
    assert not result.succeeded
    assert result.conflicts[0].path == "ppqn"


def test_deletion_vs_independent_note_change_conflicts():
    base = example()
    ours = replace(base, tracks=(base.tracks[0].replace_notes(()),), revision=1)
    theirs = with_note(base, replace(base.notes[0], velocity=50))
    result = merge_scores(base, ours, theirs)
    assert not result.succeeded
    assert result.conflicts[0].path.endswith("notes/n1")


def test_track_deletion_vs_edit_conflicts():
    original = example()
    second = ScoreTrack("b", "Drums", 9, 0, 0, 0, ())
    base = original.with_tracks(original.tracks + (second,))
    ours = base.with_tracks((base.tracks[0],))
    theirs = base.with_tracks((base.tracks[0], replace(base.tracks[1], name="Kit")))
    result = merge_scores(base, ours, theirs)
    assert not result.succeeded
    assert result.conflicts[0].path == "tracks/b"


def test_track_deletion_vs_noop_is_allowed():
    original = example()
    second = ScoreTrack("b", "Kit", 9, 0, 0, 0, ())
    base = original.with_tracks(original.tracks + (second,))
    ours = base.with_tracks((base.tracks[0],))
    result = merge_scores(base, ours, base)
    assert result.succeeded
    assert len(result.score.tracks) == 1


def test_conflicting_track_properties_rejected():
    base = example()
    ours = base.with_tracks((replace(base.tracks[0], name="First"),))
    theirs = base.with_tracks((replace(base.tracks[0], name="Second"),))
    result = merge_scores(base, ours, theirs)
    assert not result.succeeded
    assert result.conflicts[0].path.endswith("properties")


def test_invalid_score_inputs_rejected():
    base = example()
    with pytest.raises(MidiWorkstationError):
        diff_scores(base, None)
    with pytest.raises(MidiWorkstationError):
        merge_scores(base, None, base)
    with pytest.raises(MidiWorkstationError):
        merge_scores(base, base, base, strict_revision=1)


def test_strict_revision_lease_prevents_stale_branch():
    base = replace(example(), revision=9)
    ours = replace(base, revision=8)
    with pytest.raises(MidiIntegrityError):
        merge_scores(base, ours, base, strict_revision=True)


def test_conflict_receipt_is_json_safe_and_side_effect_free():
    import json
    base = example()
    ours = with_note(base, replace(base.notes[0], pitch=70))
    theirs = with_note(base, replace(base.notes[0], pitch=71))
    result = merge_scores(base, ours, theirs)
    json.dumps(result.as_dict(), allow_nan=False)
    assert result.as_dict()["no_external_mutation"] is True
    assert base.notes[0].pitch == 60
