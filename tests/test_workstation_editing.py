"""Digest-pinned atomic score transaction invariants and collisions."""
from dataclasses import replace
import pytest

from sonumio.midi_workstation.contracts import (
    MidiIntegrityError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.editing import (
    AddNote, RemoveNotes, ReplaceNote, ShiftNotes, TransposeNotes,
    ScaleNoteVelocities, ResizeNotes, DuplicateNotes, SplitNote, DeleteRegion,
    SetTrackFlags, RenameTrack, ChangePatch, AddTrack, RemoveTrack,
    SetTempo, RemoveTempo, SetMeter, EditTransaction,
    apply_transaction, transaction_for,
)
from sonumio.midi_workstation.score import (
    Score, ScoreTrack, ScoreNote, note_sort_key,
)
from sonumio.midi_workstation.tempo import TempoPoint, MeterPoint


def n(id, start, length=100, pitch=60, velocity=100):
    return ScoreNote(id, start, length, pitch, velocity)


def base():
    return Score("Edits", 480, (
        ScoreTrack("a", "Piano", 0, 0, 0, 0, (
            n("n0", 0), n("n1", 200), n("n2", 400),
        )),
        ScoreTrack("b", "Bass", 1, 32, 0, 0, (
            n("bass", 0, 500, 40),
        )),
    ), (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 1920)


def result(*ops):
    s = base()
    return apply_transaction(s, transaction_for(s, *ops))


def test_add_note_is_atomic_and_preserves_original():
    source = base()
    digest = source.sha256
    out, receipt = apply_transaction(source, transaction_for(
        source, AddNote("a", n("added", 120, 70, 67)),
    ))
    assert out.note_count == source.note_count + 1
    assert out.note_by_id("a", "added").pitch == 67
    assert source.sha256 == digest
    assert out.revision == source.revision + 1
    assert receipt.created_notes == 1
    assert receipt.deleted_notes == 0
    assert receipt.as_dict()["source_mutated"] is False


def test_remove_multiple_existing_notes():
    out, receipt = result(RemoveNotes("a", ("n0", "n2")))
    assert [n.id for n in out.track_by_id("a").notes] == ["n1"]
    assert receipt.deleted_notes == 2
    assert receipt.notes_after == 2


def test_replace_retains_stable_identifier():
    out, receipt = result(ReplaceNote("a", "n1", n("n1", 240, 140, 61)))
    assert out.note_by_id("a", "n1").start == 240
    assert receipt.altered_notes == 1


def test_shift_notes_applies_signed_integer_offsets():
    out, receipt = result(ShiftNotes("a", ("n1", "n2"), 30))
    assert out.note_by_id("a", "n0").start == 0
    assert out.note_by_id("a", "n1").start == 230
    assert out.note_by_id("a", "n2").start == 430
    assert receipt.altered_notes == 2


def test_transpose_notes_keeps_unselected_notes_unchanged():
    out, _ = result(TransposeNotes("a", ("n0", "n1"), 12))
    assert out.note_by_id("a", "n0").pitch == 72
    assert out.note_by_id("a", "n1").pitch == 72
    assert out.note_by_id("a", "n2").pitch == 60


def test_velocity_scale_uses_rational_saturation():
    out, _ = result(ScaleNoteVelocities("a", ("n0", "n1"), 3, 2))
    assert out.note_by_id("a", "n0").velocity == 127
    assert out.note_by_id("a", "n1").velocity == 127
    low, _ = result(ScaleNoteVelocities("a", ("n0",), 0, 1))
    assert low.note_by_id("a", "n0").velocity == 1


def test_note_resize_about_onset_and_release():
    a, _ = result(ResizeNotes("a", ("n1",), 2, 1))
    assert a.note_by_id("a", "n1").start == 200
    assert a.note_by_id("a", "n1").duration == 200
    b, _ = result(ResizeNotes("a", ("n1",), 2, 1, pivot="release"))
    assert b.note_by_id("a", "n1").start == 100
    assert b.note_by_id("a", "n1").stop == 300


def test_duplicate_notes_uses_explicit_suffix_and_offset():
    out, receipt = result(DuplicateNotes("a", ("n0", "n1"), 600, "repeat"))
    assert out.note_by_id("a", "n0_repeat").start == 600
    assert out.note_by_id("a", "n1_repeat").start == 800
    assert receipt.created_notes == 2
    assert len(out.track_by_id("a").notes) == 5


def test_note_split_produces_equal_union_of_intervals():
    out, receipt = result(SplitNote("a", "n1", 250, "n1-tail"))
    first = out.note_by_id("a", "n1")
    second = out.note_by_id("a", "n1-tail")
    assert (first.start, first.stop, second.start, second.stop) == (200, 250, 250, 300)
    assert receipt.created_notes == 1
    assert receipt.altered_notes == 1


def test_region_delete_with_ripple_updates_following_notes():
    out, _ = result(DeleteRegion("a", 150, 350, ripple=True))
    assert out.note_by_id("a", "n0").start == 0
    assert out.note_by_id("a", "n2").start == 200
    assert not any(note.id == "n1" for note in out.track_by_id("a").notes)


def test_region_split_keeps_tail_and_uses_new_stable_id():
    out, _ = result(DeleteRegion("b", 100, 300, ripple=False))
    first = out.note_by_id("b", "bass")
    second = out.note_by_id("b", "bass_right")
    assert (first.start, first.stop) == (0, 100)
    assert (second.start, second.stop) == (300, 500)


def test_region_refuses_cuts_when_disabled():
    s = base()
    with pytest.raises(MidiWorkstationError):
        apply_transaction(s, transaction_for(
            s, DeleteRegion("b", 100, 300, split_crossing=False),
        ))
    assert s.note_count == 4


def test_track_flags_and_rename_in_one_transaction():
    out, receipt = result(
        SetTrackFlags("a", True, False),
        RenameTrack("b", "Deep Bass"),
    )
    assert out.track_by_id("a").muted
    assert out.track_by_id("b").name == "Deep Bass"
    assert receipt.operations == 2
    assert out.active_track_ids == ("b",)


def test_changing_patch_does_not_modify_notes():
    out, receipt = result(ChangePatch("a", 12, 1, 3))
    assert out.track_by_id("a").patches == (1, 3, 12)
    assert out.track_by_id("a").notes == base().tracks[0].notes
    assert receipt.altered_notes == 0


def test_add_track_and_remove_track_within_one_atomic_transaction():
    fresh = ScoreTrack("c", "Plucked", 2, 24, 0, 0, (n("c1", 300),))
    out, receipt = result(AddTrack(fresh), RemoveTrack("b"))
    assert [t.id for t in out.tracks] == ["a", "c"]
    assert receipt.created_notes == 1
    assert receipt.deleted_notes == 1


def test_tempo_and_meter_changes_are_validated_at_commit():
    out, receipt = result(
        SetTempo(960, 400000),
        SetMeter(MeterPoint(960, 3, 4)),
    )
    assert len(out.tempo) == 2
    assert out.tempo_map.tempo_at(1000).microseconds_per_quarter == 400000
    assert out.meter_map.position(960).meter.numerator == 3
    assert receipt.operations == 2


def test_existing_nonzero_tempo_point_can_be_removed():
    source, _ = result(SetTempo(960, 400000))
    result2, _ = apply_transaction(source, transaction_for(source, RemoveTempo(960)))
    assert result2.tempo == (TempoPoint(0, 500000),)


def test_failed_edit_does_not_change_original_or_return_partial_score():
    s = base()
    digest = s.sha256
    tx = transaction_for(
        s, AddNote("a", n("valid", 600)),
        TransposeNotes("a", ("n0",), 100),
    )
    with pytest.raises(MidiWorkstationError):
        apply_transaction(s, tx)
    assert s.sha256 == digest
    assert s.note_count == 4


def test_optimistic_concurrency_rejects_changed_digest():
    s = base()
    tx = transaction_for(s, AddNote("a", n("new", 800)))
    updated = s.with_tracks(s.tracks)
    with pytest.raises(MidiIntegrityError):
        apply_transaction(updated, tx)


def test_optimistic_concurrency_rejects_changed_revision():
    s = base()
    tx = EditTransaction(s.sha256, (AddNote("a", n("new", 800)),), expected_revision=99)
    with pytest.raises(MidiIntegrityError):
        apply_transaction(s, tx)


def test_missing_track_fails_without_source_mutation():
    s = base()
    with pytest.raises(MidiWorkstationError):
        apply_transaction(s, transaction_for(s, RemoveNotes("missing", ("n0",))))
    assert s.note_count == 4


def test_missing_note_in_selection_fails():
    s = base()
    with pytest.raises(MidiWorkstationError):
        apply_transaction(s, transaction_for(s, ShiftNotes("a", ("none",), 100)))


def test_same_id_duplicate_create_is_rejected():
    s = base()
    with pytest.raises(MidiWorkstationError):
        apply_transaction(s, transaction_for(s, AddNote("a", n("n0", 700))))


def test_cannot_remove_final_track():
    s = Score.empty()
    with pytest.raises(MidiWorkstationError):
        apply_transaction(s, transaction_for(s, RemoveTrack("track-1")))


def test_duplicate_selected_note_ids_are_invalid():
    with pytest.raises(MidiWorkstationError):
        ShiftNotes("a", ("n0", "n0"), 100)


def test_split_tick_must_be_strictly_inside_note():
    s = base()
    for tick in (200, 300, 400):
        with pytest.raises(MidiWorkstationError):
            apply_transaction(s, transaction_for(s, SplitNote("a", "n1", tick, "other")))


def test_transaction_receipt_is_json_serializable():
    import json
    out, receipt = result(AddNote("a", n("new", 700)))
    saved = json.dumps(receipt.as_dict(), allow_nan=False)
    assert out.sha256 in saved
    assert receipt.as_dict()["operations_atomic"] is True
