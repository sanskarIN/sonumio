"""Snapshot-based undo/redo, stale lease rejection and journal tamper proofs."""
from dataclasses import replace
import json

import pytest

from sonumio.midi_workstation.contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.journal import RevisionJournal
from sonumio.midi_workstation.score import Score, ScoreNote


def initial():
    source = Score.empty(end=1920)
    return source.with_tracks((source.tracks[0].replace_notes((
        ScoreNote("first", 100, 200, 60),
    )),))


def revision(score, pitch):
    note = replace(score.notes[0], pitch=pitch)
    return score.with_tracks((score.tracks[0].replace_notes((note,)),))


def history():
    original = initial()
    one = revision(original, 64)
    two = revision(one, 67)
    journal = RevisionJournal.begin(original)
    journal = journal.commit(one, "Transpose to E", expected_current_sha256=original.sha256)
    journal = journal.commit(two, "Transpose to G", expected_current_sha256=one.sha256)
    return journal


def test_beginning_has_only_one_entry_and_no_undo_or_redo():
    original = initial()
    journal = RevisionJournal.begin(original)
    assert journal.current == original
    assert journal.current_digest == original.sha256
    assert not journal.can_undo
    assert not journal.can_redo
    assert journal.cursor == 0
    assert len(journal.entries) == 1
    assert journal.entries[0].canonical_bytes > 0


def test_commit_and_undo_redo_preserve_all_score_snapshots():
    journal = history()
    assert len(journal.entries) == 3
    assert journal.cursor == 2
    assert journal.current.notes[0].pitch == 67
    undone = journal.undo()
    assert undone.current.notes[0].pitch == 64
    again = undone.undo()
    assert again.current.notes[0].pitch == 60
    assert again.redo().redo() == journal
    assert journal.current.notes[0].pitch == 67
    assert journal.summary()["model_inference"] is False


def test_commit_from_undone_cursor_disposes_redo_branch():
    source = history()
    undone = source.undo()
    next_score = revision(undone.current, 72)
    branched = undone.commit(next_score, "New branch", expected_current_sha256=undone.current_digest)
    assert len(branched.entries) == 3
    assert branched.current.notes[0].pitch == 72
    assert not branched.can_redo
    assert source.current.notes[0].pitch == 67
    assert branched.entries[-1].previous_digest == branched.entries[-2].entry_digest


def test_exact_score_and_hash_chain_roundtrip():
    original = history()
    raw = original.to_bytes()
    recreated = RevisionJournal.from_bytes(raw)
    assert recreated == original
    assert recreated.to_bytes() == raw
    assert recreated.entries[-1].entry_digest == original.entries[-1].entry_digest
    assert recreated.summary()["total_canonical_bytes"] == original.total_canonical_bytes


def test_noncanonical_whitespace_is_permitted_but_digest_chain_verified():
    original = history()
    source = json.dumps(json.loads(original.to_bytes()), indent=3).encode()
    output = RevisionJournal.from_bytes(source)
    assert output == original
    assert output.to_bytes() == original.to_bytes()


def test_tampered_score_note_pitch_is_rejected_even_when_structurally_valid():
    payload = json.loads(history().to_bytes())
    payload["entries"][1]["score"]["tracks"][0]["notes"][0]["pitch"] = 75
    with pytest.raises(MidiIntegrityError):
        RevisionJournal.from_bytes(json.dumps(payload).encode())


def test_tampered_predecessor_link_is_rejected():
    payload = json.loads(history().to_bytes())
    payload["entries"][2]["previous_digest"] = "0" * 64
    with pytest.raises(MidiIntegrityError):
        RevisionJournal.from_bytes(json.dumps(payload).encode())


def test_tampered_entry_checksum_is_rejected():
    payload = json.loads(history().to_bytes())
    payload["entries"][1]["entry_digest"] = "0" * 64
    with pytest.raises(MidiIntegrityError):
        RevisionJournal.from_bytes(json.dumps(payload).encode())


def test_entry_label_is_digest_bound():
    payload = json.loads(history().to_bytes())
    payload["entries"][1]["label"] = "Different description"
    with pytest.raises(MidiIntegrityError):
        RevisionJournal.from_bytes(json.dumps(payload).encode())


def test_stale_commit_and_undo_leases_are_rejected():
    journal = history()
    with pytest.raises(MidiIntegrityError):
        journal.commit(revision(journal.current, 72), "Bad", expected_current_sha256="0" * 64)
    with pytest.raises(MidiIntegrityError):
        journal.undo(expected_current_sha256="0" * 64)
    with pytest.raises(MidiIntegrityError):
        journal.undo().redo(expected_current_sha256="0" * 64)


def test_undo_and_redo_bounds_are_explicit():
    first = RevisionJournal.begin(initial())
    with pytest.raises(MidiWorkstationError):
        first.undo()
    with pytest.raises(MidiWorkstationError):
        first.redo()
    with pytest.raises(MidiWorkstationError):
        history().redo()


def test_revision_cannot_move_backwards():
    original = initial()
    journal = RevisionJournal.begin(original)
    with pytest.raises(MidiWorkstationError):
        journal.commit(original, "unchanged", expected_current_sha256=original.sha256)
    future = replace(original, revision=0)
    with pytest.raises(MidiWorkstationError):
        journal.commit(future, "older", expected_current_sha256=original.sha256)


def test_nonconsecutive_score_revision_is_allowed_if_monotone():
    source = initial()
    far = replace(revision(source, 77), revision=source.revision+100)
    journal = RevisionJournal.begin(source)
    updated = journal.commit(far, "External validated edit", expected_current_sha256=source.sha256)
    assert updated.current.revision == source.revision + 100


def test_seek_does_not_mutate_other_journal_cursors():
    original = history()
    first = original.seek(0)
    second = original.seek(1)
    assert first.current.notes[0].pitch == 60
    assert second.current.notes[0].pitch == 64
    assert original.current.notes[0].pitch == 67
    with pytest.raises(MidiWorkstationError):
        original.seek(100)


def test_invalid_journal_protocol_and_unknown_fields_rejected():
    payload = json.loads(history().to_bytes())
    payload["protocol"] = "future-version"
    with pytest.raises(MidiFormatError):
        RevisionJournal.from_bytes(json.dumps(payload).encode())
    payload["protocol"] = "sonumio-workstation-revision-journal-v1"
    payload["execute"] = "forbidden"
    with pytest.raises(MidiFormatError):
        RevisionJournal.from_bytes(json.dumps(payload).encode())


def test_duplicate_keys_and_nonfinite_json_rejected():
    with pytest.raises(MidiFormatError):
        RevisionJournal.from_bytes(b'{"cursor":0,"cursor":1}')
    with pytest.raises(MidiFormatError):
        RevisionJournal.from_bytes(b'{"cursor":NaN}')


def test_reject_non_list_and_empty_history():
    payload = json.loads(history().to_bytes())
    for invalid in ([], {}, "not-list"):
        payload["entries"] = invalid
        with pytest.raises(MidiFormatError):
            RevisionJournal.from_bytes(json.dumps(payload).encode())


def test_invalid_label_and_score_types_rejected():
    with pytest.raises(MidiWorkstationError):
        RevisionJournal.begin(None)
    with pytest.raises(MidiWorkstationError):
        RevisionJournal.begin(initial(), label="bad\nlabel")
    with pytest.raises(MidiWorkstationError):
        RevisionJournal.begin(initial(), label="")


def test_hash_chain_not_advertised_as_authentication():
    report = history().summary()
    assert report["remote_writes"] is False
    assert len(report["entries"]) == 3
    assert all(len(item["entry_digest"]) == 64 for item in report["entries"])
