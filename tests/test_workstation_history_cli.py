"""Local journal CLI commit/undo/redo/checkouts and fail-closed provenance."""
from dataclasses import replace
import hashlib
import json

import pytest

from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.journal import RevisionJournal
from sonumio.midi_workstation.score import Score, ScoreNote
from sonumio.midi_workstation.score_io import decode_score, encode_score


def original():
    score = Score.empty(end=1920)
    track = score.tracks[0].replace_notes((ScoreNote("a", 0, 120, 60),))
    return score.with_tracks((track,))


def save(tmp_path, label, score):
    path = tmp_path / f"{label}.json"
    path.write_bytes(encode_score(score))
    return path


def call(capsys, *args):
    status = main(list(map(str, args)))
    out = capsys.readouterr()
    if status == 0:
        assert not out.err
        return 0, json.loads(out.out)
    assert not out.out
    return status, out.err


def test_history_cli_complete_undo_redo_and_checkout(tmp_path, capsys):
    a = original()
    b = a.with_tracks((a.tracks[0].replace_notes((replace(a.notes[0], pitch=64),)),))
    source = save(tmp_path, "first", a)
    changed = save(tmp_path, "second", b)
    base_history = tmp_path / "history0.json"
    result, report = call(capsys, "history-init", source, "--output", base_history,
                          "--label", "Original")
    assert result == 0
    assert report["journal"]["cursor"] == 0
    history0 = RevisionJournal.from_bytes(base_history.read_bytes())
    history1_path = tmp_path / "history1.json"
    code, committed = call(capsys, "history-commit", base_history, changed,
                           "--label", "Transpose", "--output", history1_path,
                           "--expected-current-sha256", a.sha256)
    assert code == 0
    assert committed["journal"]["cursor"] == 1
    journal1 = RevisionJournal.from_bytes(history1_path.read_bytes())
    assert journal1.current.notes[0].pitch == 64
    undone = tmp_path / "undone.json"
    code, undo = call(capsys, "history-undo", history1_path,
                      "--output", undone, "--expected-current-sha256", b.sha256)
    assert code == 0
    assert undo["journal"]["can_redo"] is True
    assert RevisionJournal.from_bytes(undone.read_bytes()).current.notes[0].pitch == 60
    redo_file = tmp_path / "redo.json"
    code, redo = call(capsys, "history-redo", undone, "--output", redo_file,
                      "--expected-current-sha256", a.sha256)
    assert code == 0
    assert RevisionJournal.from_bytes(redo_file.read_bytes()).current.notes[0].pitch == 64
    checked = tmp_path / "checkout.json"
    code, extracted = call(capsys, "history-checkout", redo_file,
                           "--index", 0, "--output", checked)
    assert code == 0
    assert decode_score(checked.read_bytes()) == a
    assert extracted["selected_score_sha256"] == a.sha256


def test_history_inspect_does_not_modify_history_file(tmp_path, capsys):
    source = save(tmp_path, "source", original())
    history = tmp_path / "history.json"
    assert call(capsys, "history-init", source, "--output", history)[0] == 0
    before = history.read_bytes()
    code, receipt = call(capsys, "history-inspect", history)
    assert code == 0
    assert receipt["files_written"] == 0
    assert receipt["input_sha256"] == hashlib.sha256(before).hexdigest()
    assert history.read_bytes() == before


def test_invalid_checksum_prevents_history_export(tmp_path, capsys):
    source = save(tmp_path, "source", original())
    output = tmp_path / "history.json"
    code, message = call(capsys, "history-init", source, "--output", output,
                         "--expected-sha256", "0" * 64)
    assert code == 2
    assert not output.exists()


def test_source_destination_identity_never_overwrites_score(tmp_path, capsys):
    source = save(tmp_path, "source", original())
    old = source.read_bytes()
    code, message = call(capsys, "history-init", source, "--output", source,
                         "--overwrite")
    assert code == 2
    assert source.read_bytes() == old


def test_stale_commit_fails_without_publishing(tmp_path, capsys):
    original_score = original()
    source = save(tmp_path, "source", original_score)
    next_score = original_score.with_tracks((original_score.tracks[0].replace_notes((
        replace(original_score.notes[0], pitch=65),
    )),))
    changed = save(tmp_path, "changed", next_score)
    journal = tmp_path / "journal.json"
    assert call(capsys, "history-init", source, "--output", journal)[0] == 0
    output = tmp_path / "next.json"
    code, message = call(capsys, "history-commit", journal, changed,
                         "--label", "stale", "--output", output,
                         "--expected-current-sha256", "0" * 64)
    assert code == 2
    assert not output.exists()


def test_history_checkout_cannot_write_over_history(tmp_path, capsys):
    source = save(tmp_path, "source", original())
    journal = tmp_path / "journal.json"
    assert call(capsys, "history-init", source, "--output", journal)[0] == 0
    before = journal.read_bytes()
    code, message = call(capsys, "history-checkout", journal,
                         "--output", journal, "--overwrite")
    assert code == 2
    assert journal.read_bytes() == before


def test_score_compare_readonly_receipt(tmp_path, capsys):
    a = original()
    b = a.with_tracks((a.tracks[0].replace_notes((replace(a.notes[0], start=20),)),))
    source = save(tmp_path, "left", a)
    target = save(tmp_path, "right", b)
    a_bytes, b_bytes = source.read_bytes(), target.read_bytes()
    code, report = call(capsys, "score-compare", source, target,
                        "--onset-tolerance", 25, "--release-tolerance", 25)
    assert code == 0
    assert report["protocol"] == "sonumio-workstation-compare-cli-v1"
    assert report["comparison"]["matched_notes"] == 1
    assert report["comparison"]["mean_onset_error_ticks"] == 20
    assert report["files_written"] == 0
    assert source.read_bytes() == a_bytes and target.read_bytes() == b_bytes


def test_reject_checkout_index_outside_history(tmp_path, capsys):
    source = save(tmp_path, "source", original())
    journal = tmp_path / "journal.json"
    assert call(capsys, "history-init", source, "--output", journal)[0] == 0
    output = tmp_path / "invalid.json"
    assert call(capsys, "history-checkout", journal,
                "--index", 100, "--output", output)[0] == 2
    assert not output.exists()


def test_all_success_receipts_have_no_remote_side_effects(tmp_path, capsys):
    source = save(tmp_path, "source", original())
    history = tmp_path / "journal.json"
    code, receipt = call(capsys, "history-init", source, "--output", history)
    assert code == 0
    assert receipt["journal"]["remote_writes"] is False
    assert receipt["publication"]["remote_publication"] is False
    assert receipt["input_files_unchanged"] is True
