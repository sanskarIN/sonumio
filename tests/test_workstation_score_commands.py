"""CLI score diff/merge/crop/concat with explicit local file publication."""
import json
from dataclasses import replace

import pytest

from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.score import Score, ScoreNote
from sonumio.midi_workstation.score_io import decode_score, encode_score


def fixture_score():
    source = Score.empty(end=2880)
    notes = (ScoreNote("a", 100, 120, 60),
             ScoreNote("b", 900, 160, 64),
             ScoreNote("c", 2100, 120, 67))
    return source.with_tracks((source.tracks[0].replace_notes(notes),))


def materialize(tmp_path, label, score):
    path = tmp_path / f"{label}.json"
    path.write_bytes(encode_score(score))
    return path


def run(capsys, *arguments):
    result = main(list(map(str, arguments)))
    output = capsys.readouterr()
    assert (result == 0) == (bool(output.out) and not output.err)
    return result, json.loads(output.out) if result == 0 else output.err


def test_diff_reports_stable_note_changes_without_writes(tmp_path, capsys):
    a = fixture_score()
    b = a.with_tracks((a.tracks[0].replace_notes((
        replace(a.notes[0], pitch=65), *a.notes[1:],
    )),))
    x = materialize(tmp_path, "before", a)
    y = materialize(tmp_path, "after", b)
    before = tuple(path.read_bytes() for path in (x, y))
    code, result = run(capsys, "score-diff", x, y)
    assert code == 0
    assert result["difference"]["notes_updated"] == 1
    assert result["output_files_created"] == 0
    assert tuple(path.read_bytes() for path in (x, y)) == before


def test_three_way_merge_combines_independent_pitches(tmp_path, capsys):
    base = fixture_score()
    ours = base.with_tracks((base.tracks[0].replace_notes((
        replace(base.notes[0], pitch=65), *base.notes[1:],
    )),))
    theirs = base.with_tracks((base.tracks[0].replace_notes((
        base.notes[0], replace(base.notes[1], pitch=68), base.notes[2],
    )),))
    a = materialize(tmp_path, "base", base)
    b = materialize(tmp_path, "ours", ours)
    c = materialize(tmp_path, "theirs", theirs)
    out = tmp_path / "merged.json"
    code, receipt = run(capsys, "score-merge", a, b, c, "--output", out)
    assert code == 0
    assert receipt["merge"]["merged"] is True
    result = decode_score(out.read_bytes())
    assert [n.pitch for n in result.notes] == [65, 68, 67]
    assert base.notes[0].pitch == 60


def test_conflicting_merge_refuses_to_create_artifact(tmp_path, capsys):
    base = fixture_score()
    a = materialize(tmp_path, "base", base)
    ours = base.with_tracks((base.tracks[0].replace_notes((
        replace(base.notes[0], pitch=65), *base.notes[1:],
    )),))
    theirs = base.with_tracks((base.tracks[0].replace_notes((
        replace(base.notes[0], pitch=67), *base.notes[1:],
    )),))
    b = materialize(tmp_path, "ours", ours)
    c = materialize(tmp_path, "theirs", theirs)
    output = tmp_path / "conflict.json"
    status, error = run(capsys, "score-merge", a, b, c, "--output", output)
    assert status == 2
    assert "unresolved conflicts" in error
    assert not output.exists()


def test_score_crop_trim_and_preserve_sibling_input(tmp_path, capsys):
    base = fixture_score()
    source = materialize(tmp_path, "source", base)
    original = source.read_bytes()
    target = tmp_path / "cropped.json"
    code, receipt = run(capsys, "score-crop", source,
                        "--start-tick", 150, "--stop-tick", 1000,
                        "--output", target)
    assert code == 0
    cropped = decode_score(target.read_bytes())
    assert cropped.end == 850
    assert cropped.note_count == 2
    assert receipt["region"]["trimmed"] == 2
    assert source.read_bytes() == original


def test_score_crop_reject_crossing_fails_closed(tmp_path, capsys):
    source = materialize(tmp_path, "source", fixture_score())
    destination = tmp_path / "crop.json"
    code, _ = run(capsys, "score-crop", source,
                  "--start-tick", 150, "--stop-tick", 1000,
                  "--reject-crossing", "--output", destination)
    assert code == 2
    assert not destination.exists()


def test_concat_preserves_track_count_and_namespaced_ids(tmp_path, capsys):
    first = fixture_score()
    second = fixture_score()
    a = materialize(tmp_path, "first", first)
    b = materialize(tmp_path, "second", second)
    target = tmp_path / "concat.json"
    code, receipt = run(capsys, "score-concat", a, b,
                        "--output", target, "--title", "My joined score")
    assert code == 0
    result = decode_score(target.read_bytes())
    assert result.end == 5760
    assert result.note_count == 6
    assert result.title == "My joined score"
    assert receipt["concatenation"]["part_count"] == 2
    assert result.notes[0].id.startswith("p0-")
    assert result.notes[-1].id.startswith("p1-")


def test_concat_output_must_not_alias_any_part(tmp_path, capsys):
    a = materialize(tmp_path, "a", fixture_score())
    b = materialize(tmp_path, "b", fixture_score())
    before = b.read_bytes()
    code, _ = run(capsys, "score-concat", a, b, "--output", b, "--overwrite")
    assert code == 2
    assert b.read_bytes() == before


def test_merge_output_must_not_alias_second_input(tmp_path, capsys):
    source = fixture_score()
    a = materialize(tmp_path, "a", source)
    b = materialize(tmp_path, "b", source)
    c = materialize(tmp_path, "c", source)
    before = b.read_bytes()
    code, _ = run(capsys, "score-merge", a, b, c, "--output", b, "--overwrite")
    assert code == 2
    assert b.read_bytes() == before


def test_invalid_sha_pins_fail_before_any_output(tmp_path, capsys):
    a = materialize(tmp_path, "a", fixture_score())
    b = materialize(tmp_path, "b", fixture_score())
    output = tmp_path / "never.json"
    code, _ = run(capsys, "score-merge", a, b, b, "--output", output,
                  "--base-sha256", "0" * 64)
    assert code == 2
    assert not output.exists()
