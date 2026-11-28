"""End-to-end workstation entrypoint and checksum-safe local file workflows."""
import hashlib
import json
import subprocess
import sys
import os
from pathlib import Path

import pytest

from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.score import Score, ScoreNote
from sonumio.midi_workstation.score_io import decode_score, encode_score
from sonumio.midi_workstation.smf import parse_smf


def execute(argv, capsys):
    status = main(argv)
    captured = capsys.readouterr()
    if status == 0:
        assert captured.err == ""
        return status, json.loads(captured.out)
    assert captured.out == ""
    assert "Sonumio MIDI workstation:" in captured.err
    return status, None


def create(tmp_path, capsys):
    path = tmp_path / "original.json"
    status, receipt = execute([
        "score-new", "--output", str(path), "--ppqn", "480",
        "--end", "1920", "--title", "Original piece",
    ], capsys)
    assert status == 0
    assert path.exists()
    assert receipt["protocol"] == "sonumio-workstation-new-score-v1"
    return path


def test_score_new_produces_canonical_score_and_receipt(tmp_path, capsys):
    path = create(tmp_path, capsys)
    result = decode_score(path.read_bytes())
    assert result.ppqn == 480
    assert result.end == 1920
    assert result.title == "Original piece"
    assert result.note_count == 0


def test_score_new_fails_when_target_exists(tmp_path, capsys):
    path = create(tmp_path, capsys)
    original = path.read_bytes()
    code, _ = execute(["score-new", "--output", str(path)], capsys)
    assert code == 2
    assert path.read_bytes() == original


def test_score_new_explicit_overwrite(tmp_path, capsys):
    path = create(tmp_path, capsys)
    code, result = execute([
        "score-new", "--output", str(path), "--overwrite",
        "--title", "New title",
    ], capsys)
    assert code == 0
    assert decode_score(path.read_bytes()).title == "New title"
    assert result["published"]["overwrote"] is True


def test_score_new_pinned_output_hash_checked_before_publication(tmp_path, capsys):
    path = tmp_path / "score.json"
    code, _ = execute([
        "score-new", "--output", str(path),
        "--expected-output-sha256", "0" * 64,
    ], capsys)
    assert code == 2
    assert not path.exists()


def test_score_inspect_returns_structured_analysis(tmp_path, capsys):
    path = create(tmp_path, capsys)
    code, result = execute(["score-inspect", str(path)], capsys)
    assert code == 0
    assert result["source"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["analysis"]["source_mutated"] is False if "source_mutated" in result["analysis"] else True


def test_score_inspect_optional_checksum_validates_source(tmp_path, capsys):
    path = create(tmp_path, capsys)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    code, _ = execute(["score-inspect", str(path), "--expected-sha256", actual], capsys)
    assert code == 0
    code, _ = execute(["score-inspect", str(path), "--expected-sha256", "0" * 64], capsys)
    assert code == 2


def test_score_quantize_creates_new_revision_and_protects_source(tmp_path, capsys):
    path = create(tmp_path, capsys)
    original = decode_score(path.read_bytes())
    note = ScoreNote("n1", 161, 100, 64)
    track = original.tracks[0].replace_notes((note,))
    original = original.with_tracks((track,))
    path.write_bytes(encode_score(original))
    before = path.read_bytes()
    dest = tmp_path / "quantized.json"
    code, receipt = execute([
        "score-quantize", str(path), "--output", str(dest),
        "--step-ticks", "120", "--quantize-releases",
    ], capsys)
    assert code == 0
    changed = decode_score(dest.read_bytes())
    assert changed.notes[0].start == 120
    assert changed.notes[0].stop == 240
    assert path.read_bytes() == before
    assert receipt["quantize"]["changed_notes"] == 1


def test_humanize_same_seed_is_deterministic(tmp_path, capsys):
    path = create(tmp_path, capsys)
    score = decode_score(path.read_bytes())
    track = score.tracks[0].replace_notes((
        ScoreNote("a", 0, 100, 60), ScoreNote("b", 120, 100, 64),
    ))
    path.write_bytes(encode_score(score.with_tracks((track,))))
    a = tmp_path / "human-a.json"
    b = tmp_path / "human-b.json"
    command = ["score-humanize", str(path), "--seed", "172",
               "--timing-ticks", "4", "--velocity-units", "6"]
    code, first = execute(command + ["--output", str(a)], capsys)
    assert code == 0
    code, second = execute(command + ["--output", str(b)], capsys)
    assert code == 0
    assert a.read_bytes() == b.read_bytes()
    assert first["expression"]["result_sha256"] == second["expression"]["result_sha256"]


def test_export_and_validate_midi_end_to_end(tmp_path, capsys):
    path = create(tmp_path, capsys)
    midi_file = tmp_path / "export.mid"
    code, receipt = execute(["midi-export", str(path), "--output", str(midi_file)], capsys)
    assert code == 0
    assert midi_file.read_bytes().startswith(b"MThd")
    digest = hashlib.sha256(midi_file.read_bytes()).hexdigest()
    assert receipt["publication"]["output_sha256"] == digest
    code, checked = execute(["midi-validate", str(midi_file)], capsys)
    assert code == 0
    assert checked["input_sha256"] == digest
    assert parse_smf(midi_file.read_bytes()).midi.end == 1920


def test_midi_inspection_with_note_option(tmp_path, capsys):
    source = create(tmp_path, capsys)
    output = tmp_path / "score.mid"
    assert execute(["midi-export", str(source), "--output", str(output)], capsys)[0] == 0
    status, value = execute(["midi-inspect", str(output), "--notes"], capsys)
    assert status == 0
    assert value["protocol"] == "sonumio-midi-workstation-inspection-v1"
    assert value["audio_rendered"] is False
    assert value["model_inference"] is False


def test_invalid_midi_does_not_create_import_output(tmp_path, capsys):
    malformed = tmp_path / "malformed.mid"
    malformed.write_bytes(b"not-a-midi")
    output = tmp_path / "score.json"
    code, _ = execute(["midi-import", str(malformed), "--output", str(output)], capsys)
    assert code == 2
    assert not output.exists()


def test_missing_input_does_not_create_output(tmp_path, capsys):
    unknown = tmp_path / "missing.json"
    target = tmp_path / "export.mid"
    code, _ = execute(["midi-export", str(unknown), "--output", str(target)], capsys)
    assert code == 2
    assert not target.exists()


def test_output_cannot_alias_source_even_when_overwrite_requested(tmp_path, capsys):
    path = create(tmp_path, capsys)
    original = path.read_bytes()
    code, _ = execute(["score-quantize", str(path), "--output", str(path),
                        "--step-ticks", "120", "--overwrite"], capsys)
    assert code == 2
    assert path.read_bytes() == original


def test_output_can_be_created_in_explicit_parent(tmp_path, capsys):
    output = tmp_path / "new-folder" / "new.json"
    status, _ = execute(["score-new", "--output", str(output),
                          "--create-parent"], capsys)
    assert status == 0
    assert output.exists()


def test_nonexistent_midi_input_gives_no_false_success_json(tmp_path, capsys):
    target = tmp_path / "missing.mid"
    status, payload = execute(["midi-inspect", str(target)], capsys)
    assert status == 2
    assert payload is None


def test_python_module_entrypoint_supports_subprocess_cli(tmp_path):
    target = tmp_path / "subprocess.json"
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    proc = subprocess.run([
        sys.executable, "-m", "sonumio.midi_workstation", "score-new",
        "--output", str(target),
    ], capture_output=True, text=True, env=env, timeout=15)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["published"]["bytes_written"] > 0
    assert target.exists()


def test_invalid_score_title_fails_closed(tmp_path, capsys):
    target = tmp_path / "bad.json"
    code, _ = execute(["score-new", "--output", str(target),
                        "--title", "bad\nname"], capsys)
    assert code == 2
    assert not target.exists()


def test_bad_quantize_grid_prevents_output_file(tmp_path, capsys):
    source = create(tmp_path, capsys)
    output = tmp_path / "quantized.json"
    code, _ = execute(["score-quantize", str(source),
                        "--step-ticks", "0", "--output", str(output)], capsys)
    assert code == 2
    assert not output.exists()


def test_source_midi_read_has_no_network_or_model_side_effects(tmp_path, capsys):
    path = create(tmp_path, capsys)
    midi = tmp_path / "original.mid"
    assert execute(["midi-export", str(path), "--output", str(midi)], capsys)[0] == 0
    before = midi.read_bytes()
    code, analysis = execute(["midi-inspect", str(midi)], capsys)
    assert code == 0
    assert analysis["model_inference"] is False
    assert midi.read_bytes() == before
