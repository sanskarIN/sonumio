"""Strict symbolic recipe JSON, deterministic playback and CLI integration."""
from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.contracts import MidiFormatError, MidiWorkstationError
from sonumio.midi_workstation.patterns import PatternPlan, PatternCell, PatternVoice
from sonumio.midi_workstation.pattern_io import (
    PatternRecipe, decode_pattern_recipe, load_pattern_recipe,
)
from sonumio.midi_workstation.score_io import decode_score


def recipe():
    return PatternRecipe(
        PatternPlan(bars=4, seed=33, title="Original pattern test"),
        (
            PatternVoice("melody", "Original melody", 0, 0, (
                PatternCell(0, 60, 2, 90, 100),
                PatternCell(4, 64, 2, 88, 100),
                PatternCell(8, 67, 2, 95, 100),
            )),
            PatternVoice("drums", "Drum trigger", 9, 0, (
                PatternCell(0, 36), PatternCell(4, 38),
                PatternCell(8, 36), PatternCell(12, 42),
            )),
        ),
    )


def test_recipe_roundtrip_exact_and_digest():
    original = recipe()
    raw = original.to_bytes()
    assert decode_pattern_recipe(raw) == original
    assert original.sha256 == sha256(raw).hexdigest()
    assert decode_pattern_recipe(raw).render() == original.render()


def test_noncanonical_json_is_normalized_without_changing_score():
    original = recipe()
    noncanonical = json.dumps(json.loads(original.to_bytes()), indent=2).encode()
    parsed = decode_pattern_recipe(noncanonical)
    assert parsed.sha256 == original.sha256
    assert parsed.to_bytes() == original.to_bytes()


def test_load_binds_optional_exact_input_sha(tmp_path):
    raw = recipe().to_bytes()
    file = tmp_path / "recipe.json"
    file.write_bytes(raw)
    value, report = load_pattern_recipe(file, expected_sha256=sha256(raw).hexdigest())
    assert value == recipe()
    assert report["source_sha256"] == sha256(raw).hexdigest()
    assert report["canonical_sha256"] == value.sha256
    assert report["network_uses"] == 0
    with pytest.raises(MidiWorkstationError):
        load_pattern_recipe(file, expected_sha256="0" * 64)


@pytest.mark.parametrize("field", ["protocol", "plan", "voices"])
def test_missing_required_fields_rejected(field):
    payload = json.loads(recipe().to_bytes())
    del payload[field]
    with pytest.raises(MidiFormatError):
        decode_pattern_recipe(json.dumps(payload).encode())


def test_unknown_fields_are_rejected_at_every_schema_depth():
    for level in ("root", "plan", "voice", "cell"):
        payload = json.loads(recipe().to_bytes())
        if level == "root":
            payload["script"] = "forbidden"
        elif level == "plan":
            payload["plan"]["source_wav"] = "forbidden.wav"
        elif level == "voice":
            payload["voices"][0]["audio_file"] = "forbidden.wav"
        else:
            payload["voices"][0]["cells"][0]["expression"] = "forbidden"
        with pytest.raises(MidiFormatError):
            decode_pattern_recipe(json.dumps(payload).encode())


def test_duplicate_fields_nonfinite_numbers_and_extra_large_inputs_rejected():
    raw = recipe().to_bytes()
    with pytest.raises(MidiFormatError):
        decode_pattern_recipe(b'{"protocol":"duplicate",' + raw[1:])
    with pytest.raises(MidiFormatError):
        decode_pattern_recipe(raw.replace(b'"velocity":90', b'"velocity":NaN'))
    with pytest.raises(MidiWorkstationError):
        decode_pattern_recipe(b" " * (256 * 1024 + 1))


def test_pattern_cli_publishes_deterministic_original_score(tmp_path, capsys):
    source = tmp_path / "pattern.json"
    source.write_bytes(recipe().to_bytes())
    a = tmp_path / "first.json"
    b = tmp_path / "second.json"
    cmd = ["score-pattern", str(source)]
    assert main(cmd + ["--output", str(a)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["protocol"] == "sonumio-score-pattern-cli-v1"
    assert receipt["recipe"]["canonical_sha256"] == recipe().sha256
    assert receipt["render"]["score_notes"] == 28
    assert receipt["publication"]["output_sha256"] == sha256(a.read_bytes()).hexdigest()
    assert main(cmd + ["--output", str(b)]) == 0
    capsys.readouterr()
    assert a.read_bytes() == b.read_bytes()
    score = decode_score(a.read_bytes())
    assert score.note_count == 28
    assert score.end == 7680
    assert source.read_bytes() == recipe().to_bytes()


def test_cli_wrong_input_sha_creates_no_output(tmp_path, capsys):
    source = tmp_path / "pattern.json"
    source.write_bytes(recipe().to_bytes())
    output = tmp_path / "never.json"
    status = main(["score-pattern", str(source), "--output", str(output),
                   "--expected-sha256", "0" * 64])
    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert not output.exists()


def test_pattern_cli_refuses_source_destination_alias(tmp_path, capsys):
    source = tmp_path / "source.json"
    original = recipe().to_bytes()
    source.write_bytes(original)
    status = main(["score-pattern", str(source), "--output", str(source), "--overwrite"])
    assert status == 2
    assert capsys.readouterr().out == ""
    assert source.read_bytes() == original


def test_invalid_and_missing_source_never_publishes(tmp_path, capsys):
    source = tmp_path / "missing.json"
    destination = tmp_path / "out.json"
    assert main(["score-pattern", str(source), "--output", str(destination)]) == 2
    assert capsys.readouterr().out == ""
    source.write_text("not json")
    assert main(["score-pattern", str(source), "--output", str(destination)]) == 2
    assert capsys.readouterr().out == ""
    assert not destination.exists()
