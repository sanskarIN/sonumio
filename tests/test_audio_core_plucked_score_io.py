"""Canonical plucked string scores, deterministic WAV export, CLI safety."""
import hashlib
import json
import os

import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError
from sonumio.audio_core.plucked_string import (
    PluckedStringPlan, PluckNote, StringVoice,
)
from sonumio.audio_core.plucked_score_io import (
    PluckedScore, parse_plucked_score, load_plucked_score,
    export_plucked_score,
)
from sonumio.audio_core.waveio import read_wav_file
from sonumio.cli import main


def score():
    return PluckedScore(
        PluckedStringPlan(
            sample_rate=8000, frames=9000, channels=2, gain=.1,
            voice=StringVoice(damping=.99, release_frames=100),
        ),
        (
            PluckNote(0, 4000, 60, velocity=.7, pan=-.5, seed=98),
            PluckNote(3000, 4000, 64, velocity=.7, pan=.5, seed=99),
        ),
    )


def test_score_roundtrip_canonical_digest(tmp_path):
    original = score()
    raw = original.to_bytes()
    assert parse_plucked_score(raw) == original
    assert hashlib.sha256(raw).hexdigest() == original.sha256
    path = tmp_path / "pluck.json"
    path.write_bytes(raw)
    assert load_plucked_score(path).to_bytes() == raw


@pytest.mark.parametrize("field,value", [
    ("protocol", "future"),
    ("plan", {}),
    ("notes", "other"),
    ("unsafe_path", "external.wav"),
])
def test_malformed_top_level_schema_rejected(field, value):
    payload = json.loads(score().to_bytes())
    payload[field] = value
    with pytest.raises(AudioContractError):
        parse_plucked_score(json.dumps(payload).encode())


def test_unknown_nested_plan_voice_note_fields_rejected():
    for target in ("plan", "voice", "note"):
        payload = json.loads(score().to_bytes())
        if target == "plan":
            payload["plan"]["source_wav"] = "forbidden.wav"
        elif target == "voice":
            payload["plan"]["voice"]["plugin"] = "forbidden"
        else:
            payload["notes"][0]["script"] = "forbidden"
        with pytest.raises(AudioContractError):
            parse_plucked_score(json.dumps(payload).encode())


def test_duplicate_key_and_oversize_rejected(tmp_path):
    raw = score().to_bytes()
    with pytest.raises(AudioContractError):
        parse_plucked_score(b'{"protocol":"duplicate",' + raw[1:])
    file = tmp_path / "oversized.json"
    file.write_bytes(b" " * 250001)
    with pytest.raises(AudioContractError):
        load_plucked_score(file)


def test_atomic_export_and_byte_identical_pinned_replay(tmp_path):
    p = score()
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    result = export_plucked_score(p, a)
    assert result.score_sha256 == p.sha256
    assert result.output_sha256 == hashlib.sha256(a.read_bytes()).hexdigest()
    assert result.frames == 9000
    assert result.channels == 2
    assert result.note_count == 2
    assert result.maximum_polyphony == 2
    export_plucked_score(p, b, expected_output_sha256=result.output_sha256)
    assert a.read_bytes() == b.read_bytes()
    decoded = read_wav_file(a)
    assert decoded.sample_rate == 8000
    assert decoded.frames == 9000


def test_seeded_dither_replay_matches(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    c = tmp_path / "c.wav"
    result = export_plucked_score(score(), a, dither_seed=321)
    export_plucked_score(score(), b, dither_seed=321)
    export_plucked_score(score(), c, dither_seed=322)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    assert result.dither_seed == 321


def test_bad_output_hash_never_creates_destination(tmp_path):
    target = tmp_path / "output.wav"
    with pytest.raises(AudioIntegrityError):
        export_plucked_score(score(), target,
                             expected_output_sha256="0" * 64)
    assert not target.exists()


def test_existing_destination_requires_explicit_overwrite(tmp_path):
    target = tmp_path / "out.wav"
    target.write_bytes(b"existing")
    with pytest.raises(AudioContractError):
        export_plucked_score(score(), target)
    assert target.read_bytes() == b"existing"
    export_plucked_score(score(), target, overwrite=True)
    assert target.read_bytes()[:4] == b"RIFF"


def test_score_file_cannot_alias_destination(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(score().to_bytes())
    with pytest.raises(AudioContractError):
        export_plucked_score(score(), source, score_path=source, overwrite=True)
    assert source.read_bytes() == score().to_bytes()


def test_hardlinked_alias_rejected(tmp_path):
    score_file = tmp_path / "score.json"
    score_file.write_bytes(score().to_bytes())
    target = tmp_path / "out.wav"
    try:
        os.link(score_file, target)
    except OSError:
        pytest.skip("Hardlinks unavailable")
    with pytest.raises(AudioContractError):
        export_plucked_score(score(), target,
                             score_path=score_file, overwrite=True)
    assert score_file.read_bytes() == score().to_bytes()


def test_plucked_cli_emits_sha_verified_receipt(tmp_path, capsys):
    path = tmp_path / "score.json"
    path.write_bytes(score().to_bytes())
    output = tmp_path / "output.wav"
    args = ["plucked-render", "--score", str(path), "--output", str(output)]
    assert main(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["protocol"] == "sonumio-verified-plucked-wav-v1"
    assert doc["score_sha256"] == score().sha256
    assert doc["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert doc["model_inference"] is False
    assert doc["source_recording_used"] is False
    assert main(args) == 2
    assert capsys.readouterr().out == ""


def test_bad_score_does_not_overwrite_wav(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_bytes(b"not valid JSON")
    output = tmp_path / "output.wav"
    output.write_bytes(b"old")
    assert main(["plucked-render", "--score", str(path),
                 "--output", str(output), "--overwrite"]) == 2
    assert output.read_bytes() == b"old"
    assert capsys.readouterr().out == ""


def test_explicit_parent_creation(tmp_path, capsys):
    path = tmp_path / "score.json"
    path.write_bytes(score().to_bytes())
    output = tmp_path / "newfolder" / "result.wav"
    assert main(["plucked-render", "--score", str(path),
                 "--output", str(output), "--create-parent"]) == 0
    assert output.exists()
