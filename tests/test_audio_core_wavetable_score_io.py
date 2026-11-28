"""Trusted canonical wavetable score CLI, exact publication and error safety."""
import hashlib
import json
import os

import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError
from sonumio.audio_core.wavetable import (
    WavetablePlan, WavetableNote, VoiceEnvelope,
)
from sonumio.audio_core.wavetable_score_io import (
    WavetableScore, parse_wavetable_score, load_wavetable_score,
    export_wavetable_score,
)
from sonumio.audio_core.waveio import read_wav_file
from sonumio.cli import main


def score(**opts):
    return WavetableScore(
        WavetablePlan(
            sample_rate=8000, frames=3000, channels=2, waveform="triangle",
            table_size=2048, max_harmonics=32, master_gain=.15,
            envelope=VoiceEnvelope(4, 30, .7, 80),
        ),
        (
            WavetableNote(0, 1000, 60, velocity=.5, pan=-.5),
            WavetableNote(450, 800, 64, velocity=.5, pan=.5),
        ),
    )


def test_score_round_trip_and_canonical_sha(tmp_path):
    original = score()
    raw = original.to_bytes()
    decoded = parse_wavetable_score(raw)
    assert decoded == original
    assert original.sha256 == hashlib.sha256(raw).hexdigest()
    path = tmp_path / "score.json"
    path.write_bytes(raw)
    assert load_wavetable_score(path).to_bytes() == raw


def test_canonical_serialization_does_not_depend_on_dict_order():
    original = score().to_bytes()
    object_data = json.loads(original)
    reserialized = json.dumps(object_data, indent=3).encode()
    assert parse_wavetable_score(reserialized).sha256 == hashlib.sha256(
        original
    ).hexdigest()


@pytest.mark.parametrize("field,value", [
    ("protocol", "future-protocol"),
    ("render", {}),
    ("notes", {}),
    ("extra", True),
])
def test_top_level_unknown_fields_are_rejected(field, value):
    payload = json.loads(score().to_bytes())
    payload[field] = value
    with pytest.raises(AudioContractError):
        parse_wavetable_score(json.dumps(payload).encode())


def test_unknown_render_note_and_envelope_fields_rejected():
    for category, key in (
        ("render", "path"), ("envelope", "script"), ("note", "source_wav"),
    ):
        payload = json.loads(score().to_bytes())
        if category == "render":
            payload["render"][key] = "forbidden"
        elif category == "envelope":
            payload["render"]["envelope"][key] = "forbidden"
        else:
            payload["notes"][0][key] = "forbidden"
        with pytest.raises(AudioContractError):
            parse_wavetable_score(json.dumps(payload).encode())


def test_duplicate_json_field_rejected():
    raw = score().to_bytes()
    with pytest.raises(AudioContractError):
        parse_wavetable_score(b'{"protocol":"duplicate",' + raw[1:])


def test_oversized_json_rejected(tmp_path):
    path = tmp_path / "too-large.json"
    path.write_bytes(b" " * 250001)
    with pytest.raises(AudioContractError):
        load_wavetable_score(path)


def test_atomic_export_digest_and_pinned_replay(tmp_path):
    original = score()
    one = tmp_path / "one.wav"
    two = tmp_path / "two.wav"
    first = export_wavetable_score(original, one)
    assert first.score_sha256 == original.sha256
    assert first.output_sha256 == hashlib.sha256(one.read_bytes()).hexdigest()
    assert first.output_frames == 3000
    assert first.output_channels == 2
    assert first.notes == 2
    assert first.maximum_polyphony == 2
    assert first.output_bits == 16
    second = export_wavetable_score(
        original, two, expected_output_sha256=first.output_sha256,
    )
    assert first.output_sha256 == second.output_sha256
    assert one.read_bytes() == two.read_bytes()
    read_back = read_wav_file(one)
    assert read_back.frames == 3000
    assert read_back.sample_rate == 8000


def test_seeded_tpdf_is_reproducible_and_auditable(tmp_path):
    original = score()
    a = tmp_path / "dither-a.wav"
    b = tmp_path / "dither-b.wav"
    c = tmp_path / "dither-c.wav"
    result = export_wavetable_score(original, a, dither_seed=123)
    export_wavetable_score(
        original, b, dither_seed=123,
        expected_output_sha256=result.output_sha256,
    )
    export_wavetable_score(original, c, dither_seed=124)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()
    assert result.as_dict()["source_recording_used"] is False


def test_reject_bad_expected_output_digest_before_write(tmp_path):
    target = tmp_path / "result.wav"
    with pytest.raises(AudioIntegrityError):
        export_wavetable_score(score(), target,
                               expected_output_sha256="0" * 64)
    assert not target.exists()


def test_no_clobber_and_explicit_overwrite(tmp_path):
    target = tmp_path / "output.wav"
    target.write_bytes(b"previous data")
    with pytest.raises(AudioContractError):
        export_wavetable_score(score(), target)
    assert target.read_bytes() == b"previous data"
    result = export_wavetable_score(score(), target, overwrite=True)
    assert result.output_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()


def test_source_score_output_alias_rejected(tmp_path):
    source = tmp_path / "score.wav"
    source.write_bytes(score().to_bytes())
    with pytest.raises(AudioContractError):
        export_wavetable_score(score(), source, score_path=source, overwrite=True)
    assert source.read_bytes() == score().to_bytes()


def test_source_score_hardlink_alias_rejected(tmp_path):
    source = tmp_path / "score.json"
    source.write_bytes(score().to_bytes())
    target = tmp_path / "linked.wav"
    try:
        os.link(source, target)
    except OSError:
        pytest.skip("Hardlink not supported")
    with pytest.raises(AudioContractError):
        export_wavetable_score(score(), target, score_path=source, overwrite=True)
    assert source.read_bytes() == score().to_bytes()


def test_too_loud_score_is_rejected_without_clipping(tmp_path):
    base = score()
    p = WavetablePlan(
        sample_rate=8000, frames=1000, channels=1,
        waveform="sine", master_gain=1.,
        envelope=VoiceEnvelope(0, 0, 1., 0),
    )
    many = tuple(WavetableNote(0, 1000, 69) for _ in range(5))
    target = tmp_path / "overrange.wav"
    with pytest.raises(AudioContractError, match="lower master_gain"):
        export_wavetable_score(WavetableScore(p, many), target)
    assert not target.exists()


def test_wavetable_cli_emits_one_parseable_receipt(tmp_path, capsys):
    plan = tmp_path / "score.json"
    plan.write_bytes(score().to_bytes())
    output = tmp_path / "render.wav"
    args = ["wavetable-render", "--score", str(plan), "--output", str(output)]
    assert main(args) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["protocol"] == "sonumio-verified-wavetable-render-v1"
    assert document["score_sha256"] == score().sha256
    assert document["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert document["model_inference"] is False
    assert main(args) == 2
    assert capsys.readouterr().out == ""


def test_cli_bad_plan_does_not_touch_destination(tmp_path, capsys):
    plan = tmp_path / "bad.json"
    plan.write_bytes(b"not json")
    output = tmp_path / "render.wav"
    output.write_bytes(b"old")
    assert main(["wavetable-render", "--score", str(plan),
                 "--output", str(output), "--overwrite"]) == 2
    assert output.read_bytes() == b"old"
    assert capsys.readouterr().out == ""


def test_cli_can_create_parent_explicitly(tmp_path, capsys):
    plan = tmp_path / "score.json"
    plan.write_bytes(score().to_bytes())
    output = tmp_path / "folder" / "generated.wav"
    assert main(["wavetable-render", "--score", str(plan),
                 "--output", str(output), "--create-parent"]) == 0
    assert output.exists()
