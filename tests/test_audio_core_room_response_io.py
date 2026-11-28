import hashlib
import json
import os

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError
from sonumio.audio_core.room_response import Point3D, RoomConfig, ShoeboxRoom
from sonumio.audio_core.room_response_io import (
    RoomPlan, export_room_plan, load_room_plan, parse_room_plan,
)
from sonumio.audio_core.waveio import encode_wav_bytes, read_wav_file
from sonumio.cli import main


def example_plan(**kwargs):
    return RoomPlan(
        ShoeboxRoom(4, 4, 3), (Point3D(1, 2, 1),),
        (Point3D(2, 2, 1), Point3D(3, 2, 1)),
        RoomConfig(sample_rate=8000, reflection_order=0), **kwargs,
    )


def wav(path, amplitude=.25):
    source = make_buffer(np.array([amplitude, 0., 0.], dtype=np.float32), 8000)
    path.write_bytes(encode_wav_bytes(source))
    return path


def test_plan_round_trip_and_canonical_digest(tmp_path):
    plan = example_plan(output_gain=.8)
    raw = plan.to_bytes()
    assert parse_room_plan(raw) == plan
    assert plan.sha256 == hashlib.sha256(raw).hexdigest()
    path = tmp_path / "room.json"
    path.write_bytes(raw)
    assert load_room_plan(path).to_bytes() == raw


@pytest.mark.parametrize("field,value", [
    ("protocol", "unknown"), ("output_gain", 1.1),
    ("include_tail", 1), ("sources", []),
    ("receivers", [{"x": 1, "y": 2, "z": 1, "path": "local.wav"}]),
    ("config", {}), ("room", {}),
    ("extra", "unknown"),
])
def test_plan_rejects_unknown_or_invalid_data(field, value):
    payload = json.loads(example_plan().to_bytes())
    payload[field] = value
    with pytest.raises(AudioContractError):
        parse_room_plan(json.dumps(payload).encode())


def test_duplicate_key_and_oversized_plan_rejected(tmp_path):
    raw = example_plan().to_bytes()
    with pytest.raises(AudioContractError):
        parse_room_plan(b'{"protocol":"duplicate",' + raw[1:])
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 16385)
    with pytest.raises(AudioContractError):
        load_room_plan(path)


def test_verified_export_and_pinned_replay(tmp_path):
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "render.wav"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = example_plan()
    report = export_room_plan(plan, source, output, expected_source_sha256=source_sha)
    assert report.source["source_sha256"] == source_sha
    assert report.plan_sha256 == plan.sha256
    assert report.output_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report.output_channels == 2
    assert report.path_count == 2
    assert report.output_frames == report.impulse_frames + 2
    rendered = read_wav_file(output)
    assert rendered.channels == 2
    assert rendered.frames == report.output_frames
    assert np.max(np.abs(rendered.samples)) <= .25
    second = tmp_path / "replay.wav"
    export_room_plan(plan, source, second, expected_output_sha256=report.output_sha256)
    assert output.read_bytes() == second.read_bytes()


def test_bad_hash_preserves_destination(tmp_path):
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "out.wav"
    output.write_bytes(b"existing bytes")
    with pytest.raises(AudioIntegrityError):
        export_room_plan(example_plan(), source, output,
                         expected_output_sha256="0" * 64, overwrite=True)
    assert output.read_bytes() == b"existing bytes"


def test_no_clobber_and_alias(tmp_path):
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "out.wav"
    output.write_bytes(b"existing bytes")
    with pytest.raises(AudioContractError):
        export_room_plan(example_plan(), source, output)
    with pytest.raises(AudioContractError):
        export_room_plan(example_plan(), source, source, overwrite=True)
    assert output.read_bytes() == b"existing bytes"
    alias = tmp_path / "alias.wav"
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("Hard links unavailable")
    with pytest.raises(AudioContractError):
        export_room_plan(example_plan(), source, alias, overwrite=True)


def test_overrange_requires_explicit_attenuation(tmp_path):
    source = wav(tmp_path / "source.wav", .75)
    p = Point3D(1, 1, 1)
    room = ShoeboxRoom(3, 3, 3)
    config = RoomConfig(sample_rate=8000, reflection_order=0)
    target = tmp_path / "out.wav"
    with pytest.raises(AudioContractError, match="reduce output_gain"):
        export_room_plan(RoomPlan(room, (p,), (p,), config), source, target)
    assert not target.exists()
    report = export_room_plan(RoomPlan(room, (p,), (p,), config, output_gain=.25),
                              source, target)
    assert report.peak == pytest.approx(.75)


def test_cli_receipt_and_missing_input(tmp_path, capsys):
    source = wav(tmp_path / "source.wav")
    plan = tmp_path / "room.json"
    plan.write_bytes(example_plan().to_bytes())
    output = tmp_path / "render.wav"
    args = ["room-response", "--plan", str(plan), "--source", str(source),
            "--output", str(output)]
    assert main(args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["protocol"] == "sonumio-verified-room-wav-v1"
    assert receipt["model_inference"] is False
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert main(args) == 2
    assert "overwrite" in capsys.readouterr().err.lower()


def test_cli_plan_cannot_be_output(tmp_path, capsys):
    source = wav(tmp_path / "source.wav")
    plan = tmp_path / "room.wav"
    raw = example_plan().to_bytes()
    plan.write_bytes(raw)
    assert main(["room-response", "--plan", str(plan), "--source", str(source),
                 "--output", str(plan), "--overwrite"]) == 2
    assert plan.read_bytes() == raw


def test_explicit_partitioned_export_replay_and_legacy_receipt(tmp_path):
    source = wav(tmp_path / "source.wav")
    plan = example_plan()
    baseline = tmp_path / "matrix.wav"
    candidate = tmp_path / "partitioned.wav"
    replay = tmp_path / "replay.wav"
    old = export_room_plan(plan, source, baseline)
    new = export_room_plan(plan, source, candidate, engine="partitioned")
    assert old.as_dict()["protocol"] == "sonumio-verified-room-wav-v1"
    assert "engine" not in old.as_dict()
    assert new.as_dict()["protocol"] == "sonumio-verified-room-wav-v2"
    assert new.as_dict()["engine"] == "partitioned"
    assert old.plan_sha256 == new.plan_sha256
    assert old.output_frames == new.output_frames
    assert old.output_channels == new.output_channels
    np.testing.assert_allclose(read_wav_file(baseline).samples,
                               read_wav_file(candidate).samples, atol=1 / 32768)
    export_room_plan(plan, source, replay, engine="partitioned",
                     expected_output_sha256=new.output_sha256)
    assert candidate.read_bytes() == replay.read_bytes()


def test_invalid_backend_does_not_touch_output(tmp_path):
    source = wav(tmp_path / "source.wav")
    target = tmp_path / "output.wav"
    target.write_bytes(b"old bytes")
    with pytest.raises(AudioContractError, match="Unknown room convolution engine"):
        export_room_plan(example_plan(), source, target,
                         engine="auto", overwrite=True)
    assert target.read_bytes() == b"old bytes"


def test_partitioned_cli_receipt(tmp_path, capsys):
    source = wav(tmp_path / "source.wav")
    plan = tmp_path / "room.json"
    plan.write_bytes(example_plan().to_bytes())
    output = tmp_path / "render.wav"
    args = ["room-response", "--plan", str(plan), "--source", str(source),
            "--output", str(output), "--engine", "partitioned"]
    assert main(args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["engine"] == "partitioned"
    assert receipt["protocol"] == "sonumio-verified-room-wav-v2"
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("tail", [False, True])
def test_room_api_backend_parity(tail):
    from sonumio.audio_core.room_response import render_room
    plan = example_plan()
    audio = make_buffer(np.random.default_rng(98).normal(0, .1, 211), 8000)
    old = render_room(audio, plan.room, plan.sources, plan.receivers,
                      plan.config, include_tail=tail)
    new = render_room(audio, plan.room, plan.sources, plan.receivers,
                      plan.config, include_tail=tail, engine="partitioned")
    assert old.spec == new.spec
    np.testing.assert_allclose(old.samples, new.samples, atol=1e-7)
    with pytest.raises(AudioContractError, match="Unknown room convolution engine"):
        render_room(audio, plan.room, plan.sources, plan.receivers,
                    plan.config, engine=None)
