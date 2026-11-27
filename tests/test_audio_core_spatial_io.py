"""Verified spatial binding, automatic duration and end-to-end export."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError,AudioIntegrityError,AudioLimitError
from sonumio.audio_core.hashing import canonical_json_bytes
from sonumio.audio_core.spatial_motion import Position,MotionKey,Trajectory,Receiver,Room,render_spatial_scene
from sonumio.audio_core.spatial_io import (
    SpatialPlacement,SpatialPlan,SourceBinding,parse_spatial_plan,export_spatial_plan,
)
from sonumio.audio_core.spatial_cli import main
from sonumio.audio_core.waveio import encode_wav_bytes,read_wav_file


@pytest.fixture
def sources(tmp_path):
    bindings={}
    for name,f in [("tone",200),("bell",440)]:
        audio=make_buffer((.2*np.sin(2*np.pi*f*np.arange(800)/8000)).astype(np.float32),8000)
        path=tmp_path/(name+".wav")
        data=encode_wav_bytes(audio)
        path.write_bytes(data)
        bindings[name]=SourceBinding(path,hashlib.sha256(data).hexdigest())
    return bindings


def plan():
    return SpatialPlan(8000,(
        SpatialPlacement("tone",Trajectory.stationary(3,4),start_frame=0),
        SpatialPlacement("bell",Trajectory.stationary(4,3),start_frame=200),
        SpatialPlacement("tone",Trajectory.stationary(2,4),start_frame=400,gain_db=-6),
    ),receiver=Receiver(Position(2,2),spacing_m=0,sound_speed=320),
        room=Room(6,6,left=.3,right=.2,bottom=0,top=0),tail_frames=10)


def test_auto_duration_includes_last_reflection(sources):
    cfg=plan()
    audios={k:read_wav_file(v.path) for k,v in sources.items()}
    scene=cfg.bind(audios)
    r=render_spatial_scene(scene)
    assert all(not p.truncated for p in r.report.paths)
    last=max(p.last_arrival_frame for p in r.report.paths)
    assert scene.frames==int(np.ceil(last))+1+cfg.tail_frames
    assert cfg.source_ids==("bell","tone")
    assert len(scene.sources)==3


def test_plan_json_roundtrip():
    cfg=plan()
    decoded=parse_spatial_plan(canonical_json_bytes(__import__("dataclasses").asdict(cfg)))
    assert decoded==cfg
    assert decoded.sha256==cfg.sha256


@pytest.mark.parametrize("raw",[
    b'{}',b'{"sample_rate":8000,"placements":[]}',
    b'{"sample_rate":8000,"placements":[],"path":"input.wav"}',
    b'{"sample_rate":8000,"placements":[],"sample_rate":24000}',
    b'{"sample_rate":8000,"placements":[{"source_id":"../x","trajectory":{"keys":[{"frame":0,"position":{"x":1,"y":2}}]}}]}',
    b'{"sample_rate":8000,"placements":[{"source_id":"a","trajectory":{"keys":[{"frame":0,"position":{"x":NaN,"y":2}}]}}]}',
    b'{"sample_rate":8000,"placements":[{"source_id":"a","trajectory":{"keys":[{"frame":0,"position":{"x":1,"y":2,"z":3}}]}}]}',
])
def test_invalid_json(raw):
    with pytest.raises((AudioContractError,TypeError)):
        parse_spatial_plan(raw)


@pytest.mark.parametrize("bits",[8,16,24,32])
def test_verified_export_is_deterministic(sources,tmp_path,bits):
    cfg=plan()
    a=tmp_path/"a.wav"
    b=tmp_path/"b.wav"
    first=export_spatial_plan(cfg,sources,a,output_bits=bits,dither_seed=3,packet_frames=19)
    second=export_spatial_plan(cfg,sources,b,output_bits=bits,dither_seed=3,packet_frames=8192,
        expected_output_sha256=first.output_sha256)
    assert first==second
    assert first.output_sha256==hashlib.sha256(a.read_bytes()).hexdigest()
    assert first.output_bits==bits
    assert first.plan_sha256==cfg.sha256
    assert read_wav_file(a).frames==first.render.frames
    assert set(first.source_receipts)==set(sources)
    for name,binding in sources.items():
        assert first.source_receipts[name]["source_sha256"]==binding.expected_sha256
        assert hashlib.sha256(binding.path.read_bytes()).hexdigest()==binding.expected_sha256
    json.dumps(first.as_dict(),allow_nan=False)


def test_invalid_bindings_and_bad_digest(sources,tmp_path):
    cfg=plan()
    out=tmp_path/"absent"/"out.wav"
    with pytest.raises(AudioContractError,match="exactly match"):
        export_spatial_plan(cfg,{"tone":sources["tone"]},out,create_parent=True)
    bad=dict(sources)
    bad["bell"]=replace(bad["bell"],expected_sha256="0"*64)
    with pytest.raises(AudioIntegrityError):
        export_spatial_plan(cfg,bad,out,create_parent=True)
    assert not out.parent.exists()


def test_no_clobber_and_source_alias(sources,tmp_path):
    cfg=plan()
    out=tmp_path/"existing.wav"
    out.write_bytes(b"old")
    with pytest.raises(AudioContractError):
        export_spatial_plan(cfg,sources,out)
    with pytest.raises(AudioIntegrityError):
        export_spatial_plan(cfg,sources,out,overwrite=True,expected_output_sha256="0"*64)
    assert out.read_bytes()==b"old"
    with pytest.raises(AudioContractError,match="differ"):
        export_spatial_plan(cfg,sources,sources["tone"].path,overwrite=True)


def test_explicit_truncation_and_tail_options(sources):
    cfg=plan()
    with pytest.raises(AudioContractError):
        replace(cfg,frames=1000)
    explicit=replace(cfg,frames=700,tail_frames=0)
    result=render_spatial_scene(explicit.bind({k:read_wav_file(v.path) for k,v in sources.items()}))
    assert any(p.truncated for p in result.report.paths)


def test_aggregate_input_budget(sources,tmp_path,monkeypatch):
    import sonumio.audio_core.spatial_io as module
    monkeypatch.setattr(module,"MAX_INPUT_PCM_BYTES",4000)
    target=tmp_path/"out.wav"
    with pytest.raises(AudioLimitError):
        export_spatial_plan(plan(),sources,target)
    assert not target.exists()


def test_cli_named_sources(sources,tmp_path,capsys):
    from dataclasses import asdict
    cfg=plan()
    path=tmp_path/"plan.json"
    path.write_bytes(canonical_json_bytes(asdict(cfg)))
    output=tmp_path/"out.wav"
    args=["spatial-scene","--plan",str(path),"--output",str(output)]
    for name,binding in sources.items():
        args.extend(["--source",name+"="+str(binding.path),
                     "--source-sha256",name+"="+binding.expected_sha256])
    assert main(args)==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["plan_sha256"]==cfg.sha256
    assert receipt["output_sha256"]==hashlib.sha256(output.read_bytes()).hexdigest()


def test_cli_duplicate_source_rejected(tmp_path,capsys):
    output=tmp_path/"out.wav"
    assert main(["spatial-scene","--plan",str(tmp_path/"missing.json"),
        "--source","a=one.wav","--source","a=two.wav","--output",str(output)])==2
    assert "unique" in capsys.readouterr().err
    assert not output.exists()
