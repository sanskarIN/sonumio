"""Verified export and original procedural source integration."""
import hashlib
import json
import os
from dataclasses import replace
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError,AudioIntegrityError,AudioLimitError
from sonumio.audio_core.granular import GranularSettings
from sonumio.audio_core.granular_wav import (
    GranularExportOptions,ProceduralGrainSource,export_granular_wav,export_procedural_grains,
)
from sonumio.audio_core.waveio import encode_wav_bytes,read_wav_file
from sonumio.audio_core.granular_cli import main


@pytest.fixture
def source(tmp_path):
    path=tmp_path/"input.wav"
    wave=(.2*np.sin(np.arange(2400)*.2)).astype(np.float32)
    path.write_bytes(encode_wav_bytes(make_buffer(wave,24000)))
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


def settings():
    return GranularSettings(4801,grain_frames=256,hop_frames=97,seed=31,
        position_jitter=.3,pitch_jitter_semitones=3,reverse_probability=.4,
        stereo_spread=.8,time_jitter_frames=16)


@pytest.mark.parametrize("bits",[8,16,24,32])
def test_export_geometry_digest_and_partition(source,tmp_path,bits):
    path,pin=source
    cfg=GranularExportOptions(settings(),bits,37,17)
    a=export_granular_wav(path,tmp_path/"a.wav",options=cfg,expected_sha256=pin)
    b=export_granular_wav(path,tmp_path/"b.wav",options=replace(cfg,packet_frames=8192),
        expected_sha256=pin,expected_output_sha256=a.output_sha256)
    assert a.output_sha256==b.output_sha256
    assert a.output_pcm_sha256==b.output_pcm_sha256
    assert a.source["source_sha256"]==pin
    assert a.output_bits==bits
    out=read_wav_file(tmp_path/"a.wav")
    assert out.frames==4801 and out.channels==2 and out.sample_rate==24000
    assert a.output_bytes==(tmp_path/"a.wav").stat().st_size
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
    json.dumps(a.as_dict(),allow_nan=False)


@pytest.mark.parametrize("kind",["rain","wind","ocean","thunder","birds","stream","fire"])
def test_original_nature_sources(tmp_path,kind):
    source=ProceduralGrainSource(kind,frames=4800,seed=12)
    cfg=GranularExportOptions(GranularSettings(4096,grain_frames=128,hop_frames=64,
        stereo_spread=.6,seed=19))
    path=tmp_path/(kind+".wav")
    result=export_procedural_grains(source,path,options=cfg)
    assert result.source["kind"]=="original_procedural_nature"
    assert result.render.source_frames==4800
    assert result.render.output_channels==2
    assert result.render.output_peak>0
    assert not result.as_dict()["model_inference"]
    repeated=export_procedural_grains(source,tmp_path/(kind+"-again.wav"),options=cfg,
        expected_source_pcm_sha256=result.source["source_pcm_sha256"],
        expected_output_sha256=result.output_sha256)
    assert repeated.output_sha256==result.output_sha256


def test_failed_pins_do_not_publish(source,tmp_path):
    path,pin=source
    cfg=GranularExportOptions(settings())
    target=tmp_path/"missing"/"out.wav"
    with pytest.raises(AudioIntegrityError):
        export_granular_wav(path,target,options=cfg,expected_sha256="0"*64,create_parent=True)
    assert not target.parent.exists()
    with pytest.raises(AudioIntegrityError):
        export_procedural_grains(ProceduralGrainSource(),target,options=cfg,
            expected_source_pcm_sha256="0"*64,create_parent=True)
    assert not target.parent.exists()


def test_existing_output_and_aliases(source,tmp_path):
    path,pin=source
    cfg=GranularExportOptions(settings())
    target=tmp_path/"existing.wav"
    target.write_bytes(b"original")
    with pytest.raises(AudioContractError):
        export_granular_wav(path,target,options=cfg)
    with pytest.raises(AudioIntegrityError):
        export_granular_wav(path,target,options=cfg,overwrite=True,expected_output_sha256="0"*64)
    assert target.read_bytes()==b"original"
    with pytest.raises(AudioContractError,match="must differ"):
        export_granular_wav(path,path,options=cfg,overwrite=True)
    alias=tmp_path/"alias.wav"
    try:
        os.link(path,alias)
    except OSError:
        pytest.skip("hardlink unavailable")
    with pytest.raises(AudioContractError,match="file identity"):
        export_granular_wav(path,alias,options=cfg,overwrite=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin


def test_export_budget(source,tmp_path,monkeypatch):
    import sonumio.audio_core.granular_wav as module
    monkeypatch.setattr(module,"MAX_EXPORT_PCM_BYTES",100)
    path,_=source
    target=tmp_path/"out.wav"
    with pytest.raises(AudioLimitError):
        export_granular_wav(path,target,options=GranularExportOptions(settings()))
    assert not target.exists()


def test_cli_procedural_and_local(source,tmp_path,capsys):
    path,pin=source
    opts=["--frames","4096","--grain-frames","128","--hop-frames","64","--stereo-spread",".8"]
    a=tmp_path/"a.wav"
    b=tmp_path/"b.wav"
    assert main(["granular","--nature","rain","--output",str(a),*opts])==0
    report=json.loads(capsys.readouterr().out)
    assert report["source"]["kind"]=="original_procedural_nature"
    assert main(["granular","--input",str(path),"--output",str(b),"--expected-sha256",pin,*opts])==0
    report=json.loads(capsys.readouterr().out)
    assert report["source"]["source_sha256"]==pin
    assert a.exists() and b.exists()


def test_cli_rejects_incompatible_pins(tmp_path,capsys):
    out=tmp_path/"bad.wav"
    assert main(["granular","--nature","rain","--frames","1024","--output",str(out),
                 "--expected-sha256","0"*64])==2
    assert "Procedural source uses" in capsys.readouterr().err
    assert not out.exists()
