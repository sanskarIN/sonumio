"""Real finite-Fourier preview smoke tests, run in the full Sonumio repo.

The isolated interchange workspace has no audio_core; CI of the whole project
runs these against the existing audio renderer, WAV codec and publication path.
"""
from dataclasses import replace
from hashlib import sha256
import json

import pytest

pytest.importorskip("sonumio.audio_core.wavetable")
from sonumio.midi_workstation.musicxml_export import encode_musicxml
from sonumio.midi_workstation.musicxml_preview import export_musicxml_preview
from sonumio.midi_workstation.musicxml_cli import main
from sonumio.midi_workstation.mxl import encode_mxl
from sonumio.midi_workstation.score import Score, ScoreNote
from sonumio.midi_workstation.score_preview import ScorePreviewOptions
from sonumio.midi_workstation.wavetable_bridge import WavetableBridgeSettings


def example():
    s=Score.empty(end=480)
    return replace(s,tracks=(replace(s.tracks[0],
        notes=(ScoreNote("original",0,240,69,100,0,64),)),))


def options(**kw):
    bridge=WavetableBridgeSettings(sample_rate=8000,channels=1,waveform="sine",
        table_size=512,max_harmonics=1,master_gain=.08,attack_frames=0,
        decay_frames=0,sustain_level=1.,release_frames=0)
    return ScorePreviewOptions(bridge=bridge,output_bits=16,**kw)


def test_real_musicxml_score_renders_original_wav(tmp_path):
    raw,_=encode_musicxml(example())
    result=tmp_path/"preview.wav"
    receipt,info=export_musicxml_preview(raw,result,options=options())
    data=result.read_bytes()
    assert data.startswith(b"RIFF") and data[8:12]==b"WAVE"
    assert len(data)>44
    assert receipt.wav_sha256==sha256(data).hexdigest()
    assert receipt.notes==1
    assert receipt.output_frames==4000
    assert receipt.output_channels==1
    assert receipt.sample_rate==8000
    assert info["render"]["trained_model_inference"] is False


def test_real_mxl_score_renders_same_original_waveform(tmp_path):
    xml,_=encode_musicxml(example())
    packed,_=encode_mxl(xml)
    output=tmp_path/"music.wav"
    report,_=export_musicxml_preview(packed,output,compressed=True,options=options())
    assert report.notes==1
    assert report.wav_sha256==sha256(output.read_bytes()).hexdigest()


def test_real_preview_is_deterministic_across_destinations(tmp_path):
    raw,_=encode_musicxml(example())
    a=tmp_path/"a.wav"
    b=tmp_path/"b.wav"
    export_musicxml_preview(raw,a,options=options())
    export_musicxml_preview(raw,b,options=options())
    assert a.read_bytes()==b.read_bytes()


def test_real_cli_preview_score(tmp_path,capsys):
    source=tmp_path/"source.musicxml"
    source.write_bytes(encode_musicxml(example())[0])
    dest=tmp_path/"output.wav"
    assert main(["preview-xml",str(source),"--output",str(dest),
                 "--sample-rate","8000","--channels","1","--waveform","sine",
                 "--gain","0.08","--bits","16"])==0
    data=json.loads(capsys.readouterr().out)
    assert data["preview"]["wav_sha256"]==sha256(dest.read_bytes()).hexdigest()
    assert data["preview"]["notes"]==1
    assert main(["preview-xml",str(source),"--output",str(dest)])==2
    assert capsys.readouterr().out==""


def test_real_preview_wav_does_not_overwrite_source(tmp_path):
    source=tmp_path/"source.musicxml"
    xml,_=encode_musicxml(example())
    source.write_bytes(xml)
    dest=tmp_path/"preview.wav"
    export_musicxml_preview(xml,dest,options=options(),source_path=source)
    assert source.read_bytes()==xml
