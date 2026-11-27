"""Contact scheduling, strict JSON, file integrity and CLI coverage."""
import hashlib
import json
from dataclasses import replace
import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError,AudioIntegrityError,AudioLimitError
from sonumio.audio_core.contact_patterns import ContactPattern,pattern_scene
from sonumio.audio_core.contact_io import parse_contact_scene,export_contact_scene
from sonumio.audio_core.contact_cli import main
from sonumio.audio_core.modal_synthesis import render_contact_scene
from sonumio.audio_core.waveio import read_wav_file


def test_pattern_timing_is_exact_on_sample_clock():
    pattern=ContactPattern(count=8,interval_ms=137,event_ms=105,
                           initial_delay_ms=17,sample_rate=44100)
    scene=pattern_scene(pattern)
    starts=[round((17+index*137)*44100/1000) for index in range(8)]
    assert [e.start_frame for e in scene.events]==starts
    assert scene.frames==starts[-1]+(105*44100+500)//1000
    assert [e.pan for e in scene.events]==[-.25,.25]*4


def test_seeded_variation_and_timing_bounds():
    pattern=ContactPattern(count=10,interval_ms=150,timing_jitter_ms=40,seed=19)
    a=pattern_scene(pattern)
    assert a==pattern_scene(pattern)
    assert a!=pattern_scene(replace(pattern,seed=20))
    for index,event in enumerate(a.events):
        assert abs(event.start_frame-index*3600)<=960
        assert .425<=event.strength<=.5750000000000001
        assert abs(event.pitch_semitones)<=1
    assert len(set(e.seed for e in a.events))==10
    assert [e.start_frame for e in a.events]==sorted(set(e.start_frame for e in a.events))


def test_parse_pattern_and_explicit_scene():
    pattern=parse_contact_scene(b'{"pattern":{"material":"gravel","count":3,"seed":42}}')
    assert pattern==pattern_scene(ContactPattern(material="gravel",count=3,seed=42))
    explicit=parse_contact_scene(b'{"sample_rate":8000,"frames":1000,"channels":1,"events":[{"start_frame":0,"frames":1000,"resonator":{"material":"metal"}}]}')
    assert explicit.channels==1
    assert render_contact_scene(explicit).report.output_peak>0
    custom=parse_contact_scene(b'{"sample_rate":8000,"frames":1000,"events":[{"start_frame":0,"frames":1000,"resonator":{"modes":[{"frequency_hz":440,"decay_seconds":0.1,"amplitude":1}],"contact_noise":0}}]}')
    assert custom.events[0].resonator.modes[0].frequency_hz==440


@pytest.mark.parametrize("raw",[
    b'{}',b'{"pattern":{},"frames":1000}',b'{"pattern":{"seed":1,"seed":2}}',
    b'{"pattern":{"unknown":1}}',b'{"pattern":{"count":true}}',
    b'{"pattern":{"strength":NaN}}',b'{"events":[]}',
    b'{"sample_rate":8000,"frames":1000,"events":[{"start_frame":0,"frames":1000,"resonator":{"material":"wood","modes":[]}}]}',
    b'{"sample_rate":8000,"frames":1000,"events":[{"start_frame":0,"frames":1000,"resonator":{"modes":[{"frequency_hz":440,"decay_seconds":0.1,"amplitude":1,"unknown":0}]}}]}',
])
def test_strict_scene_validation(raw):
    with pytest.raises((AudioContractError,TypeError)):
        parse_contact_scene(raw)


@pytest.mark.parametrize("bits",[8,16,24,32])
def test_verified_publication_and_repeat(tmp_path,bits):
    scene=pattern_scene(ContactPattern(count=3,event_ms=80,interval_ms=100,seed=8))
    a=tmp_path/"a.wav"
    b=tmp_path/"b.wav"
    first=export_contact_scene(scene,a,output_bits=bits,dither_seed=1)
    second=export_contact_scene(scene,b,output_bits=bits,dither_seed=1,
                                expected_output_sha256=first.output_sha256)
    assert first==second
    assert first.output_sha256==hashlib.sha256(a.read_bytes()).hexdigest()
    audio=read_wav_file(a)
    assert audio.frames==scene.frames and audio.channels==2
    assert first.output_bytes==a.stat().st_size
    json.dumps(first.as_dict(),allow_nan=False)


def test_failed_export_preserves_old_file(tmp_path):
    scene=pattern_scene(ContactPattern(count=2))
    path=tmp_path/"existing.wav"
    path.write_bytes(b"old")
    with pytest.raises(AudioContractError):
        export_contact_scene(scene,path)
    with pytest.raises(AudioIntegrityError):
        export_contact_scene(scene,path,overwrite=True,expected_output_sha256="0"*64)
    assert path.read_bytes()==b"old"


def test_pattern_and_export_limits(tmp_path,monkeypatch):
    with pytest.raises(AudioLimitError):
        ContactPattern(count=512,interval_ms=10000)
    with pytest.raises(AudioContractError):
        ContactPattern(interval_ms=100,timing_jitter_ms=34)
    import sonumio.audio_core.contact_io as module
    monkeypatch.setattr(module,"MAX_EXPORT_BYTES",1)
    with pytest.raises(AudioLimitError):
        export_contact_scene(pattern_scene(ContactPattern(count=1)),tmp_path/"out.wav")


def test_cli_pattern_and_json(tmp_path,capsys):
    a=tmp_path/"a.wav"
    assert main(["contact-sound","--material","wood","--count","2",
                 "--event-ms","80","--output",str(a)])==0
    receipt=json.loads(capsys.readouterr().out)
    assert len(receipt["render"]["event_reports"])==2
    scene=tmp_path/"scene.json"
    scene.write_text('{"pattern":{"material":"paper","count":2,"event_ms":80}}')
    b=tmp_path/"b.wav"
    assert main(["contact-sound","--scene",str(scene),"--output",str(b)])==0
    assert json.loads(capsys.readouterr().out)["output_sha256"]==hashlib.sha256(b.read_bytes()).hexdigest()


def test_cli_json_override_rejected(tmp_path,capsys):
    scene=tmp_path/"scene.json"
    scene.write_text('{"pattern":{"count":2}}')
    out=tmp_path/"out.wav"
    assert main(["contact-sound","--scene",str(scene),"--count","3","--output",str(out)])==2
    assert "cannot override" in capsys.readouterr().err
    assert not out.exists()
