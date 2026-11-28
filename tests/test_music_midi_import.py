"""Standard MIDI parsing and constrained import regression oracles."""
from dataclasses import replace
import hashlib
import json
import struct

import pytest

from sonumio.arrangement import ArrangementPlan, compose_arrangement, arrangement_digest
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.music_midi import encode_midi, midi_tick, vlq
from sonumio.music_midi_read import decode_midi, inspect_midi, read_midi, duration_microseconds
from sonumio.music_midi_import import (
    Mapping, ImportPlan, decode_mapping, import_arrangement, infer_sonumio_mapping,
)
from sonumio.music_score_io import encode_score, read_score


def track(data):
    return b"MTrk"+struct.pack(">I",len(data))+data


def midi(*tracks,fmt=1,ppq=480):
    return b"MThd"+struct.pack(">IHHH",6,fmt,len(tracks),ppq)+b"".join(track(t) for t in tracks)


def msg(delta,data):
    return vlq(delta)+data


def eot(delta=0):
    return msg(delta,b"\xff\x2f\x00")


def tempo(value,delta=0):
    return msg(delta,b"\xff\x51\x03"+value.to_bytes(3,"big"))


def note(delta,pitch=60,velocity=100,channel=0):
    return msg(delta,bytes((0x90|channel,pitch,velocity)))


def off(delta,pitch=60,channel=0):
    return msg(delta,bytes((0x80|channel,pitch,0)))


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(style="rhythmic",bars=4,bpm=120,seed=17,swing=.2))


def test_original_export_preserves_note_lifetimes(score):
    raw=encode_midi(score)
    document=decode_midi(raw)
    report=inspect_midi(document)
    assert document.format==1 and document.ppq==8000
    assert len(document.tracks)==9
    assert len(document.notes)==len(score.events)
    assert report["source_sha256"]==hashlib.sha256(raw).hexdigest()
    assert report["end_tick"]==midi_tick(score.plan,score.plan.ticks)
    assert report["duration_microseconds"]==8000000
    observed=sorted((document.tracks[n.track].name,n.pitch,n.start,n.stop,n.velocity) for n in document.notes)
    expected=sorted((e.track,e.pitch,midi_tick(score.plan,e.tick),midi_tick(score.plan,e.tick+e.length),
                     max(1,min(127,round(e.velocity*127)))) for e in score.events)
    assert observed==expected


def test_running_status_and_velocity_zero_note_off():
    body=msg(0,b"\xc2\x21")+note(0,60,100,2)+msg(120,bytes((64,80)))
    body+=msg(120,bytes((60,0)))+msg(120,bytes((64,0)))+eot()
    document=decode_midi(midi(body,fmt=0))
    assert [(n.channel,n.pitch,n.start,n.stop,n.program) for n in document.notes]==[
        (2,60,0,240,33),(2,64,120,360,33)]


def test_tempo_map_integrates_variable_tempo():
    body=tempo(500000)+note(0)+off(480)+tempo(1000000)+eot(480)
    document=decode_midi(midi(body,fmt=0))
    assert duration_microseconds(document)==1500000
    assert inspect_midi(document)["tempo_map"]==[
        {"tick":0,"microseconds_per_beat":500000},{"tick":480,"microseconds_per_beat":1000000}]


def test_conflicting_tempos_are_rejected():
    document=decode_midi(midi(tempo(500000)+eot(),tempo(600000)+eot()))
    with pytest.raises(GenerationError,match="Conflicting"):
        inspect_midi(document)


def test_controller_and_sysex_report_without_applying():
    body=msg(0,b"\xb0\x40\x7f")+msg(0,b"\xf0\x02\x01\xf7")+note(0)+off(480)+eot()
    document=decode_midi(midi(body,fmt=0))
    report=inspect_midi(document)
    assert report["tracks"][0]["unapplied_channel_events"]=={"0xb0":1}
    assert report["tracks"][0]["sysex_events"]==1
    assert document.notes[0].stop==480


@pytest.mark.parametrize("body",[
    b"\x81\x81\x81\x81\x00"+b"\xff\x2f\x00",
    msg(0,b"\x3c\x40")+eot(),
    note(0)+note(10)+off(10)+eot(),
    off(10)+eot(),
    note(0)+eot(100),
    note(0)+off(0)+eot(),
    note(0)+off(10),
    eot()+b"\x00",
    msg(0,b"\xff\x2f\x01\x00"),
    msg(0,b"\xff\x51\x03\x00\x00\x00")+eot(),
    msg(0,b"\xff\x58\x04\x00\x02\x18\x08")+eot(),
    msg(0,b"\x90\x80\x40")+eot(),
    msg(0,b"\xf1\x00")+eot(),
    note(0)+msg(10,b"\xff\x01\x01x")+msg(10,b"\x3c\x00")+eot(),
])
def test_malformed_midi_is_rejected(body):
    with pytest.raises(GenerationError):
        decode_midi(midi(body,fmt=0))


@pytest.mark.parametrize("data",[
    b"",b"MThd"+b"\x00"*10,
    midi(eot(),eot(),fmt=0),midi(eot(),fmt=2),
    midi(eot(),ppq=0),midi(eot(),ppq=0x8000),
    midi(eot())+b"extra",midi(eot())[:-1],
    midi(*([eot()]*33)),
])
def test_invalid_header_and_chunk_geometry(data):
    with pytest.raises(GenerationError):
        decode_midi(data)


def test_message_limit_is_enforced():
    body=msg(0,b"\xb0\x01\x00")*32768+eot()
    with pytest.raises(GenerationError,match="budget"):
        decode_midi(midi(body,fmt=0))


def test_own_export_import_preserves_grid_and_explicit_losses(score):
    document=decode_midi(encode_midi(score))
    imported,report=import_arrangement(document,score)
    assert imported.plan==score.plan
    assert imported.sections==score.sections and imported.harmony==score.harmony
    assert [(e.track,e.pitch,e.tick,e.length) for e in imported.events]==[
        (e.track,e.pitch,e.tick,e.length) for e in score.events]
    assert report["quantized_endpoints"]==0
    assert report["exact_original_score_reconstructed"] is False
    assert report["pan_source"]=="explicit track mapping"
    assert encode_midi(imported)==encode_midi(score)


def _external(score,*,delta=0,controller=False,tempo_value=500000):
    # External PPQ 8000 permits the exact template swing grid.
    start=midi_tick(score.plan,8)+delta
    stop=midi_tick(score.plan,16)+delta
    end=midi_tick(score.plan,score.plan.ticks)
    pitch=score.plan.root
    body=tempo(tempo_value)
    if controller:
        body+=msg(0,b"\xb0\x40\x7f")
    body+=msg(0,b"\xc0\x00")+note(start,pitch)+off(stop-start,pitch)+eot(end-stop)
    return decode_midi(midi(body,fmt=0,ppq=8000))


def test_explicit_external_mapping_and_bounded_quantization(score):
    document=_external(score,delta=3)
    with pytest.raises(GenerationError,match="grid"):
        import_arrangement(document,score,ImportPlan((Mapping(0,0,"lead"),)))
    result,report=import_arrangement(document,score,ImportPlan((Mapping(0,0,"lead",-.2,0),),3))
    event=result.events[0]
    assert event.tick==8 and event.length==8 and event.pan==-.2
    assert report["quantized_endpoints"]==2
    assert report["max_source_tick_error"]==3.


@pytest.mark.parametrize("kwargs,match",[
    ({"controller":True},"performance controllers"),
    ({"tempo_value":600000},"tempo"),
])
def test_import_rejects_unsupported_semantics(score,kwargs,match):
    with pytest.raises(GenerationError,match=match):
        import_arrangement(_external(score,**kwargs),score,ImportPlan((Mapping(0,0,"lead"),)))


def test_mapping_must_cover_source_and_expected_program(score):
    document=_external(score)
    with pytest.raises(GenerationError,match="cover exactly"):
        import_arrangement(document,score,ImportPlan((Mapping(0,1,"lead"),)))
    with pytest.raises(GenerationError,match="program"):
        import_arrangement(document,score,ImportPlan((Mapping(0,0,"lead",expected_program=80),)))
    with pytest.raises(GenerationError):
        infer_sonumio_mapping(document)


@pytest.mark.parametrize("data",[
    b'{"format":"sonumio-midi-import-v1","max_tick_error":0,"mappings":[]}',
    b'{"format":"sonumio-midi-import-v1","max_tick_error":0,"max_tick_error":1,"mappings":[]}',
    b'{"format":"sonumio-midi-import-v1","max_tick_error":0,"mappings":[{"track":0,"channel":0,"target":"lead","pan":NaN}]}',
    b'{"format":"sonumio-midi-import-v1","max_tick_error":17,"mappings":[{"track":0,"channel":0,"target":"lead"}]}',
    b'{"format":"sonumio-midi-import-v1","max_tick_error":0,"mappings":[{"track":0,"channel":9,"target":"lead"}]}',
])
def test_invalid_mapping_json(data):
    with pytest.raises(GenerationError):
        decode_mapping(data)


def test_mapping_json_valid():
    raw=b'{"format":"sonumio-midi-import-v1","max_tick_error":3,"mappings":[{"track":0,"channel":0,"target":"keys","pan":-0.3}]}'
    assert decode_mapping(raw)==ImportPlan((Mapping(0,0,"keys",-.3),),3)


def test_cli_inspection_and_atomic_import(score,tmp_path,capsys):
    source=tmp_path/"source.mid"
    source.write_bytes(encode_midi(score))
    template=tmp_path/"template.json"
    template.write_bytes(encode_score(score))
    target=tmp_path/"output.json"
    expected=hashlib.sha256(source.read_bytes()).hexdigest()
    assert main(["music-midi-inspect",str(source),"--expected-midi-sha256",expected])==0
    assert json.loads(capsys.readouterr().out)["note_count"]==len(score.events)
    args=["music-midi-import",str(source),"--template",str(template),
          "--expected-template-sha256",arrangement_digest(score),"--output",str(target)]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert report["result_score_sha256"]==arrangement_digest(read_score(target))
    data=target.read_bytes()
    assert main(args)==2
    assert target.read_bytes()==data
    assert main(args+["--overwrite","--expected-output-sha256","0"*64])==2
    assert target.read_bytes()==data
    assert main(["music-midi-import",str(source),"--template",str(template),
                 "--output",str(template),"--overwrite"])==2
    with pytest.raises(GenerationError,match="SHA256"):
        read_midi(source,expected_sha256="0"*64)
