"""Score interchange, independent MIDI parsing and local CLI integration."""
from dataclasses import asdict
import hashlib
import json
import struct
import wave
import pytest

from sonumio.arrangement import ArrangementPlan, compose_arrangement, arrangement_digest, TRACKS
from sonumio.music_score_io import encode_score, decode_score, MAX_SCORE_BYTES
from sonumio.music_midi import encode_midi, midi_tick, vlq, PPQ, CHANNELS
from sonumio.music_publish import publish_music_artifact
from sonumio.arrangement_cli import main
from sonumio.errors import GenerationError


def sample():
    return compose_arrangement(ArrangementPlan(bars=4,bpm=137,swing=.33,seed=17))


def test_score_roundtrip_and_digest_pinning():
    score=sample()
    data=encode_score(score)
    assert decode_score(data,expected_sha256=arrangement_digest(score))==score
    assert encode_score(decode_score(data))==data
    with pytest.raises(GenerationError):
        decode_score(data,expected_sha256="0"*64)
    with pytest.raises(GenerationError):
        decode_score(data,expected_sha256="A"*64)


@pytest.mark.parametrize("transform",[
    lambda x:dict(x,format="unknown"),
    lambda x:dict(x,sha256="0"*64),
    lambda x:dict(x,extra=True),
    lambda x:dict(x,score=dict(x["score"],events=[])),
    lambda x:dict(x,score=dict(x["score"],plan=dict(x["score"]["plan"],bars=True))),
    lambda x:dict(x,score=dict(x["score"],harmony=[0]*4)),
])
def test_tampered_score_rejected(transform):
    raw=json.loads(encode_score(sample()))
    with pytest.raises(GenerationError):
        decode_score(json.dumps(transform(raw)).encode())


@pytest.mark.parametrize("data",[
    b"",b"null",b"[]",b'{"format":1,"format":2}',b'{"x":NaN}',b"\xff",
    b" "* (MAX_SCORE_BYTES+1),
])
def test_malformed_score_rejected(data):
    with pytest.raises(GenerationError):
        decode_score(data)


@pytest.mark.parametrize("value,expected",[
    (0,b"\x00"),(127,b"\x7f"),(128,b"\x81\x00"),(16383,b"\xff\x7f"),
    (16384,b"\x81\x80\x00"),(0xfffffff,b"\xff\xff\xff\x7f"),
])
def test_midi_vlq_known_values(value,expected):
    assert vlq(value)==expected


def _read_vlq(data,pos):
    value=0
    for _ in range(4):
        byte=data[pos]
        pos+=1
        value=(value<<7)|(byte&127)
        if byte<128:
            return value,pos
    raise AssertionError("Invalid MIDI VLQ")


def parse_midi(data):
    assert data[:4]==b"MThd"
    assert struct.unpack_from(">I",data,4)[0]==6
    format,tracks,division=struct.unpack_from(">HHH",data,8)
    assert format==1 and tracks==9 and division==PPQ
    position=14
    result=[]
    for _ in range(tracks):
        assert data[position:position+4]==b"MTrk"
        size=struct.unpack_from(">I",data,position+4)[0]
        start=position+8
        chunk=data[start:start+size]
        assert len(chunk)==size
        position=start+size
        tick=0
        i=0
        events=[]
        while i<len(chunk):
            delta,i=_read_vlq(chunk,i)
            tick+=delta
            status=chunk[i]
            i+=1
            if status==0xff:
                kind=chunk[i]
                i+=1
                count,i=_read_vlq(chunk,i)
                payload=chunk[i:i+count]
                i+=count
                events.append((tick,status,kind,payload))
            else:
                assert status&0xf0 in (0x80,0x90,0xc0)
                count=1 if status&0xf0==0xc0 else 2
                payload=chunk[i:i+count]
                i+=count
                assert all(x<128 for x in payload)
                events.append((tick,status,None,payload))
        assert events[-1][1:]==(0xff,0x2f,b"")
        assert i==len(chunk)
        result.append(events)
    assert position==len(data)
    return result


def test_midi_matches_every_score_event():
    score=sample()
    data=encode_midi(score)
    assert data==encode_midi(score)
    tracks=parse_midi(data)
    end=midi_tick(score.plan,score.plan.ticks)
    assert all(events[-1][0]==end for events in tracks)
    tempo=next(e[3] for e in tracks[0] if e[1:3]==(0xff,0x51))
    assert abs(int.from_bytes(tempo,"big")-60_000_000/score.plan.bpm)<=.5
    markers=[(t,p.decode()) for t,s,k,p in tracks[0] if (s,k)==(0xff,6)]
    assert markers==[(midi_tick(score.plan,s.first_bar*32),s.kind) for s in score.sections]
    for name,events in zip(TRACKS,tracks[1:],strict=True):
        channel=CHANNELS.get(name,9)
        actual_on=[(t,p[0],p[1]) for t,s,k,p in events if s==0x90|channel]
        actual_off=[(t,p[0],p[1]) for t,s,k,p in events if s==0x80|channel]
        original=[e for e in score.events if e.track==name]
        expected_on=sorted((midi_tick(score.plan,e.tick),e.pitch,max(1,round(e.velocity*127))) for e in original)
        expected_off=sorted((midi_tick(score.plan,e.tick+e.length),e.pitch,0) for e in original)
        assert actual_on==expected_on
        assert actual_off==expected_off
        active=set()
        for _,s,k,p in events:
            if s==0x90|channel:
                assert p[0] not in active
                active.add(p[0])
            elif s==0x80|channel:
                active.remove(p[0])
        assert not active


def test_score_artifact_atomic_no_clobber(tmp_path):
    path=tmp_path/"score.json"
    data=encode_score(sample())
    receipt=publish_music_artifact(path,data,kind="score")
    assert path.read_bytes()==data
    assert receipt.sha256==hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):
        publish_music_artifact(path,data,kind="score")
    with pytest.raises(GenerationError):
        publish_music_artifact(path,b"bad",kind="score",overwrite=True,expected_sha256="0"*64)
    assert path.read_bytes()==data
    assert not list(tmp_path.glob("*.partial"))


def test_full_cli_compose_inspect_midi_and_render(tmp_path,capsys):
    score=tmp_path/"score.json"
    midi=tmp_path/"score.mid"
    audio=tmp_path/"score.wav"
    assert main(["compose","--bars","4","--bpm","160","--seed","17",
                 "--output",str(score)])==0
    receipt=json.loads(capsys.readouterr().out)
    digest=receipt["score_sha256"]
    assert receipt["output_sha256"]==hashlib.sha256(score.read_bytes()).hexdigest()
    assert main(["inspect",str(score),"--expected-score-sha256",digest])==0
    assert json.loads(capsys.readouterr().out)["score_sha256"]==digest
    assert main(["midi",str(score),"--output",str(midi)])==0
    assert json.loads(capsys.readouterr().out)["output_sha256"]==hashlib.sha256(midi.read_bytes()).hexdigest()
    parse_midi(midi.read_bytes())
    assert main(["render",str(score),"--solo","lead","--output",str(audio)])==0
    report=json.loads(capsys.readouterr().out)
    assert report["output_sha256"]==hashlib.sha256(audio.read_bytes()).hexdigest()
    assert report["tracks"]["lead"]["active"]
    assert not report["tracks"]["pad"]["active"]
    with wave.open(str(audio),"rb") as wav:
        assert wav.getnchannels()==2 and wav.getframerate()==24000
        assert wav.getnframes()==144000
    with pytest.raises(SystemExit):
        main(["midi",str(score),"--output",str(midi)])
    with pytest.raises(SystemExit):
        main(["render",str(score),"--expected-score-sha256","0"*64,"--output",str(tmp_path/"bad.wav")])
    assert not (tmp_path/"bad.wav").exists()
