"""Standard MIDI File type 1 export for original symbolic arrangements.

Uses explicit tempo, time signature, channels, programs and end markers.
MIDI stores the arrangement, not its procedural instrument timbres.
"""
from __future__ import annotations

import struct
from .arrangement import Arrangement, TRACKS, PITCHED
from .errors import GenerationError

PPQ=8000
PROGRAMS={"keys":0,"pad":89,"bass":33,"lead":80,"pluck":24}
CHANNELS={name:i for i,name in enumerate(PITCHED)}
MAX_MIDI_BYTES=1024*1024


def vlq(value):
    if type(value) is not int or not 0<=value<=0x0fffffff:
        raise GenerationError("Invalid MIDI variable length integer")
    out=[value&127]
    value>>=7
    while value:
        out.append((value&127)|128)
        value>>=7
    return bytes(reversed(out))


def midi_tick(plan,tick):
    plan.frame(tick)  # Reuse score-domain validation.
    beat,sub=divmod(tick,8)
    swing=round(plan.swing*1000)
    offset=sub*(1000+swing) if sub<=4 else 4*(1000+swing)+(sub-4)*(1000-swing)
    return beat*PPQ+offset


def _meta(kind,payload):
    return bytes((0xff,kind))+vlq(len(payload))+payload


def _track(events,end):
    events=sorted(events,key=lambda item:(item[0],item[1],item[2]))
    cursor=0
    out=bytearray()
    for tick,_,message in events:
        if not cursor<=tick<=end:
            raise GenerationError("Invalid MIDI event time")
        out.extend(vlq(tick-cursor))
        out.extend(message)
        cursor=tick
    out.extend(vlq(end-cursor))
    out.extend(b"\xff\x2f\x00")
    return b"MTrk"+struct.pack(">I",len(out))+out


def _conductor(score,end):
    plan=score.plan
    tempo=(60_000_000+plan.bpm//2)//plan.bpm
    events=[(0,0,_meta(3,b"Sonumio arrangement")),
            (0,1,_meta(0x51,tempo.to_bytes(3,"big"))),
            (0,2,_meta(0x58,bytes((4,2,24,8))))]
    for section in score.sections:
        tick=midi_tick(plan,section.first_bar*32)
        events.append((tick,3,_meta(6,section.kind.encode("ascii"))))
    return _track(events,end)


def _voice_track(score,name,end):
    channel=CHANNELS.get(name,9)
    events=[(0,0,_meta(3,name.encode("ascii")))]
    if name in PROGRAMS:
        events.append((0,1,bytes((0xc0|channel,PROGRAMS[name]))))
    for e in score.events:
        if e.track!=name:
            continue
        start=midi_tick(score.plan,e.tick)
        stop=midi_tick(score.plan,e.tick+e.length)
        velocity=max(1,min(127,round(e.velocity*127)))
        # Note-off priority before note-on at the same absolute tick.
        events.append((stop,2,bytes((0x80|channel,e.pitch,0))))
        events.append((start,3,bytes((0x90|channel,e.pitch,velocity))))
    return _track(events,end)


def encode_midi(score):
    if not isinstance(score,Arrangement):
        raise GenerationError("Expected Arrangement for MIDI export")
    end=midi_tick(score.plan,score.plan.ticks)
    tracks=[_conductor(score,end)]
    tracks.extend(_voice_track(score,name,end) for name in TRACKS)
    data=b"MThd"+struct.pack(">IHHH",6,1,len(tracks),PPQ)+b"".join(tracks)
    if len(data)>MAX_MIDI_BYTES:
        raise GenerationError("MIDI exceeds size budget")
    return data
