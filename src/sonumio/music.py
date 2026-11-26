"""Original procedural instrumental composition and rendering (not AI).

A bounded symbolic score is generated before rendering. Independent
random streams control motif, articulation and percussion so that the
musical plan is not accidentally coupled to waveform noise. No melody
library, external recordings, pretrained weights or lyrics are used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import hashlib
import json

import numpy as np

from .errors import GenerationError
from .audio import SAMPLE_RATE, encode_wav_channels, metrics_channels

TICKS_PER_BEAT = 8
TICKS_PER_BAR = 32
MAX_BARS = 8
MAX_EVENTS = 512
PROTOCOL = "sonumio-original-symbolic-music-v1"
MODES = {"major": (0,2,4,5,7,9,11), "minor": (0,2,3,5,7,8,10)}
STYLES = ("calm","cinematic","rhythmic")
TRACKS = ("pad","bass","lead","kick","hat")


@dataclass(frozen=True, slots=True)
class MusicPlan:
    style: str = "calm"
    root: int = 60
    mode: str = "major"
    bpm: int = 100
    bars: int = 4
    seed: int = 0

    def __post_init__(self):
        if type(self.style) is not str or type(self.mode) is not str or self.style not in STYLES or self.mode not in MODES:
            raise GenerationError("Unsupported music style or mode")
        for name,value,low,high in (("root",self.root,48,71),("bpm",self.bpm,60,160),
                                    ("bars",self.bars,1,MAX_BARS),("seed",self.seed,0,2**32-1)):
            if type(value) is not int or not low<=value<=high:
                raise GenerationError("Invalid music "+name)

    @property
    def ticks(self):
        return self.bars*TICKS_PER_BAR

    def frame(self,tick):
        if type(tick) is not int or not 0<=tick<=self.ticks:
            raise GenerationError("Music tick outside score")
        # Quantize absolute positions, never accumulate rounded intervals.
        numerator=tick*60*SAMPLE_RATE
        denominator=self.bpm*TICKS_PER_BEAT
        return (2*numerator+denominator)//(2*denominator)


@dataclass(frozen=True, slots=True)
class Note:
    track: str
    pitch: int
    start: int
    duration: int
    velocity: float
    pan: float

    def __post_init__(self):
        if self.track not in TRACKS:
            raise GenerationError("Unknown music track")
        if (type(self.pitch) is not int or not 24<=self.pitch<=96
                or type(self.start) is not int or self.start<0
                or type(self.duration) is not int or self.duration<1):
            raise GenerationError("Invalid music note geometry")
        for value,low,high in ((self.velocity,0.,1.),(self.pan,-1.,1.)):
            if type(value) not in (int,float) or not low<=value<=high:
                raise GenerationError("Invalid music articulation")


@dataclass(frozen=True, slots=True)
class Score:
    plan: MusicPlan
    harmony: tuple[int,...]
    notes: tuple[Note,...]

    def __post_init__(self):
        if not isinstance(self.plan,MusicPlan):
            raise GenerationError("Invalid music plan")
        if (type(self.harmony) is not tuple or len(self.harmony)!=self.plan.bars
                or any(type(x) is not int or not 0<=x<=6 for x in self.harmony)
                or self.harmony[-1]!=0):
            raise GenerationError("Invalid music harmony")
        if (type(self.notes) is not tuple or not 1<=len(self.notes)<=MAX_EVENTS
                or any(not isinstance(n,Note) or n.start+n.duration>self.plan.ticks
                       for n in self.notes)):
            raise GenerationError("Invalid bounded music score")
        if self.notes!=tuple(sorted(self.notes,key=_note_key)):
            raise GenerationError("Music score must use canonical event ordering")


def _note_key(n):
    return (n.start,TRACKS.index(n.track),n.pitch,n.duration,n.velocity,n.pan)


def _degree(plan,degree):
    octave,index=divmod(degree,7)
    return plan.root+MODES[plan.mode][index]+octave*12


def _chord(plan,degree):
    return tuple(_degree(plan,degree+i) for i in (0,2,4))


def _voicing(plan,degree,previous):
    pitches=tuple(p%12 for p in _chord(plan,degree))
    choices=[]
    for inversion in range(3):
        ordered=pitches[inversion:]+pitches[:inversion]
        for first in range(48,73):
            if first%12!=ordered[0]:
                continue
            notes=[first]
            for pc in ordered[1:]:
                notes.append(notes[-1]+(pc-notes[-1])%12)
            if notes[-1]<=84:
                choices.append(tuple(notes))
    return min(choices,key=lambda x:(sum(abs(a-b) for a,b in zip(x,previous,strict=True)),
                                     abs(sum(x)/3-plan.root),x))


def _progression(plan,rng):
    options=((0,4,5,3),(0,3,1,4),(0,5,3,4)) if plan.mode=="major" else (
             (0,5,2,6),(0,3,5,4),(0,6,5,4))
    motif=options[int(rng.integers(len(options)))]
    values=[motif[i%4] for i in range(plan.bars)]
    values[-1]=0
    if plan.bars>=3:
        values[-2]=4
    return tuple(values)


def _melody_pitch(plan,degree,offset,previous,downbeat):
    candidates=range(plan.root, min(plan.root+25,96)+1)
    scale={_degree(plan,i)%12 for i in range(7)}
    chord={p%12 for p in _chord(plan,degree)}
    allowed=[p for p in candidates if p%12 in (chord if downbeat else scale)]
    target=previous+offset
    return min(allowed,key=lambda p:(abs(p-target),abs(p-previous),p))


def compose(plan):
    if not isinstance(plan,MusicPlan):
        raise GenerationError("Expected MusicPlan")
    rng=np.random.default_rng(np.random.SeedSequence([plan.seed,0x534F4E]))
    articulation=np.random.default_rng(np.random.SeedSequence([plan.seed,0x415254]))
    harmony=_progression(plan,rng)
    # A short interval motif repeated and varied in alternating phrases.
    motif=tuple(int(x) for x in rng.choice((-3,-2,-1,1,2,3,4),size=4))
    notes=[]
    previous_chord=(plan.root-12,plan.root-8,plan.root-5)
    previous_lead=plan.root+12
    def add(track,pitch,start,duration,velocity,pan=0.):
        notes.append(Note(track,pitch,start,duration,round(float(velocity),6),float(pan)))
    for bar,degree in enumerate(harmony):
        start=bar*TICKS_PER_BAR
        final=bar==plan.bars-1
        chord=_voicing(plan,degree,previous_chord)
        previous_chord=chord
        for pitch,pan in zip(chord,(-.55,0.,.55),strict=True):
            add("pad",pitch,start,32,.42 if plan.style=="calm" else .33,pan)
        root=_degree(plan,degree)-12
        bass_pattern=(0,16) if plan.style=="calm" else (0,8,16,24)
        for i,tick in enumerate(bass_pattern):
            add("bass",root if i%2==0 else _degree(plan,degree+4)-12,start+tick,
                14 if plan.style=="calm" else 7,.5)
        rhythm=(0,8,16,24) if plan.style!="rhythmic" else (0,6,12,16,22,28)
        for i,tick in enumerate(rhythm):
            next_tick=rhythm[i+1] if i+1<len(rhythm) else 32
            offset=motif[(i+bar%2)%4]
            if bar%4==2:
                offset=-offset
            pitch=_melody_pitch(plan,degree,offset,previous_lead,tick%8==0)
            if final and i==len(rhythm)-1:
                pitch=min((plan.root,plan.root+12,plan.root+24),key=lambda p:abs(p-previous_lead))
            previous_lead=pitch
            add("lead",pitch,start+tick,max(1,next_tick-tick-1),
                .55+float(articulation.uniform(-.07,.07)),.12)
        if plan.style!="calm":
            for tick in (0,16) if plan.style=="cinematic" else (0,12,16,24):
                add("kick",36,start+tick,4,.65)
            for tick in (4,12,20,28) if plan.style=="cinematic" else range(0,32,4):
                add("hat",72,start+tick,2,.22 if tick%8 else .32,-.2 if tick%8 else .2)
    return Score(plan,harmony,tuple(sorted(notes,key=_note_key)))


def _envelope(frames,attack,release):
    out=np.ones(frames,dtype=np.float64)
    a=min(attack,frames//2)
    r=min(release,frames-a)
    if a:
        out[:a]=np.sin(np.linspace(0,math.pi/2,a,endpoint=False))**2
    if r:
        out[-r:]=np.cos(np.linspace(0,math.pi/2,r))**2
    return out


def _tone(note,frames,noise_rng):
    t=np.arange(frames,dtype=np.float64)/SAMPLE_RATE
    frequency=440*2**((note.pitch-69)/12)
    angle=2*math.pi*frequency*t
    if note.track=="kick":
        phase=2*math.pi*(45*t+65*.035*(1-np.exp(-t/.035)))
        signal=np.sin(phase)*np.exp(-t/.09)
        env=_envelope(frames,48,240)
    elif note.track=="hat":
        white=noise_rng.standard_normal(frames+1)
        signal=np.diff(white)*.35*np.exp(-t/.035)
        env=_envelope(frames,24,120)
    else:
        harmonics={"pad":(1.,.35,.18,.09),"bass":(1.,.25,.1),
                   "lead":(1.,.3,.13,.06,.025)}[note.track]
        signal=np.zeros(frames,dtype=np.float64)
        for i,amplitude in enumerate(harmonics,1):
            if frequency*i<SAMPLE_RATE*.48:
                signal+=amplitude*np.sin(i*angle)
        signal/=sum(harmonics)
        if note.track=="pad":
            env=_envelope(frames,2400,3600)*(.92+.08*np.cos(2*math.pi*1.3*t))
        else:
            env=_envelope(frames,120 if note.track=="lead" else 96,960)
    return signal*env*note.velocity


def render(score):
    if not isinstance(score,Score):
        raise GenerationError("Expected validated music score")
    plan=score.plan
    frames=plan.frame(plan.ticks)
    output=np.zeros((2,frames),dtype=np.float64)
    gains={"pad":.20,"bass":.27,"lead":.35,"kick":.25,"hat":.13}
    rng=np.random.default_rng(np.random.SeedSequence([plan.seed,0x445255]))
    for note in score.notes:
        start=plan.frame(note.start)
        end=plan.frame(note.start+note.duration)
        wave=_tone(note,end-start,rng)*gains[note.track]
        angle=(note.pan+1)*math.pi/4
        output[0,start:end]+=wave*math.cos(angle)
        output[1,start:end]+=wave*math.sin(angle)
    # Short finite stereo cross-feed echo, not an unbounded reverb feedback loop.
    dry=output.copy()
    for delay,gain in ((.083,.10),(.137,.06)):
        shift=round(delay*SAMPLE_RATE)
        if shift<frames:
            output[:,shift:]+=dry[::-1,:-shift]*gain
    output*=_envelope(frames,240,1200)
    peak=float(np.max(np.abs(output)))
    if peak>.9:
        output*=.9/peak
    result=output.astype(np.float32)
    if not np.isfinite(result).all():
        raise GenerationError("Nonfinite procedural music output")
    return result


def score_digest(score):
    encoded=json.dumps(asdict(score),sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(PROTOCOL.encode()+b"\0"+encoded).hexdigest()


def generate_music(plan):
    score=compose(plan)
    audio=render(score)
    wav=encode_wav_channels(audio,SAMPLE_RATE)
    report={"protocol":PROTOCOL,"plan":asdict(plan),"frames":audio.shape[1],
            "sample_rate":SAMPLE_RATE,"channels":2,"events":len(score.notes),
            "harmony_degrees":list(score.harmony),"score_sha256":score_digest(score),
            "wav_sha256":hashlib.sha256(wav).hexdigest(),
            "track_counts":{track:sum(n.track==track for n in score.notes) for track in TRACKS},
            "metrics":metrics_channels(audio),"learned_model":False,
            "external_recordings_used":False,"perceptual_quality_established":False}
    return wav,report
