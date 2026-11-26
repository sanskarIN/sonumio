"""Versioned long-form symbolic arrangement, separate from music v1.

No external melodies or recordings. Exact beat clock, bounded event
counts, deterministic independent phrase streams and explicit sections.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np

from .audio import SAMPLE_RATE
from .errors import GenerationError
from .music import MODES, STYLES, TICKS_PER_BEAT, TICKS_PER_BAR

PROTOCOL = "sonumio-arrangement-v1"
MAX_BARS = 32
MAX_FRAMES = 120 * SAMPLE_RATE
MAX_EVENTS = 4096
TRACKS = ("keys", "pad", "bass", "lead", "pluck", "kick", "snare", "hat")
PITCHED = TRACKS[:5]
SECTIONS = ("intro", "theme", "variation", "break", "finale")
METER = 4


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise GenerationError("Invalid arrangement " + name)
    return value


def real(value, low, high, name):
    if type(value) not in (int, float) or not low <= value <= high:
        raise GenerationError("Invalid arrangement " + name)
    return float(value)


@dataclass(frozen=True, slots=True)
class ArrangementPlan:
    style: str = "cinematic"
    root: int = 60
    mode: str = "minor"
    bpm: int = 96
    bars: int = 16
    seed: int = 0
    swing: float = 0.0

    def __post_init__(self):
        if type(self.style) is not str or self.style not in STYLES:
            raise GenerationError("Unknown arrangement style")
        if type(self.mode) is not str or self.mode not in MODES:
            raise GenerationError("Unknown arrangement mode")
        integer(self.root,48,71,"root")
        integer(self.bpm,60,160,"bpm")
        integer(self.bars,4,MAX_BARS,"bars")
        integer(self.seed,0,2**32-1,"seed")
        real(self.swing,0,.6,"swing")
        if self.frame(self.ticks)>MAX_FRAMES:
            raise GenerationError("Arrangement exceeds 120 seconds")

    @property
    def ticks(self):
        return self.bars*TICKS_PER_BAR

    def frame(self, tick):
        integer(tick,0,self.ticks,"tick")
        # Delay the midpoint of every beat, using piecewise-linear timing.
        # Beat/bar endpoints remain exactly aligned and the map is monotone.
        beat, sub = divmod(tick,TICKS_PER_BEAT)
        s = round(self.swing * 1000)
        if sub <= 4:
            scaled = sub*(1000+s)
        else:
            scaled = 4*(1000+s)+(sub-4)*(1000-s)
        units = beat*TICKS_PER_BEAT*1000+scaled
        numerator=units*60*SAMPLE_RATE
        denominator=self.bpm*TICKS_PER_BEAT*1000
        return (2*numerator+denominator)//(2*denominator)


@dataclass(frozen=True, slots=True)
class Section:
    kind: str
    first_bar: int
    bars: int

    def __post_init__(self):
        if self.kind not in SECTIONS:
            raise GenerationError("Unknown arrangement section")
        integer(self.first_bar,0,MAX_BARS-1,"section start")
        integer(self.bars,1,MAX_BARS,"section length")


@dataclass(frozen=True, slots=True)
class Event:
    track: str
    pitch: int
    tick: int
    length: int
    velocity: float
    pan: float

    def __post_init__(self):
        if type(self.track) is not str or self.track not in TRACKS:
            raise GenerationError("Unknown arrangement instrument")
        integer(self.pitch,24,96,"pitch")
        integer(self.tick,0,MAX_BARS*TICKS_PER_BAR-1,"event start")
        integer(self.length,1,MAX_BARS*TICKS_PER_BAR,"event length")
        real(self.velocity,.001,1,"velocity")
        real(self.pan,-1,1,"pan")


def event_key(e):
    return (e.tick,TRACKS.index(e.track),e.pitch,e.length,e.velocity,e.pan)


@dataclass(frozen=True, slots=True)
class Arrangement:
    plan: ArrangementPlan
    sections: tuple[Section,...]
    harmony: tuple[int,...]
    events: tuple[Event,...]

    def __post_init__(self):
        if not isinstance(self.plan,ArrangementPlan):
            raise GenerationError("Expected ArrangementPlan")
        if type(self.sections) is not tuple or not 1<=len(self.sections)<=16:
            raise GenerationError("Invalid arrangement sections")
        cursor=0
        for section in self.sections:
            if not isinstance(section,Section) or section.first_bar!=cursor:
                raise GenerationError("Sections must exactly partition the score")
            cursor+=section.bars
        if cursor!=self.plan.bars or self.sections[-1].kind!="finale":
            raise GenerationError("Incomplete arrangement structure")
        if (type(self.harmony) is not tuple or len(self.harmony)!=self.plan.bars
                or any(type(d) is not int or not 0<=d<=6 for d in self.harmony)
                or self.harmony[-1]!=0):
            raise GenerationError("Invalid arrangement harmony")
        if (type(self.events) is not tuple or not 1<=len(self.events)<=MAX_EVENTS
                or any(not isinstance(e,Event) or e.tick+e.length>self.plan.ticks
                       for e in self.events)):
            raise GenerationError("Invalid arrangement events")
        if tuple(sorted(self.events,key=event_key))!=self.events:
            raise GenerationError("Arrangement events must be canonical")
        if len(set(self.events))!=len(self.events):
            raise GenerationError("Duplicate arrangement events")
        scale={(self.plan.root+p)%12 for p in MODES[self.plan.mode]}
        if any(e.track in PITCHED and e.pitch%12 not in scale for e in self.events):
            raise GenerationError("Out-of-scale pitched event")
        # Per-track polyphony limits protect render cost and avoid ambiguous
        # note-off semantics when exporting same-pitch MIDI events.
        for track in TRACKS:
            active={}
            bounds=[]
            for e in self.events:
                if e.track==track:
                    bounds.extend(((e.tick,1,e.pitch),(e.tick+e.length,-1,e.pitch)))
            count=0
            for _,delta,pitch in sorted(bounds):
                count+=delta
                active[pitch]=active.get(pitch,0)+delta
                if active[pitch]>1 or count>(4 if track in ("keys","pad") else 1):
                    raise GenerationError("Arrangement polyphony exceeded")


def _rng(plan,domain,bar=0):
    return np.random.default_rng(np.random.SeedSequence([plan.seed,domain,bar]))


def sections(plan):
    if plan.bars<=6:
        counts=(1,plan.bars-2,1)
        kinds=("intro","theme","finale")
    else:
        intro=max(1,plan.bars//8)
        finale=max(2,plan.bars//4)
        pause=max(1,plan.bars//8)
        remaining=plan.bars-intro-finale-pause
        theme=max(1,remaining//2)
        counts=(intro,theme,remaining-theme,pause,finale)
        kinds=("intro","theme","variation","break","finale")
    output=[]
    cursor=0
    for kind,count in zip(kinds,counts,strict=True):
        if count:
            output.append(Section(kind,cursor,count))
            cursor+=count
    return tuple(output)


def degree_pitch(plan,d):
    octaves,index=divmod(d,7)
    return plan.root+MODES[plan.mode][index]+12*octaves


def chord_classes(plan,d):
    return tuple(degree_pitch(plan,d+i)%12 for i in (0,2,4))


def harmony_for(plan,parts):
    choices=((0,4,5,3),(0,3,1,4),(0,5,3,4)) if plan.mode=="major" else (
             (0,5,2,6),(0,3,5,4),(0,6,5,4))
    rng=_rng(plan,0x4841524D)
    base=choices[int(rng.integers(len(choices)))]
    output=[]
    for section in parts:
        for i in range(section.bars):
            index=i%4
            d=base[index]
            if section.kind=="variation":
                d=(base[index]+2)%7
            elif section.kind=="break":
                d=3 if i%2==0 else 5
            elif section.kind=="intro":
                d=0 if i==0 else base[index]
            if section.kind=="finale" and i==section.bars-1:
                d=0
            output.append(d)
    output[-2:]=[4,0]
    return tuple(output)


def voice_chord(plan,d,previous):
    classes=chord_classes(plan,d)
    candidates=[]
    for inversion in range(3):
        pcs=classes[inversion:]+classes[:inversion]
        for first in range(48,73):
            if first%12!=pcs[0]:
                continue
            chord=[first]
            for pc in pcs[1:]:
                chord.append(chord[-1]+(pc-chord[-1])%12)
            if chord[-1]<=84:
                candidates.append(tuple(chord))
    return min(candidates,key=lambda c:(sum(abs(a-b) for a,b in zip(c,previous)),
                                         abs(sum(c)/3-plan.root),c))


def _lead_pitch(plan,d,target,previous,strong):
    scale=set(chord_classes(plan,d)) if strong else {(plan.root+x)%12 for x in MODES[plan.mode]}
    candidates=[p for p in range(plan.root,min(96,plan.root+24)+1) if p%12 in scale]
    return min(candidates,key=lambda p:(abs(p-target),abs(p-previous),p))


def compose_arrangement(plan):
    if not isinstance(plan,ArrangementPlan):
        raise GenerationError("Expected ArrangementPlan")
    parts=sections(plan)
    harmony=harmony_for(plan,parts)
    motif=tuple(int(x) for x in _rng(plan,0x4D4F5449).choice((-4,-2,-1,1,2,3,5),size=8))
    previous_chord=(plan.root-12,plan.root-8,plan.root-5)
    previous_lead=plan.root+12
    out=[]
    def emit(track,pitch,tick,length,velocity,pan=0):
        out.append(Event(track,pitch,tick,length,round(float(velocity),6),float(pan)))
    for section in parts:
        for local in range(section.bars):
            bar=section.first_bar+local
            start=bar*TICKS_PER_BAR
            d=harmony[bar]
            final=bar==plan.bars-1
            rng=_rng(plan,0x41525449,bar)
            chord=voice_chord(plan,d,previous_chord)
            previous_chord=chord
            energy={"intro":.55,"theme":.8,"variation":.9,"break":.48,"finale":1.}[section.kind]
            for pitch,pan in zip(chord,(-.65,0,.65),strict=True):
                emit("pad",pitch,start,32,.44*energy,pan)
            # Broken chords on keys leave space for the melody.
            keys=(0,8,16,24) if plan.style=="calm" else (0,6,12,16,22,28)
            for i,tick in enumerate(keys):
                length=(keys[i+1] if i+1<len(keys) else 32)-tick
                emit("keys",chord[i%3],start+tick,max(1,length-1),.46*energy,-.3)
            if section.kind!="intro" or local>0:
                pattern=(0,16) if section.kind=="break" or plan.style=="calm" else (0,8,16,24)
                for i,tick in enumerate(pattern):
                    pitch=degree_pitch(plan,d+(4 if i%2 else 0))-12
                    while pitch>59:
                        pitch-=12
                    emit("bass",pitch,start+tick,14 if len(pattern)==2 else 7,.65*energy)
            lead_ticks=(0,8,16,24) if plan.style!="rhythmic" else (0,6,12,16,22,28)
            if section.kind=="break":
                lead_ticks=(0,16)
            if section.kind!="intro" or bar==0:
                for i,tick in enumerate(lead_ticks):
                    offset=motif[(i+local*2)%len(motif)]
                    if section.kind=="variation":
                        offset=-offset
                    pitch=_lead_pitch(plan,d,previous_lead+offset,previous_lead,tick%8==0)
                    if final and i==len(lead_ticks)-1:
                        pitch=min((plan.root,plan.root+12,plan.root+24),key=lambda p:abs(p-previous_lead))
                    previous_lead=pitch
                    stop=lead_ticks[i+1] if i+1<len(lead_ticks) else 32
                    emit("lead",pitch,start+tick,max(1,stop-tick-1),
                         (.64+float(rng.uniform(-.05,.05)))*energy,.18)
            if section.kind in ("variation","finale"):
                for i,tick in enumerate((4,12,20,28)):
                    emit("pluck",chord[i%3]+12,start+tick,3,.36*energy,.5)
            if plan.style!="calm" and section.kind not in ("intro","break"):
                for tick in ((0,16) if plan.style=="cinematic" else (0,12,16,24)):
                    emit("kick",36,start+tick,4,.75*energy)
                for tick in (8,24):
                    emit("snare",38,start+tick,4,.52*energy)
                for tick in range(0,32,4):
                    emit("hat",42,start+tick,2,(.42 if tick%8==0 else .24)*energy,
                         .27 if tick%8==0 else -.27)
    return Arrangement(plan,parts,harmony,tuple(sorted(out,key=event_key)))


def arrangement_digest(score):
    if not isinstance(score,Arrangement):
        raise GenerationError("Expected Arrangement")
    data=json.dumps(asdict(score),sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(PROTOCOL.encode()+b"\0"+data).hexdigest()
