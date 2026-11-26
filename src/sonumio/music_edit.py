"""Immutable, validated editing of original symbolic arrangements.

Every edit produces a new score with a new semantic digest. Timing edits
change the sample clock but not the symbolic beat positions. All event
geometry, scale membership, ordering and polyphony are revalidated by
the Arrangement constructor. The original score is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from .arrangement import (
    Arrangement, ArrangementPlan, Event, PITCHED, TRACKS, MAX_BARS,
    TICKS_PER_BAR, event_key, arrangement_digest, chord_classes,
)
from .errors import GenerationError

MAX_EDITS=32


def _integer(value,lo,hi,name):
    if type(value) is not int or not lo<=value<=hi:
        raise GenerationError("Invalid score edit "+name)
    return value


def _number(value,lo,hi,name):
    if type(value) not in (float,int) or not lo<=value<=hi:
        raise GenerationError("Invalid score edit "+name)
    return float(value)


def _score(value):
    if not isinstance(value,Arrangement):
        raise GenerationError("Expected validated Arrangement")
    return value


def _track(value):
    if type(value) is not str or value not in TRACKS:
        raise GenerationError("Unknown score edit track")
    return value


def _range(score,first,last):
    _integer(first,0,score.plan.bars-1,"first bar")
    _integer(last,first+1,score.plan.bars,"last bar")
    return first*TICKS_PER_BAR,last*TICKS_PER_BAR


def _replace_events(score,events):
    return replace(score,events=tuple(sorted(events,key=event_key)))


@dataclass(frozen=True,slots=True)
class Retime:
    bpm:int|None=None
    swing:float|None=None

    def __post_init__(self):
        if self.bpm is None and self.swing is None:
            raise GenerationError("Empty score retiming edit")
        if self.bpm is not None:
            _integer(self.bpm,60,160,"tempo")
        if self.swing is not None:
            _number(self.swing,0,.6,"swing")

    def apply(self,score):
        _score(score)
        fields={}
        if self.bpm is not None:
            fields["bpm"]=self.bpm
        if self.swing is not None:
            fields["swing"]=self.swing
        return replace(score,plan=replace(score.plan,**fields))


@dataclass(frozen=True,slots=True)
class Transpose:
    semitones:int

    def __post_init__(self):
        _integer(self.semitones,-23,23,"transpose semitones")

    def apply(self,score):
        _score(score)
        # Preserve every pitched interval exactly. Never silently wrap an
        # individual note into another octave to make the operation fit.
        root=score.plan.root+self.semitones
        _integer(root,48,71,"transposed tonic")
        events=[]
        for e in score.events:
            if e.track in PITCHED:
                pitch=e.pitch+self.semitones
                _integer(pitch,24,96,"transposed pitch")
                events.append(replace(e,pitch=pitch))
            else:
                events.append(e)
        return replace(score,plan=replace(score.plan,root=root),
                       events=tuple(sorted(events,key=event_key)))


@dataclass(frozen=True,slots=True)
class TrackArticulation:
    track:str
    first_bar:int
    last_bar:int
    velocity_scale:float=1.
    pan_offset:float=0.

    def __post_init__(self):
        _track(self.track)
        _integer(self.first_bar,0,MAX_BARS-1,"first bar")
        _integer(self.last_bar,self.first_bar+1,MAX_BARS,"last bar")
        _number(self.velocity_scale,.01,8,"velocity scale")
        _number(self.pan_offset,-2,2,"pan offset")

    def apply(self,score):
        _score(score)
        first,last=_range(score,self.first_bar,self.last_bar)
        events=[]
        for event in score.events:
            if event.track==self.track and first<=event.tick<last:
                # Explicit saturation is part of articulation editing,
                # not an implicit modification of pitches or timing.
                v=round(min(1.,max(.001,event.velocity*self.velocity_scale)),6)
                pan=round(min(1.,max(-1.,event.pan+self.pan_offset)),6)
                events.append(replace(event,velocity=v,pan=pan))
            else:
                events.append(event)
        return _replace_events(score,events)


@dataclass(frozen=True,slots=True)
class RemoveTrack:
    track:str
    first_bar:int
    last_bar:int

    def __post_init__(self):
        _track(self.track)
        _integer(self.first_bar,0,MAX_BARS-1,"first bar")
        _integer(self.last_bar,self.first_bar+1,MAX_BARS,"last bar")

    def apply(self,score):
        _score(score)
        first,last=_range(score,self.first_bar,self.last_bar)
        return _replace_events(score,[e for e in score.events
            if not(e.track==self.track and first<=e.tick<last)])


@dataclass(frozen=True,slots=True)
class LeadBar:
    bar:int
    notes:tuple[tuple[int,int,int,float],...]
    pan:float=.18

    def __post_init__(self):
        _integer(self.bar,0,MAX_BARS-1,"lead bar")
        _number(self.pan,-1,1,"lead pan")
        if type(self.notes) is not tuple or len(self.notes)>TICKS_PER_BAR:
            raise GenerationError("Invalid bounded lead replacement")
        end=0
        for note in self.notes:
            if type(note) is not tuple or len(note)!=4:
                raise GenerationError("Lead note requires onset, duration, pitch, velocity")
            onset,duration,pitch,velocity=note
            _integer(onset,0,TICKS_PER_BAR-1,"lead onset")
            _integer(duration,1,TICKS_PER_BAR-onset,"lead duration")
            _integer(pitch,24,96,"lead pitch")
            _number(velocity,.001,1,"lead velocity")
            if onset<end:
                raise GenerationError("Lead replacement overlaps or is unsorted")
            end=onset+duration

    def apply(self,score):
        _score(score)
        start,end=_range(score,self.bar,self.bar+1)
        # An existing note crossing the edited boundary is ambiguous:
        # reject rather than clipping it and changing an adjacent bar.
        for event in score.events:
            if event.track!="lead":
                continue
            stop=event.tick+event.length
            if event.tick<start<stop or event.tick<end<stop:
                raise GenerationError("Lead note crosses edited bar boundary")
        events=[e for e in score.events if not(e.track=="lead" and start<=e.tick<end)]
        for onset,length,pitch,velocity in self.notes:
            events.append(Event("lead",pitch,start+onset,length,velocity,self.pan))
        return _replace_events(score,events)


@dataclass(frozen=True,slots=True)
class QuantizeVelocity:
    steps:int=8

    def __post_init__(self):
        _integer(self.steps,2,127,"velocity steps")

    def apply(self,score):
        _score(score)
        events=[]
        for e in score.events:
            value=max(1,round(e.velocity*self.steps))/self.steps
            events.append(replace(e,velocity=value))
        return _replace_events(score,events)


OPERATIONS=(Retime,Transpose,TrackArticulation,RemoveTrack,LeadBar,QuantizeVelocity)


def edit(score,operations):
    _score(score)
    if (type(operations) is not tuple or not 1<=len(operations)<=MAX_EDITS
            or any(type(op) not in OPERATIONS for op in operations)):
        raise GenerationError("Invalid bounded score edit transaction")
    original=arrangement_digest(score)
    current=score
    trace=[]
    for op in operations:
        before=arrangement_digest(current)
        current=op.apply(current)
        after=arrangement_digest(current)
        trace.append({"operation":type(op).__name__,"before_sha256":before,"after_sha256":after,
                      "events_after":len(current.events)})
    if arrangement_digest(score)!=original:
        raise GenerationError("Score editing changed original")
    return current,{"source_score_sha256":original,"result_score_sha256":arrangement_digest(current),
                    "operations":trace,"source_events":len(score.events),
                    "result_events":len(current.events),"original_unchanged":True}
