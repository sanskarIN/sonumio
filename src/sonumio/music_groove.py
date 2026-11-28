"""Score-pinned original groove generation with exact hit-count constraints."""
from __future__ import annotations

from dataclasses import asdict,dataclass,replace
import hashlib
import json
import struct

from .arrangement import Arrangement,Event,TRACKS,arrangement_digest,event_key,integer,real
from .errors import GenerationError
from .music_automation import _object,_unique,_constant
from .music_groove_solver import GrooveConstraints,GrooveObjective,solve_groove,SLOTS

FORMAT="sonumio-constrained-groove-v1"
DRUMS=("kick","snare","hat")
PITCHES={"kick":36,"snare":38,"hat":42}
STYLES=("balanced","syncopated","backbeat","sparse","rolling")
MAX_BYTES=64*1024
MAX_TOTAL_TRANSITIONS=12_000_000


@dataclass(frozen=True,slots=True)
class DrumRule:
    track:str
    constraints:GrooveConstraints
    style:str="balanced"
    identity_weight:int=3
    accent_weight:int=2
    gap_weight:int=1
    variation_weight:int=1
    velocity:float=.55
    accent_boost:float=.2
    length_ticks:int=2
    pan:float=0.

    def __post_init__(self):
        if type(self.track) is not str or self.track not in DRUMS:
            raise GenerationError("Expected percussion groove track")
        if not isinstance(self.constraints,GrooveConstraints):
            raise GenerationError("Expected groove constraints")
        if type(self.style) is not str or self.style not in STYLES:
            raise GenerationError("Unknown groove style")
        for key in ("identity_weight","accent_weight","gap_weight","variation_weight"):
            integer(getattr(self,key),0,16,"groove "+key)
        for key,lo,hi in (("velocity",.001,1.),("accent_boost",0.,1.),("pan",-1.,1.)):
            object.__setattr__(self,key,real(getattr(self,key),lo,hi,"groove "+key))
        integer(self.length_ticks,1,4,"groove note length")


@dataclass(frozen=True,slots=True)
class GroovePlan:
    score_sha256:str
    first_bar:int
    bars:int
    seed:int
    rules:tuple[DrumRule,...]
    preserve_intro:bool=True
    preserve_break:bool=True
    alternate_bars:bool=False

    def __post_init__(self):
        if (type(self.score_sha256) is not str or len(self.score_sha256)!=64
                or any(c not in "0123456789abcdef" for c in self.score_sha256)):
            raise GenerationError("Invalid groove score SHA256")
        integer(self.first_bar,0,31,"groove first bar")
        integer(self.bars,1,32-self.first_bar,"groove bars")
        integer(self.seed,0,2**32-1,"groove seed")
        if (type(self.rules) is not tuple or not 1<=len(self.rules)<=len(DRUMS)
                or any(not isinstance(r,DrumRule) for r in self.rules)):
            raise GenerationError("Invalid groove track rules")
        names=tuple(r.track for r in self.rules)
        if names!=tuple(t for t in DRUMS if t in names):
            raise GenerationError("Groove rules must be unique and canonical")
        for flag in (self.preserve_intro,self.preserve_break,self.alternate_bars):
            if type(flag) is not bool:
                raise GenerationError("Groove switches must be boolean")

    @property
    def sha256(self):
        return hashlib.sha256(encode_groove(self)).hexdigest()


def encode_groove(plan):
    if not isinstance(plan,GroovePlan):
        raise GenerationError("Expected GroovePlan")
    data=json.dumps({"format":FORMAT,**asdict(plan)},sort_keys=True,
                    separators=(",",":"),allow_nan=False).encode("ascii")
    if len(data)>MAX_BYTES:
        raise GenerationError("Groove plan exceeds byte budget")
    return data


def decode_groove(data):
    if type(data) is not bytes or not 1<=len(data)<=MAX_BYTES:
        raise GenerationError("Invalid groove document byte count")
    try:
        raw=_object(json.loads(data.decode("utf-8"),object_pairs_hook=_unique,
                               parse_constant=_constant),
                    ("format","score_sha256","first_bar","bars","seed","rules"),
                    ("preserve_intro","preserve_break","alternate_bars"))
        if raw["format"]!=FORMAT or type(raw["rules"]) is not list or not 1<=len(raw["rules"])<=3:
            raise GenerationError("Invalid groove document")
        rules=[]
        for row in raw["rules"]:
            record=_object(row,("track","constraints"),tuple(k for k in DrumRule.__dataclass_fields__
                                                           if k not in ("track","constraints")))
            constraints=_object(record["constraints"],("hits",),
                                ("minimum_gap","maximum_gap","mandatory","forbidden","cyclic"))
            for name in ("mandatory","forbidden"):
                if name in constraints:
                    if type(constraints[name]) is not list or len(constraints[name])>SLOTS:
                        raise GenerationError("Invalid groove constraint array")
                    constraints[name]=tuple(constraints[name])
            rules.append(DrumRule(**{**record,"constraints":GrooveConstraints(**constraints)}))
        return GroovePlan(**{k:v for k,v in raw.items() if k not in ("format","rules")},rules=tuple(rules))
    except (ValueError,TypeError,UnicodeError,OverflowError,RecursionError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed groove document") from exc


def _section(score,bar):
    return next(s.kind for s in score.sections if s.first_bar<=bar<s.first_bar+s.bars)


def _accent(track,style,slot):
    downbeat=slot%8==0
    beat=slot%4==0
    backbeat=slot in (4,12)
    offbeat=slot%4==2
    if style=="syncopated":
        return 3 if offbeat else 2 if slot%2 else 0
    if style=="rolling":
        return 2 if slot%2==0 else 1
    if style=="backbeat":
        return 4 if (backbeat if track=="snare" else downbeat) else 1 if beat else 0
    if style=="sparse":
        return 4 if downbeat else 1 if backbeat else 0
    if track=="kick":
        return 3 if downbeat else 2 if beat else 1 if offbeat else 0
    if track=="snare":
        return 4 if backbeat else 1 if offbeat else 0
    return 3 if offbeat else 2 if beat else 1 if slot%2==0 else 0


def _jitter(seed,track,bar,slot):
    payload=struct.pack("<IIII",seed,DRUMS.index(track),bar,slot)
    digest=hashlib.sha256(b"sonumio-groove-slot-v1\0"+payload).digest()
    return int.from_bytes(digest[:2],"little")%7-3


def _objective(rule,original,previous,seed,bar,alternate):
    costs=[]
    for slot in range(SLOTS):
        identity=-rule.identity_weight*4 if slot in original else 0
        accent=-rule.accent_weight*_accent(rule.track,rule.style,slot)
        variation=rule.variation_weight*_jitter(seed,rule.track,bar,slot)
        if alternate and previous is not None and slot in previous:
            variation+=rule.variation_weight*2
        costs.append(identity+accent+variation)
    return GrooveObjective(tuple(costs),rule.gap_weight)


def _materialize(rule,pattern,start,kind):
    result=[]
    section_gain={"intro":.65,"theme":.9,"variation":1.,"break":.55,"finale":1.}[kind]
    for i,slot in enumerate(pattern):
        tick=start+slot*2
        next_tick=start+(pattern[i+1]*2 if i+1<len(pattern) else 32)
        length=min(rule.length_ticks,next_tick-tick)
        accent=_accent(rule.track,rule.style,slot)/4.
        velocity=round(min(1.,max(.001,(rule.velocity+rule.accent_boost*accent)*section_gain)),6)
        result.append(Event(rule.track,PITCHES[rule.track],tick,length,velocity,rule.pan))
    return tuple(result)


def generate_groove(score,plan):
    if not isinstance(score,Arrangement) or not isinstance(plan,GroovePlan):
        raise GenerationError("Expected arrangement and groove plan")
    if arrangement_digest(score)!=plan.score_sha256:
        raise GenerationError("Groove score SHA256 mismatch")
    if plan.first_bar+plan.bars>score.plan.bars:
        raise GenerationError("Groove region exceeds score")
    selected={}
    reports=[]
    previous={}
    transitions=0
    for bar in range(plan.first_bar,plan.first_bar+plan.bars):
        kind=_section(score,bar)
        if (kind=="intro" and plan.preserve_intro) or (kind=="break" and plan.preserve_break):
            continue
        start,stop=bar*32,(bar+1)*32
        for rule in plan.rules:
            source=tuple(e for e in score.events if e.track==rule.track and e.tick<stop and e.tick+e.length>start)
            if any(e.tick<start or e.tick+e.length>stop for e in source):
                raise GenerationError("Groove replacement boundary cuts a percussion note")
            original=tuple(sorted({(e.tick-start)//2 for e in source}))
            objective=_objective(rule,original,previous.get(rule.track),plan.seed,bar,plan.alternate_bars)
            pattern,receipt=solve_groove(rule.constraints,objective)
            transitions+=receipt["transitions"]
            if transitions>MAX_TOTAL_TRANSITIONS:
                raise GenerationError("Groove arrangement transition budget exceeded")
            selected[(bar,rule.track)]=_materialize(rule,pattern,start,kind)
            previous[rule.track]=pattern
            reports.append({"bar":bar,"section":kind,"track":rule.track,
                            "original_slots":list(original),"selected_slots":list(pattern),
                            "source_events":len(source),"new_events":len(pattern),**receipt})
    events=[e for e in score.events if (e.tick//32,e.track) not in selected]
    events.extend(e for rows in selected.values() for e in rows)
    result=replace(score,events=tuple(sorted(events,key=event_key)))
    return result,{
        "protocol":FORMAT,"source_score_sha256":plan.score_sha256,
        "result_score_sha256":arrangement_digest(result),"groove_sha256":plan.sha256,
        "bars_processed":len({bar for bar,_ in selected}),"tracks_processed":len(selected),
        "transitions":transitions,"rows":reports,"source_events":len(score.events),
        "result_events":len(result.events),"pitched_events_unchanged":True,
        "score_clock_unchanged":True,"harmony_unchanged":True,"original_unchanged":True,
        "learned_model":False,"perceptual_quality_established":False,
    }
