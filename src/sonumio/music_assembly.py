"""Digest-pinned assembly of original symbolic arrangement bar spans.

The output is a newly validated score. Sources are immutable and verified
before traversal. Tempo, swing, root and mode must match the target clock;
cutting a note across a segment boundary is rejected unless clipping was
explicitly selected. Percussion is re-synthesized at its new score position.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass,replace
import hashlib
import json
import re

from .arrangement import (
    Arrangement,ArrangementPlan,Section,Event,TRACKS,SECTIONS,MAX_BARS,MAX_EVENTS,
    TICKS_PER_BAR,event_key,arrangement_digest,
)
from .errors import GenerationError
from .music_score_io import _construct,_object,_unique,_nonfinite

FORMAT="sonumio-arrangement-assembly-v1"
MAX_RECIPE_BYTES=64*1024
MAX_SOURCES=8
MAX_SEGMENTS=16
MAX_REPEAT=8
MAX_ALIAS=32
POLICIES=("reject","clip")


def _integer(value,lo,hi,label):
    if type(value) is not int or not lo<=value<=hi:
        raise GenerationError("Invalid assembly "+label)
    return value


def _alias(value):
    if type(value) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}",value):
        raise GenerationError("Invalid assembly source alias")
    return value


def _digest(value):
    if type(value) is not str or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
        raise GenerationError("Invalid assembly source SHA256")
    return value


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,
                      allow_nan=False).encode("ascii")


@dataclass(frozen=True,slots=True)
class SourcePin:
    alias: str
    score_sha256: str

    def __post_init__(self):
        _alias(self.alias)
        _digest(self.score_sha256)


@dataclass(frozen=True,slots=True)
class Segment:
    source: str
    first_bar: int
    last_bar: int
    section: str
    repeat: int=1
    boundary: str="reject"

    def __post_init__(self):
        _alias(self.source)
        _integer(self.first_bar,0,MAX_BARS-1,"first bar")
        _integer(self.last_bar,self.first_bar+1,MAX_BARS,"last bar")
        _integer(self.repeat,1,MAX_REPEAT,"repeat count")
        if type(self.section) is not str or self.section not in SECTIONS:
            raise GenerationError("Unknown assembly destination section")
        if type(self.boundary) is not str or self.boundary not in POLICIES:
            raise GenerationError("Unknown assembly boundary policy")

    @property
    def bars(self):
        return (self.last_bar-self.first_bar)*self.repeat


@dataclass(frozen=True,slots=True)
class AssemblyPlan:
    plan: ArrangementPlan
    sources: tuple[SourcePin,...]
    segments: tuple[Segment,...]

    def __post_init__(self):
        if not isinstance(self.plan,ArrangementPlan):
            raise GenerationError("Expected assembly output plan")
        if (type(self.sources) is not tuple or not 1<=len(self.sources)<=MAX_SOURCES
                or any(not isinstance(s,SourcePin) for s in self.sources)):
            raise GenerationError("Invalid bounded assembly source pins")
        aliases=tuple(s.alias for s in self.sources)
        if aliases!=tuple(sorted(set(aliases))):
            raise GenerationError("Assembly source aliases must be unique and sorted")
        if (type(self.segments) is not tuple or not 1<=len(self.segments)<=MAX_SEGMENTS
                or any(not isinstance(s,Segment) for s in self.segments)):
            raise GenerationError("Invalid bounded assembly segments")
        if {s.source for s in self.segments}!=set(aliases):
            raise GenerationError("Assembly segments must use exactly the pinned sources")
        if sum(s.bars for s in self.segments)!=self.plan.bars:
            raise GenerationError("Assembly segments do not cover output bar count")
        if self.segments[-1].section!="finale":
            raise GenerationError("Assembly final segment must be a finale")

    @property
    def sha256(self):
        return hashlib.sha256(FORMAT.encode("ascii")+b"\0"+_canonical(asdict(self))).hexdigest()


def encode_assembly(plan):
    if not isinstance(plan,AssemblyPlan):
        raise GenerationError("Expected AssemblyPlan")
    data=_canonical({"format":FORMAT,"sha256":plan.sha256,"assembly":asdict(plan)})
    if len(data)>MAX_RECIPE_BYTES:
        raise GenerationError("Assembly recipe exceeds byte budget")
    return data


def _parse_recipe(raw):
    raw=_object(raw,("plan","sources","segments"))
    for key,limit in (("sources",MAX_SOURCES),("segments",MAX_SEGMENTS)):
        if type(raw[key]) is not list or not 1<=len(raw[key])<=limit:
            raise GenerationError("Invalid assembly recipe "+key)
    return AssemblyPlan(_construct(ArrangementPlan,raw["plan"]),
                        tuple(_construct(SourcePin,x) for x in raw["sources"]),
                        tuple(_construct(Segment,x) for x in raw["segments"]))


def decode_draft(data):
    """An unsigned human-authored recipe can be pinned explicitly."""
    if type(data) is not bytes or not 1<=len(data)<=MAX_RECIPE_BYTES:
        raise GenerationError("Invalid assembly draft byte count")
    try:
        obj=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_nonfinite)
        _object(obj,("format","assembly"))
        if obj["format"]!=FORMAT:
            raise GenerationError("Unsupported assembly draft format")
        return _parse_recipe(obj["assembly"])
    except (UnicodeError,RecursionError,TypeError,OverflowError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed assembly draft") from exc


def decode_assembly(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_RECIPE_BYTES:
        raise GenerationError("Invalid assembly recipe byte count")
    if expected_sha256 is not None:
        _digest(expected_sha256)
    try:
        obj=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_nonfinite)
        _object(obj,("format","sha256","assembly"))
        if obj["format"]!=FORMAT:
            raise GenerationError("Unsupported assembly recipe format")
        plan=_parse_recipe(obj["assembly"])
        if obj["sha256"]!=plan.sha256 or (expected_sha256 is not None and expected_sha256!=plan.sha256):
            raise GenerationError("Assembly recipe digest mismatch")
        return plan
    except (UnicodeError,RecursionError,TypeError,OverflowError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed assembly recipe") from exc


def read_assembly(path,*,expected_sha256=None):
    with open(path,"rb") as stream:
        return decode_assembly(stream.read(MAX_RECIPE_BYTES+1),expected_sha256=expected_sha256)


def _clock(plan):
    # Style and seed are intentionally excluded. The output declares its
    # own style metadata and percussion seed. Timing and scale must match.
    return plan.root,plan.mode,plan.bpm,round(plan.swing*1000)


def verify_sources(plan,sources):
    if not isinstance(plan,AssemblyPlan) or type(sources) is not dict:
        raise GenerationError("Expected assembly plan and source dictionary")
    if set(sources)!={s.alias for s in plan.sources}:
        raise GenerationError("Assembly source bindings differ from recipe")
    for pin in plan.sources:
        score=sources[pin.alias]
        if not isinstance(score,Arrangement):
            raise GenerationError("Expected validated arrangement for source "+pin.alias)
        if arrangement_digest(score)!=pin.score_sha256:
            raise GenerationError("Assembly source digest mismatch: "+pin.alias)
        if _clock(score.plan)!=_clock(plan.plan):
            raise GenerationError("Assembly source clock or tonality differs: "+pin.alias)
    return {pin.alias:pin.score_sha256 for pin in plan.sources}


@dataclass(frozen=True,slots=True)
class Span:
    events: tuple[Event,...]
    harmony: tuple[int,...]
    clipped_left: int
    clipped_right: int
    source_events: int

    @property
    def clipped(self):
        return self.clipped_left+self.clipped_right


def extract_span(score,segment):
    """Return events relative to the span, not a standalone score."""
    if not isinstance(score,Arrangement) or not isinstance(segment,Segment):
        raise GenerationError("Expected validated score and segment")
    if segment.last_bar>score.plan.bars:
        raise GenerationError("Assembly segment exceeds source bar count")
    lo=segment.first_bar*TICKS_PER_BAR
    hi=segment.last_bar*TICKS_PER_BAR
    result=[]
    left_count=right_count=source_count=0
    for event in score.events:
        start=event.tick
        stop=start+event.length
        if stop<=lo or start>=hi:
            continue
        source_count+=1
        left=start<lo
        right=stop>hi
        if (left or right) and segment.boundary=="reject":
            raise GenerationError("Assembly note crosses a segment boundary: "+segment.source+
                                  " / "+event.track+" / tick "+str(event.tick))
        left_count+=int(left)
        right_count+=int(right)
        clipped_start=max(start,lo)
        clipped_stop=min(stop,hi)
        result.append(replace(event,tick=clipped_start-lo,length=clipped_stop-clipped_start))
    return Span(tuple(sorted(result,key=event_key)),score.harmony[segment.first_bar:segment.last_bar],
                left_count,right_count,source_count)


def _sections(segments):
    sections=[]
    cursor=0
    for segment in segments:
        if sections and sections[-1].kind==segment.section:
            previous=sections.pop()
            sections.append(Section(previous.kind,previous.first_bar,previous.bars+segment.bars))
        else:
            sections.append(Section(segment.section,cursor,segment.bars))
        cursor+=segment.bars
    return tuple(sections)


def assemble(plan,sources):
    verified=verify_sources(plan,sources)
    events=[]
    harmony=[]
    trace=[]
    cursor=0
    for segment in plan.segments:
        source=sources[segment.source]
        span=extract_span(source,segment)
        span_bars=segment.last_bar-segment.first_bar
        predicted_count=len(events)+len(span.events)*segment.repeat
        if predicted_count>MAX_EVENTS:
            raise GenerationError("Assembly output exceeds event budget before allocation")
        for repetition in range(segment.repeat):
            offset=(cursor+repetition*span_bars)*TICKS_PER_BAR
            events.extend(replace(event,tick=event.tick+offset) for event in span.events)
            harmony.extend(span.harmony)
        trace.append({"source":segment.source,"source_score_sha256":verified[segment.source],
                      "source_first_bar":segment.first_bar,"source_last_bar":segment.last_bar,
                      "output_first_bar":cursor,"output_last_bar":cursor+segment.bars,
                      "repeat":segment.repeat,"section":segment.section,"boundary":segment.boundary,
                      "source_events":span.source_events,"output_events":len(span.events)*segment.repeat,
                      "left_clipped_endpoints":span.clipped_left*segment.repeat,
                      "right_clipped_endpoints":span.clipped_right*segment.repeat})
        cursor+=segment.bars
    if cursor!=plan.plan.bars or len(harmony)!=cursor:
        raise GenerationError("Assembly traversal did not cover target clock")
    # Full validation checks final tonic, section partition, scale, clock,
    # duplicate events and cross-segment polyphony. No implicit cadence or
    # silent collision resolution is performed.
    result=Arrangement(plan.plan,_sections(plan.segments),tuple(harmony),tuple(sorted(events,key=event_key)))
    if verify_sources(plan,sources)!=verified:
        raise GenerationError("Assembly source changed during traversal")
    return result,{
        "format":FORMAT,"assembly_sha256":plan.sha256,
        "source_score_sha256":verified,"result_score_sha256":arrangement_digest(result),
        "bars":result.plan.bars,"events":len(result.events),
        "events_by_track":{track:sum(e.track==track for e in result.events) for track in TRACKS},
        "sections":[asdict(s) for s in result.sections],"harmony":list(result.harmony),
        "segments":trace,"source_scores_unchanged":True,
        "clipped_endpoints":sum(row["left_clipped_endpoints"]+row["right_clipped_endpoints"] for row in trace),
        "output_seed":result.plan.seed,"percussion_re_synthesized_on_render":True,
        "audio_rendered":False,
    }


def describe_assembly(plan,sources):
    # Inspect performs the same validation and traversal as publication.
    _,report=assemble(plan,sources)
    return report
