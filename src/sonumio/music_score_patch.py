"""Digest-pinned, reversible structural patches for original scores.

Event identity uses the exact canonical JSON representation instead of
Python dataclass equality: integers and floats compare equal in Python
but may have different score digests. Patches are minimal exact event
set differences with source and target metadata snapshots.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .arrangement import (
    Arrangement, ArrangementPlan, Event, Section, MAX_EVENTS, MAX_BARS,
    arrangement_digest, event_key,
)
from .music_score_io import _construct, _object, _unique, _nonfinite
from .errors import GenerationError

FORMAT = "sonumio-score-patch-v1"
MAX_PATCH_BYTES = 2 * 1024 * 1024
MAX_CHAIN = 16


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")


def _digest(value,label):
    if (type(value) is not str or len(value)!=64
            or any(c not in "0123456789abcdef" for c in value)):
        raise GenerationError("Invalid "+label+" SHA256")
    return value


def _event_identity(event):
    if not isinstance(event,Event):
        raise GenerationError("Expected validated patch event")
    return _canonical(asdict(event))


def _ordered(events):
    return tuple(sorted(events,key=lambda event:(event_key(event),_event_identity(event))))


def _event_index(events):
    result={}
    for event in events:
        identity=_event_identity(event)
        if identity in result:
            raise GenerationError("Duplicate exact score event")
        result[identity]=event
    return result


@dataclass(frozen=True,slots=True)
class ScoreMetadata:
    plan: ArrangementPlan
    sections: tuple[Section,...]
    harmony: tuple[int,...]

    def __post_init__(self):
        if not isinstance(self.plan,ArrangementPlan):
            raise GenerationError("Expected patch arrangement plan")
        if (type(self.sections) is not tuple or not 1<=len(self.sections)<=16
                or any(not isinstance(s,Section) for s in self.sections)):
            raise GenerationError("Invalid patch section metadata")
        cursor=0
        for section in self.sections:
            if section.first_bar!=cursor:
                raise GenerationError("Patch sections must partition score")
            cursor+=section.bars
        if cursor!=self.plan.bars or self.sections[-1].kind!="finale":
            raise GenerationError("Invalid patch finale or section length")
        if (type(self.harmony) is not tuple or len(self.harmony)!=self.plan.bars
                or any(type(d) is not int or not 0<=d<=6 for d in self.harmony)
                or self.harmony[-1]!=0):
            raise GenerationError("Invalid patch harmony metadata")

    @classmethod
    def from_score(cls,score):
        if not isinstance(score,Arrangement):
            raise GenerationError("Expected Arrangement")
        return cls(score.plan,score.sections,score.harmony)

    @property
    def identity(self):
        return _canonical(asdict(self))

    def with_events(self,events):
        return Arrangement(self.plan,self.sections,self.harmony,_ordered(events))


@dataclass(frozen=True,slots=True)
class Endpoint:
    score_sha256: str
    event_count: int
    metadata: ScoreMetadata

    def __post_init__(self):
        _digest(self.score_sha256,"patch endpoint")
        if type(self.event_count) is not int or not 1<=self.event_count<=MAX_EVENTS:
            raise GenerationError("Invalid patch endpoint event count")
        if not isinstance(self.metadata,ScoreMetadata):
            raise GenerationError("Invalid patch endpoint metadata")

    @classmethod
    def from_score(cls,score):
        return cls(arrangement_digest(score),len(score.events),ScoreMetadata.from_score(score))

    def verify(self,score):
        if not isinstance(score,Arrangement):
            raise GenerationError("Expected validated patch source score")
        if arrangement_digest(score)!=self.score_sha256:
            raise GenerationError("Patch endpoint score digest mismatch")
        if len(score.events)!=self.event_count or ScoreMetadata.from_score(score).identity!=self.metadata.identity:
            raise GenerationError("Patch endpoint metadata or count differs")


def _validate_events(events,label):
    if (type(events) is not tuple or len(events)>MAX_EVENTS
            or any(not isinstance(e,Event) for e in events)):
        raise GenerationError("Invalid bounded patch "+label+" event tuple")
    identities=tuple(_event_identity(e) for e in events)
    if len(set(identities))!=len(events) or len(set(events))!=len(events):
        raise GenerationError("Duplicate patch "+label+" event")
    if identities!=tuple(_event_identity(e) for e in _ordered(events)):
        raise GenerationError("Patch "+label+" events must be canonical")
    return set(identities)


@dataclass(frozen=True,slots=True)
class ScorePatch:
    source: Endpoint
    target: Endpoint
    removed: tuple[Event,...]
    added: tuple[Event,...]

    def __post_init__(self):
        if not isinstance(self.source,Endpoint) or not isinstance(self.target,Endpoint):
            raise GenerationError("Expected patch endpoints")
        removed=_validate_events(self.removed,"removed")
        added=_validate_events(self.added,"added")
        if removed&added:
            raise GenerationError("Nonminimal patch contains cancelling event identities")
        if len(removed)>self.source.event_count or len(added)>self.target.event_count:
            raise GenerationError("Patch event changes exceed endpoint counts")
        if self.source.event_count-len(removed)+len(added)!=self.target.event_count:
            raise GenerationError("Patch endpoint event count equation differs")
        if self.source.score_sha256==self.target.score_sha256:
            if removed or added or self.source.metadata.identity!=self.target.metadata.identity:
                raise GenerationError("Equal patch endpoints cannot have changes")
        elif not removed and not added and self.source.metadata.identity==self.target.metadata.identity:
            raise GenerationError("Distinct patch endpoints require a structural change")

    @property
    def sha256(self):
        return hashlib.sha256(FORMAT.encode("ascii")+b"\0"+_canonical(asdict(self))).hexdigest()

    def reversed(self):
        return ScorePatch(self.target,self.source,self.added,self.removed)


def make_patch(source,target):
    if not isinstance(source,Arrangement) or not isinstance(target,Arrangement):
        raise GenerationError("Expected two validated scores")
    a,b=_event_index(source.events),_event_index(target.events)
    return ScorePatch(Endpoint.from_score(source),Endpoint.from_score(target),
                      _ordered(a[key] for key in a.keys()-b.keys()),
                      _ordered(b[key] for key in b.keys()-a.keys()))


def apply_patch(source,patch):
    if not isinstance(patch,ScorePatch):
        raise GenerationError("Expected ScorePatch")
    patch.source.verify(source)
    remaining=_event_index(source.events)
    for event in patch.removed:
        key=_event_identity(event)
        if key not in remaining:
            raise GenerationError("Patch removes a missing source event")
        del remaining[key]
    for event in patch.added:
        key=_event_identity(event)
        if key in remaining:
            raise GenerationError("Patch adds an existing source event")
        remaining[key]=event
    if len(remaining)!=patch.target.event_count:
        raise GenerationError("Patch result event count mismatch")
    target=patch.target.metadata.with_events(remaining.values())
    patch.target.verify(target)
    # Reject structurally redundant or fabricated changes even if the
    # endpoint check passed. The canonical minimal delta is unique.
    canonical=make_patch(source,target)
    if canonical.sha256!=patch.sha256:
        raise GenerationError("Patch is not the canonical minimal delta")
    return target


def _metadata_changes(source,target):
    a,b=asdict(source),asdict(target)
    return [field for field in ("plan","sections","harmony")
            if _canonical(a[field])!=_canonical(b[field])]


def inspect_patch(patch):
    if not isinstance(patch,ScorePatch):
        raise GenerationError("Expected ScorePatch")
    from .arrangement import TRACKS
    return {"format":FORMAT,"patch_sha256":patch.sha256,
            "source_score_sha256":patch.source.score_sha256,
            "target_score_sha256":patch.target.score_sha256,
            "source_events":patch.source.event_count,"target_events":patch.target.event_count,
            "removed_events":len(patch.removed),"added_events":len(patch.added),
            "unchanged_events":patch.source.event_count-len(patch.removed),
            "metadata_changed":_metadata_changes(patch.source.metadata,patch.target.metadata),
            "tracks":{track:{"removed":sum(e.track==track for e in patch.removed),
                             "added":sum(e.track==track for e in patch.added)} for track in TRACKS},
            "noop":patch.source.score_sha256==patch.target.score_sha256,
            "endpoint_verified":False}


def apply_with_receipt(source,patch):
    result=apply_patch(source,patch)
    receipt=inspect_patch(patch)
    receipt["endpoint_verified"]=True
    receipt["original_unchanged"]=arrangement_digest(source)==patch.source.score_sha256
    return result,receipt


def chain_patches(source,patches):
    """Verify each intermediate score and collapse a bounded chain."""
    if not isinstance(source,Arrangement):
        raise GenerationError("Expected source arrangement for patch chain")
    if (type(patches) is not tuple or not 1<=len(patches)<=MAX_CHAIN
            or any(not isinstance(p,ScorePatch) for p in patches)):
        raise GenerationError("Expected bounded score patch chain")
    current=source
    trace=[]
    for patch in patches:
        current=apply_patch(current,patch)
        trace.append({"patch_sha256":patch.sha256,"result_score_sha256":arrangement_digest(current)})
    combined=make_patch(source,current)
    if arrangement_digest(apply_patch(source,combined))!=arrangement_digest(current):
        raise GenerationError("Collapsed patch chain did not replay")
    receipt=inspect_patch(combined)
    receipt.update({"endpoint_verified":True,"chain":trace,"intermediate_scores_verified":len(trace)})
    return combined,receipt


def encode_patch(patch):
    if not isinstance(patch,ScorePatch):
        raise GenerationError("Expected ScorePatch")
    data=_canonical({"format":FORMAT,"sha256":patch.sha256,"patch":asdict(patch)})
    if len(data)>MAX_PATCH_BYTES:
        raise GenerationError("Score patch exceeds byte budget")
    return data


def _parse_metadata(raw):
    obj=_object(raw,("plan","sections","harmony"))
    if (type(obj["sections"]) is not list or not 1<=len(obj["sections"])<=16
            or type(obj["harmony"]) is not list or not 1<=len(obj["harmony"])<=MAX_BARS):
        raise GenerationError("Invalid patch metadata arrays")
    return ScoreMetadata(_construct(ArrangementPlan,obj["plan"]),
                         tuple(_construct(Section,x) for x in obj["sections"]),
                         tuple(obj["harmony"]))


def _parse_endpoint(raw):
    obj=_object(raw,("score_sha256","event_count","metadata"))
    return Endpoint(obj["score_sha256"],obj["event_count"],_parse_metadata(obj["metadata"]))


def decode_patch(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_PATCH_BYTES:
        raise GenerationError("Invalid bounded patch JSON byte count")
    if expected_sha256 is not None:
        _digest(expected_sha256,"expected patch")
    try:
        obj=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_nonfinite)
        _object(obj,("format","sha256","patch"))
        if obj["format"]!=FORMAT:
            raise GenerationError("Unsupported score patch format")
        raw=_object(obj["patch"],("source","target","removed","added"))
        for key in ("removed","added"):
            if type(raw[key]) is not list or len(raw[key])>MAX_EVENTS:
                raise GenerationError("Invalid patch event array")
        patch=ScorePatch(_parse_endpoint(raw["source"]),_parse_endpoint(raw["target"]),
                         tuple(_construct(Event,row) for row in raw["removed"]),
                         tuple(_construct(Event,row) for row in raw["added"]))
        if obj["sha256"]!=patch.sha256 or (expected_sha256 is not None and expected_sha256!=patch.sha256):
            raise GenerationError("Score patch digest mismatch")
        return patch
    except (UnicodeError,RecursionError,TypeError,OverflowError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed score patch JSON") from exc


def read_patch(path,*,expected_sha256=None):
    with open(path,"rb") as stream:
        data=stream.read(MAX_PATCH_BYTES+1)
    return decode_patch(data,expected_sha256=expected_sha256)
