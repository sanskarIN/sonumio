"""Strict bounded JSON interchange for original symbolic arrangements."""
from __future__ import annotations

from dataclasses import asdict, fields
import json

from .arrangement import (
    Arrangement, ArrangementPlan, Section, Event, PROTOCOL,
    MAX_EVENTS, arrangement_digest,
)
from .errors import GenerationError

MAX_SCORE_BYTES=1024*1024
FORMAT="sonumio-score-json-v1"


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise GenerationError("Duplicate score JSON key")
        result[key]=value
    return result


def _nonfinite(value):
    raise GenerationError("Nonfinite score JSON number")


def _object(value,expected):
    if type(value) is not dict or set(value)!=set(expected):
        raise GenerationError("Unexpected score object fields")
    return value


def _construct(cls,record):
    return cls(**_object(record,(f.name for f in fields(cls))))


def encode_score(score):
    if not isinstance(score,Arrangement):
        raise GenerationError("Expected Arrangement")
    payload={"format":FORMAT,"protocol":PROTOCOL,
             "sha256":arrangement_digest(score),"score":asdict(score)}
    data=json.dumps(payload,sort_keys=True,separators=(",",":"),
                    ensure_ascii=True,allow_nan=False).encode("ascii")
    if len(data)>MAX_SCORE_BYTES:
        raise GenerationError("Score serialization exceeds byte budget")
    return data


def decode_score(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_SCORE_BYTES:
        raise GenerationError("Invalid score JSON byte count")
    if expected_sha256 is not None:
        if (type(expected_sha256) is not str or len(expected_sha256)!=64
                or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise GenerationError("Expected canonical lowercase score SHA256")
    try:
        obj=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_nonfinite)
        _object(obj,("format","protocol","sha256","score"))
        if obj["format"]!=FORMAT or obj["protocol"]!=PROTOCOL:
            raise GenerationError("Unsupported score format")
        raw=_object(obj["score"],("plan","sections","harmony","events"))
        for name,limit in (("sections",16),("harmony",32),("events",MAX_EVENTS)):
            if type(raw[name]) is not list or not 1<=len(raw[name])<=limit:
                raise GenerationError("Invalid score array "+name)
        score=Arrangement(_construct(ArrangementPlan,raw["plan"]),
                          tuple(_construct(Section,x) for x in raw["sections"]),
                          tuple(raw["harmony"]),
                          tuple(_construct(Event,x) for x in raw["events"]))
        digest=arrangement_digest(score)
        if obj["sha256"]!=digest or (expected_sha256 is not None and expected_sha256!=digest):
            raise GenerationError("Score digest mismatch")
        return score
    except (UnicodeError,RecursionError,TypeError,OverflowError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed score JSON") from exc


def read_score(path,*,expected_sha256=None):
    # Read at most one byte beyond the budget; never trust metadata size.
    with open(path,"rb") as stream:
        data=stream.read(MAX_SCORE_BYTES+1)
    return decode_score(data,expected_sha256=expected_sha256)
