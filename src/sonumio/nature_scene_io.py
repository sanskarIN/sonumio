"""Strict scene interchange for the original controllable nature DSP engine."""
from __future__ import annotations

from dataclasses import asdict,fields
import json

from .nature_scene import (
    NatureScene,NatureLayer,AutomationPoint,MAX_LAYERS,MAX_KEYS,
    PROTOCOL,scene_digest,
)
from .nature_synthesis import NatureVoice
from .errors import GenerationError

FORMAT="sonumio-nature-scene-json-v1"
MAX_SCENE_BYTES=128*1024


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise GenerationError("Duplicate nature JSON key")
        result[key]=value
    return result


def _nonfinite(value):
    raise GenerationError("Nonfinite nature JSON number")


def _fields(value,cls):
    if type(value) is not dict or set(value)!={f.name for f in fields(cls)}:
        raise GenerationError("Unexpected nature scene fields")
    return value


def encode_scene(scene):
    if not isinstance(scene,NatureScene):
        raise GenerationError("Expected NatureScene")
    payload={"format":FORMAT,"protocol":PROTOCOL,
             "sha256":scene_digest(scene),"scene":asdict(scene)}
    data=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    if len(data)>MAX_SCENE_BYTES:
        raise GenerationError("Nature scene serialization exceeds size budget")
    return data


def _layer(raw):
    values=dict(_fields(raw,NatureLayer))
    values["voice"]=NatureVoice(**_fields(values["voice"],NatureVoice))
    for name in ("gain","pan"):
        points=values[name]
        if type(points) is not list or not 1<=len(points)<=MAX_KEYS:
            raise GenerationError("Invalid nature automation array")
        values[name]=tuple(AutomationPoint(**_fields(p,AutomationPoint)) for p in points)
    return NatureLayer(**values)


def decode_scene(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_SCENE_BYTES:
        raise GenerationError("Invalid bounded nature scene bytes")
    if expected_sha256 is not None and (
            type(expected_sha256) is not str or len(expected_sha256)!=64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise GenerationError("Expected canonical lowercase scene SHA256")
    try:
        obj=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_nonfinite)
        if type(obj) is not dict or set(obj)!={"format","protocol","sha256","scene"}:
            raise GenerationError("Invalid scene envelope")
        if obj["format"]!=FORMAT or obj["protocol"]!=PROTOCOL:
            raise GenerationError("Unsupported nature scene format")
        values=dict(_fields(obj["scene"],NatureScene))
        if type(values["layers"]) is not list or not 1<=len(values["layers"])<=MAX_LAYERS:
            raise GenerationError("Invalid nature layer array")
        values["layers"]=tuple(_layer(item) for item in values["layers"])
        scene=NatureScene(**values)
        digest=scene_digest(scene)
        if obj["sha256"]!=digest or (expected_sha256 is not None and expected_sha256!=digest):
            raise GenerationError("Nature scene digest mismatch")
        return scene
    except (UnicodeError,RecursionError,TypeError,OverflowError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed nature scene JSON") from exc


def read_scene(path,*,expected_sha256=None):
    with open(path,"rb") as stream:
        data=stream.read(MAX_SCENE_BYTES+1)
    return decode_scene(data,expected_sha256=expected_sha256)
