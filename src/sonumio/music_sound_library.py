"""Reusable, content-addressed original timbres independent of a score.

The library contains only validated declarative oscillator and noise
parameters. A separate assignment binds those entries to a specific score
after checking its actual pitches and durations. Library names are labels,
not paths, and are never interpreted as executable expressions.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
import hashlib
import json
import re

from .arrangement import Arrangement,PITCHED,TRACKS,arrangement_digest
from .errors import GenerationError
from .music_patch_synthesis import InstrumentPatch
from .music_patch_bank import PatchBank,PatchRoute,decode_patch,describe_patch_bank
from .music_drum_patches import DrumPatch,DRUMS
from .music_drum_bank import DrumBank,DrumRoute,decode_drum_patch,describe_combined_banks
from .music_patch_presets import NAMES,preset
from .music_drum_presets import PRESETS,drum_preset
from .music_score_io import _object,_unique,_nonfinite

FORMAT="sonumio-original-sound-library-v1"
ENTRY_FORMAT="sonumio-original-sound-entry-v1"
MAX_ENTRIES=64
MAX_BYTES=512*1024
NAME_PATTERN=re.compile(r"[a-z][a-z0-9-]{0,47}\Z")
KINDS=("pitched",*DRUMS)


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _valid_name(value):
    if type(value) is not str or not NAME_PATTERN.fullmatch(value):
        raise GenerationError("Sound library name must be a lowercase ASCII identifier")
    return value


def _pin(value):
    if (type(value) is not str or len(value)!=64
            or any(c not in "0123456789abcdef" for c in value)):
        raise GenerationError("Invalid sound library SHA256")
    return value


@dataclass(frozen=True,slots=True)
class SoundEntry:
    name:str
    kind:str
    patch:InstrumentPatch|DrumPatch

    def __post_init__(self):
        _valid_name(self.name)
        if type(self.kind) is not str or self.kind not in KINDS:
            raise GenerationError("Unknown sound library entry kind")
        if not isinstance(self.patch,InstrumentPatch if self.kind=="pitched" else DrumPatch):
            raise GenerationError("Sound library patch differs from declared kind")

    @property
    def timbre_sha256(self):
        # Name is not a synthesis parameter. Renaming an entry changes the
        # library identity but not its reusable timbre identity.
        payload={"kind":self.kind,"patch":asdict(self.patch)}
        return _sha(ENTRY_FORMAT.encode("ascii")+b"\0"+_canonical(payload))

    def record(self):
        return {"name":self.name,"kind":self.kind,"timbre_sha256":self.timbre_sha256,
                "patch":asdict(self.patch)}


@dataclass(frozen=True,slots=True)
class SoundLibrary:
    entries:tuple[SoundEntry,...]=()

    def __post_init__(self):
        if (type(self.entries) is not tuple or len(self.entries)>MAX_ENTRIES
                or any(not isinstance(e,SoundEntry) for e in self.entries)):
            raise GenerationError("Invalid sound library entries")
        names=tuple(e.name for e in self.entries)
        if names!=tuple(sorted(set(names))):
            raise GenerationError("Sound library names must be unique and canonical")

    @property
    def sha256(self):
        return _sha(FORMAT.encode("ascii")+b"\0"+_canonical(
            [{"name":e.name,"kind":e.kind,"timbre_sha256":e.timbre_sha256} for e in self.entries]))

    def get(self,name):
        _valid_name(name)
        for entry in self.entries:
            if entry.name==name:
                return entry
        raise GenerationError("Unknown sound library entry: "+name)


def make_library(entries):
    if type(entries) is not tuple or any(not isinstance(e,SoundEntry) for e in entries):
        raise GenerationError("Expected tuple of sound library entries")
    return SoundLibrary(tuple(sorted(entries,key=lambda e:e.name)))


def encode_library(library):
    if not isinstance(library,SoundLibrary):
        raise GenerationError("Expected SoundLibrary")
    data=_canonical({"format":FORMAT,"library_sha256":library.sha256,
                     "entries":[e.record() for e in library.entries]})
    if len(data)>MAX_BYTES:
        raise GenerationError("Sound library exceeds byte budget")
    return data


def decode_library(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_BYTES:
        raise GenerationError("Invalid sound library byte count")
    if expected_sha256 is not None:
        _pin(expected_sha256)
    try:
        raw=_object(json.loads(data.decode("utf-8"),object_pairs_hook=_unique,
                               parse_constant=_nonfinite),("format","library_sha256","entries"))
        if raw["format"]!=FORMAT or type(raw["entries"]) is not list or len(raw["entries"])>MAX_ENTRIES:
            raise GenerationError("Invalid sound library document")
        entries=[]
        for row in raw["entries"]:
            row=_object(row,("name","kind","timbre_sha256","patch"))
            if type(row["kind"]) is not str or row["kind"] not in KINDS:
                raise GenerationError("Invalid sound library entry kind")
            patch=decode_patch(row["patch"]) if row["kind"]=="pitched" else decode_drum_patch(row["patch"])
            entry=SoundEntry(row["name"],row["kind"],patch)
            if row["timbre_sha256"]!=entry.timbre_sha256:
                raise GenerationError("Sound library timbre digest mismatch")
            entries.append(entry)
        library=SoundLibrary(tuple(entries))
        if raw["library_sha256"]!=library.sha256 or (expected_sha256 is not None and expected_sha256!=library.sha256):
            raise GenerationError("Sound library identity mismatch")
        if _canonical(raw)!=encode_library(library):
            raise GenerationError("Sound library entries are not canonical")
        return library
    except (ValueError,TypeError,UnicodeError,OverflowError,RecursionError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed sound library") from exc


def read_library(path,*,expected_sha256=None):
    with open(path,"rb") as stream:
        return decode_library(stream.read(MAX_BYTES+1),expected_sha256=expected_sha256)


def original_library():
    entries=[SoundEntry("pitched-"+name,"pitched",preset(name)) for name in NAMES]
    entries.extend(SoundEntry(track+"-"+name,track,drum_preset(track,name))
                   for track,names in PRESETS.items() for name in names)
    return make_library(tuple(entries))


def library_from_banks(score,pitched=None,drums=None,*,prefix="custom"):
    _valid_name(prefix)
    if not isinstance(score,Arrangement) or (pitched is None and drums is None):
        raise GenerationError("Expected score and at least one patch bank")
    if pitched is not None:
        describe_patch_bank(score,pitched)
    if drums is not None:
        describe_combined_banks(score,drums,pitched)
    rows=[]
    if pitched is not None:
        rows.extend(SoundEntry(prefix+"-"+r.track,"pitched",r.patch) for r in pitched.routes)
    if drums is not None:
        rows.extend(SoundEntry(prefix+"-"+r.track,r.track,r.patch) for r in drums.routes)
    return make_library(tuple(rows))


def rename_entry(library,old,new):
    if not isinstance(library,SoundLibrary):
        raise GenerationError("Expected SoundLibrary")
    entry=library.get(old)
    _valid_name(new)
    if old==new:
        return library
    if any(e.name==new for e in library.entries):
        raise GenerationError("Sound library target name already exists")
    return make_library(tuple(SoundEntry(new,e.kind,e.patch) if e==entry else e for e in library.entries))


def merge_libraries(left,right):
    """Conservative name-based merge, with no silent overwriting."""
    if not isinstance(left,SoundLibrary) or not isinstance(right,SoundLibrary):
        raise GenerationError("Expected two sound libraries")
    entries={e.name:e for e in left.entries}
    conflicts=[]
    identical=[]
    added=[]
    for entry in right.entries:
        current=entries.get(entry.name)
        if current is None:
            entries[entry.name]=entry
            added.append(entry.name)
        elif current==entry:
            identical.append(entry.name)
        else:
            conflicts.append({"name":entry.name,"left_sha256":current.timbre_sha256,
                              "right_sha256":entry.timbre_sha256})
    report={"left_sha256":left.sha256,"right_sha256":right.sha256,"conflicts":conflicts,
            "identical":identical,"added":added,"merged":not conflicts}
    if conflicts:
        return None,report
    result=make_library(tuple(entries.values()))
    report["result_sha256"]=result.sha256
    return result,report
