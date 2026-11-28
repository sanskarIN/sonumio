"""Explicit score-bound sound-library assignment and conservative revision transfer."""
from __future__ import annotations

from dataclasses import asdict,dataclass
import json

from .arrangement import Arrangement,PITCHED,TRACKS,arrangement_digest
from .errors import GenerationError
from .music_patch_bank import PatchBank,PatchRoute
from .music_drum_bank import DrumBank,DrumRoute,describe_combined_banks
from .music_sound_library import SoundLibrary,_canonical,_sha,_pin,_valid_name
from .music_score_io import _object,_unique,_nonfinite

FORMAT="sonumio-score-sound-assignment-v1"
MAX_BYTES=32*1024


@dataclass(frozen=True,slots=True)
class Assignment:
    track:str
    name:str
    timbre_sha256:str

    def __post_init__(self):
        if type(self.track) is not str or self.track not in TRACKS:
            raise GenerationError("Unknown sound assignment track")
        _valid_name(self.name)
        _pin(self.timbre_sha256)


@dataclass(frozen=True,slots=True)
class SoundAssignment:
    score_sha256:str
    library_sha256:str
    assignments:tuple[Assignment,...]

    def __post_init__(self):
        _pin(self.score_sha256)
        _pin(self.library_sha256)
        if (type(self.assignments) is not tuple or not 1<=len(self.assignments)<=len(TRACKS)
                or any(not isinstance(row,Assignment) for row in self.assignments)):
            raise GenerationError("Invalid sound assignment rows")
        names=tuple(row.track for row in self.assignments)
        if names!=tuple(track for track in TRACKS if track in names):
            raise GenerationError("Sound assignment tracks must be unique and canonical")

    @property
    def sha256(self):
        return _sha(encode_assignment(self))


def encode_assignment(plan):
    if not isinstance(plan,SoundAssignment):
        raise GenerationError("Expected SoundAssignment")
    data=_canonical({"format":FORMAT,**asdict(plan)})
    if len(data)>MAX_BYTES:
        raise GenerationError("Sound assignment exceeds byte budget")
    return data


def decode_assignment(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_BYTES:
        raise GenerationError("Invalid sound assignment byte count")
    if expected_sha256 is not None:
        _pin(expected_sha256)
    try:
        raw=_object(json.loads(data.decode("utf-8"),object_pairs_hook=_unique,
                               parse_constant=_nonfinite),("format","score_sha256","library_sha256","assignments"))
        if raw["format"]!=FORMAT or type(raw["assignments"]) is not list or len(raw["assignments"])>len(TRACKS):
            raise GenerationError("Invalid sound assignment document")
        plan=SoundAssignment(raw["score_sha256"],raw["library_sha256"],
            tuple(Assignment(**_object(row,("track","name","timbre_sha256"))) for row in raw["assignments"]))
        if expected_sha256 is not None and plan.sha256!=expected_sha256:
            raise GenerationError("Sound assignment identity mismatch")
        return plan
    except (ValueError,TypeError,UnicodeError,OverflowError,RecursionError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed sound assignment") from exc


def read_assignment(path,*,expected_sha256=None):
    with open(path,"rb") as stream:
        return decode_assignment(stream.read(MAX_BYTES+1),expected_sha256=expected_sha256)


def make_assignment(score,library,tracks):
    if (not isinstance(score,Arrangement) or not isinstance(library,SoundLibrary)
            or type(tracks) is not dict or not tracks or not set(tracks)<=set(TRACKS)):
        raise GenerationError("Invalid sound assignment source")
    rows=[]
    for track in TRACKS:
        if track in tracks:
            entry=library.get(tracks[track])
            rows.append(Assignment(track,entry.name,entry.timbre_sha256))
    plan=SoundAssignment(arrangement_digest(score),library.sha256,tuple(rows))
    compile_assignment(score,library,plan)
    return plan


def compile_assignment(score,library,plan):
    if not isinstance(score,Arrangement) or not isinstance(library,SoundLibrary) or not isinstance(plan,SoundAssignment):
        raise GenerationError("Expected score, library and assignment")
    if plan.score_sha256!=arrangement_digest(score):
        raise GenerationError("Sound assignment score SHA256 mismatch")
    if plan.library_sha256!=library.sha256:
        raise GenerationError("Sound assignment library SHA256 mismatch")
    pitched=[]
    drums=[]
    for row in plan.assignments:
        entry=library.get(row.name)
        if entry.timbre_sha256!=row.timbre_sha256:
            raise GenerationError("Assigned timbre digest mismatch")
        expected="pitched" if row.track in PITCHED else row.track
        if expected!=entry.kind:
            raise GenerationError("Timbre is incompatible with assigned track")
        if row.track in PITCHED:
            pitched.append(PatchRoute(row.track,entry.patch))
        else:
            drums.append(DrumRoute(row.track,entry.patch))
    pitch_bank=PatchBank(plan.score_sha256,tuple(pitched))
    drum_bank=DrumBank(plan.score_sha256,tuple(drums))
    report=describe_combined_banks(score,drum_bank,pitch_bank)
    return pitch_bank,drum_bank,{
        "protocol":FORMAT,"assignment_sha256":plan.sha256,"library_sha256":library.sha256,
        "score_sha256":plan.score_sha256,"pitched_bank_sha256":pitch_bank.sha256,
        "drum_bank_sha256":drum_bank.sha256,"routing":report,
        "assignments":[asdict(row) for row in plan.assignments],
    }


def transfer_assignment(old_score,new_score,library,plan):
    # The caller explicitly supplies and verifies both scores. Changing
    # notes or tempo cannot silently carry stale bank score identifiers.
    _,_,old=compile_assignment(old_score,library,plan)
    if not isinstance(new_score,Arrangement):
        raise GenerationError("Expected target score for sound assignment transfer")
    result=SoundAssignment(arrangement_digest(new_score),plan.library_sha256,plan.assignments)
    _,_,report=compile_assignment(new_score,library,result)
    return result,{
        "source_assignment_sha256":plan.sha256,"result_assignment_sha256":result.sha256,
        "source_score_sha256":old["score_sha256"],"result_score_sha256":report["score_sha256"],
        "library_sha256":library.sha256,"timbres_unchanged":True,
        "source_cost":old["routing"]["cost"],"result_cost":report["routing"]["cost"],
        "target_budget_verified":True,"assignments":[asdict(row) for row in result.assignments],
    }


def transfer_library(score,old_library,new_library,plan,*,rename_map=None):
    compile_assignment(score,old_library,plan)
    if not isinstance(new_library,SoundLibrary):
        raise GenerationError("Expected replacement sound library")
    renames={} if rename_map is None else rename_map
    names={row.name for row in plan.assignments}
    if type(renames) is not dict or not set(renames)<=names:
        raise GenerationError("Invalid explicit timbre name mapping")
    rows=[]
    for row in plan.assignments:
        name=renames.get(row.name,row.name)
        entry=new_library.get(name)
        if entry.timbre_sha256!=row.timbre_sha256:
            raise GenerationError("Library transfer would change an assigned timbre")
        rows.append(Assignment(row.track,name,row.timbre_sha256))
    result=SoundAssignment(plan.score_sha256,new_library.sha256,tuple(rows))
    compile_assignment(score,new_library,result)
    return result,{"source_library_sha256":old_library.sha256,"result_library_sha256":new_library.sha256,
                   "source_assignment_sha256":plan.sha256,"result_assignment_sha256":result.sha256,
                   "timbres_unchanged":True,"renames":dict(renames)}
