"""Self-contained, bounded original-music project documents.

A project contains a canonical symbolic score and its canonical automation
document, not external audio, paths, model weights or executable code.
Revision links are content identifiers, not authentication or signatures.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import unicodedata

from .arrangement import Arrangement, arrangement_digest
from .errors import GenerationError
from .music_automation import (
    AutomationPlan, encode_automation, decode_automation, generate_automated_music,
)
from .music_score_io import encode_score, decode_score, _unique, _nonfinite, _object

FORMAT="sonumio-music-project-v1"
MAX_PROJECT_BYTES=2*1024*1024
MAX_REVISION=2**31-1
MAX_TITLE=80


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),
                      ensure_ascii=True,allow_nan=False).encode("ascii")


def _digest(value,label):
    if (type(value) is not str or len(value)!=64
            or any(c not in "0123456789abcdef" for c in value)):
        raise GenerationError("Invalid project "+label+" SHA256")
    return value


def _title(value):
    if (type(value) is not str or not 1<=len(value)<=MAX_TITLE or value!=value.strip()
            or unicodedata.normalize("NFC",value)!=value
            or any(unicodedata.category(c).startswith("C") for c in value)):
        raise GenerationError("Project title must be 1–80 printable NFC characters without outer whitespace")
    return value


@dataclass(frozen=True,slots=True)
class MusicProject:
    title: str
    score: Arrangement
    automation: AutomationPlan
    revision: int=0
    parent_project_sha256: str|None=None

    def __post_init__(self):
        _title(self.title)
        if not isinstance(self.score,Arrangement) or not isinstance(self.automation,AutomationPlan):
            raise GenerationError("Project requires validated score and automation")
        if type(self.revision) is not int or not 0<=self.revision<=MAX_REVISION:
            raise GenerationError("Invalid project revision number")
        if self.revision==0:
            if self.parent_project_sha256 is not None:
                raise GenerationError("Initial project cannot declare a parent")
        else:
            _digest(self.parent_project_sha256,"parent")
        self.automation.validate_score(self.score)

    @property
    def score_sha256(self):
        return arrangement_digest(self.score)

    @property
    def automation_sha256(self):
        return self.automation.sha256

    @property
    def score_bytes_sha256(self):
        return hashlib.sha256(encode_score(self.score)).hexdigest()

    @property
    def identity_record(self):
        return {"title":self.title,"revision":self.revision,
                "parent_project_sha256":self.parent_project_sha256,
                "score_sha256":self.score_sha256,"score_bytes_sha256":self.score_bytes_sha256,
                "automation_sha256":self.automation_sha256}

    @property
    def sha256(self):
        return hashlib.sha256(FORMAT.encode("ascii")+b"\0"+_canonical(self.identity_record)).hexdigest()


def create_project(title,score,automation=None):
    return MusicProject(title,score,AutomationPlan(()) if automation is None else automation)


def revise_project(project,*,title=None,score=None,automation=None):
    if not isinstance(project,MusicProject):
        raise GenerationError("Expected MusicProject")
    if project.revision>=MAX_REVISION:
        raise GenerationError("Project revision budget exhausted")
    target=MusicProject(project.title if title is None else title,
                        project.score if score is None else score,
                        project.automation if automation is None else automation,
                        project.revision+1,project.sha256)
    if (target.title==project.title and target.score_sha256==project.score_sha256
            and target.automation_sha256==project.automation_sha256):
        raise GenerationError("Project revision contains no semantic change")
    return target


def describe_project(project):
    if not isinstance(project,MusicProject):
        raise GenerationError("Expected MusicProject")
    frames=project.score.plan.frame(project.score.plan.ticks)
    return {"format":FORMAT,"project_sha256":project.sha256,**project.identity_record,
            "events":len(project.score.events),"bars":project.score.plan.bars,"frames":frames,
            "sample_rate":24000,"duration_seconds":frames/24000,
            "automation_lanes":len(project.automation.lanes),
            "automation_keys":sum(len(lane.keys) for lane in project.automation.lanes),
            "mix":{"gains":list(project.automation.mix.gains),"mute":list(project.automation.mix.mute),
                   "solo":list(project.automation.mix.solo),"room":project.automation.mix.room,
                   "peak_ceiling":project.automation.mix.peak_ceiling},
            "embedded_audio":False,"embedded_weights":False,"external_paths":False,
            "parent_verified":False,"authenticated":False}


def verify_revision(parent,child):
    if not isinstance(parent,MusicProject) or not isinstance(child,MusicProject):
        raise GenerationError("Expected two verified projects")
    if child.revision!=parent.revision+1 or child.parent_project_sha256!=parent.sha256:
        raise GenerationError("Project revision link does not match supplied parent")
    changed=[]
    if child.title!=parent.title:
        changed.append("title")
    if child.score_sha256!=parent.score_sha256:
        changed.append("score")
    if child.automation_sha256!=parent.automation_sha256:
        changed.append("automation")
    if not changed:
        raise GenerationError("Project revision link contains no semantic change")
    return {"parent_project_sha256":parent.sha256,"project_sha256":child.sha256,
            "revision":child.revision,"changed":changed,"parent_verified":True,
            "authenticated":False}


def encode_project(project):
    if not isinstance(project,MusicProject):
        raise GenerationError("Expected MusicProject")
    payload={"format":FORMAT,"project_sha256":project.sha256,
             "metadata":project.identity_record,
             "score":json.loads(encode_score(project.score)),
             "automation":json.loads(encode_automation(project.automation))}
    data=_canonical(payload)
    if len(data)>MAX_PROJECT_BYTES:
        raise GenerationError("Project document exceeds byte budget")
    return data


def decode_project(data,*,expected_sha256=None):
    if type(data) is not bytes or not 1<=len(data)<=MAX_PROJECT_BYTES:
        raise GenerationError("Invalid bounded project byte count")
    if expected_sha256 is not None:
        _digest(expected_sha256,"expected")
    try:
        obj=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_nonfinite)
        _object(obj,("format","project_sha256","metadata","score","automation"))
        if obj["format"]!=FORMAT:
            raise GenerationError("Unsupported music project format")
        meta=_object(obj["metadata"],("title","revision","parent_project_sha256","score_sha256",
                                     "score_bytes_sha256","automation_sha256"))
        for key in ("score_sha256","score_bytes_sha256","automation_sha256"):
            _digest(meta[key],key)
        _digest(obj["project_sha256"],"declared")
        score_data=_canonical(obj["score"])
        automation_data=_canonical(obj["automation"])
        if hashlib.sha256(score_data).hexdigest()!=meta["score_bytes_sha256"]:
            raise GenerationError("Embedded score bytes digest mismatch")
        score=decode_score(score_data,expected_sha256=meta["score_sha256"])
        automation=decode_automation(automation_data)
        if automation.sha256!=meta["automation_sha256"]:
            raise GenerationError("Embedded automation digest mismatch")
        project=MusicProject(meta["title"],score,automation,meta["revision"],meta["parent_project_sha256"])
        if project.sha256!=obj["project_sha256"] or (expected_sha256 is not None and expected_sha256!=project.sha256):
            raise GenerationError("Music project identity mismatch")
        # Nested documents must use their canonical representations. Outer
        # whitespace is allowed, but equivalent noncanonical nested fields
        # cannot silently change embedded byte identities.
        if score_data!=encode_score(score) or automation_data!=encode_automation(automation):
            raise GenerationError("Embedded project documents are not canonical")
        if _canonical(meta)!=_canonical(project.identity_record):
            raise GenerationError("Project metadata differs from verified content")
        return project
    except (UnicodeError,RecursionError,TypeError,OverflowError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed music project JSON") from exc


def read_project(path,*,expected_sha256=None):
    with open(path,"rb") as stream:
        return decode_project(stream.read(MAX_PROJECT_BYTES+1),expected_sha256=expected_sha256)


def render_project(project):
    if not isinstance(project,MusicProject):
        raise GenerationError("Expected MusicProject")
    wav,report=generate_automated_music(project.score,project.automation)
    if (report["score_sha256"]!=project.score_sha256
            or report["automation_sha256"]!=project.automation_sha256
            or report["wav_sha256"]!=hashlib.sha256(wav).hexdigest()):
        raise GenerationError("Project render receipt does not match embedded documents")
    return wav,{**report,"project_sha256":project.sha256,"project_revision":project.revision,
                "project_title":project.title}
