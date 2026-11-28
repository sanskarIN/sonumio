"""Bounded project interchange for the browser and local API.

Project bytes are never interpreted as a path and are not persisted by
inspection or editing. Original document strings are retained intact so
JSON number spelling cannot be lost by a browser's parse/stringify cycle.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .arrangement import arrangement_digest
from .errors import GenerationError
from .music_automation import MAX_PLAN_BYTES, AutomationPlan, decode_automation
from .music_note_edits import NoteEdits, apply_note_edits
from .music_project import (
    MAX_PROJECT_BYTES, MusicProject, create_project, revise_project,
    describe_project, decode_project, encode_project, verify_revision, render_project,
)
from .music_score_io import MAX_SCORE_BYTES, decode_score, _unique, _nonfinite
from .music_studio import MAX_PREVIEW_FRAMES, ENGINE_ID, DISCLAIMER, describe_score, export_score, MusicMixRequest

MAX_MUTATION_BYTES = 5 * 1024 * 1024
Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]


class ProjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CreateProjectRequest(ProjectModel):
    title: str = Field(min_length=1, max_length=80)
    score_document: str = Field(min_length=1, max_length=MAX_SCORE_BYTES)
    expected_score_sha256: Digest
    automation_document: str | None = Field(default=None, max_length=MAX_PLAN_BYTES)
    expected_automation_sha256: Digest | None = None


class ReviseProjectRequest(ProjectModel):
    project_document: str = Field(min_length=1, max_length=MAX_PROJECT_BYTES)
    expected_project_sha256: Digest
    title: str | None = Field(default=None, min_length=1, max_length=80)
    score_document: str | None = Field(default=None, max_length=MAX_SCORE_BYTES)
    expected_score_sha256: Digest | None = None
    note_edits: NoteEdits | None = None
    automation_document: str | None = Field(default=None, max_length=MAX_PLAN_BYTES)
    expected_automation_sha256: Digest | None = None


    mix: MusicMixRequest | None = None


class ProjectRenderIdentity(ProjectModel):
    # Only verified content identifiers enter the queue's idempotency
    # fingerprint; the original uploaded bytes are not persisted in a ledger.
    project_sha256: Digest
    score_sha256: Digest
    automation_sha256: Digest
    operation: str = "native-project-render-v1"


def _bytes(text,limit,label):
    if type(text) is not str:
        raise GenerationError("Expected "+label+" document string")
    try:
        data=text.encode("utf-8",errors="strict")
    except UnicodeError as exc:
        raise GenerationError("Invalid Unicode in "+label+" document") from exc
    if not 1<=len(data)<=limit:
        raise GenerationError("Invalid bounded "+label+" document byte count")
    return data


def parse_mutation(data, model):
    if model not in (CreateProjectRequest,ReviseProjectRequest):
        raise GenerationError("Unsupported project mutation type")
    if type(data) is not bytes or not 1<=len(data)<=MAX_MUTATION_BYTES:
        raise GenerationError("Invalid project mutation byte count")
    try:
        raw=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_nonfinite)
        return model.model_validate(raw)
    except (ValidationError,UnicodeError,RecursionError,TypeError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        # Avoid returning arbitrary document content in validation errors.
        raise GenerationError("Invalid project mutation fields") from exc


def _automation(document,expected):
    if document is None:
        if expected is not None:
            raise GenerationError("Automation pin requires an automation document")
        return None
    if expected is None:
        raise GenerationError("Automation document requires its verified digest")
    plan=decode_automation(_bytes(document,MAX_PLAN_BYTES,"automation"))
    if plan.sha256!=expected:
        raise GenerationError("Project automation digest mismatch")
    return plan


def project_from_request(request):
    if not isinstance(request,CreateProjectRequest):
        raise GenerationError("Expected project creation request")
    score=decode_score(_bytes(request.score_document,MAX_SCORE_BYTES,"score"),
                       expected_sha256=request.expected_score_sha256)
    automation=_automation(request.automation_document,request.expected_automation_sha256)
    return create_project(request.title,score,automation)


def revision_from_request(request):
    if not isinstance(request,ReviseProjectRequest):
        raise GenerationError("Expected project revision request")
    parent=decode_project(_bytes(request.project_document,MAX_PROJECT_BYTES,"project"),
                          expected_sha256=request.expected_project_sha256)
    if request.score_document is not None and request.note_edits is not None:
        raise GenerationError("Choose score replacement or note edits, not both")
    if request.score_document is not None:
        if request.expected_score_sha256 is None:
            raise GenerationError("Score replacement requires a verified digest")
        score=decode_score(_bytes(request.score_document,MAX_SCORE_BYTES,"score"),
                           expected_sha256=request.expected_score_sha256)
    elif request.note_edits is not None:
        if parent.score.plan.bars>16 or parent.score.plan.frame(parent.score.plan.ticks)>MAX_PREVIEW_FRAMES:
            raise GenerationError("Project note editing is limited to studio preview scores")
        score,_=apply_note_edits(parent.score,request.note_edits)
        if request.expected_score_sha256 is not None and arrangement_digest(score)!=request.expected_score_sha256:
            raise GenerationError("Edited project score digest mismatch")
    else:
        if request.expected_score_sha256 is not None:
            raise GenerationError("Score pin requires a replacement or note transaction")
        score=None
    automation=_automation(request.automation_document,request.expected_automation_sha256)
    if request.mix is not None:
        if automation is not None:
            raise GenerationError("Choose automation document replacement or static mixer edits, not both")
        automation=AutomationPlan(parent.automation.lanes,request.mix.to_plan())
    child=revise_project(parent,title=request.title,score=score,automation=automation)
    verify_revision(parent,child)
    return parent,child


def project_snapshot(project):
    if not isinstance(project,MusicProject):
        raise GenerationError("Expected verified MusicProject")
    report=describe_project(project)
    preview=project.score.plan.bars<=16 and report["frames"]<=MAX_PREVIEW_FRAMES
    report["preview_supported"]=preview
    report["score_plan"]=describe_score(project.score) if preview else None
    # Canonical nested automation values, no path or raw arbitrary bytes.
    report["automation_lanes"]=[
        {"track":lane.track,"parameter":lane.parameter,
         "keys":[{"tick":key.tick,"value":float(key.value),"law":key.law}
                 for key in lane.keys]} for lane in project.automation.lanes]
    report["automation_lane_count"]=len(project.automation.lanes)
    report["content_sha256"]=sha256(encode_project(project)).hexdigest()
    return report


def project_download(project):
    data=encode_project(project)
    return data,{
        "Content-Disposition":'attachment; filename="music.sonumio"',
        "X-Sonumio-Project-SHA256":project.sha256,
        "X-Sonumio-Score-SHA256":project.score_sha256,
        "X-Sonumio-Automation-SHA256":project.automation_sha256,
        "X-Sonumio-Content-SHA256":sha256(data).hexdigest(),
        "Cache-Control":"no-store",
    }


def project_score_download(project,format_name):
    data,media,headers=export_score(project.score,format_name)
    return data,media,{**headers,"X-Sonumio-Project-SHA256":project.sha256}


def project_render_identity(project):
    snapshot=describe_project(project)
    if snapshot["frames"]>MAX_PREVIEW_FRAMES or project.score.plan.bars>16:
        raise GenerationError("Project exceeds the local studio preview limit")
    return ProjectRenderIdentity(project_sha256=project.sha256,
        score_sha256=project.score_sha256,automation_sha256=project.automation_sha256)


def render_project_preview(project):
    identity=project_render_identity(project)
    wav,report=render_project(project)
    frames=project.score.plan.frame(project.score.plan.ticks)
    if (report["project_sha256"]!=identity.project_sha256
            or report["score_sha256"]!=identity.score_sha256
            or report["automation_sha256"]!=identity.automation_sha256
            or report["wav_sha256"]!=sha256(wav).hexdigest()
            or report["frames"]!=frames or report["sample_rate"]!=24000
            or report["channels"]!=2 or len(wav)!=44+frames*4):
        raise GenerationError("Native project renderer receipt mismatch")
    return wav,{
        "engine_id":ENGINE_ID,"model_id":ENGINE_ID,"disclaimer":DISCLAIMER,
        "project_sha256":identity.project_sha256,"score_sha256":identity.score_sha256,
        "automation_sha256":identity.automation_sha256,"wav_sha256":report["wav_sha256"],
        "project_revision":project.revision,"frames":frames,"sample_rate":24000,
        "duration_seconds":frames/24000,"channels":2,"metrics":report["metrics"],
        "track_metrics":report["tracks"],"master_attenuation":report["master_attenuation"],
        "learned_model":False,"external_recordings_used":False,
    }


def inspect_score_document(data):
    score=decode_score(data)
    frames=score.plan.frame(score.plan.ticks)
    return {"score_sha256":arrangement_digest(score),"frames":frames,
            "bars":score.plan.bars,"events":len(score.events),
            "preview_supported":score.plan.bars<=16 and frames<=MAX_PREVIEW_FRAMES}


def inspect_automation_document(data):
    plan=decode_automation(data)
    return {"automation_sha256":plan.sha256,"lanes":len(plan.lanes),
            "keys":sum(len(lane.keys) for lane in plan.lanes)}
