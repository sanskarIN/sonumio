"""Streaming-bounded local project routes with no user-selected file paths."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from .errors import GenerationError
from .music_project import MAX_PROJECT_BYTES, decode_project
from .music_project_workspace import (
    MAX_MUTATION_BYTES, CreateProjectRequest, ReviseProjectRequest,
    parse_mutation, project_from_request, revision_from_request,
    project_snapshot, project_download, project_score_download,
    project_render_identity, render_project_preview,
)


async def bounded_json_body(request,limit):
    media=request.headers.get("content-type","").split(";",1)[0].strip().lower()
    if media!="application/json":
        raise HTTPException(status_code=415,detail="Project requests require application/json")
    if request.headers.get("content-encoding","identity").lower()!="identity":
        raise HTTPException(status_code=415,detail="Compressed project requests are unsupported")
    lengths=request.headers.getlist("content-length")
    declared=None
    if lengths:
        raw=lengths[0]
        if len(lengths)!=1 or not raw.isascii() or not raw.isdigit() or len(raw)>10:
            raise HTTPException(status_code=400,detail="Invalid project Content-Length")
        declared=int(raw)
        if declared>limit:
            raise HTTPException(status_code=413,detail="Project request exceeds byte limit")
        if declared==0:
            raise HTTPException(status_code=422,detail="Empty project request")
    data=bytearray()
    try:
        async for chunk in request.stream():
            if len(data)+len(chunk)>limit:
                raise HTTPException(status_code=413,detail="Project request exceeds byte limit")
            data.extend(chunk)
    except ClientDisconnect as exc:
        raise HTTPException(status_code=400,detail="Project upload disconnected") from exc
    if declared is not None and len(data)!=declared:
        raise HTTPException(status_code=400,detail="Project request length differs from header")
    if not data:
        raise HTTPException(status_code=422,detail="Empty project request")
    return bytes(data)


def _pin(request):
    values=request.headers.getlist("x-sonumio-project-sha256")
    if not values:
        return None
    value=values[0]
    if len(values)!=1 or len(value)!=64 or any(c not in "0123456789abcdef" for c in value):
        raise HTTPException(status_code=422,detail="Invalid project SHA256 pin")
    return value


async def _validated(function,*args,**kwargs):
    try:
        return await run_in_threadpool(function,*args,**kwargs)
    except GenerationError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc


async def _project(request,*,require_pin=False):
    pin=_pin(request)
    if require_pin and pin is None:
        raise HTTPException(status_code=422,detail="Verified project SHA256 pin required")
    data=await bounded_json_body(request,MAX_PROJECT_BYTES)
    return await _validated(decode_project,data,expected_sha256=pin)


def install_project_routes(app,*,enqueue,store_for_current,store_audio,requested_job_id):
    router=APIRouter()

    @router.post("/v1/music/project/inspect")
    async def inspect_project(request:Request):
        project=await _project(request)
        return await _validated(project_snapshot,project)

    @router.post("/v1/music/project/create")
    async def create(request:Request):
        raw=await bounded_json_body(request,MAX_MUTATION_BYTES)
        model=await _validated(parse_mutation,raw,CreateProjectRequest)
        project=await _validated(project_from_request,model)
        data,headers=await _validated(project_download,project)
        return Response(data,media_type="application/json",headers=headers)

    @router.post("/v1/music/project/revise")
    async def revise(request:Request):
        raw=await bounded_json_body(request,MAX_MUTATION_BYTES)
        model=await _validated(parse_mutation,raw,ReviseProjectRequest)
        _,child=await _validated(revision_from_request,model)
        data,headers=await _validated(project_download,child)
        return Response(data,media_type="application/json",headers=headers)

    @router.post("/v1/music/project/score")
    async def score(request:Request):
        project=await _project(request,require_pin=True)
        data,media,headers=await _validated(project_score_download,project,"json")
        return Response(data,media_type=media,headers=headers)

    @router.post("/v1/music/project/midi")
    async def midi(request:Request):
        project=await _project(request,require_pin=True)
        data,media,headers=await _validated(project_score_download,project,"midi")
        return Response(data,media_type=media,headers=headers)

    @router.post("/v1/jobs/music-project",status_code=202)
    async def render(request:Request,job_id:str|None=Depends(requested_job_id)):
        project=await _project(request,require_pin=True)
        identity=await _validated(project_render_identity,project)
        store=store_for_current()

        def produce():
            wav,result=render_project_preview(project)
            identifier=store_audio(wav,store)
            return {**result,"id":identifier,"audio_url":f"/v1/generations/{identifier}/audio"}

        return enqueue("music",identity,produce,store,job_id)

    app.include_router(router)
