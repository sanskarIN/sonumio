"""Local, bounded browser transport for integrated render recipes."""
from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import GenerationError
from .music_project import MAX_PROJECT_BYTES, decode_project
from .music_performance import MAX_DOCUMENT_BYTES, decode_performance
from .music_rack import MAX_RACK_BYTES, decode_rack
from .music_render_recipe import (
    RenderRecipe, MAX_RECIPE_BYTES, decode_recipe, encode_recipe, describe_recipe, render_recipe,
)
from .music_project_api import bounded_json_body, _validated
from .music_project_workspace import Digest, _bytes
from .music_score_io import _unique, _nonfinite
from .music_studio import ENGINE_ID, DISCLAIMER, MAX_PREVIEW_FRAMES

MAX_CREATE_BYTES = 6 * 1024 * 1024


class RecipeCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    project_document: str = Field(min_length=1, max_length=MAX_PROJECT_BYTES)
    expected_project_sha256: Digest
    performance_document: str | None = Field(default=None, max_length=MAX_DOCUMENT_BYTES)
    rack_document: str | None = Field(default=None, max_length=MAX_RACK_BYTES)


class RecipeJobIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    operation: str = "integrated-project-render-v1"
    recipe_sha256: Digest
    project_sha256: Digest
    score_sha256: Digest
    automation_sha256: Digest
    performance_sha256: Digest | None
    rack_sha256: Digest | None


def parse_create(data):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_CREATE_BYTES:
        raise GenerationError("Invalid render recipe creation envelope")
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
        model = RecipeCreate.model_validate(raw)
    except (ValueError, TypeError, UnicodeError, RecursionError, ValidationError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Invalid render recipe creation fields") from exc
    project = decode_project(_bytes(model.project_document,MAX_PROJECT_BYTES,"project"),
                             expected_sha256=model.expected_project_sha256)
    performance = (decode_performance(_bytes(model.performance_document,MAX_DOCUMENT_BYTES,"performance"))
                   if model.performance_document is not None else None)
    rack = (decode_rack(_bytes(model.rack_document,MAX_RACK_BYTES,"rack"))
            if model.rack_document is not None else None)
    return RenderRecipe(project,performance,rack)


def recipe_snapshot(recipe):
    report = describe_recipe(recipe)
    data = encode_recipe(recipe)
    report.update(title=recipe.project.title,
                  content_sha256=hashlib.sha256(data).hexdigest(),
                  preview_supported=(report["frames"] <= MAX_PREVIEW_FRAMES
                                     and recipe.project.score.plan.bars <= 16),
                  bars=recipe.project.score.plan.bars)
    return report


def recipe_download(recipe):
    data = encode_recipe(recipe)
    return data, {
        "Content-Disposition": 'attachment; filename="sonumio-render-recipe.json"',
        "X-Sonumio-Recipe-SHA256": recipe.sha256,
        "X-Sonumio-Project-SHA256": recipe.project.sha256,
        "X-Sonumio-Content-SHA256": hashlib.sha256(data).hexdigest(),
        "Cache-Control": "no-store",
    }


def job_identity(recipe):
    if not recipe_snapshot(recipe)["preview_supported"]:
        raise GenerationError("Render recipe exceeds studio preview limits")
    return RecipeJobIdentity(recipe_sha256=recipe.sha256, **recipe.identity)


def render_preview(recipe):
    identity = job_identity(recipe)
    wav, report = render_recipe(recipe)
    if any(report[key] != value for key,value in identity.model_dump().items() if key != "operation"):
        raise GenerationError("Integrated render receipt differs from queued identity")
    # Full receipts remain available in offline bundles. Keep queue metadata
    # small and avoid serializing the complete input documents into a ledger.
    details = report["details"]
    if recipe.engine == "performance":
        details = details["mix"]
    return wav, {
        **recipe.identity, "recipe_sha256": recipe.sha256,
        "receipt_sha256": report["receipt_sha256"], "wav_sha256": report["wav_sha256"],
        "engine_id": ENGINE_ID, "model_id": ENGINE_ID, "recipe_engine": recipe.engine,
        "frames": report["frames"], "sample_rate": report["sample_rate"], "channels": 2,
        "duration_seconds": report["frames"]/report["sample_rate"],
        "metrics": details["metrics"], "master_attenuation": details["master_attenuation"],
        "learned_model": False, "external_recordings_used": False, "disclaimer": DISCLAIMER,
    }


def _pin(request, required):
    values = request.headers.getlist("x-sonumio-recipe-sha256")
    if not values and not required:
        return None
    if (len(values)!=1 or len(values[0])!=64
            or any(c not in "0123456789abcdef" for c in values[0])):
        raise HTTPException(status_code=422,detail="Valid render recipe SHA256 pin required")
    return values[0]


async def _recipe(request,required=False):
    pin=_pin(request,required)
    data=await bounded_json_body(request,MAX_RECIPE_BYTES)
    return await _validated(decode_recipe,data,expected_sha256=pin)


def install_recipe_routes(app,*,enqueue,store_for_current,store_audio,requested_job_id):
    router=APIRouter()

    @router.post("/v1/music/render-recipe/create")
    async def create(request:Request):
        data=await bounded_json_body(request,MAX_CREATE_BYTES)
        recipe=await _validated(parse_create,data)
        payload,headers=await _validated(recipe_download,recipe)
        return Response(payload,media_type="application/json",headers=headers)

    @router.post("/v1/music/render-recipe/inspect")
    async def inspect(request:Request):
        recipe=await _recipe(request)
        return await _validated(recipe_snapshot,recipe)

    @router.post("/v1/jobs/music-recipe",status_code=202)
    async def render(request:Request,job_id:str|None=Depends(requested_job_id)):
        recipe=await _recipe(request,required=True)
        identity=await _validated(job_identity,recipe)
        store=store_for_current()

        def produce():
            wav,result=render_preview(recipe)
            identifier=store_audio(wav,store)
            return {**result,"id":identifier,"audio_url":f"/v1/generations/{identifier}/audio"}

        return enqueue("music",identity,produce,store,job_id)

    app.include_router(router)
