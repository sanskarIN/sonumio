"""Local-only initial REST API. Do not expose publicly without auth/rate limits."""
from functools import lru_cache
from pathlib import Path
import os
from fastapi import FastAPI, HTTPException
from . import __version__
from fastapi.responses import Response
from pydantic import BaseModel, Field
from .errors import GenerationError
from .registry import list_models
from .scenes import SceneLayer, compose_scene
from .service import generate_audio
from .output_store import LocalAudioStore, OutputStoreError

app = FastAPI(title="Sonumio experimental API", version=__version__, docs_url="/docs")
OUTPUT_DIR = Path(os.environ.get("SONUMIO_OUTPUT_DIR", "outputs")).resolve()

class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=800)
    duration_seconds: float = Field(default=5.0, ge=0.5, le=15.0)
    model_id: str = "procedural-sfx-v0"
    seed: int = Field(default=0, ge=0, lt=2**32)

class SceneLayerRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=800)
    start_seconds: float = Field(default=0.0, ge=0, le=30)
    duration_seconds: float = Field(default=3.0, ge=0.5, le=15)
    gain_db: float = Field(default=0.0, ge=-60, le=12)
    gain_end_db: float | None = Field(default=None, ge=-60, le=12)
    pan: float = Field(default=0.0, ge=-1, le=1)
    pan_end: float | None = Field(default=None, ge=-1, le=1)
    fade_in_seconds: float = Field(default=0.02, ge=0, le=10)
    fade_out_seconds: float = Field(default=0.05, ge=0, le=10)
    seed: int = Field(default=0, ge=0, lt=2**32)

class SceneRequest(BaseModel):
    duration_seconds: float = Field(default=5.0, ge=0.5, le=30)
    layers: list[SceneLayerRequest] = Field(min_length=1, max_length=8)


@lru_cache(maxsize=16)
def _store_for(root: Path) -> LocalAudioStore:
    """Keep API outputs separate from user-managed CLI audio files."""
    return LocalAudioStore(root / ".sonumio-api-audio")


def _store_audio(wav_bytes: bytes, store: LocalAudioStore) -> str:
    try:
        return store.write(wav_bytes)
    except (OutputStoreError, OSError) as exc:
        raise HTTPException(status_code=507, detail="Local audio storage is unavailable or full") from exc


def _generate_scene(request: SceneRequest, store: LocalAudioStore) -> dict:
    try:
        layers = [SceneLayer(**layer.model_dump()) for layer in request.layers]
        generated = compose_scene(layers, request.duration_seconds)
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    identifier = _store_audio(generated.wav_bytes, store)
    return {
        "id": identifier, "engine_id": generated.result.engine_id,
        "audio_url": f"/v1/generations/{identifier}/audio",
        "sample_rate": generated.result.sample_rate,
        "duration_seconds": generated.result.duration_seconds,
        "layer_count": len(layers),
        "channels": (2 if generated.result.samples.ndim == 2 else 1),
        "metrics": generated.metrics,
        "disclaimer": "Procedural scene synthesis; not a neural generative model",
    }


@app.post("/v1/scenes", status_code=201)
def generate_scene(request: SceneRequest) -> dict:
    return _generate_scene(request, _store_for(OUTPUT_DIR))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "original_sonumio_models_available": 0}

@app.get("/v1/models")
def models() -> list[dict]:
    return list_models()

def _generate(request: GenerationRequest, store: LocalAudioStore) -> dict:
    try:
        generated = generate_audio(**request.model_dump())
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    generation_id = _store_audio(generated.wav_bytes, store)
    return {
        "id": generation_id,
        "model_id": generated.result.engine_id,
        "duration_seconds": generated.result.duration_seconds,
        "sample_rate": generated.result.sample_rate,
        "seed": generated.result.seed,
        "audio_url": f"/v1/generations/{generation_id}/audio",
        "metrics": generated.metrics,
        "disclaimer": "Procedural DSP is not a trained AI model" if request.model_id == "procedural-sfx-v0" else "Third-party pretrained model",
    }


@app.post("/v1/generations", status_code=201)
def generate(request: GenerationRequest) -> dict:
    return _generate(request, _store_for(OUTPUT_DIR))


@app.get("/v1/generations/{generation_id}/audio")
def audio(generation_id: str):
    try:
        wav_bytes = _store_for(OUTPUT_DIR).read(generation_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Audio not found") from exc
    except (OutputStoreError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Local audio storage unavailable") from exc
    return Response(
        content=wav_bytes, media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="sonumio.wav"',
                 "Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# Development-only, bounded in-process jobs. These endpoints deliberately do
# not expose multi-user/public inference, arbitrary model paths or voice tasks.
# The queue owns one background worker, not a thread per request.
# ---------------------------------------------------------------------------
from .jobs import LocalJobQueue, JobNotFoundError, JobConflictError, QueueFullError
from .scenes import validate_scene_plan
from .validation import validate_request
from .engines.procedural import recognized_sound_classes

JOBS = LocalJobQueue(max_active=4, max_history=32, ttl_seconds=1800)


def _cleanup_job_audio(result: dict, store: LocalAudioStore) -> None:
    """Remove only the audio associated with this queued job."""
    store.delete(result.get("id") if isinstance(result, dict) else None)


def _queue_response(job: dict) -> dict:
    return {
        **job,
        "status_url": f"/v1/jobs/{job['id']}",
        "cancel_url": f"/v1/jobs/{job['id']}",
        "disclaimer": "Local development queue; active work cannot be forcibly interrupted",
    }


def _enqueue(kind: str, callback, store: LocalAudioStore) -> dict:
    # Capture the exact storage instance alongside the queued callback:
    # changing OUTPUT_DIR later cannot redirect the render or its cleanup.
    try:
        task = JOBS.submit(
            kind, callback,
            lambda result: _cleanup_job_audio(result, store),
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return _queue_response(task)


@app.post("/v1/jobs/generations", status_code=202)
def enqueue_generation(request: GenerationRequest) -> dict:
    if request.model_id != "procedural-sfx-v0":
        raise HTTPException(status_code=422, detail="The local job queue supports procedural-sfx-v0 only")
    try:
        validate_request(request.prompt, request.duration_seconds)
        recognized_sound_classes(request.prompt)
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store = _store_for(OUTPUT_DIR)
    return _enqueue("generation", lambda: _generate(request, store), store)


@app.post("/v1/jobs/scenes", status_code=202)
def enqueue_scene(request: SceneRequest) -> dict:
    layers = [SceneLayer(**layer.model_dump()) for layer in request.layers]
    try:
        validate_scene_plan(layers, request.duration_seconds)
        for layer in layers:
            recognized_sound_classes(layer.prompt)
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store = _store_for(OUTPUT_DIR)
    return _enqueue("scene", lambda: _generate_scene(request, store), store)


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return _queue_response(JOBS.get(job_id))
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found or expired") from exc


@app.delete("/v1/jobs/{job_id}")
def cancel_job(job_id: str) -> dict:
    try:
        return _queue_response(JOBS.cancel(job_id))
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found or expired") from exc
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
