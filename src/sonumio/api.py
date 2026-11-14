"""Local-only initial REST API. Do not expose publicly without auth/rate limits."""
from pathlib import Path
import os
import re
import tempfile
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from .errors import GenerationError
from .registry import list_models
from .scenes import SceneLayer, compose_scene
from .service import generate_audio

app = FastAPI(title="Sonumio experimental API", version="0.1.0-dev", docs_url="/docs")
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
    fade_in_seconds: float = Field(default=0.02, ge=0, le=10)
    fade_out_seconds: float = Field(default=0.05, ge=0, le=10)
    seed: int = Field(default=0, ge=0, lt=2**32)

class SceneRequest(BaseModel):
    duration_seconds: float = Field(default=5.0, ge=0.5, le=30)
    layers: list[SceneLayerRequest] = Field(min_length=1, max_length=8)


def _store_audio(wav_bytes: bytes) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    identifier = str(uuid4())
    filename = OUTPUT_DIR / f"{identifier}.wav"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=OUTPUT_DIR, suffix=".tmp", delete=False) as tmp:
            tmp.write(wav_bytes)
            temp_path = Path(tmp.name)
        os.replace(temp_path, filename)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return identifier


@app.post("/v1/scenes", status_code=201)
def generate_scene(request: SceneRequest) -> dict:
    try:
        layers = [SceneLayer(**layer.model_dump()) for layer in request.layers]
        generated = compose_scene(layers, request.duration_seconds)
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    identifier = _store_audio(generated.wav_bytes)
    return {
        "id": identifier, "engine_id": generated.result.engine_id,
        "audio_url": f"/v1/generations/{identifier}/audio",
        "sample_rate": generated.result.sample_rate,
        "duration_seconds": generated.result.duration_seconds,
        "layer_count": len(layers), "metrics": generated.metrics,
        "disclaimer": "Procedural scene synthesis; not a neural generative model",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "original_sonumio_models_available": 0}

@app.get("/v1/models")
def models() -> list[dict]:
    return list_models()

@app.post("/v1/generations", status_code=201)
def generate(request: GenerationRequest) -> dict:
    try:
        generated = generate_audio(**request.model_dump())
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    generation_id = _store_audio(generated.wav_bytes)
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

@app.get("/v1/generations/{generation_id}/audio")
def audio(generation_id: str):
    if not re.fullmatch(r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", generation_id):
        raise HTTPException(status_code=404, detail="Audio not found")
    path = OUTPUT_DIR / f"{generation_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav", filename="sonumio.wav")

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


def _cleanup_job_audio(result: dict, root: Path) -> None:
    """Remove only locally generated, ID-validated WAV outputs."""
    identifier = result.get("id")
    if isinstance(identifier, str) and re.fullmatch(
        r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}", identifier
    ):
        (root / f"{identifier}.wav").unlink(missing_ok=True)


def _queue_response(job: dict) -> dict:
    return {
        **job,
        "status_url": f"/v1/jobs/{job['id']}",
        "cancel_url": f"/v1/jobs/{job['id']}",
        "disclaimer": "Local development queue; active work cannot be forcibly interrupted",
    }


def _enqueue(kind: str, callback) -> dict:
    # Keep the output path fixed per task so later config changes do not delete
    # unrelated files when a cancelled/expired job is cleaned up.
    output_root = OUTPUT_DIR
    try:
        task = JOBS.submit(kind, callback,
                           lambda result: _cleanup_job_audio(result, output_root))
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
    return _enqueue("generation", lambda: generate(request))


@app.post("/v1/jobs/scenes", status_code=202)
def enqueue_scene(request: SceneRequest) -> dict:
    layers = [SceneLayer(**layer.model_dump()) for layer in request.layers]
    try:
        validate_scene_plan(layers, request.duration_seconds)
        for layer in layers:
            recognized_sound_classes(layer.prompt)
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _enqueue("scene", lambda: generate_scene(request))


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
