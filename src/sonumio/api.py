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
from .service import generate_audio

app = FastAPI(title="Sonumio experimental API", version="0.1.0-dev", docs_url="/docs")
OUTPUT_DIR = Path(os.environ.get("SONUMIO_OUTPUT_DIR", "outputs")).resolve()

class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=800)
    duration_seconds: float = Field(default=5.0, ge=0.5, le=15.0)
    model_id: str = "procedural-sfx-v0"
    seed: int = Field(default=0, ge=0, lt=2**32)

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generation_id = str(uuid4())
    filename = OUTPUT_DIR / f"{generation_id}.wav"
    try:
        with tempfile.NamedTemporaryFile(dir=OUTPUT_DIR, suffix=".tmp", delete=False) as tmp:
            tmp.write(generated.wav_bytes)
            temp_path = Path(tmp.name)
        os.replace(temp_path, filename)
    finally:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)
    return {
        "id": generation_id, "model_id": generated.result.engine_id,
        "duration_seconds": generated.result.duration_seconds,
        "sample_rate": generated.result.sample_rate, "seed": generated.result.seed,
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
