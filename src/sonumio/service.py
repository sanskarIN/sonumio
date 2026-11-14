"""Validated request routing and signal/performance metadata."""
from dataclasses import dataclass
import os
from time import perf_counter
from .audio import AudioResult, encode_wav, metrics
from .engines import procedural
from .errors import GenerationError, UnavailableModelError
from .registry import ENTRIES
from .validation import validate_request

@dataclass(frozen=True)
class Generated:
    result: AudioResult
    wav_bytes: bytes
    metrics: dict[str, float]

def generate_audio(prompt: str, duration_seconds: float = 5.0,
                   model_id: str = "procedural-sfx-v0", seed: int = 0) -> Generated:
    start = perf_counter()
    prompt = validate_request(prompt, duration_seconds)
    if model_id not in ENTRIES:
        raise GenerationError(f"Unknown model_id: {model_id}")
    if not isinstance(seed, int) or not (0 <= seed < 2**32):
        raise GenerationError("seed must be an integer in [0, 2**32)")
    if model_id == "procedural-sfx-v0":
        result = procedural.generate(prompt, duration_seconds, seed)
    elif model_id == "meta-musicgen-small":
        if os.environ.get("SONUMIO_ENABLE_MUSICGEN") != "1":
            raise UnavailableModelError("MusicGen is disabled. Its CC-BY-NC-4.0 weights require explicit research opt-in.")
        from .engines.musicgen import generate
        result = generate(prompt, duration_seconds, seed)
    else:
        raise GenerationError(f"Unsupported model_id: {model_id}")
    wav_bytes = encode_wav(result.samples, result.sample_rate)
    quality = metrics(result.samples)
    elapsed = perf_counter() - start
    quality["generation_wall_seconds"] = round(elapsed, 6)
    quality["real_time_factor"] = round(elapsed / result.duration_seconds, 6)
    return Generated(result=result, wav_bytes=wav_bytes, metrics=quality)
