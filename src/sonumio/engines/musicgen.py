"""Research-only optional third-party Meta MusicGen adapter; weights not bundled.

This has not been validated with a real model checkpoint in baseline CI.
"""
from ..audio import AudioResult, to_mono
from ..errors import UnavailableModelError

MODEL_NAME = "facebook/musicgen-small"

def generate(prompt: str, duration_seconds: float, seed: int = 0) -> AudioResult:
    try:
        import numpy as np
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise UnavailableModelError("Install 'pip install -e .[research]' before using MusicGen") from exc
    torch.manual_seed(seed)
    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline("text-to-audio", model=MODEL_NAME, device=device)
    steps = min(1500, max(25, round(duration_seconds * 50)))
    try:
        result = generator(prompt, forward_params={"max_new_tokens": steps})
    except Exception as exc:
        raise UnavailableModelError(f"MusicGen inference failed: {type(exc).__name__}") from exc
    samples = to_mono(np.asarray(result["audio"], dtype=np.float32))
    return AudioResult(samples, int(result["sampling_rate"]), "meta-musicgen-small", seed)
