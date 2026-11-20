"""Research-only adapter for Meta MusicGen-small weights.

Uses a pinned Hugging Face revision, refuses remote code, and requests safetensors.
Weights are CC-BY-NC-4.0 and are not bundled with Sonumio. This optional
adapter has not been weight/GPU-tested in the baseline test environment.
"""
from ..audio import AudioResult, to_mono
from ..errors import UnavailableModelError

MODEL_NAME = "facebook/musicgen-small"
MODEL_REVISION = "4c8334b02c6ec4e8664a91979669a501ec497792"

def generate(prompt: str, duration_seconds: float, seed: int = 0) -> AudioResult:
    try:
        import numpy as np
        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
    except ImportError as exc:
        raise UnavailableModelError(
            "Install optional dependencies with 'pip install -e .[research]' before using MusicGen"
        ) from exc

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=False,
    )
    model = MusicgenForConditionalGeneration.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, trust_remote_code=False,
        use_safetensors=True,
    ).to(device)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
    steps = min(1500, max(25, round(duration_seconds * 50)))
    try:
        with torch.inference_mode():
            audio_values = model.generate(**inputs, max_new_tokens=steps)
        samples = to_mono(np.asarray(audio_values[0, 0].detach().cpu(), dtype=np.float32))
        rate = int(model.config.audio_encoder.sampling_rate)
    except Exception as exc:
        raise UnavailableModelError(f"MusicGen inference failed: {type(exc).__name__}") from exc
    return AudioResult(samples, rate, "meta-musicgen-small", seed)
