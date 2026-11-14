import io
import wave
import numpy as np
import pytest
from sonumio.audio import encode_wav, metrics
from sonumio.errors import GenerationError, UnsupportedPromptError, UnavailableModelError
from sonumio.registry import ENTRIES
from sonumio.service import generate_audio

@pytest.mark.parametrize("prompt", [
    "Rain and thunder in a forest", "बारिश और गरज जंगल में",
    "Ocean waves", "Wind and birds", "cinematic ambient drone",
])
def test_real_wav_generation(prompt):
    generated = generate_audio(prompt, 1.25, seed=42)
    with wave.open(io.BytesIO(generated.wav_bytes), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 30_000
        assert np.frombuffer(wav.readframes(30_000), dtype="<i2").std() > 0
    assert generated.metrics["clipping_fraction"] == 0.0

def test_deterministic_output():
    a = generate_audio("rain storm", 0.5, seed=17)
    b = generate_audio("rain storm", 0.5, seed=17)
    c = generate_audio("rain storm", 0.5, seed=18)
    assert a.wav_bytes == b.wav_bytes
    assert a.wav_bytes != c.wav_bytes

@pytest.mark.parametrize("seconds", [0, -1, 15.1, float("nan"), float("inf")])
def test_invalid_duration(seconds):
    with pytest.raises(GenerationError):
        generate_audio("rain", seconds)

@pytest.mark.parametrize("prompt", ["", "  ", "x" * 801, "Make John say hello", "clone the voice of a teacher"])
def test_invalid_or_voice_prompt(prompt):
    with pytest.raises(GenerationError):
        generate_audio(prompt, 1)

def test_no_fake_generation():
    with pytest.raises(UnsupportedPromptError):
        generate_audio("A loud applause in a stadium", 1)

def test_musicgen_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("SONUMIO_ENABLE_MUSICGEN", raising=False)
    with pytest.raises(UnavailableModelError):
        generate_audio("orchestral music", 1, model_id="meta-musicgen-small")
    assert ENTRIES["meta-musicgen-small"].original_sonumio_model is False

def test_invalid_pcm():
    with pytest.raises(ValueError):
        encode_wav(np.array([np.nan], dtype=np.float32), 24000)
    with pytest.raises(ValueError):
        encode_wav(np.array([0.1], dtype=np.float32), 1000)
    assert metrics(np.array([0.0, 0.5], dtype=np.float32))["clipping_fraction"] == 0
