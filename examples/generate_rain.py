"""Run this after installing Sonumio in editable mode."""
from pathlib import Path
from sonumio.service import generate_audio

audio = generate_audio("forest rain with thunder", 3.0, seed=123)
Path("sonumio-rain.wav").write_bytes(audio.wav_bytes)
print("Wrote sonumio-rain.wav", audio.metrics)
