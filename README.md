# Sonumio

**Imagine it. Hear it.** Open-source generative audio platform for creating speech, music, and sound effects from text prompts, powered by evolving AI models.

> Experimental development: the default procedural DSP synthesizer generates actual WAV audio, but is NOT an AI model. Original Sonumio-1 does NOT exist yet.

## Setup
Python 3.11+:
1. `python -m venv .venv`
2. Activate the environment; on Windows run `.venv\Scripts\activate`
3. `python -m pip install -e '.[test]'`
4. `sonumio generate "Rain and thunder in a forest" --seconds 3 --output outputs/rain.wav`
5. `python -m pytest -q`
6. `sonumio serve` then open http://127.0.0.1:8000/docs

The procedural baseline supports rain, thunder, wind, birds, water and ambient tones in English/Hindi. Unsupported prompts raise an error. Sample format: 24 kHz mono PCM WAV.

The optional Meta MusicGen-small adapter is a THIRD-PARTY pretrained text-to-music model, research-only, disabled by default. Meta weights are CC-BY-NC-4.0 and not covered by Sonumio's Apache-2.0 source license; you must read and respect the upstream terms and separately install research dependencies. Set `SONUMIO_ENABLE_MUSICGEN=1` to enable. This adapter has NOT undergone real model weight/GPU validation in this prototype.

API: `GET /health`, `GET /v1/models`, `POST /v1/generations` and `GET /v1/generations/{id}/audio`. Do not expose the local server publicly without authentication, quotas, rate limits and cleanup.

Original Sonumio-1 and later model generations are research goals, NOT software version numbers or existing trained weights. See `docs/` for architecture, evaluation and roadmap. All releases/model weight publications need owner approval.

Source license: Apache-2.0. External weights and datasets retain their own terms.

## Original-model research scaffolding (development only)

From the repository root, after installing the optional PyTorch research dependencies:

- `python -m research.spectral_smoke --steps 40 --seed 7`: trains a tiny synthetic-spectrum regression network, records held-out error and saves a local NPZ checkpoint. This is **not** a neural waveform generator and not Sonumio-1.
- `python -m research.dataset_manifest path/to/manifest.jsonl`: checks data-rights declarations, voice exclusions, recording/checksum duplicates and group-isolated evaluation splits. This cannot verify legal rights without human review.

The MusicGen adapter uses a pinned upstream revision with safetensors loading and remote code disabled. Actual neural checkpoint execution is still unverified.

## Development-only performance measurements

Each successful generation reports `generation_wall_seconds`, `real_time_factor`, peak/RMS dBFS and the hard-clipping fraction. Run `python benchmarks/latency.py` after installing the package. The benchmark measures **procedural CPU synthesis**, not pretrained or original neural model latency.

## Sound scene generation (experimental procedural mixing)

The `scene` command and `POST /v1/scenes` can position up to eight DSP sound
layers on a 30-second timeline with per-layer fades, start time, seed and gain.

```bash
sonumio scene examples/forest_storm_scene.json --output outputs/forest_storm.wav
```

See [scene documentation](docs/SCENES.md). This is an audio mixing feature, not
neural sound generation.

## Original generative-model research experiment

An initial **trainable waveform diffusion** experiment now synthesizes very short,
8 kHz waveforms from six category conditions. Its training examples are created
by Sonumio's procedural engine. This is an experimental neural generator, **not
Sonumio-1** and not production-quality prompt-driven audio.

```bash
python -m pip install -e '.[research,test]'
python -m research.wave_diffusion train --steps 50 --output checkpoints/wave-diffusion
python -m research.wave_diffusion sample --checkpoint checkpoints/wave-diffusion --prompt rain --output outputs/research-wave.wav
```

See [diffusion research limitations](docs/DIFFUSION_EXPERIMENT.md). Locally trained
weights are excluded from Git, and no weights, releases or pre-releases are authorized.
