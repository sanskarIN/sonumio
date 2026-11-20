# Sonumio Architecture — ADR 001

**Current working data path:** CLI or loopback FastAPI -> typed input validation -> safety heuristics -> explicit engine registry -> synthesis -> PCM WAV encoding -> output store + RMS/peak/clipping metadata.

## First working platform
- Python 3.11+, numpy DSP generator, FastAPI endpoints, CLI, local WAV outputs, pytest.
- `procedural-sfx-v0` is NOT an AI model; it produces deterministic real WAV samples with English and Hindi keyword matching.
- Third-party `meta-musicgen-small` is optional, noncommercial-weights research adapter, disabled by default, not checkpoint-tested.
- No pretrained Sonumio model or voice cloning capability.

## Next architecture
1. `sonumio-core`: task routing, configurable model registry, resampler, loudness controls, metadata.
2. `sonumio-models`: isolated adapters with upstream license, revision, hash, supported use cases and weight provenance.
3. `sonumio-api`: local first, future async generation queue, storage quotas, auth, cancellation, telemetry.
4. React + TypeScript + Vite frontend: accessible prompt form, model/task selection, waveform preview and downloadable audio.
5. `research/`: dataset manifests, reproducible training, checkpoints, evaluation and safety gates.
6. Rust and Tauri later only with measured motivation.

## Security gaps before public service
No auth, rate limiting, queue limits, retention/deletion or public abuse monitoring in this prototype. Do NOT bind to a public network address. Safetensors-only trusted checkpoint allowlist, revision pinning, datasets licensing and explicit voice consent are future work. Never publish weights without rights review.
