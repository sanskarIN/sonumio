# Learned English/Hindi text conditioning — research only

This experiment introduces an **original neural text encoder** connected to
Sonumio's small waveform diffusion denoiser. It is a real, trainable
text-to-waveform prototype, but **not Sonumio-1**, not open-domain generative
audio, and not ready for creator use. No weight distribution is authorized.

## Technical implementation

- `research/text_audio.py`: UTF-8 byte tokenizer, 259 vocabulary positions,
  at most 128 UTF-8 bytes (strict failure instead of silent truncation).
- Learned byte embeddings, learned positions, one Transformer encoder layer
  with four attention heads, masked mean pooling, and a learned null prompt
  embedding for classifier-free guidance.
- Original text/time-conditioned six-residual-block 1D Conv1D denoiser, DDPM
  forward corruption and iterative noise-prediction sampling.
- The neural model sees byte-level text features rather than hand-coded sound
  categories. **The training signal still covers only six original DSP sound
  categories and 15 two-sound combinations**, not arbitrary natural language.
- 8 kHz, 2,048 samples per clip (0.256 s), no stereo, no TTS, no editing,
  no music-quality control, no long-duration generation.
- Saves local numeric-only NumPy `.npz` checkpoints plus JSON manifest and
  SHA-256 integrity. Loading rejects unexpected tensor names, shapes, dtypes
  and nonfinite values. No pickle deserialization, no downloaded weights.

## Synthetic experiment

The self-generated DSP training set has 21 category combinations, four seeds,
and two language captions per combination = **168 examples**. Held-out
recording seeds 101 and 102 produce **84 evaluation examples**. These are
seed-separated, but captions are a tiny hand-written template collection,
so a low validation loss does **not** prove generalization to natural prompts.

```bash
python -m pip install -e '.[research,test]'
OMP_NUM_THREADS=2 python -m research.text_audio train \
  --steps 50 --seed 10 --channels 16 --diffusion-steps 8 \
  --output checkpoints/private-text-experiment
python -m research.text_audio sample \
  --checkpoint checkpoints/private-text-experiment \
  --prompt 'बारिश और हवा' \
  --output outputs/text-experiment.wav
python -m pytest -q tests/test_text_audio.py
```

For Windows PowerShell, omit `OMP_NUM_THREADS=2` or use
`$env:OMP_NUM_THREADS = '2'` before calling Python. These short training runs
are smoke tests, **not** meaningful model quality training.

## Rights-gated local WAV option

```bash
python -m research.text_audio train --steps 50 \
  --manifest private-data/reviewed.jsonl \
  --audio-root private-data/approved-recordings \
  --output checkpoints/private-text-reviewed
```

The loader reuses Sonumio's existing local WAV checksum, decoding, metadata,
manual rights-declaration and recording-group split checks, then retains
captions for byte-text training. This **does not independently validate legal
rights or inspect recordings for voices**. Humans must review dataset consent,
licenses, captions and voice exclusion before training. Currently the rights
loader accepts only **one recognized sound category per WAV caption**.

## Evaluation and honesty rules

Each training report contains: training and held-out noise MSE, parameter
count, runtime, train/evaluation dataset fingerprints, the training source,
and a **matched-versus-shuffled caption probe**. The latter corrupts audio
once with fixed noise and evaluates the same noisy audio against its paired
caption, a shuffled caption and no caption. It is **not** an audio-quality,
language-comprehension, source-separation or memorization benchmark.

On a local smoke run (8 train steps, 16 channels, 8 DDPM steps):

- Held-out noise MSE: approximately **0.8812**.
- Paired-caption noise MSE: **0.8540**; shuffled-caption noise MSE: **0.8531**.
- Matched prompts performed *slightly worse*, not better, in this diagnostic.
  This is not evidence of successful text-audio alignment.

Before any original model-generation milestone or distribution, Sonumio needs
longer audio modeling, licensed human-reviewed varied audio datasets, genuinely
held-out descriptions and recorded sessions, objective quality checks,
independent listening studies, source attribution, voice privacy review,
memorization evaluation and robust hardware/resource benchmarks.

**DO NOT merge this draft as a release, publish pre-releases/tags, or publish
weights without explicit project-owner approval.**
