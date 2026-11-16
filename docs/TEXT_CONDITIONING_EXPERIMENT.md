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

The `active-window-dsp-v2` recipe renders one second per source combination
and takes a seeded 256 ms crop starting between 300 and 400 ms. The original
crop ended before the first bird call and reused the exact same solo ambient
waveform across splits. The new window contains bird activity and varies the
ambient phase. Regression tests check that all 168 underlying single-source
and mixture clips across the fixed train, validation, and benchmark seeds have
distinct waveform bytes. English and Hindi captions intentionally share their
underlying clip. All splits still use the same DSP generators.

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

Each training report includes training and validation noise MSE, split
fingerprints, training source, parameter count, and runtime. The legacy
`caption_alignment_probe` rotates the first 16 captions. Rotated captions
can still name the correct sound, so use the separate benchmark below to
measure synthetic caption retrieval.

### Held-out caption retrieval

```bash
python -m research.text_alignment \
  --checkpoint checkpoints/private-text-experiment --compare-initialization
```

The benchmark uses 42 clips from DSP seeds 1001 and 1002. It scores English and
Hindi captions with both the familiar training templates and held-out wording
that reverses the order of two-source captions while retaining the same sound
names. Each caption competes against every caption in the same language and
style with exactly one source replaced. The matched, wrong, and blank captions
all receive identical corrupted audio at each of three fixed noise levels.
The report covers 504 cases and 3,600 wrong-caption comparisons. Those cases
reuse the 42 clips, so they are correlated.

- **Pairwise accuracy** is the share of wrong captions beaten by the matched
  caption, averaged per case. A tie gives half credit.
- **Top-1 retrieval** is the share of cases where the matched caption has the
  lowest noise-prediction error. Tied winners split credit. Uniform chance is
  16.7% for one sound, 11.1% for two sounds, and 12.7% overall.
- **Mean MSE advantage** is wrong-caption MSE minus matched-caption MSE.
  Positive values favor the matched caption.

The report records the benchmark fingerprint, checkpoint hash, training recipe,
and separate summaries by language and wording, source count, noise level,
and recording seed. The optional initialization baseline reconstructs the
starting architecture from the checkpoint's recorded seed using the current
PyTorch runtime.

### CPU baseline

From the repository root, repeat this pair of commands with seeds 10, 19,
and 31. The measured runs used Python 3.12, PyTorch 2.15.0 CPU, two PyTorch
threads, 16 model channels, and a 256-step diffusion schedule.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m research.text_audio train \
  --steps 400 --seed 10 --channels 16 --diffusion-steps 256 \
  --output checkpoints/text-alignment-10
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m research.text_alignment \
  --checkpoint checkpoints/text-alignment-10 --compare-initialization
```

| Training seed | Initial pairwise | Trained pairwise | Initial top-1 | Trained top-1 |
| --- | ---: | ---: | ---: | ---: |
| 10 | 50.7% | 62.3% | 9.3% | 18.7% |
| 19 | 50.2% | 62.2% | 11.5% | 15.3% |
| 31 | 51.9% | 59.9% | 11.9% | 13.5% |

At 400 steps, pairwise retrieval improved for all three initializations.
High-noise pairwise accuracy stayed between 49.2% and 51.8%. Two-source top-1
retrieval cleared its 11.1% chance baseline in one run out of three.

Two exploratory reruns used the same settings with `--steps 1600` and fresh
output paths. Both improved two-source retrieval, though high-noise scores
remained much lower than the overall scores.

| Training seed | Overall pairwise | Overall top-1 | Two-source top-1 | High-noise pairwise |
| --- | ---: | ---: | ---: | ---: |
| 10 | 69.7% | 27.4% | 21.7% | 52.4% |
| 19 | 68.1% | 24.6% | 18.3% | 56.0% |

These results measure denoising on synthetic clips with six known source
names. The longer reruns reuse the same benchmark and are exploratory.
Generated-audio quality and natural-language prompt adherence remain
unevaluated.

Before any original model-generation milestone or distribution, Sonumio needs
longer audio modeling, licensed human-reviewed varied audio datasets, genuinely
held-out descriptions and recorded sessions, objective quality checks,
independent listening studies, source attribution, voice privacy review,
memorization evaluation and robust hardware/resource benchmarks.

**DO NOT merge this draft as a release, publish pre-releases/tags, or publish
weights without explicit project-owner approval.**
