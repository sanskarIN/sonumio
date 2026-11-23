# Sonumio: independent original English/Hindi audio–text model replication

**Experimental representation learning only. No Sonumio-1 weights, audio exports or open-vocabulary claims.**

## Why compare independent initializations

Sonumio's small original audio–text embedding experiment learns to
match English/Hindi captions to 21 combinations of six procedurally
generated DSP sound categories. Its small 1D audio encoder and UTF-8
Transformer text encoder are trained from scratch; no pretrained CLAP,
speech recognizer or public model embedding is used.

One model can rank held-out template captions well or poorly based
on its initial parameters. Moreover, two different recordings within
the same source family should not be treated as semantic negatives,
and English/Hindi captions for the same recording do not make a
second independent sound.

`research/audio_text_alignment_replicates.py` repeats the **actual**
training for 2–4 independent model initialization seeds, keeping all
recording data, semantic categories and optimization settings fixed.

## Evidence validation before any aggregate

The replication refuses any run with:
- a different or forged canonical configuration SHA256,
- a missing/mixed Sonumio original model protocol,
- changed first-party training and validation recording fingerprints,
- a different 21-audio held-out retrieval candidate panel or duplicate
  recording hash,
- changed audio family coverage or bilingual caption counts,
- repeated initial or trained neural model state SHA256,
- missing training updates, inconsistent loss minima/maxima,
  published neural weights or unearned perceptual claims,
- training minibatches with **duplicate sound-combination groups**,
- a category name, recording seed, English/Hindi caption flag or source
  index that does not match the original synthetic source layout,
- missing candidate retrieval ranks or summary scores not matching
  their underlying ranking arithmetic.

The model reports rank@1, rank@5 and mean reciprocal rank separately
for English, Hindi and a deliberately wrong cyclic caption shuffle.

It **recomputes** each ranked retrieval metric from all 21 actual
category ranks. A forged "100% accuracy" headline with incorrect
per-category ranks fails the study. The source dataset's 21 categories
are still highly related synthetic sounds, not independently licensed
real-world environmental sources.

## Which outcomes are aggregated

Each training initialization contributes one independent observation
per language and metric. The descriptive summary includes observed
mean, median, sample standard deviation, min/max and sign counts
across model initializations. English/Hindi variants are **paired
measurements on the same 21 audio candidates**, not 42 independent
acoustic recordings. The cyclic wrong-caption retrieval panel is
an evaluation-only negative control, not another trained model.

The English/Hindi same-category text-vector cosine is also reported
across model initializations as a descriptive measure. A high cosine
does not establish translation fidelity, language understanding,
speech content or naturally generated audio.

No model is declared the winner based on these small pilot scores,
and no p-values or population confidence bounds are invented.

## CPU developer commands

```bash
python -m pytest -q tests/test_audio_text_alignment_replicates.py

python -m research.audio_text_alignment_replicates \
  --model-seeds 17,29 --updates 2 \
  --batch-categories 4 --training-seeds 0 \
  --validation-seeds 101
```

The CI smoke trains two real from-scratch neural models on 21 unique
source-category recordings and evaluates retrieval with one held-out
recording per category, each with English/Hindi descriptions.

A larger experiment, still far from production readiness:

```bash
python -m research.audio_text_alignment_replicates \
  --model-seeds 17,29,41 --updates 32 \
  --batch-categories 8 --training-seeds 0,1 \
  --validation-seeds 101,102
```

Every run uses local CPU PyTorch, first-party deterministic DSP,
explicit seeded data/optimizers and SHA256 state checks. The CLI
outputs bounded metadata and numerical JSON, **not** audio files,
neural embeddings, model weights, proprietary datasets or trained
model distribution.

## Remaining research requirements

A useful generative text-to-audio system needs natural-language
paraphrases, real multilingual event captions, robust source-disjoint
data rights, advanced acoustic embeddings, longer contextual
generation, subjective listening tests and meaningful model-seed
replication across genuinely independent acoustic sources.

These results are not evidence of natural speech/music generation,
sound originality, copyright provenance, human prompt adherence or
Sonumio-1 model maturity. No release, tag or model weights publication
is authorized by this experimental validation.
