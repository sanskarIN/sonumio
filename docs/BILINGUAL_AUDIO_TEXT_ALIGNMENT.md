# Sonumio original bilingual text–audio alignment research

**Experimental learned embedding, not a pretrained CLAP/CLIP model, speech recognizer, copyright classifier, or Sonumio-1.**

## What problem this addresses

A generative audio model should eventually respond to the meanings of
text prompts, not just their bytes, category tokens or seeded sampling
noise. Sonumio's current original waveform/latent models use a small
UTF-8-byte Transformer text encoder, but numerical diffusion loss and
prompt-dependent waveform differences cannot by themselves establish
that a model understands "rain", "wind", "forest" or Hindi equivalents.

`research/audio_text_alignment.py` adds a **separately trainable**
small original audio–text alignment probe. This is an *evaluation and
representation-learning baseline*, not a perceptual metric.

The audio tower:
- accepts one-second 8192-sample, 8-kHz mono float32 first-party DSP;
- uses four small strided 1D convolution stages;
- combines learned global audio features with a 16-bin acoustic energy
  envelope before an original projection;
- outputs a normalized 16, 32 or 64-dimensional embedding.

The text tower:
- reuses Sonumio's original UTF-8 byte tokenizer and small learned
  Transformer; no external NLP models are loaded;
- outputs a matching normalized embedding after a separate projection;
- supports short English and Hindi caption templates.

Cosine similarity measures a **trained alignment model's own embedding
geometry**, not human-perceived similarity or real-world sound semantics.

## Strict category-disjoint InfoNCE

One first-party procedural sound combination is an acoustic *category
group* for this limited experiment. The six individual sound types and
their fifteen two-type mixtures form 21 groups.

An important contrastive-learning failure: if "recording of rain" in
English and "आवाज़ बारिश" in Hindi are paired with the exact same
rain waveform, they are **two valid positive descriptions of the same
sound**, not unrelated recordings. Similarly, two seeded clips from
"rain" should not be treated as mutually exclusive semantic negatives
simply because their waveforms differ.

Training samples therefore contain **at most one example per category
combination in each batch**. The other examples are from different
sound-combination groups, and only those may serve as negatives.

For each batch, normalized audio/text embeddings `A` and `T` give

```text
logits = A @ T.T / temperature
L = [cross_entropy(logits, diag_identity)
     + cross_entropy(logits.T, diag_identity)] / 2
```

Temperature is finite and bounded. Duplicate category IDs, malformed
text tokens, mismatched source batches, nonfinite embeddings, wrong
device/dtypes and unbounded audio are rejected **before** the loss.

The loss is differentiable through both neural towers and uses no
trained external audio model.

## Train and evaluate fairly

`research/audio_text_alignment_study.py` actually trains this
original embedding model with one category-group sample per minibatch,
cycling deterministic source-seed and language choices. It uses
first-party procedural Sonumio recordings only, with exact train/heldout
source SHA disjointness checks.

Held-out retrieval is **one recording per sound category** in each
language. That provides 21 candidate category labels. The same audio
recording is used for English and Hindi text separately; the pair is
not falsely counted as two independent audio recordings.

The study reports:
- English and Hindi category recall@1 / recall@5 and mean reciprocal rank;
- per-category retrieval ranks, and matched-versus-other cosine;
- English/Hindi same-category text-embedding cosine;
- a deliberately wrong **cyclically shifted caption-category** control;
- neural initial/trained parameter SHA256 and unchanged evaluation state;
- full first-party recording fingerprints and source-level counts.

A high recall on **21 templated categories** does **not** establish
open-vocabulary semantics, speech transcription, general music
understanding, speaker identity, license provenance or natural sound
quality. It only reports a narrow, synthetic category-matching task.

## Reproduce in CPU research environment

```bash
python -m pytest -q tests/test_audio_text_alignment.py \
  tests/test_audio_text_alignment_study.py

python -m research.audio_text_alignment_study \
  --seed 17 --updates 2 --batch-categories 4 \
  --temperature 0.1 --learning-rate 0.0005 \
  --training-seeds 0 --validation-seeds 101
```

This small pilot uses 21 unique training recordings and 21 held-out
recordings (each annotated in English and Hindi) and evaluates the
held-out 21-category retrieval panel. For broader but still limited
synthetic evaluation, use:

```bash
python -m research.audio_text_alignment_study \
  --seed 17 --updates 32 --batch-categories 8 \
  --training-seeds 0,1 --validation-seeds 101,102
```

Train for only 1–64 research steps, never claiming this is sufficient
for meaningful language understanding. The CLI prints numeric JSON
with hashes and category IDs; no model checkpoints, audio, embeddings,
pretrained model files or WAVs are written.

## Long-term scientific next steps

A useful text–audio alignment benchmark needs:
- genuinely independent acoustic source families and legally
  cleared real-world recordings,
- many unrelated and compositional natural-language descriptions in
  both English and Hindi, with human-validated positives/negatives,
- independent model-seed replication with uncertainty across real
  recording groups,
- matching sound identity, environment context and temporal events,
  not merely memorizing the 21 source-category templates,
- blind listening and audio–text adherence measurements, and
- clarity about what a similarity score can and cannot prove.

This code creates no Sonumio-1 model generation, release, weights,
external data downloads or claims about perceptual quality.
