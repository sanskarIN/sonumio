# Research: adjacent-window velocity continuation

**Experimental, CPU-only, 8-kHz, first-party synthetic sounds. No model weights, WAVs, pre-releases, or Sonumio-1.**

## Why the dataset matters

Prior clean-prefix work tested the mechanics of attaching a known prefix to
a generated one-second waveform, but its training clips did not represent
two windows from a **single continuously generated recording**. For a
meaningful continuation experiment, history and target must share the
same actual source and sample-level overlap.

`research/adjacent_audio.py` generates one 3-second first-party DSP
recording for each supported source-combination and recording seed. It
converts the complete source once from 24 kHz to 8 kHz using the
existing anti-alias filter, applies a deterministic seed-derived slow
gain envelope, and extracts two adjacent 8192-frame windows with a
specified shared prefix of 128–4096 frames (multiples of 64).

```text
source                0 ------------------------ 3 seconds
previous              [======= 8192 =======]
future                            [======= 8192 =======]
                         <-- shift = 8192 - prefix_frames -->
previous[-prefix:] == future[:prefix]       (exact float32 equality)
reconstructed source region:
  cat(previous, future[prefix:]) == source[crop_start:crop_end]
```

The windows come from the **same resampled waveform**. Resampling each
window independently would introduce edge differences that undermine
exact adjacency. Dataset constructors verify full reconstruction,
waveform hashes, source length/range, finite values and exact overlap.

## Provenance and leakage prevention

The 21 supported first-party DSP source groups (six individual types and
15 two-type combinations) are each rendered with an explicit group seed.
Two captions, English and Hindi, point to the **same audio waveform**.
For two recording seeds, this creates **42 acoustic recordings and
84 caption examples**, not 84 independent sound recordings.

Each recording has:
- a deterministic source identity/fingerprint,
- a previous/future waveform hash and exact stitch hash,
- source-combination IDs and the recording seed,
- bilingual caption data and their shared recording-group identity,
- a deterministic seed-dependent crop offset and slow gain envelope.

Some procedural sources (especially the oscillator/ambient category)
ignore their original generator seed. The deterministic gain envelope
prevents the resulting raw source bytes from accidentally being identical
across recording seeds; **it does not make those sources independent
natural recordings**. This remains a tightly related synthetic task.
Recordings derived from the same group are never split between training
and held-out evaluation.

`audit_adjacent_split` rejects any shared recording seed, repeated
source-hash, group-identity collision or incompatible prefix length
across training and held-out sets. A seed-disjoint split is still
**not evidence of out-of-distribution generalization**.

## Real paired training experiment

`research/adjacent_prefix_study.py` trains the exact same raw
16-channel velocity-conditioned Sonumio denoiser architecture twice,
starting with bit-identical initial parameters:

1. **True-history model:** sees the exact clean prefix from the
   preceding recording window.
2. **Hidden-history model:** sees a zero prefix in the same input positions.

Both variants receive identical caption examples, future targets,
noisy future suffixes, timesteps, Gaussian noise, optimizer type and
learning rate, updates, gradient clipping and loss mask. The only
intended difference is visibility of actual previous audio samples.

```text
clean future   = current recording's later 8192-frame window
noise/t        = shared local generator draw
mixed          = sqrt(alpha_bar_t) * future + sqrt(1-alpha_bar_t) * noise
v target       = sqrt(alpha_bar_t) * noise - sqrt(1-alpha_bar_t) * future
true input     = concat(previous[-prefix:], mixed[prefix:])
hidden input   = concat(zeros(prefix), mixed[prefix:])
loss           = mean((pred_v[prefix:] - v_target[prefix:]) ** 2)
```

The prefixes are **not** prediction targets. The model sees exactly
the type of clean input it will receive at prefix-conditioned inference.

### Held-out controls

Each model is evaluated over **every bilingual caption example** in
a disjoint set, using the **same** Gaussian corruption and selected
timestep. A third diagnostic sends the true-history network an
unrelated prefix from the *next acoustic recording group*. This is
an inference-only context-sensitivity test, not an additional trained
model and not an independent model seed.

Report three v-prediction MSE values:

- `true_history`: trained with correct history; evaluated with correct history.
- `hidden_history`: trained with zero history; evaluated with zero history.
- `wrong_history`: true-history model evaluated with wrong source history.

The report includes `hidden_minus_true` and `wrong_minus_true`.
The sign is descriptive, **not a guarantee**. If these values are near
zero or negative, the tiny denoiser may not use the visible history
meaningfully, or the task may not reward it. Scores are additionally
summarized **once per acoustic recording group**, rather than treating
English/Hindi versions of the same waveform as independent observations.

For a small number of held-out examples the study samples with clean
actual history through the matching velocity-DDIM sampler. It reports
the first generated sample jump versus the **true future waveform's**
first new-sample jump, plus sample-aligned MAE/RMSE and SHA256 hashes.
These are numerical waveform measures only. A good sample error is not
proof of listening quality, scene continuity or prompt semantics.

## Reproduce

With CPU PyTorch research dependencies installed:

```bash
python -m pytest -q tests/test_adjacent_audio.py \
  tests/test_adjacent_prefix_study.py

python -m research.adjacent_prefix_study \
  --model-seed 17 --updates 2 \
  --prefix-frames 512 --diffusion-steps 8 \
  --evaluations 2 --batch-size 2 \
  --training-seeds 0 --heldout-seeds 101
```

For a larger but still nonconverged exploratory experiment:

```bash
python -m research.adjacent_prefix_study \
  --model-seed 17 --updates 64 \
  --prefix-frames 1024 --diffusion-steps 16 \
  --evaluations 8 --batch-size 4 \
  --training-seeds 0,1 --heldout-seeds 101,102
```

The first command tests **21 acoustic training recordings** (42 English/
Hindi captions) and 21 held-out recordings; the second uses 42 and 42.
No audio or neural model data is saved. The CLI emits numeric-only JSON
to standard output.

## Limits and next research

This is still a small short-context, synthetic, 8-kHz denoiser. It
has not learned general text-to-audio semantics, singing, speech,
voice identity, musical structures, realistic environmental acoustics,
or long-range temporal coherence. Some synthetic sounds share generated
oscillator structure across source seeds; the seed-specific envelope
does not resolve that generalization limitation.

A strong experimental roadmap requires at least:
- multiple independent model initialization seeds and held-out group
  comparisons before choosing a training protocol,
- increasing training time based on validation rather than raw line counts,
- longer receptive field or explicit continuation memory,
- new rights-cleared, speaker/source-disjoint recordings,
- numerical and blind listener evaluations of transitions and
  prompt adherence,
- more representative sample rates and production safety review.

There is no inference that the Sonumio-1 model exists or that any model
weights may be published.
