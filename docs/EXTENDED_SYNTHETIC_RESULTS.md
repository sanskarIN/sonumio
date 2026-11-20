# Actual synthetic audio training results — three-seed extended experiment

**Research evidence, 16 November 2026 — experimental only.**

- **GitHub Actions training run:** [43848233806](https://github.com/sanskarIN/sonumio/actions/runs/43848233806) (**success**).
- Source commit: [1dca95e8](https://github.com/sanskarIN/sonumio/commit/1dca95e8f5875c7aba01c1f8efc6716c2bc2b997).
- Methods and provenance: [EXTENDED_SYNTHETIC_TRAINING.md](EXTENDED_SYNTHETIC_TRAINING.md).
- Project: **Sonumio**, not Sonumio-1; no releases, tags, or public model weights.
- Training used only first-party procedural DSP sound recordings. **No real-world,
  licensed external recordings** were submitted or used.

## Prespecified setup and completed execution

| Setting | Verified run |
| --- | --- |
| Initialization seeds | 17, 29, 41 |
| Codec updates for each seed | 128 |
| Raw DDPM updates per seed | 256 |
| Latent DDPM updates per seed | 256 |
| Training audio | 42 unique first-party synthetic waveforms, 84 bilingual caption pairs |
| Full evaluation audio | 42 distinct synthetic waveforms, 84 English/Hindi captions |
| Categories and mixtures | Six base categories and all 15 two-source mixtures |
| Clip duration | 1.024 seconds |
| Sample rate | 8 kHz mono |
| Diffusion schedule | Eight steps |
| Raw DDPM parameters | 23,409 |
| Latent DDPM parameters | 20,128 |
| Separately trained codec parameters | 6,337 |
| Training environment | Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6; 2 CPU threads |
| Configuration SHA256 | c9d87311c83b2ee46dff6f740a2c50c72b42bf6c51f4e9ee4da5c7df164c576d |
| Hard-negative token SHA256 | 6fd3f689c54b11ec80acb0a1c24d1f50908778750d0889eac61affdf7ce2b3c1 |
| Checkpoint and weight files | **None generated/published** |
| Experimental report | Ephemeral JSON on GitHub runner and logged text |

The workflow checked that the only file written under \`checkpoints/\` was
the local metrics JSON, all 21 held-out source combinations were evaluated,
the protocol matched expectations, no exact train/evaluation audio overlap
occurred and no weight publication flags were present.

**This is actual optimizer training**, not an invented result or a single
forward-pass test. Every model parameter update was computed from the
first-party DSP recordings. The resulting model weights were **discarded**
after the run, not released for download.

## Within-representation denoising

Mean reduction of noise-prediction MSE relative to an all-zero noise
predictor, calculated as \`1 − paired MSE / zero-predictor MSE\`:

| Noise band | Raw waveform DDPM | Latent DDPM | Positive initialization seeds |
| --- | ---: | ---: | --- |
| Low | 65.84% | 14.78% | 3/3 for both |
| Medium | 86.98% | 20.83% | 3/3 for both |
| High | 90.72% | 21.98% | 3/3 for both |

A larger reduction is better **within that model's own signal domain**, but
the numbers **cannot** be compared directly between the raw and latent arms:
they operate on different tensor representations with unequal models and
training costs. These statistics evaluate predicted noise, not a generated
waveform against a real reference distribution.

## Held-out bilingual caption checks

Each true caption was compared with a wrong caption in the *same language*,
with the *same number of sounds* but **zero overlapping sound categories**.
All caption candidates in a comparison received the same corrupted signal,
noise target and timestep.

The quantity below is \`wrong-caption noise MSE − matched-caption noise MSE\`.
Positive means the matched caption gave a lower noise-prediction error.

| Noise band | Raw: mean MSE advantage (positive seeds) | Latent: mean MSE advantage (positive seeds) |
| --- | ---: | ---: |
| Low | +0.006323 (3/3) | +0.000423 (3/3) |
| Medium | +0.002774 (3/3) | +0.000228 (3/3) |
| High | +0.001236 (3/3) | +0.000131 (2/3) |

These are small, synthetic, category-template-based text-alignment signals.
One latent high-noise seed still had the *wrong* caption performing slightly
better. The full 21-combination holdout does not add unseen semantic classes
or independent sound recordings beyond the procedural DSP family.
No conclusion about open-ended prompt comprehension or realistic sounds follows.

## Latent codec reconstruction

Across the three runs, the codec's mean held-out **waveform MAE** was
**0.116474**, median **0.116102**, sample standard deviation **0.002119**,
and observed per-seed range **0.114565–0.118754**. This is a
sample-aligned reconstruction error, **not** a listening score. No
human listening evaluation or full spectral reconstruction report was run
on the trained weights before they were discarded.

## Comparison with the early pilot

The preceding [48-update pilot](MULTI_SEED_PILOT_RESULTS.md) used fewer
updates and only **four** evaluated source combinations, while this
extended experiment uses **all 21**. Both used the same synthetic train
seed corpus and v2 category-disjoint controls, but their evaluation
distributions differ. Consequently, differences in reported statistics
cannot be attributed solely to increased training steps. The extended
results are stronger evidence that the training pipeline learns **noise
prediction on these synthetic sounds**, not a controlled proof of
better real-world generation or architecture superiority.

## Remaining gates

- Acquire and independently review a legally usable, varied non-voice
  audio dataset, including consent and license evidence for each source.
- Train on longer 16–32 kHz or higher quality recordings, with
  session-disjoint train/validation/test splits.
- Compare similarly sized models and total training compute; account for
  codec training and reconstruction losses.
- Keep private, validated checkpoints only when a later dedicated
  local reproducibility workflow is authorized. **Do not publish weights.**
- Perform actual prompt-conditioned audio sampling, FAD/CLAP-style reference
  diagnostics with fixed feature extractors, independent blinded listening,
  memorization audits, and privacy/copyright checks.
- Complete production-quality storage, model lifecycle and UI/UX.
- Obtain separate explicit project-owner approval for any PR merge, model
  weights, stable release, pre-release or tag.

**Verdict:** real research training succeeded and the held-out synthetic
noise metrics improved relative to the zero-predictor baselines. The
actual end-user generative-audio model is **not yet release-ready**.
