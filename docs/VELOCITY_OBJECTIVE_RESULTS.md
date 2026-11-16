# Measured velocity versus epsilon DDPM results — Sonumio

**16 November 2026 · experimental DSP-only neural research · no model release**

The matched velocity-prediction experiment **completed successfully** in
GitHub Actions and substantially reduced severe waveform saturation on
both original research architectures. It did **not** establish natural-
sounding or correctly prompted sound effects, speech or music.

## Verifiable evidence

- **Corrected, completed paired training and decoder audit:**
  [GitHub Actions run #43865832372](https://github.com/sanskarIN/sonumio/actions/runs/43865832372)
  at code commit [4d99aa2e](https://github.com/sanskarIN/sonumio/commit/4d99aa2e2b21f4cbd9545595cabe4e12e2247eb6).
- All four GitHub workflows passed on that code commit: Python tests,
  web checks, CPU research (**175 tests passed**) and actual paired
  in-memory training/generation.
- **Corrected report configuration SHA256:**
  \`d6f338ca954c7372b92fd5e6bb1dfc8dce00460c5b8ccfafebf7869d097fea7d\`.
- Math and training protocol: [VELOCITY_OBJECTIVE_STUDY.md](VELOCITY_OBJECTIVE_STUDY.md).
- **No model weights, generated audio/WAV files, release artifacts,
  prerelease/stable version tags or model-generation releases were saved
  or published.** Only numeric JSON metrics existed ephemerally on the
  GitHub runner.

## Controlled design

The two prediction objectives **epsilon** and **velocity (v)** started
from bit-identical raw and latent neural parameter arrays and saw
the same data examples, Gaussian noise draws, chosen diffusion timesteps,
text-conditioning drop masks, initial model width, AdamW settings, and
gradient clipping within each model architecture and seed.

| Setting | Recorded configuration |
| --- | --- |
| Model-initialization seeds | 17, 29, 41 |
| Dataset | Sonumio-owned six-source synthetic DSP and two-source mixtures |
| Unique training recordings | 42, with 84 English/Hindi caption pairs |
| Unique held-out recordings | 42, with 84 English/Hindi caption pairs |
| Sampling | 8-step cosine schedule, alpha-bar terminal ≈ 0.00003747 |
| Codec optimization | 128 updates per seed, **one shared frozen codec per pair** |
| Denoiser optimization | 256 updates per objective and per model arm |
| Model arms | 1×8192 raw waveform; 16×128 audio latent |
| Final generated output | 1.024-second, 8-kHz mono |
| Fixed prompts | rain/बारिश and wind/हवा |
| Sampling noise seed | 1500 + initialization seed |

This is a paired **prediction-objective** comparison under the cosine
schedule, not a raw-vs-latent architecture comparison. The epsilon
and velocity losses have **different targets and cannot be compared
as though their values are equivalent**.

## Numerical generated-audio outcomes

Mean over four fixed bilingual prompts per seed, followed by the mean
over three model-initialization seeds:

| Metric | Raw epsilon | Raw velocity | Latent epsilon | Latent velocity |
| --- | ---: | ---: | ---: | ---: |
| Final clipped/saturated waveform samples | **94.642%** | **0.004%** | **71.131%** | **0%** |
| Final waveform RMS | 0.9819 | 0.1673 | 0.9576 | 0.0479 |
| Pre-limiter signal peak | 207.41 | 0.849 | 88.12 | 0.230 |
| Pre-limiter signal RMS | 16.97 | 0.167 | 16.96 | 0.048 |
| Pre-limiter abs(value) >1 | 94.637% | 0.004% | 91.673% | 0% |
| Rain-vs-wind mean absolute waveform difference | 0.3416 | 0.0413 | 0.0496 | 0.00516 |

**Critical interpretation of "pre-limiter":** For raw waveform it is
the neural DDPM output *before external amplitude clamping*.
For the latent architecture it is the convolutional codec decoder
**before its trained internal Tanh nonlinearity**. The latter is not
already an actual audio waveform exceeding [-1,1]; it is a neural
activation representing the risk of internal decoder saturation.

The latent **Tanh saturation fraction**, measured where pre-Tanh
abs(activation) ≥ atanh(0.999), was **71.131%** for epsilon
and **0%** for velocity. The actual decoded-waveform clipping fraction
matches because the Tanh output gets extremely close to ±1.
The pre-activation was independently checked to reproduce the decoded
waveform when Tanh was applied, so this internal limiter cannot hide
unbounded values from the audit.

The velocity objective reduced the mean final saturated/clipped samples
by approximately **94.638 percentage points** for raw and
**71.131 percentage points** for latent. These reductions occurred for
**all 3 paired initialization seeds** in both architectures.

## Individual-seed clipping results

| Seed | Raw epsilon | Raw v | Latent epsilon | Latent v |
| --- | ---: | ---: | ---: | ---: |
| 17 | 94.168% | 0% | 75.119% | 0% |
| 29 | 95.004% | 0.0122% | 82.312% | 0% |
| 41 | 94.754% | 0% | 55.963% | 0% |

The raw velocity model still produced some out-of-range samples for
one seed; no audio-quality acceptance threshold has been reached.
Velocity may suppress saturation while producing muted or acoustically
unconvincing sounds. This experiment did not include blinded listening.

## Evidence limitations: more than clipping remains

1. **Audio correctness unknown.** The metrics do not prove that
   "rain" actually sounds like rain or that Hindi/English produce
   semantically equivalent audio. Waveform-change magnitude is not a
   language-understanding score.
2. **Small synthetic data.** No diverse real-world rights-reviewed
   recordings, fresh recording sessions, listening tests, FAD/CLAP,
   sound-class recognition, or music/TTS evaluation were used.
3. **Training not converged.** Only 256 updates per denoiser
   and an extremely short 8-step sampling schedule were evaluated.
4. **Codec bottleneck.** Held-out codec reconstruction MAE across
   seeds was approximately 0.10094, 0.11464 and 0.11332.
   That is not a perceptual fidelity guarantee.
5. **Losses are different targets.** Raw vs latent and epsilon vs
   velocity noise-objective MSE are not directly comparable quality
   scores. The validated improvement is specifically *numerical
   saturation reduction after inference*.
6. **Cross-run byte identity is unverified.** The exact synthetic
   data fingerprints on the initial and corrected v-audit workflows
   differ even though source seeds and reported NumPy version were
   unchanged. This does *not* invalidate the paired comparisons
   **within each execution**, which share exactly the same in-memory
   examples, but it does prevent claiming bitwise-identical dataset
   outputs across those two separate GitHub jobs. The cause (e.g.
   floating-point environment differences) has not been established
   and requires investigation before cross-job comparisons.

## Research recommendation — next controlled work

Continue developing velocity prediction as a **promising numerical
research direction**, not an approved production model. Next:

1. Investigate cross-run synthetic recording fingerprint drift and
   add reproducibility checks for audio generation and numeric arrays.
2. Evaluate multi-noise-seed audio diversity, amplitude, spectral
   reconstruction and output distribution against held-out DSP,
   with separate tests for prompt meaning.
3. Test longer 16–32 kHz audio, stronger codec capacity, more meaningful
   training budgets, and less extreme diffusion schedules using controlled
   A/B experiments.
4. Before any real-world data, verify licensing, attribution, consent and
   non-voice restrictions; maintain recording-session-disjoint splits.
5. Add blinded listener evaluations, sound-class recognition, memorization
   and privacy checks before any actual Sonumio model generation.

**Draft PR #1 remains unmerged. No prerelease, stable release, tag, or
model weight publication follows from passing this experiment.**
