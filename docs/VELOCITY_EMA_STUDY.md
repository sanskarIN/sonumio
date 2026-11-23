# Research: EMA averaging for original Sonumio velocity denoisers

**Experimental CPU-only work. No Sonumio-1 model, released weights, or original sound quality claim.**

## What this experiment tests

An exponential moving average (EMA) keeps a separate smoothed copy of the
model parameters during normal velocity-prediction training:

```text
shadow_new = decay * shadow_old + (1 - decay) * online_after_optimizer_step
```

The online model always receives the gradients and optimizer updates. The
EMA copy never receives gradients or optimizer steps. The two candidates
therefore share the **same online optimization trajectory**. Comparing
one live model to its EMA shadow is more controlled than comparing separately
trained models or changing random seeds between runs.

No EMA shadow is authorized for production or publication.

## Security and numerical correctness

`research/weight_ema.py` implements all-or-nothing updates for finite CPU
float32 model tensors. It verifies every tensor name, dtype, shape and device
before mutation, makes every candidate updated tensor, checks their finite
values, and only then writes to the in-memory shadow. Integral buffers are
copied without interpolation. A source NaN, mismatched model or float32
interpolation overflow cannot leave a partially updated EMA state.

Shadows are private, frozen and evaluated through independent deep copies.
The source's parameters and train/eval mode remain unchanged during EMA
updates. The operation does not use the global torch RNG, the disk, or any
model serialization interface. One-way hashes give identity evidence
without publishing the actual tensors.

A warmup option caps the effective decay after `u` completed updates at
`min(decay, (1+u)/(10+u))`. It is disabled by default and is always
explicitly recorded; **EMA decay and warmup are experimental hyperparameters**.

## Controlled study protocol

`research/velocity_ema_study.py` trains two existing architectures:
an 8192-frame raw waveform v-predictor and a 16-channel × 128-frame latent
v-predictor paired with a small 64× neural audio codec.

- **Training data:** Sonumio first-party procedural sound effects only.
- **Split:** 42 unique recordings for training, 42 independent held-out
  recordings. Each recording has an English and Hindi caption, giving 84
  caption examples per split. Exact recording SHA overlap is rejected.
- **Learning:** Both models use a cosine diffusion schedule and the
  established velocity target. After every online AdamW step, the EMA is
  updated. The held-out split is not used for optimization.
- **Held-out numerical objective:** For each arm, all 84 caption examples
  contribute to noise-prediction MSE. Online and EMA see identical
  timesteps, Gaussian corruption and text tokens. Raw-vs-latent domain
  MSE values are not directly comparable.
- **Generated audio numerical controls:** Online and EMA use the same
  fixed caption/seed with Sonumio's **full-budget deterministic v-DDIM**
  sampler. Reports clipping, RMS, spectral statistics, an exact waveform
  hash, numerical waveform difference and signal source provenance.
  Latent activation levels are explicitly **pre-Tanh neural decoder
  activations**, not unclipped externally rendered waveform amplitudes.
- **Model integrity:** Hash the original, EMA, and post-evaluation states.
  Any unexpected mutation during evaluation fails the experiment.
- **Output:** Strict JSON with metrics and digests, not WAV audio,
  weights, checkpoints, embedding arrays, or source recordings.

## Reproduce the bounded smoke

From repo root with CPU PyTorch research dependencies installed:

```bash
python -m research.velocity_ema_study \
  --seed 17 --steps 2 --codec-steps 1 \
  --diffusion-steps 8 --decay 0.9 \
  --sampling-seeds 301
```

For explicit warmup, add `--warmup`. The real experiment allows up to
64 denoiser updates, 16 codec updates and two sampling seeds. These are
research guardrails, not an indication that 64 steps are enough to
train a working audio model.

```bash
python -m pytest -q tests/test_weight_ema.py tests/test_velocity_ema_study.py
```

## What is **not** established

This does not establish that EMA improves perceptual audio quality,
speech intelligibility, music generation, sound-category classification,
English/Hindi semantic alignment, originality, copyright provenance,
convergence, broad data generalization, model release readiness, or a
preferred model checkpoint.

A negative `heldout_velocity_mse_ema_minus_online` means EMA predicts
the **specific held-out v target** more closely under one fixed corruption
draw. It is not a human-listening preference, and a short single-seed
pilot cannot establish statistical significance. Numerical clipping
reductions do not by themselves guarantee rich or usable sound.
Later studies should independently replicate model seeds, expand
rights-reviewed datasets, introduce listening tests and semantic sound
metrics, and measure inference hardware cost.

**No release, source tag, model identity, or weights publication follows
from passing this experiment.**
