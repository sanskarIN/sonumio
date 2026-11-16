# Paired velocity (\`v\`) versus epsilon DDPM experiment

**Sonumio research only; no Sonumio-1 model, weights, WAV or release.**

The [three-seed diffusion schedule experiment](SCHEDULE_ABLATION_RESULTS.md)
proved a negative result: under the same short 8-step training budget, the
cosine-plus-epsilon prediction sampler produced **94.62% clipping**
in the raw branch and **63.86% clipping** in the latent branch. Those
outputs must not be presented as acceptable creator audio.

This follow-up changes the **prediction objective and corresponding
reverse posterior**, rather than only changing a diffusion schedule.

## Mathematical protocol

Let clean audio or codec latent be x₀, Gaussian noise be ε, and
cumulative signal power at a sampled timestep be ᾱₜ.

- Forward noising: xₜ = √ᾱₜ · x₀ + √(1−ᾱₜ) · ε.
- Previous objective: predict ε.
- New velocity target: **vₜ = √ᾱₜ · ε − √(1−ᾱₜ) · x₀**.
- Recover x₀: **x̂₀ = √ᾱₜ · xₜ − √(1−ᾱₜ) · v̂ₜ**.
- Recover noise: **ε̂ = √(1−ᾱₜ) · xₜ + √ᾱₜ · v̂ₜ**.

The velocity sampler computes its reverse DDPM posterior mean from
x̂₀ and xₜ with the analytically correct posterior coefficients:
βₜ√ᾱₜ₋₁/(1−ᾱₜ) and √αₜ(1−ᾱₜ₋₁)/(1−ᾱₜ).
The stochastic posterior variance is βₜ(1−ᾱₜ₋₁)/(1−ᾱₜ).
At t=0, the returned sample is x̂₀ with zero added stochastic variance.

Unlike the earlier near-terminal epsilon sampler, this posterior
does **not** divide a network prediction error by √αₜ when the last
cosine beta is 0.999. This improves the theoretical conditioning of
the update but **does not prove** audible quality, realism or convergence.

The older epsilon models and checkpoints **remain epsilon-trained**.
Mixing an epsilon-trained checkpoint with the v reverse equations
is forbidden; these are distinct model objectives.

## Strict paired training controls

The [runner](../research/velocity_ablation.py) initializes raw and
latent denoisers once per seed, then clones their initial weights so
epsilon and velocity networks start **bit-identically**.

For every training step within a model arm, the two objectives share:

- Exactly the same minibatch indices, 1.024-second Sonumio DSP clips,
  English/Hindi captions and 8-kHz representation.
- Exactly the same selected timestep and **same Gaussian noise tensor**,
  hence the same corrupt waveform or 16×128 codec latents.
- The same classifier-free conditioning dropout mask.
- Identical architecture, width, parameter counts, AdamW settings,
  clipping thresholds and optimizer-update counts.
- A *shared pretrained convolutional codec* within the latent arm
  (trained once per seed, then frozen for both objectives).

The only deliberate intervention is the objective and corresponding
inference equations. The raw and latent model arms **do not** have
matched compute budgets or directly comparable loss units.

The study uses three initialization seeds (17, 29, 41), 42 unique
synthetic training recordings and 84 bilingual caption pairs,
42 disjoint synthetic validation recordings, 128 codec optimizer
steps and **256 denoiser steps for each objective of each model arm**,
eight cosine diffusion steps, and two fixed English/Hindi
categories: rain/बारिश and wind/हवा.

## Actual generated-audio audit before and after clipping

The [generation code](../research/velocity_prediction.py) returns
**unclipped** 1.024-second raw waveforms or decoded latent waveforms
before any volume limiter. The comparator computes:

- Preclip peak amplitude and root mean square.
- Fraction of samples strictly outside [-1, 1] **before** limiting.
- Postclip peak, RMS, hard-clipping fraction and near-silence fraction.
- Postclip spectral centroid.
- Same-noise-seed rain-versus-wind change for English and Hindi.
- Observed per-seed metric differences (v − epsilon) with
  mean, median, standard deviation, range and sign counts.

All waveform buffers and neural weights are discarded after
the calculation. The only persisted file is private, ephemeral,
numeric JSON under ignored \`checkpoints/\`.

The code rejects nonfinite, improperly typed or absurd-amplitude
unclipped waveforms rather than pretending they're good sound.
No actual audio sample, user data or model artifact is uploaded.

## Run locally

~~~bash
python -m pip install -e '.[research,test]'
python -m research.velocity_ablation \
  --seeds 17,29,41 \
  --codec-steps 128 --denoiser-steps 256 \
  --schedule-steps 8 --channels 16 \
  --output checkpoints/private-research/velocity-objective.json

python -m pytest -q tests/test_velocity_prediction.py
~~~

In PowerShell enter each command on one line or use PowerShell's
backtick line continuation instead of Bash's backslash.

The development PR's GitHub Actions CPU training workflow executes
this fixed configuration. It verifies 3 completed seeds, the
paired model/conditioning controls, finite measured audio statistics,
and no saved WAVs or weight files.

## Interpret results cautiously

A reduction in clipped samples, even a large one, is evidence
of **numerical waveform health only**. It is not a listening result
or proof that the output sounds like rain or wind. A larger
rain-versus-wind waveform difference is similarly NOT a
semantic/class recognition score.

The model still sees only simple first-party DSP synthetic sounds
and templated short captions. Real licensed source recordings,
higher sample rate, longer sound clips, independent listening,
perceptual feature metrics, memorization and copyright review,
and robust English/Hindi prompt generalization are outstanding.

**No release, pretrained weights, pre-release tag, stable release,
or merge of draft PR #1 is authorized or performed by this study.**
