# Experimental velocity-DDIM step-budget study

**Status: development research, not Sonumio-1, not a release or weights approval.**

## Why this exists

Sonumio's short raw and latent DDPM research has already shown that
**velocity prediction** can reduce numerical saturation under the cosine
schedule. It has **not** established realistic text-conditioned audio,
English/Hindi semantic alignment, human listening quality, or production
readiness. This study separates the **sampling algorithm** from training
and makes the compute/trajectory tradeoff measurable.

Unlike stochastic DDPM inference, this candidate follows deterministic
DDIM with `eta=0` and a fixed, strictly descending subset of the
**already-trained** diffusion timesteps. It NEVER reinterprets an
epsilon-trained checkpoint as a v-model and does not alter the existing
`sample_velocity` implementation.

## Exact v-DDIM equations

Let `alpha_bar_t` be the cumulative noise schedule, `x_t` the noisy
state and `v_theta(x_t,t,c)` the learned velocity. Then:

```text
x0_hat = sqrt(alpha_bar_t) * x_t - sqrt(1-alpha_bar_t) * v_theta
eps_hat = sqrt(1-alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v_theta

for previous trained timestep s < t:
  x_s = sqrt(alpha_bar_s) * x0_hat + sqrt(1-alpha_bar_s) * eps_hat

for terminal step s = -1:
  output = x0_hat
```

No random noise is added during a DDIM step. The only stochastic input is
the initial locally generated Gaussian, supplied by an isolated CPU
`torch.Generator` with an explicit unsigned 32-bit seed. The first
selected timestep is always `T-1`, the last always `0`. Selected
timesteps are integer-respaced without modifying training alphas.
Full-budget DDIM therefore means **DDIM with T evaluations**, not the
original stochastic v-DDPM path.

With classifier-free guidance factor 1, each step needs **one** denoiser
evaluation. Other values within [0,4] use conditional and unconditional
passes, for **two** denoiser evaluations per step. Compute savings are
expressed as model-evaluation counts; runtime, model quality and memory
are separate measurements.

## Implemented experiments

- `research/velocity_ddim.py`: a standalone sampler for **both**
  original experimental architectures: raw 1×8192 samples and latent
  16×128 followed by the frozen local neural decoder.
- `research/velocity_ddim_study.py`: trains each v-denoiser on **only
  first-party Sonumio procedural SFX**, while training the tiny codec
  separately. The optimizer does not access held-out waveforms.
- Original bilingual synthetic pairs: **42 distinct training recordings
  and 42 held-out recordings**, with **84 English/Hindi captions per split**.
  Their recording-group hashes and train/validation fingerprints are
  checked for exact overlap.
- All held-out caption examples contribute to the reported v-objective
  MSE. Duplicated English/Hindi waveforms do **not** count as independent
  source recordings; cross-domain raw/latent v-MSE must not be directly
  ranked.
- Each budget uses the **same learned model weights** and **same initial
  Gaussian for a given sampling seed**. Output is compared with the
  corresponding full-budget **deterministic DDIM** output, NOT a human
  audio reference.
- Measures full-step numerical MAE, waveform cosine, post-clamp RMS and
  clipping, pre-clamp magnitude, and decoder **pre-Tanh activations** for
  latent inference. Records rain-vs-wind waveform differences in both
  English and Hindi as a *prompt-change control*, not semantic scoring.
- Produces strictly bounded **numeric JSON only**; neither audio WAVs
  nor any neural weights/checkpoints or third-party recordings are saved.

## Running locally

Install optional research dependencies from the repository (PyTorch CPU is
sufficient). This command uses a deliberately small, non-converged pilot:

```bash
python -m research.velocity_ddim_study \
  --seed 17 \
  --train-steps 1 \
  --codec-steps 1 \
  --schedule-steps 8 \
  --budgets 2,4,8 \
  --sample-seeds 201,202
```

The command writes its numerical report to stdout; redirect it to a local
JSON file only if desired. No `--model-path`, `--output-wav`, or external
dataset flag is part of this experiment. This is **not a demonstration of
trained model quality**; a one-step smoke deliberately cannot converge.

A separate multi-initialization control repeats the paired study across
independent model-initialization seeds while enforcing identical data
fingerprints, sample counts, sampler budgets, and source-group coverage.
This provides descriptive observed ranges, **not significance or a
confidence interval**:

```bash
python -m research.velocity_ddim_replicates \
  --seeds 17,29,41 --train-steps 8 --codec-steps 4 \
  --schedule-steps 8 --budgets 2,4,8 --sample-seeds 201,202
```

For tests:

```bash
python -m pytest -q tests/test_velocity_ddim.py tests/test_velocity_ddim_study.py tests/test_velocity_ddim_replicates.py
```

## Verification and interpretation

The test suite uses an *ideal known x0/noise oracle*: every selected
velocity transition must recover the exact corresponding forward-noised
tensor within finite float32 tolerance, even with timestep skipping.
Further checks cover invalid ordering/budgets, RNG isolation, repeatable
waveforms, decoder activation provenance, and no output artifacts.

An actual measurement is only interpretable with its configuration
hash, source fingerprints, model-initialization seed, sampled noise seeds,
trained prediction objective, schedule profile, and selected timesteps.
Smaller full-DDIM trajectory differences are **not** necessarily better
sounding. Reduced model evaluations imply lower nominal denoiser call
count, not automatically wall-time gains on every CPU/GPU.

Future research requires substantially longer independent training,
authentic rights-reviewed non-voice recordings, semantic classifier/
embedding and human-blind listening protocols, memorization controls,
and hardware measurements. No model weight or release publication is
authorized by these experiments.
