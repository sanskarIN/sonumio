# Device-aware velocity training

This research runner trains the existing raw, text-conditioned waveform
velocity denoiser on Sonumio's synthetic audio. It is not a new model
generation or a production training service. The data, optimizer state,
and trained weights remain in memory.

## Training design

A logical batch contains up to 32 caption rows. The runner constructs its
indices, timesteps, conditioning-dropout mask, and Gaussian noise using
an isolated CPU generator. It hashes these complete inputs before splitting
the batch into microbatches of up to eight rows. Changing the microbatch
partition or requested device does not change that random input plan.

Each microbatch's mean velocity loss is weighted by its actual number of
rows divided by the logical batch size. Gradients accumulate across all
microbatches, including a smaller final remainder. Gradient clipping and
the AdamW update occur once per logical batch, rather than once per
microbatch. Gradients are cleared if a partial update fails. Nonfinite
losses and gradient norms are rejected.

The runner uses float32 tensors and accepts an explicit CPU or CUDA device.
Unavailable CUDA requests raise an error instead of silently falling back.
The model is returned to CPU for the existing research DDIM sampler.
No mixed precision, multi-GPU, distributed execution, GPU memory sizing,
or bitwise cross-device equivalence is claimed.

## Run the controlled experiment

Install the project's research dependencies and a compatible PyTorch build.
The default experiment compares two CPU microbatch partitions with identical
initial model parameters and random training inputs:

```bash
python -m research.device_velocity_study --device cpu --steps 24 --batch-size 5 --microbatch-size 5 --alternative-microbatch 2 --seed 17
```

On a machine with a working CUDA-enabled PyTorch installation, the following
adds a third training arm using the specified CUDA device:

```bash
python -m research.device_velocity_study --device cuda:0 --steps 24 --batch-size 5 --microbatch-size 5 --alternative-microbatch 2 --seed 17
```

The report records the actual training device, logical and microbatch sizes,
optimizer update count, loss, gradient norm, timestep histogram, model-state
fingerprints, and CPU input-plan digest. All arms are evaluated on CPU
using held-out synthetic recordings and identical corruption noise. Generated
audio diagnostics are computed only when the independent open-set source
probe passes its own reference gate. Matched, rotated-label, and unconditional
generation are reported without writing audio files.

The study compares trained model parameters numerically. Floating-point
accumulation order and backend algorithms can produce differences even with
the same input trace. The CPU CI workflow verifies two independently seeded
microbatch experiments; the CUDA smoke test skips when hardware is unavailable.
CUDA correctness and performance require a separate run on real CUDA hardware.
Training timing includes device transfers and model fingerprinting, so it is
not a GPU kernel benchmark.

## Limitations

This is a bounded synthetic-only training experiment at 8 kHz, using
1.024-second waveform clips and the existing 16-channel research architecture.
Successful optimization does not establish perceptual quality, general
prompt grounding, long-duration generation, or Sonumio-1 readiness.


## Timestep weighting ablation

The existing default uniform velocity objective is preserved. Two opt-in
policies keep the same stochastic sampling plan, model initialization and
logical-batch normalization:

- Min-SNR velocity weighting: min(alpha_bar, gamma * (1 - alpha_bar)).
  This is algebraically equivalent to min(SNR, gamma)/(SNR + 1)
  and avoids division by very small terminal signal powers.
- Clean-space MSE weighting: 1 - alpha_bar. For identical noisy input,
  clean-waveform MSE equals velocity-prediction MSE times this factor.

The first policy follows Hang et al., [Efficient Diffusion Training via
Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556), ICCV 2023.
The second is a separate algebraic diagnostic. Neither policy is assumed
to improve audio quality. In particular Min-SNR strongly downweights the
near-zero terminal SNR, while clean-space weighting emphasizes it.

Use the controlled comparison:

```bash
python -m research.device_velocity_study --device cpu --steps 32 --batch-size 5 --microbatch-size 2 --alternative-microbatch 1 --compare-objectives
```

The four arms share random indices, timesteps, condition dropout, Gaussian
noise and initial model parameters. Weighted training objectives are
reported separately from raw velocity MSE; a lower weighted objective is
not evidence that one policy is better than another. Compare unweighted
held-out waveform errors and matched versus rotated/null generation.
The weighting formula and uneven microbatch gradient equivalence have
dedicated regression tests. The experiment makes no claim of successful
prompt grounding or hardware acceleration.
