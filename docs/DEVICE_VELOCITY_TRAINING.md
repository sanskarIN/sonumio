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


## Stratified timestep and paired-caption sampling

The default independent uniform sampler remains unchanged in its random
draw sequence. The opt-in stratified policy consumes complete random
permutations of the diffusion timesteps, carrying unused positions across
logical batches. Each completed cycle covers all timesteps exactly once.
It uses a separate CPU random stream, preserving waveform indices, noise
and condition dropout when only timestep sampling changes.

The opt-in bilingual pair policy draws adjacent recording pairs, with an
even logical batch. Every adjacent pair must have exactly identical
waveform samples and distinct non-null caption tokens. Both rows share
one sampled timestep, one Gaussian noise tensor and one dropout decision.
The policy checks recording adjacency but does not independently verify
the language or translation of arbitrary caption tokens. The synthetic
study uses known English/Hindi pairs. Pairs can span microbatch boundaries;
each row retains its correct loss normalization.

```bash
python -m research.device_velocity_study --steps 16 --batch-size 4 --microbatch-size 3 --alternative-microbatch 1 --timestep-sampling stratified --caption-sampling bilingual_pair
```

With eight diffusion timesteps, this example covers 32 pair draws,
or exactly eight caption rows at every timestep. The report records
timestep count spread, distinct caption rows observed and minimum/maximum
row exposure. Pair sampling changes the number of independent recordings
per logical batch; it is not equivalent to independently sampled rows.
No training efficiency or semantic grounding improvement is asserted.


## Training-only semantic supervision

The optional auxiliary head predicts multi-label source membership directly
from the shared UTF-8 text encoder. The six categories and their one- or
two-source membership targets are derived from the exact canonical synthetic
caption table, with strict token order checks. No source label or class
lookup is added to denoiser inference. The head is initialized with its own
CPU seed and discarded at the end of the experiment.

The loss is mean binary cross-entropy across the six source coordinates,
summed over retained caption rows and divided by the entire logical batch.
Classifier-free dropped captions contribute zero auxiliary loss. The shared
encoder, denoiser and auxiliary head are jointly optimized and gradient
clipped, including when a bilingual pair crosses a microbatch boundary.

```bash
python -m research.device_velocity_study --steps 48 --batch-size 4 --microbatch-size 2 --alternative-microbatch 1 --timestep-sampling stratified --caption-sampling bilingual_pair --compare-semantics
```

The comparison uses a fixed auxiliary weight of 0.5 and an otherwise
identical baseline. An explicit `--semantic-weight` is available for
separate runs. Initial and final classification BCE, exact source-set
matches, bit accuracy and micro-F1 are reported for training and held-out
caption rows. The latter may share caption templates with training even
when waveform recordings are seed-disjoint; it is not a novel-language
generalization test. A reduction in text classification loss alone is not
evidence of audio grounding. Evaluate matched, rotated-label and null
generated waveform controls independently. The source probe does not
provide a training gradient. No pretrained weights or audio assets are
downloaded, and neither the auxiliary head nor the denoiser is published.


## Multi-negative caption margin

The optional margin objective extends the earlier single-negative
matched-only intervention to up to four distinct wrong captions. Each
negative preserves the recording-seed template, language and source
cardinality, while naming disjoint sources. A deterministic rotating
selection requires no new random draws. All negatives are evaluated on
the same corrupted recording and timestep as the matched caption.

For retained caption rows the smallest wrong-caption velocity MSE is
detached from the gradient graph. The auxiliary loss is
max(0, matched_MSE - (1 - margin) * minimum_wrong_MSE).
It is summed over retained rows and divided by the entire logical batch.
Dropped captions contribute zero. This does not directly reward worsening
wrong predictions, although shared parameter changes can affect them.

```bash
python -m research.device_velocity_study --steps 48 --batch-size 4 --microbatch-size 2 --alternative-microbatch 1 --caption-sampling bilingual_pair --timestep-sampling stratified --compare-margins
```

The baseline and treatment share initialization and stochastic inputs.
The fixed treatment weight is 0.5, with two disjoint negatives and a
relative margin of 0.05 by default. Independent runs can set
`--margin-weight`, `--relative-margin` and `--negative-count`.

Every arm now reports held-out matched, wrong and null caption errors
on identical corruptions, stratified by language and source count.
Generated single-source panels also enumerate all 720 balanced label
assignments separately for each language, with accepted predictions and
forced nearest-source rankings reported separately. Repeated labels and
abstention are preserved in the null distribution. These narrow synthetic
diagnostics do not establish general language or audio quality.
