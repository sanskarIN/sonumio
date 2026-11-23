# Sonumio: audited full-history embedding reuse during v-DDIM inference

**Numerical research improvement only. No generated model release, audio asset, checkpoint or inference-performance claim.**

## The redundant computation

An original Sonumio `GlobalHistoryVelocityDenoiser` receives the
preceding 8192-frame audio segment, text tokens, a diffusion timestep
and the current noisy future. The preceding audio is **identical for
every reverse-diffusion step**. Its `HistoryEncoder` is the same
multiscale learned 1D CNN during all these evaluations.

Before this experiment, the model recomputed the full history vector
on every one of its `N` denoiser calls. When classifier-free text
guidance is enabled, it made another **unconditioned text** call at
each timestep while still re-encoding the identical history. In that
case it performed **2N history-encoder evaluations**.

The correct optimization is not to skip the text, time, or future
residual model, all of which change each step. Instead the full audio
history representation can be prepared once and reused for both text
conditions and every selected noise timestep.

## What has been implemented

`research/global_history_denoiser.py` now accepts an **optional
detached validated history embedding** into the existing residual
conditioning route. With this optional argument absent, its original
forward path, gradients, model parameters and behavior are unchanged.

`research/cached_global_history.py` introduces:

- `prepare_history(model, audio)`: evaluate the original trained
  history CNN **one time** on exactly the prior 8192 samples.
- `PreparedHistory`: a short-lived, in-memory feature with
  recorded SHA256 digests for the exact source waveform, current model
  state, and feature bytes. It cannot be interpreted as durable
  serialization, a model file or a user-audio upload.
- `sample_cached_global_history_ddim(...)`: recompute current
  timesteps, text embeddings and future denoiser residuals normally,
  but reuse the validated feature. Supports 2–T deterministic
  velocity-DDIM inference timesteps, explicit unsigned 32-bit seeds,
  English/Hindi captions and bounded text guidance.
- `parity_check`: run the **same original neural network**, same
  start Gaussian, same history, same text, and same trained diffusion
  schedule on both uncached and cached paths. Demand **bitwise-equal
  output float32 samples and matching SHA256** rather than relying
  on a tolerance or judging the sound by listening.

All outputs stay in process memory, with no output clamp, scaling,
normalization or file write. The cache is not shared across neural
model versions, source waveforms or mutated model parameters.

### Explicit provenance and invalidation

A prepared history is bound to:

```text
history_sha256 = SHA256(float32 raw history bytes)
model_sha256   = SHA256(original full model state)
feature_sha256 = SHA256(original encoded history vector bytes)
```

Every reuse validates all three, the exact channel count and shape,
dtype, CPU device, finite values, no-autograd flag and supported
protocol. An optimizer step, different source clip, cache feature
mutation, different model initialization or invalid metadata causes
immediate rejection. The digest protects against accidental misuse;
it is **not** a signed public checkpoint or a cryptographic trust
attestation for remote attackers.

Network mode and global torch RNG state are restored. Cached
embeddings contain no raw audio array in the numeric report.

## What saves neural work

For N DDIM steps:

| Text guidance | Uncached history CNN calls | Cache created in call | Already prepared cache |
| --- | ---: | ---: | ---: |
| guidance = 1 | N | 1 | 0 |
| guidance ≠ 1 | 2N | 1 | 0 |

For an 8-step text-guided inference, the original path invokes the
history encoder 16 times, while the cached path can use one newly
prepared feature (or zero calls for reuse). The time/text residual
denoiser still runs 16 times.

This is an **operation count**, not proof of real wall-clock speedup.
Feature hashing, model-state verification, Python overhead and CPU
hardware vary. A future benchmark should compare actual latencies,
memory peak and median timings on identical machines and model
parameters. Avoid unverified 10× or faster-audio claims.

## Reproduce parity

```bash
python -m pytest -q tests/test_cached_global_history.py
```

```python
from research.cached_global_history import (
    prepare_history, sample_cached_global_history_ddim, parity_check
)

# model = an original global-history v-trained denoiser
# schedule = its EXACT trained diffusion schedule
# prior = CPU float32 8192 samples of preceding audio
prepared = prepare_history(model, prior)
sample, report = sample_cached_global_history_ddim(
    model, schedule, "बारिश", prior,
    prefix_frames=1024, evaluations=8,
    seed=41, guidance=1.5,
    prepared_history=prepared,
)
assert report["history_encoder_evaluations_for_this_call"] == 0
assert report["known_prefix_preserved_exactly"] is True
```

The test suite checks exact cached-vs-uncached waveforms over multiple
timesteps and guidance values, actual history-module forward-call
counts using a PyTorch hook, model/feature/audio mutation rejection,
valid source metadata, deterministic replay, mode/RNG integrity,
corrupt input prevention, and no audio/weights release.

## Why this does not create a model

A performance optimization to an experimental denoiser cannot make
it produce natural music or speech. The global history model still
needs substantial high-quality rights-cleared training, real
source-disjoint listening evaluation, more output sample rates, scene
continuity validation and inference resource profiling.

No Sonumio-1 generation, model weights, audio exports, network
service or new software release is created by this cache.
