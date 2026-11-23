# Sonumio: global-history audio-conditioned velocity model

**Independent new research architecture; not Sonumio-1, not a production model.**

## Motivation

The first Sonumio prefix-anchored waveform denoiser is a local 1D CNN.
It can see a known clean prefix but has a limited effective temporal
receptive field. A distant generated sample may be unable to use useful
history from the previous one-second audio window. Merely matching a
prefix sample-for-sample does not imply coherent continued sound.

`research/global_history_denoiser.py` introduces a separate
`GlobalHistoryVelocityDenoiser` that receives:

- **Noisy future**: `[B,1,8192]`, float32, 8-kHz mono.
- **Clean full history**: `[B,1,8192]`, from the immediately previous clip.
- **Known overlap**: 128–4096 frames, with the future input's first
  `P` frames equal to the full history's final `P` frames.
- **UTF-8 text tokens**: `[B,130]`, English/Hindi handled by the
  existing byte tokenizer and original learned text encoder.
- **Diffusion timestep**: `[B]`, integer.

The history encoder uses learned multiscale strided 1D convolutions
over the **full preceding window**, global pooling, and an explicit
16-bin energy envelope plus mean/RMS/peak summary. It yields an
independent context vector fused with time and text embeddings. That
fused vector conditions each existing-style residual block, so
information from the earliest part of the previous window can
influence predictions even at the far end of the future clip.

This is an explicit architectural possibility, **not proof** that a
model learns useful history conditioning or human-perceived coherence.

## Training objective

For true adjacent 8192-frame previous/future windows:

```text
t      ~ Uniform(discrete diffusion timesteps)
epsilon ~ N(0,I) with explicit CPU torch.Generator
x_t    = sqrt(alpha_bar_t)*clean_future + sqrt(1-alpha_bar_t)*epsilon
x_t[:P] = previous[-P:]                  # known clean overlap
v_t    = sqrt(alpha_bar_t)*epsilon - sqrt(1-alpha_bar_t)*clean_future
loss   = MSE(model(x_t, t, text, previous)[P:], v_t[P:])
```

The future-only velocity target excludes known samples. The full
history encoder receives gradients as well as the text encoder and
residual denoiser. Source input is validated before forward passes;
false adjacency, nonfinite samples, bad types, wrong shapes and
unsupported model dimensions fail closed.

This is a **new checkpoint/model protocol**. Existing epsilon-trained
models or older v-trained models cannot be silently used with it.
No automatic checkpoint migration is defined.

## Sampling and continuation

`sample_global_history_ddim` begins with a reproducible local
Gaussian generator, anchors the first `P` samples to the exact
preceding clip tail at every timestep, and predicts the remaining
unknown samples with original deterministic velocity-DDIM algebra.
Optional classifier-free text guidance keeps the SAME audio history
for both the conditional and unconditional text predictions.

The previous clip's encoder signal is provided on every reverse step.
The final clean overlap is preserved exactly, and the returned audio
is **not** normalized/clipped. The sample is returned only in memory,
with peak, clipping fraction, first unknown-sample jump, parameter-state
SHA256, history/sample hashes, inference counts and RNG audit flags.

`extend_global_history_sequence` accepts a caller-provided first
clip (never invents a lead clip from a continuation-only model), plus
one shared caption or one caption per **future** clip. It iteratively
passes the last full 8192 frames of the assembled recording as the next
history input. Up to 8 clips are supported.

```text
frames = 8192 + (segments - 1)*(8192 - prefix_frames)
```

Sample-level matching of the known prefix is a hard invariant, not an
audio quality metric. It is possible for the first newly generated
sample to be discontinuous or for the sound to drift in semantics.

## Developer smoke

```python
import torch
from research.global_history_denoiser import (
    GlobalHistoryVelocityDenoiser, masked_history_velocity_loss,
    sample_global_history_ddim,
)
from research.text_audio import ByteTextTokenizer
from research.wave_diffusion import DiffusionSchedule

network = GlobalHistoryVelocityDenoiser(16)
schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
history = torch.zeros(1, 1, 8192)
future = torch.zeros(1, 1, 8192)
tokens = ByteTextTokenizer().encode(["rain"])
loss = masked_history_velocity_loss(
    network, schedule, future, tokens, history,
    prefix_frames=512,
    generator=torch.Generator(device="cpu").manual_seed(17),
)
loss.backward()
```

The example only validates shapes and gradients. **The model must
actually be trained** using correctly paired adjacent recordings before
its output can be interpreted as generated continuation.

For a trained original model after validation, users can call
`sample_global_history_ddim` with matching history and P. Do not
publish the model or its weights without separate authorization.

```bash
python -m pytest -q tests/test_global_history_denoiser.py
```

The dedicated GitHub CPU workflow executes the full test suite and
one real forward/backpropagation/sampling smoke. Tests require:
finite and correct neural gradients, remote-history sensitivity
without changing the local overlap, reproducible seeded sampling,
sample-prefix identity, immutable original model weights, global
RNG isolation, English/Hindi tokenization and malformed-data
rejection. They do **not** certify perceptual sound quality.

## Next scientific questions

The new encoder needs matched adjacent-history training, independent
seed replication and held-out recording-group analysis. Evaluate its
future-only MSE against a fair same-parameter-count zero-history
baseline, test wrong-history perturbations, and independently assess
first-unknown-sample jumps, long-window coherence and listening
quality. More realistic audio rates, rights-cleared datasets and
multi-second contextual windows are future work.

No model release, model weights, external sound data, public API
endpoint, speech/music model or original Sonumio-1 generation
is established by the existence of this architecture.
