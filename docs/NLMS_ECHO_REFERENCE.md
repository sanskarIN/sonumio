# Explicit-reference adaptive NLMS echo suppression

Sonumio includes an opt-in normalized least-mean-squares (NLMS)
adaptive FIR reference for **known, aligned** far-end audio and
multichannel microphone captures.

This is a research and testing facility, not a production-certified
acoustic echo canceller or a learned noise-removal model.

## Signal model

The caller supplies one source `reference` channel and one or more
microphone `microphone` channels on the **same sample clock** and
with exactly equal frame counts.

Each output channel has its own causal length-`T` adaptive filter.
At every sample:

`predicted[n] = dot(weights, recent_reference_samples)`

`error[n] = microphone[n] - predicted[n]`

When adaptation is explicitly permitted, the filter updates as

`weights <- (1-leakage)*weights +
step_size * error[n] * history / (epsilon + dot(history,history))`.

All arithmetic is local and deterministic. There is no implicit
resampling, unknown delay detection, voice identification, or
automatic double-talk classification.

## Python example

```python
import numpy as np
from sonumio.audio_core.nlms_echo import (
    NlmsSettings, suppress_reference_echo
)

settings = NlmsSettings(
    taps=128,
    step_size=.5,
    epsilon=1e-6,
    leakage=0.,
    maximum_weight_l2=100.,
)
# reference is a mono AudioBuffer; microphone has one or more channels.
# Freeze adaptation over manually reviewed near-end speech intervals.
mask = np.zeros(reference.frames, dtype=bool)
result = suppress_reference_echo(
    reference, microphone, settings,
    freeze_mask=mask, chunk_frames=8192,
)
residual = result.residual
predicted_echo = result.prediction
weights = result.weights
print(result.summary.as_dict())
```

A stateful `NlmsEchoStream` offers a fixed-clock push API for explicit
reference/microphone chunks and optional boolean freeze gates.
Its predictions and filter updates are deterministic regardless of
caller chunk boundaries.

## Limitations and safety

- The far-end reference must be aligned with the microphone signal.
  Propagation delays longer than the adaptive FIR cannot be recovered.
- Near-end speech can corrupt unconstrained adaptation. Freezing is
  explicit; no automatic detector is implemented.
- Stationary or insufficiently excited reference signals may leave
  filter coefficients ambiguous or poorly estimated.
- A reported residual RMS reduction is *not* calibrated echo-return
  loss enhancement or perceptual-quality proof.
- The operator rejects invalid topology, excessive amplitudes,
  floating-point overflow and too-large adaptive geometry.
- The result contains immutable PCM and detached filter coefficients.
- No microphone recording is downloaded or written in place; the
  caller owns any external file publication or privacy decisions.

The module can serve synthetic acoustic research and controlled
A/B experiments. A production voice-conferencing echo canceller
would require explicit delay estimation, robust double-talk handling,
nonlinear loudspeaker modeling, and latency/performance qualification.
