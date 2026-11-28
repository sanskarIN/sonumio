# Log-sine excitation and bounded linear impulse identification

Sonumio provides a CPU-based research reference for generating an
original logarithmic sine sweep and estimating a **linear time-invariant**
impulse response from a recording of that sweep.

The reference is suitable for controlled laboratory experiments and
synthetic test systems. It cannot certify the acoustics of an uncontrolled
room, instrument, microphone, speaker, or playback/recording chain.

## Generation

`SweepPlan` declares a bounded sample clock, duration, start/end
frequency, amplitude and short raised-cosine edge fades. The phase
integral of a logarithmic frequency ramp is used to generate sample
values directly, avoiding accumulated phase-drift steps.

The generated `AudioBuffer` is single-channel, immutable, and
uses only original mathematical oscillator content.

## Identification

Given a declared excitation `x[n]` and recorded responses `y_c[n]`,
the algorithm estimates a per-capture-channel transfer response:

`H_c[k] = conj(X[k]) * Y_c[k] / (abs(X[k])**2 + lambda * max(abs(X)**2))`

A regularized inverse FFT produces a finite impulse estimate. The
regularization parameter is explicit. The implementation does **not**
automatically detect or compensate propagation latency: use the
`latency_frames` setting to extract the relevant delayed interval.
A fitted output residual and relative RMS error are reported separately
for every capture channel.

The resulting impulse uses the same `ImpulseMatrix` representation
as Sonumio's matrix and partitioned FIR engines, with one excitation
input and 1..8 capture output channels.

## Python example

```python
from sonumio.audio_core.sweep_identification import (
    SweepPlan, IdentificationSettings, render_log_sweep,
    identify_linear_response,
)
from sonumio.audio_core.matrix_convolution import convolve_matrix

plan = SweepPlan(
    sample_rate=24000,
    frames=24000,
    start_hz=50.,
    end_hz=9000.,
    amplitude=.2,
    fade_frames=128,
)
excitation = render_log_sweep(plan)

# Record the excitation through an independently controlled system,
# retaining the same sample clock. 'recording' is an AudioBuffer.
settings = IdentificationSettings(
    taps=2048,
    regularization=1e-10,
    latency_frames=0,
    block_frames=1024,
)
result = identify_linear_response(excitation, recording, settings)
estimated_ir = result.impulse
print(result.report.as_dict())
predicted = convolve_matrix(excitation, estimated_ir)
```

## Important assumptions

- Linear and time-invariant response throughout the measurement.
- Stable sample clock for excitation and capture; sample-rate
  conversion is not silently applied.
- Known propagation/digital latency supplied as integer frames.
- One excitation input; multiple capture outputs are independent.
- Recorded signal should contain sufficient broadband excitation
  energy and minimal unrelated noise.
- Without independent calibration, an amplitude response cannot be
  assigned physical sound-pressure units.

A sweep-based estimate may be biased by system nonlinearities, ambient
noise, automatic gain controls, recording alignment and unknown
clock drift. The supplied fit residual is a numerical sanity check,
not a calibrated uncertainty bound or psychoacoustic quality metric.

All input buffers are immutable, sample clocks are validated and
FFT/work memory is bounded before estimation. The method does not
modify source recordings, open external resources or call an AI model.
