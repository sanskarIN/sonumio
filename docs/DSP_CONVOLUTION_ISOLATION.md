# Isolating original NumPy convolution differences

**Experimental first-party DSP only; not a new audio model or data recipe.**

The [stage-trace experiment](DSP_PIPELINE_TRACE_RESULTS.md) reproduced
a cross-CPU exact-byte divergence specifically inside Sonumio's
existing rain, thunder and wind sound layers on an AMD EPYC 7763.
It did not affect the time grid, the FIR resampling kernel or
the original forest, water and ambient layers.

Those three differing source families share relatively long
**float32 NumPy convolutions**, while water uses a much shorter
convolution. This makes source-convolution arithmetic a testable
hypothesis; it does not yet prove the cause.

## Precise original operations

The new [convolution isolation probe](../research/dsp_convolution_probe.py)
reconstructs only original operations already used in
\`src/sonumio/engines/procedural.py\`, without replacing them
with a modified convolution algorithm or changing training data.

| Source | DSP convolution input | Kernel length | NumPy mode |
| --- | --- | ---: | --- |
| Rain | Seeded float32 impulse train | 180 | full, cropped to source length |
| Thunder | Seeded float32 Gaussian noise | 140 | same |
| Wind | Seeded float32 Gaussian noise | 60 | same |
| Water/ocean control | Seeded float32 Gaussian noise | 12 | same |

For **each source × four original recording seeds** (0, 1, 101, 102),
the probe captures native-dtype exact SHA256 and separate coarse
quantized hashes of:

- The actual seeded Gaussian noise or rain impulse-array input.
- The exact float32 original convolution kernel.
- The NumPy convolution output before later normalization/envelopes.
- The complete original source-layer output.
- Eight selected scalar convolution references calculated with
  Python double products and \`math.fsum\` rather than NumPy's
  vectorized convolution reduction.
- For rain, the separate seeded hiss layer, because that sound
  uses an additional random noise draw before convolution.

An independent invocation of the unmodified original source
function must agree **byte-for-byte** with the reconstructed
source layer. The scalar reference is a *numerical diagnostic*;
it is checked with a stated finite absolute tolerance, not
required to match floating-point SIMD arithmetic bit-for-bit.

The [cross-host CPU matrix](../.github/workflows/dsp-trace-matrix.yml)
runs two independent processes on each hosted CPU and records the
per-source input/kernel/convolution/layer SHA256s. Logs contain
only SHA/CPU/selected-error metrics, not waveform samples.

## How to interpret a matched or mismatched stage

- **Input differs:** investigate seed/RNG generation, source
  construction and NumPy random implementation before blaming
  convolution.
- **Inputs agree but kernel differs:** investigate float32 kernel
  construction, including exponential for rain.
- **Input and kernel agree but convolution differs:** evidence
  narrows the difference to NumPy convolution arithmetic or
  the backend/compiler/runtime implementing that convolution.
  It still does **not** identify a specific processor instruction.
- **Convolution matches but final source differs:** investigate
  subsequent source normalization, oscillator, envelope, mixing
  and dtype conversion.
- **All stages match:** that host pair does not reproduce the
  historical numerical divergence.

The water/ocean 12-tap control is especially informative; a
different 60-/140-/180-tap path with matching inputs and kernels
would support a convolution-length-specific hypothesis.

## Reproducible research commands

~~~bash
python -m research.dsp_convolution_probe \
  --output checkpoints/private-research/dsp-conv-hashes.json
python -m research.dsp_convolution_probe --summary-only
python -m pytest -q tests/test_dsp_convolution_probe.py
~~~

This numerical experiment does not certify realism, prompt
semantics, audio fidelity, license compliance of future datasets,
or production readiness. Keep velocity-prediction experiments,
the original corpus and the private draft PR intact. **Do not
publish WAVs, neural weights, tags, prereleases or stable releases.**
