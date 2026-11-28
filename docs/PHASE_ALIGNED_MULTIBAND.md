# Phase-aligned FIR multiband dynamics

Sonumio's opt-in multiband reference module separates a signal into
complementary frequency bands, optionally applies independently configured
offline compressors, and recombines the results on the original sample clock.

## Crossover design

Given strictly increasing cutoff frequencies `f_1, ..., f_n`, design
windowed-sinc lowpass kernels `L_1, ..., L_n` of one odd length `T`.
The bank is

- `B_0 = L_1`
- `B_i = L_(i+1) - L_i` for intermediate bands
- `B_n = delta_(T//2) - L_n`

Therefore the sum of all kernels is **exactly a centered unit impulse to
floating-point roundoff**, even near the zero-padded boundaries. The
implementation removes the common group delay of `(T-1)//2` samples by
returning the centered `N` samples of the full convolution.

This is **offline zero-phase-aligned output**, not a causal zero-latency
streaming filter. The band filters are real-valued and linear phase. For
large filters or signals, explicit working-memory and FFT-work limits
reject the operation before allocation. No model weights are involved.

## Python API

```python
from sonumio.audio_core.multiband import (
    CrossoverPlan, split_bands, recombine_bands, multiband_compress
)
from sonumio.audio_core.dynamics import CompressorConfig

plan = CrossoverPlan(
    sample_rate=48000,
    cutoffs_hz=(160., 2600.),
    taps=257,
    window="blackman",
)
bands = split_bands(audio, plan)
reconstruction = recombine_bands(bands)
settings = (
    CompressorConfig(threshold_dbfs=-20., ratio=3.),
    CompressorConfig(threshold_dbfs=-18., ratio=2.),
    CompressorConfig(threshold_dbfs=-16., ratio=2.5),
)
outcome = multiband_compress(audio, plan, settings, wet=1.,
                             output_gain_db=0.)
processed = outcome.audio
report = outcome.report.as_dict()
```

Here `audio` is an existing canonical `AudioBuffer` on a 48 kHz sample
clock. The module does not fetch audio from external files.

## Declarative processing graph

An allowlisted `multiband_compress` operation accepts one input node and
a data-only configuration:

```json
{
  "id": "multiband",
  "operation": "multiband_compress",
  "inputs": ["source"],
  "parameters": {
    "cutoffs_hz": [160.0, 2600.0],
    "taps": 257,
    "window": "blackman",
    "thresholds_dbfs": [-20.0, -18.0, -16.0],
    "ratios": [3.0, 2.0, 2.5],
    "makeup_db": [0.0, 0.0, 0.0],
    "attack_seconds": 0.01,
    "release_seconds": 0.1,
    "knee_db": 6.0,
    "link_channels": true,
    "wet": 1.0,
    "output_gain_db": 0.0
  }
}
```

This node must appear inside a normal versioned `SignalGraph` with a
prior declared source node. The three parameter arrays must contain
exactly one setting per frequency band.

## Quality, accuracy and reproducibility

The regression suite compares complementary kernel reconstruction,
same-frame band recombination, sine-wave routing, declared gain, stereo
layout preservation, graph serialization, source immutability, and invalid
parameter rejection.

The compressor uses Sonumio's existing attack/release sample detector and
optionally links detector gain across channels. The report describes
numerical band RMS/peaks and explicit settings, **not** perceptual quality
or calibrated loudness. Output is never automatically hard-limited, clipped,
or normalized; review peaks before encoding to integer PCM.

Do not infer that the multiband treatment improves every source. A/B
listening, objective measurement, and actual production recordings are
needed to evaluate sound quality.
