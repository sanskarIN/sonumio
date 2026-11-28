# Median harmonic/percussive spectral decomposition

Sonumio has an opt-in, deterministic harmonic/percussive source separation
(HPSS) reference. The algorithm applies frequency- and time-axis median
filters to one local STFT and constructs complementary soft masks. This
helps reveal stationary tonal structure and transient energy for creative
processing and research.

It is **not** a trained musical-instrument separator, stem extraction
system or guarantee of independent, original instruments.

## Mathematical model

Given a complex STFT `X[c, f, t]`:

- The harmonic estimate is a local median of `|X|` along time.
  A sustained harmonic stays relatively stable at its frequency.
- The percussive estimate is a local median of `|X|` across
  neighboring frequencies. A broad transient spans several bins.
- The two nonnegative maps compete in a softened power mask.
  With `a=harmonic_estimate/margin_harmonic`,
  `b=percussive_estimate/margin_percussive`, exponent `p`:
  `harmonic_mask = a**p / (a**p + b**p)`.
  Its exact complement is `percussive_mask=1-harmonic_mask`.
- Both outputs retain the *original complex phase*, using
  Sonumio's bounded inverse STFT. The two masked spectra sum
  to the original spectrum to floating-point precision.

The masks assign equal weights to completely silent/ambiguous bins.
The sum of two reconstructed signals should approximate the original
input within the normal STFT/float32 reconstruction error.

The time-median width, frequency-median width, mask exponent and
two margin factors are explicit settings and finite bounded values.

## Python API

```python
from sonumio.audio_core.hpss import (
    HpssSettings, split_harmonic_percussive
)

settings = HpssSettings(
    n_fft=1024,
    hop_frames=256,
    harmonic_median_frames=17,
    percussive_median_bins=17,
    mask_power=2.,
    margin_harmonic=1.,
    margin_percussive=1.,
)
result = split_harmonic_percussive(audio, settings)
harmonic = result.harmonic
percussive = result.percussive
report = result.stats.as_dict()
```

Here `audio` must already be a canonical immutable `AudioBuffer`.
Both results retain exactly the input duration, layout and sample rate.

## Safety and reproducibility

A three-dimensional median tensor can allocate large temporary arrays.
Sonumio therefore checks FFT bins, channel count, analysis frame count,
median widths, estimated intermediate bytes and an independent median
operation budget **before** median filtering.

The algorithm never opens files or network URLs and never modifies
the supplied PCM. Source licensing remains with the caller.

The report includes relative component energies, peaks and direct
numerical recombination error. It explicitly marks the decomposition
as non-neural and makes no perceptual-quality or instrument-perfect
separation claim.

## Limitations

Harmonic instruments with sharp attacks may be partly percussive.
Percussive components with ringing resonances may be partly harmonic.
Short signals provide too little time context for reliable tonal
separation, and output often changes with median size. Listening
and evaluation against independently licensed reference sources
are necessary before claims about quality.
