# Regularized Wiener-style noise-profile suppression

Sonumio offers an opt-in CPU spectral suppressor based on an explicit
noise-only reference interval. It is distinct from the existing hard
spectral gate and is designed for careful comparison, not an automatic
enhancement or trained speech-cleaning model.

## Required noise reference

The caller supplies start/stop times for a source interval presumed to
contain **only stationary background noise**. Sonumio does not claim
to discover or verify that interval automatically. A contaminated
reference can suppress valid musical or speech components.

The reference STFT frames are summarized by a declared median, mean
or maximum statistic per frequency bin and channel, then squared to
obtain a nonnegative power profile. By default channels are profiled
independently; optional channel linkage selects the largest observed
noise power per bin across channels.

The profile is an immutable, detached array with explicit source
sample rate and FFT geometry, preventing accidental reuse with
different FFT sizes or channel layouts.

## Gain law

For each input power `P(c,k,t)` and estimated noise profile `N(c,k)`:

`G = sqrt( max(P - beta*N, 0) / max(P, epsilon) )`

The oversubtraction factor `beta` is bounded. The resulting amplitude
gain is smoothed with short, explicitly bounded odd-size moving
averages over time and frequency. A configurable minimum gain prevents
forced exact spectral zeros. Gains cannot exceed unity.

The gain multiplies the original complex spectrum, preserving its
phase in each bin. The shared inverse-STFT uses weighted overlap-add
to restore the input duration. Optional dry/wet control enables
an exact dry bypass, including for repeatable A/B evaluation.

## Python example

```python
from sonumio.audio_core.wiener_suppression import (
    WienerSettings, suppress_wiener
)

settings = WienerSettings(
    n_fft=1024,
    hop_frames=256,
    noise_start_seconds=0.,
    noise_stop_seconds=.1,
    noise_statistic="median",
    oversubtraction=1.5,
    minimum_gain=.05,
    smoothing_time=3,
    smoothing_frequency=3,
    channel_linked_profile=False,
    wet=1.,
)
result = suppress_wiener(audio, settings)
cleaned = result.audio
report = result.report.as_dict()
```

The source `audio` is a canonical immutable `AudioBuffer`, not a URL
or filename. No network access, model weights or in-place mutation
occurs. The processing preserves sample rate, channel count and
frame count.

## Accuracy and resource caveats

- A linear spectral power subtraction is a heuristic, not a perfect
  stationary noise estimator. Over-suppression can create musical
  noise or damage low-level harmonics.
- The reference region is a user-supplied assumption, not a
  machine-verified ground truth.
- The output is never hard-clipped or silently normalized; its peak
  and RMS are measured separately for diagnostics.
- The STFT engine bounds spectral storage, and additional mask
  scratch and smoothing work limits are checked.
- Any claimed improvement requires independent audio-listening
  evaluation and real noise-only datasets.

This operator complements, but does not replace, existing spectral
gating and source-identity verification facilities.
