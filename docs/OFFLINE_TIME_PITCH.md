# Offline duration and pitch DSP

Sonumio provides two bounded, original NumPy reference algorithms for editing
its generated sounds and other local canonical integer PCM WAVs. They do not
train or invoke a model, synthesize a voice, preserve vocal formants, or
establish subjective audio quality.

## Algorithms

- WSOLA: overlapping fixed-length Hann-windowed grains, nominal integer
  analysis centers derived from the exact requested output length, and a
  bounded joint-channel normalized correlation search. Per-channel mean
  removal avoids a DC bias; anti-phase stereo remains matchable. The
  selected center is shared across all channels.
- Phase vocoder: centered Hann-window analysis and synthesis, actual
  integer analysis-hop measurement, expected bin phase increment
  subtraction, wrapped residual phase and one linked rotation per bin.
  The rotation preserves the current inter-channel spectral relationship.
  DC and Nyquist remain real. Squared-window overlap normalization covers
  the exact requested interval.
- Pitch shift: pitch-preserving stretch to an intermediate length followed
  by ratio-aware sinc conversion at an explicitly reported virtual integer
  sample clock. Duration is restored exactly, including a possible one
  frame boundary correction.

The time-scale identity path preserves PCM values without overlap-add.
Both transforms are offline and deliberately bounded. Longer clips, larger
searches, high rates, or unsupported virtual sample rates may be rejected.
No silent fallback to a different method occurs. Neither transient phase
locking nor psychoacoustic quality optimization is implemented yet.

## CLI

The mutually exclusive operation flags select a duration multiplier, exact
output frame count, or pitch change in semitones.

```bash
sonumio wav-transform original.wav --ratio 1.5 --output slower.wav --peak-ceiling .95
sonumio wav-transform original.wav --target-frames 48000 --method phase_vocoder --output fitted.wav --peak-ceiling .95
sonumio wav-transform original.wav --semitones -5 --output lower.wav --peak-ceiling .95
```

Duration factors range from 0.5 to 2; pitch shifts range from -12 to +12
semitones, subject to the valid virtual sample clock. Source bit depth is
retained by default. Optional output depth and explicit seeded dither
are supported. Output peak attenuation is never a loudness measurement.

The file adapter verifies the full canonical WAV source before synthesis.
It rejects aliasing source/destination paths, overrange output without an
explicit peak policy, unexpected digests and existing destinations unless
overwrite is explicitly enabled. Publication uses the existing verified
atomic WAV writer. No directories are created unless requested.

The JSON report includes input and output file SHA256, exact frame geometry,
peak attenuation, method-specific work diagnostics and algorithm identity.
It does not claim a listening test or model inference.

## Test strategy

Analytic tone frequency preservation, exact integer frame counts, identity
PCM, stereo polarity, deterministic repeated runs, zero-energy signals,
short and odd clip lengths, numerical bounds, direct correlation reference,
source integrity failures, alias rejection, deterministic WAV bytes across
reader packet partitions, and CLI argument validation are covered. Existing
audio-core CI exercises the tests on Linux and Windows with NumPy 1.x/2.x.
