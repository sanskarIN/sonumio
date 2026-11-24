# Verified streaming spectral energy inspection

This independent developer tool measures decoded PCM spectral energy on a
fixed original-source block grid. It does not modify the WAV, access the
network, load model weights, or evaluate perceptual or semantic audio quality.

## Command line

From an installed Sonumio source checkout:

```bash
python -m sonumio.audio_core.spectrum_cli input.wav --fft-frames 1024
python -m sonumio.audio_core.spectrum_cli input.wav --fft-frames 2048 --include-bins
```

The optional `--expected-sha256` accepts a trusted lowercase whole-file
digest. `--packet-frames` selects the bounded PCM transport size, not the FFT
resolution. Invalid input exits with status 2 and no JSON on stdout. A
successful inspection emits one JSON object after source verification.

## Python

```python
from sonumio.audio_core.stream_spectrum import (
    SpectrumConfig, inspect_wav_spectrum,
)

report = inspect_wav_spectrum("input.wav", config=SpectrumConfig(2048))
print(report.spectrum.channel_spectra[0].dominant_frequency_hz)
```

## Numerical protocol

* FFT length N is a power of two from 64 to 8192; bin centers are k*sample_rate/N.
* Non-overlapping rectangular blocks begin at frame zero. An incomplete last
  block is zero-padded to N; padded_tail_frames counts original frames in
  that partial block, not synthetic zeros.
* One-sided energy contributions for each block are squared FFT magnitudes
  divided by N, doubled for interior bins only. DC and Nyquist are not doubled.
  The sum across blocks is divided by the original number of source frames.
* The sum of per-bin mean-square contributions equals decoded time-domain
  mean square (Parseval), including a partial tail. This is not a density
  expressed per Hz. The optional bin list exposes each contribution.
* Dominant bin includes DC and Nyquist. Exact computed ties choose the lower
  bin; nearly equal peaks can change order with floating-point reductions.
  The centroid is energy-weighted over all one-sided bins. Silent channels
  have null dominant frequency, centroid and DC/Nyquist fractions.
* DC bin energy includes blockwise means; it is not solely whole-file DC
  offset. A rectangular window exhibits spectral leakage on non-bin-centered
  tones and on padded tails. Short or nonstationary sounds need careful
  interpretation; the report is not a spectrogram or pitch tracker.

The accumulator retains at most one incomplete FFT block plus one bounded
energy vector per channel. FFT work is batched with at most 65,536 original
frames per batch. Incoming validated packets and optional per-bin JSON
introduce additional bounded memory. The underlying canonical reader accepts
8–192 kHz, 1–8 channels, integer 8/16/24/32-bit PCM, up to one hour subject to
the file-byte cap. SHA256 verification includes the original RIFF bytes.

## Validation

The regression suite compares results to a separate direct DFT matrix and
time-domain mean-square reference across fragmented, randomly partitioned and
large packets. It includes exact-bin tones, DC, Nyquist, silence, partial
tails, rejected stream geometry, digest mismatches and malformed source files.
The tool deliberately does not infer a source category, quality score,
copyright status, phase response or model readiness.
