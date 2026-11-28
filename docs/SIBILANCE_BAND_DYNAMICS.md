# Dynamic high-frequency sibilance-band attenuation

Sonumio includes a non-neural, opt-in dynamic attenuator often called
a de-esser. This version detects acoustic energy in a declared
frequency band; it does not classify speech phonemes, transcribe
dialogue, or learn a vocal model.

## Signal flow

The existing bounded Hann STFT selects the frequency bins inside
the declared sibilance band. Weighted overlap-add reconstructs
the isolated band without changing the source sample rate,
channel count or original frame count.

A causal per-channel RMS-like detector, optionally linked across
all channels, applies attack/release smoothing. For signal level
above a declared threshold, a finite compression ratio proposes
attenuation in decibels. A separate maximum-reduction control
prevents unrestricted gain loss.

Only the attenuated portion of the isolated band is subtracted
from the original full-band signal:

`result = original - wet * band * (1 - gain)`

Other frequencies should remain approximately unchanged, subject
to ordinary finite-window spectral leakage. The dry position is
exactly bitwise identical to the original PCM.

## Python example

```python
from sonumio.audio_core.sibilance_dynamics import DeEsserSettings, deess

settings = DeEsserSettings(
    n_fft=1024,
    hop_frames=256,
    band_low_hz=3500.,
    band_high_hz=7800.,
    threshold_dbfs=-28.,
    ratio=4.,
    attack_seconds=.001,
    release_seconds=.05,
    maximum_reduction_db=18.,
    wet=1.,
    link_channels=True,
)
outcome = deess(audio, settings)
processed = outcome.audio
report = outcome.report.as_dict()
```

The `audio` argument is an immutable Sonumio `AudioBuffer`.
Band limits must fit below its Nyquist frequency. No file is
opened, network request performed, or model weights loaded.

## Validation, resources and limitations

Tests compare an original synthetic low-frequency sinusoid mixed
with a strong high-frequency burst. The high-frequency projection
must decrease while the lower frequency remains largely intact.
Other tests check dry bypass, independent and linked stereo,
attack/release behavior, gain floors, source immutability,
config validation and resource-budget rejection.

Every detector option has finite bounds. The implementation checks
input duration, selected sample rate, channel topology, and
intermediate CPU control and memory budgets.

An upper-frequency percussion hit or cymbal can trigger this
processor even without speech. Excessive reduction can create
dullness and unnatural transients. The report therefore explicitly
disclaims phoneme recognition, clipping prevention and calibrated
perceptual-quality improvement. Listening evaluation is necessary
before considering it ready for a particular voice or music mix.

## Checksum-verified WAV rendering

```sh
sonumio wav-deess --source spoken-input.wav --output cleaned.wav \
  --band-low-hz 3500 --band-high-hz 7800 --ratio 4
```

The command uses Sonumio's canonical WAV reader with optional
`--source-sha256`, applies the same bounded effect, then publishes a
complete WAV using an atomic no-clobber operation. Use
`--expected-output-sha256` to enforce byte-identical playback
reproducibility, and `--dither-seed` when adding explicitly seeded
integer-PCM dithering. The output format defaults to 16-bit PCM.

Source and destination file identities are checked before and after
processing; output may be overwritten only with `--overwrite`.
Failure returns a nonzero status without a misleading success JSON
receipt or truncated destination WAV.

The receipt records input file provenance, output SHA-256, PCM digest,
output bit-depth, effect settings and gain statistics. Input audio is
not automatically downloaded, uploaded, or modified.
