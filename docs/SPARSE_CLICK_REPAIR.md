# Opt-in sparse waveform click restoration

Sonumio has an experimental CPU-only detector for **isolated sample outliers**.
It is intended for sparse waveform corruption, not continuous hiss,
de-reverberation, vocal separation, general noise reduction, or restoration
of unknown musical content.

## Detection

An odd rolling window determines a robust median `m[n]` and median absolute
deviation `MAD[n]` for every channel. A sample is a candidate when

`abs(x[n] - m[n]) > max(minimum_deviation, mad_multiplier * 1.4826 * MAD[n])`.

The detector considers only short runs that fit between unmodified
neighboring samples. It refuses a correction if candidate density exceeds
the configurable maximum fraction. These are *outliers*, not guaranteed
noise: a true percussive attack may meet the same criterion. Review proposed
events before using results for any important recording.

The default detector treats channels independently; an explicit linkage
flag duplicates candidate timestamps to every channel. No implicit
downmixing, resampling, gain normalization or model inference occurs.

## Interpolation

Each accepted run has an unchanged sample immediately before and after it.
A linear interpolation is the default. An optional monotone cubic Hermite
interpolant uses nearby anchor slopes, bounded to prevent overshoot outside
the two neighboring sample values.

Both methods preserve the original frame count, sample rate and channel
layout. Every unmasked sample is copied exactly. Neither approach can
reconstruct arbitrary missing harmonics. Boundary samples and long damaged
spans are deliberately not repaired.

## Python example

```python
from sonumio.audio_core.click_repair import (
    ClickRepairOptions, detect_clicks, repair_clicks
)

options = ClickRepairOptions(
    window_frames=11,
    mad_multiplier=10.,
    minimum_deviation=.04,
    max_run_frames=3,
    max_fraction=.01,
    interpolation="cubic",
    link_channels=False,
)
detection = detect_clicks(audio, options)
print(detection.count, detection.fraction)
# Inspect candidate locations and listen to an A/B comparison.
result = repair_clicks(audio, options, explicit_mask=detection.mask)
cleaned = result.audio
print(result.report.as_dict())
```

`audio` must be a canonical `AudioBuffer`. An explicit boolean mask
can be used after manual review to limit changes to known corruption.
The API validates its shape, sparsity and run lengths, then copies it into
immutable storage before processing.

The report contains per-channel start/stop frames, interpolation type,
the number of corrected samples and peak levels. This report does not
measure perceptual improvement. It makes no trained-model claim.

## Safety and resource bounds

The rolling-window median requires proportional workspace; preflight
limits bound it to 192 MiB. Every automatic or manual repair is subject to
a sparse-event threshold and maximum event count. Samples and masks are
never modified in place. Failure before publication cannot silently
overwrite the source recording.

Prefer a new output file for evaluated restoration. Keep the original,
review any repaired transient that sounds unusual and do not treat this
heuristic as a substitute for professional restoration or provenance.
