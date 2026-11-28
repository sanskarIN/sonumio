# GCC-PHAT signed sample-clock alignment reference

Sonumio's optional GCC-PHAT module estimates a **relative time delay**
between two independently supplied, already-loaded PCM buffers.
It is intended for microphone synchronization research, acoustic impulse
analysis and testing aligned echo-reference captures.

A positive lag indicates the observed signal arrives later than the
reference. A negative lag means it arrives earlier.

## Estimation

The algorithm chooses explicitly declared channels from the reference
and observed audio. The two signals must share a sample rate; their
durations may differ, but both need enough frames to evaluate the
requested lag bound.

After optional DC subtraction and Hann taper, calculate Fourier
spectra `X` and `Y`. Generalized cross-correlation with phase
transform applies the normalized cross spectrum:

`cross = (Y * conj(X)) / max(abs(Y * conj(X)), epsilon) ** exponent`

The exponent varies from 0 (plain cross correlation) to 1
(full PHAT whitening). A real inverse FFT produces lag evidence
for each integer offset in an explicit interval. The largest
absolute correlation peak supplies the signed integer lag;
optional parabolic refinement yields a fractional-lag hypothesis.

The result includes the strongest and second independent peak
scores, relative peak ratio, the selected channels and an ambiguity
flag. The scores are not calibrated probabilities.

Silent inputs or signals below the minimum RMS explicitly return
**no hypothesis**, rather than guessing that their delay is zero.

## Python API

```python
from sonumio.audio_core.gcc_phat_alignment import (
    AlignmentSettings, estimate_delay, align_observed_to_reference
)

settings = AlignmentSettings(
    maximum_lag_frames=2000,
    reference_channel=0,
    observed_channel=0,
    remove_dc=True,
    taper=True,
    phat_exponent=1.,
    minimum_signal_rms=1e-5,
)
report = estimate_delay(reference, observed, settings)
print(report.as_dict())
if report.hypothesis is not None:
    integer_aligned = align_observed_to_reference(observed, report)
```

Here `reference` and `observed` are immutable `AudioBuffer` objects.
The alignment utility shifts by **integer** frames only: it adds zeros
before delayed playback or discards leading samples for an advance,
preserving output duration and channel layout.

Fractional lag refinement is reported for analysis and is not applied
by that integer shift. Users requiring fractional-delay filtering need
a separately declared interpolation process.

## Scope and limitations

- No automatic clock-drift compensation or resampling.
- No trained speaker localization, speech recognition or source separation.
- Correlation can be ambiguous in periodic or reverberant audio.
- Unknown path reflections can make the highest peak differ from direct
  physical propagation latency.
- Channel choices are explicit; opposite-phase signals are never
  guessed together.
- FFT length, memory and lag-search range are bounded before
  allocation.
- Every source buffer remains unchanged, including on an error path.

Validate estimated delays against known test pulses or independently
recorded sync markers before using them as physical timing evidence.
