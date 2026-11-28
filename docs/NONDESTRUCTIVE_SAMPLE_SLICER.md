# Non-destructive sample slicing and timeline playback

Sonumio's opt-in sample slicer treats a single immutable source
`AudioBuffer` as a bounded library of half-open sample-frame slices.
Trigger events play those slices on an exact output sample clock
without changing the source or accessing filesystem assets.

This supports procedural percussion arrangements, test stimuli and
simple sample-based game/audio research where source material is
appropriately licensed or authored by the user.

## Slice and trigger geometry

A `SampleSlice` declares `[start_frame, stop_frame)` inside the
source. There are no hidden resampling rules or seconds-to-frame
ambiguities. A `SamplerTrigger` declares:

- Output start frame and zero-based slice reference.
- Velocity between 0 and 1.
- Explicit playback rate from 0.125 to 4 times original.
- Optional reverse playback.
- Mono-to-stereo equal-power pan.
- Short bounded in/out fades.

The output duration is `ceil(slice_frames / playback_rate)`.
At non-integer sample positions, the renderer uses linear
interpolation between adjacent input samples. Playback speed **does
change pitch** and is not a pitch-preserving time-stretch algorithm.
Neither anti-aliasing nor high-fidelity rate conversion is claimed.

## Python example

```python
from sonumio.audio_core.sample_slicer import (
    SampleSlice, SamplerTrigger, SamplerSettings,
    sample_slices_from_onsets, render_sampler,
)

slices = (
    SampleSlice(0, 1200, label="kick"),
    SampleSlice(1200, 2200, label="snare"),
)
triggers = (
    SamplerTrigger(0, 0, velocity=.8, fade_frames=8, pan=-.3),
    SamplerTrigger(6000, 1, velocity=.7, fade_frames=8, pan=.3),
    SamplerTrigger(12000, 0, velocity=.6, rate=.8, reverse=True),
)
options = SamplerSettings(frames=24000, channels=2, gain=.5)
result = render_sampler(source, slices, triggers, options)
mixed = result.audio
report = result.report.as_dict()
```

The `source` must be an existing canonical immutable AudioBuffer.
Explicit onset frame positions from the rhythm analyzer can be
converted to sample slices with
`sample_slices_from_onsets(source, onset_frames)`, after reviewing
the onset hypotheses.

## Bounded offline audio contracts

- At most 1024 slice descriptors, 2048 triggers and 64 simultaneous
  playback voices.
- Total rendered event samples, work memory and output PCM duration
  are bounded before synthesis.
- Unsupported channel changes are rejected. Native channel layout
  is preserved; only mono-to-stereo conversion is explicitly supported.
- A stereo source may not use implicit per-event pan remapping.
- Every event must fit within declared output frames; samples beyond
  a slice cannot be read.
- There is no clipping, volume normalization or trained model
  inference. Overrange PCM remains explicit for downstream validation.

This is an offline deterministic reference. It is not a real-time
low-latency sampler engine and does not manage soundfonts, sample
libraries, MIDI devices, voice stealing or external asset licenses.
