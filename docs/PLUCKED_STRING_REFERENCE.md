# Original damped-delay plucked string synthesis

Sonumio contains a deterministic, CPU-only plucked-string-like oscillator
based on a short feedback delay and frequency-dependent damping.

This Karplus-Strong-style method provides a lightweight source of
original pitched sounds without sample libraries, model weights or
network access. It is not calibrated to any physical instrument and
its nominal pitch is only approximate because of delay rounding.

## Excitation

For each note, a seeded, bounded random-noise ring is generated
independently. A short moving average controls initial high-frequency
content. A spatial comb difference with an explicit pluck-position
parameter introduces repeatable overtone notches.

The excitation is peak-scaled to a normalized range, but note playback
is **not** silently peak-normalized after combining multiple voices.
The data in the delay ring are local to each note and never reused
between independently rendered arrangements.

## Damped feedback

The circular buffer has roughly one fundamental period of delay.
At each sample, the oldest ring value is emitted; the same slot
receives a damped blend between the newest and previous output:

`new = damping * (brightness * current + (1-brightness)*(current+previous)/2)`

The feedback gain is strictly below one. This produces a decaying,
approximately periodic sound. At high pitches, the small integer
delay introduces audible tuning error. The report includes the
approximate feedback-loop frequency so consumers can inspect it.

Sample-accurate note start and duration, velocity, bounded attack,
and linear fade-out release are independent of rendering chunk size.

## Python example

```python
from sonumio.audio_core.plucked_string import (
    PluckedStringPlan, StringVoice, PluckNote, render_plucked_strings
)

plan = PluckedStringPlan(
    sample_rate=24000,
    frames=48000,
    channels=2,
    gain=.2,
    voice=StringVoice(
        damping=.995,
        brightness=.5,
        pluck_position=.22,
        attack_frames=8,
        release_frames=256,
        excitation_smoothing=3,
    ),
)
notes = (
    PluckNote(0, 10000, 60, velocity=.8, pan=-.5, seed=101),
    PluckNote(6000, 10000, 64, velocity=.7, pan=0., seed=102),
    PluckNote(12000, 10000, 67, velocity=.7, pan=.5, seed=103),
)
result = render_plucked_strings(plan, notes)
audio = result.audio
report = result.report.as_dict()
```

The result is canonical immutable PCM that can be routed into Sonumio's
existing effect and WAV encoding pipelines. No external audio is
automatically downloaded or copied.

## Constraints

- Maximum of 1024 note events and at most 32 sounding simultaneously.
- Every note's release must fit the declared output frame count.
- The loop delay must contain at least four samples; excessively high
  fundamental frequencies are rejected rather than silently aliased.
- Stereo pan uses equal-power gains; mono permits center only.
- Work, rendering duration and PCM memory have explicit upper bounds.
- Identical plans, note arrays, and seeds produce identical output.
- No clipped, mastered, learned, or physically verified sound claim.

This reference provides a basis for controlled audio-synthesis research.
Real instrument likeness and tuning stability require listening tests,
measurement and a more sophisticated fractional-delay feedback design.
