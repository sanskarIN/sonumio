# Original finite-Fourier wavetable synthesis

Sonumio now has a deterministic local reference oscillator for mathematical
sine, sawtooth, square and triangle waveforms. It renders MIDI-style note
events on an exact integer sample clock, supporting bounded polyphony,
individual note envelopes, velocity and optional stereo panning.

This is *not* a trained AI music generator, a general sampler, a license
to emulate proprietary recordings, or a completely alias-free virtual
analog synthesizer.

## Harmonic tables

Every note frequency is computed from standard equal-tempered MIDI note
numbers, using A4=440 Hz. The table contains only integer harmonics
whose frequencies lie strictly below the source Nyquist limit.

The Fourier series are mathematical original waveforms:

- Sine: first harmonic only.
- Saw: alternating-sign harmonics with amplitude proportional to 1/n.
- Square: odd harmonics with amplitude proportional to 1/n.
- Triangle: alternating odd harmonics proportional to 1/n².

A fixed-length table is peak-normalized and reused for notes with the same
harmonic limit. Phase advances on the original sample clock and uses
linear interpolation between adjacent table values.

Truncating the Fourier series suppresses high-frequency harmonics before
Nyquist but does not remove every possible interpolation image or alias.
No true-peak oversampling or perceptual waveform matching is claimed.

## Polyphonic note events

Each `WavetableNote` has a start frame, positive gate duration, MIDI
pitch 0..127, 0..1 velocity, -1..1 pan and normalized initial phase.
A `VoiceEnvelope` has bounded attack/decay/sustain/release controls.
The release starts from the actual gate-end level, including when the
gate closes before attack or decay completes.

The renderer refuses notes whose release would exceed the declared
output duration. A sweep-line validation limits simultaneous active
voices to 64, and separate budgets constrain total event samples,
table bank construction and PCM work memory.

Stereo panning uses explicit equal-power gains. Mono notes must use
center pan; no arbitrary channel topology conversion takes place.

## Python example

```python
from sonumio.audio_core.wavetable import (
    VoiceEnvelope, WavetablePlan, WavetableNote, render_wavetable
)

plan = WavetablePlan(
    sample_rate=24000,
    frames=48000,
    channels=2,
    waveform="triangle",
    table_size=2048,
    max_harmonics=64,
    master_gain=.2,
    envelope=VoiceEnvelope(
        attack_frames=120,
        decay_frames=800,
        sustain_level=.65,
        release_frames=1600,
    ),
)
notes = (
    WavetableNote(0, 8000, 60, velocity=.7, pan=-.4),
    WavetableNote(4000, 8000, 64, velocity=.7, pan=0.),
    WavetableNote(8000, 8000, 67, velocity=.7, pan=.4),
)
outcome = render_wavetable(plan, notes)
audio = outcome.audio
report = outcome.report.as_dict()
```

The synthesis report lists each note's sample span and harmonic count,
peak/RMS, maximum polyphony and distinct table count. Audio is kept in
bounded immutable `AudioBuffer` storage with no implicit PCM clipping or
final normalization. A downstream WAV encoder may reject overrange audio.

No file, microphone, network, model weights or proprietary score is
required by this reference oscillator. Tests verify deterministic
replay, frequency at A4, Fourier table immutability, pan law, gate/release
boundaries and unsafe geometry rejection.
