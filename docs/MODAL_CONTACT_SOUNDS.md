# Procedural modal contact sounds

The original contact synthesizer constructs damped sinusoidal modes and
seeded, spectrally shaped contact noise. It is a source-only DSP tool,
not a physical acoustic simulator, trained model or recording library.

Eight authored material profiles (wood, metal, glass, ceramic, stone,
rubber, paper and gravel) specify relative frequencies, decay times and
contact-noise parameters. Size changes mode frequency, damping changes
decay, and hardness changes mode excitation weights. A custom resonator
can instead declare its own sorted frequency/decay/amplitude list.

Strike, tap, scrape, rattle and step excitations use bounded, independent
event schedules. Mode frequencies above the sample-clock guard are
omitted explicitly and reported. Each event receives an attack and
optional end taper; stereo mixing uses constant-power pan. The output
ceiling only attenuates the complete mix.

A deterministic pattern planner places events on an absolute sample
clock, with configurable interval, timing jitter, strength, pitch and
alternating pan. It does not infer walking gait or real-world motion.

```bash
python -m sonumio.audio_core.contact_cli contact-sound --material gravel --excitation step --count 6 --interval-ms 450 --event-ms 160 --timing-jitter-ms 20 --seed 12 --output gravel.wav
```

The command also accepts a strict JSON file:

```json
{
  "sample_rate": 24000,
  "frames": 24000,
  "events": [
    {
      "start_frame": 0,
      "frames": 12000,
      "resonator": {"material": "metal", "fundamental_hz": 360},
      "excitation": "tap",
      "strength": 0.6,
      "pan": -0.4
    }
  ]
}
```

The scene digest, mode counts, impulse counts, PCM digest and WAV SHA256
are reported. Work and memory are bounded before rendering. Export uses
the existing atomic no-clobber writer, with explicit bit depth and optional
seeded dither. Numerical tests cover pitch, decay, sample-clock geometry,
determinism, pan isolation, clipping policy and malformed inputs.
