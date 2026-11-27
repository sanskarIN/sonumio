# Original granular synthesis

A deterministic, offline source-to-texture engine creates overlapping
grains from original procedural nature audio or an explicitly provided
canonical PCM WAV. It does not train or run a neural model.

The schedule is seeded by event index, uses independent fixed property
draws, and is computed before the samples are inspected. Controls cover
source scanning, position and timing jitter, pitch and reverse playback,
amplitude variation, and stereo spread. Each grain uses a finite
rate-dependent sinc kernel to reduce aliasing when reading faster than
the source clock. Wrap, reflect and zero extension are distinct boundary
policies; Hann, triangular and Tukey windows are available.

Overlap normalization divides by at least one, preserving tapered
isolated grain edges. Sparse grains may leave intentional silence.
A common peak guard only attenuates and does not hard clip. Mono sources
can become stereo; stereo inputs retain the original layout, and
multichannel panning is explicitly rejected.

Examples (run from the installed source tree):

```bash
python -m sonumio.audio_core.granular_cli granular --nature rain --frames 48000 --grain-frames 256 --hop-frames 128 --pitch-jitter 4 --reverse-probability .3 --stereo-spread .8 --output rain-texture.wav
python -m sonumio.audio_core.granular_cli granular --input rain-texture.wav --frames 24000 --grain-frames 128 --hop-frames 64 --pitch -5 --output transformed.wav
```

The source sample clock is retained. Source duration for procedural
nature is specified in frames at 24 kHz. Settings, schedules, input
PCM and output file hashes provide reproducibility diagnostics, not
authentication of ownership, permission or subjective sound quality.

WAV inputs are fully verified before rendering. Output bit depth defaults
to the source depth (16 bits for a generated source); optional seeded
TPDF dither is available. Output is staged and atomically published
only after a successful render and expected digest check. Explicit
overwrite is required to replace an existing ordinary file.

The renderer caps grain count, estimated interpolation work, PCM output
geometry, memory and per-grain scratch. Larger requests can fail even
when their nominal duration is within the general audio-core limit.
