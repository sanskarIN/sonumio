# Canonical original plucked-string scores

The `plucked-render` command consumes a strict local JSON score,
renders its original mathematical damped-delay string voices, and
atomically publishes a digest-verified WAV. It reads no instrument
samples, commercial music, user recordings or trained model weights.

## Command

```sh
sonumio plucked-render \
  --score examples/original-plucked-string-score.json \
  --output generated-string.wav
```

Optional flags include `--output-bits 24`,
`--dither-seed 1234`, `--expected-output-sha256`,
`--overwrite` and `--create-parent`.

A successful command emits a single canonical JSON receipt with
the score SHA-256, output PCM SHA-256, byte-exact WAV SHA-256,
number of voices, output sample clock, peak and RMS. Invalid
input prints a concise error to stderr and returns nonzero.

## Score schema

The `protocol` is `sonumio-original-plucked-score-v1`.
The `plan` declares sample rate, total frames, channel count,
output gain and bounded string-voice parameters. Each note contains
start/duration in integer frames, MIDI pitch, velocity, pan and an
explicit deterministic noise seed.

Unknown/missing fields, nonfinite values, duplicate keys,
unbounded note counts, oversized JSON, invalid frequency/loop
geometry and forbidden output paths inside the score are rejected.

The score's canonical digest is based on sorted UTF-8 JSON keys;
the source file may use any ordinary JSON whitespace and key order.
No URLs or filesystem output paths belong to its data schema.

## Verified publication

The score is read from a bounded, regular file descriptor without
following leaf symlinks where supported. Source/destination aliases
are checked before rendering and before publication. The output
PCM memory budget is checked before synthesis.

A generated buffer with a peak exceeding normalized PCM is rejected
rather than silently clipped; adjust the declared master gain.
The canonical encoder supports 8/16/24/32-bit integer PCM with
optional explicitly seeded TPDF dithering. Publication stages the
complete WAV and hashes its bytes before an atomic no-clobber
or explicit overwrite operation.

Failures cannot emit a success receipt or partially replace an
existing output file under the documented local-filesystem
assumptions. This is not a hostile multi-process filesystem sandbox.

## Reproducibility and scope

The same score, implementation, quantization and seed produce
repeatable output in the supported build environment. The resulting
sound is synthesized from an original procedural algorithm, but
neither perceptual instrument accuracy nor exact pitch is claimed:
the Karplus-Strong-like feedback loop has an approximate integer
delay period.

The score is intended for local sound design prototypes,
regression tests, procedural game sounds, and open-source research.
