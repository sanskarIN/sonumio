# Canonical original wavetable scores and verified WAV publication

The `wavetable-render` CLI turns a strict, local JSON score into one
atomic, checksum-verified PCM WAV. It uses only Sonumio's mathematical
finite-Fourier synthesizer and does not require user audio recordings,
network services, trained models, or third-party melody samples.

## Example

```sh
sonumio wavetable-render \
    --score examples/original-wavetable-score.json \
    --output original-triad.wav
```

Optional controls include `--output-bits 16`, a declared
`--dither-seed`, `--expected-output-sha256`, and explicit
`--overwrite` or `--create-parent` flags.

The command writes a JSON receipt to stdout only after the output
has been successfully published and verified. Failures report a
nonzero status to stderr without emitting a misleading success
receipt or truncating an existing WAV file.

## Strict score schema

`protocol: sonumio-original-wavetable-score-v1`

Each plan specifies the exact sample rate, frame count, channel count,
waveform name, harmonic table size, maximum harmonic count, master
gain and attack/decay/sustain/release envelope. Every note declares
exact integer start and gate frames, MIDI number, velocity, pan and
normalized initial phase.

Missing or unexpected JSON keys are rejected, as are duplicate keys,
nonfinite values, deep structures, invalid notes and excessive score
sizes. A plan never chooses the destination file path. The output is
an independent, explicit command-line argument.

## Publication safety

The scoring engine first validates synthesis duration, voice count,
table geometry, output PCM memory and all note release boundaries.
It creates canonical float32 PCM in memory, rejects output peaks
above the normalized range, and optionally uses seeded TPDF dither
for deterministic integer conversion.

The resulting WAV is checked and atomically published by Sonumio's
existing file-publication service. A pinned SHA-256 mismatch prevents
publication, and an existing destination requires explicit overwrite.
The parser rejects score-path/destination identity and does not
follow score leaf symlinks.

The receipt identifies the canonical score SHA-256, byte-identical
WAV SHA-256, in-memory PCM SHA-256, PCM resolution, output geometry,
note count, maximum simultaneous polyphony, peak and RMS.

## Reproducibility notes

Canonical scores are serialized with sorted UTF-8 keys for stable
content hashing. The same plan, synthesizer implementation, Python
and NumPy behavior, quantization and declared seed produce repeatable
results on the supported test matrix; there is no claim of universal
bit-exactness across unrelated future DSP versions.

This is source-only original synthesis. It provides no real-time
performance guarantee, automatic content rights clearance,
perceptual-quality certification, or AI music model weights.
