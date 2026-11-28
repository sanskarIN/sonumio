# Original finite-Fourier stems — verified local source preview

This source-only subsystem creates an independent audio recording of every
explicitly selected pitched track in an immutable Sonumio workstation score.
It does not synthesize General MIDI sound banks or commercial samples. Each
track's waveform is selected from the original Fourier instruments provided
by Sonumio's existing `audio_core.wavetable` implementation. It makes no
claim of trained neural generation or perceptual music quality.

## Reproducible command sequence

From the repository root after installing Sonumio with Python 3.11+:

```sh
python -m sonumio.midi_workstation.stem_package_cli plan \
  examples/original-stems-score.json \
  --recipe examples/original-stems-recipe.json

python -m sonumio.midi_workstation.stem_package_cli render \
  examples/original-stems-score.json \
  --recipe examples/original-stems-recipe.json \
  --output original-stems.zip

python -m sonumio.midi_workstation.stem_package_cli inspect \
  original-stems.zip

python -m sonumio.midi_workstation.stem_mixdown_cli unity-recipe \
  original-stems.zip --output original-stems-mix.json

python -m sonumio.midi_workstation.stem_mixdown_cli plan \
  original-stems.zip --recipe original-stems-mix.json

python -m sonumio.midi_workstation.stem_mixdown_cli render \
  original-stems.zip --recipe original-stems-mix.json \
  --output original-stems-master.wav
```

The original example score and its exact canonical recipe were generated with
`encode_score` and `encode_recipe`; **the recipe contains the actual source
SHA-256, not a placeholder**. A changed score requires a new recipe. The
`unity-recipe` command similarly pins the digest of the actual archive.

## Resources and clock

Every stem WAV has identical frame count, sample rate, number of channels and
bit depth. All notes retain their original global onset and offset frame
positions, independent of the order in which stems are rendered. Different
release envelopes retain their requested length and are padded with zero PCM
to the common maximum tail. No per-stem normalization silently changes gain.

Limits are deliberate: at most eight explicitly mapped tracks, 2,048 pitched
notes, 900,000 aggregate per-track samples, 4 MiB per package, and integer PCM
8/16/24/32-bit output. The underlying synthesizer may enforce tighter
resource limits. Empty or muted tracks require explicit policy, and
unpitched percussion cannot be silently treated as a pitched waveform;
`skip_percussion` must be explicitly enabled when present. Preflight computes
budgets before any audio samples are generated.

## ZIP container contract

A canonical archive contains `manifest.json` followed by sorted
`stems/<stable-track-id>.wav` entries. Members use fixed modification times,
byte-preserving compression mode, no unknown paths or encrypted members,
and a canonical JSON manifest containing original score SHA-256, original
mathematical patch settings, real PCM clocks, per-member SHA-256 and numerical
audio statistics. SHA-256 values are **integrity identifiers, not digital
signatures or origin authentication**.

Inspections reject malformed headers, mismatched RIFF geometry, invalid
sample widths, truncated PCM, unexpected ZIP entries, traversal names,
symlinks, duplicates, noncanonical manifests, and checksum mismatches.
No archive content is executed. The archive is intentionally small enough
for bounded in-memory validation.

## Mixdown semantics

Mixdown reads the package's PCM recordings into a bounded float64 accumulator.
The complete canonical gain list names every package stem, with integers
measured in thousandths (1,000 means unity; 0 means explicit mute; 2,000
means 2×). All stems share a clock and are never automatically stretched,
time-shifted, sample-rate converted or independently peak-normalized.

The default policy rejects a mix whose peak exceeds the target (980 means
0.98); deliberate `"peak_policy":"normalize"` applies a single master
attenuation to the entire sum, preserving relative track balances. Receipts
record pre- and post-normalization peaks, the normalization factor, output
bit depth, immutable source package digest and a SHA-256 of the rendered
float32 sample buffer. The generated PCM WAV is atomically published through
Sonumio's existing generated-WAV writer.

## Source, file safety and limits

Scores and recipes are read as bounded regular files with optional expected
SHA-256 values, no symbolic links, duplicate fields, executable code or
unrecognized JSON schema attributes. The CLI refuses to overwrite original
source score, recipe and stem ZIP paths; it will not overwrite an existing
artifact without an explicit switch. The publication step verifies expected
output SHA-256 before creating the file and stages output in the destination
directory. No network upload or GitHub mutation occurs.

ZIP publication is a single file; extracting multiple WAV files by arbitrary
ZIP tools is outside Sonumio's atomic-publication guarantees. Preview
stems may be used in other audio editors but require human listening and
proper licensing review before any public release. Source-only build and
CI checks are distinct from user-facing release approval.
