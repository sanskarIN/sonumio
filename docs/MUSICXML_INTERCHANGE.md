# MusicXML performance interchange (source-only)

Sonumio's symbolic MIDI workstation includes an **opt-in**, locally executed
MusicXML 3.1/4.0 `score-partwise` interchange projection. This is **not** an
engraving engine, a general MusicXML implementation, a General MIDI soundfont,
or a neural music model. Source files are never modified by inspection.

## CLI examples

From an installed editable Sonumio checkout, or `PYTHONPATH=src`:

```bash
python -m sonumio.midi_workstation.musicxml_cli to-xml original.json --output original.musicxml
python -m sonumio.midi_workstation.musicxml_cli to-mxl original.json --output original.mxl
python -m sonumio.midi_workstation.musicxml_cli inspect original.musicxml
python -m sonumio.midi_workstation.musicxml_cli audit original.musicxml
python -m sonumio.midi_workstation.musicxml_cli from-xml edited.musicxml --output imported.json
python -m sonumio.midi_workstation.musicxml_cli from-mxl edited.mxl --output imported.json
python -m sonumio.midi_workstation.musicxml_cli preview-xml edited.musicxml --output original.wav
python -m sonumio.midi_workstation.musicxml_cli preview-mxl edited.mxl --output original.wav
```

Preview commands use Sonumio's existing deterministic, mathematical wavetable
renderer and hash-pinned PCM publication path; there are no trained weights.
Use `--sample-rate 24000 --channels 2 --waveform triangle --gain 0.15 --bits 16`
to set explicit preview parameters. Additional flags permit finite dithering,
output SHA256 pinning, explicit overwrite or parent directory creation.

## Accepted MusicXML semantics

- MusicXML 3.1 or 4.0, UTF-8, `score-partwise` with consecutive numbered measures.
- Integer `divisions` shared across parts; compatible with the Sonumio score's PPQN.
- Global tempo `direction/sound@tempo` and bar-aligned time signatures.
- Explicit MIDI pitched note values (`step`, integer `alter`, `octave`).
- Integer durations, rests, `forward`/`backup` multi-voice timing, and chords.
- Bar-crossing tie start/stop chains (validated for continuity and identity).
- Part names and explicit MIDI channel/program routing when present.
- Sonumio note IDs and exact project end time on Sonumio-origin exports.

Unsupported performance or notation constructs (unpitched drums, grace notes,
repeat navigation, tuplets, instrument transposition, ornament rendering,
per-note MIDI controllers, microtonal alterations, arbitrary midbar meter
changes) are deliberately rejected rather than silently guessed.

### Semantic loss is explicit

Sonumio tracks additional metadata that the restricted MusicXML performance
projection cannot carry: exact original velocity, release velocity, per-note
pan, score revision, track mute/solo flags, MIDI bank data and engraving.
Imported notes use stable defaults **velocity 100, release velocity 0, pan 64**;
receipts state these changes. A MusicXML round-trip does **not** claim to
reconstruct every byte or expressive setting of the original Sonumio score.

### Source and ZIP security

- XML byte budget: 4 MiB; 160,000 elements; depth 48; 16,384 text characters.
- No DTD, entity declarations, processing instructions, remote resource loads,
  arbitrary XML execution, or external sample paths.
- Bounded part count, note fragments, voices, measures and program channels.
- `.mxl` archives use exactly three members: uncompressed `mimetype`,
  `META-INF/container.xml`, and `score.musicxml`.
- Archives reject unwanted members, traversal paths, symlinks, encrypted
  entries, invalid metadata, checksum corruption and invalid compressed data.
- File I/O reuses strict regular-file reads, SHA-256 guards, atomic publication,
  no-clobber output defaults, and symlink / hardlink safety checks.

`.mxl` support is a canonical intentionally restricted OCF subset; arbitrary
third-party `.mxl` packages with extra resources require a different importer
policy and are not accepted as equivalent.

### Provenance / release limitations

A source-only build has no trained model weights. MusicXML WAV preview uses
simple original mathematical waveforms; rendering a MusicXML score does not
establish artistic or perceptual audio quality. Performance editing, sound
design, audio mastering, and a final commercial-quality DAW UX remain future
engineering work. This feature requires repository CI and maintainer review
before inclusion in a stable release.
