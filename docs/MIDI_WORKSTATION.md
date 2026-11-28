# Sonumio MIDI workstation — local-first optional subsystem

This is an additive symbolic-music module built independently from Sonumio's existing strict arrangement and MIDI importer. It deliberately does **not** replace `sonumio.music_midi_read`, the existing audio core, or the web studio.

## Entry points

Run from an installed Sonumio package, or with `PYTHONPATH=src`:

```shell
python -m sonumio.midi_workstation --help
python -m sonumio.midi_workstation midi-inspect input.mid --notes
python -m sonumio.midi_workstation midi-validate input.mid
python -m sonumio.midi_workstation midi-import input.mid --output score.json
python -m sonumio.midi_workstation midi-export score.json --output exported.mid
python -m sonumio.midi_workstation score-new --output blank.json
python -m sonumio.midi_workstation score-inspect score.json
python -m sonumio.midi_workstation score-quantize score.json --step-ticks 120 --output quantized.json
python -m sonumio.midi_workstation score-humanize score.json --seed 42 --output expressive.json
python -m sonumio.midi_workstation score-pattern examples/original-midi-pattern.json --output original.json
python -m sonumio.midi_workstation score-diff score.json expressive.json
python -m sonumio.midi_workstation score-compare score.json expressive.json
python -m sonumio.midi_workstation score-crop score.json --start-tick 0 --stop-tick 960 --output excerpt.json
python -m sonumio.midi_workstation score-concat excerpt.json excerpt.json --output joined.json
python -m sonumio.midi_workstation history-init score.json --output history.json
python -m sonumio.midi_workstation history-inspect history.json
```

Every mutation requires an explicit destination and refuses to overwrite by default. Optional pinned input and output SHA-256 flags are available on supported commands. Source and destination alias checks include same-path and hardlink detection. The publication layer uses an atomic local staging protocol, but is not a multi-user adversarial filesystem sandbox.

## Architecture

- `smf`, `byteio`, `model` — bounded MIDI formats 0–2, running status, SysEx preservation, explicit PPQN/SMPTE clocks, and semantic roundtrip checks.
- `tempo`, `transport`, `controllers` — exact rational timestamp conversion, sample-clock scheduling and controller/bend interpolation.
- `performance`, `score`, `score_io` — expressive note reconstruction, typed immutable symbolic scores, canonical JSON interchange.
- `editing`, `regions`, `revisions`, `journal` — atomic note editing, half-open clip transforms, conflict-aware three-way merges and reversible digest-chained score snapshots.
- `groove`, `quantize`, `harmony`, `patterns`, `pattern_io` — musical timing, scales and triads, deterministic original pattern composition.
- `pianoroll` — backend-neutral viewport queries, rectangle selection, optimistic drag/resize gestures and safe clipboard.
- `analysis`, `compare` — inspectable structure, timing and note matching diagnostics; no claims of perceptual musical correctness.
- `wavetable_bridge` — optional lazy adapter to Sonumio's real `audio_core.wavetable`. The standalone tests for this adapter use a stub to validate parameter contracts. Real audio-core integration must be tested after applying the patch to a repository checkout.
- `publish`, `cli`, `score_commands`, `history_cli` — bounded local-only file interchange with deterministic machine-readable JSON receipts.

## Important invariants

1. This subpackage does not request network access, write GitHub metadata, or load external model weights.
2. Scores and controllers are immutable typed values; optimistic edit actions pin their source digests.
3. All event times use integer ticks and all audio transport scheduling uses explicit rational time conversion.
4. SMPTE MIDI timing is supported for inspection and timing analysis, but not losslessly represented in the separate PPQN-based editable `Score` object. This is an explicit limitation.
5. Imported MIDI CC, aftertouch and SysEx are never silently interpreted as equivalent symbolic score events. MIDI program changes are refused by default. An explicit `midi-import --allow-patch-rechannel` option can preserve note/program groups by assigning additional fixed patches to unused non-percussion MIDI channels; it records every reassignment, refuses drum-kit changes and refuses channel exhaustion. Such reassignment changes channel routing and is not a byte-identical MIDI roundtrip.
6. A key-on/key-off interval and a sustain-extended audible interval are distinct. In explicitly shared-channel performance mode, the originating MIDI track is retained even when a different track sends key-off, sustain release or channel-wide stop messages.
7. A decoded score or model is never proof that a melody is original, licensed or free of copyright. Pattern generator inputs are supplied by the user.
8. SHA-256 chains provide corruption detection, not authorship proof or authentication.

## Validation

In the isolated working directory:

```shell
PYTHONPATH=src python -m compileall -q src/sonumio/midi_workstation
PYTHONPATH=src pytest -q tests/test_workstation_*.py
```

The local test suite covers MIDI parse/encode, malformed inputs, controller state, tempo math, performance overlap, score edits, history, CLI, filesystem publication, and cross-feature integration. It does not replace Sonumio's existing full-repository test matrix, packaging checks, or platform CI. That requires applying this patch to a full local checkout and running the existing build.

## Integration guidance

The new module is discoverable under the existing `setuptools` package scan, because it resides in `src/sonumio/midi_workstation`. Do not automatically route existing top-level Sonumio commands through it until the legacy MIDI behavior, web studio and any public API consumers have been regression-tested. This subsystem is opt-in through `python -m sonumio.midi_workstation`.

A responsible release should additionally verify cross-platform path safety (Windows + Linux), full repo unit/integration tests, note-count performance, real wavetable output, and input boundary fuzzing.

## Read-only MIDI linter

The optional `midi-lint` command reports structurally permitted but musically
ambiguous conditions without editing or rendering the file. Findings cover
unmatched and unreleased notes, same-key overlaps, zero-length notes, cross-track
releases, controller closures, sustain left down, patch changes while held,
and conflicting or duplicate conductor metadata. Format-2 tracks remain
independent; format-0/1 channels are shared. Each finding has an original track
and tick, with deterministic bounded output (1–2048 detailed findings).

```bash
python -m sonumio.midi_workstation midi-lint original.mid
python -m sonumio.midi_workstation midi-lint original.mid --max-findings 32 --no-information
python -m sonumio.midi_workstation midi-lint original.mid --fail-on-errors
python -m sonumio.midi_workstation midi-lint original.mid --fail-on-warnings
```

The default returns a JSON report, including error findings, without mutation.
The optional nonzero gates leave standard output empty. No SysEx is executed.
Note pairing is a conservative FIFO diagnostic, not a claim about every synth.

## Raw MIDI note transform with loss-aware safety

`midi-transform` is for files that should retain original SMF events rather
than undergo lossy conversion into Sonumio's editable score. It preserves MIDI
ticks, track order, tempo, meter, controllers, arbitrary opaque metadata and
SysEx as decoded events, and changes only selected note-on/off pitch, polyphonic
key pressure pitch, and nonzero note-on velocity. Percussion is excluded by
default, note-on velocity zero remains a release, and note-off release velocity
remains unchanged. Export canonically re-encodes running status, so this is
event-semantic rather than byte-identical interchange.

```bash
python -m sonumio.midi_workstation midi-transform original.mid --output higher.mid --semitones 2
python -m sonumio.midi_workstation midi-transform original.mid --output softer.mid --velocity-numerator 3 --velocity-denominator 4
python -m sonumio.midi_workstation midi-transform original.mid --output selected.mid --track 1 --channel 0 --pitch-low 48 --pitch-high 84 --semitones -1
python -m sonumio.midi_workstation midi-transform original.mid --output other-key.mid --semitones 2 --transpose-key-signatures
```

Velocity scales as an exact rational, rounded half up, with an optional offset
and a documented [1,127] clamp. Pitch overflow aborts. New overlapping
same-key note-on identities abort unless `--allow-new-overlaps` is explicitly
set; existing overlaps are not silently repaired. Key-signature meta changes
require a whole-score selection and choose a minimal-accidental enharmonic
spelling. Without that flag, transposition retains original key-signature
metadata and exposes that fact in the receipt. Publication is atomic,
SHA-256-pinned, no-clobber by default, and never overwrites an input file.

## Hash-verified score-to-WAV preview

The opt-in `score-preview-wav` command renders a Sonumio symbolic workstation
score through the existing `sonumio.audio_core.wavetable` synthesizer. It does
not train or call a generative model, load third-party recordings or execute
MIDI SysEx. The synthesizer uses original finite-Fourier waveforms; audio
quality is not established by correctness tests.

```bash
python -m sonumio.midi_workstation score-preview-wav original-score.json --output preview.wav
python -m sonumio.midi_workstation score-preview-wav original-score.json --output preview.wav --sample-rate 24000 --channels 2 --waveform sine --master-gain 0.12 --output-bits 24 --dither-seed 2026
```

The score's exact rational tempo clock determines sample frames. Only active
pitched notes render by default; percussion notes are excluded because this
is not a drum sampler. Use `--include-muted` only when intentionally rendering
muted tracks. The preview enforces a 24 MiB float32 intermediate budget,
integer PCM bits 8/16/24/32, an explicit optional 32-bit deterministic dither
seed, a 64 MiB WAV transport bound and a peak limit of 1.0 before quantization.

The existing local generated-WAV atomic publisher enforces no-clobber by
default, checksum-pinned output, and symlink refusal. Score and output paths
must be distinct. The JSON receipt identifies the canonical source score,
output WAV SHA-256, float audio digest, sample clock, peak, RMS, rendered notes,
and whether any percussion notes were skipped. Wavetable render and real-WAV
round-trip tests run against Sonumio's actual audio core in full-repo CI.
