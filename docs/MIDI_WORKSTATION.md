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
5. Imported MIDI CC, aftertouch, SysEx and program changes are never silently interpreted as equivalent symbolic score events. Some workflows refuse such conversions rather than inventing semantics.
6. A key-on/key-off interval and a sustain-extended audible interval are distinct.
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
