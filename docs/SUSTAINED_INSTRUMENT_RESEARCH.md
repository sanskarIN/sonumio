# Sustained harmonic instrument research

The experimental windowed harmonic model preserves the original short-note
network layout while evaluating global note phase, control anchors and
edge envelopes at absolute sample positions. The research API accepts
notes up to 96000 samples (four seconds at 24 kHz). Each differentiable
window is limited to 8192 samples; long inference uses a bounded iterator.
A long note is never approximated by stitching separately restarted notes.

The default training grid contains five pitched instruments, three pitches,
two velocities and two durations (0.5 and 1.5 seconds). Evaluation uses
disjoint pitch, velocity, duration and jointly held-out panels; held-out
lengths are 1 and 2 seconds. Source waveforms are first-party procedural
renders, with exact source identity and waveform hashes checked.

Training cycles through attack, sustain and release thirds and note
lengths, using deterministic 2048-frame windows. Attack and release
windows include the actual start and end of the note. Sustain windows
are sampled within the middle third. The existing waveform and
compressed-STFT objective is applied to each window. Evaluation visits
every frame in every held-out note and reports separate region metrics
alongside full-note relative waveform MSE, RMS ratio and waveform cosine.

The three independent initialization replicates compare trained matched
conditions against initialization, generic sine and all four mismatched
instrument labels. These are waveform reconstruction measurements of a
synthetic family. They do not establish subjective quality, performance
on real recordings, or improved results for all instruments. Four-second
inference geometry is tested; the default training study does not
establish four-second reconstruction quality. The existing hybrid
score renderer is unchanged until this separate study is qualified.

Run locally with the research dependencies:

```bash
python -m pytest -q tests/test_instrument_long_research.py
python -m research.instrument_long_study
```

The study writes only a JSON report to standard output. No checkpoints,
external recordings, generated audio or pretrained weights are published.
