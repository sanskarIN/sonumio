# Sustained-note edge coordinate factorial study

The original sustained harmonic model predicts normalized attack and
release fractions. Fixed-time procedural source edges do not scale
proportionally with note duration. This experiment maps the identical
two learned control outputs into bounded logarithmic absolute sample
durations instead. It supplies no source-instrument-specific edge
constants and preserves the same oscillator model and parameter count.

Four prespecified arms compare normalized versus absolute coordinates
and the original waveform/spectral loss versus the same loss plus a
compressed RMS-envelope error. All arms share model initialization
weights per seed, source corpus, deterministic edge-distance schedule,
optimizer and update count. The sampler covers the actual endpoints
and multiple distances from them. Training uses only the existing
disjoint training controls.

The primary comparison is absolute-coordinate + envelope objective
against normalized-coordinate + original loss, with separate contrasts
for each factor. Full notes, thirds of the note and the first/last
2048 samples are evaluated on duration-heldout and joint-heldout panels.
Metrics include relative and absolute waveform MSE, RMS calibration,
and phase-insensitive envelope error with a fixed near-silence floor.
The report retains per-instrument averages, matched initialization
controls, fixed sine, all wrong labels for the primary arm, and seed
ranges. It reports negative as well as positive effects.

Run the CPU study:

```bash
python -m pytest -q tests/test_instrument_edge_research.py
python -m research.instrument_edge_study
```

No production renderer change, trained checkpoint, generated audio
file, external recording, or perceptual-quality claim is introduced.
