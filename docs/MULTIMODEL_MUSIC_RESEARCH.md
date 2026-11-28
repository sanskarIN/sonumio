# Multi-model music reconstruction

This research pipeline combines the existing short-note harmonic model,
sustained harmonic model, and stochastic percussion model without
publishing checkpoints or altering the original production synthesizer.

Eligible pitched events are routed to the short (2048–8192 frames) or
sustained (12000–96000 frames) model. The gap between these domains is
procedural, as are unsupported pitches or velocities. Percussion
supports 2048–8192 frames. Exact pitched training controls and all
percussion training source seeds are excluded. This is a control-domain
check, not a claim of generalization to recordings or all durations.

The same original arrangement clock, event order, track pan and mixing
settings are preserved. Learned drum excitation is independently
derived from each event identity and evaluation replica. Every
substituted note receives family-specific diagnostics. Pitched
comparison is phase-sensitive relative waveform error. Stochastic
percussion is compared by temporal and spectral feature errors, not
the phase of the original random waveform.

The study trains three independent models for each of three
initialization replicates. It compares generic synthesis, initialization,
trained matched labels, three single-family ablations restoring
procedural notes, one shifted pitched label and one shifted percussion
label. Wrong-label interventions leave unrelated model families
unchanged. Full-mix features compare each stereo channel in bounded
windows; the final short block is zero padded identically and weighted
by its actual frame count. These features do not establish listening
quality or realism.

Run locally with research dependencies:

```bash
python -m pytest -q tests/test_multimodel_music_research.py
python -m research.multimodel_music_study
```

The full study is intended for a CPU research environment. No audio
files, external recordings, model weights or checkpoints are published.
