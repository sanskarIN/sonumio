# Stochastic percussion research

This isolated CPU experiment learns parameters of a differentiable modal
and noise-bank synthesizer for kick, snare and hi-hat. Training targets
come from Sonumio's original procedural percussion renderer. No external
recordings or pretrained weights are used.

The model conditions on instrument, velocity and duration, and receives
an independently generated bounded random excitation. Source seeds are
disjoint between training and all evaluation panels. Independent noise
domains and replica identities do not give the model access to the source
PCM or require matching the original random phase.

Six bounded sinusoidal modes have learned frequency, decay and downward
frequency sweep. Three noise bases provide low-, mid- and high-frequency
emphasis, with learned decay. A simplex bounds their combined gain and
a learned attack/release envelope preserves note boundaries. The
instantaneous modal frequency is capped below Nyquist; this is not a
strict proof of alias-free output.

The objective combines compressed time-domain energy envelopes, band
spectra, time-frequency band envelopes and normalized spectral centroid.
All terms are phase-insensitive, and do not establish perceptual quality.
The study compares three trained initialization replicates against their
untrained states, a generic decaying broadband control, and both wrong
instrument labels with all other controls and excitation held fixed.

The default study evaluates held-out source seeds, velocity, duration and
joint conditions. It uses two independent excitation realizations per
target. It measures synthetic feature reconstruction only. No improved
percussion quality is assumed until measured results are reviewed.

Run with research dependencies:

```bash
python -m pytest -q tests/test_percussion_research.py
python -m research.percussion_study
```

No audio, checkpoints or weights are exported. The existing procedural
music engine and hybrid score routing remain unchanged.
