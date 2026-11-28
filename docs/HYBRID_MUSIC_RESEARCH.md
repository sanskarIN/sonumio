# Hybrid learned arrangement reconstruction

This research-only workflow connects the learned harmonic note model to
complete original arrangements without publishing checkpoints or audio.

Run `python -m research.hybrid_music_study` in the CPU research environment.
The study trains from the existing first-party synthetic note grid, then
compares four prespecified arrangements with the same score and mix.

The note renderer substitutes only pitched notes within the model's MIDI
pitch (36–88), velocity (0.100–1.000, quantized for conditioning), and
frame-count (2048–8192) ranges. Exact training control identities are
excluded from substitution. All remaining events use the original
procedural engine, including percussion and longer notes. Velocity
rounding is compensated in the waveform amplitude, but not the learned
conditioning. This is not a trained full-arrangement neural synthesizer.

Each score is compared against its procedural reference using:
- a generic sine synthesizer on exactly the eligible notes;
- the initialized neural model before training;
- the trained model with correct instrument conditions;
- each of the four alternative instrument conditions, with the same
  trained weights and identical score timing.

The report counts event and sample coverage, identifies all fallback
reasons, checks fallback-only tracks for exact PCM identity, verifies the
model state is unchanged during rendering, and compares stereo waveform
errors and rendered WAV digests. Each comparison uses the same original
mix settings; peak attenuation may differ between reference and
candidate. Both pre-mix stem and post-mix diagnostics are available.

The results measure reconstruction of Sonumio's own synthetic source
family; they are not claims of perceptual quality, novelty, realistic
instrument imitation, or open-vocabulary text-to-music capability.
No generated files, external recordings or learned weights are included.
