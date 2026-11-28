# Quiet-region sustained instrument research

The earlier absolute-edge factorial study improved attack reconstruction
but regressed held-out release waveform error. This follow-up keeps the
normalized-coordinate architecture fixed and investigates training loss
normalization instead of replacing the production engine.

Three arms share the original model weights at initialization, source
corpus, edge-distance window schedule, AdamW optimizer and update count.
The baseline uses the existing relative waveform and spectral loss.
The floor arm changes only the reference waveform power denominator
from a minimum of 1e-8 to 1e-4. The third arm adds the existing
compressed RMS-envelope loss. These values are set before evaluation.

Reports retain the original relative error, absolute waveform MSE,
envelope error, RMS difference, per-instrument comparisons, initial
and wrong-label controls, and gradient clipping/floor frequencies.
Positive and negative effects are both reported. A reduced training
loss alone is not evidence of improved generated audio.

    python -m pytest -q tests/test_instrument_quiet_research.py
    python -m research.instrument_quiet_study

No weights or generated audio are published. No held-out controls enter
training and no perceptual or real-recording generalization is claimed.
