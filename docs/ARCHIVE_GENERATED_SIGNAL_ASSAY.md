# Generated-signal evaluation for the experimental waveform baseline

Teacher-forced noise prediction does not establish that sampled sound has
the requested content. This separate numerical assay samples the model
using deterministic epsilon-DDIM (eta=0), with no final hard clipping.
It exposes peak, RMS, DC, overrange, near-clip fraction and crest factor
before any output normalization. No samples or weights are saved.

```bash
python -m research.archive_generated_assay private.scorpus model.stcheckpoint \
  --archive-sha256 APPROVED_ARCHIVE --checkpoint-sha256 APPROVED_CHECKPOINT \
  --steps 20 --channels 8 --batch-size 4 --diffusion-steps 8 --reverse-steps 8
```

Use the exact original checkpoint training settings. Every requested
category and guidance strength starts from the same seed-specific
Gaussian tensor. Null guidance (0) must produce exactly the same output
for all category IDs. Guidance 1 is ordinary conditional prediction;
larger guidance mixes conditional and null predictions. The report
binds initial noise and raw waveform hashes. Nonfinite or extreme
intermediates are rejected, not silently clipped.

## Transparent reference probe

The reference descriptor uses 256-frame tapered FFTs with a 128-frame
hop, relative band powers and their temporal variation, spectral
centroid, zero crossings and spectral flatness. It is intentionally
small and interpretable, not a pretrained semantic representation.

Only eligible **training** recordings fit the prototype centroids and
category-balanced feature scaling. Each recording group contributes
equally within a category. A separate held-out validation report
measures whether the probe distinguishes these categories at all.
The test split is not scored; archive integrity and readiness preflight
still inspect it. Silence or constant signals are unscorable, and ties
are not counted as correct.

The generated panel reports distance margins to the nearest wrong
reference category and rank for the requested category, with paired
changes relative to null guidance. A positive margin is favorable
only in this feature space. Review held-out reference accuracy and
raw signal statistics together; a weak probe is not evidence of
semantic alignment. A strong probe can still be fooled by artificial
spectral cues. No metric here establishes listening quality,
generalization, or original Sonumio-1 readiness.

The diffusion schedule's terminal cumulative alpha is reported because
a Gaussian initial sample may mismatch the terminal training distribution
(particularly for short linear schedules). No claim is made that this
mismatch has been solved. The evaluator uses bounded, fixed-size CPU
samples; it does not support long-form generation or GPU inference.
