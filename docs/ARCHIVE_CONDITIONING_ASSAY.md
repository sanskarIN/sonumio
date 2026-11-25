# Private archive category-conditioning research assay

A low held-out diffusion noise MSE does not show that the network uses its
category conditioning. This paired intervention evaluates each eligible
validation clip under its correct category, every other supported category,
and the classifier-free null condition. Every alternative receives the same
noisy waveform, timestep and noise target. A zero-noise predictor is included.

Run only on approved local recordings and your own private checkpoint:

```bash
python -m research.archive_conditioning_assay private.scorpus experiment.stcheckpoint \
  --archive-sha256 ARCHIVE_DIGEST --checkpoint-sha256 CHECKPOINT_DIGEST \
  --steps 20 --channels 8 --batch-size 4 --diffusion-steps 8
```

Use the exact original training configuration, except the total step target
may be at least the checkpoint step. The same runtime restrictions as resume
apply. No optimizer updates or files are produced. The test split is never
scored, although archive verification and readiness preflight inspect it.

Three deterministic positions (first, middle, final diffusion timestep) are
tested; the actual noise levels depend on the schedule. A fixed per-row,
repeat and timestep seed binds the corruption trace independently of batch
geometry. Report values may differ slightly across numerical kernels.

Metrics include matched MSE, mean and hardest wrong-condition MSE, null and
zero controls, signed margins (positive favors the matched condition),
tie-aware mean rank and strict win fraction, and a true-category by
supplied-condition error matrix at each timestep. Recording-group macro
summaries counter overrepresentation by crop-rich groups. An all-zero model
must tie every conditioning alternative and equal the zero predictor.
Reports are aggregate only and contain no recording identifiers or PCM.

This assay measures teacher-forced prediction, not semantic accuracy of
generated audio, naturalness, statistical significance, or Sonumio-1
readiness. It is a diagnostic for choosing further controlled research.
No trained weights or real recordings are part of this contribution.


## Matched initialization control

Add `--compare-initial` to the same command to reconstruct the checkpoint's
original seeded model initialization and evaluate it with the identical
validation corruption panel before restoring trained parameters. The output
contains the initialization assay and trained-minus-initial deltas for
each available scalar metric, category, timestep and group-macro summary.
A changed noise trace, configuration, or coverage is rejected. Positive
margin changes favor matched conditioning; negative rank changes favor the
matched category. Lower matched MSE alone does not prove conditioning
improved. This comparison is one initialization, not an independent
multi-seed statistical study.
