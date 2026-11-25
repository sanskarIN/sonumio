# Prespecified paired label-supervision study

This experiment tests whether correct category supervision changes a small
waveform diffusion model's teacher-forced prediction relative to a control
trained with independent uniform random category labels. It is not a
generated audio listening test, nor a text-to-audio benchmark.

Run on your own reviewed archive with its independently retained digest:

```bash
python -m research.archive_label_study private.scorpus --expected-sha256 APPROVED_DIGEST \
  --seeds 17,29,41 --steps 64 --batch-size 8 --channels 16 --diffusion-steps 32
```

Each seed creates one original initialization and clones it into two arms.
Each optimizer step uses one audited training batch, a single diffusion
noise tensor and timestep vector, and the same null-condition dropout mask.
The only intervention is the supplied category for non-null examples:
the matched arm receives the true category and the control receives
uniform independent categories with replacement. Accidental matches are
expected (approximately one in six), not forcibly excluded. Counts are
reported. A fixed shared training trace binds actual sampled indices,
timesteps, noise, masks and both labels. The arms train on identical clean
waveforms with separate optimizers.

The initial and both trained models use the same eligible validation rows,
noise panel, timesteps and batch geometry. The validation panel is fixed
across training seeds. This enables descriptive paired deltas, not
independent validation corpus replications. We report matched-minus-initial,
random-minus-initial, and matched-minus-random deltas at the clip and
recording-group levels, with category and timestep strata. Positive
wrong-minus-matched margins and negative rank changes favor the matched
condition. Lower MSE alone does not prove semantic conditioning.

The plan is bounded before optimization (2–8 distinct seeds, at most
100,000 training examples across arms and 2,000,000 evaluation prediction
rows). These are workload caps, not runtime or process-memory guarantees.
The plan binds source archive and eligibility fingerprints. CPU execution
preserves caller RNG; numerical reproducibility is scoped to the same
runtime. Report values contain no private identifiers or sample PCM.
No weights or audio files are written. The test split is not scored;
its archive integrity and readiness are inspected during preflight.

Use the per-seed observations, mean, median, sample standard deviation,
range and sign counts as descriptive evidence. Do not treat these ranges
as confidence intervals, report significance, or infer generalization to
unseen recordings, languages, or open-vocabulary prompts.
