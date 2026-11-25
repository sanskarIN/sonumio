# Private archive crop latent experiment

This local research command combines the reviewed archive reader and
readiness audit with the crop-compatible quantized codec and class-conditioned
velocity denoiser. The archive must have an independently retained SHA256.
Use only recordings that are approved for the intended research.

```bash
python -m research.archive_latent_study private.scorpus --expected-sha256 ARCHIVE_SHA256 \
  --codec-steps 32 --diffusion-updates 64 --batch-size 4 --bits 8 \
  --width 32 --schedule-steps 32 --reverse-steps 8
```

Stage 1 trains the codec on group/asset/category-balanced training batches.
Then the codec is frozen. A streamed calibration scans each eligible
training clip once. By default, the moments are weighted uniformly by
category, then recording group, asset and crop. This matches the sampler's
intended hierarchy rather than the number of crops in each recording.
Use `--calibration-mode clip_weighted` for the original uniform-crop
control. The report includes category mass and a weight-plan digest,
without identifiers. The weighted channel variance includes both within-
and between-clip variation.
Stage 2 uses a separate deterministic balanced sampler and encodes each
minibatch on demand. No full waveform or latent corpus is materialized.
Each crop is checked against audited row and PCM identity before use.

Validation scores all wrong categories, correct category, null and zero
controls at fixed early/middle/late timesteps, with paired corruption.
The report distinguishes codec reconstruction error from decoded x0 error,
native latent projection and quantization. Group-macro summaries are
aggregate, with no recording group identifiers. Evaluation scans one crop
at a time for repeatable numerical geometry. The test split is inspected
during readiness but is not used for optimization, calibration or scoring.

The bounded generated panel uses the same initial noise across categories
and guidance 0, 1 and 2. It reports raw signal summaries, decoder
preactivation and native projection, not semantic accuracy or listening
quality. All weights and generated audio remain in memory. The report
contains data/plan/model fingerprints and aggregate calibration values;
treat it as private when the input corpus is private.

Limits include 4096 eligible crops per train/validation split, 16 clips
per optimizer batch, at most 32,000 combined training draws and separate
validation and generation forward-call budgets. This is CPU research,
not a large-scale corpus trainer, a production codec, or Sonumio-1.
It does not include optimizer checkpoint recovery or GPU support yet.
