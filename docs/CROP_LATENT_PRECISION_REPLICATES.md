# Multi-seed crop latent precision ablation

A codec's latent quantization changes the distribution that the diffusion
model learns. Comparing standardized latent MSE between different precision
arms is therefore not a meaningful cross-arm quality ranking.

This controlled experiment trains float and selected fixed-width latent
precision variants for each independent initialization seed. Within a seed
the codec and denoiser initializations, training batch indices, diffusion
timestep/noise/dropout traces and optimizer settings must match. Across all
seeds, the synthetic source data and held-out/generated noise panels are
identical. Each arm retains its own train-only latent calibration.

```bash
python -m research.crop_latent_replicates --model-seeds 17,29,41 \
  --precision-bits 4,8 --codec-steps 32 --diffusion-updates 64 \
  --width 32 --schedule-steps 32 --reverse-steps 8
```

Cross-arm differences compare waveform-domain codec reconstruction and
teacher-forced decoded x0, matched by source category, source seed position
and diffusion timestep. The generated-signal comparison matches requested
category, Gaussian seed and guidance strength. It measures raw waveform
statistics and explicit native latent projection, not semantic quality.
Each per-seed difference is averaged across paired cases before independent
seed summaries are computed. No significance, confidence interval or
winner is declared.

All precision arms retain their individual within-latent diagnostics but
the cross-arm summary never subtracts those velocity errors. The experiment
rejects changes in source hashes, training trace, seed list, model
initialization, evaluation panel, or null control. Numeric reports may be
serialized to JSON and revalidated. No samples, weights or private data
are published. Current inputs are Sonumio's own synthetic DSP clips;
this is not a real-corpus or Sonumio-1 benchmark.
