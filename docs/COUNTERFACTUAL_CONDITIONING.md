# Counterfactual velocity conditioning study

This private synthetic-only research compares the existing raw and latent
velocity-denoising objective against a matched-caption emphasis term. It
addresses the previous generated-source-grounding pilot's failure to
outperform zero guidance. It does not claim to fix that failure.

## Method

For each training row with classifier-free conditioning retained, select
a caption naming a completely disjoint source set, preserving language,
recording seed/template variant and source count. Rotate through all
eligible negatives without consuming random numbers. Verify the token
ordering against the actual dataset before using the index table.

With MSE averaged per recording over channels and samples, the treatment is:

    L = L_velocity + weight * mean(max(0, MSE_matched - 0.95 * stopgrad(MSE_wrong)))

The auxiliary mean is over retained conditional rows only. Both predictions
receive the same corrupted audio, timestep and velocity target. Wrong-caption
prediction is computed without gradients; the penalty cannot directly reward
increasing wrong-caption error. Shared parameters can still change wrong-caption
predictions indirectly. Weight is bounded to [0, 2]; zero preserves the original
training objective. The default study treatment uses 0.5.

The comparison resets the initialization seed, verifies exact initialization
hashes and training-input trace hashes (codec minibatch indices, denoising
indices, timesteps, dropout masks and noise), and requires the frozen codecs to
be tensor-identical. Training and validation corpus fingerprints must agree.
This design isolates the objective change within a fixed budget and hardware
environment; it does not guarantee bitwise reproducibility across platforms.

## Evaluation

Teacher-forced diagnostics use all 42 held-out recordings (84 caption rows)
at three fixed timesteps: 0, 4, 7. English/Hindi pairs share the underlying
recording and corruption noise. Correct, wrong, and null captions use the
identical noise and timestep for each row. Results are reported separately by
language and one/two-source caption count, with absolute MSE, paired differences
and strict win counts. A larger wrong-minus-matched difference alone is not
improvement: inspect absolute matched MSE, wrong MSE and null MSE together.

The free-generation check runs both frozen model arms using the existing
DDIM sampler, fixed sampling noise and the original six-source spectral probe.
Its reference fit, calibration and holdout recordings are disjoint from both
training and teacher-forced validation. It reports matched, rotated and null
controls, including exclusion of categories accepted for noise. These reference
scores are limited synthetic proxies, not listener judgments. If the reference
gate fails, the free-generation check is skipped and its absence is explicit.
No weights or waveforms leave memory.

## Run

```bash
python -m pip install -e '.[research,test]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m research.counterfactual_conditioning_study \
  --seed 17 --train-steps 32 --codec-steps 16 --weight 0.5 \
  --evaluations 8 --sampling-seeds 301 302
```

Repeat with seed 29. The dedicated CPU workflow runs both configurations and
prints numeric metrics. A passing workflow verifies execution and experimental
controls, not an improvement in generation quality. No performance outcome is
assumed in advance. Do not pool dependent caption rows as independent
observations, and do not publish checkpoints or claim Sonumio-1 readiness.
