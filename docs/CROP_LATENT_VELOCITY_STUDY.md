# Crop-scale latent velocity diffusion research

The crop codec and the denoiser form two separately trained stages.
The codec is trained on first-party synthetic 2048-sample crops and
then frozen. Its native latent is 16 channels by 32 frames, with
optional hard fixed-width quantization. A training-only per-channel
calibration uses mergeable Float64 moments, with a 0.01 scale floor
for constant or nearly constant channels.

```bash
python -m research.crop_latent_study --codec-steps 32 --diffusion-updates 64 \
  --bits 8 --width 32 --schedule-steps 32 --reverse-steps 8
```

The new class-conditioned denoiser uses a cosine diffusion schedule
and velocity prediction. Unlike epsilon division near a very noisy
terminal timestep, velocity inversion obtains x0 and epsilon directly
from the orthogonal transform. Every optimizer step uses recorded
seeded indices, noise, timesteps and classifier-free dropout.

The validation panel measures held-out velocity MSE for correct, null
and rotated wrong labels at three fixed timesteps, with a zero-predictor
control. It also compares teacher-forced decoded x0 against both the
original waveform and the frozen codec's own reconstruction ceiling.
These are not freely generated audio quality scores.

The generation panel uses deterministic velocity-DDIM and shared
initial noise across all six categories and guidance 0, 1 and 2.
The decoder requires bounded native latents, so restored values outside
[-1,1] are measured before explicit projection. Optional requantization
is measured separately. Reports expose decoder pre-tanh peak and
overrange, not merely tanh-capped output. Null guidance must produce
identical waveforms for different requested categories.

All experiments are bounded CPU diagnostics and keep weights and samples
in memory. They do not use the reviewed private corpus yet. The study
has no open-vocabulary language conditioning, long-form generation,
independent listening score, or original Sonumio-1 model claim.
