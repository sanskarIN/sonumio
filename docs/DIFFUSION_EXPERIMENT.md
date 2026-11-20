# Waveform diffusion experiment: research only

`research/wave_diffusion.py` is an original **trainable generative** implementation,
not a production model, and **not Sonumio-1**. It trains a 1D convolutional
conditional denoiser to estimate diffusion noise and then uses a small DDPM
reverse process to synthesize waveforms from random noise.

## What is implemented

- Six fixed semantic categories: rain, thunder, wind, forest, ocean, ambient.
- Synthetic training examples derived entirely from Sonumio's DSP generator.
- 8 kHz, 2,048-sample clips (0.256 seconds); 32 diffusion steps by default.
- A temporal sinusoidal encoding, class embedding and residual dilated 1D convolutions.
- Condition dropout during training and bounded classifier-free guidance for sampling.
- Deterministic held-out source seeds, gradient clipping, numerical-only NPZ checkpoints,
  SHA256 verification and strict tensor-name/shape/dtype validation.
- Tests cover the training gradient, output determinism, checkpoint tampering and
  an end-to-end two-step research training run.

## Commands

```sh
python -m pip install -e '.[research,test]'
python -m research.wave_diffusion train --steps 50 --seed 9 --output checkpoints/wave-diffusion
python -m research.wave_diffusion sample --checkpoint checkpoints/wave-diffusion --prompt rain --seed 10 --output outputs/research-wave.wav
```

The generated WAV is real neural-model output, but at this size and training
budget it is expected to sound poor and does **not** demonstrate useful text-to-audio
quality or generalization. Neither saved weights nor model cards have passed a
release gate and the checkpoints must not be published.

## Next scientifically meaningful work

1. Add robust data ingestion on manually rights-reviewed recordings rather than
   using only the current synthetic procedural sounds.
2. Evaluate longer audio using efficient representations or overlap-add decoding.
3. Use independent validation and listening evaluation, not only a training loss.
4. Compare a mel-spectrogram generator and a compact latent-audio model on the
   same reviewed datasets and measured compute limits.
5. Audit memorization and speech/voice misuse before any deployment or weight release.

The numerical training loss is noise-prediction MSE, not perceived audio fidelity.

## Local WAV training research

The optional rights-gated WAV training path is documented in
[DATASET_INGESTION.md](DATASET_INGESTION.md). Passing --manifest together
with --audio-root trains from declared, checksum-verified local PCM recordings.
It retains 0.256-second 8 kHz clips and single-category conditioning.
A human must independently verify rights and that recordings contain no voices.
No checkpoints are published.

Checkpoint loading additionally preflights the NPZ container before NumPy
allocates any tensor: exact expected member names (no duplicates), declared
uncompressed byte budgets, finite bounded model dimensions, and NumPy header
shapes/dtypes. A SHA-256 checksum alone is not a guarantee that a checkpoint
is safe to deserialize; **checkpoints remain local and unpublished**.
