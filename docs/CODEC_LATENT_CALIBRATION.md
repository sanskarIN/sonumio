# Paired codec and latent calibration study

The open-set source probe rejected every latent generator output in the
initial two-seed study, while accepting the same synthetic references as the
previous probe. This motivates measuring the codec's reconstruction ceiling
separately from denoising and generation. The current study makes no claim
that the codec is the sole cause.

## Architecture and calibration

The native legacy codec compresses a mono 8192-sample waveform to 16 channels
by 128 frames, which is a 64-fold reduction of the temporal length but only
a fourfold reduction in the number of scalar values. There is no quantizer,
entropy coder, compressed audio file or measured bitrate.

A second research codec preserves this latent shape. Its encoder uses
three stride-four downsampling blocks with residual convolutional units.
The decoder uses explicit nearest-neighbor interpolation, stride-one
convolutions and residual units. Its output is bounded by tanh. This tests
a different upsampling design, not an equal-parameter comparison. Both
codecs use the same reconstruction objective and paired training minibatches.

Channel-wise affine calibration fits means and population standard deviations
over the 42 unique training recordings and their temporal positions only.
Each scale is floored at 0.001 to avoid division by nearly zero. The denoiser
can then operate in normalized coordinates, while a frozen adapter restores
native codec coordinates before decoding. The original codec weights are not
changed by this adapter. Held-out distribution shifts remain visible, and
calibration does not imply independence or Gaussianity of latent channels.

## Three paired arms

| Arm | Encoder/decoder | Diffusion coordinates |
| --- | --- | --- |
| legacy_native | Legacy codec | Native |
| legacy_calibrated | Identical frozen legacy codec | Channel calibrated |
| residual_calibrated | Residual resize-convolution codec | Channel calibrated |

The denoisers have identical starting weights and see identical caption indices,
noise tensors, timesteps and conditional dropout masks. The codec minibatches
are also paired. Different codec parameter counts and learned representations
mean this is not a controlled comparison of parameter efficiency.

## Evaluation

- Check exact training, validation and reference waveform hash separation.
- Report channel moments, scale floors and large normalized latent values.
- Measure reconstruction MAE, relative squared error, waveform cosine,
  multiresolution log-magnitude error, RMS ratio, clipping and DC bias on
  42 unique held-out recordings.
- Reconstruct each of the 18 independent source-probe holdout originals from
  its encoded latent. Compare the decoded waveform's source match with the
  original's match. This is an oracle reconstruction diagnostic, not generation.
- Evaluate the trained denoiser on 84 held-out caption rows at three timesteps.
  Bilingual pairs share corruption noise. Velocity MSE in different
  representations is not directly comparable.
- If the frozen open-set reference gate passes, sample real DDIM waveforms
  with matched, rotated and unconditional captions and compare source matches.
  A passing reference gate does not establish general real-world sound identity.

No data, model weights or generated audio are saved. A green CI verifies
experiment execution and controls, not a positive research result.

## Run

```bash
python -m pip install -e '.[research,test]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m research.codec_calibration_study --seed 17 --codec-steps 64 --train-steps 64 --evaluations 8 --sampling-seeds 301 302
```

Repeat with seed 29. All output is numeric JSON. Both runs are included in the
dedicated CPU workflow.

## Background literature

- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752):
  a precedent for separating a learned representation from diffusion. It
  concerns images, not proof that this audio calibration improves generation.
- Pons et al., [Upsampling artifacts in neural audio synthesis](https://arxiv.org/abs/2010.14356):
  motivation for testing alternative upsampling operators rather than assuming
  transposed convolution is artifact-free. Interpolation can also introduce
  spectral artifacts.
- Zeghidour et al., [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312):
  a reference for learned audio compression using quantized representations.
  This project experiment does not implement SoundStream, its quantization,
  training scale or bitrate/quality claims.

The next decision should be based on actual reconstruction and free-generation
results, rather than adopting the residual model merely because it is larger.
