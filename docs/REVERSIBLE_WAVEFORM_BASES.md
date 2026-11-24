# Reversible waveform coordinates

The previous compressed latent experiment is motivated by reducing model
cost, but loss of source structure at the codec boundary can prevent a
generative model from recovering it. This experiment changes the coordinate
basis of the existing multiscale waveform denoiser without compression.

Each eight-sample block is represented in one of three fixed bases:
identity (raw sample packing), orthonormal DCT-II, or orthonormal Haar.
All eight coefficients are retained, and synthesis uses the transpose
of the analysis matrix. This is an invertible patch-coordinate change,
not a learned audio codec or a bitrate reduction. The original waveform
noise, velocity objective, spectral loss and DDIM sampler are unchanged.

## Controlling initial functions

All arms first receive the same random trainable parameters. For basis
B and physical samples x, analysis gives y = Bx. The input convolution is
rotated by W B^T, and the final convolution and bias by B V and B b.
Consequently the initial waveform-domain mapping is preserved up to
floating-point roundoff. The study checks this before training using
shared held-out corruptions at low and terminal noise levels.
The fixed basis is reported with its own SHA-256 digest; the learnable
state digest excludes that deterministic nonpersistent buffer.

The comparison uses equal trainable parameter counts and identical
training indices, caption dropout, timesteps and noise. AdamW is not
invariant to arbitrary orthogonal reparameterizations, so subsequent
training trajectories can differ even when initial functions match.
The comparison studies that interaction, not merely initialization.

## Run the experiment

```bash
python -m research.reversible_basis_study --seed 17 --train-steps 96 --evaluations 8 --sampling-seeds 301 302
```

The CPU CI workflow runs two initializations with all three bases.
Reports include held-out raw velocity and clean-waveform errors, matched
versus disjoint/null conditioning, independent open-set source probe
results, exact 720-label-assignment panels, and held-out caption wording
controls. Generated panels run only when the reference probe passes.
The study uses synthetic audio, no pretrained model or external corpus.
No weights or audio are saved. Results do not establish general prompt
understanding, perceptual audio quality or Sonumio-1 readiness.
