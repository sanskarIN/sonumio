# Native 24 kHz codec high-band objective ablation

This independent research experiment retains the original 24 kHz synthetic
DSP source rather than downsampling it to 8 kHz. It does not alter the
existing archive, 8 kHz neural models, CLI or deployed synthesis. It is
an autoencoder reconstruction experiment, not a text-to-audio generator.

## Geometry and numerical objectives

Each mono crop has 12,288 frames (512 ms). A reversible 3-phase
rearrangement maps consecutive samples to three channels on an 8 kHz
internal clock; unlike decimation it retains all sample values. Learned
stride-4 convolution at three stages gives a total stride of 192 on the
original sample clock. The latent has 16 channels and 64 time steps:
1,024 float32 values, a 12:1 scalar-count reduction and 125 time steps/s.
This is not a compressed byte stream, entropy coding, or measured rate
distortion; 4,096 latent bytes exclude parameters and metadata.

The paired arms share model architecture, exact initialization, batches,
AdamW settings and training steps. The only change is an extra weighted
high-band magnitude-error term (.5 vs zero) on the native sample clock.
Both arms use waveform MAE, log magnitude MAE, and normalized spectral
convergence. Magnitude diagnostics use 256/512/1024 FFTs, quarter-hop,
Hann windows without implicit padding and one-sided energy weighting.
The high-band starts at 4.5 kHz, beyond the 8 kHz Nyquist limit. Its
error is normalized by total source spectral energy, not by potentially
zero high-band energy. Signal and reconstruction high-band energy
fractions are separate. This is magnitude, not phase fidelity.

A deterministic 24 -> 8 -> 24 kHz Kaiser-window polyphase resample is
an explicitly bandwidth-limited DSP control. Silence is another control.
Neither is a matched learned codec or a fair bitrate comparison. The
7 kHz tone regression checks removal of high-frequency energy, whereas
the 1 kHz tone regression checks low-band preservation.

## Run the study

```bash
python -m research.native_codec_study --model-seeds 17,29,41 \
  --training-seeds 0,1,2 --validation-seeds 101,102 \
  --updates 32 --batch-size 4
```

Event-aware windows are derived from the existing synthetic recipe but
rendered at native rate and extended to 512 ms. Exact PCM overlap checks
reject duplicates within and across splits. Distinct seeds or offsets
are not evidence of independent real-world recordings. The study records
source digests, initialization and batch traces, held-out per-category
averages, model fingerprints and paired seed differences. A negative
highband-minus-base delta means a lower numeric error, not necessarily
better perceived quality. For SNR, larger is better. Source-band fraction
should be invariant between arms. Three initializations are descriptive.

The study uses CPU float32 model training, bounded source sets, at most
256 updates and 8,192 training examples across all arms. No private
recordings, checkpoints or waveform files are written. CI runs both a
small regression training and the prespecified three-seed pilot. The
pilot JSON is logged on the CI runner; its outcomes need separate review.
No original Sonumio-1 weights, perceptual fidelity, semantic alignment,
or generation quality are established.

Remaining steps: inspect the measured results, evaluate real rights-reviewed
24 kHz clips and broader held-out sources, compare capacity/compute
matched codec alternatives, and validate a separately versioned quantized
packet format before attempting longer native-band latent diffusion.


## Numerical correction from first CI run

At commit a676f03749372c29ae02c22f7195f5454b2f0eb7,
[the dedicated CPU job](https://github.com/sanskarIN/sonumio/actions/runs/45803426585)
passed 18 tests and failed one: identical silent waveforms returned loss
4.729372449219227e-08 instead of zero (tolerance 1e-8). The training pilot
was skipped, so no three-seed outcome exists at that revision.

The source of the problem is subtraction of two separately rounded square
roots in the smoothed residual norm. The implementation now uses the
algebraically equivalent rationalized expression

    E / (sqrt(E + epsilon) + sqrt(epsilon))

where E is nonnegative spectral residual energy and epsilon is 1e-8.
The tensor epsilon is explicitly created in the input dtype. This produces
exact zero for E=0 without subtractive cancellation and retains a finite
derivative at zero. The original test tolerance is unchanged, and extra
tests cover tiny positive energies, monotonicity, a float64 reference,
finite derivatives and exact waveform identity at several amplitudes.
The corrected-head hosted CI and training pilot remain separate gates.
