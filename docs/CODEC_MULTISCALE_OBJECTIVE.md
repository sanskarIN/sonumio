# Sonumio original codec objective research

**CPU-only experimental waveform reconstruction. No released neural weights, music/speech model or perceptual-quality claim.**

## Why investigate the codec

The current Sonumio raw and latent research branches use an original tiny
8-kHz mono neural autoencoder that compresses the time axis by 64×. The
existing codec training objective combines waveform L1 and log-magnitude
STFT across three FFT sizes. That objective is valid, but sample L1
alone cannot distinguish the particular cost of missing transients,
deleting high-frequency energy, or flattening the amplitude envelope.

This research adds an **explicit, inspectable differentiable loss** to
investigate those differences. It does not silently replace any legacy
training, change model checkpoint compatibility or declare that any
loss improves audible quality.

## Eight separate differentiable signal losses

`research/codec_objectives.py` computes all components on exact
sample-aligned 8192-frame float32 waveforms (8 kHz mono):

1. `waveform_l1`: mean absolute sample error.
2. `preemphasis_l1`: mean error after `y[t] - k*y[t-1]`, making
   rapid changes more visible.
3. `envelope_rms_l1`: framewise RMS mismatch at 64, 256 and 1024
   samples (roughly 8, 32 and 128 ms).
4. `log_magnitude_l1`: compressed log(1 + STFT magnitude) mismatch.
5. `linear_magnitude_l1`: absolute STFT magnitude error.
6. `spectral_convergence`: spectral error norm / source spectral norm
   (epsilon guarded for silence).
7. `spectral_flux_l1`: absolute mismatch of frame-to-frame changes
   in log magnitude.
8. `highband_energy_ratio_l1`: mismatch in relative spectral power
   above a selected high-frequency threshold.

STFT resolutions are selected from 256, 512, 1024 and 2048 frames
with Hann windows and quarter-window hops. High-frequency spectral
weights can be set explicitly but are bounded, as are all objective
weights. The sample rate and highband cutoff are audited.

The loss accepts only finite, aligned CPU float32 [-1,1] recordings,
batch sizes 1–8. Invalid FFT configurations, NaN/Inf weights, nonfinite
sample values or unsupported shapes are rejected. No auto-normalization,
file writing or external embeddings are performed.

The component decomposition is returned as real differentiable
`torch.Tensor` scalars when requested; do not convert to Python
floats before backpropagation. The separate numeric audit returns
sample MAE/RMSE, finite-epsilon SNR and explicit component values.

```python
from research.codec_objectives import (
    CodecObjectiveWeights, CodecSpectralConfig,
    weighted_codec_loss,
)

loss, metrics = weighted_codec_loss(
    clean, codec(clean), return_components=True,
    weights=CodecObjectiveWeights(),
    config=CodecSpectralConfig(),
)
loss.backward()
```

## Matched training study, not a marketing comparison

`research/codec_loss_study.py` initializes **two independent
TinyAudioAutoencoder networks with identical weights**. Both models
receive exactly the same first-party DSP training minibatches,
optimizer, steps, learning rate and gradient norm clipping.

Only the training objective changes:

- Original: existing `research.latent_audio.reconstruction_loss`.
- Candidate: explicit multiresolution waveform/transient/spectral loss.

Train and held-out examples use separate DSP recording seeds.
English and Hindi captions for one waveform are **not counted twice**;
the study trains on unique recordings and evaluates all held-out
recordings only once. The 21 sound-combination families remain related;
source-group descriptive differences are not a population-confidence
interval.

Each held-out neural reconstruction is scored using exactly the
**same new objective components** for both training arms. A mean
candidate-minus-original delta below zero indicates lower numerical
reconstruction error on that component. It is **not a listener
preference**, not a semantic alignment score, not an original sound
generation test, and not evidence that the codec has converged.
Different training loss totals are not directly comparable.

The study checks per-model source-independent weights, source WAV
hashes, exact train/held-out separation, model weight SHA256 before
and after held-out evaluation, and standard torch RNG restoration.

## Reproduce

```bash
python -m pytest -q tests/test_codec_objectives.py \
  tests/test_codec_loss_study.py

python -m research.codec_loss_study \
  --model-seed 17 --updates 2 --training-batch 2 \
  --learning-rate 0.001 \
  --training-seeds 0 --heldout-seeds 101
```

For more steps within the guarded CPU pilot:

```bash
python -m research.codec_loss_study \
  --model-seed 17 --updates 32 --training-batch 4 \
  --training-seeds 0,1 --heldout-seeds 101,102
```

The first compares 21 unique training and 21 held-out source
recordings (42 captions per split); the second compares 42 and 42
(84 captions per split). The CLI emits numeric-only JSON, not WAV,
weights, checkpoints, external model embeddings or other artifacts.
The default maximum 64 optimization steps is not enough to claim
convergence.

## Honest quality limitations

Spectral statistics can improve while **audible quality gets worse**:
an over-smoothed or tonally reduced signal can be numerically closer
on one loss yet sound muted or unnatural. Likewise, a brief training
pilot on Sonumio's six related synthetic source categories is not
a reliable evaluation of real speech, music, long-form environment
sounds, intelligibility, originality or rights provenance.

Before promoting the loss to a default model, independently replicate
seeds, train much longer with early stopping on group-disjoint
validation, compare numerical and blind-listening quality, measure
aliasing and sample-rate limitations, and verify licensed datasets.
No published Sonumio-1 weights, release or model generation follows
from these tests.
