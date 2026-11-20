# One-second text-conditioned latent audio — isolated research prototype

**Status: experimental, draft-only, not a Sonumio model generation. No releases,
pre-releases, tags or model-weight publication are authorized.**

This isolated CPU-oriented experiment tests whether an original, trainable
convolutional waveform codec combined with a learned byte-text-conditioned
latent diffusion network is a useful direction **beyond Sonumio's original
0.256-second raw waveform DDPM**.

It generates **1.024-second mono clips at 8 kHz**. One second is not long-form
audio, broadcast quality, streaming, speech synthesis, music generation or a
general text-to-sound product. It does not run in the production/local API or
React audio studio.

## Architecture

| Component | Current experiment |
| --- | --- |
| Audio input | One 8 kHz mono clip, 8,192 float32 frames |
| Trainable codec | Three strided Conv1d stages (4 × 4 × 4), plus matching transpose-convolution decoder |
| Latent representation | 16 channels × 128 frames; 64× temporal reduction, **not** 64× file-size compression |
| Codec objective | Waveform L1 plus three log-magnitude STFT losses (FFT sizes 128, 512, 1024) |
| Text | Existing bilingual UTF-8 byte tokenizer and learned Transformer encoder |
| Latent denoiser | Five conditioned, dilated residual 1D convolution blocks |
| Diffusion | Original discrete DDPM, classifier-free guidance, bounded 8–256 timesteps |
| Checkpoints | Private numeric-only NPZ + JSON and SHA256 with archive header/tensor-size checks |
| Dataset | Six Sonumio procedural sound categories and their 15 two-source combinations; English/Hindi synthetic captions |

No pretrained encoder, neural codec, speech recognizer, vocoder, text embedding
API, outside recording or third-party weight is included or downloaded.

## Training protocol

1. DSP audio is generated for 1.5 seconds at 24 kHz and downsampled to 8 kHz.
   A deterministic **seed-dependent crop** preserves a **1.024-second** waveform
   for each source combination. This is necessary because the uncropped
   ambient synthesizer ignores seeds and otherwise duplicates evaluation audio.
2. A codec learns reconstructions using waveform and multi-resolution spectral
   objectives. The encoder is then frozen for the next stage.
3. The DDPM learns to predict noise on encoded latents, conditioned on
   learned embeddings of short English/Hindi captions.
4. Separate training and validation seeds yield separate data fingerprints.
   The initial synthetic run contains **84 train pairs** and **84 held-out pairs**.
   Each underlying DSP recording is intentionally reused once for English and
   once for Hindi.
5. A report records reconstruction loss, held-out noise MSE, parameters,
   data fingerprints, training steps and runtime. None of those scores alone
   demonstrate intelligible, natural, aesthetically pleasing or safe audio.

A tiny training run is an **execution smoke test**, not proof of convergence.

## Running locally

From the repository root, after installing Python 3.11+:

~~~bash
python -m pip install -e '.[research,test]'

python -m research.latent_audio train \
  --ae-steps 10 --diffusion-steps 10 --schedule-steps 8 \
  --channels 16 --seed 7 \
  --output checkpoints/private-latent-experiment

python -m research.latent_audio sample \
  --checkpoint checkpoints/private-latent-experiment \
  --prompt 'बारिश और हवा' --seed 20 --guidance 1.0 \
  --output outputs/latent-experiment.wav

python -m pytest -q tests/test_latent_audio.py tests/test_latent_alignment.py
~~~

On Windows PowerShell the commands work on single lines without the backslash
continuations. All generated checkpoints and audio are local only. No release
artifacts or trained weights are committed to Git.

## Hard-negative text alignment diagnostic

~~~bash
python -m research.latent_alignment \
  --checkpoint checkpoints/private-latent-experiment
~~~

The separate caption benchmark uses held-out DSP seeds 1001/1002 and three
fixed noise levels. For each underlying clip, the correct description competes
with **one-source-replaced** English/Hindi descriptions of the same source
count, in training-template and held-out-wording styles. All candidate captions
see precisely the same noisy latent, timestep and target noise. An
unconditioned/blank caption acts as an additional reference.

- Pairwise accuracy assigns half credit to ties.
- Top-1 retrieval splits ties equally; the uniform chance baseline is
  **16.7%** for one sound and **11.1%** for two-source mixtures.
- Mean noise-MSE advantage is the average wrong-caption loss **minus**
  matched-caption loss. A positive advantage favors correct text.
- A caption-blind denoiser is regression-tested to score at chance.
- Evaluation seeds are disjoint from the training and validation seed sets.

These observations still use the same six procedural source families and a
narrow set of template-derived words. They do **not** establish real-world
semantic understanding, open-vocabulary generation, acoustic fidelity or
memorization safety. Independent listening tests and rights-reviewed
recordings are required.

## Research gates before expanding the experiment

- Test codec reconstructions with independent listening and multi-resolution
  spectral comparisons; check silence, transients and clipping.
- Benchmark truly held-out sessions, novel wording, novel microphones and
  new recording environments under human-confirmed license/consent review.
- Compare a matched-parameter raw waveform baseline against the latent
  approach without conflating different sample rates or datasets.
- Profile CPU/GPU throughput, VRAM, checkpoint size and end-to-end real-time
  factor before increasing clip duration or sample rate.
- Add provenance/memorization studies and auditable quality controls before
  claiming any original Sonumio model generation.

**No trained Sonumio-1 model exists. Keep research checkpoints private.**


## Objective reconstruction and model-health evaluation

~~~bash
python -m research.latent_quality \
  --checkpoint checkpoints/private-latent-experiment \
  --bootstrap-iterations 200 \
  --output checkpoints/private-latent-experiment/quality.json

python -m pytest -q tests/test_latent_quality.py
~~~

This command loads a **private** numeric-only checkpoint. It evaluates the
**codec reconstruction**, *not* DDPM generation quality, on all 42 unique
recordings from validation seeds 101/102, across 6 single categories and
15 pairs. Each recording is counted **once**, although the text dataset
intentionally pairs it with separate English and Hindi captions.

A new mandatory exact-waveform collision check compares validation bytes with
the fixed training seeds 0/1, and rejects duplicate evaluation recordings.
This catches accidental duplicate waveforms, but does not prevent semantically
similar or related DSP content across splits.

### Metrics and controls

| Statistic | Meaning | Better |
| --- | --- | --- |
| Waveform MAE/RMSE | Sample-aligned absolute/squared errors | Lower |
| Log-spectral L1 | Mean difference of log1p magnitude STFT at 128/512/1024 bins | Lower |
| Spectral convergence | Normalized Frobenius norm of magnitude difference | Lower |
| Signal-to-noise ratio | Paired waveform energy/error ratio in dB; **not** scale-invariant | Higher |

The codec is directly compared with a **silent waveform**, an **unchanged
identity waveform** (perfect-reconstruction ceiling), and a fixed **9-tap
moving-average DSP** control. Those are sanity checks, **not a matched-parameter
raw waveform DDPM or a state-of-the-art audio model**.

Metrics are reported overall, by number of sound sources, and by category
membership (some two-source clips appear in multiple category breakouts).
A seeded bootstrap resamples the 21 source-combination groups, instead of
incorrectly treating paired English/Hindi captions as independent clips.
Its exploratory 95% interval **does not imply generalization outside the
tiny shared procedural audio family** and is not a population-confidence
claim. No significance threshold automatically triggers promotion or release.

The same command checks DDPM-generated audio only for **finite samples,
clipping, RMS, peaks, duration and deterministic wave hashes**. It does not
evaluate real audible quality or whether a prompt matches what is heard.
Independent listening tests remain a separate required gate.

The training report also now evaluates **all 42 unique validation waveforms**
for codec reconstruction and **all 84 validation caption pairs** for
noise-prediction loss (in bounded batches). Previously it reported just the
first English/Hindi pair of a single recording as its held-out result.
This was misleading about validation coverage; older reports should not be
compared as though they used the corrected protocol.
