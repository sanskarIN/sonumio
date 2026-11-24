# Multiscale waveform diffusion research

The earlier raw waveform velocity experiment produced source matches at the
same rate as zero guidance. This study tests a wider temporal receptive
field and a separately controlled clean-signal spectral objective.

## Model

Input and output remain 8192 mono samples at 8 kHz. Reversible sample packing
rearranges eight adjacent samples into eight channels at 1024 positions.
Packing itself loses no samples and is not a learned audio codec. The
network then processes 1024, 256 and 64 positions with conditioned residual
blocks, skip connections and a low-rate self-attention bottleneck. A learned
UTF-8 text encoder and timestep embedding condition each resolution. The
decoder reconstructs the original waveform shape via unpacking. Existing
model checkpoints and production inference defaults are not changed.

The optional auxiliary loss evaluates log(1 + magnitude) STFT distance
between the actual clean waveform and the clean waveform estimate recovered
from the predicted velocity. FFT sizes are 128, 512 and 1024 with quarter-hop
Hann windows. Each per-example spectral error is multiplied by
(1 - alpha_bar_t); ordinary velocity MSE remains unchanged. This puts more
auxiliary emphasis on higher noise levels without replacing the trained
velocity parameterization. Magnitude loss does not guarantee phase fidelity
or perceptual similarity.

## Experimental controls

Three arms share training recording order, noise, timesteps, classifier-free
dropout and minibatch sizes:

- Legacy 16-channel waveform velocity model.
- Multiscale waveform velocity model.
- Identically initialized multiscale model with spectral auxiliary weight 0.2.

The last two arms isolate the objective change within one architecture.
Legacy versus multiscale is not an equal-parameter or equal-compute
comparison. All arms train on the same 42 unique first-party synthetic
recordings and 84 bilingual caption rows. Held-out teacher-forced metrics
are reported at three timesteps on 42 other recordings. English/Hindi pairs
share corruption noise. The independent open-set probe is frozen; a failed
reference gate prevents generation scoring.

Free DDIM generation uses paired sampling seeds and compares correct
captions, rotated labels and unconditional predictions. The experiment
records unclamped RMS and out-of-range samples as well as source acceptance
and source rank. It does not claim that a source match establishes real-world
sound identity.

## Run

```bash
python -m pip install -e '.[research,test]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m research.multiscale_waveform_study --seed 17 --train-steps 96 --evaluations 8 --sampling-seeds 301 302 --spectral-weight 0.2
```

Repeat with seed 29. The CPU workflow executes both configurations and
regression tests. Training and sampling are in-memory only. No model weights,
audio files, external datasets or model releases are produced. Passing CI
establishes correct execution and paired controls, not an improvement in
prompt grounding or Sonumio-1 readiness.

## Held-out wording diagnostic

For each arm, the first sampling seed also generates all six categories
with a fixed alternative caption per language. Token identities are checked
against every synthetic training template (both variants, both languages,
single and mixed categories). The alternative wording is disjoint from these
training templates; the underlying sound vocabulary and synthetic source
recipes are still shared. This is not an open-vocabulary or human-language
understanding benchmark.

The same Gaussian initial noise is used for template and alternative captions.
The panel records joint correct accepted source matches, template-only and
alternative-only matches, exact balanced-label assignment distributions,
distinct accepted source counts, and same-category versus different-category
waveform differences. The latter can reveal numerical sensitivity or collapse,
but do not establish semantic fidelity. Zero-guidance generation must remain
identical even when the prompt changes from a training template to a held-out
alternative. No generated audio is persisted.
