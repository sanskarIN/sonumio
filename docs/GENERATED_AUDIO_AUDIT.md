# Numerical audit of actual neural audio generation (draft research)

Sonumio's research DDPMs were previously assessed only by **noise-prediction
error**. The new isolated generation audit evaluates **waveforms actually
sampled from trained models**, while retaining strict limits on what these
numbers can establish.

**Not Sonumio-1. Not a production model, checkpoint, release, or perceptual
audio-quality benchmark.** No weights, WAV files, embeddings or datasets are
uploaded or committed by this tool.

## What it does

The [training workflow](EXTENDED_SYNTHETIC_TRAINING.md) now calls the
replicated [architecture study](MULTI_SEED_STUDY.md) with
\`--audit-generated-audio\` enabled. The raw model's existing standalone
2,048-frame sampling CLI was never appropriate for a raw denoiser trained on
8,192 frames. An explicit **one-second raw DDPM sampler** was implemented
separately for this research protocol. The latent model's existing one-second
sampler is reused.

For each model initialization and one designated independent sample-noise
seed, it generates these fixed minimal prompt controls:

| Audio class | English prompt | Hindi prompt |
| --- | --- | --- |
| Rain | rain | बारिश |
| Wind | wind | हवा |

The same random Gaussian starting field is used for each prompt *within
each model arm*, so differences between English/Hindi or between rain/wind
are not caused by arbitrary changes in the initial random field. The raw
and latent models still have **different tensor spaces and random fields**,
so waveform-level comparison between models is *not* paired.

For each generated waveform, the code checks:

- Shape **exactly 8,192 mono frames at 8 kHz**, with finite float samples
  within the allowed [-1, 1] range.
- RMS, peak amplitude, clipping fraction and near-silence fraction.
- Spectral centroid, low-frequency (below 300 Hz) energy share, and
  spectral flatness using a bounded one-second Hann-windowed FFT.
- The SHA256 of the in-memory samples, for numerical reproducibility.
- Same-noise-seed mean absolute waveform difference and cosine similarity
  when changing rain ↔ wind or English ↔ Hindi prompt strings.

The report also includes these diagnostics for held-out Sonumio-owned DSP
reference clips (seed 101). Reference statistics only provide basic
numerical context. The audit does **not** claim a distributional distance,
FAD, CLAP, trained sound classifier, listening preference, or
matched-acoustic-pair quality measurement.

The summary across training initializations uses *within-arm* means and
observed variability. Audit configuration and report consistency are checked
before results from independent runs can be summarized. Any mixed audited/
unaudited run set or alleged saved weight/audio artifact causes failure.

## Run in development

~~~bash
python -m pip install -e '.[research,test]'

python -m research.multi_seed_study \
  --seeds 17,29,41 \
  --codec-steps 128 --denoiser-steps 256 \
  --schedule-steps 8 --evaluation-combinations 21 \
  --audit-generated-audio \
  --output checkpoints/private-research/generation-audit.json

python -m pytest -q tests/test_generation_audit.py
~~~

On PowerShell place arguments on a single line or use backtick
continuations instead of Bash's backslash syntax.

No standalone trained weights need to be downloaded because the audit
evaluates both models **immediately before in-memory weights are discarded**.
The only written output is the optional local metrics JSON inside the
ignored checkpoints folder. CI asserts this invariant.

## Critical limitations

A numerical demonstration that two prompts produce **different** audio
does **not** demonstrate that either audio actually sounds like the
corresponding prompt. A high raw waveform amplitude or spectral match
to procedural DSP does not imply realism or musical quality. A low
caption-MSE signal does not prove general text understanding.

Independent human listening, phonetic intelligibility (for future
voice models), blind sound-class recognition, rights-audited held-out
audio, memorization checks, and stronger sampling/quality comparisons
remain required before any Sonumio model-generation or product claims.

**No release, pre-release, tag, PR merge or model-weight publication
is authorized by successful tests or an apparently good numerical score.**


## Verified generation pilot and sampling-prior warning

The first [numerical generated-audio pilot report](GENERATED_AUDIO_PILOT_RESULTS.md)
documents the *measured* 3-seed results: severe raw-branch clipping,
small latent prompt-change differences and no listener-quality
certification. The audit now records the final diffusion cumulative
alpha and terminal clean/noise coefficients, flagging the potential
sampling-prior mismatch. This is a **hypothesis requiring controlled
experiments**, not an automatic explanation of every waveform defect.
