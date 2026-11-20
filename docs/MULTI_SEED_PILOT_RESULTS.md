# Pilot results: three-seed shared-data audio DDPM study

**Research record — 16 November 2026. NOT Sonumio-1, not a release,
and not evidence of perceptually good or prompt-reliable audio.**

Source: [GitHub Actions CPU research run 43846262743](https://github.com/sanskarIN/sonumio/actions/runs/43846262743)
at development commit [915dc9c7](https://github.com/sanskarIN/sonumio/commit/915dc9c73701d260c501e8f3e018af56b1c24cfa).

This report is transcribed from the **successfully executed** JSON diagnostic
in CI, not a forecast or claim about future Sonumio model generations.
All neural weights were trained in memory and **not saved/published**.

## Fixed configuration

| Field | Recorded value |
| --- | --- |
| Benchmark protocol | raw-vs-latent-multi-seed-diagnostics-v2 |
| Single-run protocol | shared-data-raw-vs-latent-text-ddpm-v2 |
| Initialization seeds | 17, 29, 41 (3 repetitions) |
| Codec update count per seed | 24 |
| Denoiser updates per arm/seed | 48 |
| DDPM schedule | 8 steps, text conditioning |
| Nominal denoiser width | 16 channels |
| Training sample count | 42 distinct audio recordings, 84 bilingual captions |
| Validation dataset size | 42 distinct recordings, 84 bilingual captions |
| Evaluated sound combinations | 4 per run, using both held-out recording seeds |
| Audio duration | 1.024 seconds, 8 kHz mono |
| Execution | Python 3.11.18, PyTorch 2.15.0+cpu, NumPy 2.4.6, 2 CPU threads |
| Configuration SHA256 | 1162d7dbb8779460716a37bdcc102c8f5cccd8f22b0d9299cfa5d5dddbadeb6a |
| Hard-negative token SHA256 | 8b938ddffe2cbefc5323204a611983f1088668716d328101af5dbba319385649 |
| Research tests in run | 130 passed |

The corpus source, recordings and prompt templates are identical across all
three runs. Only the model/training random initialization changes. Therefore,
this is an **initialization sensitivity** check, not three independent
real-world audio datasets or statistically generalizable listening studies.

## Denoising better than a zero-noise predictor

Each value below is the **mean within-arm relative improvement fraction**:
\`1 − paired noise MSE / zero-predictor noise MSE\`, converted to percent.
This describes noise prediction on **different representations**; do not
directly compare the raw and latent columns as an audio-quality ranking.

| Noise level | Raw waveform: mean | Latent: mean | Seeds with positive improvement |
| --- | ---: | ---: | --- |
| Low | 50.89% | 1.89% | 3/3 for both |
| Medium | 72.53% | 3.18% | 3/3 for both |
| High | 76.25% | 3.41% | 3/3 for both |

These results show that 48 denoiser updates established a nonzero
**noise-prediction training signal** compared with the within-domain
zero-noise baseline. They do not imply audible naturalness or superiority
of one audio representation over the other. Higher percent improvement
for raw here is not an acceptable model-selection criterion.

## Category-disjoint caption test

The v2 protocol replaced ambiguous circular-caption rotations with
counterfactual captions that preserve one-/two-source count and language
while sharing **zero true sound categories** with the recording.
Here, positive \`wrong-caption MSE − correct-caption MSE\` means the matched
description has lower noise-prediction error *within that one arm*.

| Noise level | Raw mean MSE advantage | Raw positive seeds | Latent mean MSE advantage | Latent positive seeds |
| --- | ---: | ---: | ---: | ---: |
| Low | -0.001810 | 1/3 | -0.000490 | 1/3 |
| Medium | -0.000184 | 2/3 | +0.000501 | 3/3 |
| High | +0.000065 | 1/3 | +0.000530 | 3/3 |

Signs fluctuate across noise levels and seeds, and the positive latent
medium-/high-noise advantages are small. The three outcomes are correlated
by a common narrow synthetic dataset. There is **no conclusive prompt
alignment** claim, no statistical population interval and no basis to say
that one model produces better-sounding audio.

## Codec reconstruction sanity check

The trainable latent codec's held-out **waveform MAE** over the selected
validation recordings had mean **0.149738**, median **0.106852**, observed
range **0.081191–0.261171**, and sample SD **0.097353** across the three
initializations. The codec was trained for only **24 optimizer updates**.
These are not listening scores and are not evidence of high fidelity.

## Honest limitations and next experiment

1. **Capacity/FLOP mismatch.** Raw denoiser, latent denoiser and latent
   codec differ in model parameter count, compute and receptive field.
2. **Tiny training budget.** 48 denoiser updates and an eight-step schedule
   do not establish convergence or useful generative capabilities.
3. **Synthetic-only data.** Evaluations use six known DSP sound families
   and templated English/Hindi captions rather than rights-reviewed
   real-world recordings, novel sessions or diverse utterances.
4. **Denoising is not generation.** The experiment does not sample audio
   for blind human listening, FAD, CLAP or memorization analysis.
5. **Three seeds are exploratory.** This is descriptive observed variability,
   not a general population confidence interval or a preregistered benchmark.

Before further scaling, expand to verified licensed diverse non-voice
recordings, multi-seed convergence studies with matched compute budgets,
codecs at higher sample rates, independent blind listening and privacy/
rights audits. **No promotion, merge, weight publication, tag, pre-release
or stable release follows from these results.**
