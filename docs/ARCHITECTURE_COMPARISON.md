# Controlled raw waveform vs latent DDPM research comparison

**Status: experimental, synthetic-only, private and unreleased.**
This implementation exists only in Sonumio's draft development PR. It is not
Sonumio-1, not an official quality ranking, and does not generate publishable
model weights or any release artifacts.

## Research question

What do the current original text-conditioned DDPM prototypes do when trained
on **exactly the same 1.024-second, 8-kHz mono synthetic audio/caption
examples**, with the same minibatch indices, optimizer-update count, model
nominal width and diffusion schedule?

Two research paths are compared:

| Property | Raw waveform arm | Latent arm |
| --- | --- | --- |
| Input waveform duration | 8,192 samples (1.024 s) | 8,192 samples (1.024 s) |
| DDPM training representation | 1 × 8,192 raw samples | 16 × 128 latent samples |
| Text conditioning | Existing UTF-8 byte Transformer | Same *architecture*, separate trained weights |
| Audio codec | None | Separate small trainable convolutional codec (64× time reduction) |
| Denoiser architecture | 6 dilated convolutional blocks | 5 dilated convolutional blocks |
| Training optimizer | AdamW, lr 0.0005 | AdamW, lr 0.0005 |
| Classifier-free dropout | 10% | 10% |
| Noise schedule | Same linear-beta DDPM | Same linear-beta DDPM |
| Train data and minibatch indices | Same DSP+English/Hindi pairs | Same pairs and indices |
| Model parameters | Reported after instantiation | Reported separately, not equal |
| Codec training expense | None | Reported separately |

**Equal update counts do not equal matched FLOPs or model capacity.** The
comparison records runtime on the current CPU, but a single short timing run
is not a statistically reliable performance comparison.

## Dataset integrity

Both branches load one Sonumio-owned synthetic dataset with six sound classes
and 15 two-source mixtures. Training recordings use seeds **0 and 1** (42
unique source/seed waveforms, 84 English/Hindi caption pairs). Evaluation
recordings use seeds **101 and 102** (42 unique source/seed waveforms, 84
captions). Evaluation selects 2..21 source combinations, counting both
languages and both validation seeds in the selected combinations.

The benchmark verifies both language captions point to the exact same
underlying recording, and rejects *exact-byte* duplicate waveforms within a
split or across train and validation. Different seed IDs alone are not accepted
as proof of independent waveforms. Even with no exact duplicates, the
recordings reuse the same DSP generators and do **not** represent a real-world
holdout.

## Measurement design

The codec is trained on all **unique** training waveforms only, with
reconstruction loss combining waveform L1 and multiresolution log-spectral
error. Its encoder is then frozen. Both denoisers learn noise prediction from
the same bilingual training pairs. Neither model's weights are saved by this
experiment.

Held-out **low-, medium-, and high-noise** checks use the same diffusion
timestep for each arm and the exact same noisy sample/target for each candidate
caption within that arm. The tested prompt conditions are:

- Correct English/Hindi caption (paired).
- Another source combination's caption (four-row circular shuffle).
- Blank/unconditional text.
- An all-zero noise predictor with a known numerical baseline.

For each noise level, the report records paired, shuffled, unconditional and
zero-predictor mean-squared errors; shuffled-minus-paired and
unconditional-minus-paired advantages; and improvement relative to the
zero-noise-prediction baseline.

**Raw MSE and latent MSE must never be compared numerically to declare a
winning audio architecture**: they are errors measured in *different
representations* with different tensor sizes, learned signal distributions
and receptive fields. Similarly, a positive caption advantage may reflect a
narrow synthetic template association, not natural-language understanding.

## Try it locally

Install dependencies from the repository root:

~~~bash
python -m pip install -e '.[research,test]'

# Tiny diagnostic. This does not train useful audio.
python -m research.architecture_compare \
  --codec-steps 2 --denoiser-steps 2 \
  --schedule-steps 8 --channels 16 \
  --evaluation-combinations 4 --seed 17 \
  --output checkpoints/private-architecture-compare/report.json

python -m pytest -q tests/test_architecture_compare.py
~~~

Windows PowerShell: enter each command on one line or use its backtick
continuation instead of the Unix backslash.

The report is a **JSON file only**; no checkpoint, neural weights or audio
clips are saved or uploaded. Its default path should be kept in the ignored
local `checkpoints/` directory. For more comprehensive experiments, increase
the number of steps and the number of evaluated source combinations
(2–21), then report **multiple seeds** and measured hardware resources.
Do not interpret a 1–2 step smoke test as convergence or audio quality.

## What this experiment cannot settle

- Neither independent human listening nor generated waveform similarity is
  evaluated here.
- The fixed codec introduces an information bottleneck that the raw branch
  does not have; neither path's trained-parameter count is matched.
- The same text encoder design is used but independently initialized, and
  each branch receives different-dimensional Gaussian noise.
- The synthetic descriptions cover only a small handful of templated English
  and Hindi words.
- A larger real-world, rights-reviewed, diverse dataset and strict recording
  session splits remain mandatory.
- CLAP/FAD, human preference, throughput, peak VRAM and expensive long-form
  sampling were **not** measured.
- One-second 8-kHz sounds cannot represent mature music generation, TTS,
  expressive sound design, or the final Sonumio vision.

**Research next gate:** matched parameter/FLOP budgets, meaningful
convergence experiments with multiple seeds, trained model sampling, blind
human evaluation, audited data rights and memorization testing. Do not
publish a release, model tag or weights until independently authorized.


## Follow-up: multiple initialization seeds

Use the [replicated study](MULTI_SEED_STUDY.md) when exploring whether
caption improvements are robust to different model initialization seeds.
Results are summarized **within** the raw and latent model representations,
with the 42 training recordings and held-out data identical across runs.
Do not interpret variation across initializations as generalization to
different sound datasets. Both arms still differ in FLOPs and codec cost.
