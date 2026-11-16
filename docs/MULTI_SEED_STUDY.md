# Multi-seed raw vs latent audio research

**Research only; no Sonumio-1 model, downloadable checkpoints, releases,
pre-releases or version tags.** Work is confined to draft PR #1.

This protocol repeats the [same-data architecture diagnostic](ARCHITECTURE_COMPARISON.md)
over several explicitly chosen **model-initialization random seeds**. It adds
replication, descriptive variability, cross-run consistency validation and
negative controls. It does **not** add a real-world audio dataset or solve
the unequal-compute problem.

## Controlled design

Every independent initialization sees:

- The same 8 kHz **1.024-second** non-voice DSP recordings, with distinct
  training and validation seed sets and exact-byte overlap checks.
- Identical bilingual caption templates, optimizer-update counts,
  minibatch indices *within that run*, schedule, channel choice and
  evaluation source-combination subset.
- A raw text-conditioned DDPM on waveforms and a separate text-conditioned
  DDPM on 16 × 128 learned audio latents.
- A latent convolutional codec separately trained on unique recording WAVs.
  Its cost and reconstruction MAE are reported independently.
- Paired, wrong-source and blank text plus zero-noise-predictor controls at
  low, medium and high noise. All caption conditions within an arm are
  evaluated against exactly the same corrupt sample and target noise.

The data sources, sound category families and caption wording are **not**
independently sampled across model initialization seeds. Thus, a multi-seed
study tests *initialization sensitivity only*, not performance across novel
recording sessions, real sounds or different languages beyond the templates.

The raw and latent paths have different representations, parameter counts,
model capacity, codec overhead and numerical noise. **Do not rank their
absolute MSEs against each other.**

## Local reproducible command

From repository root:

~~~bash
python -m pip install -e '.[research,test]'

python -m research.multi_seed_study \
  --seeds 17,29,41 \
  --codec-steps 32 \
  --denoiser-steps 64 \
  --schedule-steps 8 \
  --channels 16 \
  --evaluation-combinations 8 \
  --output checkpoints/private-research/multi-seed-17-29-41.json

python -m pytest -q tests/test_multi_seed_study.py
~~~

For PowerShell, place the command on one line, or use PowerShell's backtick
line continuation rather than Bash backslashes.

The code permits 2–8 distinct explicitly listed 32-bit seeds, 1–1,000
optimization steps per component, 8–256 diffusion steps and 2–21 evaluated
sound-source combinations. More steps or more seeds increase CPU time and
RAM requirements. **Short two-/three-step runs test execution only.**
Preselect seeds and configurations before examining outcome metrics.

The JSON report includes each unabridged run, shared train/validation data
fingerprints, model parameter counts, codec reconstruction error, system
version metadata, and a stable SHA256 of the *experiment configuration*
(not the machine-specific measurements). It is saved only to the explicit
local path; no neural weights are saved, trained or uploaded as artifacts.

## Statistical interpretation

Summaries are calculated **separately** for raw and latent predictions at
each noise band. For each metric, report:

- Number of seeds, mean, median, sample standard deviation.
- Observed minimum and maximum, not a confidence interval.
- The count of positive, zero and negative seed outcomes.

The three most interpretable within-model diagnostic quantities are:

| Quantity | If positive, it means... |
| --- | --- |
| wrong-caption MSE − matched-caption MSE | Matched captions have lower noise error |
| unconditional MSE − matched-caption MSE | Text lowers error against blank conditioning |
| 1 − paired MSE / zero-predictor MSE | The denoiser beats a zero-noise prediction reference |

A **caption-blind model** should show zero caption advantage. All metrics
remain noise-prediction diagnostics, not a naturalness, text-relevance or
copyright-safety test. Running more random initializations does not turn
this restricted DSP benchmark into an independent listening test.

There is no automatic release gate, best-model promotion, winner announcement
or confidence-level claim. Selection after inspecting results would also
require fresh, independent preregistered evaluation to avoid overfitting
these known DSP templates.

## Failure gates

The experiment aborts rather than averaging invalid/incompatible results if:

- Initializer seeds repeat, are malformed, or fail to match report metadata.
- Training or validation fingerprints, sample dimensions, source-combination
  selection, update counts, channels, or model parameter counts differ.
- A run reports exact training/evaluation WAV overlap or saved/published
  weights.
- Any model's noise metric is absent, nonfinite, invalid, or inconsistent
  with its documented arithmetic.
- A model generation is claimed.

## Scientific milestones still missing

The next quality gate is substantial multi-seed training on **human-reviewed,
diverse, consent-/license-cleared sound recordings** with session-disjoint
holdouts, real sample generation, real listening panels, matched compute
constraints, and CLAP/FAD or other properly licensed and documented
reference metrics. Even if this synthetic experiment improves noise MSE, it
would not establish Sonumio-1 readiness.


## Corrected negative-caption protocol

The former circular-caption shuffle did not always replace **every** sound
mentioned by a caption. `v2` now requires the wrong caption to name
**zero overlapping sound categories**, with the same number of sources and
the same English/Hindi style. Each run includes the exact negative-token hash
and the multi-seed aggregator rejects incompatible or older `v1` results.

A caption advantage that improves under this correction would still be
limited to known DSP sources and short templates. It is not proof of
open-ended prompt understanding or usable generated-audio quality.

## Measured pilot result (16 November 2026)

See the [three-initialization experimental results](MULTI_SEED_PILOT_RESULTS.md)
for the verified 24-codec-update/48-denoiser-update CPU run, exact
configuration hash, category-disjoint caption scores and limitations.
This result does **not** establish usable generated-audio quality.
