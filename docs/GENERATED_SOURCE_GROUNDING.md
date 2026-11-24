# Generated waveform source grounding

This CPU experiment measures whether a sampled waveform is closer to the
synthetic DSP reference category named in its prompt. The six categories are
rain, thunder, wind, forest, ocean, and ambient. The experiment scores 1.024 s
mono clips at 8 kHz and uses short templated English and Hindi prompts.

## Run

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m research.generated_source_grounding \
  --model-seed 17 --train-steps 32 --codec-steps 16 \
  --evaluations 8 --sampling-seeds 301 302
# Repeat with --model-seed 29 for an independent initialization.
```

Use the optional CPU PyTorch research environment. The command writes a
numeric JSON report to stdout. Audio and parameters stay in memory.

## Reference probe

The probe uses relative power in ten frequency bands and the variation of band
power and frame energy over 512-frame Hann windows at a 256-frame hop. A
centroid per category comes from four generator-independent DSP seeds
(24 recordings). The distance is root-mean-square deviation from the centroid,
scaled by pooled within-category feature variation with a 2 dB floor. The
category acceptance radius is 1.5 times its furthest calibration recording's
distance to its own centroid, with a 0.75 minimum. Calibration uses twelve
different recordings.

An untouched panel of 18 recordings (three further seeds) checks the probe
before either neural arm trains. The study requires a correct accepted
prediction for each category and at least 12 correct accepted predictions
overall. Seed separation and exact waveform hashes guard against overlap
between the three panels and the DDIM trainer's own training and validation
splits. The report includes white noise at 8 kHz, Gaussian noise passed through
the same downsampler as the training data, and a silence abstention check.
These controls expose the probe's tendency to mistake noise for a source.

## Generated-waveform comparisons

Both the raw and latent velocity models use the existing DDIM trainer and
sampler. For each arm and sampling seed, the initial noise is held fixed as
the caption changes. Each generated waveform is clamped to the playable range
for feature extraction, and the unclamped out-of-range sample fraction is
reported separately.

- **Matched** scores the requested category.
- **Rotated** scores a cyclically reassigned category on the exact same
  waveform, keeping the target labels balanced.
- **Zero guidance** checks that two distinct prompts give sample-identical
  unconditional waveforms. It scores that single waveform against the same
  balanced targets to expose which scores can be obtained without conditioning.

The report gives the nearest-category rank, accepted rank-one count, distance
margin to the closest competing category, and mean reciprocal rank. The forced
rank-one chance for balanced six-way choices is 1/6. Distance acceptance is
reported separately so a forced choice on distant audio remains visible.
The report also counts accepted rank-one cases after excluding every category
accepted as a match for any noise control in that run.
Categories and languages share sampling noise, so the scores are descriptive.

The reference probe measures similarity to these particular synthetic DSP
families. Real-world sound identity, perceived quality, open-vocabulary
language comprehension and model readiness require other evaluations.

## Two-initialization pilot

The commands above were run for model seeds 17 and 29 on CPU with Python
3.12.14, NumPy 2.3.5, and PyTorch 2.15.0+cpu. The probe ranked the correct
category first for all 18 held-out reference clips and accepted 17. It also
accepted all six noise controls as rain. All 96 generated samples had rain
as their nearest reference.

| Arm | Condition | Rank one | Accepted rank one | Accepted rank one excluding noise confounds |
| --- | --- | ---: | ---: | ---: |
| Raw | Matched | 8/48 | 8/48 | 0/48 |
| Raw | Rotated | 8/48 | 6/48 | 0/48 |
| Raw | Zero guidance | 8/48 | 8/48 | 0/48 |
| Latent | Matched | 8/48 | 4/48 | 0/48 |
| Latent | Rotated | 8/48 | 3/48 | 0/48 |
| Latent | Zero guidance | 8/48 | 4/48 | 0/48 |

Under this training budget and probe, matched captions did not improve on
zero guidance in either arm. The noise false positives rule out using the
rain matches as evidence of prompt grounding. The 48 cases per arm combine
both model seeds, both sampling seeds, both languages and six categories.


## Paired balanced-label audit (v1)

The numeric report can now be audited independently using only Python's standard
library. For example, redirect a study to a local JSON file and run:

```bash
python -m research.paired_grounding_audit grounding.json
```

The audit rejects incomplete or duplicate cases, mismatched rotated predictions,
contradictory ranks, missing null invariance checks, inconsistent unconditional
predictions and noise-confound declarations inconsistent with the raw controls.
It does not load weights or re-score the original audio; the report is the
input evidence, not an independently authenticated record of the experiment.

Each arm, sampling seed and language forms a six-category panel. The audit
holds the observed predictions fixed and enumerates all 720 one-to-one label
assignments. Its upper-tail fraction includes ties and the observed assignment.
Abstentions remain abstentions, so expected accepted correct assignments are
coverage divided by six rather than always one. Both forced and accepted
nonconfounded results are reported. Every category accepted for either noise
control family is excluded from accepted evidence for all three conditions.

A descriptive screen requires a tail fraction at most 0.05 and strictly more
accepted nonconfounded matches than both rotated and zero-guidance controls.
This is not a corrected significance test across panels, a population estimate,
or a release gate: languages and arms share noise and reference data. Do not
pool their tail fractions or equate passing with model readiness. Mode collapse
predicting rain for all targets produces a forced rank-one count of one per
panel under every permutation, regardless of whether the probe accepts rain.
The reported earlier pilot would therefore fail the descriptive screen. This
is a deduction from the documented pilot, not a new training run.

The CPU research workflow runs unit tests and audits both fresh trained-model
reports without saving weights or audio.
