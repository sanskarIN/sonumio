# Controlled diffusion-prior schedule ablation — 16 November 2026

**Status: experimental; no audio quality winner and no release approval.**
This experiment tests the specific failure hypothesis from the
[actual generated-audio pilot](GENERATED_AUDIO_PILOT_RESULTS.md): the
eight-step linear-beta training schedule ends with a cumulative clean-signal
fraction of about 0.922, although DDPM sampling starts from pure noise.

## Scientific question

Does changing only the eight-step DDPM diffusion schedule to a
**cosine terminal-noise profile** reduce the observable raw-waveform
clipping and improve numerical prompt-change controls after training?

The alternative is intentionally **not** called a solution until
actual paired-seed results support it. Even if clipping improves,
these numerical diagnostics do not establish a realistic or
prompt-correct sound effect.

## Prespecified profiles

| Property | Original | Alternative |
| --- | --- | --- |
| Profile ID | \`linear_beta_v1\` | \`cosine_zero_terminal_v1\` |
| Beta recipe | Linear 0.0001–0.02 | Cosine cumulative alpha curve, offset 0.008 |
| Steps | 8 | 8 |
| Last beta | 0.02 | Max clamped to 0.999 |
| Approximate terminal cumulative alpha | 0.922212 | 0.00003747 |
| Assumed sample starting field | Standard normal noise | Same standard normal noise |

The cosine schedule is computed in float64 and stored in finite float32
buffers. The terminal beta cap prevents a true zero alpha and the
resulting division-by-zero problem. A final beta of 0.999 is **very
large**; numerical instability, saturation, and sampling failure
remain possible. Both schedule profiles are bounded and validate their
monotonic alpha trajectory. The prior linear default is unchanged,
preserving earlier research checkpoint behavior.

## Controlled setup

Both profiles run the **same initialization seeds (17, 29, 41)**,
the same first-party procedural sound dataset (42 unique training
waveforms and 84 bilingual training captions; 42 distinct held-out
waveforms and 84 caption pairs), the same sound combinations, the
same model widths and independent initial parameter values, the same
batch-index generators and optimizer settings, and:

- 128 codec optimizer updates per initialization
- 256 raw-waveform DDPM updates per initialization
- 256 latent DDPM updates per initialization
- 8 diffusion timesteps and 1.024-second, 8-kHz mono clips
- All 21 held-out source combinations
- In-memory generated-audio checks for rain/बारिश and wind/हवा,
  with the same sample-noise seeds within each model arm

The **only intended intervention** is the diffusion schedule.
The comparison validates matching dataset fingerprints, training
parameters, model parameter counts, seed IDs and successful in-memory
generated-audio audits. It rejects inconsistent or missing runs.

## Measurements

For each random initialization and each of the raw/latent model arms,
the report records the **cosine minus original linear** difference in:

- Generated RMS and peak amplitude.
- Hard-clipping fraction and near-silence fraction.
- Spectral centroid, low-frequency energy fraction and spectral flatness.
- Same-starting-noise rain-versus-wind waveform mean absolute difference
  and cosine similarity, separately evaluated for English and Hindi.

It reports paired per-seed outcomes and descriptive mean/median,
sample SD, observed range, and sign counts. The research system retains
the per-arm denoising results for diagnostic context only. Comparing
raw-model and latent-model MSE directly as though they were in one
unit remains scientifically invalid.

**Interpretation constraints:** A decline in clipping is a useful numeric
health improvement, but not perceptual fidelity; an increase in
rain-vs-wind waveform difference does not prove the generated sounds match
the descriptions. Higher terminal Gaussian-prior agreement is a
structural benefit, **not** evidence of correctly trained reverse
dynamics. A change in spectral centroid is descriptive rather than a
quality grade. Three initialization seeds are exploratory.

## Run (development branch only)

~~~bash
python -m pip install -e '.[research,test]'
python -m research.schedule_ablation \
  --seeds 17,29,41 \
  --codec-steps 128 --denoiser-steps 256 \
  --schedule-steps 8 --channels 16 \
  --evaluation-combinations 21 \
  --output checkpoints/private-research/schedule-ablation.json

python -m pytest -q tests/test_schedule_ablation.py
~~~

For Windows PowerShell, place each command on one line or use backtick
continuation instead of the shown Bash backslash.

The GitHub
[Private synthetic audio training](../.github/workflows/synthetic-training.yml)
job executes this same paired study. Only the JSON metrics report
exists in the ignored \`checkpoints/\` directory; **no weights, WAVs,
model artifacts, tags, or releases are written or uploaded**.
The workflow asserts all 3 matched seeds and both schedule profiles.

## What remains even if cosine is numerically healthier

- Higher-fidelity codec and sample-rate improvements.
- Longer clips and multiple independent audio categories.
- Independently licensed, rights-reviewed real recordings with
  session-disjoint train, validation and final test data.
- Independent human listening, blind sound-class relevance assessment,
  dynamic range, loudness and clipping review on actual audio.
- An honest architecture/compute-matched model comparison and
  convergence tests with more seeds/updates.
- Copyright/memorization/privacy review and consent and data retention
  gates before considering voice and music use cases.
- Full product security, job/storage reliability and UX.

**Maintain draft PR #1; do not merge or publish weights, prereleases,
stable releases or tags without separate explicit owner approval.**


## Verified outcome — negative result

The full [three-seed paired schedule experiment](SCHEDULE_ABLATION_RESULTS.md)
**completed successfully**. Despite a near-zero cumulative terminal
alpha, the cosine-plus-epsilon candidate severely worsened clipping:
raw-model clipping increased from 13.71% to 94.62%, and latent-model
clipping from 0% to 63.86%. All matched seeds were worse. Accordingly
**do not select** this cosine schedule for normal inference; it
remains an experimental failure control. The next research question
is stable velocity-prediction DDPM sampling, not amplitude hiding.
