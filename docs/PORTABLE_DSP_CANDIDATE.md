# Portable ambient synthesis: full-corpus numerical experiment

**Research stage · 17 November 2026 · PR #1 remains draft**

## Motivation and verified precursor

The earlier four-host [full fast DSP experiment](FAST_DSP_RESULTS.md)
reproduced the verified scalar reference for the complete **84-case**
training/evaluation-source grid and, separately, all **105
previously unseen synthetic seed/source pairs**
([evidence](HELDOUT_DSP_RESULTS.md)), with NumPy 2.4.6.
These were source-level waveform experiments, not trained
generative model evaluations.

The controlled [NumPy version audit](AMBIENT_SINE_RESEARCH.md)
revealed a **real portability limitation**: the original
NumPy float32 sine outputs for the ambient 110-Hz and
220-Hz oscillators changed between NumPy 1.26.4 and
2.4.6 on the *same* host, affecting the final
waveform SHA256 in twelve of forty inspected
source/seed combinations.

A subsequent controlled same-CPU
[oscillator comparison #44077947598](https://github.com/sanskarIN/sonumio/actions/runs/44077947598)
found **identical full 36,000-frame float32 SHA256 output**
from both isolated alternatives when switching NumPy
1.26.4 to 2.4.6:

- **numpy_float64**: exact original float32 phases,
  promoted once to float64 for `np.sin`, rounded
  back to float32 before original mixing;
- **python_math_sin**: exact original float32 phases,
  promoted to Python binary64 for scalar `math.sin`,
  rounded back to float32 before original mixing.

The original `dsp._drone` differed between versions.
Both candidates measured the same numerical maximum
difference **1.1920928955078125e-7** from the
version-local unchanged original for this specific
1.5-second ambient sound. This test does *not*
establish cross-CPU or whole-pipeline stability.

## Separate 189-case complete DSP experiment

`research/portable_dsp_candidate.py` is a newly
identified non-production source recipe:

`research-f64-source-conv-plus-f64-ambient-sine-v1-no-training`

It reconstructs original first-party synthesis and
substitutes only two **already isolated** operations:

1. Four source sound convolutions use the verified
   vectorized float64 research candidate, which
   reproduced the scalar fsum source reference on
   previously tested NumPy 2.x/CPUs.
2. The ambient/drone source replaces original
   float32 `np.sin` with one of the explicit
   `numpy_float64` or `python_math_sin`
   alternatives, based on a method field.

All other elements—float32 time grid, original
recognized source order, RNG draws, gains,
envelopes, 24-kHz layer mix, 25-ms fades,
peak normalization, legacy float64 FIR
downsampling to 8 kHz, and original
seed-specific 8192-frame crop—are
independently reconstructed, preserving
the original baseline as an oracle.

The corpus contains **189 cases per method**:
42 original train source recordings from
seeds 0/1, 42 validation source recordings
from seeds 101/102, and 105 research-only
holdout source recordings from seeds
3/7/17/29/41, all with six single sounds
and every one of fifteen two-source pairings.

Both oscillator options therefore create
**378 separate candidate recordings**.
These are a *numerical test grid*, not a
proposal to train two models or publish
378 WAV files. No models are trained and
no waveform files are emitted.

## Numerical integrity and anti-leakage gates

For each case and method, the tool:

- Calculates the 36,000-frame fast
  float32 source waveform, final
  8-kHz crop and exact SHA256s.
- Independently reruns the original
  first-party generator and checks the
  original train/validation crop against
  the historical neural data oracle.
- Validates output lengths, dtype,
  all-finite values, exact seed/split
  and crop offsets, and maximum
  original-versus-candidate sample
  difference no greater than `2e-5`.
- Refuses duplicates in 8192-sample
  final cropped fingerprints **across
  the entire 189-case grid per method**,
  including train/validation/holdout
  overlap. Identical raw seed-invariant
  ambient waves remain allowed before
  cropping, where they are expected.
- Validates source combination and
  seeded identity coverage; rejects
  tampered numerical provenance
  aggregate hashes.
- Records only SHA256 digest strings,
  descriptive source identities, timings
  and numeric errors in local JSON.
  **No audio arrays, weights, WAV
  files or original training data
  are written or replaced.**

## Independent verification plan

A dedicated
[portable DSP experiment workflow](https://github.com/sanskarIN/sonumio/actions/workflows/dsp-portable-matrix.yml)
is installed but **its specific result must be
observed before making portability claims**.
Its four parallel CPU runners use NumPy
2.4.6 and print candidate per-split
24-kHz and 8-kHz exact fingerprints.
A fifth job starts on one Ubuntu
runner with NumPy 1.26.4, generates
the entire 378-case fingerprint
report, switches that **same runner**
to NumPy 2.4.6, and independently
regenerates the complete report
before comparing *per-recording*
SHA256s for each oscillator method.

The workflow intentionally treats
the cross-version fingerprint equality
as an **empirical outcome**, not a
presupposed assertion. If it differs,
it prints the values and requires
further stage localization; if it
matches, exact golden hashes can be
recorded and made into future
automated gates.

## Engineering boundary

No step promotes the alternative
ambient synth into
`src/sonumio/engines/procedural.py`,
the public service, original latent
research data, diffusion training,
model weights or inference packaging.

A stable CPU synthetic-byte SHA
does not prove neural generative
audio fidelity, speech/music
quality, pronunciation, bilingual
prompt following, long-form
music structure, real-world
audio dataset licenses or GPU
portability. Those are distinct
future Sonumio research programs.

**PR #1 remains draft and unmerged.
No tags, releases, prereleases,
model weights or downloadable audio
artifacts have been authorized.**
