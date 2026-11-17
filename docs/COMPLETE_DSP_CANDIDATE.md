# Isolated complete DSP candidate: 24 kHz source to 8 kHz training crop

**Experimental first-party synthetic research · 17 November 2026**

This evaluates a **separate, new research recipe**
`candidate-dsp-source-scalar-fsum-v1-not-for-training`.
It is not called by the public API, production procedural renderer,
experimental neural training jobs, or any model checkpoint. Do not replace
the existing historical DSP source in place or interpret the candidate
results as a higher-quality AI audio model.

## Why investigate the whole pipeline?

The earlier [full-source convolution CPU study](FULL_CONVOLUTION_RESULTS.md)
found exact source-convolution output hashes for a scalar,
compensated-summation candidate on four CPU runners, including
environments where the original float32 NumPy convolution differed.
That did not establish reproducible final waveforms because mixing,
fades, normalization, 24-to-8 kHz FIR filtering and cropping all
execute *after* source convolution.

`research/complete_dsp_candidate.py` reconstructs those stages,
preserving the original **seed, RNG draws and order, sound-layer
envelopes/gains, 24 kHz sample grid, sound-combination order,
normalization, and original float64 FIR resampler**. It replaces
only four source-level convolutions (rain, thunder, wind, water)
with `full_scalar_convolution`; forest and ambient retain the
original DSP functions. No monkey-patching or global NumPy changes.

For each case, an independent invocation of
`sonumio.engines.procedural.generate` is downsampled and cropped,
and the original 8 kHz result **must match exactly**
`research.latent_audio.render_seeded_waveform` — the historical
research-data oracle. The candidate is never passed to neural training.

## Evaluated recording grid

| Scope | Distinct source sets | Original train seeds | Held-out seeds | Total clips |
| --- | ---: | --- | --- | ---: |
| smoke | All 6 singles plus four pairs (rain+thunder, rain+ambient, wind+forest, ocean+ambient) | 0, 1 | 101, 102 | 40 |
| full | All 6 singles plus all 15 two-source pairs | 0, 1 | 101, 102 | 84 |

Every recording is a 1.5-second (36,000-frame) 24 kHz synthetic
source, FIR-downsampled to 8 kHz and cropped to 8,192 frames (1.024s).
`full` scope covers exactly 42 distinct recordings per split.
English/Hindi captions are not represented as separate waveform
recordings in this numerical source-comparison report.

The report stores only digest/metric metadata:

- SHA256 of original and candidate complete 24 kHz output, uncropped
  8 kHz result, and final exact training crop;
- maximum absolute error, RMS error, and differing float32 value count
  at all three output stages; the validation cap is `2e-5`;
- split-level aggregate exact hashes, full recording identities,
  execution environment and approximate total processing time;
- anti-leakage controls: no duplicate exact waveform SHA within
  a split and no train/validation waveform SHA overlap, for
  **both** the original and experimental candidate crops;
- failure on malformed identifiers, mismatched original oracle,
  wrong sample shape, nonfinite audio/metric values, forged/inconsistent
  aggregate, or unsupported experiment mode.

No audio arrays, WAVs, private recordings, model weights or checkpoints
are serialized, uploaded or published. The source and its JSON reports
do not independently prove provenance or licensing of future datasets.

## Verified four-host, 40-recording pilot

The [pilot workflow #44063033855](https://github.com/sanskarIN/sonumio/actions/runs/44063033855)
successfully completed **4/4 independent CPU jobs**. Each
job produced the complete 40-recording smoke report twice in separate
processes, verifying byte-identical within-host candidate results.
The four hosted CPU identifiers included Intel Xeon 6975P-C,
AMD EPYC 9V74 and AMD EPYC 7763. All four host jobs reported
**identical entire 24 kHz candidate and 8 kHz cropped corpus hashes**:

| Stage / split | Identical candidate SHA256 across four jobs |
| --- | --- |
| 24 kHz raw / train | `31334881e53ada3a0a918c404c2756dbdace8af68496e2c6426feb52712205d9` |
| 24 kHz raw / validation | `417a621ef9028033bdcf54903a272cd3286a311246a0db8f2d81e817ef1b111d` |
| 8 kHz cropped / train | `1c5f66450a469e056638066c864f560e2ed9ea86ffdd6afc926b4febf892419e` |
| 8 kHz cropped / validation | `92770a0ec18e869efce37a0a1dc1ac5dfedb83504a59e3bd7db9d4d15968a3cf` |

On EPYC 7763, the original renderer's 8 kHz crop hashes **still
differed** from the other hosts, while candidate hashes matched.
Maximum observed absolute candidate-versus-original error across
the 40 pilot recordings was **1.1920928955078125e-7** at 24 kHz
and **8.940696716308594e-8** at the final 8 kHz crop.

These are complete-sample error maxima **within the 40 tested
recordings**, not universal bounds on new data, hardware,
dependencies or DSP implementation changes. Cross-host exact
candidate hashes on four hosted machines do not prove absolute
portability on all operating systems or processors.

The accompanying [CPU research workflow #44063033967](https://github.com/sanskarIN/sonumio/actions/runs/44063033967)
passed **275 research tests** including strict original-oracle
comparison, source/mix cases and deliberate report-tampering tests.
That milestone preceded the subsequently added anti-leakage checks,
which require an additional CI rerun.

## Research commands

```bash
python -m pytest -q tests/test_complete_dsp_candidate.py
python -m research.complete_dsp_candidate --scope smoke \
  --output checkpoints/private-research/complete-dsp-smoke.json
python -m research.complete_dsp_candidate --scope full \
  --output checkpoints/private-research/complete-dsp-full.json
```

The CPU matrix now also executes the full 84-recording scope to
check all pairings on independently selected runners. The specific
full-scope reproducibility result must be established from completed
CI logs; **do not report it as passing solely because this step
has been added**.

## Gates before any future recipe promotion

1. Verify all 84 source/seed combinations on divergent and matching
   host environments with full candidate raw/crop exact hashes.
2. Verify train/validation SHA disjointness, bilingual-caption
   association and a separate, explicitly versioned manifest.
3. Benchmark actual candidate generation **independently of
   oracle/test overhead**, with a documented CPU-time/performance
   budget; this Python scalar implementation is a slow reference.
4. Review audible output via independent listening and multiple
   objective metrics; small numeric errors alone do not prove
   perceptual equivalence, realism or safety.
5. Keep original neural model experiments and existing data recipes
   unchanged unless a separately reviewed experiment explicitly
   requests new candidate data. Model performance requires fresh
   paired experiments and external rights-reviewed evaluation.

**The Sonumio-1 model does not exist as a finished model. PR #1 remains
draft and unmerged. No releases, prereleases, tags, WAV artifacts,
model weights or deployment are authorized by this research.**
