# Portable ambient DSP — first verified results and remaining version drift

**Sonumio source-level numeric research · 17 November 2026**
**Four-CPU results:** [GitHub Actions #44080000639](https://github.com/sanskarIN/sonumio/actions/runs/44080000639)

## Executive result

The separate `research/portable_dsp_candidate.py` reconstructs
Sonumio's original first-party procedural sound generator using
two isolated numerical substitutions: vectorized float64
convolution for source sound smoothing, and either float64
NumPy sine or scalar `math.sin` for ambient synthesis.

Four separate GitHub-hosted Ubuntu CPU jobs completed
successfully, each running **189 recording cases × 2
ambient methods = 378 waveform candidates**, with
NumPy **2.4.6**. In *every* job the SHA256
fingerprints of the complete 24-kHz waveform
and final 8-kHz crop matched exactly,
and the two ambient alternatives produced
identical bytes for every tested case.

A separate **fifth** CI job on one host
recomputed all 378 candidates under
NumPy **1.26.4**, replaced only NumPy
with **2.4.6** in the same runner,
and recomputed them independently.

The result was **mixed**:
- Original source-data train grid, 42
  recording cases per method: **exact match**
  across the two NumPy versions.
- Original source-data validation grid,
  42 recording cases per method: **exact match**
  across the two NumPy versions.
- Additional numerical stress grid, 105
  cases per method: the *aggregate* 24-kHz
  and final 8-kHz crop hashes **differed**
  across those versions, for **both** methods.

Thus: both candidate methods are bit-reproducible
on four tested NumPy-2.4.6 machines, and both
repair the original 84-case NumPy 1.26→2.4
mismatch, but **neither is yet a proven
cross-version replacement for all 189 cases**.

The initial five-job workflow was a
measurement workflow. Subsequent CI work
adds explicit assertions for the proven
84-case cross-version grid and frozen
golden references for all 378
NumPy-2.4.6 recordings. Those *new
assertion runs must succeed separately*
before their gate implementation itself
is considered verified.

## Exact SHA256 reference for the 189-case corpus per method

All four NumPy-2.4.6 runner jobs reported
the **same** values below, for both
`numpy_float64` and `python_math_sin`.
Hashes are SHA256 of the concatenated
*per-recording digest bytes* in canonical
source/seed order, not hashes of WAV
files, neural checkpoints, audio streams,
dataset releases or model weights.

| Numerical split | Cases per method | 24-kHz aggregate SHA256 | Final 8-kHz crop aggregate SHA256 |
| --- | ---: | --- | --- |
| Original train | 42 | `25482fb0bd0415c3db39787e835088c4af5b29b16ea993e79b1ae936d9422f34` | `ee0fd7edd68e2fc5bc3bd6ceaa4472b875290d82f5ae5bde3e3249e055b40f79` |
| Original validation | 42 | `df1da429546460fcf5faee0f76f1742cf324160e10a0565ff52a34fcf06bf48e` | `d16ac7558a1f4f5ccec2490ca5746ada995cebb27fbf12d98e06c94a2aaebd01` |
| Additional numerical seeds | 105 | `afe0e669d9577f284eaa50bf7d85d1e2722140aa8f3abb1f9fd8f88cd4ed8079` | `30eaf47c085fec411fe5861037a73934bb149176fd553dbbeceb875a5dd3c8ca` |

The four runners reported CPUs including Intel Xeon
6975P-C, Intel Xeon Platinum 8370C, and AMD EPYC 9V74.
Hosted hardware assignments can change between reruns,
so the inference is restricted to these exact runs.

The corresponding verified per-split values are frozen
in `research/portable_dsp_reference.py`, with
`verify_reference(report)` checking the full
378-recording in-memory report on **NumPy 2.4.6 only**.
The original host-sensitive float32 DSP
hashes are *deliberately not pinned*.

## Cross-NumPy divergence is limited to the extra-seed split

The fifth job on an Intel Xeon 6975P-C reported:

| Numerical split | 24-kHz portable aggregate cross-version | Final 8-kHz crop aggregate cross-version |
| --- | --- | --- |
| Original train 42 | Identical | Identical |
| Original validation 42 | Identical | Identical |
| Additional 105 | **Not identical** | **Not identical** |

The NumPy-1.26.4 heldout numerical stress
aggregates were:

- 24 kHz: `618fdb3a92d735c8ac52016fc6da43f8a802a60630c6185e6a6014091fa4b486`
- 8 kHz cropped: `b64725c3c5d62efa5625a6d7be397f6447e299521c4a20ae67e9d78f7c78b976`

The paired NumPy-2.4.6 aggregates are in the table above.
**An unequal aggregate does not reveal how many of the
105 individual recordings differ.** The code now
records precise per-seed and per-recording difference
counts in `research/portable_dsp_candidate.compare_reports`,
and a new independent
`research/portable_seed_stage_probe.py` traces the
additional sound layers and RNG state to find the
first divergent execution stage on the same runner.
Their results are pending and should not be inferred
from the old aggregate results.

## Important dataset lineage

The 105 additional source/seed cases use five
seed values 3, 7, 17, 29 and 41, disjoint
from the original latent-audio 84-case source
grid. However, **seed 3 overlaps**
`research.text_audio.SYNTHETIC_TRAIN_SEEDS`.
Therefore these 105 are valid additional
**numerical reproducibility** stress cases
only: they are **not** a globally untouched
neural-model evaluation split, and their
matching source SHA does not establish
model generalization or absence of training
set leakage in other pipelines.

## Performance and limits

Earlier controlled microbenchmarks showed
the source vectorized float64 convolution
can be substantially faster than a Python
`math.fsum` scalar implementation, but
it is a numerical source-DSP comparison,
not generative model performance. Four
single-shot ambient-oscillator timing
samples measured the float64 NumPy
oscillator in approximately 1.3–1.8 ms
and the scalar `math.sin` oscillator
in approximately 8.0–12.4 ms on
the tested hosted CPUs. These single
CI timings are indicative only and
do not establish reliable speed ratios
or multi-tenant inference performance.

**No existing Sonumio generator, neural dataset,
training pipeline, checkpoints or deployed service
has been changed to use the experimental methods.
No audio files or model weights are published.**
PR #1 remains draft and unmerged; Sonumio
has no approved release or prerelease.
