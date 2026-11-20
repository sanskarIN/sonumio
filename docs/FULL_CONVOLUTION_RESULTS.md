# Full-frame scalar source convolution — verified multi-CPU result

**17 November 2026 · Sonumio experimental draft PR #1 · not a release**

[Four-runner GitHub Actions experiment #44060065472](https://github.com/sanskarIN/sonumio/actions/runs/44060065472)
successfully completed the full-frame,
[separately implemented scalar convolution study](FULL_CONVOLUTION_CANDIDATE.md)
at commit [a90cf379](https://github.com/sanskarIN/sonumio/commit/a90cf3793a7ff36d50e938286f265ed5dbc1afdd).

**All four independent runner jobs succeeded.** In each job, eight
golden-vector tests passed, then the full study independently
reconstructed all 16 source/seed cases **twice**, producing identical
candidate and original NumPy hashes across processes on that host.

## CPU environments tested

| Runner instance | Image | Reported CPU model | Original NumPy behavior | Scalar candidate |
| --- | --- | --- | --- | --- |
| 1 | Ubuntu 24.04 | AMD EPYC 7763 | Divergent original NumPy hashes | Match |
| 2 | Ubuntu 24.04 | Intel Xeon 6975P-C | Baseline original hashes | Match |
| 3 | Ubuntu 22.04 | AMD EPYC 7763 | Divergent original NumPy hashes | Match |
| 4 | Ubuntu 22.04 | AMD EPYC 9V45 | Baseline original hashes | Match |

The **EPYC 7763 mismatch is directly represented in this run**.
The [earlier selected-only run #44059208693](SCALAR_CONVOLUTION_RESULTS.md)
used two EPYC 9V74 and two Intel Xeon hosts; one EPYC 9V74
also showed the historically divergent original NumPy hashes.
Therefore processor model alone is not a proven cause.

## Complete output SHA256s (aggregated across four seeds)

The scalar candidate was evaluated for every first-party
source-convolution output sample, not merely 264 positions:

- Rain: 36,179 output frames × 4 seeds, including the full-mode
  tail; the existing renderer crops the first 36,000 frames.
- Thunder, wind and water: 36,000 output frames × 4 seeds each.
- Four sources × four seeds = **16 complete convolution cases**.

| Source | Identical candidate full-output SHA256 across 4 jobs | Original full-output SHA differs on EPYC 7763? | Maximum absolute candidate-vs-NumPy difference over all output samples |
| --- | --- | --- | ---: |
| Rain | `4b252e3bc9361b2a50cc5bdd0a65a12ac91f28de50504590270abd068fd81ff5` | Yes | 1.1920928955078125 × 10⁻⁷ |
| Thunder | `e2244e816126366e569ee03b0f9a70807adb1209811617933efdaf764fa0eba6` | Yes | 5.960464477539063 × 10⁻⁸ |
| Wind | `a57ea85fe044f6b2f17ea66a9ca5c8232be099039c1c1e95becc2660aeeb1bf1` | Yes | 5.960464477539063 × 10⁻⁸ |
| Water | `875fbf371c3282499838d142b909737821e36706d69eda6a3c5191c2a92a24ab` | No | 1.1920928955078125 × 10⁻⁷ |

Rain example: original full-convolution aggregate SHA256 on the
baseline hosts was
`6b9ddc4bf42e74d7dc262f683bbc9f70e27a90aea3d6e05a2f899da64ee92079`,
whereas on EPYC 7763 it was
`0bddc025b45b8b8e2bb2469c9912c0b84d6a42febd97de475b09c7be8d3f4c3c`.
The corresponding scalar candidate SHA remained exactly
`4b252e3bc9361b2a50cc5bdd0a65a12ac91f28de50504590270abd068fd81ff5`
on **all four** CPU jobs.

The candidate is **not bit-identical to NumPy**; indeed several
float32 values differ because NumPy uses different reduction
arithmetic. The table reports the **maximum over the complete
convolution output** for each source and four seeds, not an
arbitrary sampling estimate. It does not bound other input
waveforms or downstream synthesis stages.

## Tests, provenance, and security boundaries

- Full-frame candidate output is assembled in 1,024-position
  blocks; regression tests enforce bitwise equality across
  chunk boundaries.
- Eight independent fixed float32 goldens cover 12-, 60-,
  140-, and 180-tap kernels in both `full` and `same` modes;
  all eight passed on each matrix runner.
- Every source/seed case validates its input/kernel SHA against
  the untouched original `research.dsp_convolution_probe`.
- The original NumPy legacy-crop output SHA is checked against
  the existing source oracle before candidate evaluation.
- The experiment writes only **hashes, numeric metrics and
  allowlisted CPU metadata**, no WAV samples or weights. The
  existing source data recipe, real-user data and trained
  neural objectives are unchanged.

## Interpretation — one specific problem solved, many still open

**Supported by this experiment:** on these four hosted Linux
jobs with the same input/kernel bytes, the separate scalar
candidate produced **exactly matching complete source-convolution
bytes** where the original NumPy path did not. This is practical
evidence for a more reproducible arithmetic primitive.

**Not proven:** identical *post-convolution* layer rendering,
multi-layer mixing, normalization, FIR resampling/cropping, final
8-kHz train/validation source hashes, cross-version portability,
acceptable inference/training speed, perceptual audio quality,
sound relevance to English/Hindi prompts, or external data rights.
One successful four-runner trial is not a proof for every CPU.

The immediate next research experiment should recompute **entire
DSP recordings from source layers through final 8-kHz crop** with
a separately versioned candidate arithmetic path, while leaving
all existing recordings and neural comparison data untouched.
It must test runtime cost, source/seed splits, exact hashes and
waveform errors; retain separate human listening and rights
evaluation gates.

**Development policy:** PR #1 stays DRAFT and UNMERGED; no
pre-release, release, tag, WAV package, model weight or public
deployment has been authorized or performed.
