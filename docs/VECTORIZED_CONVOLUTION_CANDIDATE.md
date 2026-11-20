# Accelerating deterministic DSP convolution: float64 vectorized hypothesis

**17 November 2026 · isolated research candidate · not for model training**

## Why this experiment exists

The [paired performance experiment](DSP_CANDIDATE_BENCHMARK.md)
on CPU research [run #44064895590](https://github.com/sanskarIN/sonumio/actions/runs/44064895590)
reported **281 passed research tests** and, for one 1.5-second
first-party rain recording on AMD EPYC 7763 with two measured
repetitions and no warmup:

| Renderer | Reported median rendering time |
| --- | ---: |
| Original 24-kHz NumPy DSP | 0.001956907 seconds |
| Separate compensated scalar candidate | 1.3280277505 seconds |
| Candidate / original median ratio | **678.64×** |

The scalar method proved useful as a numerical **reference**, but
that microbenchmark makes it a poor immediate production candidate.
It is one host and one source with very few repetitions; the number
is **not** a universal performance ratio or a model speed benchmark.

## New, experimentally separate candidate

The [float64 vectorized candidate](../research/vectorized_convolution_candidate.py)
computes

```python
np.convolve(
    original_float32_input.astype(np.float64),
    original_float32_kernel.astype(np.float64),
    mode=original_mode,
).astype(np.float32)
```

This preserves exact original float32 inputs and kernels,
changes the accumulation to vectorized binary64 arithmetic, and
rounds exactly once to float32. It is intended to approximate the
scalar `math.fsum` reference while enabling NumPy's fast vectorized
implementation. It is **not automatically bit-identical to scalar
`math.fsum`**, particularly at float32 rounding midpoints. Since
NumPy vectorized implementations and CPUs can differ, actual
cross-host SHA comparison is essential.

The experiment covers **16 cases**: rain (180-tap, full mode),
thunder (140-tap, same), wind (60-tap, same) and water (12-tap,
same) × original seeds (0, 1, 101, 102). Each case:

- Recreates and hashes the exact original DSP source inputs and
  kernels, validating against the untouched `isolate_one` oracle.
- Computes a full vectorized float64 convolution and casts to
  float32; records its exact full-output SHA256 and original-length
  crop SHA256. No sample arrays or WAVs are exported.
- Evaluates a deterministic set of reference positions with
  independent scalar `math.fsum` products and measures exact
  float32 disagreements plus maximum sampled absolute difference.
- Aggregates four full-output SHA256 digests per source and
  **compares the aggregate** against the independent scalar
  full-convolution SHA256s previously measured in
  [four-runner workflow #44060065472](FULL_CONVOLUTION_RESULTS.md).
- Records observed per-call vectorized arithmetic latency, with
  no unearned model/audio quality claims.

The independent CPU matrix now runs two Python processes per
host and checks exact **within-host** vectorized bytes; it
prints the per-source SHA aggregates for manual cross-host
verification. It **does not require a match to the scalar
reference yet**: the purpose of the experiment is to discover
whether that hypothesis is true, not to assume the outcome.

## Research invocation

```bash
python -m pytest -q tests/test_vectorized_convolution_candidate.py
python -m research.vectorized_convolution_candidate \
  --output checkpoints/private-research/vectorized-float64-study.json
```

The output is JSON numeric metrics and hashes only. Reproduction
depends on NumPy version, processor/runtime and the original input
recipe. It is inappropriate to silently overwrite the earlier
8-kHz dataset hashes or claim universal determinism.

## Promotion gates

Before evaluating this faster operation in a completely separate
24-kHz-to-8-kHz candidate:

1. Require full source-convolution SHA equality across independently
   provisioned CPU hosts and report each source's exact agreement
   or disagreement with the slower scalar reference.
2. If mismatches exist, determine whether they arise only from
   benign float32 rounding or from host/runtime-dependent
   output bytes. An individual mismatch prevents claiming
   exact scalar identity even if error is extremely small.
3. Repeat all **84 synthetic recording sources**, single
   and dual-layer mixing, original FIR resampling and
   exact 8-kHz research crop comparisons under a new recipe ID.
4. Benchmark at least three warm-up/replicate configurations,
   multiple source pairs and independent machines; do not
   impose an arbitrary production latency gate based on one CI job.
5. Keep new data fingerprints strictly separate from historical
   neural experiments and require new listening, caption,
   licensing and model-performance evaluations.

**Do not release weights, WAV artifacts, tags, stable or pre-release
versions. Keep development draft PR #1 unmerged and untouched
original model-training sources.**
