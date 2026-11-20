# Scalar convolution reproducibility candidate (research only)

**17 November 2026 · draft PR #1 · not a new Sonumio model or training recipe**

Earlier [measured CPU experiments](DSP_CONVOLUTION_RESULTS.md) isolated
a repeatable difference between NumPy float32 convolution output bytes
on one AMD EPYC 7763 runner and multiple other runners, despite matching
inputs and kernels. This study evaluates a separately implemented
*scalar reference candidate* **without changing** the original DSP
generator, original synthetic training/evaluation corpus, checkpoint
formats, model configurations, or benchmark histories.

## Precisely what is new

`research/stable_convolution_candidate.py` exposes
`selected_scalar_convolution(samples, kernel, positions, mode=...)`.
It accepts bounded finite, one-dimensional **float32** inputs/kernels.
For each requested output position:

1. The input and kernel values are exactly promoted from float32 to
   Python binary64 scalars.
2. Each pair is multiplied as a Python binary64 operation.
3. `math.fsum` combines those products using compensated summation.
4. The result is rounded **once** to float32.

The sample alignment agrees with NumPy's full and same modes for the
supported case `len(samples) >= len(kernel)`, including even-length
kernels and output boundary positions. Each input is capped at 50,000
frames, kernels at 256 taps, and each call at 2,048 selected positions.
The algorithm is intentionally slow, CPU-only, does not support
arbitrary tensor shapes, and is not part of production inference.

`probe_one` recreates the **identical original** first-party
convolution inputs and kernels for rain (180 taps, full/cropped),
thunder (140, same), wind (60, same), and water (12, same). Critically,
it independently invokes the pre-existing unmodified
`isolate_one` probe and requires exact SHA256 agreement of the
original inputs and kernels **before** any candidate comparison.

Every case compares deterministic, selected positions from the
scalar candidate against NumPy's original output:

- All **four research seeds**: 0, 1, 101 and 102; 16 source/seed cases.
- At least 257 positions sampled across each complete 36,000-frame
  24-kHz source (including boundaries and convolution-specific offsets).
- Original *full-output* SHA256 plus two separate hashes of the
  *selected output* values (original NumPy and scalar candidate).
- Exact float32 disagreement counts, selected maximum absolute
  difference, and sampled RMS absolute difference.
- Separate digest-only per-source aggregates; no audio bytes,
  WAVs, model weights, private recordings or checkpoints are emitted.

The independent four-runner CPU CI matrix re-runs this probe twice
within each job and compares its exact hashes. Printed SHA256s can
then be compared **across hosts** to test whether the scalar candidate
is more reproducible on the known divergent CPU.

## How to run

```bash
python -m pytest -q tests/test_stable_convolution_candidate.py
python -m research.stable_convolution_candidate \
  --output checkpoints/private-research/scalar-candidate.json
```

The JSON contains numeric metrics and hashes, not audio arrays.
The test suite includes short full-output convolution checks across
kernel sizes 1, 2, 3, 12, 60, 140 and 180, both alignment modes,
float32 conversion, full-mode tails, malformed shapes, oversized
inputs, invalid positions, the original-signal input-hash oracle,
complete source coverage and report-tampering controls.

## Interpretation and promotion gates

A candidate selected-output SHA256 matching across two CPUs is
**not** sufficient to assert full-audio byte identity, model
reproducibility, numerical equivalence for all possible inputs,
perceptual quality, or acceptable throughput. Even a small measured
maximum **over the sampled positions** does not bound the unmeasured
samples. A same-host result cannot prove cross-host consistency.

Before any future separate, versioned training-data recipe:

1. Reproduce the original NumPy mismatch on EPYC 7763 or another
   independently confirmed divergent host.
2. Compare scalar-candidate digests across that host and at least one
   baseline host, with exact input/kernel digests matching.
3. Extend to **every output frame**, at least four original seeds,
   source mixtures, complete resampling/cropping and explicit
   train/validation fingerprints.
4. Benchmark runtime and memory against the current vectorized DSP
   baseline; decide whether an optimized deterministic/fixed-point
   method is needed.
5. Run independent numerical-error characterization, human listening
   evaluation and a separate provenance/version migration plan.
6. Re-run paired neural experiments using an independently reviewed,
   explicitly versioned dataset only after these gates.

This is a controlled numerical research comparison, **not**
completion of Sonumio-1 or proof of viable long-form speech/music/SFX
generation. Keep PR #1 **DRAFT / UNMERGED**. No release, pre-release,
tag, trained weights or waveform artifacts should be published.
