# Entire source-convolution reproducibility candidate

**Development-only research protocol · 17 November 2026 · not a production renderer**

This is the follow-on to the first [264-position scalar study](SCALAR_CONVOLUTION_RESULTS.md).
The first study found matching candidate sampled-output SHA256s across
four hosted CPU jobs despite one host producing different original
NumPy convolution hashes. It did not test all output frames.

## New full-frame verification

`research/full_convolution_candidate.py` extends the exact same
bounded float32-input / scalar binary64-product / `math.fsum` /
single float32-rounding candidate to **every output position**,
in 1,024-output-frame blocks. The output bytes do not depend on
chunk boundaries, as checked by regression tests.

The research protocol covers 4 first-party sources × 4 original
seeds (0, 1, 101, 102):

| Sound | NumPy mode | Input frames | Kernel taps | Candidate full-output frames | Historical original crop |
| --- | --- | ---: | ---: | ---: | ---: |
| Rain | full | 36,000 | 180 | 36,179 | First 36,000 |
| Thunder | same | 36,000 | 140 | 36,000 | All 36,000 |
| Wind | same | 36,000 | 60 | 36,000 | All 36,000 |
| Water | same | 36,000 | 12 | 36,000 | All 36,000 |

Before evaluating candidate bytes, each case independently invokes
the **unchanged** original `isolate_one` source oracle and verifies
identical SHA256s for its float32 input and kernel. It also checks
that NumPy's entire original output or legacy crop matches the
historical source-convolution SHA from that oracle.

Every 16-case report stores:

- Exact SHA256 of the original and candidate complete convolution
  outputs, and separate SHA256s of the historical 36,000-frame crop.
- Exact counts of float32 samples that differ between the two
  arithmetic implementations.
- **Entire-output** maximum absolute difference and RMS difference,
  after promoting both float32 outputs to float64 for comparison.
- Strict source/seed coverage, dimensions, hash syntax, numerical
  finiteness and report-aggregate consistency checks.
- Allowlisted CPU/NumPy version metadata; no waveform array values,
  WAV files, neural checkpoints or model weights.

## Verification commands

```bash
python -m pytest -q tests/test_full_convolution_candidate.py
python -m research.full_convolution_candidate \
  --output checkpoints/private-research/full-source-convolution.json
```

The independent four-runner workflow
`.github/workflows/dsp-trace-matrix.yml`
executes the 16-case report twice per CPU runner, checks same-host
exact equality, and prints only source-level digests and error metrics.
The CPU research suite separately runs full-frame chunk equivalence,
per-source original-input gates, mutation and oversize regressions.

## How to interpret the next matrix

The complete-convolution candidate may produce matching bytes where
the original NumPy implementation does not. If so, that establishes
only **full first-party source-convolution arithmetic identity**
for those tested inputs and runtime environments. It does **not**
prove that original whole generated WAVs are identical, because
downstream source envelopes, normalization, multi-source mixing,
resampling and cropping can differ. It also does not yet guarantee
a stable input RNG across library changes.

A separate *versioned, independent* downstream full-corpus
experiment and objective listening study are necessary before
this alternate arithmetic can be considered for actual model
training or inference.

If results are inconsistent, investigate the first divergence
without changing the original source recipe or earlier experiment
data to make comparisons look better.

**No release, pre-release, tag, published audio files or trained
model weights. Draft PR #1 must remain open, draft and unmerged.**
