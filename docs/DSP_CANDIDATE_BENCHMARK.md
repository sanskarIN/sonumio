# Original versus scalar DSP candidate: bounded performance protocol

**17 November 2026. Research-only. Not a model speed, quality or release benchmark.**

The separately versioned [complete DSP candidate](COMPLETE_DSP_CANDIDATE.md)
has shown exact final 8-kHz crop hashes for 40 recordings across four
CPU hosts. However, the scalar binary64 `math.fsum` reference uses Python
loops and may be much slower than vectorized `numpy.convolve`. Numeric
reproducibility is useful only if its computing cost is made explicit
before considering optimized alternatives.

## Experiment and limitations

Use `benchmarks/dsp_candidate.py` to measure **in-memory 1.5-second
24-kHz procedural rendering**, with identical source combinations and
seed for each variant. Both paths preserve all DSP sound semantics.
The baseline is the original `dsp.generate`; the candidate replaces
only source convolutions with the separately tested scalar arithmetic.

- Fixed **36,000 float32 output frames** per render; no loading models.
- Use a single explicit seed, with **1–8 measured paired repetitions**
  and **0–2 warm-ups** per method. The execution order alternates
  baseline-then-candidate versus candidate-then-baseline to reduce
  simple order bias.
- Only the actual renderer call is timed using `perf_counter`.
  Waveform finiteness validation, SHA256, error calculations and JSON
  writing are **excluded** from the timed region.
- Report median, min, max, nearest-rank p95, median real-time factor
  and **candidate median / baseline median**. A ratio greater than
  one means the candidate was slower in that environment.
- Require repeated identical seed/variant renders to have identical
  exact float32 SHA256 on the same host. Include full-array maximum
  candidate-versus-original float32 difference.
- The output records allowlisted CPU/NumPy metadata and numbers,
  **not waveform samples, WAV artifacts, model weights, training data
  or private recordings**.

Choose from six single-source sounds and four explicitly declared
source pairings. Other prompts cannot be supplied to this research
microbenchmark. An unpinned hosted runner's available load,
processor frequency and CPU architecture may vary; no speed
generalization across machines is warranted. A 1- or 2-sample
observation is primarily a CI smoke measurement, not a reliable
performance distribution.

The benchmark **does not measure** neural generation, full 8-kHz
resampling/cropping, Python/NumPy peak native memory use, whole-model
throughput, quality or training performance. Its timer does not
measure WAV writing or JSON reporting. No throughput thresholds
have been installed into CI that would reject a numerically
correct experiment based on incidental runner slowdown.

## Usage

```bash
python -m benchmarks.dsp_candidate --case rain --repeats 3 --warmups 1
python -m benchmarks.dsp_candidate --case rain-thunder \
  --seed 17 --repeats 5 --warmups 2 \
  --output checkpoints/private-research/dsp-bench-rain-thunder.json
python -m pytest -q tests/test_dsp_benchmark.py
```

The CPU research workflow executes a small two-repeat rain case
with **no** warm-up only to establish that timings can be recorded
without saving audio. The CI source alone does not imply completed
benchmark measurements; inspect the workflow logs for actual values.

## Engineering decision gates

The scalar Python implementation is intentionally a diagnostic
reference, not a production solution. Before promotion to a new,
explicitly versioned source dataset or inference backend:

1. Confirm the entire 84-recording train/validation dataset has
   exact final-crop hashes on divergent CPU hosts, including duplicate
   waveform and split-overlap checks.
2. Benchmark single-source and two-source pairs on representative
   desktop CPUs and quantify median, p95 and real-time-factor limits.
3. Measure actual peak resident memory in a separate isolated process;
   Python `tracemalloc` alone misses some native NumPy allocations.
4. Investigate an optimized fixed-reduction-order / fixed-point
   convolution implementation with separately specified float32
   rounding and numeric or bitwise guarantees.
5. Repeat the full 24-kHz-to-8-kHz corpus, anti-leakage and waveform
   quality tests for the new optimized kernel; record all changed
   exact dataset fingerprints under a **new** recipe identifier.
6. Keep perceptual listening, diverse rights-cleared real-world
   captions, bilingual prompt adherence, long-form generation and
   private model training studies as distinct research requirements.

**PR #1 remains draft and unmerged. No prereleases, stable releases,
tags, downloads of WAVs, or trained model weights are authorized.**
