# Research-only DSP process memory and latency measurement

**17 November 2026 · Experimental Sonumio CPU source rendering**

`benchmarks/dsp_resource_profile.py` evaluates five first-party
24-kHz procedural DSP methods for the **same 1.5-second sound**
and seed, in **five separate bounded Python subprocesses**:

1. Unchanged original procedural `dsp.generate`.
2. Existing first-party fast float64 source convolution.
3. Existing first-party fast double ambient sine candidate.
4. New forest-aware double float64 sine candidate.
5. New forest-aware Python scalar `math.sin` candidate.

This is a numerical research benchmark, not an in-production
audio API, long-duration music renderer, GPU workload,
speech evaluation or neural-model inference benchmark.

## Timing and memory methodology

For each allowlisted source case, seed and renderer:

- Load the variant and its needed research dependencies
  **before timing** (import costs excluded from render timings).
- Record the process resource high-water mark after imports,
  optionally render 0–3 warmups, then render 1–9
  fixed-seed repetitions with monotonic `perf_counter`.
- Validate fixed finite 36,000-frame float32 sound and
  compute SHA256 **outside** the timed interval.
- Report median, minimum, maximum and nearest-rank p95
  per-render elapsed times and one exact SHA256 per variant.
- Require the same variant/seed to produce
  identical sound SHA256s across within-process repetitions.
- Report Linux/macOS `RUSAGE_SELF.ru_maxrss`
  whole-process peak resident set as KiB
  (native Linux ru_maxrss is KiB, macOS is bytes
  and is normalized to KiB). On Windows and
  unsupported runtimes report `null`, **not zero**.
- Expose the peak seen after imports and a
  nonnegative increase during the remainder
  of process work, with an explicit caution
  that this difference **is not per-render peak
  live allocation**. Values include
  NumPy/Torch/CPython dynamic library
  footprints, process startup state,
  caches, native and Python allocations.

**Important validity caveat:** different candidate modules
load different imported dependencies. The process-level high-water
numbers describe real isolated research *process* footprints,
but cannot be used to claim one DSP arithmetic operation itself
uses that many bytes, or to rank GPU/VRAM/streaming inference.
CPU dispatch, thread scheduling, OS caching, warmup length,
Python/NumPy versions and hosted CPU instances may also
change timings. A three-repeat benchmark is a diagnostic,
not a reliable production performance guarantee.

## Supported local commands

```bash
python -m benchmarks.dsp_resource_profile \
  --case thunder-forest --seed 41 --repeats 3 --warmups 1 \
  --output checkpoints/private-research/dsp-resources.json

python -m pytest -q tests/test_dsp_resource_profile.py
```

The CLI uses fixed allowlisted cases and variant names.
It never executes shell-provided code, writes audio,
opens user private recordings, downloads models or
publishes any weights. The optional file is
**numerical JSON only**, written only when an explicit
`--output` path is supplied. A research CI step
executes an abbreviated thunder+forest profile
and prints median render seconds and
whole-process peak RSS per method; **those
new measurements remain unverified until the
relevant CI step finishes**.

For accurate end-to-end inference budgets,
an eventual production evaluation will need
trained and rights-cleared Sonumio models,
CPU/GPU memory watermarking, batch and
streaming variation, concurrency,
cold-start and sustained-load profiling
and a meaningful sound-quality study.

The source generators and training data are
unchanged. Draft PR #1 remains unmerged; no
release, pre-release, tags, sound files or
model weights have been published.


## First measured CPU example (completed CI)

[Research workflow #44097168528](https://github.com/sanskarIN/sonumio/actions/runs/44097168528)
completed successfully with **411 passed tests**,
including this new isolated-process
benchmark. For a 1.5-second
`thunder-forest` prompt, first-party
seed 41 and two timed repetitions
per variant, one hosted CPU reported:

| Separate process | Median render seconds | Whole-process peak RSS (KiB) |
| --- | ---: | ---: |
| Original procedural | 0.001667 | 45,400 |
| Fast float64 source | 0.001649 | 299,796 |
| Portable ambient float64 | 0.001562 | 299,948 |
| Forest-aware float64 sine | 0.001708 | 299,952 |
| Forest-aware Python scalar sine | 0.002480 | 299,856 |

The very small number of timed
repetitions cannot justify sustained
throughput or benchmark ranking.
The substantially larger process
high-water mark for research variants
reflects extra imported dependencies
(such as PyTorch), not 255 MB of
additional audio samples or the
isolated algorithm's live memory.
A production source-only build
must be profiled separately without
research imports before making
deployment recommendations.
