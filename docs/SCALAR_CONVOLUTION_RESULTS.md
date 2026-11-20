# Measured scalar convolution candidate — first multi-host result

**17 November 2026 · development-only evidence · no Sonumio-1 claim**

The independent [four-job CPU research matrix #44059208693](https://github.com/sanskarIN/sonumio/actions/runs/44059208693)
ran the [candidate experiment](SCALAR_CONVOLUTION_CANDIDATE.md)
on commit `480345109cbdf65869dfb5c918a2a9c6f90acf70`.
**All four matrix jobs succeeded**, each repeating the probe in a
separate Python process and verifying exact same-host fingerprints.
The original procedural synthesizer and model datasets were **not modified**.

## The observed environments

All four jobs reported NumPy **2.4.6**, and their respective runner
images and CPU identifiers were:

| Matrix instance | Image | CPU model reported | Original rain NumPy SHA family | Scalar candidate sampled SHA |
| --- | --- | --- | --- | --- |
| 1 | Ubuntu 24.04 | Intel Xeon 6975P-C | baseline | equal |
| 2 | Ubuntu 24.04 | Intel Xeon 6975P-C | baseline | equal |
| 3 | Ubuntu 22.04 | AMD EPYC 9V74 | baseline | equal |
| 4 | Ubuntu 22.04 | AMD EPYC 9V74 | **different** | equal |

Importantly, **both AMD jobs identified as EPYC 9V74** yet had
different original NumPy convolution hashes for rain, thunder and
wind. The divergent NumPy full-output rain SHA prefix `45bdecdb`
is the same value reported in the earlier 16 November
[EPYC 7763 experiment](DSP_CONVOLUTION_RESULTS.md), but the
**17 November matrix did not contain a CPU identified as EPYC
7763**. Therefore, do **not** attribute the discrepancy to CPU
model alone. The environment/runtime cause remains unproven.

## Exact source-level evidence

For each of four sources × four seeds (0, 1, 101, 102),
the experiment re-created the original first-party float32
convolution inputs and kernels and compared their SHA256 digests
to the unmodified source oracle. It then evaluated exactly
**264 positions of 36,000** per source recording, not the
entire output. Selected positions are deterministic and include
both boundaries and positions around kernel offsets.

| Source | Scalar candidate digest, aggregated across four seeds | Maximum observed absolute difference versus original NumPy output at sampled positions |
| --- | --- | ---: |
| Rain | `eb64ccac6da346aeaed15c0a8e5fedfaa8cd9a74c7ba93948ea3fe22b83724cc` | 5.96 × 10⁻⁸ |
| Thunder | `b1d721768273a6cfcbcf8124dd5f8c04e355470acba2944725f51d44a267ec71` | 2.98 × 10⁻⁸ |
| Wind | `1bb76e2e5a8a2d50e5c0dd3e568b8224eaa4eb6d3f690a064104854a424c8c10` | 2.98 × 10⁻⁸ |
| Water | `557bd85a685e1a9cb2486cdfbb1baf5061aacd8c0947278829f3d1863296be9d` | 5.96 × 10⁻⁸ |

**All four candidate aggregated digests were exactly identical
on all four jobs.** The original NumPy selected digests for
rain, thunder and wind differed on instance 4; water was
stable. These metrics reflect the logged **sampled positions
only** and do not provide an error bound for the other
35,736 frames of each recording.

The scalar candidate promotes original float32 inputs to Python
binary64 scalar products, accumulates using `math.fsum`, and
casts back to float32. Small differences against NumPy are
expected and were measured; matching candidate hashes are
consistent with reproducibility on the tested configurations,
not a universal guarantee.

## What this DOES and does NOT establish

**Supported:** the original NumPy behavior can differ even when
two GitHub jobs report the same broad CPU model. The proposed
reference computation produced byte-identical **selected samples**
on these four sampled CPU jobs. The original source-data
input/kernel fingerprints were explicitly validated.

**Not established:** a full 36,000-frame deterministic
alternative, bitwise equality of complete 8-kHz research data,
cross-version/runtime guarantees, acceptable training/serving
throughput, audio quality, prompt adherence, or whether a future
learned generative model benefits. These are numerical
reproducibility observations, **not Sonumio-1 milestones**.

## Next experimental gates

1. Preserve the original wave renderer and earlier checkpoint
   fingerprints; do not replace a working data recipe silently.
2. Gate independent fixed-byte golden test vectors for float32
   scalar convolution on every CPU matrix job, covering four
   tap counts and both full/same alignments.
3. Benchmark **full-length scalar output**, peak absolute waveform
   error and throughput on the independently divergent CPU
   environments, including the host-specific NumPy paths.
4. Validate complete 24-kHz source rendering and 8-kHz
   resampling/cropping, source mixtures, train/validation splits
   and their SHA-based leakage safeguards.
5. If performance is unacceptable, investigate explicitly
   specified fixed-point or numerically stable acceleration,
   keeping versioned data fingerprints and perceptual evaluation
   separate from merely matching numerical hashes.

**Policy:** draft PR #1 stays unmerged. No pre-releases, releases,
tags, WAV downloads or model-weight distribution. This report
uses only public workflow numerical logs.
