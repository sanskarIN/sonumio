# Verified source of cross-CPU DSP byte differences: convolution operation

**Sonumio first-party research · 16 November 2026 · numerical evidence only**

The previous [stage-trace matrix](DSP_PIPELINE_TRACE_RESULTS.md)
showed rain, thunder and wind producing different original
waveform bytes on an AMD EPYC 7763 CPU, while the other three
sound families matched. The follow-up targeted probe has now
identified the **first differing numerical operation** on that host.

## Reproducible CI evidence

- Source-code experiment:
  [commit 0fa50577](https://github.com/sanskarIN/sonumio/commit/0fa50577d48c450383207034339c7a0d72957ee5).
- [Independent CPU convolution + stage matrix #43882937960](https://github.com/sanskarIN/sonumio/actions/runs/43882937960):
  **4/4 jobs successful**; each job repeated its input synthesis in
  two independent Python processes and verified exact stage hashes.
- [CPU research tests #43882938356](https://github.com/sanskarIN/sonumio/actions/runs/43882938356):
  **196 tests passed**, including original source-layer reconstruction,
  input/kernel hashing, scalar reference checks and forgery rejection.
- Python tests, web checks and the private paired epsilon/velocity
  model-training workflow on the same commit also **passed**.
- Original DSP and 8-kHz research training data are **unchanged**.
  No WAV files, model weights or release artifacts were uploaded.

All four convolution jobs used NumPy 2.4.6, the same repository
commit and same declared original recording seeds, but the
individual GitHub-hosted execution environments were not a fully
controlled hardware/runtime experiment.

| Matrix job | Reported CPU | Exact convolution stage outcome |
| --- | --- | --- |
| 1 (Ubuntu 24.04) | Intel Xeon 6975P-C | Baseline |
| 2 (Ubuntu 24.04) | AMD EPYC 9V74 | Baseline |
| 3 (Ubuntu 22.04) | **AMD EPYC 7763** | **Three long convolution outputs differ** |
| 4 (Ubuntu 22.04) | Intel Xeon Platinum 8573C | Baseline |

## Isolated input/kernel/convolution findings

The [original DSP convolution probe](DSP_CONVOLUTION_ISOLATION.md)
replicated the original source functions **exactly** and hashed
16 cases: 4 sounds × 4 original recording seeds.

| Original source | Kernel taps | Input float32 SHA | Kernel float32 SHA | NumPy convolution output SHA | Final source-layer SHA |
| --- | ---: | --- | --- | --- | --- |
| Rain | 180 | Match | Match | **Differs on 7763** | **Differs** |
| Thunder | 140 | Match | Match | **Differs on 7763** | **Differs** |
| Wind | 60 | Match | Match | **Differs on 7763** | **Differs** |
| Water/ocean control | 12 | Match | Match | Match | Match |

Input arrays include the actual seeded random normal values or
rain impulse positions and weights; the kernels are the original
float32 smoothing or rain decay kernel. Therefore the difference
appears **after** identical inputs/kernels are passed to
\`numpy.convolve\`, before the source's subsequent envelope,
normalization or sound mixing.

For example, complete SHA256 hashes of the aggregated source
convolution outputs across seeds 0, 1, 101 and 102:

| Source | Baseline SHA256 | AMD EPYC 7763 SHA256 |
| --- | --- | --- |
| Rain | \`4a60f2b89bcab34654a5c63735245400c17f340e36ff1cc9db926d990911a29e\` | \`45bdecdb58fab4d56809876f8e25f18f8dc0585bd6d46b5debd377cea9eaf0e0\` |
| Thunder | \`a8d8af6320bedd442757ea7b428f003994a14c636e5b119f8f9653775bd16ce7\` | \`b195f0994fdd4f8429ce10448ac0b81a76d7fba9726bddf25eca8d1fb63f8ca9\` |
| Wind | \`53334e023409d2be25d44174027eceee1ab42fdf8e760aae733f916c6047e21a\` | \`d65bac138e1d65a8a7e325500a87b23305ddbd7140ad019a0b4962691224eb24\` |
| Water | \`6d9a8a0aed1c8ab873c0d160d124710f92387629da418394ce71f85ba8aeaa9f\` | Same as baseline |

As an independent numerical control, the probe computed selected
convolution values with Python double products and \`math.fsum\`.
Those eight selected positions per recording had **identical
scalar-reference hashes** across all four CPU jobs.
At those sampled positions, the maximum absolute difference
between the scalar reference and actual NumPy convolution
was around **1.2–2.2 × 10⁻⁸**.

This is strong evidence of **host-sensitive float32 convolution
rounding/accumulation**, but does not prove a specific CPU instruction,
vector width, compiler optimization or NumPy C routine as the exact
implementation cause. Hash differences do not quantify an entire
36,000-sample waveform's maximum deviation; the sampled scalar
differences are limited to eight positions per recording.

## Practical engineering decision

1. **Preserve the original DSP recipe.** Do not silently modify
   historic synthetic recordings, their hashes or the earlier neural
   training evaluations.
2. Keep exact **within-job** training/validation source SHA gates,
   bilingual pairing checks and same-input model comparisons.
3. When cross-host bytewise identity becomes a model-training
   requirement, evaluate a **newly versioned** reproducible
   convolution/data recipe. Options include specified accumulation
   order or integer/fixed-point convolution, with explicit
   performance, waveform-error and listening-quality trade-offs.
4. Build operation-specific tests on CPUs where the original
   60-/140-/180-tap NumPy convolutions differ and verify the new
   recipe against them, not just the hosts that already match.
5. Do not equate 0.01-amplitude bucket hash agreement with
   exact bytes, audible equivalence or a maximum error bound.

**The core Sonumio generative models still need higher-quality
recordings, stronger codec/generator training, blind listening and
independent prompt relevance before any creator-quality claim.**
This numerical DSP finding is not Sonumio-1 readiness evidence.

**No PR merge, release, prerelease, version tag, downloadable WAV
or neural model weights publication was authorized or performed.**
