# DSP reproducibility across NumPy versions — measured limits

**17 November 2026 — experimental source/AI-training separation retained**

## Verified same-version result

[Four-host CPU workflow #44068924332](https://github.com/sanskarIN/sonumio/actions/runs/44068924332)
verified the separate vectorized binary64 convolution candidate
against the scalar fsum reference on **all 16 original sound
convolution probes**. Every tested CPU used NumPy **2.4.6**.
The complete fast 24-kHz generation pipeline, historical FIR
resampler and final 8-kHz crop also reproduced the scalar
reference across **all 84 synthetic recordings**, on all
four tested CPU runners.

[CPU research workflow #44068924393](https://github.com/sanskarIN/sonumio/actions/runs/44068924393)
passed **311 research tests**, and the standard Python matrix,
web checks and private synthetic training also passed.
No production or training-data recipe was changed.

## Cross-version experiment: newly identified incompatibility

The independent
[NumPy compatibility matrix #44075335237](https://github.com/sanskarIN/sonumio/actions/runs/44075335237)
executed the *same source code* and complete fast 84-recording
research protocol under three supported NumPy releases:

| NumPy version | Hosted CPU observed | All four fast source convolutions match scalar SHA? | Complete 84-recording candidate matches scalar reference? |
| --- | --- | --- | --- |
| 1.26.4 | AMD EPYC 9V74 | Yes | **No** |
| 2.2.6 | Intel Xeon 6975P-C | Yes | Yes |
| 2.4.6 | AMD EPYC 9V45 | Yes | Yes |

**Important:** the three matrix entries ran on different CPUs.
The observed difference is associated with NumPy 1.26.4
*in that run*, but is not yet a controlled proof that
the version alone caused the divergence. The source
convolution hashes were nevertheless bitwise equal across all
three environments, narrowing the remaining numerical issue
to other stages of complete DSP processing.

### Exact fast complete-pipeline SHA256 aggregate differences

Each digest aggregates the byte digests of **42 distinct
first-party synthetic source recordings per split** in a
fixed source/seed ordering. The values do not refer to public
audio files or neural checkpoints.

| NumPy | Train 24 kHz | Train 8 kHz cropped |
| --- | --- | --- |
| 1.26.4 | `0431a25de64f7c174191610a7195484eaa4abe74de756613836e9b15f75031fc` | `1da65a6bd7b7369195d632da8d8e8759af61884215d8c28d69f6bad26f12c4ab` |
| 2.2.6 and 2.4.6 | `ee2c324eb111cd60ae4451fea8e769ae2fc6888eeac5e00124c5b64a910129af` | `3b0f97711eed11348c8fde76c60aef73ba549d1ee434d9a2c257b9f612dd4de3` |

| NumPy | Validation 24 kHz | Validation 8 kHz cropped |
| --- | --- | --- |
| 1.26.4 | `6b662ed35df9cba13d9f8f03a448382f104360ef5a92d9fbfcff6133ebc3ccae` | `f79974950761ac0161a0a8dbe066ef53a678702d677a8f7d97a11b91a53fa075` |
| 2.2.6 and 2.4.6 | `3dcc50e5949675b03c3cd3edeb8f0614727a2a98dacc1c405f9cd4ed8cd3d5e3` | `8a5676f2de55435952b481df218e5ea9d872fcee4729e2adda3fd0a309984cb3` |

This demonstrates that **a cross-host, single-library-version
success does not establish cross-library-version exact
reproducibility**, even when the source-level convolutions match.

## New same-runner stage investigation

A separate `research/numpy_stage_probe.py` records only
SHA256 of 24-kHz time grids, individual source layers,
mix accumulation, fade ramp, post-fade mix, normalization,
FIR filter kernel/filtered output, downsampled 8-kHz
waveform, and final seeded crop.

The [same-runner NumPy stage-difference workflow](https://github.com/sanskarIN/sonumio/actions/workflows/dsp-numpy-stage-diff.yml)
runs the exact same **40 first-party source/seed test
cases** twice in fresh Python processes on one Ubuntu
runner. It starts with NumPy 1.26.4, saves only numerical
JSON stage SHA digests, then replaces NumPy with 2.4.6
on that same runner, and records the earliest differing
stage per source/seed.

**The same-runner stage audit has now succeeded.** It found
28/40 exact recordings and 12/40 differing recordings,
all involving the ambient/drone sound. After fixing
JSON's alphabetic stage-key ordering, the first
*executed* divergent stage is `layer_drone` for all
12 cases. The separate [original ambient-expression
probe](AMBIENT_SINE_RESEARCH.md) pins the first arithmetic
difference to the float32 `sin_110` result (and
also observes a different `sin_220`). The time
grid, source float64 convolution kernels, fade
ramp and original FIR kernel fingerprints matched
on the same tested CPU.

## New unseen-seed portability checks

`research/heldout_dsp_probe.py` tests five
source seeds **3, 7, 17, 29 and 41**, disjoint
from the historical train/validation seeds 0, 1,
101 and 102, across all **21 source combinations**.
It records **105 additional synthetic first-party
recording fingerprints** without publishing any
audio, training new weights or introducing a new
train/validation split. The [independent four-runner
held-out result](HELDOUT_DSP_RESULTS.md) subsequently
passed all **105** cases with identical full 24-kHz
and final cropped 8-kHz SHA256 aggregates across
all tested CPU hosts at NumPy 2.4.6. These hashes
are now frozen in the matrix as candidate-only
regression gates.

Different SHA values are treated as an experiment
to investigate, not suppressed or silently rewritten.

## Next engineering gates

1. Confirm whether isolated **float64 NumPy sine**
   or **Python scalar math.sin** candidates produce
   identical ambient bytes across NumPy versions
   and independently provisioned CPU hosts.
2. Only after a candidate is proven, consider a
   separately versioned *complete* DSP variant
   through mixing, normalization, unchanged FIR
   and final 8-kHz crop, without modifying any
   historical neural training data or public
   procedural generator.
3. Extend the verified five-seed holdout probe
   to NumPy 1.26.4 and 2.x under the same host,
   and document any remaining source-family drift.
4. Re-run full numerical bit tests after any fix,
   and measure latency and native peak memory
   in independently provisioned environments.
5. Preserve independent acoustic listening, real
   rights-cleared data, bilingual caption fidelity
   and longer-duration neural-model evaluation.

**This is source-level numerical research. The original
generator, neural data, weights, releases and project
tag state have not been modified. Draft PR #1
remains unmerged and unreleased.**
