# Verified vectorized float64 and complete fast DSP experiment

**Evidence: 17 November 2026 · [four-host GitHub Actions run #44068924332](https://github.com/sanskarIN/sonumio/actions/runs/44068924332)**

## Result

**Four of four** independent Ubuntu CPU jobs succeeded. The research
used NumPy 2.4.6 on every runner, but different CPUs. The isolated
`research/vectorized_convolution_candidate.py` returned **exactly the
same full float32 convolution SHA256s** as the separately verified
Python binary64 scalar `math.fsum` reference, for all four
original procedural convolution sources and all four source seeds
(**16 cases**). On the selected scalar reference positions,
maximum observed float32 difference was **0.0** for all sources.

The independently implemented
`research/fast_dsp_candidate.py` then generated
**all 84 original synthetic train/validation recordings** (6
individual sources + 15 pairs, seeds 0, 1, 101 and 102)
as 1.5-second 24-kHz waveforms, original FIR resampling,
and final 8,192-sample 8-kHz crops. It preserved the
**exact SHA256 aggregates** of the previously verified scalar
candidate across all four CPU jobs. Each host repeated
the full experiment in two independent Python processes;
each passed the same-host byte agreement checks.

All four jobs also independently required the original
Sonumio DSP generator to match the original neural training
data crop exactly for each source and seed, *without*
using this new candidate for model training.

## Independent environments

| Runner | Reported host CPU | Candidate exact full source SHA reference | Fast 84-recording scalar-reference match |
| --- | --- | --- | --- |
| Ubuntu 22.04, instance 3 | AMD EPYC 7763 | 4/4 sources | True |
| Ubuntu 22.04, instance 4 | AMD EPYC 9V45 | 4/4 sources | True |
| Ubuntu 24.04, instance 1 | AMD EPYC 9V45 | 4/4 sources | True |
| Ubuntu 24.04, instance 2 | Intel Xeon 6975P-C | 4/4 sources | True |

The actual physical CPUs available to hosted Actions may change.
This table records only what job logs reported for this run.

## Candidate fast complete-pipeline SHA256 aggregates

Hashes are SHA256 of the **concatenated per-recording SHA256
digest bytes**, ordered by source combination and seed;
they do not identify a published WAV or downloadable dataset.

| Stage | Training (42) | Validation (42) |
| --- | --- | --- |
| 24-kHz candidate sound | `ee2c324eb111cd60ae4451fea8e769ae2fc6888eeac5e00124c5b64a910129af` | `3dcc50e5949675b03c3cd3edeb8f0614727a2a98dacc1c405f9cd4ed8cd3d5e3` |
| 8-kHz 8192-sample crop | `3b0f97711eed11348c8fde76c60aef73ba549d1ee434d9a2c257b9f612dd4de3` | `8a5676f2de55435952b481df218e5ea9d872fcee4729e2adda3fd0a309984cb3` |

**These exact byte fingerprints matched on all four hosts** and
match the existing `research/candidate_reference.py`
golden scalar corpus. Original host-sensitive NumPy float32
rendering was not altered; its hashes are not expected to
be stable across all machines.

All full fast candidate crop comparisons to the original
in this run had maximum absolute error **no greater than
1.1920928955078125e-7**. These tiny differences do
not on their own prove subjective audio quality, perceptual
equivalence or real-world generation ability.

## What the evidence does and does not establish

We have verified a **reproducible vectorized alternative for this
first-party synthetic corpus on NumPy 2.4.6 and four CPU runners**.
This supports the next step of measuring performance and testing
the algorithm across NumPy versions.

It does **not** prove:
- identical bytes on all CPUs, operating systems or NumPy versions;
- safe behavior for arbitrary new waveforms/filters/durations;
- performance under realistic concurrent inference workloads;
- that this tiny synthesized corpus supports robust natural-language
  or long-duration generative audio;
- quality or licensing of real-world recordings, model weights,
  music, speech or source rights.

The newly added CPU workflow now **asserts** the verified
four-source and 84-recording corpus hashes match their
frozen scalar references. A separate NumPy compatibility matrix
is exploratory until results are obtained; its success must be
verified against explicit reported hashes and not inferred from
this same-version run.

The original procedural generator, original synthetic datasets,
original model experiments, checkpoint formats and public
service remain untouched. This candidate is a separate experiment.

**No Sonumio model generation is being published. PR #1 stays
draft and unmerged. No releases, pre-releases, tags, WAVs,
weights, or production deployment.**
