# Full 84-recording candidate DSP reproducibility — measured multi-CPU result

**Verified 17 November 2026 · experimental first-party numerical research**

[Independent CPU matrix #44063872705](https://github.com/sanskarIN/sonumio/actions/runs/44063872705)
executed `python -m research.complete_dsp_candidate --scope full`
at development commit
[`742d71d0`](https://github.com/sanskarIN/sonumio/commit/742d71d077a0d770a8b73bb171b6aa995ff8fcdb).
**4/4 independently provisioned jobs succeeded**, each processing
**all 84 unique first-party synthetic source recordings**:
42 train recordings (seeds 0 and 1) plus 42 validation recordings
(seeds 101 and 102), covering six single sources and 15 two-source
combinations. English/Hindi captions share the corresponding
recording and are not counted as separate waveforms here.

## Observed CPU environments

| Workflow runner | Image | Reported CPU | Full 84-case job outcome |
| --- | --- | --- | --- |
| 1 | Ubuntu 24.04 | AMD EPYC 9V45 | Success |
| 2 | Ubuntu 24.04 | Intel Xeon 6975P-C | Success |
| 3 | Ubuntu 22.04 | AMD EPYC 7763 | Success |
| 4 | Ubuntu 22.04 | AMD EPYC 7763 | Success |

All four logs reported **NumPy 2.4.6** and the same
candidate *entire-corpus* SHA256 aggregate values for both raw
24-kHz waveforms and the final exact 8192-frame 8-kHz crops.

## Measured exact SHA256 aggregates

Each aggregate below is SHA256 of the concatenated per-recording
SHA256 bytes in canonical source/seed order. It is NOT a hash of a
single WAV file or a published audio dataset.

| Candidate pipeline stage | Train: 42 original synthetic recordings | Validation: 42 original synthetic recordings |
| --- | --- | --- |
| Final 24-kHz waveform | `ee2c324eb111cd60ae4451fea8e769ae2fc6888eeac5e00124c5b64a910129af` | `3dcc50e5949675b03c3cd3edeb8f0614727a2a98dacc1c405f9cd4ed8cd3d5e3` |
| Final 8-kHz 8192-frame crop | `3b0f97711eed11348c8fde76c60aef73ba549d1ee434d9a2c257b9f612dd4de3` | `8a5676f2de55435952b481df218e5ea9d872fcee4729e2adda3fd0a309984cb3` |

**All four runners matched every candidate aggregate exactly.**
On the EPYC 7763 hosts, the original NumPy renderer continued
to yield different float32 source/render hashes, while the
separate scalar candidate's final-crop aggregates remained
identical to Intel Xeon and EPYC 9V45. This is evidence for
numerical reproducibility on the **tested recipe/inputs/host
environments**, not a universal processor or NumPy-version
guarantee.

## Difference measurements against the unchanged original

The maximum absolute candidate-versus-original sample difference
over the full tested source waveforms was:

| Stage | Train maximum | Validation maximum |
| --- | ---: | ---: |
| 24-kHz waveform | 1.7881393432617188 × 10⁻⁷ | 1.1920928955078125 × 10⁻⁷ |
| 8-kHz final crop | 1.1920928955078125 × 10⁻⁷ | At most 8.940696716308594 × 10⁻⁸ |

These are maxima over the **84 experimental waveforms on the four
tested runners**, not a bound on arbitrary audio, independent
listening preference, model quality or future library versions.
Matching hashes are byte-level evidence and **do not** certify
license validity, authentic natural sound, safety or semantics.

## Automated future regression guard

`research/candidate_reference.py` now records exactly these four
verified candidate aggregate SHA256s, the expected 42/42 split
counts, the corpus recipe identifier and the public reference
workflow. The independent CPU matrix validates the fully generated
research report and then requires its candidate aggregates to
match these known values. The original host-sensitive NumPy
SHA256s are intentionally **not** pinned to a single host-dependent
value. A failed numerical reference gate is an invitation to
investigate and **version** an intentional changed recipe, not to
silently overwrite historical experiments.

## What remains unverified

The reference is not evidence that the scalar implementation is
fast enough to serve generation requests. A separate
[paired in-memory performance benchmark](DSP_CANDIDATE_BENCHMARK.md)
measures legacy-versus-candidate rendering latency and avoids
asserting a speed threshold without measured evidence.

Independent remaining work includes rights-reviewed real audio,
human listening and quality tests, more than 1.024-second model
research, multi-lingual prompt understanding outside six source
families, private original neural model training comparisons,
production security, resource isolation and GPU tests.

**Keep draft PR #1 unmerged. No release, prerelease, tag,
weights, downloadable WAVs or production deployment was created.**
