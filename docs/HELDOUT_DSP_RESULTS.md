# Held-out fast DSP reproduction — 105 additional source/seed cases

**Verified 17 November 2026 · first-party synthetic numerical research**

## Why the held-out study matters

The original 84-source waveform corpus covers source seeds 0, 1, 101
and 102. A candidate can match that fixed test grid without
necessarily reproducing every future seed.

`research/heldout_dsp_probe.py` therefore evaluates a **separate
five-seed, 21-source-combination probe**, seeds **3, 7, 17, 29 and
41**, for **105 additional complete 1.5-second, 24-kHz synthetic
recordings**. This is not a new neural training dataset or
a release. It does not change any existing audio generator, training
recipes, neural checkpoints, real-world audio rights or APIs.

Each sample uses the fast vectorized float64 convolution in the
research-only DSP candidate, then original mixing, fade,
normalization, anti-aliasing FIR, downsampling and an independent
seed-offset 8-kHz crop. It compares against the **unchanged**
official procedural DSP renderer. JSON diagnostics contain only
source identities, SHA256 hashes and finite error metrics, no
audio arrays, WAVs or weights. Exact candidate and legacy crop
duplicates are rejected; duplicate raw ambient/drone waves are
allowed before the seed-dependent crop.

## Independently verified cross-CPU result

The complete [four-runner workflow #44075765225](https://github.com/sanskarIN/sonumio/actions/runs/44075765225)
passed **4/4 independent CPU jobs**, each examining all 105
recordings. Runner-observed CPU models included:

- AMD EPYC 7763 (Ubuntu 22.04).
- AMD EPYC 9V45 (Ubuntu 24.04).
- Intel Xeon 6975P-C (Ubuntu 24.04).
- Intel Xeon Platinum 8573C (Ubuntu 22.04).

Every job reported **NumPy 2.4.6**, and every job independently
produced the **same candidate aggregate SHA256 fingerprints** of
complete 24-kHz samples and final 8-kHz crops for each of the
five unseen seeds.

| Held-out seed | Candidate 24-kHz aggregate SHA256 | Final 8-kHz crop aggregate SHA256 |
| ---: | --- | --- |
| 3 | `e1076d40c4d72e2b75bd3588f408c077d961d407d98c95c6c7c30d6f1893ef95` | `aab15df63345413c2c5075803d81194413a1bee5a5994799ea277af101a8765b` |
| 7 | `2840877f38f17e3ea0b39263cfa4533781473b2ef85688b1ee448962cdf8dda1` | `68991ec70a90f369947d11b318fbcf43a71be78dd38a515d73d907b31ef3d806` |
| 17 | `30df175004b4909e2cbb76971d6e0b0d3bb39179a45b3b2709355c680d4261b8` | `5eed92f868e9a03fa64c461ffd51ea690caaaca36848c77a699000dcb144d897` |
| 29 | `00776f177d1dfa0de268e89ca424f3f826f16128ccf57a1834f9b7f6b35393a9` | `db8a189f7fb0ea536670d9eeb54cd75abefe27917b94a72cadcf98c30a89718f` |
| 41 | `9f59d6cb67af7c2671e321fb75baa60ab968f114475977af89627af07d28858e` | `a76d3a12854b6d59b90d084cb40e79223876deebfc099526b8ad0b37bf673726` |

Each SHA256 value is computed from the concatenated per-recording
SHA256 **digest bytes** for the 21 source combinations in canonical
order, not from a published WAV or audio collection.

The resulting **189 first-party synthetic recording cases** (84
original + 105 held-out) have been tested across multiple CPUs
at NumPy 2.4.6. The cases are not 189 independent real-world
audio environments and are not evidence of usable text-to-audio
quality, semantic generalization or model safety.

The research-only
`verify_heldout_reference` function now checks the complete
105-recording report against these independently observed hashes,
and the CPU matrix calls it on both independently generated
reports per host. Original NumPy float32 SHA256s are deliberately
not pinned because host-sensitive differences were previously
observed.

## Limitations and next experiments

The [NumPy-version study](DSP_NUMPY_PORTABILITY.md) discovered
that NumPy 1.26.4 produces different complete DSP output
fingerprints from 2.2.6/2.4.6 in a separate experiment,
despite matching source-level float64 convolution hashes.
The held-out result **does not** resolve that cross-library
mismatch. Separate same-runner phase/sine and full-stage
localization work is underway. Other necessary experiments
include longer clips, native-memory peak measurement,
separate listening preference tests and rights-reviewed
natural audio and captions.

**Keep draft PR #1 unmerged and unreleased. No tags,
pre-releases, stable releases, WAV files, public model weights
or training-dataset substitutions.**


## Important provenance clarification: numerical holdout is not global model holdout

These five seeds were selected **outside the 84-recording latent-audio
source/seed grid**, not as globally untouched sources across every Sonumio
neural experiment. In particular, seed **3** is already part of
`research.text_audio.SYNTHETIC_TRAIN_SEEDS = (0, 1, 2, 3)`.
Consequently the 105 audio fingerprints are valid additional *numerical
reproducibility stress cases*, but **must not be used to claim neural-model
generalization, absence of training leakage or an independently held-out
model-quality evaluation set**. The research report now declares this overlap,
refuses a false global-holdout flag, and preserves the previously measured
byte fingerprints unchanged.
