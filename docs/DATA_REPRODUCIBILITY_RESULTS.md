# Actual multi-run DSP reproducibility findings

**Sonumio research audit · 16 November 2026 · synthetic-only**

The recording-level reproducibility code is now exercised by CI on
different GitHub-hosted CPUs. The observed cross-run float32 byte
differences are real; exact sample identity **must not** be assumed
across runners. This is a research integrity result, not a model release.

## Evidence and execution

| CI workflow | Run | CPU |
| --- | --- | --- |
| [Research tests (CPU)](https://github.com/sanskarIN/sonumio/actions/runs/43876282152) | SUCCESS; **182 passed** | AMD EPYC 9V45 96-Core Processor |
| [Private synthetic audio training (CPU)](https://github.com/sanskarIN/sonumio/actions/runs/43876282111) | SUCCESS; paired epsilon-v training | AMD EPYC 7763 64-Core Processor |
| [Python tests](https://github.com/sanskarIN/sonumio/actions/runs/43876282319) | SUCCESS | Independent workflow |
| [Web checks](https://github.com/sanskarIN/sonumio/actions/runs/43876282225) | SUCCESS | Independent workflow |

Code commit:
[\`0381fa4b\`](https://github.com/sanskarIN/sonumio/commit/0381fa4bbbb70f8a5a9bc27cb04ce428cd14fe8f).
Protocol, CLI and strict rules:
[DATA_REPRODUCIBILITY.md](DATA_REPRODUCIBILITY.md).

Both primary runs generated the same **declared** first-party DSP recipe:
42 unique training recordings from seeds 0, 1 and 42 held-out
recordings from seeds 101, 102; every recording has English/Hindi
captions for a total of 84 captions (42 bilingual pairs) per split. Same duration
1.024 seconds, sample rate 8 kHz, float32 arrays, and same project
source commit.

## Byte-level and coarse fingerprints

| Research input | Research runner: EPYC 9V45 | Training runner: EPYC 7763 | Equal? |
| --- | --- | --- | --- |
| Train exact dataset SHA256 (prefix) | \`0a0aedb1b8a0\` | \`95eb5929bffcc\` | **No** |
| Held-out exact SHA256 (prefix) | \`3cb5f4806c42\` | \`ff7c20fb6c42\` | **No** |
| Train aggregated 0.001-bucket SHA256 (prefix) | \`dd0ddee8ee47\` | \`db08216b5361\` | **No** |
| Held-out aggregated 0.001-bucket SHA256 (prefix) | \`8ff622e6bb53\` | \`8ff622e6bb53\` | Yes |
| Train aggregated 0.01-bucket SHA256 (prefix) | \`bc2d18c3397a\` | \`bc2d18c3397a\` | Yes |
| Held-out aggregated 0.01-bucket SHA256 (prefix) | \`47a5efb98556\` | \`47a5efb98556\` | Yes |

These are SHA256 of deterministic source/recording entries;
the **training dataset SHA256 also includes captions and source
metadata**, while the coarse summaries aggregate bucketized recording
hashes. The prefixes shown are for human-readable comparison; the
complete hashes appear in the CI logs.

**What this proves:** data was **not bitwise identical across those
two CI executions**; it is unsafe to assume identical float arrays
merely because the source code, seeds, and reported library versions
are identical.

**What it does not prove:** a matching 0.01 quantized hash is NOT proof
of a maximum sample error under 0.01, a spectral closeness guarantee,
or a meaningful listening-equivalence result. Bucket collisions and
within-bucket differences are possible. These observations do not
isolate the responsible instruction, CPU flag or floating-point
library operation.

## Within-run controls succeeded

The research runner generated the entire corpus in two **independent
Python processes** with the same environment; all 84 source recording
hashes and both dataset hashes matched **exactly**.

The research runner also repeated generation with OMP/BLAS/MKL
thread settings changed from its CI configuration to single-threaded;
the exact per-recording SHA hashes remained identical and no coarse
bucket mismatches were observed.

The separate training runner produced an independent **preflight**
manifest. Its train and held-out exact SHA hashes agreed with the
datasets **actually consumed by the neural training process**.
Training therefore had no preflight-vs-execution dataset drift
within that GitHub job.

These tests protect **within-execution validity** of Sonumio's paired
epsilon-versus-velocity experiments, where the two objectives see
the same in-memory examples and noise. They do not restore byte
identity between machines.

## Hypothesis and next investigation

The two observed GitHub runners used distinct AMD EPYC processor models,
so a CPU-dependent numerical implementation of a DSP function,
NumPy array operation, convolution, or transcendental is a plausible
factor. **That is a hypothesis, not an established cause.**
Compiler flags, library builds, environment variables and other
non-reported system details could also explain the difference.

Before promising bitwise cross-host reproducibility, identify **which
individual source recordings and sample operations** first diverge.
Compare actual sample arrays under controlled environments,
capture SIMD/backend versions, and optionally define a new,
explicitly versioned deterministic data recipe only after
measuring its fidelity trade-offs. Do not retrospectively relabel
an old audio recipe as the same training data.

For all current research experiments:

- Compare model objectives using exactly the same audio tensors
  **within the same run**, as Sonumio's paired trainer does.
- Record per-run exact source fingerprints, CPU/NumPy/PyTorch
  environment and numerical metrics.
- Never pool raw vs latent MSE or claim cross-host byte identity
  without verifying it.
- Do not replace exact leakage checks with approximate bucket matches.
- Keep proprietary/uncleared real recordings, speech and music
  outside this experimental synthetic corpus until licensing, consent
  and privacy review are complete.

**No WAV, neural weight, model generation, tag, release, prerelease
or draft PR merge was produced or authorized.**
