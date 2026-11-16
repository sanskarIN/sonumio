# Exact synthetic-audio reproducibility and dataset provenance

**Experimental, synthetic-only research. No weights, voice recordings,
WAV assets, pretrained models or releases are saved or published.**

## Why Sonumio needs this gate

An earlier matched velocity-vs-epsilon DDPM experiment had stable
numerical saturation results but **different exact training/validation
audio fingerprints** in separate GitHub Actions executions despite
the same declared recording seed IDs and reported NumPy version.
Two models trained within the same execution saw exactly the same
recordings, which preserves the intended paired comparison; however,
that does **not** establish bitwise identity across machines or jobs.

The actual cause of the cross-job drift **has not yet been proven**.
Possible influences include differences in host CPU/FFT/convolution
numerics or dependency builds; do not assert any as the root cause
without an isolated reproduction. Even sub-micro differences can change
a full SHA256, so a changed hash does not automatically imply material
acoustic differences. It does mean byte-identical datasets must not be
claimed.

## Implementation

The private [dataset provenance audit](../research/data_reproducibility.py)
generates Sonumio's existing original 8-kHz 1.024-second DSP recordings
**without changing the input generation algorithm**:

| Field | Description |
| --- | --- |
| Train seeds | 0, 1 |
| Evaluation seeds | 101, 102 |
| Source combinations | 6 individual categories + 15 two-source mixtures |
| Unique WAV recordings per split | 42 |
| Language captions per split | 84 (two captions per identical audio recording) |
| Signal layout | float32, 8,192 mono samples at 8 kHz |
| Primary identity | SHA256 of canonical little-endian float32 sample bytes |
| Coarse diagnostics | SHA256 of quantized 0.001 and 0.01 amplitude buckets |
| Context | Exact dataset fingerprint, source membership, seed, peak/RMS, Python/NumPy/PyTorch/CPU/thread settings |

For every audio row, the code enforces finite float32 samples,
expected length, and an allowed waveform range. It explicitly verifies
that the English and Hindi captions of each recording share *exactly*
the same sound data, counts each audio source once, and rejects
exact duplicate waveforms inside or across train/validation splits.
This is a **research leakage guard**, not evidence that a narrow DSP
sound family generalizes to independent real-world acoustic scenes.

### Important distinction: exact identity versus coarse numerical similarity

- **Exact SHA256 equality** is the test for recording-byte identity.
- The bucketed hashes can indicate how coarse numerical rounding
  changes a recording's fingerprint, **without** quantizing or
  altering the recordings supplied to model training.
- Matching bucketed hashes are **not** proof that source arrays
  are identical, nor do they establish any bound on maximum absolute
  waveform error. For example, two different values in the same bucket
  yield the same quantized hash. Never use coarse hashes to clear
  training/evaluation overlap or to claim cross-run identity.
- A SHA mismatch alone cannot quantify a numerical error; actual
  original float sample arrays would be needed for that.

The manifest contains hashes, IDs, source labels, peak/RMS and
allowlisted software/thread information only. **No waveform samples,
audio files, embeddings or model parameters** are serialized.
It also enforces a bounded JSON manifest size for comparison.

## CI validation

The [CPU research workflow](../.github/workflows/research-ci.yml):

1. Produces two independent Python-process manifests with the same
   environment and checks **exact** per-recording and dataset SHA
   identity across both invocations. If they differ, CI fails.
2. Generates a third manifest using single-threaded OMP, BLAS and MKL
   settings and reports exact/hash-bucket differences **diagnostically**.
   Thread-profile comparison is not incorrectly required to be exact
   across heterogeneous compute environments.
3. Logs train and held-out source fingerprints and a limited number
   of differing record IDs, not raw waveforms.

The [private synthetic training workflow](../.github/workflows/synthetic-training.yml)
generates an independent preflight manifest in a separate process.
After actual paired epsilon/velocity model training, its CI validation
compares the **exact** train/validation dataset fingerprints recorded
by the model training runner with those of this independent manifest.
This prevents training from silently proceeding on different
procedural audio while reporting a supposedly identical experiment.

CI passing does not prove bitwise reproducibility across all hosts or
future dependency versions; two independent jobs are not necessarily
the same CPU or underlying BLAS/SIMD environment.

## Run locally

From the project root, after installing the documented CPU research
dependencies:

~~~bash
python -m research.data_reproducibility \
  --output checkpoints/private-research/manifest-1.json
python -m research.data_reproducibility \
  --output checkpoints/private-research/manifest-2.json
python -m research.data_reproducibility \
  --compare checkpoints/private-research/manifest-1.json \
            checkpoints/private-research/manifest-2.json \
  --strict-exact
python -m pytest -q tests/test_data_reproducibility.py
~~~

PowerShell: place each command on one line or use PowerShell backtick
continuation rather than Bash backslashes.

Note: these are private local JSON reports and deliberately not
release artifacts. You can delete them after inspection. The
research workflow writes its manifests to the runner's \`/tmp\` and
does not upload them.

## Remaining science and engineering

- Investigate **which** individual sounds and CPU/dependency
  configurations cause any future cross-run waveform hash drift.
- Isolate numerical operations that change floats across machines;
  pin and record the actual library versions and CPU environment.
- Repeat the experiment on separate hosts and architectures.
- Before declaring Sonumio-1 readiness, add independent
  rights-reviewed real-world sound data, reliable session splits,
  longer high-rate clips, independent blind listening and sound-class
  prompt-relevance evaluations.
- **No draft PR merge, public model weights, tags, pre-releases or
  stable releases** without explicit later project-owner authorization.
