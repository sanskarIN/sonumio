# Original DSP pipeline: recording-stage reproducibility investigation

**Sonumio research note — 16 November 2026**

The first-party Sonumio DSP dataset is reproducible byte-for-byte
across independent Python processes on the same machine, but separate
GitHub runner CPU environments previously produced different exact
float32 fingerprints. **The exact numerical operation responsible
remains unknown.** The goal of this investigation is to narrow the
first divergent stage without changing the historical sound dataset.

## Audit, not a new dataset or model

The [standalone pipeline tracer](../research/dsp_pipeline_trace.py)
reconstructs the original procedural sound generator using the
same first-party functions and NumPy operations. The final in-memory
24-kHz waveform is checked with **exact** array equality against a
second call to the unmodified original DSP generator. Its final
8-kHz cropped training waveform is also checked byte-for-byte
against the original Sonumio research rendering function.

The tracer covers **all 84 distinct first-party synthetic sound
recordings** (42 training, 42 held-out). Every English and Hindi
caption shares the same original recording, so caption duplication
does not cause double-counting here.

The recorded stages include:

| Stage | Intended diagnostic |
| --- | --- |
| \`time_grid\` | Float32 source timebase |
| \`layer_*\` | Individual original rain, thunder, wind, forest, water, ambient DSP components |
| \`mix_after_*\` | Original float32 accumulation in source order |
| \`mix_pre_fade\` | Combined unwindowed source audio |
| \`mix_post_fade\` | After 25-ms edge fade |
| \`normalized_24k\` | After the original peak normalization |
| \`fir_kernel\` | Existing 49-tap sinc/Hamming downsampling kernel |
| \`fir_filtered_24k\` | Float64 filtering result |
| \`downsampled_8k\` | Original float32 decimation |
| \`cropped_8k\` | Exact final 8,192-frame training waveform |

Every stage includes its native float32 or float64 type, frame count,
**exact SHA256 of canonical little-endian sample bytes**, and
separate 0.001/0.01 amplitude quantization-bucket SHA256s.

The stage tracer additionally groups stage digests across all
recordings and per individual sound category to narrow whether
a discrepancy begins before mixing, inside an individual source,
in the FIR kernel, or only in the final conversion/crop.

**No WAV files, raw sample arrays or model weights are serialized.**
The JSON output contains only source identities, SHA256 hashes,
lengths, categories, seeds, a bounded CPU model description and
allowlisted Python/NumPy/PyTorch versions and thread settings.

## Verification protocol

The [CPU research workflow](../.github/workflows/research-ci.yml)
generates **two independent pipeline traces in separate Python
processes** on the same runner and strictly requires all stage arrays
to match exactly. It independently checks every final cropped
stage digest against the previous per-recording dataset manifest.

The [private synthetic training workflow](../.github/workflows/synthetic-training.yml)
generates the same trace separately on its own CPU runner and checks
that all 84 final cropped stage hashes match the recordings in its
independently regenerated pretraining manifest.

Both jobs log aggregate stage hashes and per-single-source layer
and crop hashes. This enables manual comparison of **which stage first
differs across distinct runner models**, without requiring uploads of
any sound recordings or neural network weights.

Do not force cross-CPU equality until the exact cause is understood.
A lower-resolution bucket-hash agreement is not byte equality, not
a maximum sample-error guarantee and not a substitute for exact
train/validation leakage checks.

## Hypothesis testing and next action

A mismatch at \`time_grid\` points to the float32 timebase;
a mismatch at a \`layer_*\` with a matching timebase narrows the
problem to that source's signal generation; matching layers with
mismatched \`mix_after_*\` suggests the float32 layer combination.
Matching normalized audio but diverging \`fir_kernel\` suggests kernel
construction; matching both but diverging \`fir_filtered_24k\`
suggests convolution arithmetic. Different \`cropped_8k\` hashes
with matching earlier stages would implicate downsampling/crop.

These are **diagnostic clues, not automatic causal conclusions**;
host-specific NumPy builds, math libraries or SIMD paths require
independent controlled tests before assigning responsibility.

The current dataset recipe and all prior training evidence stay
unchanged. Any future bitwise-stable replacement must use a distinct
versioned recipe, reproducibility tests across CPUs, and independent
evaluation of audible fidelity before replacing earlier experiments.

## Practical development commands

~~~bash
python -m research.dsp_pipeline_trace \
  --output checkpoints/private-research/dsp-trace.json
python -m research.dsp_pipeline_trace --summary-only
python -m pytest -q tests/test_dsp_pipeline_trace.py
~~~

The full JSON report is intended for private research inspection
only. It contains **no audio or model parameter data** and is not
published as an artifact.

Sonumio-1, realistic audio quality, open-ended prompting, speech,
music, real-world data rights, and creator readiness **remain unproven**.
No draft PR merge, prerelease, stable release, version tag or
model weights publication is authorized.


## Independent hosted-CPU matrix

The [independent DSP CPU pipeline trace workflow](../.github/workflows/dsp-trace-matrix.yml)
also schedules **four separate** Ubuntu 24.04/22.04 hosted jobs on the same
GitHub commit. Each job installs the same Python/CPU research dependencies,
runs the original 84-recording tracer twice in independent Python
processes, strictly validates same-host byte identity, and prints the
allowlisted CPU model, per-stage source hashes, and coarse bucket hashes.

The individual jobs are separate machines; differing exact stage hashes
between their logs are recorded as **data-environment evidence**, not as
a failed neural model test or a reason to rewrite the DSP generator.
A matching stage hash only certifies that stage's bytes; differences
locate a first observed divergence but do not prove the causal
instruction, compiler implementation, or libm/kernel behavior.

No audio WAV, neural model weights, named models or releases are
created. Workflow logs contain hash/CPU metadata only. The
experiment is bounded to four parallel hosted machines and is not
intended as ongoing large-scale CPU benchmarking.


The [first measured four-runner results](DSP_PIPELINE_TRACE_RESULTS.md)
showed **exact stage identity** on one AMD EPYC 9V45 and three
Intel Xeon 6975P-C runners. The formerly divergent AMD EPYC 7763
was not represented; therefore the earlier host-specific numerical
difference remains unexplained, not fixed. A follow-up mixed
Ubuntu 22.04/24.04 matrix broadens the sample without changing
original source waveforms or training.


## Reproduced source-level divergence on EPYC 7763

The [second actual cross-CPU study](DSP_PIPELINE_TRACE_RESULTS.md)
included two Ubuntu 24.04 AMD EPYC 9V74, one Ubuntu 22.04
Intel Xeon 6973P-C and one Ubuntu 22.04 AMD EPYC 7763.
All 84 original audio traces passed within-job validation, but
**rain, thunder and wind layer hashes diverged on EPYC 7763**,
while time grid, FIR kernel, forest, water and ambient matched.
The new [convolution probe](DSP_CONVOLUTION_ISOLATION.md)
tests original 180-/140-/60-tap convolutions against a
12-tap water control, separating input/kernel/output hashes.
