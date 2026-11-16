# Multi-CPU stage-trace evidence: original Sonumio DSP

**16 November 2026 · numeric research verification only**

The original Sonumio synthetic dataset was traced through 20
intermediate signal stages on separately allocated GitHub CPU runners.
The tracer did **not** modify, quantize, store or publish any WAV sound
or training model weights.

## First four-runner study: confirmed exact identity

Workflow: [Independent DSP CPU pipeline trace #43880973720](https://github.com/sanskarIN/sonumio/actions/runs/43880973720)
at code commit [ae3dba1c](https://github.com/sanskarIN/sonumio/commit/ae3dba1c32f1fab3befd116f6ea24446e90fe6de).

| Job | Hosted CPU model | Exact 84-recording pipeline result |
| --- | --- | --- |
| CPU stage trace 1 | AMD EPYC 9V45 96-Core Processor | All 20 aggregate stage hashes match |
| CPU stage trace 2 | Intel Xeon 6975P-C | All 20 aggregate stage hashes match |
| CPU stage trace 3 | Intel Xeon 6975P-C | All 20 aggregate stage hashes match |
| CPU stage trace 4 | Intel Xeon 6975P-C | All 20 aggregate stage hashes match |

All four GitHub jobs finished **successfully**. The runner labels
are the models as reported by Linux \`/proc/cpuinfo\` in each job.
Each individual job generated all 84 original first-party recordings
in **two independent Python processes**, checked strict stage-hash
identity within that host, and asserted the reconstructed waveform
was byte-identical to the unmodified DSP renderer and original
research training crop.

Across the four jobs, exact hashes agreed for:

- The 24-kHz float32 source time grid.
- All six original DSP source-family layers: rain, thunder, wind,
  forest, ocean/water and ambient/drone.
- Each incremental layer mix, edge fade and peak normalization.
- The 49-tap float64 FIR kernel and filtered 24-kHz audio.
- The float32 8-kHz downsample and final 8192-frame crop.
- All 12 recorded source-specific aggregate digests
  (six single-source layer digests and six single-source crop digests).

Example complete aggregate SHA256 results shared by all four CPUs:

| Original processing stage | Exact aggregate SHA256 |
| --- | --- |
| \`time_grid\` | \`e63757ce0becd2a6c923547e38aba59d8eab98bb90e9a3a7a281359e863cd4f1\` |
| \`layer_rain\` | \`493b260f5fe1b73705f6a93a472077786775d6d31b44cd319bf41fe674f761a4\` |
| \`layer_forest\` | \`f5b7d9c54a2fad9f4c28d5b14d061acdb87b423f295f6532d24499d2cb93922f\` |
| \`normalized_24k\` | \`2631d25b82e770996ce87b8bf970a0b0027942a51e4bb2e0a48e07200c8f9116\` |
| \`fir_kernel\` | \`f7b5bbe93362671a8b84234c94ce312b2f456a18455a2d3fb2bb92449d6d729f\` |
| \`fir_filtered_24k\` | \`fc618cef612df47639ba34b8c64a9763de7ea54c5901860f30299c1a0c8138bd\` |
| \`downsampled_8k\` | \`9896feff280360c9e8c967a4d9e3cb0ae045f243c465bd3f6539702e423f6e86\` |
| \`cropped_8k\` | \`11822dfb6181eb8772af4c7dffd4963189bcb2ef2a91bf6e1dd5a046b9c6e5f9\` |

Research CI and private training CI passed on the implementation
commit \`15f2b514\`; the CPU research suite reported **192 passing tests**.
Both jobs happened to use AMD EPYC 9V74 and matched all trace hashes,
so that run alone did not offer heterogeneous CPU evidence.

## Why the prior fingerprint discrepancy is still open

An earlier, separately hosted AMD EPYC 7763 produced a
*different* exact train/validation recording fingerprint than
the AMD EPYC 9V74, 9V45 and Intel Xeon hosts observed so far.
The new four-host stage trace **did not include an EPYC 7763**,
so the earlier discrepant run has not been reproduced at the
intermediate-stage level.

**It would be misleading to declare the root cause fixed.**
The new results demonstrate reproducibility across several actual CPU
models and source stages, not universal DSP bitwise stability.

An extended GitHub CI matrix now also tests a legacy
Ubuntu 22.04 hosted runner pool to broaden the environments.
Runner-image version is an extra variable, so if it differs
we must avoid attributing the cause exclusively to CPU hardware
without independently controlled binary/library builds.

The next valid step is to compare step-level hashes when an
environment that reproduces the old exact-hash divergence
becomes available. Do not substitute coarse quantized hash matches
for exact waveform identity.

## Safety and scientific limitations

The sounds here are simple first-party synthetic DSP reference data;
these checks do not demonstrate that a neural Sonumio model can
produce realistic rain/wind, music, speech or accurate multilingual
audio prompts. High-fidelity codecs, longer/high-rate recordings,
independent listening tests and rights-cleared real-world source
data remain necessary.

These tests use only diagnostic SHA256s, CPU strings and source labels.
**No release, prerelease, tag, PR merge or pretrained model weights
were published.**


## Second four-runner study reproduced EPYC 7763 differences

Workflow: [Independent DSP CPU pipeline trace #43881702284](https://github.com/sanskarIN/sonumio/actions/runs/43881702284).
**All four hosted jobs passed**, with exactly matched stage hashes
between the two processes **within each job**.

| Job | Ubuntu image | Processor | Match to baseline 20-stage digest list |
| --- | --- | --- | --- |
| 1 | 24.04 | AMD EPYC 9V74 | All 20 stages equal |
| 2 | 24.04 | AMD EPYC 9V74 | All 20 stages equal |
| 3 | 22.04 | **AMD EPYC 7763** | **15 stages differ** |
| 4 | 22.04 | Intel Xeon 6973P-C | All 20 stages equal |

The previous EPYC 7763 result is **now reproducible as a source-layer
difference**. Importantly, the other Ubuntu 22.04 Intel runner matches
the baseline exactly, so the older Ubuntu image by itself is
insufficient to explain the difference.

On EPYC 7763, the following first-party source layer hashes were
different: **rain, thunder, wind**. The other three source layers
**forest, water/ocean and ambient/drone** matched exactly.
The original float32 time grid, FIR downsampling kernel, and
the per-source component hashes of forest, water and drone
also matched. After rain/thunder/wind layer differences were
introduced, mixed audio, fades, normalization, filtered and
cropped arrays naturally inherited the divergence.

The three differing source functions all use **long float32
NumPy convolutions (kernels of 180, 140 and 60 taps)**.
The water control uses a **12-tap** smoothing convolution and
matched. This provides a precise operation-level hypothesis,
**not proof** that NumPy convolution or any particular CPU
instruction is responsible.

A [targeted convolution probe](DSP_CONVOLUTION_ISOLATION.md)
now isolates original seeded input arrays, kernels, individual
convolution outputs and scalar reference calculations without
changing source DSP or writing WAV files. Its cross-host
results must be inspected separately before causal conclusions.


## Follow-up probe measurement

The [successful 4-host convolution experiment](DSP_CONVOLUTION_RESULTS.md)
confirmed this hypothesis at the operation boundary: original
long-kernel seeded float32 inputs and kernels match, but the
NumPy convolution output hashes differ on AMD EPYC 7763.
The water 12-tap control and independent selected scalar
reference hashes match. The cause is now localized to
host-specific convolution arithmetic or its implementation,
**not** to different random seeds or source-layer kernels.
Do not claim a specific machine instruction or audible-quality
effect from SHA differences alone.
