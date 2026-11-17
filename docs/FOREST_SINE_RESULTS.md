# Forest-aware portable DSP: complete CPU/NumPy reproducibility results

**Measured 17 November 2026 · Synthetic first-party waveform research only**

## Result and evidence

The initial five-job
[forest-aware portable CPU workflow #44095705429](https://github.com/sanskarIN/sonumio/actions/runs/44095705429)
completed **5/5 successful jobs**.

- **4 independent CPU runner jobs**, each with NumPy 2.4.6,
  produced the **same exact 24-kHz float32 SHA256s and final
  8-kHz 8192-frame crop SHA256s** for all
  **189 source/seed combinations per method**.
- **1 controlled same-CPU version-swapping job** executed
  all **378 candidate recordings** first under NumPy
  1.26.4, then 2.4.6 in a separate Python process
  on the same Intel Xeon 6975P-C host. It found
  **zero changed 24-kHz waveform SHA256s and zero
  changed final 8-kHz crop SHA256s** for either
  oscillator method across the complete **189-case
  per-method** grid.
- Both `numpy_float64` and `python_math_sin`
  methods produced identical SHA256 fingerprints
  in this experiment, despite different
  arithmetic implementations and runtimes.

This resolves the previously observed NumPy
1.26.4→2.4.6 reproducibility limitation
for the **tested source family, sample duration,
seeds, CPU hosts and Python/NumPy versions**.
It does **not** establish universal
bitwise equality on arbitrary inputs,
other NumPy versions, operating systems,
GPU libraries or compiler runtimes.

### Measured CPUs under NumPy 2.4.6

The four independent jobs reported processors
including AMD EPYC 9V45, AMD EPYC 9V74,
AMD EPYC 7763 and another AMD EPYC 9V74.
GitHub-hosted CPU assignments are not
under the project's control, so this is a
four-runner verification, not a guarantee
about four distinct CPU microarchitectures.

### Frozen candidate SHA256 reference

Each SHA256 below aggregates the
**concatenated SHA256 digest bytes of
individual float32 waveforms** in canonical
source-combination/seed order. They are
not hashes of downloadable WAVs, music files,
neural model weights or training datasets.

| Source grid | Cases per method | 24-kHz full raw reference | Final 8-kHz crop reference |
| --- | ---: | --- | --- |
| Original train seeds 0, 1 | 42 | `05b1891546041d895c15be6125c511f0563fb29c331fe70bdabcd395ebe0b18d` | `5e8b09ca8de0918728d7c8bd4b00d21afd97f13689697e0a9651146567ae9fe8` |
| Original validation seeds 101, 102 | 42 | `e487d27a49294f50537e19f9d5b78d969607581d88dfad566f1ac58c577d42c5` | `67f7e5dadd51a429f770b8c41638667cde4f688fc51a9094a66fb8df0d3b22fb` |
| Extra numerical seeds 3, 7, 17, 29, 41 | 105 | `47be2e54b572f55b9970364e3167338aa1969e55f147f3d3373420337408d9cb` | `cc27a25309724f54582ceddc0170b73423ecd19da629fb5f1961e96506502b97` |

All three split SHA rows matched for **both**
alternative sine methods across NumPy 1.26.4
and 2.4.6 on the same CPU, as well
as across all four NumPy-2.4.6 CPU jobs.

## Forensic cause of the last two mismatches

The earlier first-generation portable DSP
candidate fixed the original ambient sine
version differences but left
`dsp._forest` unchanged. It therefore
reproduced **187 of 189** numerical
source/seed cases per method across
NumPy 1.26.4→2.4.6.

The remaining two affected seed-41 cases:
`thunder+forest` and `wind+forest`,
were traced by the 210-case stage
report to the first forest sound layer,
without preceding RNG state drift.

`research/forest_sine_probe.py` reconstructed
forest chirps independently for 36
source/seed contexts and verified that
the reconstruction reproduces **the
unchanged original source bytes and final
RNG state exactly on each host**.

On the same CPU, the two affected
contexts had **identical original
frequency/phase inputs**, but the
original float32 `np.sin` output
was different across NumPy versions
at the **first chirp**. The differences
then propagated through the chirp
contribution, forest mix, 24-kHz raw
waveform, resampling and crop.

In contrast, both alternate methods
(`np.sin` on the float64-promoted
original float32 phase, then one
float32 round; or Python scalar
`math.sin` with the same promotion)
produced identical chirp-layer hashes
for the affected contexts under both
NumPy versions. The forest-aware
complete DSP candidate also passed
the original independent waveform
oracle with bounded numerical error.

## Exact future regression gates

`research/portable_forest_reference.py`
contains these independently **measured**
split SHA256 values and a fail-closed
`verify_reference(report)` API. It
requires both source methods, all
378 cases, original numerical data
provenance, and NumPy version
**1.26.4 or 2.4.6**. It intentionally
does **not** pin the original
host-sensitive float32 generator's
SHA256 fingerprints.

The four-host portable matrix has
been updated to **assert** exact golden
hashes per 378-case report.
The same-runner cross-NumPy job
has been updated to **assert**
all per-recording hashes match,
and verify both versioned reports
against the same frozen values.
A separate test file rejects
fabricated NumPy compatibility
labels, recomputed altered waveform
hashes, missing source identity
and false neural-training flags.

**These new assertions were committed
after the original successful five-job
measurement. Their own post-change
CI results must be verified separately
before claiming that the new automated
gates themselves have passed.**

## Scope and remaining research

Sonumio's first-party DSP sources are
short synthetic sound effects, not
rights-reviewed natural speech,
music, long-duration soundtrack
generation or a general natural
text-to-audio training corpus.
A matching SHA256 establishes
deterministic numerical output for
the tested input grid; it does
not establish perceptual quality,
semantic text following, source
rights, adversarial robustness,
or model generalization.

The extra seed 3 also appears
in the separate experimental
text-conditioned neural training
recipe, so the additional 105
cases are **numerical stress cases,
not an independent global
neural-model holdout**.

Other next gates include sample
durations outside 1.5 seconds,
additional seeds and multimodal
mix combinations, cross-OS and
newer NumPy/Python testing,
performance under sustained load,
native process memory measurements,
proper rights-verified corpora,
unseen acoustic evaluation,
fair neural architecture
comparisons and independent human
listening tests.

**None of the original DSP generator,
original neural dataset, model weights
or deployed Sonumio API has been
modified to use this alternative
research source. Draft PR #1
remains UNMERGED and no release,
prerelease, tag, WAV, checkpoint
or model weight has been published.**
