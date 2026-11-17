# Fast float64 DSP pipeline candidate — research-only experiment

**17 November 2026 · Draft PR #1 · no neural training or releases**

The earlier verified [84-recording scalar candidate](COMPLETE_DSP_RESULTS.md)
is bit-for-bit stable across four tested CPU environments, but the
scalar Python convolution is very slow. A [separate measurement](DSP_CANDIDATE_BENCHMARK.md)
observed approximately **679×** slowdown for one 1.5-second rain
sound in a two-repeat CI benchmark. This does not imply a universal
slowdown across processors or sources.

## Goal and implementation

The new `research/fast_dsp_candidate.py` constructs a **separate**
experimental source recipe:

`experimental-vectorized-f64-convolution-dsp-v1-not-for-training`

Like the scalar reference, it reconstructs first-party seeded sounds
at 24 kHz while preserving the original source ordering, source
envelopes and gains, random draws, fades, peak normalization,
original float64 FIR 24-kHz-to-8-kHz resampling and seeded crop.

**Only the original four source convolutions are replaced**, this time
with the separately isolated
[NumPy float64 convolution candidate](VECTORIZED_CONVOLUTION_CANDIDATE.md).
The float32 source input and float32 kernel are promoted to float64,
`numpy.convolve` is used, and the complete result is rounded exactly
once to float32. Forest and ambient layers are unchanged.

The new candidate is **not** added to or imported by any official
Sonumio procedural renderer, model inference service, original
wave-diffusion training dataset, original latent trainer, web API,
or checkpoint builder. There is no monkey patching of global NumPy.

## Corpus and safety gates

The separate `--scope full` research run generates **84 synthetic
source recordings** (42 train and 42 validation) spanning 6
single-source and all 15 two-source combinations at distinct
recording seeds 0, 1, 101 and 102. A 40-recording smoke scope
is available for fast development.

For each independent source recording:

- Compute the entire 1.5-second 36,000-frame float32 fast 24-kHz
  waveform and its 8-kHz original-FIR result.
- Crop exactly 8192 contiguous samples using the historical
  seed-specific offset.
- Independently regenerate the **unchanged** original DSP recording.
  Verify the original crop is bitwise equal to the official
  `render_seeded_waveform` research-data oracle; fail closed if it
  differs.
- Record full 24-kHz, full 8-kHz and final 8-kHz crop SHA256 hashes,
  plus maximum/RMS original-vs-fast sample difference.
- Refuse nonfinite audio or metrics; refuse incorrect sample counts,
  invalid source/split identities, unapproved source names, duplicate
  waveforms in either split and exact overlap between train and
  validation. Report JSON only; **no WAVs, weights or audio arrays**.
- Record candidate source-render timing as a diagnostic, not a
  production-level benchmark: the measured method and environment
  may differ from the original benchmark conditions.

Both the 24-kHz and final 8-kHz split aggregates are SHA256 over
concatenated per-recording digest bytes in canonical source/seed
order; they are **not** hashes of an audio file or a trained model.

The `matches_scalar_reference` check uses the four independently
verified scalar aggregate values from
[the complete-corpus four-host run](COMPLETE_DSP_RESULTS.md) and
reports a Boolean. The fast method may fail exact scalar identity;
this is an open, measured question, so neither the unit tests nor
the first-pass matrix **assume** that the reference matches.

## Tests and multi-host comparison

```bash
python -m pytest -q tests/test_fast_dsp_candidate.py
python -m research.fast_dsp_candidate --scope smoke \
  --output checkpoints/private-research/fast-smoke.json
python -m research.fast_dsp_candidate --scope full \
  --output checkpoints/private-research/fast-full.json
```

The research CI covers all six individual sources, selected pairs,
full-pipeline oracle crops, numerical errors, invalid inputs,
duplicate-waveform and train/validation exact-SHA leakage
rejection. The independent four-host CPU matrix runs **two
separate full-corpus Python processes** and compares their
candidate 24-kHz and 8-kHz crop fingerprints on each runner.
It also prints the entire 42/42 train/validation split SHA
aggregates and Boolean scalar-reference match for cross-host
inspection.

The matrix comparison is **observational** until the experimental
hypothesis is verified. A successful process/shape/same-host check
is not equivalent to success in cross-host bitwise identity.
Do not declare the fast candidate reproducible merely because
the matrix jobs finish successfully.

## Decision gates

1. Confirm all source-convolution fast candidate SHA256s across
   independently provisioned CPU hosts.
2. Confirm **all 84 final fast 8-kHz crop SHA256s** and both
   split aggregates across the same CPU hosts, including EPYC
   hardware where the original NumPy path was host-sensitive.
3. Compare fast full-corpus 24-kHz and 8-kHz hashes with the verified
   scalar reference. If mismatch, report sample-level errors and
   document whether bit identity is impossible without fixed
   rounding order.
4. Benchmark fast-vs-original and fast-vs-scalar throughput with
   independent warmup/repetitions on multiple CPUs, plus separate
   native-memory peak measurements. Do not extrapolate from CI.
5. Require separate listening tests, human scoring, independent
   rights-reviewed audio and bilingual text-adherence evaluation.
6. Only consider any new training recipe after explicit model
   research review and a **new immutable dataset fingerprint**;
   never silently replace historical generated datasets.

**The project remains unreleased; draft PR #1 is unmerged. No
pre-release, stable release, tags, weights or public WAV artifacts.**
