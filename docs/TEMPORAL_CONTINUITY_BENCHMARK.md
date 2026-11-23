# Sonumio source-grounded temporal continuation benchmark

**Numerical audio research only. No model weights, WAV files, external audio, trained model identity, or public audio quality claim.**

## Why an objective reference matters

A model can preserve the preceding audio's final P samples perfectly
without knowing what should come next. It can also produce a small
boundary jump while generating silence, an unrelated sound category,
or a tonally meaningless waveform. Therefore a simple seam-jump
number or crossfade smoothness cannot measure musical rhythm, speaker
consistency or auditory realism.

Sonumio has first-party procedural audio recordings that are generated
as *continuous sources* and then divided into two truly adjacent,
sample-aligned windows. That data permits an **exact ground-truth
future waveform** to serve as a numerical reference.

`research/temporal_continuity_benchmark.py` compares a proposed
future against that exact source. There is no pretrained perceptual
embedding, license inference, audio classifier, ASR or external data.

## Three controlled conditions

For each unique recording group:

**True adjacent future (oracle).** The original future window from the
same source. Its first P samples exactly match the previous window's
final P samples. Its future audio after P is the reference. All
sample-aligned and spectral error metrics must be **zero**. This
control verifies bookkeeping, not the learned model's intelligence.

**Silence after prefix.** The correct known overlap, followed by zero
samples. This can make some seam metrics look deceptively good while
losing all meaningful future sound. It must produce nonzero
sample-aligned prediction error for the source recordings used.

**Unrelated future.** The correct known overlap, followed by the
future suffix of the next distinct acoustic recording group. This
preserves exactly the same prefix and evaluation duration but
changes the true sound source. It must produce nonzero future error.

No example receives a false independent count simply because it
has both Hindi and English captions. All 21 DSP source families per
recording seed are scored once.

## What the eight errors actually measure

`score_continuation` accepts the exact previous clip, real following
clip, and one proposed future (each 8192-frame float32 mono) and
requires bitwise identity in the first P samples.

Only unknown samples after P contribute to:

1. `future_sample_mae`: mean absolute sample-level error.
2. `future_sample_rmse`: root mean-square sample-level error.
3. `future_first_difference_mae`: average mismatch in consecutive
   sample changes, exposing timing/oscillation differences.
4. `future_log_spectral_l1`: log-magnitude STFT mismatch in the
   first 1024 or more future samples at 256/512/1024-point FFT scales.
5. `future_spectral_convergence`: normalized magnitude-spectrum
   residual over the same early future window.
6. `highband_power_fraction_error`: the mismatch in relative
   high-frequency spectral energy above an explicit cutoff.
7. `first_new_sample_jump_absolute_error`: the absolute difference
   between the reference's and generated audio's first *unknown*
   sample jumps.
8. `first_new_sample_slope_change_absolute_error`: the difference
   in local second-order change at the start of unknown future audio.

Every metric is nonnegative, finite, and designed so the exact source
oracle yields 0. They have different scales and must **not be summed
into a single 'audio quality' grade** without validation.

We also report the generated waveform's peak and fraction outside
the standard unit audio interval, hashes for the prior/source/sample,
and exact known-prefix identity. No silent clipping or normalization
is performed, and no waveform bytes leave the Python process.

### Important interpretation limitations

A low sample-aligned MAE is not a psychoacoustic metric: tiny waveform
timing shifts and phase rotations can yield large sample MAE despite
some perceptual similarity. A low spectral magnitude error does not
preserve temporal phase, linguistic correctness or naturalness.
A small boundary-jump error does not establish that a song/voice/sound
scene continues semantically. A silent output is a crucial negative
control for exactly that reason.

The 21 procedural source families and their seed-perturbed versions
also remain highly related. They are not real source-disjoint
independent examples of general music/speech/environmental audio.

## Compare models without inventing quality

`compare_continuations(real, baseline, candidate, previous,
prefix_frames=P)` gives the per-metric difference:

```text
candidate_minus_reference_error
   = error(candidate versus true future)
     - error(baseline versus true future)
```

A negative value means lower **that particular error** under a
specified exact source. It does not imply the candidate sounds better,
contains more content, understands the Hindi prompt, or should be
published.

The full dataset benchmark returns one record per unique acoustic
recording group, then source-group means/minima/maxima, with zero-
and positive-error counts. Group means are descriptive, never p-values
or confidence intervals.

## Reproduce with first-party synthetic audio

```bash
python -m pytest -q tests/test_temporal_continuity_benchmark.py
```

```python
from research.adjacent_audio import adjacent_bilingual_dataset
from research.temporal_continuity_benchmark import (
    SeamEvaluationConfig, benchmark_adjacent_dataset,
)

data = adjacent_bilingual_dataset((101,), prefix_frames=512)
report = benchmark_adjacent_dataset(
    data, config=SeamEvaluationConfig(prefix_frames=512),
)
assert report["ground_truth_oracle_zero_error_verified"]
assert report["source"]["independent_recordings"] == 21
```

One seed gives 21 acoustic recordings and 42 templated bilingual
captions. Standard two-seed evaluation yields 42 distinct source
recordings and 84 captions.

The dedicated GitHub CI tests multiple exact source widths, waveform
data rejection, negative controls, deliberate first-sample corruption,
constant waveform identity, complete source-group counts, fixed
numeric categories, and repeated deterministic results.

## What remains before a trustworthy long-form model

To support Sonumio-1, source-grounded metrics should complement
rather than replace long-form blind listening, speech intelligibility,
melody and rhythm coherence, speaker/scene consistency, semantic
prompt adherence, rights-cleared and truly source-disjoint datasets,
larger model context windows and independent training replicas.

This benchmark creates no original Sonumio-1 weights, public audio,
release tags or new software versions.
