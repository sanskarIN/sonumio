# Private neural waveform replication screen

This optional research tool compares generated audio with **every eligible
training crop** and measures a held-out validation baseline against the
same training gallery. It targets the current 8 kHz, 2048-frame CPU waveform
checkpoint, not the latent codec, original Sonumio-1, music or speech.

## Motivation and literature

Bralios et al., *Generation or Replication: Auscultating Audio Latent
Diffusion Models* (ICASSP 2024), compare retrieval representations and
report advantages of mel spectrograms for identifying replicated audio:
https://arxiv.org/abs/2310.10604.

*Mitigating data replication in text-to-audio generative diffusion models
through anti-memorization guidance* (2025) investigates a distinct
inference-time mitigation problem: https://arxiv.org/abs/2509.14934.
Sonumio implements no such mitigation here. These sources motivate
measurement; this implementation is independently defined and is not a
reproduction of either paper's metrics, models or experimental results.

## Run locally

Use the original private archive and checkpoint with their independently
retained SHA256 receipts, and the exact checkpoint training configuration:

```bash
python -m research.archive_replication_screen private.scorpus private.stcheckpoint \
  --archive-sha256 ARCHIVE_SHA256 --checkpoint-sha256 CHECKPOINT_SHA256 \
  --steps 50 --channels 32 --diffusion-steps 32 --reverse-steps 8
```

Adjust all training arguments to match the checkpoint. The restore
operation verifies numeric checkpoint state, configuration and eligibility
through the existing loader; no optimization occurs. Use only authorized
local data. Never publish the private archive, weights or report by default.

## Experimental protocol

- Use the existing readiness gate and recheck each eligible crop digest.
- Enumerate all eligible training crops as a single fixed gallery.
- Treat each eligible validation crop as a held-out query. The test split
  is inspected by archive/readiness verification but never scored here.
- Generate once per seed with null guidance; for each nonzero guidance
  generate all six categories using the same initial noise. Null outputs
  are not counted six times. Do not clip generated waveforms.
- Compare each query to each gallery crop. Retain independent maximum
  similarities for each metric, not the identity of the matching crop.
- The budget is checked before generation. Exceeding it aborts rather
  than silently using an incomplete gallery or validation panel.

### Representations

Waveform metric: absolute cosine of mean-centered unit-norm aligned
waveforms; invariant to gain, DC offset and polarity where defined.

Spectral metric: 256-frame periodic Hann analysis, 64-frame hop and 24
triangular area-normalized mel bands. The log power transform uses the
whole crop mean power as its reference. The metric compares mean-centered,
unit-normalized time-frequency arrays over each actual overlapping region
at lags -2 through +2 STFT hops, taking the highest cosine. It is not the
same metric as the cited papers. It preserves some temporal structure,
unlike a time-averaged category embedding, and can flag certain short shifts.

Signals with centered RMS below 1e-5 are unscorable. Nonfinite or
out-of-bound signals fail. The waveform threshold 0.9995 and spectral
threshold 0.995 are provisional engineering review triggers, not
empirically validated decision boundaries. Report scorable and unscorable
counts, medians, maxima and candidate counts separately for each metric
and guidance panel. The nearest neighbor need not be the same for the
two representations.

## Scope and resource limits

At most 256 validation queries and 100,000 total gallery/query pairs;
default pair budget 20,000. Retain only query signatures and a single
gallery signature while iterating verified archive shards. The query
count is also bounded by the existing generation config. No WAV,
individual recording identifiers, captions, matching pairs or new weights
are written. Aggregate counts and digests may still be sensitive.

The validation baseline is descriptive, not a statistical membership
inference classifier. Related sounds can have high spectral similarities;
different timing, tempo, pitch, remixing or short excerpts can evade both
metrics. The validation crops are not independently sampled if multiple
crops belong to one recording group. High similarity merits human review,
while low similarity does not establish originality or a safe model.
No legal, semantic, perceptual or privacy conclusion follows.

## Acceptance and next experiment

The synthetic tests include exact-copy, affine-copy, integer-hop-shift,
independent-noise and unscorable controls; a verified private one-update
checkpoint exercises genuine DDIM sampling without parameter updates.
The full-gallery positive control substitutes known copied training audio
for generated samples, explicitly distinct from neural model results.

Next, execute longer multi-seed private training with an independently
reviewed non-voice corpus; calibrate the screen against edited-copy and
nonmember recording groups on disjoint development data, then freeze the
protocol before final testing. Add broader time-shift waveform comparison,
multi-scale features, and independent listening review. None of those
future results are claimed by this change.


## Group-balanced threshold stress test

An additional development-only CLI measures fixed threshold behavior using
known transformed training members and validation nonmembers:

```bash
python -m research.archive_replication_controls private.scorpus \
  --expected-sha256 ARCHIVE_SHA256 --seed 61 --pair-budget 50000
```

There is exactly one anchor (lowest eligible crop index) per recording group
from each split. Every training anchor has eight deliberately related
variants: identity, gain/DC/polarity, zero-padded shifts of 64 and 128
frames, additive centered noise at 30 and 15 dB signal-to-noise ratios,
a three-tap low-pass filter, and PCM16 quantization. Noise uses a
deterministic seed derived from the study seed and crop index.

Both these known-member variants and the validation nonmember anchors use
the same full eligible training gallery. For each member query, report the
known-source similarity separately from the maximum gallery similarity.
For each representation and panel, report scorable counts, minima,
medians, maxima and candidate fractions across the prespecified thresholds
0.9, 0.95, 0.98, 0.99, 0.995, 0.999 and 0.9995. Undefined similarities
remain undefined, with explicit counts. No threshold is selected or fitted.
Validation nonmember hits are screening flags, not proven false positives,
because independent recording rights and semantic independence are not established.

Maximum 64 anchor groups per split and 100,000 complete comparisons;
default budget 50,000. An excessive study fails instead of silently
subsampling groups or gallery rows. Reports contain aggregate numbers and
archive/eligibility digests, not private identifiers or individual scores.
The test split is not scored, although archive readiness verifies it.
No model runs and no output audio or weights are written.

Use these descriptive controls to develop hypotheses. Any chosen threshold
must be frozen and assessed on a separate independent corpus before making
claims of detector sensitivity, specificity, or model memorization safety.
The synthetic fixture CI is not evidence of real-world calibration.
