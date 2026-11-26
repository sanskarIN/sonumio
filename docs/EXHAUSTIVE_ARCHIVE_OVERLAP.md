# Exhaustive private archive overlap screening

This optional research diagnostic complements the exact crop hashes in
the archive reader and the aligned/shifted sign-sketch ingestion checks.
It does not change corpus ingestion, eligibility, training, models or releases.

Run locally after installing the research and test dependencies:

```bash
python -m research.archive_overlap_audit private.scorpus \
  --expected-sha256 TRUSTED_ARCHIVE_SHA256 --max-shift 1024 \
  --pair-budget 4096 --require-clear
```

Use the SHA256 retained independently after the approved archive build.
No audio, caption, identifier, model weight or per-clip result is written.
The JSON contains an archive digest and aggregate private-corpus statistics;
keep that report local unless disclosure is separately authorized.

## Numerical method

For each of train/validation, train/test and validation/test, enumerate
every Cartesian-product pair. The pair budget is checked against the
total before any pair computation. There is no sample selection or
sign-sketch prefilter, and a budget overrun never reports partial clearance.

Normalize each finite float32 clip in float64 after global centering.
For every integer lag within the requested range, use a zero-padded linear
FFT correlation and prefix sums to compute local centered Pearson correlation
over only the actual shared interval. Positive lag compares the beginning
of the first clip with the later portion of the second. The absolute
correlation is insensitive to gain, DC offset and polarity where defined.

At most half the crop can be shifted: the default 1024 frames is 128 ms
at the archive's 8 kHz sample rate. Each compared segment has at least
half of the original frames. Overlap energy below 1e-8 relative to the
globally normalized signal is classified as numerically undefined, rather
than falsely reporting zero similarity. The default threshold is 0.9995
with a conservative 1e-7 review margin. This margin is not a formal
floating-point error bound. Constant and low-energy overlaps prevent
screening clearance.

## Costs and interpretation

The default 4096-pair budget can be raised explicitly to 100,000.
For each split pair, processing retains only two prepared shards and
working arrays; a later split's shards are reread for each earlier shard.
Each pair costs O(F log F + L) numerical work for F frames and L lags.
Python and NumPy overhead and repeated verification add real costs.
There is no measured runtime guarantee or claim of large-corpus scalability.
Use a different indexed algorithm for corpora exceeding the pair budget;
do not split into batches and mistakenly omit cross-batch pairs.

A potential overlap is a reason for human review, not proof of copying.
Periodic signals and shared recording characteristics can produce false
positives. The check cannot detect arbitrary time warps, pitch shifts,
resampling, different crop windows outside the tested lag interval,
semantic similarity, legal rights, or voice presence. It examines the
archived model crops only, not full original recordings. It does not
measure generated-audio memorization or establish model quality.

Exit status is 2 for invalid archive/policy/budget; no JSON is emitted.
With --require-clear, status 2 also indicates potential overlaps,
undefined lags or no examined pairs, but a completed aggregate JSON
report is emitted. Without that option, a completed screen returns 0
regardless of its findings. A clear report is limited to this numerical
protocol and does not establish statistical independence.

## Verification

```bash
python -m pytest -q tests/test_archive_overlap_audit.py
```

Tests compare all computed lags to a direct time-domain oracle, exercise
gain/DC/polarity and positive/negative boundary shifts, and verify privacy,
pair budgets, archive checksum rejection, and CLI behavior on synthetic
fixtures. The dedicated CPU workflow runs these tests with the existing
archive round-trip tests. No external recordings or trained weights are used.
