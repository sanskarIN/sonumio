# Sonumio Audio Core and Scientific Evaluation — Architecture Record

**Development branch:** `feat/initial-audio-platform`  
**Status:** standalone experimental audio library, NOT the original Sonumio-1 model.  
**Source license:** Apache-2.0.  
**Publication:** no release, pre-release, tag, public generated WAV, or model weight
publication is authorized by this document. Draft PR #1 remains unmerged.

## Scope and design decision

Sonumio needs real signal-processing infrastructure before its eventual
original generative-audio models can be trained or assessed credibly.
The existing procedural SFX renderer is already running and has a
highly constrained numerical reproducibility protocol, including
recordings whose exact bytes have been measured under two NumPy versions.

This development batch adds an *independent* `sonumio.audio_core` package
and *independent* `research.audio_eval` evaluation subsystem.
Neither is wired automatically into the original procedural renderer,
the REST API, published frontend, historical training data, or checkpoint
loader. This avoids accidentally invalidating existing SHA256 goldens.

The new work is local-first: no model hub, network adapter, microphone
capture, external asset fetch, voice cloning or automatic soundtrack
publication is performed. The processing graph evaluates a fixed
allowlist of transformations, **not** arbitrary code supplied through
JSON.

## Immutable audio geometry

`AudioBuffer` owns an independent **float32**, channel-first
`(channels, frames)` array, with an explicit `AudioSpec`.
Its default NumPy view is read-only. A mutable processing allocation is
available only through `mutable_copy()`; combining buffers requires
matching sample rates and channel layouts.

Sample rates: 8,000–192,000 Hz.  
Channels: 1–8.  
Audio duration: at most 120 seconds per bounded in-memory buffer.  
Allocation: no more than 128 MiB per declared canonical allocation.
Many operations have stricter local scratch or staging limits.

These are **development safety ceilings**, not promises of resource
isolation for a public upload service. The existing experimental API
remains local-only; authentication, per-user quotas and public-service
hardening are separate tasks.

## Time, waveform and numerical semantics

The `timebase` module represents timestamps as integer sample
frames and exact `Fraction` durations whenever possible.
Converting external decimal seconds to frames uses round-half-even.
A clip interval is half-open: beginning frame included, end frame excluded.
Absolute stream packet offsets must be contiguous and clock-matched.

The numerical primitives include independent dB-to-linear conversion,
explicit gain, peak/RMS inspection, float64 accumulation,
equal-power crossfade, saturation, clipping and mid/side processing.

Important distinctions:

- **RMS dBFS is not LUFS.** No BS.1770/EBU-R128
  integrated loudness compliance is claimed.
- A **sample-peak limiter is not a true-peak limiter**;
  the latter normally requires oversampling.
- Double-precision accumulation may be numerically reproducible
  for tested platforms but cannot establish universal bitwise portability.
- Stereo balancing and mono-to-stereo routing are explicit;
  a clip is never assumed to be a particular loudspeaker layout.

## PCM WAV and deterministic dither

The new `waveio` reader/writer supports bounded, uncompressed
RIFF/WAVE integer PCM with 8/16/24/32-bit samples, at sample rates
8k–192k Hz and up to eight channels. It checks exact RIFF/chunk
boundaries, byte-rate and alignment consistency, duplicate fmt/data,
finite decoded samples and ordinary, nonsymlink local file requirements.

It **does not** silently fall back for float WAV, compressed audio,
WAVE64/RF64/RIFX or arbitrary metadata encoding. Unsupported formats
fail clearly instead of producing incorrect audio.

Deterministic TPDF dither requires an explicit seed. A 24-bit PCM
encoder writes the exact signed little-endian three-byte
two's-complement payload, and the decoder checks frame alignment.

Writing a report or WAV to disk is an **explicit local opt-in**, and
existing filenames require an explicit overwrite option.

## Resampling and spectral analysis

`resample` exposes:

1. Linear interpolation, allowed for upsampling;
   downsampling requires an explicit acknowledgement of aliasing.
2. A bounded windowed-sinc reference with an anti-alias cutoff
   based on source and target Nyquist rates, and per-output
   kernel normalization at clip boundaries.

The intended behavior is independently checked using constant input,
an above-target-Nyquist tone, and original PCM preservation.
The source-to-output frame conversion uses exact rational arithmetic.

`stft` exposes centered, windowed FFT analysis, bounded complex spectra,
frequency/time axes, masked transforms and inverse overlap-add.
Centered Hann transformations are tested for accurate float32
reconstruction including edge samples.

Acoustic features include spectral centroid, bandwidth, roll-off,
flatness, flux, energy, mel-frequency mapping and mel-band power.
These are physical/numerical descriptions. They are not naturalness
scores, semantic embeddings, content filters or MOS estimates.

The separate `spectral_effects` research processor offers
noise-profile spectral gating, EQ, tilt and median-filter
harmonic/percussive soft masks. An operator must **explicitly**
select which recording interval is assumed to contain background noise.
HPSS heuristic stems do not semantically separate singers,
instruments or copyrighted performances.

## Filtering, dynamics and effects

The new IIR/FIR filter module contains stable bounded
second-order biquad designs (RBJ family), stateful causal
processing, windowed-sinc FIR, convolution, DC blocking,
low-pass, high-pass, band-pass, notch, shelving,
peak and all-pass variants.

Dynamics include an RMS/peak envelope follower,
linked-channel soft-knee compressor, hysteretic gate,
explicit peak/RMS normalization, and a conservative
offline sample peak ceiling.

Time effects include bounded delay, ping-pong,
tremolo, ring modulation, chorus, flanger and
finite impulse-response convolution reverb.
The reader should expect audible processing artifacts
at extreme settings; there is **no claim** of transparent
music mastering or pretrained AI audio enhancement.

## Multitrack and procedural test synthesis

The new `timeline` subsystem has immutable source clips,
integer frame placements, gain/pan/fades, track
mute/solo controls, explicit channel routing, and a
float64 accumulation buffer. Supported operations
include clip slicing, reverse, concatenate,
crossfade, padded extension, looping, chunking
and coarse silence detection.

`automation` contains keyframe interpolation,
explicit boundary behavior, per-sample envelopes,
gain automation and a rational BPM beat clock.

`synthesis` is an ORIGINAL, fully procedural
oscillator/noise reference generator for creating
small, easily audited test signals. It supports
bounded pitches and polyphony, ADSR envelopes,
deterministic noise, and optional harmonic
bandlimiting. **It is not a learned Sonumio model**
and should not be advertised as trained
text-to-music or high-quality natural audio.

## Safe declarative processing graphs

`SignalGraph` only accepts declared nodes
with allowlisted operators and typed parameters.
A graph must already be topologically ordered;
forward references, cycles, missing inputs,
duplicate names, nonfinite values and
unknown operators fail before execution.

This blocks arbitrary Python evaluation, shell
execution, plugin loading, filesystem paths,
network calls and implicit neural model loading.
It **does not** provide a sandbox for an
untrusted, publicly exposed service.
The total materialized frame count is capped
so a graph cannot casually build unbounded
many-generation audio tensors.

Per-node reports include absolute geometry
and a SHA256 fingerprint of PCM. Hashes
support repeatability and error detection
but are not signatures or proof of
creator rights.

## Continuous PCM stream framing

`streaming` allows fixed-clock,
contiguous PCM frame packets and a
bounded window framer. Window extraction
is invariant to the placement of
input chunk boundaries: a 128-frame
chunk and a 137-frame chunk produce
the same analytical windows if
they contain the same PCM bytes.
The framer can either discard a
short trailing frame or explicitly
pad it with zeros.

This API is **not** network streaming,
a real-time operating-system scheduler,
or a production live-audio stack.

## Human rights and evaluation provenance

`research.audio_eval` uses a
`RightsDeclaration` specifying
claimed source kind, license,
human review flag, permission
status and voice presence.
The declaration's consistency is
validated, but Sonumio software
does **not** authenticate that the
declaration is truthful.

An `EvaluationSet` rejects
reused asset IDs, duplicate PCM
checksums and group IDs across
train/validation/test/stress splits.
An explicit source group is the
independent unit for the numerical
bootstrap: several clips from
the same original recording
are NOT independent observations.

**Important:** extra first-party
numerical seed 3 appeared in a
different text-conditioned neural
training experiment. Consequently,
the old 105-case extra-seed corpus
is only a numerical stress test,
not globally unseen model evaluation.

The evaluation subsystem calculates
sample error, SI-SDR, normalized
correlation and multi-resolution
STFT discrepancies. No one metric
establishes audio realism,
prompt adherence or perceptual quality.

## Prespecified and grouped analysis

`EvaluationPlan` requires a
declared test/validation/stress
split, metric direction, candidate
engine ID, exact FFT sizes,
number of independent groups,
confidence level, fixed RNG
seed and a human listening/
rights-review requirement.

The runner reports per-case
fingerprints, per-case scores,
group-equal-weighted aggregate,
group-bootstrap interval and
threshold outcomes. A/B comparisons
require identical case IDs,
original PCM source hashes,
group identities, split and
metric before making any
numerical contrast.

A positive reported numerical
difference is **not** a
certified model improvement.

## Numeric-only, checksum-addressed archives

`research.audio_eval.report_io`
serializes only bounded plain
JSON metrics. Embedded raw PCM,
WAV bytes, known secret fields,
checkpoints, arbitrary Python
objects and NumPy arrays are
rejected. Optional local output
uses an explicit atomic write
and opt-in overwrite semantics.

Canonical SHA256 records bind
payload, declared source identity
and metadata. They are integrity
checks only. A malicious person
with authority to recompute
hashes could forge a different
record, so sensitive or
legally consequential provenance
requires authenticated storage
and human review.

## Running the tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_audio_core_*.py
python -m pytest -q tests/test_audio_eval_*.py
```

The standalone GitHub workflow
`.github/workflows/audio-core-ci.yml`
tests Python 3.11/3.12 and NumPy
1.26.4/2.4.6 in fresh runners.
It is separate from the legacy
DSP-source numerical reproducibility
matrix, neural training smoke tests
and local Web API tests.

## Engineering gates before use in production

The next generation of Sonumio
must establish the following
before any original model,
public API or release:

1. Rights-verified, auditable
   first-party and licensed
   datasets with defensible
   train/validation/test splits.
2. Neural speech/music/SFX
   architecture, objective
   ablations and measured
   generalization to previously
   unseen source groups.
3. Reproducible checkpoints
   with strict optimizer/model
   provenance, numerical
   repeatability and realistic
   compute/memory budgeting.
4. Human listening, accessibility
   review and evaluation across
   Hindi/English and future
   supported languages.
5. API authentication, quotas,
   request cancellation,
   concurrency isolation and
   a documented retention policy.
6. CPU and GPU memory profiling
   without confounding
   research-only imports.
7. Portable install/build
   testing and security review.

**No release, prerelease, tag, audio
publication, source dataset
replacement or model-weight
publication is approved by
the present work.**


## Rational frame clock and materialized partitions (18 November 2026)

An audit identified a numerical edge case in `frames_from_seconds`: converting
a high-precision `Fraction` through the default 28-digit `Decimal` context
could silently collapse positions just *above* a half-frame boundary onto the
tie, changing the selected sample frame. Rational timestamps now use exact
integer quotient/remainder arithmetic and **round-half-even**; Decimal,
float and integer input retain their existing public behavior. A 4096-bit
per-term rational precision ceiling fails closed for pathological timestamps.

The other discovered allocation hazard was `TimeGrid.partition(1)` at a valid
120-second/192-kHz clock: its former tuple comprehension could materialize
23,040,000 `FrameInterval` Python objects. `partition()` now preflights
the count and caps eager materialization at 50,000 intervals. Callers that
need finer divisions should consume the existing lazy `split_intervals()`
iterator in bounded batches rather than accumulating its full output.

This does not alter DSP samples, existing procedural source computations,
training recipes or diffusion checkpoint formats. Both changes are tested
by `tests/test_audio_core_timebase_precision.py` and included in the
independent Python/NumPy audio-core matrix.


### Decimal half-frame precision follow-up

The exact time quantizer now also converts finite `Decimal` and
decimal-form float boundary inputs into exact integer ratios before applying
round-half-even. This closes the same 28-digit precision bug for a
high-precision `Decimal` located only `1e-40` seconds above a half-frame.
Decimal textual precision and exponent are bounded before constructing any
large denominator; pathological numeric encodings fail closed. Float input
retains its established text-decimal interpretation (`Decimal(str(value))`),
not a claim that binary floating-point carries exact human-intended times.


## Optional two-pass per-channel DC removal (18 November 2026)

The local `sonumio wav-process INPUT.wav --output OUTPUT.wav --remove-dc`
switch explicitly enables full-file arithmetic mean subtraction **per channel**.
It defaults to off. The first hash-verified pass calculates channel means
using exact, bounded int64 fixed-point accumulation across the decoded
8/16/24/32-bit PCM (including 32-bit float32-decoding roundoff) and extrema
with bounded memory; the second separately verified pass subtracts
each mean in float64 before any chosen gain, ceiling, hard clipping, or
seeded PCM TPDF quantization. A centered waveform may have a larger sample
peak than its original; gain/ceiling decisions therefore use the corrected
extrema, not the uncorrected file peak. Aborted operations leave no
published partial WAV.

Reports retain uncorrected `source_peak` / `source_rms`, and contain
post-processing `output_*_before_quantization`. New fields
`dc_removed` (boolean) and `removed_dc_offsets` (ordered channel means in
normalized PCM full-scale units) make the transformation auditable. Without
the option the offset array is empty and historical PCM results are unchanged.

This is **not** a time-varying baseline remover, high-pass filter, loudness
normalizer, true-peak limiter, speech enhancement, learned audio generator,
copyright permission validator or perceptual quality claim. Quantization
can leave a small residual mean. It is not applied to the legacy procedural
generator, research data/training, API or frontend. The draft remains
unmerged and no release/weights/audio are authorized.


## Prepublication output integrity gate (18 November 2026)

For independent offline PCM file processing, optionally pass
`--expected-output-sha256 <64 lowercase hex characters>` to
`sonumio wav-process`. This requires the **entire serialized
RIFF/WAVE output** (including 44-byte header, audio bytes, and odd-sized
payload padding) to match the supplied digest. The destination is not
published if it differs. This is distinct from
`--expected-sha256`, which authenticates the **input** WAV read
in both passes. Both options are disabled unless explicitly supplied.

The gate runs within `WavPacketWriter.finish(expected_sha256=...)`
after full scratch-file hashing and fsync, but **before** atomic
link-or-replace publication. Wrong hashes leave no newly published WAV;
an existing destination remains byte-for-byte unchanged even when
`--overwrite` was requested. Scratch files are cleaned up.
Invalid hash syntax is rejected before processing. The expected hash
is a reproducibility checksum, **not** a digital signature, a rights
declaration, a perceptual quality certificate, or protection against
a malicious actor who can replace both expected digest and data.

Changes are opt-in and do not modify legacy DSP WAV outputs, neural
training sources, API, web client, model weights or release state.


## Verified, frame-exact streaming quiet-edge trim (18 November 2026)

The independent local `wav-process` command offers **opt-in**
`--trim-silence-dbfs -60` (supported threshold range: -120 to -1 dBFS).
This retains the inclusive first through exclusive last sample frame for
which **any** channel contains a decoded PCM sample whose absolute amplitude
is **strictly greater** than the selected linear sample threshold. It removes
only leading and trailing quiet frames, **never** internal quiet regions.
All channels use the same shared frame interval; no resampling or timeline
offset guessing is performed.

The first pass hashes **the entire original WAV** while locating the
retained interval. If nothing passes the threshold, processing fails
closed before creating any output parent/destination. The second pass
still reads and verifies **every input frame**, including trimmed edges,
but streams only the retained slices into an atomically published WAV.
Optional gain, per-channel full-file DC correction, sample-peak ceiling,
clipping and seeded TPDF run **after** the original-signal trim selection.
An optional expected output SHA256 gate still checks complete output bytes
before publication. Output frame offsets are rewritten to start at zero.

The report's `source_frames`, `retained_start_frame` and
`retained_end_frame` are original-source, zero-based frame coordinates,
with the end exclusive; `frames` describes the **exported** count.
`source_peak` and `source_rms` summarize the full source file.
`output_peak_before_quantization` and
`output_rms_before_quantization` use only retained processed frames.
Without the trim option, `retained_start_frame=0`,
`retained_end_frame=source_frames` and the output is unchanged.

This is neither time-domain speech detection nor noise-reduction, musical
silence detection, perceptual gating, a loudness analyzer, a mastering
grade algorithm, nor proof of user rights. Long quiet passages inside
an active recording are never collapsed. Full-file DC correction is
computed before trimming, by design. Source and neural training corpora,
historic procedural DSP goldens, local API and web UI remain untouched.
All work remains on the unmerged draft development branch without releases.

## Explicit trim context handles and fade curves (20 November 2026)

The opt-in offline `wav-process` command can retain original audio immediately
around the first and last above-threshold sample frame. These are exact integer
frames, **not** approximate milliseconds:

```bash
sonumio wav-process input.wav --output output.wav \
  --trim-silence-dbfs -42 --trim-pre-roll-frames 960 \
  --trim-post-roll-frames 1920 --fade-in-frames 240 \
  --fade-out-frames 480 --fade-curve cosine
```

At 24 kHz, 960 frames are 40 ms and 1,920 frames are 80 ms. In the
**first verified read**, the trim detector takes the first and last frames
whose absolute ORIGINAL sample value in **any channel** exceeds the chosen
amplitude threshold. It then extends the common, sample-aligned interval
by the requested preceding/following context, clamping to the source
file rather than creating synthetic padding. The original complete WAV
is verified by SHA-256 on **both** reads, including discarded samples.
A wholly below-threshold input fails with no new output. Nonzero context
without a threshold is rejected; defaults remain zero-context.

The default `linear` edge fades retain historical numerical behavior.
An explicit `--fade-curve cosine` uses a raised-cosine *amplitude* ramp
`0.5 - 0.5*cos(pi*x)` from progress 0 through 1, and the reversed ramp
at the other end. The first/last fade samples are exactly zero for every
positive fade length, including a one-frame fade. Fades covering one
another multiply; they are not equal-power crossfades, true-peak
limiters or click-removal guarantees. Every interpolation coordinate is
derived from the **absolute retained output frame**, making output
independent of WAV packet partitions. Fades occur after optional
per-channel DC subtraction and before gain/quantization/dither.
All output digests still gate atomic publication.

Keep context and fades explicitly chosen. They do not perform speech
segmentation, noise gating, voice-activity detection, source separation
or any rights/consent verification. No historical audio generator,
experiment inputs or model checkpoints are changed by this addition.
