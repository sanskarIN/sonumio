# Verified long-form WAV quality inspection

Sonumio offers an independent, deterministic, **sample-level** signal-health
inspection command for existing canonical integer PCM WAVs. This is numerical
measurement, not Sonumio-1, semantic prompt evaluation, human listening,
recording-rights verification or a neural audio-quality score.

## Inspect without writing new audio

```bash
sonumio wav-inspect existing.wav
sonumio wav-inspect existing.wav --packet-frames 4096
sonumio wav-inspect existing.wav --expected-sha256 <64-lowercase-hex>
sonumio wav-inspect existing.wav --max-peak 0.95 --fail-on-warnings
```

The command emits **one JSON object only after the entire source** is read,
validated and hashed. A bad file, a truncated read, unexpected file changes or a
mismatched expected SHA256 result in exit code **2**, a diagnostic on stderr,
and **no JSON on stdout**. On a verified source, the exit code is 0; with
`--fail-on-warnings`, it is 1 when one or more signal thresholds fail.
Quality warnings are not treated as corrupt source files.

Optional frame-accurate window diagnostics:

```bash
sonumio wav-inspect existing.wav --timeline --expected-sha256 <64-lowercase-hex>
```

`--timeline` appends a `timeline` key containing zero-based, end-exclusive
one-second window boundaries on the **original sample clock**. Each window
reports its own peak, RMS, maximum absolute per-channel DC mean, clipping
fraction and active-frame fraction. The final window may be shorter than one
second. The window layout is fixed by sample rate rather than input packet
size, so seeking and comparison do not depend on transport fragmentation.
The command will keep at most 3,600 one-second windows for its one-hour
source cap (the standalone window accumulator allows 4,096 windows).
Timeline analysis is opt-in; the default JSON shape is unchanged.
No generated audio is stored by this diagnostic.

The JSON carries `inspection_protocol` and `inspection_controls`, including
the packet frame size, whether an expected SHA256 was enforced, whether
time-window inspection was enabled, and **all seven effective quality limits**.
This is important even when no warnings are emitted: two different threshold
policies can both pass the same audio while expressing different acceptance
criteria. The source digest is always the actual verified SHA256, but a digest
computed without an expected reference must not be mistaken for a previously
pinned/approved source. These controls are non-sensitive and omit local
absolute paths.

The JSON includes exact WAV geometry, the verified whole-file SHA256, a
`source_verified` flag and the `signal_health` report. Signal measurements
include sample peak, RMS, per-channel mean DC absolute magnitude, share of
samples at or above full scale, proportion of active frames, first/last frame
boundary amplitude, and per-channel RMS imbalance. Defaults match
`sonumio.audio_core.quality.measure_quality`; the user can override:

- `--max-peak`, `--max-clipping-fraction`, `--max-dc-absolute`
- `--max-boundary-jump`, `--max-channel-imbalance-db`
- `--min-active-fraction`, `--activity-floor-dbfs`

**Limits:** 8–192 kHz, 1–8 channels, integer 8/16/24/32-bit PCM, an exact
canonical 44-byte RIFF/WAVE header and maximum 3600 seconds. Large packet
allocation is refused. Arbitrary broadcast WAV, metadata chunks, floating-point
WAVE, RF64, WAVE_FORMAT_EXTENSIBLE and symlinks are deliberately unsupported.
The reader checks RIFF sizes, padding and source metadata, and refuses to
authenticate a changed or incomplete input. No network access, model weights,
uploads, automatic source modification or audio outputs are involved.

## Python use

```python
from sonumio.audio_core import inspect_wav_quality

result = inspect_wav_quality(
    "existing.wav", packet_frames=8192, expected_sha256=None,
)
print(result.source_sha256)
print(result.signal_health.as_dict())
```

For an already-trusted stream of `FramePacket` objects, use
`StreamingQualityAccumulator(sample_rate, channels, expected_frames=...)`,
send strictly consecutive packets via `update(packet)`, and only then call
`finalize()`. Supplying a known frame count is strongly recommended: the
accumulator rejects premature finalization, clock/layout mismatches, gaps,
duplicates and excessive input. The file-based entry point additionally
verifies the entire original WAVE representation, unlike standalone packet
aggregation.

With the default whole-file report, processing memory scales with **one PCM
packet and the channel count**, not the full recording duration. When the
optional timeline is enabled, it additionally retains **one compact result per
one-second window** (at most 3,600 records for an hour-long WAV). Each packet
accumulates float64 sums and sum of squares, plus fixed-size counters; rounding
may differ minutely with different packet boundaries. The independent NumPy in-memory quality implementation
is used as the regression oracle across bit depths and layouts.

No source may be described as safe, licensed, original or perceptually good on
the strength of these checks. Human and provenance review remain separate.

## Exact integer PCM endpoint detection

The report also contains `pcm_rails` with separate positive and negative
full-scale **integer** counts, their combined fraction of all channel samples,
and `audible_clipping_proven: false`. Rail occupancy can suggest saturation
but is not definitive proof of distortion: intentional waveforms may contain
endpoint values. This count is calculated **before** PCM decoding to float32.
It is therefore accurate even for positive 8/16/24-bit maxima that decode
below +1.0 and for 32-bit adjacent integer values that round together in
float32. The existing `signal_health.clipping_fraction` remains an
independent *decoded-sample* criterion (absolute float >= 1.0), preserving
the first-party quality-report protocol. Do not confuse the two metrics.

The verified file inspector accepts an optional rail-occupancy budget:

```bash
sonumio wav-inspect source.wav --max-pcm-rail-fraction 0.001 --fail-on-warnings
```

This checks the fraction of decoded integer PCM samples at the exact
positive or negative quantization endpoint. The limit is inclusive and
must be finite and within [0, 1]. Its effective value appears in both
`inspection_controls.max_pcm_rail_fraction` and
`pcm_rails.max_rail_fraction`; `pcm_rails.passed` records whether it
was respected. Without this explicitly supplied limit, the exact rail
counts are observational and `pcm_rails.passed` is true. With a budget
breach and `--fail-on-warnings`, the verified CLI report is still
emitted and exit code is **1**. The top-level
`inspection_policy_passed` boolean combines signal-health findings and
the optional rail policy; it is false if either fails. Invalid budgets fail with **2** and
no JSON. The rail fraction does *not* automatically modify the legacy
`signal_health.clipping_fraction`, because exact quantization endpoints
and decoded float full-scale samples measure different things.

The rail counts stay **unverified until the entire file is authenticated**,
just like the source SHA256. Their calculation uses constant-sized packet
storage and two integer counters, regardless of recording duration.

## Reader identity and race safety

The source reader preflights the exact leaf pathname without following links.
It verifies that the opened regular-file descriptor still matches that
preflight identity, then checks that the pathname still refers to that
same file when full-file reading is complete. This rejects ordinary
same-byte pathname replacement, symlink swaps (including platforms lacking
`O_NOFOLLOW`), path deletion and byte modifications before declaring a
source verified. The check uses platform-reported device, file identity,
size and nanosecond timestamps; it is **not** a sandbox against hostile
privileged filesystem manipulation or a guarantee of remote filesystem
coherency. Review the trust boundary for each deployment.

## Scientific scope and numerical caveats

`QualityReport` is retained for backward compatibility and shares its
warning decisions with the streaming implementation. The optional window
measurements are independent per-window aggregates (not LUFS or true peak).
Float64 sums and sums of squares are used; changing packet boundaries may
vary the last few bits of a reduction, so tests use tight numerical
tolerances rather than incorrectly demanding identical decimal rendering.
Real-model evaluation still requires independent source rights, holdouts,
blind listening, and prompt-grounding checks.

## Repeatable CPU scanner resource benchmark

Run the optional developer benchmark from the repository root:

```bash
python -m benchmarks.wav_quality_profile --seconds 121 --timeline
python -m benchmarks.wav_quality_profile --seconds 300 --packet-frames 1024 8192 32768
```

The benchmark generates a deterministic, rights-independent synthetic sine
mix as a temporary canonical 8-kHz mono WAV. Three default packet sizes each
run a full-source SHA-verified inspection in a **separate Python process**.
The benchmark compares source hashes, PCM rail fractions, sample peaks and
RMS across those packet sizes, and records wall time, real-time throughput
and whole-process peak RSS where the OS supports it. When enabled, timeline
window counts must also agree exactly. Scratch files are automatically
removed; nothing is uploaded or released.

**Interpretation:** peak RSS includes the Python interpreter, imports and
NumPy state, and a high-water mark is not the scanner's marginal allocation.
Windows currently reports RSS as unavailable rather than fabricating a
number. Timings exclude subprocess startup and synthetic-source creation,
and observed throughput is hardware-, OS- and load-dependent. This is
an engineering resource diagnostic, not a model-inference benchmark or proof
of perceptual quality, memory safety or a universal resource ceiling.


## Optional adjacent-frame continuity inspection

The legacy `signal_health.boundary_jump` measures the magnitude of the first
and last frames of the file; it is **not** a check for internal transitions.
To measure changes *between consecutive decoded frames* anywhere in the
source, opt in to:

```bash
sonumio wav-inspect source.wav --continuity --expected-sha256 <64-lowercase-hex>
sonumio wav-inspect source.wav --max-adjacent-step 1.1 --fail-on-warnings
```

`--continuity` adds `frame_continuity`: the maximum absolute difference
between adjacent decoded samples (including across PCM packet boundaries),
the maximum by channel and the number of frame-to-frame transitions.
`--max-adjacent-step` is an optional inclusive policy in the interval
0–2 (decoded PCM is bounded to roughly [-1,1]); it automatically enables
continuity measurement. The report then includes the number/fraction of
channel-sample transitions **strictly above** the specified maximum.
The `max_step_at_frame` field identifies the **later** frame of the
largest adjacent pair, using zero-based original-source frame numbering;
`max_step_channel` identifies its zero-based channel. Equal-magnitude
ties choose the earliest frame and then the lowest channel, independent
of packet size. For single-frame sources, both positions are `null`.
The top-level `inspection_policy_passed` combines this optional criterion
with the existing signal-health and exact integer-rail criteria. No
continuity work or extra `frame_continuity` output is added by default.

This is a *sample continuity* metric, **not an audible-click detector**:
percussive sounds, sharp attacks and other intended transients can have
large adjacent steps, and some audible defects can produce modest steps.
No perceptual claims should be inferred. The accumulator retains only
the last frame per channel and bounded scratch arrays. The verified file
report remains inaccessible until the whole canonical source is authenticated.
Single-frame recordings have zero adjacent transitions and zero measured step.
