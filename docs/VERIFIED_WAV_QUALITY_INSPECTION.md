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

Memory use is proportional to **one packet and the channel count**, not the
full recording duration. Each packet accumulates float64 sums and sum of
squares, plus fixed-size counters; rounding may differ minutely with different
packet boundaries. The independent NumPy in-memory quality implementation
is used as the regression oracle across bit depths and layouts.

No source may be described as safe, licensed, original or perceptually good on
the strength of these checks. Human and provenance review remain separate.
