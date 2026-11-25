# Verified streaming WAV resampling

The independent audio-core library supports packet-based offline sinc
conversion without retaining a complete recording. Source clock, channel
count and total frames are declared before the first packet. The same
centered, cutoff-scaled kernel and DC-normalized boundary convention as
the offline resampler are used; absolute output frame coordinates make
partitioning independent of source packet length.

```python
from sonumio.audio_core.resample import ResampleSettings
from sonumio.audio_core.wave_resample import resample_wav_file

report = resample_wav_file(
    "input.wav", "converted.wav", settings=ResampleSettings(16000),
    output_bits=24, packet_frames=4096,
)
print(report.as_dict())
```

Command-line equivalent:

```bash
sonumio wav-resample input.wav --output converted.wav --target-rate 16000 --output-bits 24
```

Use `--expected-sha256` and `--expected-output-sha256` to require exact
trusted digests, `--dither-seed` for seeded TPDF, or `--clip-policy hard`
to explicitly allow clipping. Successful publication emits a JSON report;
a validation or integrity error emits no JSON and returns exit code 2.

The processor accepts up to 8192 source frames per call and retains only
the filter context required for subsequent output. Destination packets
are limited to 65536 frames. Kernel matrices respect the existing cell
budget. This is not a bound on total application memory: output packets
must be consumed and discarded by the caller. The processor requires
lookahead, so it is not designed for live device playback.

The WAV wrapper uses the canonical integer PCM reader and atomic writer.
Every source byte is hashed and the reader's identity checks must succeed
before destination publication. Optional trusted source and destination
SHA256 checks are supported. A failed source hash, output hash, clipping
policy, or frame-count check prevents publication and removes scratch.
Existing source/destination aliases are refused. The default clipping
policy is rejection; hard clipping is explicit. The standard seeded TPDF
quantizer is available with a dither seed.

The implementation is restricted to existing canonical RIFF limits and
does not add support for arbitrary codecs, RF64, WAVE_FORMAT_EXTENSIBLE,
real-time resampling or a trained audio model. It does not verify copyright
or perceptual transparency. Tests compare streamed and offline samples,
rate extremes, irregular partitions, lifecycle failures, bounded history,
WAV bytes and digest failures.
