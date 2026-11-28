# Fixed-clock streaming STFT and overlap-add reconstruction

Sonumio's independent audio core includes an incremental STFT analysis
and inverse STFT reference. It is designed for chunked, offline CPU
workflows, where application buffer boundaries must **not** change the
mathematical transform.

The implementation deliberately uses the same FFT size, window
generation, center padding, right padding and hop clock as the existing
`stft` and `istft` functions.

## Incremental analysis

`StreamingStft` accepts any nonempty float32/float64 channel-first
PCM chunks with a fixed declared channel count and sample rate.
Each push returns all FFT frames for which the input prefix is
currently complete, shaped `[channels, bins, frames]`.

An incomplete FFT window remains buffered until future samples arrive.
Calling `finish` adds the exact center/right padding required by
the selected `StftConfig`, produces the final frames, and permanently
closes the one-shot stream.

The cumulative source duration, spectral output and FFT-work budgets
are enforced independently. A rejected input chunk does not advance
the internal sample or FFT clock.

## Incremental inverse

`StreamingIstft` requires the original source frame count up front.
Each pushed complex FFT frame is transformed with the declared window.
The weighted overlap-add accumulator finalizes only sample positions
which no future FFT frame can modify. Center padding is removed by
absolute source-clock offset, not by guessing chunk lengths.

The inverse requires exactly the expected number of STFT frames.
An incomplete sequence can receive additional frames before finish.
An oversized or nonfinite complex input is rejected before updating
the reconstruction state.

For both directions, the analysis frame clock is an intrinsic
property of the declared audio geometry—not the caller's push size.

## Python example

```python
from sonumio.audio_core.stft import StftConfig
from sonumio.audio_core.streaming_stft import (
    StreamingStft, StreamingIstft
)

cfg = StftConfig(n_fft=1024, hop_length=256)
analysis = StreamingStft(channels=2, sample_rate=48000, config=cfg)

first = analysis.push(left_pcm_chunk)
second = analysis.push(right_pcm_chunk)
last = analysis.finish()

inverse = StreamingIstft(
    channels=2,
    sample_rate=48000,
    original_frames=original_frame_count,
    config=cfg,
)
# Spectral chunks can be pushed in different size batches,
# provided the declared channel/bin geometry is unchanged.
out1 = inverse.push(first)
out2 = inverse.push(second)
out3 = inverse.push(last)
out4 = inverse.finish()
```

The convenience functions `streaming_stft(AudioBuffer)` and
`streaming_istft(Spectrogram)` assemble these outputs into the
existing immutable Sonumio container types.

## Accuracy and limitations

A chunk-invariant transform must reproduce the full offline STFT
to complex floating-point precision and the inverse to normal
float32 reconstruction tolerance. Test cases cover input shorter
than one FFT, arbitrary incomplete blocks, multiple windows, center
padding enabled/disabled, and no-end-padding policy.

This is **not an allocation-free or real-time-safe callback**:
it uses NumPy heap allocations and FFTs at push boundaries. It
has no hardware clock synchronization, deadline certification,
external file access or neural model inference.

The transform itself does not denoise or create content; it is
a reusable numerical foundation for streaming audio analysis.
