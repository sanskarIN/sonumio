# Circular PCM loop search and export

This original local DSP reference helps select and repeat sections of
generated ambient sound or music. It is not an AI composition model, a
beat-alignment detector, or proof of a perceptually seamless transition.

## Endpoint score

The caller declares exclusive source stop, nominal start, independent
bounded search radii, context window, stride, period constraints and
optional overlap. The coarse grid includes the nominal endpoint and
both range edges even when the step does not divide the range. A bounded
refinement explores integer offsets near the highest-scoring coarse
candidates. All candidates are ranked deterministically by score,
distance to the requested endpoints and frame positions.

The score combines the following separately measured properties:

- RMS-normalized error between the actual boundary step and the local
  predicted step from adjacent incoming/outgoing derivatives.
- RMS-normalized change in slope and RMS level.
- Centered joint-channel waveform correlation.
- Normalized Hann-window spectral-band distribution distance.
- A silence penalty to avoid treating inactivity as universally ideal.

The raw boundary step is also reported. For blended loops the endpoint
error and slope terms are downweighted; waveform and spectrum matching
are proxies on the declared context, not perceptual predictions.
Opposite-polarity stereo is never mixed to mono for endpoint selection.
No candidate outside the declared period and source bounds is returned.

## Exact render geometry

For a source interval [start, stop) and overlap B, a loop cycle contains
stop-start-B frames. With B > 0, the cycle begins at start+B, retains the
middle, then blends the last B source frames into the first B. The last
sample follows the original head immediately before the first sample
of the cycle. The crossfade consumes overlap rather than adding silence
or altering the source clock. Linear and cosine gains sum to one; equal
power uses sine/cosine gains and may exceed the original sample peak.

The report separately measures actual circular boundary step, slope
change, interior step statistics and sample peak after any explicit
attenuation. These are numerical diagnostics, not listening-test results.

## Usage

```bash
python -m sonumio.audio_core.loop_cli loop-search ambience.wav --start-frame 24000 --stop-frame 120000 --blend-frames 256
python -m sonumio.audio_core.loop_cli loop-render ambience.wav --start-frame 24000 --stop-frame 120000 --blend-frames 256 --auto-search --repeats 3 --output loop.wav --peak-ceiling .95
```

Without --auto-search, the exact declared endpoints are used.
Source snapshots are verified over the canonical WAV descriptor before
the search or render. The source and destination must not alias. The
source bit depth is preserved unless an explicit integer output depth is
selected. Seeded TPDF is opt-in. The output is rejected if its sample peak
exceeds normalized PCM, unless a requested ceiling attenuates the signal.
Publication uses a staged, verified, atomic no-clobber WAV writer.

Work is bounded by candidate count, context-product count, cache memory,
source duration, offline snapshot bytes, output geometry and repeat
count. The reference does not promise arbitrary long-form streaming
loops. All tests use synthetic original PCM and do not load models.
