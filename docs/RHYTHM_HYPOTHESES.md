# Interpretable onset and tempo hypotheses

Sonumio's optional rhythm analysis makes a deterministic acoustic estimate
of *when* short energy onsets occur and *which periodicities* may represent
a tempo. It is deliberately not a transcription model, a beat-quality
score, or a reliable time-signature detector.

## Method

1. Compute a centered, Hann-windowed STFT over the original sample clock.
2. Compute per-channel positive spectral flux: positive differences in
   successive magnitude spectra, averaged across frequency bins.
3. Average the channel novelty curves, optionally smooth over a short
   odd window, subtract the median background and normalize its peak.
4. Find local peaks with declared minimum strength and prominence.
   Enforce a refractory interval in STFT hops, preferring stronger peaks
   and then earlier timestamps.
5. Evaluate normalized lag correlation at lags corresponding to the
   configured BPM range and retain distinct local hypotheses.

The analysis does not shift, normalize, or mutate the source sound.
Onsets are integer audio frame positions, not floating seconds rounded
later. For each tempo hypothesis, the report records a score, lag in
analysis hops and interbeat distance on the source sample clock.

The optional beat-frame preview extrapolates from the **earliest detected
onset**. It is a convenience hypothesis, not a validated measure grid.
A dense arrangement can exhibit half-time, double-time and syncopation
ambiguities, and may require human confirmation of musical meter.

## Python example

```python
from sonumio.audio_core.rhythm import RhythmSettings, analyze_rhythm

settings = RhythmSettings(
    n_fft=512,
    hop_frames=128,
    min_bpm=60.,
    max_bpm=200.,
    min_spacing_seconds=.08,
    min_strength=.15,
    prominence=.08,
)
result = analyze_rhythm(audio, settings)
for onset in result.onsets:
    print(onset.frame, onset.seconds, onset.strength)
for tempo in result.tempos:
    print(tempo.bpm, tempo.score)
print(result.beat_frames)
print(result.as_dict())
```

The `audio` parameter is an existing canonical `AudioBuffer`; no file
paths or network endpoints enter this operation.

## Failure modes and numerical contracts

- A silent signal returns an all-zero novelty curve, no onsets and no tempo.
- A clip with fewer than three accepted onsets reports no tempo.
- Strict BPM bounds may omit the actual tempo.
- Half-time and double-time interpretations can appear as separate
  candidates. Scores are *relative periodicity evidence*, not calibrated
  probabilities of correctness.
- Long or extremely finely sampled analysis is rejected before STFT
  allocation if its frame or correlation budget is too large.
- STFT memory is independently bounded by the shared spectral engine.
- All event positions are on the original sample clock; no resampling is
  performed.
- Results are immutable and explicitly tagged as non-neural hypotheses.

Use original or public-domain click tracks and independently licensed audio
for validation. Rhythm candidates should be confirmed by listening and
visual inspection before driving an automatic musical arrangement.
