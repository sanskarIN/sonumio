# Monophonic pitch hypotheses with cumulative mean normalized difference

Sonumio provides an opt-in CPU reference for estimating a single
fundamental-frequency hypothesis in each audio analysis frame.

## Algorithm

The estimator compares a fixed reference window of `W` samples against
shifted windows up to a bounded candidate lag. For each lag `tau`,

`d(tau) = sum((x[j] - x[j + tau]) ** 2)`

is computed using valid correlation and squared-energy prefix sums. The
window length remains fixed across lags, avoiding a systematic shorter-
window bias at large lags.

A cumulative mean normalized difference (CMNDF) then finds the first
candidate trough below an explicit threshold. If no candidate crosses
this threshold, an optional, stricter fallback accepts only a sufficiently
small global trough. Optional parabolic refinement makes frequencies
subsample rather than strictly integer-lag estimates.

The default channel policy selects the highest-RMS channel within the
analysis frame rather than downmixing potentially opposite-phase stereo
signals. The frame RMS is separately compared to a user-defined silence
floor. Below that level, the result is marked *unvoiced* rather than
inventing a note.

## Python example

```python
from sonumio.audio_core.yin_pitch import PitchSettings, track_pitch

settings = PitchSettings(
    frame_frames=1024,
    hop_frames=256,
    minimum_hz=65.,
    maximum_hz=1400.,
    yin_threshold=.15,
    fallback_threshold=.3,
    silence_dbfs=-60.,
)
result = track_pitch(audio, settings)
for observation in result.frames:
    print(observation.center_frame, observation.frequency_hz,
          observation.confidence, observation.voiced)
print(result.summary.as_dict())
```

The source `audio` must be an existing `AudioBuffer`. The output uses
the same source sample rate and integer frame clock, with no file or
network access and no neural-model weights.

## Limits and correct interpretation

- Suitable primarily for *monophonic*, pitched, relatively periodic audio.
- Polyphony, subharmonics, formants, distortion and octave relationships
  can produce wrong or ambiguous fundamentals.
- Confidence comes from the YIN difference curve. It is not a calibrated
  probability, nor is it a perceptual score.
- No pitch labels, music transcription, singer identification, or
  voice-source separation are claimed.
- Silence returns explicitly unvoiced observations; unvoiced frames do
  not contribute to median pitch statistics.
- Bounded work geometry prevents excessive analysis-frame counts,
  difference-function products and scratch memory.
- Input data are never modified; results contain structured observations
  and transparent method metadata.

Validation uses synthesized single-frequency and inverted-channel
waveforms with known exact sample clocks. Real recordings require listening
and independent annotations to evaluate musical accuracy.

## Verified local WAV command

```sh
sonumio wav-pitch input.wav --min-hz 65 --max-hz 1400 --summary-only
```

Use `--expected-sha256` to bind the read to a trusted lowercase
SHA-256 of the source WAV. The command verifies the original file via
the canonical packet reader and prints one deterministic JSON object.
Omit `--summary-only` to include all per-frame pitch observations.
No output file is created, and the input file is not modified.
