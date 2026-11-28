# Interpretable pitch-class and triad hypotheses

Sonumio exposes a local, bounded chroma analyzer for exploring the
distribution of 12 equal-tempered pitch classes in recorded audio.
It uses frequency-domain magnitude evidence and simple major/minor
triad templates. Its outputs are **hypotheses**, not reliable
transcriptions, automatic music theory annotations, or trained models.

## Spectral analysis

The analyzer computes a centered Hann STFT at one sample clock and
maps nonzero frequency bins within an explicit Hz range to fractional
equal-tempered semitone positions. A bin between two chromatic
classes contributes proportionally to both neighbors. Magnitudes from
the original channels are combined only *after* computing the spectrum,
so opposite-phase stereo does not cancel to silence.

Per analysis frame, energies are normalized to sum to one where
usable energy exists. The resulting array has exactly 12 rows, one
per class, with the order

`C, C#, D, D#, E, F, F#, G, G#, A, A#, B`.

No external pitch reference or key signature is assumed beyond
the explicit A4 reference pitch, default 440 Hz.

## Candidate chord scoring

All 12 roots are compared against major and minor triad masks:
root, major/minor third, and perfect fifth. Each chord score is
the fraction of pitch-class energy assigned to its three tones.

A chord is *not* emitted unless each member class has at least the
minimum declared evidence, the total score passes an explicit floor,
and its score is separated from the next candidate. Isolated notes,
dense ambiguous profiles, and silence can therefore return unknown.

Adjacent frame hypotheses with the same symbol are grouped into
bounded audio-frame intervals. This is not a probabilistic hidden
Markov model or trained chord sequence decoder; no chord progression
or musical key is automatically inferred.

## Python example

```python
from sonumio.audio_core.harmonic_chroma import (
    ChromaSettings, analyze_chroma
)

settings = ChromaSettings(
    n_fft=4096,
    hop_frames=512,
    lowest_hz=65.,
    highest_hz=4000.,
    reference_a_hz=440.,
    minimum_rms=.002,
    minimum_triad_energy=.07,
    minimum_chord_score=.55,
    minimum_score_gap=.015,
    minimum_segment_frames=2,
)
result = analyze_chroma(audio, settings)
for segment in result.segments:
    print(segment.start_frame, segment.stop_frame, segment.symbol)
print(result.mean_chroma())
print(result.as_dict())
```

`audio` is a pre-existing canonical `AudioBuffer`, not a pathname
or URL. All operations are local and do not modify the source.

## Accuracy warnings

Harmonic overtones do not preserve pitch class uniquely: a single
instrument may reinforce the fifth and third even when only one
fundamental is played. Tuning variation, vibrato, strong dissonance,
heavy distortion, bass fundamentals and short percussion can also
confuse these templates. A sounding inversion is not distinguished
from the same root-position chord.

A simple template scorer should not claim to identify every chord
in arbitrary commercial recordings. Compare to hand-annotated,
properly licensed recordings before publishing accuracy metrics.
The report explicitly marks its chord classifications as hypotheses,
with no trained-model or perceptual-quality claims.

## Verified WAV inspection command

```sh
sonumio wav-chroma local-recording.wav --fft-frames 4096 --summary-only
```

The CLI checks source WAV geometry and optional
`--expected-sha256` using Sonumio's verified local packet reader.
It writes a single JSON document containing source provenance,
a 12-class chroma summary, candidate chord segments and explicit
uncertainty metadata. Omit `--summary-only` to include each
analysis-frame hypothesis.

The WAV file is never altered; invalid input returns a nonzero
status without a JSON success receipt. The command provides no
automatic copyright clearance, model weights or trained transcription.
