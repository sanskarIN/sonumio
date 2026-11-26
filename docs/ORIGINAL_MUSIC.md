# Original procedural instrumental music

This local-only generator creates a symbolic score, then synthesizes
stereo PCM with original mathematical oscillators and seeded percussion.
It is not text-to-music AI, has no training corpus and uses no external
melody library, recordings, voices or pretrained weights.

```bash
python -m sonumio.music_cli --style calm --mode major --root 60 --bpm 100 --bars 4 --seed 17 --output outputs/calm.wav
python -m sonumio.music_cli --style cinematic --mode minor --root 57 --bpm 84 --bars 4 --seed 29 --output outputs/cinematic.wav
python -m sonumio.music_cli --style rhythmic --mode major --root 60 --bpm 128 --bars 4 --seed 41 --output outputs/rhythm.wav
```

Supported controls: three styles, major/minor, MIDI tonic 48–71, tempo
60–160 BPM, one to eight bars of 4/4, unsigned 32-bit seed. The score uses
8 ticks per beat; exact absolute tick-to-frame rounding avoids accumulated
timing drift. At most 512 bounded events. Output is stereo 24 kHz PCM16,
with up to 32 seconds at the slowest tempo.

The generator uses diatonic chord progressions, nearest-voice triad
inversions, interval-motif variation, chord-tone downbeats, and final
tonic resolution. Pad, bass and lead tracks have distinct additive
harmonic spectra and smooth envelopes; cinematic/rhythmic styles add
kick and high-pass-like seeded noise hats. Equal-power stereo pan,
finite cross-feed delays, edge fades and attenuation-only peak protection
are applied during rendering. No external model, sound library or music
file is required. This is a simple musical baseline, not an orchestra
or proof of perceptual quality. Seeded scores are reproducible within
the implementation; floating-point waveform bytes can vary across platforms.

The JSON report records the score and WAV SHA256, sample geometry,
track counts, harmony degrees and numerical signal health. Output
publication uses the existing atomic WAV exporter; existing files require
explicit overwrite and an optional expected digest can reject a mismatch.

## Research direction

Separate symbolic planning from waveform rendering so future learned
symbolic models can be evaluated for rhythm, harmony, structure and
variation independently of instrument sound. See the general review
[Computational Creativity and Music Generation Systems](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00014/full)
and [Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models](https://arxiv.org/abs/2405.09901)
for background. This implementation is an original rule-based baseline,
not an implementation of either paper's models.

Regression tests check timing, scale membership, final tonic, nonoverlap
of melody events, stereo signal health, deterministic replay and atomic
publication. Future experiments should add matched prompt/parameter
controls, symbolic diversity metrics, blind listening and learned
instrument synthesis. Do not equate these structural tests with
musical preference or guarantee of originality against all existing music.
