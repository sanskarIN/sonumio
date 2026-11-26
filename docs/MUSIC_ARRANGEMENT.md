# Structured instrumental arrangement, stems and MIDI

The original music v1 engine remains unchanged. This separate engine
generates an eight-track symbolic score with sections, then renders
stereo PCM using original procedural instruments. It is not neural AI.

## Local workflow

```bash
python -m sonumio.arrangement_cli compose --style cinematic --mode minor --bpm 96 --bars 16 --seed 29 --swing 0.18 --output outputs/score.json
python -m sonumio.arrangement_cli inspect outputs/score.json
python -m sonumio.arrangement_cli midi outputs/score.json --output outputs/score.mid
python -m sonumio.arrangement_cli render outputs/score.json --output outputs/score.wav
python -m sonumio.arrangement_cli render outputs/score.json --solo lead --room 0 --output outputs/lead.wav
```

The composer supports 4–32 bars, 60–160 BPM, 120 seconds maximum,
major/minor, MIDI root 48–71 and swing 0–0.6. The absolute tick clock
has eight ticks per beat; swing warps timing within a beat, not its
endpoints. The score partitions bars into intro, theme, variation,
break and finale where length permits. Diatonic triad voicing minimizes
movement from the prior chord. The lead uses a repeated and varied
interval motif, chord tones on strong beats and a final tonic. Bass
and plucks follow chord classes. Section-dependent density gives a
basic arrangement arc rather than repeating one full texture.

## Instruments and mix

Keys, pad, bass, lead and pluck are finite harmonic/modal synthesizers.
The kick, snare and hi-hat use original oscillator/noise models. Each
percussion event receives a seed from its own tick and pitch. No sample
library or pretrained model is loaded. Smooth note edges, conservative
partial cutoffs and explicit track polyphony bounds reduce numerical
artifacts, but are not a guarantee of perceptual realism or strict
bandlimiting under modulation.

Dry stereo stems are returned as independent read-only arrays through
the Python API. Rendering a complete score is independent of later
mute, solo and track gain choices. The mix uses equal-power pans,
finite room reflections and attenuation-only peak protection. Sample
peak is not true peak or integrated loudness. Stems consume at most
256 MiB of PCM allocation; temporary rendering/mix arrays require
additional memory.

## Interchange and validation

Versioned score JSON rejects extra/duplicate keys, nonfinite values,
oversize arrays, invalid event geometry and digest mismatch. The
score digest authenticates no author; it is a reproducibility checksum.
MIDI type 1 uses a tempo/conductor track plus eight instrument tracks,
section markers, correct delta-time encoding and note-off ordering.
MIDI playback depends on the external instrument synthesizer; it does
not reproduce Sonumio's waveform exactly.

Each CLI command publishes at most one artifact atomically, rejects
implicit overwrite and supports an optional expected output SHA256.
Input score SHA256 can be pinned separately. This is a trusted local
filesystem workflow, not a concurrent hostile-user sandbox.

Numerical score analysis includes section densities, track occupancy,
polyphony, lead intervals, chord-tone adherence and relative interval
phrase repetition. Audio analysis includes centered stereo correlation,
mono RMS retention, channel DC, peaks and half-second energy windows.
These diagnostics do not establish musical preference, copyright
clearance, perceptual fidelity or neural generation quality.
