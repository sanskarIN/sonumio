# Original music stem interchange

Sonumio can package the eight dry tracks of an existing symbolic arrangement
as synchronized stereo PCM16 WAVs. This is a local procedural workflow
for editing and mixing, not a trained source-separation model.

```bash
python -m sonumio.arrangement_cli compose --bars 8 --bpm 100 --seed 17 --output score.json
sonumio music-stems score.json --output stems.zip
sonumio music-stems-verify stems.zip --replay
sonumio music-stem-select stems.zip --track lead --output lead.wav
```

The same three commands are available via `python -m sonumio.music_stem_cli`.

## Contents and signal semantics

The deterministic stored ZIP contains a canonical score JSON, a canonical
manifest and exactly eight WAV members: keys, pad, bass, lead, pluck, kick,
snare and hat. All members start at frame zero, have identical length, are
stereo at 24 kHz and retain the instrument's original note-level stereo
placement. A single attenuation factor is calculated from the maximum
dry-stem sample peak so that no stem exceeds the 0.9 floating sample-peak
ceiling before PCM16 quantization.

The stems do not include static mix gains, room reflections, final music
mastering or score automation. They therefore do not sum byte-exactly to
the separately rendered master; their relative dry levels are preserved
apart from individual integer quantization error. A track without events
is retained as a silent, synchronized stem. No external music or recorded
instrument sample is bundled.

The package currently supports scores up to 60 seconds and ZIPs up to
50 MiB. All eight dry stems are held in memory by the existing renderer;
this is not a bounded-block streaming stem renderer. Larger arrangements
remain available through the existing arrangement render command.

## Verification

The manifest binds the score digest, serialized score hash, geometry,
declared source peak and common attenuation to each actual WAV's SHA256,
byte count, integer sample sums, squared sums, absolute peaks and nonzero
counts. Verification checks every member even when selecting only one.
It rejects wrong PCM geometry, unexpected names, noncanonical score or
ZIP bytes, duplicate/extra entries and changed receipts without extracting
files. No archives from other applications are accepted.

The archive checksum can be pinned using `--expected-pack-sha256`.
Creation can pin the input score with `--expected-score-sha256`.
Creation and selection support `--expected-output-sha256`, and require
`--overwrite` for an existing file. An input file cannot be replaced as
an output even with explicit overwrite.

A self-consistent manifest is not a signature or proof of source
generation. Only the optional replay re-renders the embedded score and
compares exact archive bytes, including the declared dry source peak.
Replay is local-runtime dependent: floating-point results may differ
between CPUs, NumPy versions or operating systems. The reported integer
sample statistics are not perceptual quality or loudness measurements.

Generated ZIPs, WAVs and any private artifacts remain outside the
source-only release. No weights or release publications are authorized
by this functionality.
