# Self-contained music project files

A `.sonumio` project file is bounded canonical JSON containing the
complete validated symbolic score and automation/mix document. It
contains no external paths, audio recordings, executable code or
model weights. The project ID binds the title, revision, parent ID,
semantic score digest, canonical score bytes digest and normalized
automation digest.

```bash
sonumio-music-project create --title "Original music" --score score.json --automation mix.json --output music.sonumio
sonumio-music-project inspect music.sonumio
sonumio-music-project render music.sonumio --output preview.wav
sonumio-music-project extract-score music.sonumio --output extracted.json
sonumio-music-project extract-automation music.sonumio --output extracted-mix.json
```

Omitting automation at creation embeds the default eight-track mix
with no automation lanes. Project validation reuses the original
arrangement and automation constraints and rejects lanes extending
past the score. Rendering uses the existing deterministic procedural
mixer and checks its score, automation and WAV receipt.

A revision records its predecessor's project digest:

```bash
sonumio-music-project revise music.sonumio --title "Second version" --output revised.sonumio
sonumio-music-project verify-parent music.sonumio revised.sonumio
```

A revision must change at least one of title, score or automation.
Revisions are immutable content links, not authenticated signatures,
a database, or proof of authorship. Parent verification requires the
actual supplied predecessor. The reader verifies embedded digests
and canonical nested documents; optional expected project digests
provide an external identity pin.

The project format has a 2 MiB byte limit. Atomic publication protects
existing output unless explicit overwrite is requested, and input
file aliases cannot be used as output. The CLI ships as a separate
entry point and can also be run with
`python -m sonumio.music_project_cli`.
