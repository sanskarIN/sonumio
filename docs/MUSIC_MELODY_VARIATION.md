# Constrained melodic variation

The melody transformer selects a new pitch sequence on a monophonic
instrument (lead, bass or pluck) while preserving rhythm, velocity, pan,
other instruments, section structure and existing harmony. It never
imports external music.

A finite candidate lattice is built from the source key, chosen pitch
range, maximum shift and optional chord tones on beat-aligned notes.
Boundary pitches and maximum leap are hard constraints. The first and
last selected notes can be preserved, and the last note can optionally
be required to have the tonic pitch class. Incompatible constraints
fail explicitly rather than silently being relaxed.

The exact dynamic program minimizes integer local contour and source
deviation costs plus interval and movement costs. Equal cost paths
resolve lexicographically. The five contour modes are original, mirror,
arch, ascending and descending. The contour is an objective target,
not a guarantee of strictly increasing or decreasing pitches. The
search is capped at 512 notes and 12 million candidate transitions.

Example variation document:

    {
      "format": "sonumio-constrained-melody-variation-v1",
      "score_sha256": "replace-with-the-source-score-digest",
      "track": "lead",
      "first_bar": 0,
      "bars": 4,
      "mode": "mirror",
      "minimum_pitch": 48,
      "maximum_pitch": 84,
      "max_leap": 12,
      "max_shift": 12,
      "preserve_ends": true
    }

Inspect a result without publishing a file:

    python -m sonumio.music_melody_cli inspect score.json variation.json

Apply the exact same transformation atomically:

    python -m sonumio.music_melody_cli apply score.json variation.json --output variation-score.json

The receipt includes source, recipe and result digests, candidate counts,
objective cost, original and transformed pitches, and geometry. No trained
model is used. The result is a normal canonical score and can be passed
to the existing MIDI, stem and audio renderers.
