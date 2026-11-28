# Sample-clock performance overlays

A symbolic score uses integer musical ticks. A performance overlay adds
sample-level timing and articulation without changing the original arrangement
or its MIDI file. Each expression addresses an index in the canonical event
array and is pinned to the original arrangement SHA256. Both start and end
offsets are signed sample counts. Velocity and pan changes saturate explicitly
at the original instrument bounds. Invalid durations, out-of-range positions,
and sample-clock polyphony violations fail before synthesis.

Generate a deterministic humanization document:

    python -m sonumio.music_performance_cli humanize score.json --seed 42 --track lead --track keys --timing-frames 120 --shortening-frames 80 --velocity-variation 0.08 > performance.json

Inspect the document and its geometry:

    python -m sonumio.music_performance_cli inspect score.json performance.json

Render locally:

    python -m sonumio.music_performance_cli render score.json performance.json --output expressive.wav

An optional automation document can be passed to render with
--automation. Its clock is the original score clock rather than the
displaced note clock. Existing MIDI exports do not contain these offsets.

The humanizer moves connected, overlapping intervals as a group, preserving
chord synchronization and preventing added polyphony. Adjacent groups are
separated at their original gap midpoint. Duration variations only shorten
notes; they never increase overlap. Seeded draws use domain-separated SHA256
and do not depend on a particular NumPy random generator release.

The expression format has a 512 KiB limit and at most 4096 rows. Start and
end offsets are bounded to 100 ms; generated timing and shortening are each
bounded to 50 ms. Full rendering is limited by the existing stem allocation
cap and an additional total note-frame budget. This is procedural
performance expression, not learned musical interpretation.
