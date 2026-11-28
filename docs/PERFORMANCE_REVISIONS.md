# Performance overlays across score revisions

Performance indices refer to a specific canonical score. Adding notes
or reordering a score can change indices. Transfer resolves the original
event identity instead of blindly copying those indices.

    python -m sonumio.music_performance_transfer_cli preview-transfer original.json edited.json take.json
    python -m sonumio.music_performance_transfer_cli transfer original.json edited.json take.json --output revised-take.json

The default policy requires exact event identity, an unchanged quantized
score clock, and a match for every expressed note. Explicitly opting into
geometry matching permits velocity and pan changes, retaining the relative
expression. Missing expressions can be dropped only with --missing drop.
The --clock sample-offset policy retains absolute sample offsets even if
BPM or swing changes. All resulting sample intervals are revalidated.

Independent overlay edits can also be reconciled:

    python -m sonumio.music_performance_transfer_cli preview-merge score.json base.json left.json right.json
    python -m sonumio.music_performance_transfer_cli merge score.json base.json left.json right.json --output merged.json

Concurrent changes on the same event conflict. Even individually valid
timing edits may conflict when combined, so the entire result undergoes
sample-clock polyphony validation. Conflicts return 3 without publishing
an output. Validation errors return 2. Receipts record source and target
identities and decisions, not authenticated authorship.
