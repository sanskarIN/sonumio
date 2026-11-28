# Reversible score patches and conservative three-way merge

Use the score patch CLI to review or replay structural arrangement changes
without replacing the complete source score blindly.

```bash
sonumio music-score-diff original.json edited.json --output change.json
sonumio music-patch-inspect change.json
sonumio music-patch-apply original.json change.json --output verified.json
sonumio music-patch-reverse original.json change.json --output undo.json
sonumio music-patch-apply verified.json undo.json --output restored.json
sonumio music-patch-chain original.json change.json undo.json --output combined.json
```

A patch stores canonical minimal event removals/additions and complete
source/target metadata snapshots (clock, sections and harmony), plus
both semantic score digests and event counts. Exact event identity uses
canonical JSON bytes to avoid conflating integer and float spellings.
Decoding validates all fields and the patch digest; applying a patch
also checks the source digest, all removals/additions, complete target
arrangement validation and target digest. Inspection without the source
does not verify endpoint claims. A chain is bounded to 16 patches, with
each intermediate score independently validated before collapsing to a
minimal delta.

To merge separate score edits:

```bash
sonumio music-score-merge base.json left.json right.json --output merged.json
```

Tracks are conservative atomic conflict domains: independent track
changes merge; different edits on the same track conflict even if
their note ranges are disjoint. Incompatible metadata changes conflict,
and a unilateral metadata change conflicts with unrelated changes in
the other branch. Identical changes and unchanged branches resolve
directly. Successful results pass arrangement validation and patch replay.
Conflicts produce a structured JSON report, return exit code 2, and
do not publish or modify an output file.

All publication uses atomic local no-clobber or explicit overwrite.
Source inputs cannot also be the output, including file aliases.
Optional expected digest pins distinguish semantic score/patch identity
from the byte-level output checksum. The schema is bounded to 2 MiB and
4096 removed/added events each. No network, audio or trained weights
are involved.
