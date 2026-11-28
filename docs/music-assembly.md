# Arrangement assembly recipes

The assembly API builds a new symbolic arrangement from bar spans of one
or more existing verified score documents. It does not concatenate WAV
files, train a model, modify source scores, or claim identical rendered
percussion at a new position.

## Create a recipe

```python
from dataclasses import replace
from sonumio.arrangement import arrangement_digest
from sonumio.music_score_io import read_score
from sonumio.music_assembly import (
    AssemblyPlan, SourcePin, Segment, encode_assembly, assemble,
)

a = read_score("a.json")
# This example assumes at least four source bars and a tonic final bar.
last = a.plan.bars
recipe = AssemblyPlan(
    replace(a.plan, bars=4),
    (SourcePin("a", arrangement_digest(a)),),
    (Segment("a", 0, 1, "intro"),
     Segment("a", 1, 2, "theme", repeat=2),
     Segment("a", last - 1, last, "finale")),
)
with open("recipe.json", "wb") as stream:
    stream.write(encode_assembly(recipe))
result, report = assemble(recipe, {"a": a})
```

Bar ranges are zero-based, half-open. Each repeat copies the exact
relative ticks, pitches, lengths, velocities and pans of the selected
span. Matching source and destination root, mode, BPM and quantized
swing are required. Style and seed may differ deliberately. The final
harmony must be tonic; no implicit reharmonization is performed.

The default boundary policy rejects notes crossing a span endpoint.
Choose `boundary="clip"` explicitly to trim such notes. Every clipped
endpoint is counted in the report; clipping can introduce a new note
attack where a source note previously continued.

## Verify and replay

```bash
python -m sonumio.music_assembly_cli inspect recipe.json --source a=a.json
python -m sonumio.music_assembly_cli assemble recipe.json --source a=a.json --output assembled.json
```

An unsigned draft uses the same JSON structure as a pinned recipe except
that its outer `sha256` field is omitted. The `pin` command validates the
draft, source pins, and complete resulting arrangement before writing a
canonical recipe. Source paths are supplied separately and never stored
inside the recipe.

```bash
python -m sonumio.music_assembly_cli pin draft.json --source a=a.json --output recipe.json
```

Optional expected-recipe and expected-output digests protect replay and
publication. Outputs cannot overwrite an input, even when overwrite
is explicitly requested. All assembly results remain subject to the
normal score duration, event count, polyphony, scale and section rules.

This is a local symbolic editing workflow, not an audio splice engine.
