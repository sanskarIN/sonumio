# Studio note editor

The browser music studio can now inspect its original symbolic notes,
select a track and bar, view a pitch/tick grid, and add, replace or
remove individual notes. Undo/redo history is local and bounded.
Note editing does not require audio or uploaded external files.

The original composed score is identified by its SHA-256 digest.
Every draft refers to the canonical original event indices; removing
a note does not shift the meaning of subsequent edits. A request
contains up to 128 total removals, replacements and additions.
The server re-composes the original plan, verifies the base digest,
applies the entire transaction and validates the final arrangement.
Edited plans have a new score digest and expose their canonical notes.

The UI checks note pitch, scale membership, tick geometry, velocity,
pan, duration and polyphony before accepting changes. The backend is
authoritative and repeats validation. Changing composition controls
clears the old draft. Rendering and export send the edited transaction
and pin the verified edited score digest; stale plans cannot silently
render a different score.

The same local queue, atomic export receipts and procedural synthesis
engine are retained. The piano roll is a symbolic visualization, not
a waveform editor or trained neural music generator. No model weights
or audio files are bundled.
