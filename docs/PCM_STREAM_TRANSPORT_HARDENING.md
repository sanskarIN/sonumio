# PCM Stream and Local WAV Transport Hardening

**Status:** independent development-branch audio-core safety patch.
**Scope:** `sonumio.audio_core.streaming` and `waveio` only.
The historical Sonumio procedural renderer, API, neural research, synthetic
training datasets, and numerical golden fingerprints are unchanged.

## Why these limits were added

The prior streaming framer bounded input chunks but not overlapping **output**
windows. A valid 1,000,000-frame packet with hop=1 could allocate nearly one
million window objects. Increasing the analysis window makes this an extreme
copy-amplification hazard even when the original PCM fits the buffer budget.

The streaming framer now preflights, before accepting a packet:

- no more than **16,384 materialized windows per push**;
- no more than **64 MiB total output window PCM bytes per push**;
- unchanged input packet/history caps and sample-rate/channel continuity.

The limits constrain *each push*; smaller contiguous FramePackets can be used
for dense overlapping analysis. Rejected pushes leave stream position/history
unchanged. This is synchronous local framing, **not real-time backpressure**,
thread isolation, or bounded memory for an application retaining all returned
windows. Such applications must bound their own accumulation.

The packet collector now caps **50,000 retained packet objects** in addition
to the pre-existing total frame/duration budget. A million one-frame packets
were previously possible without exceeding the PCM data byte limit, while
their Python object overhead could dominate memory. Its check occurs before
append; consumers may coalesce tiny packets.

## Atomic WAV destination creation

The original WAV writer encoded into a fsynced sibling temporary file, then
used `exists()` followed by `os.replace()`. When `overwrite=False`, a
competing process could create a destination after the final existence check,
and `replace` would silently overwrite that new file.

The local writer now uses an **atomic hard-link create-if-absent** when
`overwrite=False`. A collision fails and the scratch file is cleaned up.
`overwrite=True` explicitly retains atomic replacement. Unsupported
hard-link filesystems **fail closed**, rather than falling back to racy
check-then-replace or direct partially visible writes. Parent directories
must remain under the developer's control; neither mode protects arbitrary
user-supplied filesystem paths in a multi-user service.

The WAV reader now opens one file descriptor, checks `fstat` for a regular
file and a size bound, and reads at most `MAX_WAV_BYTES + 1` bytes. POSIX
`O_NOFOLLOW` rejects symlink path swaps between the preliminary check and
open. Operating systems without `O_NOFOLLOW` retain best-effort rejection
of existing symlinks; this is explicitly **not** a public hostile-filesystem
security guarantee.

## Verification

`tests/test_audio_core_boundary_safety.py` exercises retained-packet caps,
overlap amplification, output-byte budgets, transactional push rejection,
partitioned framing, destination-creation races, explicit replacement, file
size rejection, symlink swaps on POSIX, and failed-hard-link cleanup.

```bash
python -m pytest -q tests/test_audio_core_boundary_safety.py
python -m pytest -q tests/test_audio_core_*.py
```

The independent audio-core CI tests Linux Python 3.11 with NumPy
1.26.4/2.4.6, Linux Python 3.12 with NumPy 2.4.6 and newly adds Windows
Python 3.12 with NumPy 2.4.6. These jobs must succeed before claiming any
Windows compatibility. Numeric coverage is not a substitute for memory-limit
stress testing or independent security review.

Provenance `hashing.digest_file` also uses one bounded, ordinary-file
file descriptor. Where supported, no-follow and nonblocking open flags
reject symlink/FIFO swaps. It checks starting vs ending file metadata and
exact read length. File metadata can be manipulated; this is NOT a signed,
immutable content snapshot or proof of usage rights.

The Web workflow now uses Node 24 to avoid the observed npm 10.9.x
`edgesOut` peer-tree crash. Since `web/` has no committed lockfile, its
`npm install` remains nonreproducible; producing and reviewing a lockfile
is a separate gate. CI success on the new runtime is not presumed.

**No release, pre-release, tag, public output WAV, or model-weight publication
is authorized or performed. Draft PR #1 stays open and unmerged.**
