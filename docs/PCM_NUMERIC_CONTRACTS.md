# PCM numeric contracts and bounded arithmetic

The canonical channel-first conversion now checks the requested output
allocation, not only the input array's byte count. Finite double samples
outside the float32 representable interval are rejected before narrowing.
Ordinary overrange PCM remains permitted: this change does not clip or
normalize audio. The canonical buffer and immutable streaming packet use
the same conversion boundary.

The optional copy=False flag means reuse storage when possible, not forbid
all allocations. NumPy's asarray performs necessary dtype/layout conversions
on both supported NumPy major versions. Immutable buffer construction
still creates independent bytes-backed storage.

Peak statistics use signed extrema. RMS uses peak-scaled double arithmetic
in bounded blocks, and then combines channel RMS with the same scaling
method. This avoids squaring extremely large or small finite inputs
directly. Small relative floating-point changes are expected; this is not
a loudness or perceptual quality measure.

Mixing validates all source shapes and gains before accumulation. It retains
validated source views rather than widening up to 64 complete recordings,
and widens one bounded slice at a time. Float64 overflow is a contract
error; the final float32 conversion rejects overflow rather than returning
nonfinite PCM. The original gain order, zero padding, cap and explicit
normalization semantics are retained. This is a bounded in-memory mixer,
not a streaming renderer or an aggregate process-RSS guarantee.

Dedicated regressions cover both signs of float32 narrowing overflow, large
Python integers, NumPy copy behavior, buffer ownership, allocation preflight,
large/small RMS, batch partitioning, source aliasing and mix overflow.
No neural model, dataset, generated media or release is changed.
