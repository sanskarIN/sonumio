# Bounded-block music automation

The existing music automation document can be rendered without allocating
all eight full-song floating stems:

```bash
sonumio music-automate score.json automation.json --block-frames 4096 --output mix.wav
```

The optional block size is 256–16384 frames. The original note synthesis,
gain and attenuation-only stereo balance laws, score-clock compilation,
track summation order, room response, global edge fades and peak ceiling
are preserved. The first pass gathers track statistics and the global
unscaled peak. The second pass replays the exact block stream, checks its
digest and encodes PCM16 using one global attenuation.

The same shared encoder serves the static blockwise renderer and validates
finite stereo float64 blocks, contiguous absolute offsets, complete frame
coverage and identical second-pass output. The final bounded WAV byte
buffer remains in memory. This is lower floating-point memory usage,
not constant memory with respect to output bytes or real-time synthesis.
The unscaled replay hash includes block boundaries, while the output WAV
is intended to match the whole-array automation renderer independently
of the selected block partition in the same numerical runtime.

Existing automation CLI behavior is unchanged when the flag is omitted.
Both paths retain atomic publication, explicit overwrite and source,
automation and output digest pins.
