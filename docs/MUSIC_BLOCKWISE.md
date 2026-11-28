# Blockwise music rendering

Render a verified score without holding all eight full-length floating
stems in memory:

```bash
sonumio music-render-blockwise score.json --output music.wav
sonumio music-render-blockwise score.json --output music.wav --block-frames 4096 --overwrite
```

The renderer retains only active note waveforms, one bounded block of
each track, the short room-response history, scalar diagnostics and the
final encoded WAV byte buffer. Existing note synthesis, event order,
stereo placement, per-track float32 accumulation, mix summation order,
room taps, global edge ramps and PCM16 quantization are preserved.

A first deterministic pass measures the peak of the complete mix. A
second pass recomputes it, verifies the unscaled block-stream hash and
encodes globally attenuated PCM. The receipt includes the score and WAV
digests, block size, actual peak and RMS, track statistics, and replay
verification. A hash of the raw block stream is partition-dependent;
the encoded output is intended to be partition-independent and match
the existing renderer within the same numerical runtime.

The default block length is 8192 frames (about 341 ms at 24 kHz).
Supported block sizes are 256–16384 frames. The score retains the
existing 120-second limit and the note/polyphony bounds. This is not
constant memory with respect to final WAV bytes, and the two-pass
synthesis trades additional computation for lower floating PCM memory.
It is not a real-time or trained AI renderer.

The CLI reuses atomic no-clobber publication and optional score and
output SHA-256 pins. Mute, solo, room and sample-peak ceiling controls
are supported. Score-clock automation remains available through the
separate music-automate command.
