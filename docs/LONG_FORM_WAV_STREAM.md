# Sonumio local packet-to-WAV streaming export (research infrastructure)

**Development status:** independent, optional `sonumio.audio_core.wave_stream`
module. It does not modify the existing Sonumio procedural SFX renderer,
voice/music models, neural training, training datasets, API or frontend.
The published source license does not establish the rights of any recordings.

## Motivation

The existing `AudioBuffer` one-shot decoder and encoder intentionally
cap materialized PCM at 120 seconds. A future long-form text-to-audio
pipeline needs *file output that doesn't allocate the entire recording*.
This is a **file transport foundation** for longer sequences, not a longer
trained generative model or an actual streaming inference scheduler.

## Protocol

- The developer predeclares sample rate (8–192 kHz), channels (1–8),
  integer PCM bit depth (8/16/24/32) and `maximum_frames`.
- Up to 3,600 seconds and **96 MiB** of RIFF/WAVE including the header
  and padding may be exported, whichever limit is tighter.
- Each immutable `FramePacket` must have a strictly sequential absolute
  start frame, matching rate/channels and at most **65,536 frames**.
- `push()` validates and quantizes one packet before appending bytes
  to the temporary file. Invalid packets leave the frame position unchanged.
- Chunk boundaries do not affect quantization bytes when dither is off.
  Existing TPDF dithering is **not chunk invariant**, so `dither=True`
  is explicitly rejected rather than silently replaying the same RNG
  sequence on each packet. A future stream dither protocol must specify
  its own reproducibility and cross-channel semantics.
- `finish()` requires at least one frame, patches the exact RIFF sizes,
  writes any odd-byte padding, flushes and fsyncs the temporary file,
  verifies its complete SHA256, and **then** atomically publishes it.
- `overwrite=False` uses an atomic hard-link create-if-absent operation;
  races or unsupported filesystems fail closed. `overwrite=True`
  explicitly uses atomic replacement and refuses an observed symlink.
  The directory must be local and developer-controlled.
- `with` scope exit without `finish()` aborts, removes the scratch
  file and never creates the destination.

This protocol is not resumable, distributed, authenticated, real-time,
durable job scheduling or public file upload isolation. Unexpected process
termination can leave `.sonumio-stream-*.partial` temporary files that
must be safely cleaned up by a trusted maintainer.

## Example (no publication until finish)

```python
import numpy as np
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter

spec = WavPacketSpec(sample_rate=24000, channels=1,
                     maximum_frames=24000 * 180)
with WavPacketWriter("outputs/experimental-local.wav", spec,
                     create_parent=True) as writer:
    for position in range(0, 24000 * 180, 24000):
        # Example numerical silence only; no trained audio generation.
        writer.push(FramePacket(
            24000, position, np.zeros(24000, dtype=np.float32)
        ))
    result = writer.finish()
print(result.frames, result.wav_sha256)
```

## Validation

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests/test_audio_core_wave_stream.py
python -m pytest -q tests/test_audio_core_*.py
```

The regression suite covers 48 bit-depth/channel/chunk combinations,
one-shot byte equality including odd-byte padding, aborts and exceptions,
out-of-order packets, clock and size limits, atomic overwrite/no-clobber
races, SHA256 accuracy and >120-second valid standard PCM WAV output.
Only the bounded writer can export this longer file: the existing in-memory
`read_wav_file` intentionally continues rejecting >120-second files.

CI must pass across Linux and Windows before any compatibility claim.
None of the tests prove semantic prompt adherence, generated sound
quality, real-time latency, audio rights or long-form neural inference.

**Keep PR #1 DRAFT and unmerged. No release, pre-release, tag, model
weights or generated recordings are published by this work.**

## Matching canonical WAV packet reader

`WavPacketReader(path, packet_frames=8192, expected_sha256=...)` is a
strict companion to the writer. It streams canonical 44-byte-header
integer PCM, not arbitrary RIFF metadata or compressed audio. It opens a
single regular-file descriptor with available no-follow/nonblocking
protections, checks RIFF sizes and byte rate, validates end padding and
file metadata, and yields immutable channel-first `FramePacket` objects
up to 65,536 frames each.

The source's `verified_sha256` is available **only after the last packet
has passed all validations**. Incomplete consumption is not represented as
successful verification. An expected SHA256 mismatch prevents delivery
of the final packet. Files written by the incremental writer beyond 120
seconds can be read without allocating a 120+ second `AudioBuffer`.

The reader is not an untrusted internet upload parser, source-license
verifier or malicious in-place modification defense; source owners must
protect input directory permissions. The strict canonical-only protocol
is intentionally narrower than the legacy generic one-shot WAVE parser.

```python
from sonumio.audio_core.wave_reader import WavPacketReader

with WavPacketReader("outputs/experimental-local.wav") as reader:
    for packet in reader:
        # Process a bounded packet; do not accumulate all packets in memory.
        print(packet.start_frame, packet.frames)
    print(reader.verified_sha256)  # only valid after complete consumption
```

Validation: `python -m pytest -q tests/test_audio_core_wave_reader.py`.
Run and review the actual Linux and Windows CI results before claiming
cross-platform support.


### Verified checksum versus timestamp races

A Windows CI regression demonstrated that matching file size and mtime/ctime
is insufficient for detecting in-place, same-length source edits. The reader
now opens the descriptor **without Python read-ahead buffering** so that
packet reads use the current file descriptor state rather than a prefetched
full recording. When a known trusted SHA-256 is supplied, the full canonical
RIFF bytes (including header and padding) must match before a reader reports
successful verification or returns the final frame packet. In-place edits on
hostile filesystems or concurrent edits can still race a read; trusted inputs,
source path isolation and explicit reference digests remain essential.

The mutation test now supplies the original export checksum, so it is
independent of platform-specific timestamp granularity.


## Bounded two-pass postprocessing

A new opt-in `wave_pipeline.process_wav_file` adds independent
peak/RMS analysis, explicit fixed gain, optional peak ceiling and
hash-verified file-to-file processing with no full-clip PCM allocation.
Read the full safety and correctness protocol in
[VERIFIED_WAV_PROCESSING.md](VERIFIED_WAV_PROCESSING.md).
