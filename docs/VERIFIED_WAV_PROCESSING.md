# Verified streaming WAV postprocessing

Status: **experimental local audio-core research infrastructure**.
This module is not a neural audio generation engine, voice synthesis tool,
speaker recognition system, or copyright/consent verifier.

## Problem and trust model

For recordings longer than Sonumio's 120-second in-memory `AudioBuffer` cap,
a developer needs bounded memory, deterministic gain-staging and integrity
verification **before output publication**.

`process_wav_file()` performs two sequential passes over a strict canonical
integer PCM RIFF/WAV file from `WavPacketReader`:

1. Read bounded packets, compute the global PCM peak, RMS and full SHA256.
   A caller-supplied `expected_source_sha256` must match, if given.
2. Compute one fixed gain from requested dB and optional absolute peak
   ceiling. The default ceiling option only reduces peaks; `allow_peak_boost`
   permits up to an additional +12 dB gain toward the ceiling.
3. Read the source again, **expecting exactly the first pass SHA256**.
   Apply the same gain to all channels and packets, without hidden
   resampling or channel layout changes.
4. Default `clip_policy="reject"` fails closed if any prequantized sample
   leaves [-1, 1]. `"hard"` explicitly permits clipping and reports its
   sample count. These are sample peaks, NOT true peaks or LUFS.
5. Only after the second full source checksum verifies does
   `WavPacketWriter.finish()` publish the output atomically. Any rejection
   removes the scratch file. Source and destination cannot be the same
   path or hard-link alias.

Output is standard 8/16/24/32-bit integer PCM. Without `output_bits`,
source bit depth is retained. With the default `dither_seed=None`, output
remains undithered and retains the prior deterministic byte semantics.
An **explicit** integer `dither_seed` from 0 to 2^32-1 activates the
new stream-specific PCG64 frame-major TPDF protocol, whose RNG state
persists across packets. It is bit-for-bit invariant across packet size
and input chunking, and deliberately *differs* from legacy one-shot
`quantize_pcm(dither=True)` noise ordering. The result records whether
dither was applied, seed and versioned protocol. 32-bit integer source
samples decoded into float32 may lose low-order bits; an identity
pass is therefore not promised to be byte-exact for 32-bit PCM.

Peak normalization here is a fixed full-clip gain coefficient, **not**
dynamic look-ahead limiting, mastering, true-peak control, integrated
loudness measurement, noise removal or neural audio improvement.

`WavProcessingResult` records input/output hashes, file geometry, gain
applied, sample-peak and RMS statistics and any clipping, with
`original_model_inference=False`.

## Local example

```python
from sonumio.audio_core.wave_pipeline import (
    WavProcessingConfig, process_wav_file,
)

result = process_wav_file(
    "approved-input.wav",
    "local-output.wav",
    config=WavProcessingConfig(
        gain_db=3, peak_ceiling_dbfs=-1, packet_frames=8192,
    ),
    expected_source_sha256="PUT_TRUSTED_LOWERCASE_64_CHAR_SHA256_HERE",
)
print(result.source_sha256, result.output_sha256)
print(result.source_peak, result.output_peak_before_quantization)
```

## Regression coverage

```bash
python -m pytest -q tests/test_audio_core_pipeline_fades.py
python -m pytest -q tests/test_audio_core_*.py
```

Tests include 27 identity/chunk invariance combinations (8/16/24-bit,
mono/stereo/four-channel), explicit peak cap, optional bounded boost,
rejected or permitted clipping, source hash mismatch, between-pass file
mutation, same-path and inode aliases, existing-target no-clobber, and
**121-second** end-to-end WAV processing. Results are only considered
cross-platform verified after the corresponding Linux and Windows CI jobs
finish successfully.

The writer outputs its scratch file in a developer-controlled directory.
The protocol is not a sandbox for hostile paths, concurrent in-place
tampering, public uploads or a rights-approval mechanism. Maintaining a
trusted source hash and controlling filesystem permissions remain required.

**PR #1 stays draft/unmerged. No release, pre-release, tag, model
weights or public WAV artifacts are authorized or produced.**


## Explicit local CLI

For a developer-controlled, rights-approved canonical PCM WAV, the
`sonumio wav-process` CLI offers two-pass source SHA256 validation, fixed
gain, an optional peak ceiling and safe atomic output creation:

```bash
sonumio wav-process approved-input.wav \
  --output local-processed.wav \
  --expected-sha256 PUT_TRUSTED_64_CHAR_SHA256_HERE \
  --gain-db 3 --peak-ceiling-dbfs -1 --dither-seed 42
```

The CLI does **not** create parent directories, overwrite existing output
or hard-clip samples by default. `--create-parent`, `--overwrite` and
`--clip-policy hard` are independent explicit opt-ins. The command prints
one JSON result with measured source/output checksums and PCM statistics,
or returns status 2 with an error without publishing partial WAV files.

Do not assume a source is licensed merely because its SHA256 matches. This
is not a public API or authenticated upload workflow.


## Streamed TPDF postprocessing

With `WavProcessingConfig(dither_seed=42)` (or the CLI
`--dither-seed 42`), the second pass uses precisely the versioned
`pcg64-frame-major-interleaved-pairs-v1` generator to dither *only* final
integer PCM conversion, never the first-pass peak/RMS statistics.
A different valid `packet_frames` changes performance but not output bytes
for a fixed approved source, seed, output bit depth and gain policy.
Source verification, refusal to clip without permission and atomic final
publication are unchanged. This is optional quantization dither, **not**
randomness for diffusion sampling, voice transformation or trained models.

## Opt-in frame-exact edge fades (19 November 2026)

`wav-process` can now apply independent, **opt-in linear amplitude fades**
with `--fade-in-frames N` and/or `--fade-out-frames N`. Both default to
zero, preserving all historical output bytes and SHA256 values when omitted.
These are **sample-frame counts at the original WAV clock**, not milliseconds:
no automatic rounding or resampling is introduced. Each value must be a
nonnegative integer of at most 1,920,000 frames (10 seconds at 192 kHz) and
must not exceed the actual retained output length. Excessive fades fail
**before destination directory or scratch output creation**.

Example for a local 24-kHz source, trimming below -40 dBFS and then fading
its retained beginning/end over exactly 240 samples (10 ms):

```bash
sonumio wav-process approved-input.wav --output edited.wav \\
  --trim-silence-dbfs -40 --fade-in-frames 240 --fade-out-frames 240 \\
  --expected-sha256 PUT_TRUSTED_64_CHAR_SHA256_HERE
```

The envelope starts at zero on the first faded frame and reaches unity on
its last fade-in frame; fade-out does the inverse. A one-frame fade mutes
only its endpoint. If fades overlap, their amplitudes multiply. For an
intermediate output frame `i` in a file of `T` retained frames, each
`N > 1` fade contributes the linear factors
`min(i/(N-1), 1)` and/or `min((T-1-i)/(N-1), 1)`.

Fades operate **after optional full-source per-channel DC removal and before
fixed gain, clipping checks, quantization and optional seeded TPDF dither**.
The gain/ceiling estimate remains conservative because its global peak is
computed before fades. Importantly, all discarded source frames are still
read and SHA256-verified in both passes. Fade positioning is anchored to
**retained output frame coordinates** rather than packet starts, making the
bytes deterministic across reader packet sizes within the same numerical
runtime. The JSON result includes `fade_in_frames` and `fade_out_frames`.

A linear fade can soften some hard cuts but does **not** guarantee click-free
or transparent editing, psychoacoustic mastering, LUFS compliance, speaker
consent, or improved generative model quality. Source files and published
model weights are unchanged. PR #1 stays draft and unmerged, with no
release, tag, pretrained weights or generated audio publication.
