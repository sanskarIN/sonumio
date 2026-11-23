# Sonumio local CLI: atomic generated WAV publication

Sonumio's local `generate` and `scene` commands produce an in-memory
RIFF/WAVE file. Previously they used direct `Path.write_bytes` output.
That could truncate an existing recording before a full new output was
ready; failed writes could leave partial WAV data behind.

The CLI now shares `src/sonumio/atomic_cli_wav.py`, a bounded local-only
publisher with explicit destination safety and full-file verification.

## Usage

```bash
sonumio generate "rain and wind" --seconds 5 --seed 17 \
    --output outputs/rain.wav

sonumio scene examples/forest_storm_scene.json \
    --output outputs/forest.wav
```

Both commands create missing parent directories as before. If a WAV already
exists, a second call refuses to overwrite it. To intentionally replace an
ordinary existing file, use `--overwrite`:

```bash
sonumio generate "rain" --seconds 5 \
    --output outputs/rain.wav --overwrite
```

A caller with a trusted full-output digest can require it **before**
creating any output or parent directory:

```bash
sonumio generate "rain" --seed 17 \
    --expected-output-sha256 <64-character lowercase SHA256> \
    --output outputs/known.wav
```

This digest is the complete WAV file bytes, including the RIFF header,
not the raw float samples or audio payload alone. The command now prints
the result's SHA256 after success. A mismatch fails without publishing.

## How atomic publication works

1. Reject non-WAV destinations, unsafe/nonregular target files,
   symlinked parent components, wrong hash formats and malformed WAV.
2. Prevalidate the entire **canonical PCM RIFF/WAVE** byte structure:
   exact `RIFF`, `fmt `, `data` chunks, sample clock, channel, bit depth,
   frame geometry, file sizes and RIFF padding.
3. Enforce an explicit **64 MiB** maximum for one locally generated WAV.
4. Create a private temporary file in the destination's own directory.
5. Write all bytes, flush, `fsync`, and independently **rehash the staged
   bytes** and compare them against the input SHA256.
6. Revalidate the parent and destination immediately before commit.
7. Without `--overwrite`, `os.link` publishes the completed file if and
   only if the destination does not exist. A concurrent creation rejects
   the publication rather than overwriting it.
8. With `--overwrite`, `os.replace` atomically switches the pathname
   only after the completed new bytes are flushed and validated.
9. Remove the temporary file on all paths, including failures.

An existing filename that is a symbolic link, directory, or other special
file is rejected even with `--overwrite`. Replacing one hardlink name via
`os.replace` does not change the other names of its original inode.

The design is for a **trusted local user's filesystem**. It does not
implement directory file-descriptor traversal against hostile
simultaneous path-component replacement, a transactional multi-file
database, or cross-process locking.

## Behavioral compatibility

- The procedural engine and waveform/audio checksum output are unchanged.
- The API's private ephemeral preview storage remains separate.
- `wav-process` already uses its own two-pass, atomic output writer and
  retains its existing behavior.
- Local `generate` and `scene` retain existing command arguments and
  implicit parent-directory creation, but overwriting now requires the
  new explicit `--overwrite` flag.
- Only complete canonical PCM `.wav` files can be emitted through this
  publisher. WAV extensions other than `.wav` are not accepted.

## Regression gates

```bash
python -m pytest -q tests/test_atomic_cli_wav.py
python -m pytest -q tests/test_scenes.py tests/test_audio.py \
    tests/test_audio_core_cli_wave_process.py
```

The dedicated GitHub Actions workflow tests Ubuntu Python 3.11/3.13
and Windows Python 3.12. Tests include simulated failed `fsync`,
failed `os.replace`, competing no-clobber writes, checksum mismatches,
truncated/forged WAV headers, nested destinations, symlinks, hardlinks,
actual CLI generation and scene output.

This is source-code transport hardening, **not a new trained model,
a Sonumio-1 release, a neural generation claim, or model-weight
publication**.
