"""Atomic, checksum-verified publication of generated local CLI WAV files.

Normal 'generate' and 'scene' commands already hold a complete short WAV
in memory. Never publish it with Path.write_bytes(): a short write, failed
fsync or process interruption can truncate an existing preview and leave
a partial output. This module stages complete, verified PCM in the
destination directory, then publishes it with one atomic filesystem
operation. No network, models, user recordings or background work.

Only a trusted local user can select the output path. This is not a
hostile multi-process filesystem sandbox; there is no process-wide
lock protecting an arbitrary shared directory from hostile renames.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tempfile

from .audio_core.errors import (
    AudioFormatError, AudioIntegrityError, AudioLimitError,
)
from .audio_core.hashing import strict_sha256
from .audio_core.wave_reader import _decode_canonical_header

MAX_GENERATED_WAV_BYTES = 64 * 1024 * 1024
_STAGE_PREFIX = ".sonumio-cli-"
_STAGE_SUFFIX = ".partial"


@dataclass(frozen=True, slots=True)
class PublishedWav:
    path: Path
    bytes_written: int
    wav_sha256: str


def _validate_wav(audio: object) -> str:
    """Validate complete canonical integer PCM bytes before creating dirs."""
    if not isinstance(audio, bytes):
        raise AudioFormatError("Generated output must contain immutable WAV bytes")
    if not 44 <= len(audio) <= MAX_GENERATED_WAV_BYTES:
        raise AudioLimitError("Generated WAV is empty or exceeds local byte limit")
    info = _decode_canonical_header(audio[:44], len(audio))
    if info.data_bytes & 1 and audio[-1] != 0:
        raise AudioFormatError("Generated WAV has invalid RIFF padding")
    return hashlib.sha256(audio).hexdigest()


def _check_existing_target(target: Path, *, overwrite: bool) -> None:
    """Never follow or replace a leaf symlink or special filesystem entry."""
    try:
        entry = target.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(entry.st_mode):
        raise AudioFormatError("Generated WAV destination is not an ordinary file")
    if not overwrite:
        raise AudioFormatError(
            "Existing WAV requires explicit --overwrite"
        )


def _ensure_safe_parent(destination: Path, *, create_parent: bool) -> None:
    """Reject existing symlinked directory components for local CLI exports."""
    parent = Path(os.path.abspath(destination.parent))
    components = (parent, *parent.parents)
    for path in components:
        if path.is_symlink():
            raise AudioFormatError(
                "Generated WAV parent directory must not be a symlink"
            )
    if create_parent:
        parent.mkdir(parents=True, exist_ok=True)
    for path in components:
        if path.is_symlink():
            raise AudioFormatError(
                "Generated WAV parent directory changed to a symlink"
            )
    if not parent.is_dir():
        raise AudioFormatError("Generated WAV output parent must be a directory")


def publish_generated_wav(
    destination: str | Path,
    wav_bytes: bytes,
    *,
    overwrite: bool = False,
    create_parent: bool = True,
    expected_sha256: str | None = None,
) -> PublishedWav:
    """Durably stage and atomically publish a locally generated PCM WAV.

    No-clobber uses same-directory os.link(source, destination): it fails
    atomically if any destination already exists, even one created by a
    concurrent writer after validation. Explicit overwrite uses os.replace,
    which replaces a file rather than opening and truncating it.

    Validate WAV and optional trusted SHA256 before creating parent dirs;
    fsync and rehash the staged bytes before publishing. Failure leaves
    old contents intact and removes the partial staging file.
    """
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise ValueError("WAV publication overwrite/create_parent must be boolean")
    if expected_sha256 is not None:
        strict_sha256(expected_sha256)
    target = Path(destination)
    if (target.name in ("", ".", "..")
            or target.suffix.lower() != ".wav"):
        raise AudioFormatError("Generated WAV output must end in .wav")
    actual_sha256 = _validate_wav(wav_bytes)
    if expected_sha256 is not None and expected_sha256 != actual_sha256:
        raise AudioIntegrityError(
            "Generated WAV output SHA256 mismatch before publication"
        )

    _ensure_safe_parent(target, create_parent=create_parent)
    _check_existing_target(target, overwrite=overwrite)

    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=target.parent, prefix=_STAGE_PREFIX,
            suffix=_STAGE_SUFFIX, delete=False,
        ) as stream:
            staged = Path(stream.name)
            count = stream.write(wav_bytes)
            if count != len(wav_bytes):
                raise OSError("Short staged WAV output write")
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
            if size != len(wav_bytes) or digest.hexdigest() != actual_sha256:
                raise AudioIntegrityError(
                    "Staged generated WAV output failed SHA256 verification"
                )

        # A non-hostile concurrent application may have published this path
        # since the early existence check. Validate it once more and then
        # use the atomic operation as the final collision arbiter.
        _ensure_safe_parent(target, create_parent=False)
        _check_existing_target(target, overwrite=overwrite)
        if overwrite:
            os.replace(staged, target)
            staged = None
        else:
            try:
                os.link(staged, target)
            except FileExistsError as exc:
                raise AudioFormatError(
                    "Existing WAV requires explicit --overwrite"
                ) from exc
            except OSError as exc:
                raise AudioFormatError(
                    "Atomic no-clobber WAV output unsupported on filesystem"
                ) from exc
        return PublishedWav(
            path=target, bytes_written=len(wav_bytes),
            wav_sha256=actual_sha256,
        )
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
