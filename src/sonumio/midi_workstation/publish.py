"""Atomic, output-checksum-pinned publication of user-requested local artifacts.

The API never auto-publishes to GitHub or other services. It writes only to
explicit caller-provided local file destinations. Existing files require
explicit overwrite, with stale-content protection available by SHA-256.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256 as _sha256
import os
from pathlib import Path
import secrets
import stat

from .contracts import (
    MAX_MIDI_BYTES, MAX_JSON_BYTES, MidiFormatError,
    MidiIntegrityError, MidiLimitError, MidiWorkstationError,
    integer, raw_bytes, sha256,
)

MAX_PUBLICATION_BYTES = max(MAX_MIDI_BYTES, MAX_JSON_BYTES)


@dataclass(frozen=True, slots=True)
class PublicationOptions:
    overwrite: bool = False
    create_parent: bool = False
    expected_sha256: str | None = None
    expected_previous_sha256: str | None = None

    def __post_init__(self):
        if type(self.overwrite) is not bool or type(self.create_parent) is not bool:
            raise MidiWorkstationError("Publication flags must be boolean")
        if self.expected_sha256 is not None:
            sha256(self.expected_sha256, "expected output digest")
        if self.expected_previous_sha256 is not None:
            sha256(self.expected_previous_sha256, "existing output digest")
            if not self.overwrite:
                raise MidiWorkstationError("Previous digest requires explicit overwrite")


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    output_sha256: str
    bytes_written: int
    previous_sha256: str | None
    overwrote: bool
    protocol: str = "sonumio-workstation-atomic-file-v1"

    def as_dict(self):
        from dataclasses import asdict
        result = asdict(self)
        result["remote_publication"] = False
        result["model_inference"] = False
        return result


def _distinct(input_path: str | Path | None, output_path: Path):
    if input_path is None:
        return
    source = Path(input_path)
    if source.resolve() == output_path.resolve():
        raise MidiFormatError("Input and output paths must be different")
    if source.exists() and output_path.exists():
        try:
            if os.path.samefile(source, output_path):
                raise MidiFormatError("Input and output refer to the same file")
        except FileNotFoundError:
            pass


def _check_target(path: Path, options: PublicationOptions) -> str | None:
    if path.is_symlink():
        raise MidiFormatError("Refusing to write through output symlink")
    if not path.exists():
        if options.expected_previous_sha256 is not None:
            raise MidiIntegrityError("Expected existing destination is missing")
        return None
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise MidiFormatError("Existing output is not a regular file")
    if not options.overwrite:
        raise MidiWorkstationError("Refusing to overwrite existing output without approval")
    if options.expected_previous_sha256 is None:
        return None
    if metadata.st_size > MAX_PUBLICATION_BYTES:
        raise MidiLimitError("Existing output exceeds digest verification budget")
    with path.open("rb") as file:
        old = file.read(MAX_PUBLICATION_BYTES + 1)
    if len(old) > MAX_PUBLICATION_BYTES:
        raise MidiLimitError("Existing output grew during verification")
    actual = _sha256(old).hexdigest()
    if actual != options.expected_previous_sha256:
        raise MidiIntegrityError("Existing output digest mismatch; refusing to overwrite")
    return actual


def publish_bytes(destination: str | Path, content: bytes,
                  options: PublicationOptions = PublicationOptions(), *,
                  input_path: str | Path | None = None,
                  expected_extension: str | None = None) -> PublicationReceipt:
    if not isinstance(options, PublicationOptions):
        raise MidiWorkstationError("Expected PublicationOptions")
    raw_bytes(content, "publication content", MAX_PUBLICATION_BYTES)
    if not content:
        raise MidiFormatError("Refusing to publish empty artifact")
    output = Path(destination)
    if expected_extension is not None and (
        type(expected_extension) is not str or
        output.suffix.lower() != expected_extension.lower()
    ):
        raise MidiFormatError("Output file extension does not match requested artifact")
    _distinct(input_path, output)
    target_digest = _sha256(content).hexdigest()
    if options.expected_sha256 is not None and options.expected_sha256 != target_digest:
        raise MidiIntegrityError("Generated artifact digest differs from pinned output")
    parent = output.parent
    if not parent.is_dir():
        if not options.create_parent:
            raise MidiFormatError("Destination directory is missing")
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise MidiFormatError("Destination directory is not a directory")
    previous = _check_target(output, options)
    # Temporary file stays in the target directory so replacing it is atomic
    # on a normal local filesystem. Never use an unchecked temp directory.
    staging = parent / f".{output.name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = None
    try:
        fd = os.open(staging, flags, 0o600)
        with os.fdopen(fd, "wb", closefd=True) as stream:
            fd = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        with staging.open("rb") as verify:
            saved = verify.read(MAX_PUBLICATION_BYTES + 1)
        if _sha256(saved).hexdigest() != target_digest:
            raise MidiIntegrityError("Staged output failed digest validation")
        _distinct(input_path, output)
        _check_target(output, options)
        if options.overwrite:
            os.replace(staging, output)
        else:
            # Hardlink is the no-clobber commit primitive: if the name
            # already exists, it fails without replacing the other file.
            os.link(staging, output)
            staging.unlink()
        try:
            parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            # Some Windows filesystems cannot fsync directories; publication
            # remains successful but crash durability is platform-dependent.
            pass
        return PublicationReceipt(target_digest, len(content), previous,
                                  options.overwrite)
    finally:
        if fd is not None:
            os.close(fd)
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass


def publish_midi(path: str | Path, midi, options: PublicationOptions = PublicationOptions(),
                 *, input_path: str | Path | None = None) -> PublicationReceipt:
    from .smf import encode_smf
    data = encode_smf(midi)
    return publish_bytes(path, data, options, input_path=input_path,
                         expected_extension=".mid")


def publish_score(path: str | Path, score, options: PublicationOptions = PublicationOptions(),
                  *, input_path: str | Path | None = None) -> PublicationReceipt:
    from .score_io import encode_score
    data = encode_score(score)
    return publish_bytes(path, data, options, input_path=input_path,
                         expected_extension=".json")
