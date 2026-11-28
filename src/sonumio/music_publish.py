"""Atomic local publication of bounded score JSON, MIDI and stem packages.

Each command publishes one file, not a multi-file transaction. This
local filesystem helper is not a hostile multi-user sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

from .atomic_cli_wav import _ensure_safe_parent, _check_existing_target
from .music_score_io import MAX_SCORE_BYTES
from .music_midi import MAX_MIDI_BYTES
from .errors import GenerationError

MAX_STEM_PACK_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True,slots=True)
class ArtifactReceipt:
    path: Path
    bytes_written: int
    sha256: str


def publish_music_artifact(path,data,*,kind,overwrite=False,expected_sha256=None):
    formats = {"score": (MAX_SCORE_BYTES, ".json"),
               "midi": (MAX_MIDI_BYTES, ".mid"),
               "stems": (MAX_STEM_PACK_BYTES, ".zip")}
    if type(kind) is not str or kind not in formats or type(overwrite) is not bool:
        raise GenerationError("Invalid music artifact publication mode")
    limit, suffix = formats[kind]
    if type(data) is not bytes or not 1<=len(data)<=limit:
        raise GenerationError("Invalid bounded music artifact bytes")
    path=Path(path)
    if path.suffix.lower()!=suffix:
        raise GenerationError("Music artifact requires "+suffix+" extension")
    if expected_sha256 is not None:
        if (type(expected_sha256) is not str or len(expected_sha256)!=64
                or any(c not in "0123456789abcdef" for c in expected_sha256)):
            raise GenerationError("Expected lowercase SHA256")
    digest=hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest!=expected_sha256:
        raise GenerationError("Music artifact digest mismatch before publication")
    _ensure_safe_parent(path,create_parent=True)
    _check_existing_target(path,overwrite=overwrite)
    stage=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,mode="w+b",prefix=".sonumio-music-",
                                         suffix=".partial",delete=False) as stream:
            stage=Path(stream.name)
            if stream.write(data)!=len(data):
                raise OSError("Short music artifact write")
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            staged_hash = hashlib.sha256()
            staged_bytes = 0
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                staged_bytes += len(block)
                if staged_bytes > limit:
                    raise GenerationError("Music artifact staging exceeds byte limit")
                staged_hash.update(block)
            if staged_bytes != len(data) or staged_hash.hexdigest() != digest:
                raise GenerationError("Music artifact staging verification failed")
        _ensure_safe_parent(path,create_parent=False)
        _check_existing_target(path,overwrite=overwrite)
        if overwrite:
            os.replace(stage,path)
            stage=None
        else:
            os.link(stage,path)
        return ArtifactReceipt(path,len(data),digest)
    finally:
        if stage is not None:
            stage.unlink(missing_ok=True)
