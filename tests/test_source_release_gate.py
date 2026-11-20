"""Fail-closed source release integrity tests against checked-out Git HEAD."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import tomllib

import pytest

from scripts import source_release_gate as gate


def test_candidate_git_archive_matches_every_source_git_blob(tmp_path: Path):
    archive = tmp_path / "source.tar.gz"
    gate._make_archive(archive)
    report = gate._audit_archive(archive)
    assert report["tag_candidate"] == "v0.0.1"
    assert report["source_files"] >= len(gate.REQUIRED)
    assert report["canonical_blob_bytes_verified"] is True
    assert report["canonical_blob_hashes_verified"] is True
    assert report["binary_or_model_artifacts"] == 0
    assert report["published"] is False


def test_archive_refuses_same_length_content_tampering(tmp_path: Path):
    original = tmp_path / "original.tar.gz"
    modified = tmp_path / "modified.tar.gz"
    gate._make_archive(original)
    changed = False
    with tarfile.open(original, mode="r:gz") as reader:
        with tarfile.open(modified, mode="w:gz") as writer:
            for member in reader:
                if member.isfile():
                    stream = reader.extractfile(member)
                    assert stream is not None
                    data = stream.read()
                    if member.name == gate.PREFIX + ".gitignore":
                        assert data
                        data = bytes([data[0] ^ 1]) + data[1:]
                        changed = True
                    writer.addfile(member, io.BytesIO(data))
                else:
                    writer.addfile(member)
    assert changed, "The real source archive must include .gitignore"
    with pytest.raises(RuntimeError, match="Git blob checksum mismatch: .gitignore"):
        gate._audit_archive(modified)


def test_source_gate_matches_web_and_python_software_versions():
    metadata = tomllib.loads(
        (gate.ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    web = json.loads(
        (gate.ROOT / "web" / "package.json").read_text(encoding="utf-8")
    )
    assert metadata["project"]["version"] == gate.VERSION == "0.0.1"
    assert web["version"] == gate.VERSION
    assert web["private"] is True
