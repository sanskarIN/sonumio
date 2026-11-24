"""Fail-closed source release integrity tests against checked-out Git HEAD."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import stat
import zipfile
import tomllib

import pytest

from scripts import source_release_gate as gate


def test_candidate_git_archive_matches_every_source_git_blob(tmp_path: Path):
    archive = tmp_path / "source.tar.gz"
    candidate = gate.resolve_candidate()
    gate._make_archive(archive, candidate)
    report = gate._audit_archive(archive, candidate)
    assert report["tag_candidate"] == candidate.tag
    assert report["commit_candidate"] == candidate.commit
    assert report["software_version"] == candidate.version
    assert report["source_files"] >= len(gate.REQUIRED)
    assert report["canonical_blob_bytes_verified"] is True
    assert report["canonical_blob_hashes_verified"] is True
    assert report["binary_or_model_artifacts"] == 0
    assert report["published"] is False


def test_archive_refuses_same_length_content_tampering(tmp_path: Path):
    original = tmp_path / "original.tar.gz"
    modified = tmp_path / "modified.tar.gz"
    candidate = gate.resolve_candidate()
    gate._make_archive(original, candidate)
    changed = False
    with tarfile.open(original, mode="r:gz") as reader:
        with tarfile.open(modified, mode="w:gz") as writer:
            for member in reader:
                if member.isfile():
                    stream = reader.extractfile(member)
                    assert stream is not None
                    data = stream.read()
                    if member.name == candidate.prefix + ".gitignore":
                        assert data
                        data = bytes([data[0] ^ 1]) + data[1:]
                        changed = True
                    writer.addfile(member, io.BytesIO(data))
                else:
                    writer.addfile(member)
    assert changed, "The real source archive must include .gitignore"
    with pytest.raises(RuntimeError, match="Git blob checksum mismatch: .gitignore"):
        gate._audit_archive(modified, candidate)


def test_source_gate_matches_web_and_python_software_versions():
    metadata = tomllib.loads(
        (gate.ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    web = json.loads(
        (gate.ROOT / "web" / "package.json").read_text(encoding="utf-8")
    )
    candidate = gate.resolve_candidate()
    assert metadata["project"]["version"] == candidate.version
    assert web["version"] == candidate.version
    assert web["private"] is True


@pytest.mark.parametrize("version", ["0.0.1", "0.0.2", "1.2.3", "123456.0.12"])
def test_canonical_numeric_version(version):
    candidate = gate.SourceCandidate("a" * 40, version)
    assert candidate.tag == f"v{version}"
    assert candidate.prefix == f"sonumio-v{version}/"


@pytest.mark.parametrize("version", ["v0.0.2", "0.00.2", "0.0.2-alpha", "0.0.2\n",
                                    "../0.0.2", "1.2", "1.2.3.4", "", None, 2])
def test_invalid_source_version(version):
    with pytest.raises(RuntimeError, match="canonical numeric"):
        gate.SourceCandidate("a" * 40, version)


@pytest.mark.parametrize("commit", ["HEAD", "main", "-n", "a"*39, "g"*40, "A"*40])
def test_candidate_requires_full_commit_id(commit):
    with pytest.raises(RuntimeError, match="pinned Git commit"):
        gate.SourceCandidate(commit, "0.0.2")


def test_expected_version_rejects_mismatch_before_export():
    candidate = gate.resolve_candidate()
    assert gate.resolve_candidate(candidate.version) == candidate
    different = "0.0.2" if candidate.version != "0.0.2" else "0.0.1"
    with pytest.raises(RuntimeError, match="Expected source version"):
        gate.resolve_candidate(different)


def test_candidate_pins_committed_metadata_not_working_tree(monkeypatch, tmp_path):
    import subprocess
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    git("config", "user.name", "Source audit test")
    git("config", "user.email", "test@example.invalid")
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text('[project]\nname = "sonumio"\nversion = "0.0.2"\n', encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "-qm", "candidate")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    original = gate.resolve_candidate("0.0.2")
    metadata.write_text('[project]\nname = "sonumio"\nversion = "0.0.3"\n', encoding="utf-8")
    assert gate.resolve_candidate() == original
    git("add", "pyproject.toml")
    git("commit", "-qm", "next candidate")
    assert gate.resolve_candidate().version == "0.0.3"
    archive = tmp_path / "pinned.tar.gz"
    gate._make_archive(archive, original)
    with tarfile.open(archive, "r:gz") as reader:
        stream = reader.extractfile(original.prefix + "pyproject.toml")
        assert stream is not None
        assert tomllib.loads(stream.read().decode("utf-8"))["project"]["version"] == "0.0.2"


def test_archive_rejects_duplicate_directory_entries(tmp_path):
    candidate = gate.resolve_candidate()
    archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        for name in ("docs", "DOCS"):
            directory = tarfile.TarInfo(candidate.prefix + name)
            directory.type = tarfile.DIRTYPE
            writer.addfile(directory)
    with pytest.raises(RuntimeError, match="case-colliding"):
        gate._audit_archive(archive, candidate)


def test_zip_and_tar_sources_match_same_pinned_tree(tmp_path):
    candidate = gate.resolve_candidate()
    tar = tmp_path / "source.tar.gz"
    zipped = tmp_path / "source.zip"
    gate._make_archive(tar, candidate)
    gate._make_archive(zipped, candidate, format="zip")
    a = gate._audit_archive(tar, candidate)
    b = gate._audit_archive(zipped, candidate, format="zip")
    for key in ("commit_candidate", "source_files", "source_bytes", "software_version"):
        assert a[key] == b[key]
    assert b["archive_format"] == "zip"
    assert b["canonical_blob_hashes_verified"]
    assert a["archive_sha256"] != b["archive_sha256"]


def test_zip_same_size_tampering_detected(tmp_path):
    candidate = gate.resolve_candidate()
    original = tmp_path / "original.zip"
    modified = tmp_path / "modified.zip"
    gate._make_archive(original, candidate, format="zip")
    with zipfile.ZipFile(original) as reader, zipfile.ZipFile(modified, "w") as writer:
        for member in reader.infolist():
            data = reader.read(member)
            if member.filename == candidate.prefix + ".gitignore":
                data = bytes([data[0] ^ 1]) + data[1:]
            writer.writestr(member, data)
    with pytest.raises(RuntimeError, match="Git blob checksum mismatch: .gitignore"):
        gate._audit_archive(modified, candidate, format="zip")


@pytest.mark.parametrize("filename", ["../escape", "a//b", "a/./b", "x:stream", "bad."])
def test_zip_unsafe_noncanonical_paths(tmp_path, filename):
    candidate = gate.resolve_candidate()
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(candidate.prefix + filename, b"x")
    with pytest.raises(RuntimeError, match="Unsafe archive path|Noncanonical source path"):
        gate._audit_archive(archive, candidate, format="zip")


def test_zip_rejects_symlink(tmp_path):
    candidate = gate.resolve_candidate()
    archive = tmp_path / "symlink.zip"
    entry = zipfile.ZipInfo(candidate.prefix + "link")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr(entry, b"target")
    with pytest.raises(RuntimeError, match="non-regular ZIP"):
        gate._audit_archive(archive, candidate, format="zip")


def test_zip_case_collision_and_duplicate_root(tmp_path):
    candidate = gate.resolve_candidate()
    for paths, pattern in (
        (("docs/A", "docs/a"), "case-colliding"),
        (("", ""), "duplicate archive root"),
    ):
        archive = tmp_path / "duplicate.zip"
        with zipfile.ZipFile(archive, "w") as writer:
            writer.writestr(candidate.prefix + paths[0], b"")
            if paths[0] == paths[1]:
                with pytest.warns(UserWarning, match="Duplicate name"):
                    writer.writestr(candidate.prefix + paths[1], b"")
            else:
                writer.writestr(candidate.prefix + paths[1], b"")
        with pytest.raises(RuntimeError, match=pattern):
            gate._audit_archive(archive, candidate, format="zip")


def test_entry_count_bound(monkeypatch, tmp_path):
    candidate = gate.resolve_candidate()
    archive = tmp_path / "bounded.zip"
    monkeypatch.setattr(gate, "MAX_ENTRIES", 2)
    with zipfile.ZipFile(archive, "w") as writer:
        for i in range(3):
            writer.writestr(candidate.prefix + f"dir{i}/", b"")
    with pytest.raises(RuntimeError, match="entry review limits"):
        gate._audit_archive(archive, candidate, format="zip")
