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
    assert report["canonical_directory_tree_verified"] is True
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
    assert b["canonical_directory_tree_verified"]
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


@pytest.mark.parametrize("format", ["tar.gz", "zip"])
@pytest.mark.parametrize("mutation", ["extra", "file_as_parent", "missing_root", "missing_directory"])
def test_archive_directory_topology_matches_committed_tree(tmp_path, format, mutation):
    candidate = gate.resolve_candidate()
    original = tmp_path / ("original." + format)
    changed = tmp_path / ("changed." + format)
    gate._make_archive(original, candidate, format=format)

    def omit(name):
        return ((mutation == "missing_root" and name.rstrip("/") == candidate.prefix.rstrip("/"))
                or (mutation == "missing_directory" and name.rstrip("/") == candidate.prefix + "src"))

    extra = {"extra": "untracked-empty", "file_as_parent": "README.md/child"}.get(mutation)
    if format == "zip":
        with zipfile.ZipFile(original) as reader, zipfile.ZipFile(changed, "w") as writer:
            for member in reader.infolist():
                if not omit(member.filename):
                    writer.writestr(member, reader.read(member))
            if extra is not None:
                writer.writestr(candidate.prefix + extra + "/", b"")
    else:
        with tarfile.open(original, "r:gz") as reader, tarfile.open(changed, "w:gz") as writer:
            for member in reader:
                if not omit(member.name):
                    writer.addfile(member, reader.extractfile(member) if member.isfile() else None)
            if extra is not None:
                member = tarfile.TarInfo(candidate.prefix + extra)
                member.type = tarfile.DIRTYPE
                writer.addfile(member)
    with pytest.raises(RuntimeError, match="directories differ from the committed tree"):
        gate._audit_archive(changed, candidate, format=format)


def test_installed_spectrum_smoke_checks_independent_pcm_energy(tmp_path):
    import struct
    import wave
    destination = tmp_path / "sample.wav"
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(struct.pack("<h", 8192) * 12000)
    report = {
        "source_verified": True, "source_sha256": "a" * 64,
        "spectrum": {
            "frames": 12000, "fft_frames": 256, "channels": 1,
            "complete_blocks": 46, "padded_tail_frames": 224,
            "channel_spectra": [{
                "mean_square": 0.0625,
                "bin_mean_square": [0.0625] + [0.0] * 128,
            }],
        },
    }
    gate._check_spectrum_smoke(report, "a" * 64, destination)
    report["spectrum"]["channel_spectra"][0]["bin_mean_square"][0] = 0.125
    with pytest.raises(RuntimeError, match="independent PCM energy"):
        gate._check_spectrum_smoke(report, "a" * 64, destination)


def test_installed_demo_smoke_rejects_false_model_provenance(tmp_path):
    import hashlib
    import wave
    destination = tmp_path / "demo.wav"
    with wave.open(str(destination), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\x00\x00" * 288000)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    report = {
        "software_version": "0.0.2", "engine_id": "procedural-sfx-v0",
        "seed": 42, "duration_seconds": 12.0,
        "original_model_inference": False, "third_party_recordings_used": False,
        "wav_sha256": digest, "bytes_written": 576044,
        "inspection": {
            "source_sha256": digest, "source_verified": True,
            "signal_health": {"clipping_fraction": 0},
        },
    }
    gate._check_demo_smoke(report, destination, "0.0.2")
    report["original_model_inference"] = True
    with pytest.raises(RuntimeError, match="demo provenance"):
        gate._check_demo_smoke(report, destination, "0.0.2")


@pytest.mark.parametrize("filename", [
    "research/private.scorpus", "research/private.SCORPUS",
    "research/state.stcheckpoint", "research/state.STCHECKPOINT",
    "research/.sonumio-train-abc.partial",
])
def test_source_gate_refuses_private_training_artifacts(filename):
    candidate = gate.SourceCandidate("a"*40, "0.0.2")
    entries = gate.SourceEntries(candidate)
    def forbidden():
        pytest.fail("private artifact should be rejected before its contents are read")
    with pytest.raises(RuntimeError, match="Binary or sensitive artifact"):
        entries.accept(candidate.prefix+filename, directory=False, regular=True,
                       size=1, read=forbidden)


@pytest.mark.parametrize("payload", [
    b"SCORPUS1"+bytes(32), b"STCP0001"+bytes(32),
    b"SLHAND01"+bytes(32), b"SLCROP01"+bytes(32),
    b"\x93NUMPY"+bytes(32), b"PK\x03\x04"+bytes(32),
    b"PK\x05\x06"+bytes(18), b"PK\x07\x08"+bytes(12),
    b"RIFF"+bytes(4)+b"WAVE"+bytes(32),
    b"RF64"+bytes(4)+b"WAVE"+bytes(32),
    b"RIFX"+bytes(4)+b"WAVE"+bytes(32),
    b"fLaC"+bytes(32), b"OggS"+bytes(32),
])
@pytest.mark.parametrize("format", ["tar.gz","zip"])
def test_disguised_artifact_detected_before_tree_comparison(tmp_path, payload, format):
    candidate = gate.resolve_candidate()
    archive = tmp_path / ("disguised." + format)
    name = candidate.prefix + "research/ordinary.txt"
    if format == "zip":
        with zipfile.ZipFile(archive,"w") as writer:
            writer.writestr(name,payload)
    else:
        with tarfile.open(archive,"w:gz") as writer:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            writer.addfile(member,io.BytesIO(payload))
    with pytest.raises(RuntimeError,match="Binary artifact signature"):
        gate._audit_archive(archive,candidate,format=format)


def test_format_documentation_is_not_rejected():
    candidate = gate.SourceCandidate("a"*40,"0.0.4")
    entries = gate.SourceEntries(candidate)
    source = b'# Private archive format\nMAGIC = b"SCORPUS1"\n'
    entries.accept(candidate.prefix+"docs/format.md",directory=False,regular=True,
                   size=len(source),read=lambda:source)
    assert entries.files["docs/format.md"] == source


def test_sound_design_pcm_receipt_uses_actual_bytes(tmp_path):
    import hashlib
    import wave
    target = tmp_path / "contact.wav"
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(b"\x01\x00\x02\x00" * 20)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    report = {"output_sha256": digest}
    assert gate._check_generated_pcm(report, target, (8000, 2, 2, 20)) == digest
    with pytest.raises(RuntimeError, match="geometry"):
        gate._check_generated_pcm(report, target, (8000, 1, 2, 20))
    with pytest.raises(RuntimeError, match="digest"):
        gate._check_generated_pcm({"output_sha256": "0"*64}, target, (8000, 2, 2, 20))


def test_pack_smoke_cross_checks_actual_archive_and_selection(tmp_path):
    from copy import deepcopy
    from sonumio.audio_core.asset_pack import AssetRecipe, SoundPackPlan
    from sonumio.audio_core.asset_archive import create_pack_bytes, verify_pack_bytes
    from sonumio.audio_core.asset_selection import select_asset
    raw, receipt = create_pack_bytes(SoundPackPlan(
        "smoke", (AssetRecipe("rain", 2400, variants=2),)))
    archive = tmp_path / "pack.zip"
    archive.write_bytes(raw)
    selection = select_asset(raw, "rain", 2)
    target = tmp_path / "selected.wav"
    target.write_bytes(selection.wav)
    created = json.loads(json.dumps(receipt.as_dict()))
    verified = json.loads(json.dumps(verify_pack_bytes(raw, replay=True).as_dict()))
    selected = {**selection.as_dict(), "output_sha256": selection.record["wav_sha256"]}
    assert gate._check_pack_smoke(created, verified, selected, archive, target) == selected["output_sha256"]
    wrong = deepcopy(verified)
    wrong["replayed_assets"] = 1
    with pytest.raises(RuntimeError, match="verification"):
        gate._check_pack_smoke(created, wrong, selected, archive, target)
    target.write_bytes(select_asset(raw, "rain", 1).wav)
    with pytest.raises(RuntimeError, match="differs from archive"):
        gate._check_pack_smoke(created, verified, selected, archive, target)
