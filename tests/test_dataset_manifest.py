import json
import pytest
from research.dataset_manifest import ManifestError, validate_manifest

def row(**changes):
    r = {
        "asset_id": "sample-a", "source_url": "https://example.org/recording-a",
        "sha256": "a" * 64, "caption": "light rain", "language": "en",
        "group_id": "recording-session-1", "split": "train",
        "license": "CC0-1.0", "source_kind": "cc0",
        "rights_reviewed": True, "creator_permission": False,
        "voice_present": False, "sample_rate": 24000, "duration_seconds": 3.0,
    }
    return {**r, **changes}

def write(tmp_path, rows):
    path = tmp_path / "dataset.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path

def test_valid_manifest_and_counts(tmp_path):
    report = validate_manifest(write(tmp_path, [
        row(),
        row(asset_id="sample-b", sha256="b" * 64,
            group_id="recording-session-2", split="test"),
    ]))
    assert report["records"] == 2
    assert report["split_counts"] == {"test": 1, "train": 1}
    assert len(report["manifest_fingerprint"]) == 64

@pytest.mark.parametrize("changes", [
    {"rights_reviewed": False}, {"voice_present": True},
    {"source_kind": "cc0", "license": "unknown"},
    {"source_kind": "direct-permission", "creator_permission": False},
    {"sha256": "xx"}, {"sample_rate": 10},
])
def test_provenance_rules_block_invalid_records(tmp_path, changes):
    with pytest.raises(ManifestError):
        validate_manifest(write(tmp_path, [row(**changes)]))

def test_group_leakage_and_duplicate_checks(tmp_path):
    with pytest.raises(ManifestError, match="leaks"):
        validate_manifest(write(tmp_path, [
            row(), row(asset_id="sample-b", sha256="b" * 64, split="test"),
        ]))
    with pytest.raises(ManifestError, match="duplicate"):
        validate_manifest(write(tmp_path, [
            row(), row(asset_id="sample-b", sha256="a" * 64),
        ]))


@pytest.mark.parametrize("changes", [
    {"duration_seconds": float("nan")}, {"duration_seconds": float("inf")},
    {"duration_seconds": -float("inf")}, {"duration_seconds": True},
    {"split": {"unexpected": "object"}}, {"language": ["en"]},
    {"source_kind": ["cc0"]}, {"caption": "a" * 801},
])
def test_reject_nonfinite_or_wrong_typed_manifest_metadata(tmp_path, changes):
    with pytest.raises(ManifestError):
        validate_manifest(write(tmp_path, [row(**changes)]))


def test_reject_duplicate_json_keys_and_keep_original_line_numbers(tmp_path):
    invalid = tmp_path / "duplicate.jsonl"
    valid = json.dumps(row())
    invalid.write_text("\n" + valid[:-1] + ', "split": "test"}\n', encoding="utf-8")
    with pytest.raises(ManifestError, match="line 2: invalid or duplicate-key JSON"):
        validate_manifest(invalid)


def test_manifest_snapshot_fingerprint_binds_exact_file_bytes(tmp_path):
    from hashlib import sha256
    from research.dataset_manifest import load_manifest_snapshot

    path = write(tmp_path, [row()])
    records, report = load_manifest_snapshot(path)
    assert records[0][0] == 1
    assert records[0][1]["caption"] == "light rain"
    assert report["manifest_fingerprint"] == sha256(path.read_bytes()).hexdigest()
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    assert validate_manifest(path)["manifest_fingerprint"] != report["manifest_fingerprint"]


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_optional_metadata_is_rejected(tmp_path, bad):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row())[:-1] + ', "extra": ' + bad + '}\n',
                    encoding="utf-8")
    with pytest.raises(ManifestError, match="invalid or duplicate-key JSON"):
        validate_manifest(path)


def test_lone_surrogate_metadata_is_rejected(tmp_path):
    path = write(tmp_path, [row(extra="\ud800")])
    with pytest.raises(ManifestError):
        validate_manifest(path)


def test_manifest_byte_cap_before_read(tmp_path, monkeypatch):
    from research import dataset_manifest as dm
    path = write(tmp_path, [row()])
    monkeypatch.setattr(dm, "MAX_MANIFEST_BYTES", 8)
    with pytest.raises(ManifestError, match="byte cap"):
        validate_manifest(path)


def test_manifest_rejects_oversized_whitespace_line(tmp_path, monkeypatch):
    from research import dataset_manifest as dm
    path = tmp_path / "manifest.jsonl"
    path.write_bytes(b" " * 100 + b"\n" + json.dumps(row()).encode())
    monkeypatch.setattr(dm, "MAX_LINE_BYTES", 80)
    with pytest.raises(ManifestError, match="line 1"):
        validate_manifest(path)


def test_manifest_rejects_directory_and_leaf_symlink(tmp_path):
    with pytest.raises(ManifestError, match="ordinary"):
        validate_manifest(tmp_path)
    path = write(tmp_path, [row()])
    link = tmp_path / "alias.jsonl"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ManifestError, match="ordinary"):
        validate_manifest(link)


def test_snapshot_handles_short_reads(tmp_path, monkeypatch):
    from research import local_snapshot as snapshot
    source = tmp_path / "input"
    source.write_bytes(b"0123456789")
    real = snapshot.os.read
    monkeypatch.setattr(snapshot.os, "read", lambda fd, n: real(fd, min(n, 2)))
    assert snapshot.read_local_snapshot(source, max_bytes=10) == b"0123456789"


def test_snapshot_detects_growth_during_read(tmp_path, monkeypatch):
    from research import local_snapshot as snapshot
    source = tmp_path / "input"
    source.write_bytes(b"12345678")
    real = snapshot.os.read
    changed = False
    def grow(fd, count):
        nonlocal changed
        if not changed:
            changed = True
            with source.open("ab") as out:
                out.write(b"9")
        return real(fd, count)
    monkeypatch.setattr(snapshot.os, "read", grow)
    with pytest.raises(snapshot.SnapshotError, match="byte cap"):
        snapshot.read_local_snapshot(source, max_bytes=8)


def test_snapshot_detects_same_size_edit(tmp_path, monkeypatch):
    import os
    from research import local_snapshot as snapshot
    source = tmp_path / "input"
    source.write_bytes(b"0123456789")
    before = source.stat()
    real = snapshot.os.read
    changed = False
    def edit(fd, count):
        nonlocal changed
        data = real(fd, count)
        if not changed:
            changed = True
            source.write_bytes(b"9876543210")
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))
        return data
    monkeypatch.setattr(snapshot.os, "read", edit)
    with pytest.raises(snapshot.SnapshotError, match="changed"):
        snapshot.read_local_snapshot(source, max_bytes=10)


def test_snapshot_refuses_fifo_without_waiting(tmp_path):
    import os
    from research import local_snapshot as snapshot
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unavailable")
    path = tmp_path / "pipe"
    os.mkfifo(path)
    with pytest.raises(snapshot.SnapshotError, match="ordinary"):
        snapshot.read_local_snapshot(path, max_bytes=100)


def test_snapshot_rejects_path_swap_to_fifo(tmp_path, monkeypatch):
    import os
    from research import local_snapshot as snapshot
    if not (hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK")):
        pytest.skip("nonblocking FIFO unavailable")
    source = tmp_path / "input"
    source.write_bytes(b"data")
    real = snapshot.os.open
    def swap(path, flags):
        source.unlink()
        os.mkfifo(source)
        return real(path, flags)
    monkeypatch.setattr(snapshot.os, "open", swap)
    with pytest.raises(snapshot.SnapshotError):
        snapshot.read_local_snapshot(source, max_bytes=100)
