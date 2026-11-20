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
