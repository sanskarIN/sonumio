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
