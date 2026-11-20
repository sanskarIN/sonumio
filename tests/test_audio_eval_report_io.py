"""Bounded local-only metric archives, metadata tampering and audio isolation."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from sonumio.audio_core.errors import (
    AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError,
)
from research.audio_eval.report_io import (
    REPORT_PROTOCOL, NumericReport, NumericReportBundle,
    compare_numeric_report_versions, deserialize_numeric_report,
    read_numeric_report, serialize_numeric_report, write_numeric_report,
)


@pytest.fixture
def sample_report():
    return NumericReport(
        "evaluation_case_001",
        "numerical-acoustic-evaluation",
        payload={
            "metric": "sample_rmse",
            "group_ids": ["source_0", "source_1"],
            "scores": [0.01, 0.03],
            "group_weighted_mean": .02,
            "rights_authenticated": False,
            "synthetic_sound_only": True,
        },
        source_note="Synthetic fixture, no private WAV or weights",
    )


def test_numeric_report_is_canonical_and_has_two_bound_sha_bindings(sample_report):
    first = serialize_numeric_report(sample_report)
    second = serialize_numeric_report(sample_report)
    assert first == second
    decoded = deserialize_numeric_report(first)
    assert decoded.sha256 == sample_report.sha256
    assert decoded.content_sha256 == sample_report.content_sha256
    assert decoded.as_dict() == sample_report.as_dict()
    assert len(sample_report.sha256) == 64
    assert len(sample_report.content_sha256) == 64
    assert sample_report.as_dict()["audio_files_embedded"] is False
    assert sample_report.as_dict()["checkpoint_or_model_weights_embedded"] is False
    assert sample_report.as_dict()["release_authorized"] is False


def test_report_snapshot_is_recursively_immutable_and_export_is_independent(sample_report):
    original = sample_report.sha256
    with pytest.raises(TypeError):
        sample_report.payload["metric"] = "other"
    with pytest.raises(TypeError):
        sample_report.payload["scores"][0] = 99
    exported = sample_report.as_dict()
    exported["payload"]["scores"][0] = 77
    exported["payload"]["metric"] = "tampered"
    assert sample_report.sha256 == original
    assert sample_report.as_dict()["payload"]["scores"][0] == .01
    assert sample_report.as_dict()["payload"]["metric"] == "sample_rmse"
    assert deserialize_numeric_report(serialize_numeric_report(sample_report)).sha256 == original


def test_json_field_order_does_not_change_report_hash(sample_report):
    other = NumericReport(
        sample_report.report_id, sample_report.domain,
        dict(reversed(tuple(sample_report.payload.items()))),
        source_note=sample_report.source_note,
    )
    assert other.sha256 == sample_report.sha256
    assert serialize_numeric_report(other) == serialize_numeric_report(sample_report)


def test_forged_metric_value_and_recomputed_json_object_are_rejected(sample_report):
    corrupted = sample_report.as_dict()
    corrupted["payload"]["group_weighted_mean"] = .99
    encoded = json.dumps(corrupted).encode("utf8")
    with pytest.raises(AudioIntegrityError, match="digest"):
        deserialize_numeric_report(encoded)
    # Integrity hashes detect accidental edits but not a malicious
    # party with authority to recompute BOTH checksums (no signature).
    recomputed = NumericReport(
        sample_report.report_id, sample_report.domain,
        dict(sample_report.payload, group_weighted_mean=.99),
        sample_report.source_note,
    )
    assert recomputed.sha256 != sample_report.sha256


def test_report_rejects_false_release_and_rights_claims(sample_report):
    for key in (
        "release_authorized", "audio_files_embedded",
        "human_rights_authenticated", "checkpoint_or_model_weights_embedded",
    ):
        wrong = sample_report.as_dict()
        wrong[key] = True
        with pytest.raises(AudioIntegrityError):
            deserialize_numeric_report(json.dumps(wrong).encode("utf8"))


def test_archive_does_not_serialize_numpy_audio_or_secret_keys():
    for payload in (
        {"pcm_samples": [0., 1., 0.]},
        {"nested": {"wav_bytes": "raw"}},
        {"api_key": "unsafe"},
        {"model_checkpoint": "checkpoint"},
        {"spectrum": np.zeros(12, dtype=np.float32)},
        {"hidden": b"payload"},
        {"nonfinite": float("nan")},
    ):
        with pytest.raises(AudioContractError):
            NumericReport("case_0", "research-benchmark", payload)
    with pytest.raises(AudioContractError):
        NumericReport("case_0", "unknown", {"x": 1})


def test_atomic_report_output_requires_local_explicit_destination(
    sample_report, tmp_path
):
    path = tmp_path / "numerical-results.json"
    assert write_numeric_report(sample_report, path) == path
    restored = read_numeric_report(path)
    assert restored.sha256 == sample_report.sha256
    with pytest.raises(AudioFormatError, match="overwrite"):
        write_numeric_report(sample_report, path)
    written = write_numeric_report(sample_report, path, overwrite=True)
    assert written == path
    with pytest.raises(AudioFormatError):
        write_numeric_report(sample_report, tmp_path / "not-json.wav")
    with pytest.raises(AudioFormatError):
        write_numeric_report(sample_report, tmp_path / "missing" / "child.json")


def test_symlinked_report_is_never_silently_followed(sample_report, tmp_path):
    path = tmp_path / "real.json"
    write_numeric_report(sample_report, path)
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(AudioFormatError):
        write_numeric_report(sample_report, alias, overwrite=True)
    with pytest.raises(AudioFormatError):
        read_numeric_report(alias)


def test_duplicate_json_keys_rejected_even_if_one_digest_looks_valid(sample_report):
    good = serialize_numeric_report(sample_report)
    text = good.decode().replace(
        '"release_authorized":false',
        '"release_authorized":false,"release_authorized":false',
        1,
    )
    with pytest.raises(AudioContractError):
        deserialize_numeric_report(text.encode())


def test_numeric_archive_bundle_has_no_payload_or_audio_bytes(sample_report):
    second = NumericReport(
        "evaluation_case_002", "audio-core-health",
        {"peak": .4, "clipped": False},
    )
    result = NumericReportBundle((sample_report, second), "test_run")
    data = result.as_dict()
    assert data["numeric_payloads_embedded"] is False
    assert data["audio_files_embedded"] is False
    assert data["model_weights_embedded"] is False
    assert data["reports"][0]["sha256"] == sample_report.sha256
    assert len(result.sha256) == 64
    with pytest.raises(AudioContractError):
        NumericReportBundle((sample_report, sample_report), "reused")


def test_numeric_report_diff_discloses_changed_fields_not_media(sample_report):
    changed = NumericReport(
        sample_report.report_id, sample_report.domain,
        dict(sample_report.payload, new_measurement=1),
        sample_report.source_note,
    )
    report = compare_numeric_report_versions(sample_report, changed)
    assert report["changed"] is True
    assert report["added_payload_keys"] == ["new_measurement"]
    assert report["deleted_payload_keys"] == []
    assert report["changed_payload_keys"] == []
    assert report["rights_or_release_status_inferred"] is False
    assert report["binary_audio_or_weights_compared"] is False
    unchanged = compare_numeric_report_versions(sample_report, sample_report)
    assert unchanged["changed"] is False


def test_cannot_compare_unrelated_report_ids(sample_report):
    other = NumericReport("other", sample_report.domain, {"metric": "sample_rmse"})
    with pytest.raises(AudioContractError):
        compare_numeric_report_versions(sample_report, other)


def test_report_refuses_large_canonical_json_and_invalid_ids():
    with pytest.raises(AudioContractError):
        NumericReport("bad id", "audio-core-health", {"x": 1})
    with pytest.raises(AudioContractError):
        NumericReport("valid", "audio-core-health", {})
    with pytest.raises(AudioLimitError):
        NumericReport("valid", "audio-core-health", {
            "big": "A" * 20000
        })


def test_invalid_truncated_payload_and_wrong_json_types_fail_closed():
    for data in (b"", b"{}", b"[]", b"null",
                 b'{"x":NaN}', b"not json"):
        with pytest.raises((AudioContractError, AudioIntegrityError, AudioFormatError)):
            deserialize_numeric_report(data)
