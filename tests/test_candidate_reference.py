"""Reference comparison unit tests; full audio generation runs in CPU matrix."""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")

from research import candidate_reference as ref
from research.complete_dsp_candidate import CANDIDATE_RECIPE, PROTOCOL


def _synthetic_report_structure() -> dict:
    # Unit-test the comparison fields without rendering 84 audio arrays.
    # The production verify_reference() still calls the real validator.
    return {
        "protocol": PROTOCOL,
        "candidate_recipe": CANDIDATE_RECIPE,
        "scope": "full",
        "recordings": [None] * 84,
        "splits": copy.deepcopy(ref.REFERENCE_SPLITS),
    }


def test_reference_has_complete_exact_hashes_and_source_evidence():
    assert set(ref.REFERENCE_SPLITS) == {"train", "validation"}
    assert ref.REFERENCE_SOURCE_COMMIT.startswith("742d71d0")
    assert ref.REFERENCE_WORKFLOW.endswith("/44063872705")
    assert sum(x["recordings"] for x in ref.REFERENCE_SPLITS.values()) == 84
    for split in ref.REFERENCE_SPLITS.values():
        for field in ("candidate_raw_sha256", "candidate_8k_dataset_sha256"):
            assert len(split[field]) == 64
            assert set(split[field]).issubset(set("0123456789abcdef"))


def test_complete_golden_comparison_passes_without_pinning_original_numpy(monkeypatch):
    monkeypatch.setattr(ref, "validate_report", lambda report: None)
    report = _synthetic_report_structure()
    result = ref.verify_reference(report)
    assert result["candidate_full_corpus_matches_reference"] is True
    assert result["recordings_checked"] == 84
    assert result["original_numpy_hash_pinned"] is False
    assert result["no_audio_or_weights_published"] is True


@pytest.mark.parametrize("field", [
    "candidate_raw_sha256", "candidate_8k_dataset_sha256", "recordings",
])
@pytest.mark.parametrize("split", ["train", "validation"])
def test_hash_or_count_drift_rejected_even_if_other_split_matches(
    monkeypatch, field, split,
):
    monkeypatch.setattr(ref, "validate_report", lambda report: None)
    report = _synthetic_report_structure()
    if field == "recordings":
        report["splits"][split][field] += 1
    else:
        report["splits"][split][field] = "0" * 64
    with pytest.raises(ValueError, match="reference drift"):
        ref.verify_reference(report)


def test_wrong_scope_is_rejected_even_with_matching_aggregates(monkeypatch):
    monkeypatch.setattr(ref, "validate_report", lambda report: None)
    report = _synthetic_report_structure()
    report["scope"] = "smoke"
    with pytest.raises(ValueError, match="84"):
        ref.verify_reference(report)
    report = _synthetic_report_structure()
    report["recordings"] = report["recordings"][:-1]
    with pytest.raises(ValueError, match="84"):
        ref.verify_reference(report)


def test_missing_split_and_incompatible_recipe_fail_closed(monkeypatch):
    monkeypatch.setattr(ref, "validate_report", lambda report: None)
    report = _synthetic_report_structure()
    report["splits"].pop("validation")
    with pytest.raises(ValueError, match="Missing"):
        ref.verify_reference(report)
    report = _synthetic_report_structure()
    report["candidate_recipe"] = "unknown"
    with pytest.raises(ValueError, match="Incompatible"):
        ref.verify_reference(report)
