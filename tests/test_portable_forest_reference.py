"""Test measured cross-NumPy, four-CPU golden 378-case DSP fingerprints."""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("torch")

from research import portable_forest_candidate as study
from research import portable_forest_reference as gold


def test_golden_provenance_and_source_counts():
    assert gold.REFERENCE_WORKFLOW.endswith("/44095705429")
    assert gold.REFERENCE_NUMPY == ("1.26.4", "2.4.6")
    assert [gold.REFERENCE_SPLITS[s]["recordings"] for s in
            ("train", "validation", "heldout")] == [42, 42, 105]
    assert sum(r["recordings"] for r in gold.REFERENCE_SPLITS.values()) == 189
    for split in gold.REFERENCE_SPLITS.values():
        assert len(split["candidate_24k_sha256"]) == 64
        assert len(split["candidate_8k_crop_sha256"]) == 64


@pytest.fixture(scope="module")
def report():
    if np.__version__ not in gold.REFERENCE_NUMPY:
        pytest.skip("Only compare measured NumPy 1.26.4 and 2.4.6 references")
    return study.build_report()


def test_actual_full_378_candidate_corpus_matches_independent_goldens(report):
    result = gold.verify_reference(report)
    assert result["protocol"] == gold.REFERENCE_PROTOCOL
    assert result["matched_verified_full_raw_and_crop_reference"] is True
    assert result["cases_per_method"] == 189
    assert result["total_candidate_recordings"] == 378
    assert result["methods"] == list(study.METHODS)
    assert result["num_independent_numpy_2_4_cpu_jobs"] == 4
    assert result["num_same_cpu_cross_numpy_jobs"] == 1
    assert result["original_dsp_hash_pinned"] is False
    assert result["original_data_modified"] is False
    assert result["neural_training_validity_claimed"] is False
    assert result["audio_files_saved"] is False
    assert result["model_weights_saved"] is False


def test_invalid_numpy_version_is_not_silently_marked_verified(report):
    tampered = copy.deepcopy(report)
    tampered["environment"]["numpy"] = "2.2.6"
    with pytest.raises(ValueError, match="verified only"):
        gold.verify_reference(tampered)


def test_raw_sha_drift_is_caught_even_when_internal_aggregates_are_recomputed(report):
    tampered = copy.deepcopy(report)
    tampered["recordings"][0]["candidate_24k_sha256"] = "f" * 64
    tampered["splits"] = study._summary(tampered["recordings"])
    study.validate_report(tampered)
    with pytest.raises(ValueError, match="SHA drift"):
        gold.verify_reference(tampered)


def test_final_crop_sha_drift_is_caught_even_when_own_aggregates_are_updated(report):
    tampered = copy.deepcopy(report)
    tampered["recordings"][0]["candidate_8k_crop_sha256"] = "e" * 64
    tampered["splits"] = study._summary(tampered["recordings"])
    study.validate_report(tampered)
    with pytest.raises(ValueError, match="SHA drift"):
        gold.verify_reference(tampered)


def test_golden_rejects_forged_training_claim_or_missing_source_grid(report):
    bad = copy.deepcopy(report)
    bad["candidate_used_for_neural_training"] = True
    with pytest.raises(ValueError, match="metadata"):
        gold.verify_reference(bad)
    bad = copy.deepcopy(report)
    bad["splits"]["python_math_sin"].pop("validation")
    with pytest.raises(ValueError):
        gold.verify_reference(bad)
