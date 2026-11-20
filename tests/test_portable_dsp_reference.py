"""Regression gates for independently verified 378-case portable DSP goldens."""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("torch")

from research import portable_dsp_reference as gold
from research import portable_dsp_candidate as study


def test_reference_identity_and_all_expected_split_counts():
    assert gold.REFERENCE_NUMPY == "2.4.6"
    assert len(gold.REFERENCE_SPLITS) == 3
    assert [gold.REFERENCE_SPLITS[k]["recordings"]
            for k in ("train", "validation", "heldout")] == [42, 42, 105]
    assert sum(x["recordings"] for x in gold.REFERENCE_SPLITS.values()) == 189
    assert gold.REFERENCE_RUN.endswith("/44080000639")


@pytest.fixture(scope="module")
def portable_report():
    if np.__version__ != gold.REFERENCE_NUMPY:
        pytest.skip("Exact four-CPU goldens are NumPy 2.4.6-specific")
    return study.build_report()


def test_entire_actual_portable_corpus_matches_independent_four_cpu_goldens(portable_report):
    status = gold.verify_reference(portable_report)
    assert status["protocol"] == gold.REFERENCE_PROTOCOL
    assert status["recordings_per_method"] == 189
    assert status["total_candidate_recordings"] == 378
    assert status["methods_verified"] == list(study.METHODS)
    assert status["all_candidate_hashes_match_four_cpu_reference"] is True
    assert status["original_generator_hash_pinned"] is False
    assert status["original_training_data_modified"] is False
    assert status["model_weights_saved"] is False
    assert status["audio_files_saved"] is False


def test_reference_rejects_wrong_numpy_version_even_if_metadata_claims_stability(portable_report):
    changed = copy.deepcopy(portable_report)
    changed["environment"]["numpy"] = "1.26.4"
    with pytest.raises(ValueError, match="ONLY to NumPy"):
        gold.verify_reference(changed)


def test_reference_detects_changed_8k_crop_hash_with_recomputed_internal_aggregate(portable_report):
    changed = copy.deepcopy(portable_report)
    changed["recordings"][0]["portable_8k_crop_sha256"] = "a" * 64
    changed["splits"] = study._summary(changed["recordings"])
    study.validate_report(changed)
    with pytest.raises(ValueError, match="reference drift"):
        gold.verify_reference(changed)


def test_reference_detects_changed_24k_hash_with_recomputed_internal_aggregate(portable_report):
    changed = copy.deepcopy(portable_report)
    changed["recordings"][0]["portable_24k_sha256"] = "b" * 64
    changed["splits"] = study._summary(changed["recordings"])
    study.validate_report(changed)
    with pytest.raises(ValueError, match="reference drift"):
        gold.verify_reference(changed)
