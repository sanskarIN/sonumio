"""Research-only forest-aware complete 189-case dual-method DSP verification."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from research import portable_forest_candidate as study
from research.portable_dsp_candidate import portable_raw
from sonumio.engines import procedural as dsp


def test_stress_cases_include_original_and_additional_sources():
    assert len(study.CASES) == 189
    assert len(study.METHODS) == 2
    assert study.PROTOCOL.endswith("-v1")


@pytest.mark.parametrize("method", study.METHODS)
def test_canonical_forest_chirp_stays_close_to_unchanged_generator(method):
    t = np.arange(study.RAW_FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    rng = np.random.default_rng(41)
    oracle_rng = np.random.default_rng(41)
    chirps = study.candidate_forest(rng, t, method=method)
    baseline = dsp._forest(oracle_rng, t)
    assert np.isfinite(chirps).all()
    assert chirps.dtype == np.float32
    assert chirps.shape == baseline.shape
    assert np.max(np.abs(chirps.astype(np.float64) -
                         baseline.astype(np.float64))) < 2e-5
    assert rng.bit_generator.state == oracle_rng.bit_generator.state


@pytest.mark.parametrize("method", study.METHODS)
def test_non_forest_sources_keep_exact_previous_portable_fingerprint(method):
    for members in ((0,), (1,), (2,), (4,), (5,), (0, 5), (1, 5)):
        candidate = study.forest_aware_raw(members, 41, method)
        previous = portable_raw(members, 41, method)
        np.testing.assert_array_equal(candidate, previous)


def test_seed_41_affected_contexts_are_finite_and_within_original_bounds():
    for members in ((1, 3), (2, 3)):
        for method in study.METHODS:
            result = study._case("heldout", members, 41, method)
            assert result["seed"] == 41
            assert result["maximum_24k_absolute_error"] < 2e-5
            assert result["maximum_8k_crop_absolute_error"] < 2e-5
            assert len(result["candidate_8k_crop_sha256"]) == 64


def test_input_contract_rejects_wrong_dtype_nan_rng_and_unknown_method():
    t = np.arange(50, dtype=np.float32) / dsp.SAMPLE_RATE
    rng = np.random.default_rng(0)
    for bad in (t.astype(np.float64), t.reshape(10, 5),
                np.full(50, np.nan, dtype=np.float32)):
        with pytest.raises(ValueError, match="Invalid"):
            study.candidate_forest(rng, bad)
    with pytest.raises(ValueError, match="Invalid"):
        study.candidate_forest(object(), t)
    with pytest.raises(ValueError, match="Invalid"):
        study.candidate_forest(rng, t, method="numpy_float32")
    for members, seed, method in (
        ((1, 1), 41, "numpy_float64"),
        ((1, 3), -1, "numpy_float64"),
        ((1, 3), True, "numpy_float64"),
        ([1, 3], 41, "numpy_float64"),
        ((1, 3), 41, "invalid"),
    ):
        with pytest.raises(ValueError, match="Invalid"):
            study.forest_aware_raw(members, seed, method)


@pytest.fixture(scope="module")
def report():
    return study.build_report()


def test_full_forest_aware_378_case_corpus_is_hash_only_and_integrity_checked(report):
    study.validate_report(report)
    assert report["protocol"] == study.PROTOCOL
    assert report["case_count_per_method"] == 189
    assert len(report["recordings"]) == 378
    assert report["original_generator_modified"] is False
    assert report["original_neural_data_modified"] is False
    assert report["candidate_used_for_neural_training"] is False
    assert report["audio_files_saved"] is False
    assert report["model_weights_saved"] is False
    assert report["additional_seed_3_is_text_audio_training_seed"] is True
    result = study.compare_reports(report, json.loads(json.dumps(report, sort_keys=True)))
    assert result["cases_per_method"] == 189
    assert result["same_cpu_model"] is True
    assert result["all_candidate_bytes_match"] is True
    assert result["previous_candidate_hashes_match"] is True
    for method in study.METHODS:
        assert [result["by_method_split"][method][split]["cases"]
                for split in ("train", "validation", "heldout")] == [42, 42, 105]


def test_internal_rehashed_tampering_is_observable_not_wrongly_validated(report):
    changed = copy.deepcopy(report)
    row = changed["recordings"][0]
    row["candidate_24k_sha256"] = "f" * 64
    changed["splits"] = study._summary(changed["recordings"])
    study.validate_report(changed)
    result = study.compare_reports(report, changed)
    assert result["all_candidate_bytes_match"] is False
    assert result["by_method_split"]["numpy_float64"]["train"]["different_24k"] == 1
    assert result["previous_candidate_hashes_match"] is True


def test_cross_method_duplication_and_forged_metadata_rejected(report):
    changed = copy.deepcopy(report)
    changed["candidate_used_for_neural_training"] = True
    with pytest.raises(ValueError, match="metadata"):
        study.validate_report(changed)
    changed = copy.deepcopy(report)
    changed["additional_seed_3_is_text_audio_training_seed"] = False
    with pytest.raises(ValueError, match="metadata"):
        study.validate_report(changed)
    changed = copy.deepcopy(report)
    changed["recordings"][0]["candidate_24k_sha256"] = "invalid"
    with pytest.raises(ValueError, match="checksum"):
        study.validate_report(changed)
    changed = copy.deepcopy(report)
    target = changed["recordings"][-1]
    target["candidate_8k_crop_sha256"] = changed["recordings"][0]["candidate_8k_crop_sha256"]
    changed["splits"] = study._summary(changed["recordings"])
    # Duplicate detection applies PER method and may differ between methods.
    peer = next(x for x in changed["recordings"] if
                x["method"] == target["method"] and x["seed"] != target["seed"])
    target["candidate_8k_crop_sha256"] = peer["candidate_8k_crop_sha256"]
    changed["splits"] = study._summary(changed["recordings"])
    with pytest.raises(ValueError, match="Duplicate"):
        study.validate_report(changed)
