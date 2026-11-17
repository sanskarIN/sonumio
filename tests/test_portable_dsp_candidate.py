"""Complete 189-case numerical study of two separate ambient sine methods."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

pytest.importorskip("torch")

from research import portable_dsp_candidate as study
from research.complete_dsp_candidate import _difference
from research.fast_dsp_candidate import render_fast_raw
from research.wave_diffusion import CATEGORIES
from sonumio.engines import procedural as original


def test_case_grid_contains_only_disjoint_original_and_heldout_seeds():
    assert len(study.CASES) == 189
    assert len(set(study.CASES)) == 189
    assert len({(m, split, members, seed) for m in study.METHODS
                for split, members, seed in study.CASES}) == 378
    assert len(study.METHODS) == 2
    assert len(study.SYNTHETIC_COMBINATIONS) == 21


@pytest.mark.parametrize("method", study.METHODS)
def test_portable_drone_replaces_only_ambient_sine_and_is_close_to_original(method):
    actual = study.portable_raw((5,), 0, method)
    reference = original.generate("ambient", 1.5, 0).samples
    assert actual.shape == reference.shape == (36000,)
    assert actual.dtype == reference.dtype == np.float32
    assert _difference(actual, reference)["max_abs"] < 2e-5


@pytest.mark.parametrize("method", study.METHODS)
def test_no_ambient_still_matches_separate_float64_fast_source_exactly(method):
    for members in ((0,), (1,), (2,), (3,), (4,), (0, 1), (2, 3)):
        portable = study.portable_raw(members, 3, method)
        original_fast = render_fast_raw(members, 3)
        np.testing.assert_array_equal(portable, original_fast)


def test_invalid_method_sources_and_seeds_are_rejected():
    for members, seed, method in (
        ((0,), -1, "numpy_float64"),
        ((0,), True, "numpy_float64"),
        ((0, 0), 0, "numpy_float64"),
        ((99,), 0, "numpy_float64"),
        ([0], 0, "numpy_float64"),
        ((0,), 0, "unsupported"),
    ):
        with pytest.raises(ValueError, match="Invalid"):
            study.portable_raw(members, seed, method)
    for args in (
        ("train", (0,), 101, "numpy_float64"),
        ("validation", (0,), 0, "numpy_float64"),
        ("heldout", (0,), 101, "python_math_sin"),
        ("train", (0,), 0, "invalid"),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            study.trace_one(*args)


def test_independent_legacy_training_oracle_and_heldout_samples():
    a = study.trace_one("train", (0, 5), 0, "numpy_float64")
    b = study.trace_one("heldout", (1, 5), 17, "python_math_sin")
    assert a["crop_offset"] != b["crop_offset"]
    for row in (a, b):
        assert len(row["portable_24k_sha256"]) == 64
        assert len(row["portable_8k_crop_sha256"]) == 64
        assert row["raw_max_abs_error"] < 2e-5
        assert row["crop_max_abs_error"] < 2e-5
        assert row["render_24k_seconds"] > 0


@pytest.fixture(scope="module")
def corpus():
    return study.build_report()


def test_all_378_study_records_validate_and_never_train(corpus):
    study.validate_report(corpus)
    assert corpus["protocol"] == study.PROTOCOL
    assert corpus["case_count_per_method"] == 189
    assert len(corpus["recordings"]) == 378
    assert corpus["original_renderer_modified"] is False
    assert corpus["original_neural_data_modified"] is False
    assert corpus["portable_candidate_used_for_training"] is False
    assert corpus["audio_files_saved"] is False
    assert corpus["model_weights_saved"] is False
    for method in study.METHODS:
        assert corpus["splits"][method]["train"]["recordings"] == 42
        assert corpus["splits"][method]["validation"]["recordings"] == 42
        assert corpus["splits"][method]["heldout"]["recordings"] == 105


def test_same_corpus_roundtrips_json_and_comparison(corpus):
    other = json.loads(json.dumps(corpus, sort_keys=True))
    study.validate_report(other)
    report = study.compare_reports(corpus, other)
    assert report["cases_per_method"] == 189
    assert report["same_cpu_model"] is True
    assert all(report["portable_24k_exact"].values())
    assert all(report["portable_final_crops_exact"].values())
    assert report["legacy_crops_exact"] is True
    assert report["original_84_portable_exact"] is True
    for method in study.METHODS:
        for split, cases in (("train", 42), ("validation", 42),
                             ("heldout", 105)):
            metrics = report["discrepancies_by_method_split"][method][split]
            assert metrics["cases"] == cases
            assert metrics["different_24k"] == 0
            assert metrics["different_8k_crop"] == 0
            assert metrics["different_legacy_crop"] == 0
            assert metrics["first_different_cases"] == []


def test_modified_sha_rejected_even_if_summary_recomputed(corpus):
    damaged = copy.deepcopy(corpus)
    damaged["recordings"][0]["portable_8k_crop_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="aggregate"):
        study.validate_report(damaged)
    damaged["splits"] = study._summary(damaged["recordings"])
    study.validate_report(damaged)
    compared = study.compare_reports(corpus, damaged)
    assert compared["portable_24k_exact"]["numpy_float64"] is True
    assert compared["portable_final_crops_exact"]["numpy_float64"] is False
    assert compared["portable_final_crops_exact"]["python_math_sin"] is True
    assert compared["original_84_portable_exact"] is False
    changed = compared["discrepancies_by_method_split"]["numpy_float64"]["train"]
    assert changed["different_24k"] == 0
    assert changed["different_8k_crop"] == 1
    assert changed["different_legacy_crop"] == 0
    assert changed["by_seed_8k_crop"]["0"] == 1
    assert changed["by_seed_8k_crop"]["1"] == 0
    assert changed["first_different_cases"][0]["seed"] == 0


def test_exact_crop_duplicates_are_rejected_across_splits_and_seeds(corpus):
    altered = copy.deepcopy(corpus)
    records = altered["recordings"]
    source = next(r for r in records if r["method"] == "numpy_float64"
                  and r["split"] == "train")
    target = next(r for r in records if r["method"] == "numpy_float64"
                  and r["split"] == "heldout")
    target["portable_8k_crop_sha256"] = source["portable_8k_crop_sha256"]
    altered["splits"] = study._summary(records)
    with pytest.raises(ValueError, match="Duplicate exact"):
        study.validate_report(altered)


def test_fail_closed_on_neural_training_flags_bad_metadata_or_nan(corpus):
    tampered = copy.deepcopy(corpus)
    tampered["portable_candidate_used_for_training"] = True
    with pytest.raises(ValueError, match="metadata"):
        study.validate_report(tampered)
    tampered = copy.deepcopy(corpus)
    tampered["recordings"][0]["raw_max_abs_error"] = float("nan")
    with pytest.raises(ValueError, match="difference"):
        study.validate_report(tampered)
    tampered = copy.deepcopy(corpus)
    tampered["recordings"][0]["method"] = "unapproved"
    with pytest.raises(ValueError, match="identity"):
        study.validate_report(tampered)
    tampered = copy.deepcopy(corpus)
    tampered["recordings"][0]["crop_offset"] = 1601
    with pytest.raises(ValueError, match="offset"):
        study.validate_report(tampered)
