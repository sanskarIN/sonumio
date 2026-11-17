"""Held-out DSP candidate fingerprints and fail-closed metadata checks."""
from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("torch")

from research import heldout_dsp_probe as study
from research.text_audio import SYNTHETIC_COMBINATIONS


def test_heldout_seed_grid_is_disjoint_and_complete():
    assert set(study.HELDOUT_SEEDS).isdisjoint(study.TRAIN_SEEDS)
    assert set(study.HELDOUT_SEEDS).isdisjoint(study.VALIDATION_SEEDS)
    assert len(study.HELDOUT_SEEDS) == 5
    assert len(SYNTHETIC_COMBINATIONS) == 21


def test_reject_unsupported_seed_and_source_identity():
    for members, seed in (
        ((0,), 0), ((0,), 101), ((0,), True), ((0,), -1),
        ((0, 0), 3), ((0,), 2**32), ((99,), 3),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            study._case(members, seed)


@pytest.mark.parametrize("seed", study.HELDOUT_SEEDS)
def test_unseen_seed_raw_candidate_matches_original_numerically(seed):
    row = study._case((0, 1), seed)
    assert row["seed"] == seed
    assert row["raw_max_abs_error"] < 2e-5
    assert row["crop_max_abs_error"] < 2e-5
    assert row["crop_offset"] >= 400
    assert len(row["fast_crop_sha256"]) == 64


@pytest.fixture(scope="module")
def report():
    return study.build_report()


def test_complete_heldout_fingerprint_report_is_hashes_not_wav(report):
    study.validate_report(report)
    assert report["protocol"] == study.PROTOCOL
    assert len(report["recordings"]) == 105
    assert set(report["by_seed"]) == set(map(str, study.HELDOUT_SEEDS))
    assert all(x["recordings"] == 21 for x in report["by_seed"].values())
    assert report["original_generator_unchanged"] is True
    assert report["candidate_used_for_neural_training"] is False
    assert report["audio_files_saved"] is False
    assert report["model_weights_saved"] is False
    same = study.compare_reports(report, copy.deepcopy(report))
    assert same["distinct_recordings"] == 105
    assert same["identical_fast_24k_hashes"] is True
    assert same["identical_fast_8k_crop_hashes"] is True
    assert same["identical_original_8k_crop_hashes"] is True


def test_modified_sha_or_recomputed_aggregate_detected(report):
    damaged = copy.deepcopy(report)
    damaged["recordings"][0]["fast_crop_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="aggregate"):
        study.validate_report(damaged)
    damaged["by_seed"] = study._summary(damaged["recordings"])
    study.validate_report(damaged)
    compare = study.compare_reports(report, damaged)
    assert compare["identical_fast_24k_hashes"] is True
    assert compare["identical_fast_8k_crop_hashes"] is False


def test_duplicate_crop_rejected_even_when_metadata_rehashed(report):
    damaged = copy.deepcopy(report)
    damaged["recordings"][1]["fast_crop_sha256"] = (
        damaged["recordings"][0]["fast_crop_sha256"]
    )
    damaged["by_seed"] = study._summary(damaged["recordings"])
    with pytest.raises(ValueError, match="duplicate"):
        study.validate_report(damaged)


def test_model_training_and_nonfinite_metrics_cannot_be_mislabeled(report):
    damaged = copy.deepcopy(report)
    damaged["candidate_used_for_neural_training"] = True
    with pytest.raises(ValueError, match="metadata"):
        study.validate_report(damaged)
    damaged = copy.deepcopy(report)
    damaged["recordings"][0]["crop_max_abs_error"] = float("nan")
    with pytest.raises(ValueError, match="error"):
        study.validate_report(damaged)
    damaged = copy.deepcopy(report)
    damaged["recordings"][0]["seed"] = True
    with pytest.raises(ValueError, match="held-out source"):
        study.validate_report(damaged)
    damaged = copy.deepcopy(report)
    damaged["recordings"][0]["crop_offset"] = 1601
    with pytest.raises(ValueError, match="offset"):
        study.validate_report(damaged)
