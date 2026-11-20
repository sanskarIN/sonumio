"""Verify the isolated alternate DSP pipeline without switching legacy data."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.complete_dsp_candidate import (
    CANDIDATE_RECIPE, FRAMES, PROTOCOL, RAW_FRAMES, SMOKE_COMBINATIONS,
    _difference, _identities, _summarize, build_report, candidate_raw,
    compare_reports, trace_one, validate_report,
)
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES
from sonumio.engines import procedural as original


@pytest.mark.parametrize("category", range(len(CATEGORIES)))
def test_candidate_single_source_is_valid_and_close_to_unmodified_renderer(category):
    members = (category,)
    signal = candidate_raw(members, 0)
    legacy = original.generate(CATEGORIES[category], 1.5, 0).samples
    error = _difference(signal, legacy)
    assert signal.shape == (RAW_FRAMES,)
    assert signal.dtype == np.float32
    assert np.isfinite(signal).all()
    assert error["max_abs"] < 2e-5
    if CATEGORIES[category] in ("forest", "ambient"):
        assert error["max_abs"] == 0
        np.testing.assert_array_equal(signal, legacy)


@pytest.mark.parametrize("members", [(0, 1), (0, 5), (2, 3), (4, 5)])
def test_two_layer_candidate_respects_combination_order_and_original_waveform(members):
    signal = candidate_raw(members, 101)
    prompt = " and ".join(CATEGORIES[x] for x in members)
    legacy = original.generate(prompt, 1.5, 101).samples
    assert _difference(signal, legacy)["max_abs"] < 2e-5


@pytest.mark.parametrize(("members", "seed", "split"), [
    ((0,), 0, "train"), ((1, 5), 101, "validation"),
    ((3,), 102, "validation"),
])
def test_candidate_end_to_end_checks_original_training_oracle(members, seed, split):
    row = trace_one(members, seed, split)
    assert len(row["legacy_cropped"]) == 64
    assert row["crop_offset"] >= 400
    assert row["raw_error"]["max_abs"] < 2e-5
    assert row["resampled_error"]["max_abs"] < 2e-5
    assert row["cropped_error"]["max_abs"] < 2e-5
    assert row["elapsed_seconds"] >= 0


def test_invalid_source_combination_seed_scope_and_split_fail_closed():
    with pytest.raises(ValueError):
        candidate_raw((0, 0), 0)
    with pytest.raises(ValueError):
        candidate_raw((0,), -1)
    with pytest.raises(ValueError):
        candidate_raw((0,), True)
    with pytest.raises(ValueError):
        trace_one((0,), True, "train")
    with pytest.raises(ValueError):
        trace_one((0,), 101, "train")
    with pytest.raises(ValueError):
        trace_one((0,), 0, "unknown")
    with pytest.raises(ValueError):
        _identities("not-valid")
    assert len(_identities("smoke")) == len(SMOKE_COMBINATIONS) * 4
    assert len(_identities("full")) == len(SYNTHETIC_COMBINATIONS) * 4


def test_difference_rejects_wrong_types_nan_and_shapes():
    a = np.ones(12, dtype=np.float32)
    for b in (
        np.ones(12, dtype=np.float64),
        np.ones(11, dtype=np.float32),
        np.asarray([float("nan")] + [1] * 11, dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            _difference(a, b)
    equal = _difference(a, a)
    assert equal == {"max_abs": 0.0, "rms": 0.0,
                     "different_float32_values": 0}


@pytest.fixture(scope="module")
def full_smoke():
    return build_report("smoke")


def test_smoke_includes_both_splits_and_explicitly_never_trains(full_smoke):
    validate_report(full_smoke)
    assert full_smoke["protocol"] == PROTOCOL
    assert full_smoke["candidate_recipe"] == CANDIDATE_RECIPE
    assert full_smoke["scope"] == "smoke"
    assert len(full_smoke["recordings"]) == 40
    assert full_smoke["splits"]["train"]["recordings"] == 20
    assert full_smoke["splits"]["validation"]["recordings"] == 20
    assert full_smoke["sample_rate_24k"] == 24000
    assert full_smoke["sample_rate_8k"] == 8000
    assert full_smoke["cropped_frames_8k"] == FRAMES
    assert full_smoke["original_source_recipe_modified"] is False
    assert full_smoke["candidate_used_for_neural_training"] is False
    assert full_smoke["audio_files_saved"] is False
    assert full_smoke["model_weights_saved"] is False
    same = compare_reports(full_smoke, copy.deepcopy(full_smoke))
    assert same["recordings"] == 40
    assert same["candidate_24k_exact"] is True
    assert same["candidate_8k_exact"] is True
    assert same["candidate_crops_exact"] is True
    assert same["original_crops_exact"] is True


def test_hash_report_drift_is_detected_and_aggregate_forgery_rejected(full_smoke):
    corrupted = copy.deepcopy(full_smoke)
    corrupted["candidate_used_for_neural_training"] = True
    with pytest.raises(ValueError, match="metadata"):
        validate_report(corrupted)
    corrupted = copy.deepcopy(full_smoke)
    corrupted["recordings"][0]["candidate_cropped"] = "a" * 64
    with pytest.raises(ValueError, match="summary"):
        validate_report(corrupted)
    corrupted = copy.deepcopy(full_smoke)
    corrupted["recordings"][0]["candidate_cropped"] = "b" * 64
    corrupted["splits"] = _summarize(corrupted["recordings"])
    validate_report(corrupted)
    compared = compare_reports(full_smoke, corrupted)
    assert compared["candidate_crops_exact"] is False
    assert compared["candidate_24k_exact"] is True
    # Even a consistent re-hashed summary must not admit duplicate audio
    # within a split or leaked waveform identity across train and validation.
    corrupted = copy.deepcopy(full_smoke)
    corrupted["recordings"][1]["candidate_cropped"] = (
        corrupted["recordings"][0]["candidate_cropped"]
    )
    corrupted["splits"] = _summarize(corrupted["recordings"])
    with pytest.raises(ValueError, match="Duplicate waveform"):
        validate_report(corrupted)
    corrupted = copy.deepcopy(full_smoke)
    val = next(x for x in corrupted["recordings"] if x["split"] == "validation")
    val["candidate_cropped"] = corrupted["recordings"][0]["candidate_cropped"]
    corrupted["splits"] = _summarize(corrupted["recordings"])
    with pytest.raises(ValueError, match="Train/validation"):
        validate_report(corrupted)
    corrupted = copy.deepcopy(full_smoke)
    corrupted["recordings"][0]["crop_offset"] = -2
    with pytest.raises(ValueError, match="crop"):
        validate_report(corrupted)
    corrupted = copy.deepcopy(full_smoke)
    corrupted["recordings"][0]["cropped_error"]["rms"] = float("nan")
    with pytest.raises(ValueError, match="metrics"):
        validate_report(corrupted)
    corrupted = copy.deepcopy(full_smoke)
    corrupted["recordings"][0]["seed"] = corrupted["recordings"][1]["seed"]
    with pytest.raises(ValueError, match="recording"):
        validate_report(corrupted)


def test_train_validation_share_no_source_identity(full_smoke):
    train = {(tuple(r["sources"]), r["seed"])
             for r in full_smoke["recordings"] if r["split"] == "train"}
    validation = {(tuple(r["sources"]), r["seed"])
                  for r in full_smoke["recordings"] if r["split"] == "validation"}
    assert not train.intersection(validation)
    assert all(math.isfinite(r["cropped_error"]["max_abs"])
               for r in full_smoke["recordings"])
