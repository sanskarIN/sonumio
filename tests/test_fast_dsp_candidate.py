"""Regression tests for an isolated fast vectorized research DSP path."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

pytest.importorskip("torch")

from research import fast_dsp_candidate as fast
from research.candidate_reference import REFERENCE_SPLITS
from research.complete_dsp_candidate import _difference
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES
from sonumio.engines import procedural as legacy


@pytest.mark.parametrize("category", range(len(CATEGORIES)))
def test_fast_layer_single_source_preserves_shape_and_original_semantics(category):
    wave = fast.render_fast_raw((category,), 0)
    original = legacy.generate(CATEGORIES[category], 1.5, 0).samples
    error = _difference(wave, original)
    assert wave.dtype == np.float32 and wave.shape == (fast.RAW_FRAMES,)
    assert np.isfinite(wave).all()
    assert error["max_abs"] < 2e-5
    if CATEGORIES[category] in ("forest", "ambient"):
        np.testing.assert_array_equal(wave, original)


@pytest.mark.parametrize("members", [(0, 1), (0, 5), (2, 3), (4, 5)])
def test_fast_two_layer_envelopes_and_rng_are_compatible(members):
    wave = fast.render_fast_raw(members, 101)
    prompt = " and ".join(CATEGORIES[i] for i in members)
    original = legacy.generate(prompt, 1.5, 101).samples
    assert _difference(wave, original)["max_abs"] < 2e-5


def test_bounded_input_rejections():
    with pytest.raises(ValueError):
        fast.render_fast_raw([0], 0)
    for members, seed in (((0, 0), 0), ((0,), -1), ((0,), True),
                          ((99,), 0), ((0,), 2**32)):
        with pytest.raises(ValueError):
            fast.render_fast_raw(members, seed)
    with pytest.raises(ValueError):
        fast._case((0,), 101, "train")
    with pytest.raises(ValueError):
        fast._case((0,), 0, "unknown")
    with pytest.raises(ValueError):
        fast._case([0], 0, "train")


def test_train_source_crop_is_checked_against_original_oracle():
    row = fast._case((0, 1), 0, "train")
    assert row["offset"] >= 400
    assert len(row["fast_crop_sha256"]) == 64
    assert len(row["original_crop_sha256"]) == 64
    assert row["raw_error"]["max_abs"] < 2e-5
    assert row["crop_error"]["max_abs"] < 2e-5
    assert math.isfinite(row["fast_24k_seconds"]) and row["fast_24k_seconds"] > 0


@pytest.fixture(scope="module")
def smoke_report():
    return fast.build_report("smoke")


def test_smoke_completes_and_reports_independent_recipe(smoke_report):
    fast.validate_report(smoke_report)
    assert len(smoke_report["recordings"]) == 40
    assert smoke_report["splits"]["train"]["recordings"] == 20
    assert smoke_report["splits"]["validation"]["recordings"] == 20
    assert smoke_report["fast_recipe"] == fast.FAST_RECIPE
    assert smoke_report["original_generator_unchanged"] is True
    assert smoke_report["candidate_used_for_training"] is False
    assert smoke_report["audio_files_saved"] is False
    assert smoke_report["model_weights_saved"] is False
    assert fast.matches_scalar_reference(smoke_report) is None
    compared = fast.compare_reports(smoke_report, copy.deepcopy(smoke_report))
    assert compared["recordings"] == 40
    assert compared["identical_candidate_raw_sha"] is True
    assert compared["identical_candidate_cropped_sha"] is True
    assert compared["identical_original_raw_sha"] is True
    assert compared["matches_scalar_reference"] is None


def test_report_rejects_invalid_hash_errors_and_forged_metadata(smoke_report):
    mutated = copy.deepcopy(smoke_report)
    mutated["candidate_used_for_training"] = True
    with pytest.raises(ValueError, match="metadata"):
        fast.validate_report(mutated)
    mutated = copy.deepcopy(smoke_report)
    mutated["recordings"][0]["fast_crop_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="aggregate"):
        fast.validate_report(mutated)
    mutated["recordings"][0]["fast_crop_sha256"] = "g" * 64
    with pytest.raises(ValueError, match="SHA"):
        fast.validate_report(mutated)
    mutated = copy.deepcopy(smoke_report)
    mutated["recordings"][0]["crop_error"]["max_abs"] = float("nan")
    with pytest.raises(ValueError, match="metric"):
        fast.validate_report(mutated)
    mutated = copy.deepcopy(smoke_report)
    mutated["recordings"][0]["seed"] = mutated["recordings"][1]["seed"]
    with pytest.raises(ValueError, match="identity"):
        fast.validate_report(mutated)


def test_split_leakage_is_detected_even_when_aggregate_is_recomputed(smoke_report):
    mutated = copy.deepcopy(smoke_report)
    mutated["recordings"][1]["fast_crop_sha256"] = (
        mutated["recordings"][0]["fast_crop_sha256"]
    )
    mutated["splits"] = fast._aggregate(mutated["recordings"])
    with pytest.raises(ValueError, match="Duplicate waveform"):
        fast.validate_report(mutated)
    mutated = copy.deepcopy(smoke_report)
    validation = next(x for x in mutated["recordings"] if x["split"] == "validation")
    validation["fast_crop_sha256"] = mutated["recordings"][0]["fast_crop_sha256"]
    mutated["splits"] = fast._aggregate(mutated["recordings"])
    with pytest.raises(ValueError, match="Train/validation"):
        fast.validate_report(mutated)


def test_full_reference_is_a_measured_condition_not_a_default_claim(monkeypatch):
    report = copy.deepcopy(smoke_placeholder())
    monkeypatch.setattr(fast, "validate_report", lambda _: None)
    assert fast.matches_scalar_reference(report) is False
    for split, original in REFERENCE_SPLITS.items():
        report["splits"][split]["recordings"] = original["recordings"]
        report["splits"][split]["fast_raw_sha256"] = original["candidate_raw_sha256"]
        report["splits"][split]["fast_8k_crop_sha256"] = (
            original["candidate_8k_dataset_sha256"]
        )
    assert fast.matches_scalar_reference(report) is True


def smoke_placeholder():
    return {
        "scope": "full",
        "splits": {
            split: {"recordings": 0, "fast_raw_sha256": "0" * 64,
                    "fast_8k_crop_sha256": "0" * 64}
            for split in REFERENCE_SPLITS
        },
    }


def test_complete_grid_expected_distinct_recordings():
    assert len(SYNTHETIC_COMBINATIONS) == 21
    assert len(fast._identities("full")) == 84
