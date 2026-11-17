"""Ambient sine variants are separate from the official DSP generator."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

pytest.importorskip("torch")

from research import ambient_sine_candidate as study
from sonumio.engines import procedural as original


@pytest.mark.parametrize("method", study.METHODS)
def test_original_float32_phase_with_alternate_sine_is_finite(method):
    t = np.arange(study.FRAMES, dtype=np.float32) / original.SAMPLE_RATE
    candidate = study.candidate_drone(t, method=method)
    baseline = original._drone(t)
    assert candidate.dtype == np.float32
    assert candidate.shape == baseline.shape
    assert np.isfinite(candidate).all()
    delta = candidate.astype(np.float64) - baseline.astype(np.float64)
    assert np.max(np.abs(delta)) <= 0.005


def test_invalid_methods_nan_and_unsupported_dtype_fail_closed():
    t = np.arange(100, dtype=np.float32) / original.SAMPLE_RATE
    with pytest.raises(ValueError):
        study.candidate_drone(t.astype(np.float64))
    with pytest.raises(ValueError):
        study.candidate_drone(t, method="float32")
    with pytest.raises(ValueError):
        study.candidate_drone(t.reshape(10, 10))
    with pytest.raises(ValueError):
        study.candidate_drone(np.full(100, np.nan, dtype=np.float32))
    with pytest.raises(ValueError):
        study._sin_float32_phase(np.ones(100, dtype=np.float64), method="numpy_float64")
    with pytest.raises(ValueError):
        study._sin_float32_phase(t, method="unknown")


@pytest.fixture(scope="module")
def report():
    return study.build_probe()


def test_both_methods_are_hash_only_and_reference_original(report):
    study.validate_probe(report)
    assert report["protocol"] == study.PROTOCOL
    assert report["frames"] == 36000
    assert report["original_source_unchanged"] is True
    assert report["audio_files_saved"] is False
    assert report["model_weights_saved"] is False
    assert report["candidate_used_for_training"] is False
    assert len(report["methods"]) == 2
    result = study.compare_probes(report, json.loads(json.dumps(report)))
    assert result["same_cpu_model"] is True
    assert result["same_original_bytes"] is True
    assert result["candidate_hashes_match"] == {
        "numpy_float64": True, "python_math_sin": True
    }


def test_candidate_sha_drift_is_observed_not_assumed_equivalent(report):
    changed = copy.deepcopy(report)
    changed["methods"]["numpy_float64"]["sha256"] = "f" * 64
    study.validate_probe(changed)
    compared = study.compare_probes(report, changed)
    assert compared["candidate_hashes_match"]["numpy_float64"] is False
    assert compared["candidate_hashes_match"]["python_math_sin"] is True


def test_reject_fake_training_or_nonfinite_metrics(report):
    changed = copy.deepcopy(report)
    changed["candidate_used_for_training"] = True
    with pytest.raises(ValueError, match="metadata"):
        study.validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["methods"]["python_math_sin"]["rms_absolute_difference"] = float("nan")
    with pytest.raises(ValueError, match="error"):
        study.validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["methods"]["numpy_float64"]["different_float32_values"] = 999999
    with pytest.raises(ValueError, match="error"):
        study.validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["methods"].pop("numpy_float64")
    with pytest.raises(ValueError, match="Incomplete"):
        study.validate_probe(changed)
