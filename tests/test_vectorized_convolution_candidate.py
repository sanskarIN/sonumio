"""Research controls for a possible faster, NOT YET validated convolution."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

pytest.importorskip("torch")

from research import vectorized_convolution_candidate as study
from research.stable_convolution_candidate import selected_scalar_convolution


@pytest.mark.parametrize("taps", (1, 2, 12, 60, 140, 180))
@pytest.mark.parametrize("mode", ("same", "full"))
def test_vectorized_double_matches_scalar_numerically_not_assumed_bitwise(
    taps, mode
):
    rng = np.random.default_rng(129)
    samples = rng.standard_normal(211).astype(np.float32)
    kernel = rng.uniform(-0.2, 0.2, taps).astype(np.float32)
    vector = study.vectorized_float64_convolution(samples, kernel, mode=mode)
    frames = len(samples) if mode == "same" else len(samples) + taps - 1
    scalar = selected_scalar_convolution(
        samples, kernel, tuple(range(frames)), mode=mode
    )
    assert vector.shape == (frames,)
    assert vector.dtype == np.float32
    np.testing.assert_allclose(vector, scalar, atol=2e-6, rtol=0)


def test_vectorized_rejects_broken_inputs():
    samples = np.ones(40, dtype=np.float32)
    kernel = np.ones(12, dtype=np.float32)
    for x, y, mode in (
        (samples.astype(np.float64), kernel, "same"),
        (samples, kernel.astype(np.float64), "same"),
        (samples, np.ones(41, dtype=np.float32), "full"),
        (samples, kernel, "valid"),
        (samples, np.asarray([np.nan], dtype=np.float32), "same"),
        (np.ones(50001, dtype=np.float32), kernel, "same"),
    ):
        with pytest.raises(ValueError):
            study.vectorized_float64_convolution(x, y, mode=mode)


@pytest.mark.parametrize("source", study.SOURCES)
def test_source_probe_validates_original_input_and_uses_reference_samples(source):
    row = study.probe_one(source, study.SEEDS[0])
    assert row["source"] == source
    assert row["reference_selected_frames"] >= 257
    assert row["vectorized_seconds"] > 0
    assert row["reference_selected_max_abs_error"] <= 2e-5
    assert row["reference_selected_different_values"] >= 0
    assert len(row["vectorized_full_sha256"]) == 64
    assert len(row["input_sha256"]) == 64


@pytest.fixture(scope="module")
def report():
    return study.build_probe()


def test_complete_vectorized_probe_is_only_hashes_and_is_self_consistent(report):
    study.validate_probe(report)
    assert report["protocol"] == study.PROTOCOL
    assert report["original_dsp_unchanged"]
    assert report["vectorized_candidate_for_training"] is False
    assert report["model_weights_saved"] is False
    assert report["audio_files_saved"] is False
    assert len(report["recordings"]) == 16
    assert set(report["sources"]) == set(study.SOURCES)
    for source in study.SOURCES:
        assert report["sources"][source]["cases"] == 4
        assert type(report["sources"][source]["matches_verified_scalar_full_reference"]) is bool
    same = study.compare_probes(report, copy.deepcopy(report))
    assert same["cases"] == 16
    assert same["identical_source_inputs_and_kernels"]
    assert same["identical_vectorized_full_bytes"]
    # Whether matches_scalar_reference_all_sources is true is an empirical
    # question; tests intentionally do not assume this unverified result.
    assert type(same["matches_scalar_reference_all_sources"]) is bool


def test_hash_drift_and_tampering_are_detected(report):
    changed = copy.deepcopy(report)
    changed["vectorized_candidate_for_training"] = True
    with pytest.raises(ValueError, match="metadata"):
        study.validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["recordings"][0]["input_sha256"] = "q" * 64
    with pytest.raises(ValueError, match="SHA"):
        study.validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["recordings"][0]["reference_selected_max_abs_error"] = float("nan")
    with pytest.raises(ValueError, match="diagnostics"):
        study.validate_probe(changed)
    changed = copy.deepcopy(report)
    changed["recordings"][0]["vectorized_full_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="aggregation"):
        study.validate_probe(changed)
    changed["sources"] = study._aggregate(changed["recordings"])
    study.validate_probe(changed)
    comparison = study.compare_probes(report, changed)
    assert comparison["identical_source_inputs_and_kernels"] is True
    assert comparison["identical_vectorized_full_bytes"] is False


def test_reference_hashes_have_valid_source_set_and_are_distinct():
    assert set(study.SCALAR_SOURCE_REFERENCE) == set(study.SOURCES)
    assert len(set(study.SCALAR_SOURCE_REFERENCE.values())) == len(study.SOURCES)
    assert all(len(x) == 64 and set(x) <= set("0123456789abcdef")
               for x in study.SCALAR_SOURCE_REFERENCE.values())
