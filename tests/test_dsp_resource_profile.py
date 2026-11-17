"""Isolated DSP CPU process memory/latency profiler tests."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from benchmarks import dsp_resource_profile as profile


def test_all_research_variants_and_bounded_sound_case_grid():
    assert len(profile.VARIANTS) == 5
    assert profile.CASES["thunder-forest"] == (1, 3)
    assert profile.CASES["wind-forest"] == (2, 3)
    assert profile.CASES["forest-ambient"] == (3, 5)
    assert len(set(profile.VARIANTS)) == len(profile.VARIANTS)


def test_invalid_seed_repeats_warmups_case_and_unknown_variant_rejected():
    for case, seed, repeats, warmups in (
        ("wrong", 41, 1, 0), ("rain", -1, 1, 0),
        ("rain", True, 1, 0), ("rain", 2**32, 1, 0),
        ("rain", 41, 0, 0), ("rain", 41, True, 0),
        ("rain", 41, 10, 0), ("rain", 41, 1, -1),
        ("rain", 41, 1, 4), ("rain", 41, 1, False),
    ):
        with pytest.raises(ValueError, match="bounded"):
            profile._validate_controls(case, seed, repeats, warmups)
    with pytest.raises(ValueError, match="Unknown"):
        profile._renderer("not-an-allowed-variant")


def test_waveform_fingerprints_enforce_fixed_float32_shape_and_finite_samples():
    wave = np.zeros(profile.RAW_FRAMES, dtype=np.float32)
    assert profile._sample_audio(wave) == hashlib.sha256(
        wave.astype("<f4").tobytes()).hexdigest()
    for other in (
        wave.astype(np.float64), wave[:100], wave.reshape(2, -1),
        np.full(profile.RAW_FRAMES, np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError, match="finite"):
            profile._sample_audio(other)


def test_real_original_generator_in_child_only_profile():
    row = profile._child_measure(
        "original", "rain", seed=41, repeats=2, warmups=0
    )
    assert row["variant"] == "original"
    assert row["case"] == "rain"
    assert row["repeats"] == 2
    assert len(row["wave_sha256"]) == 64
    assert row["median_render_seconds"] > 0
    assert math.isfinite(row["p95_seconds_nearest_rank"])
    assert row["maximum_render_seconds"] >= row["minimum_render_seconds"]
    assert row["audio_files_saved"] is False
    assert row["model_weights_saved"] is False
    assert row["candidate_used_for_neural_training"] is False
    if row["total_process_peak_rss_kib"] is not None:
        assert row["total_process_peak_rss_kib"] > 0
        assert row["incremental_highwater_during_work_kib"] >= 0


def test_parent_distributes_only_allowlisted_variants_and_never_writes_audio(monkeypatch):
    called = []
    sha = "a" * 64

    def fake_child(variant, case, seed, repeats, warmups):
        called.append((variant, case, seed, repeats, warmups))
        return {
            "variant": variant, "case": case, "seed": seed,
            "repeats": repeats, "warmups": warmups,
            "wave_sha256": sha, "median_render_seconds": 0.01,
            "total_process_peak_rss_kib": 300000,
            "audio_files_saved": False, "model_weights_saved": False,
            "candidate_used_for_neural_training": False,
        }

    monkeypatch.setattr(profile, "_run_child", fake_child)
    result = profile.profile_case("wind-forest", seed=41, repeats=2, warmups=0)
    assert len(called) == len(profile.VARIANTS)
    assert {item[0] for item in called} == set(profile.VARIANTS)
    assert all(item[1:] == ("wind-forest", 41, 2, 0) for item in called)
    assert result["source_categories"] == ["wind", "forest"]
    assert len(result["variants"]) == 5
    assert result["original_generator_modified"] is False
    assert result["original_data_modified"] is False
    assert result["candidate_used_for_neural_training"] is False
    assert result["audio_files_saved"] is False
    assert result["model_weights_saved"] is False


def test_failure_in_subprocess_is_not_hidden(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Subprocess failed")
    monkeypatch.setattr(profile, "_run_child", fail)
    with pytest.raises(ValueError, match="Subprocess failed"):
        profile.profile_case("rain", repeats=1, warmups=0)
