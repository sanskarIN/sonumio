"""Tests for balanced, source-bounded three-way DSP performance diagnostics."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks import dsp_three_way as bench


def test_invalid_sample_controls_fail_closed():
    for bad in (
        {"members": (0, 0)}, {"members": [0]}, {"members": (99,)},
        {"seed": True}, {"seed": -1}, {"seed": 2**32},
        {"repeats": 0}, {"repeats": True}, {"repeats": 10},
        {"warmups": -1}, {"warmups": False}, {"warmups": 4},
    ):
        members = bad.get("members", (0,))
        controls = {name: value for name, value in bad.items()
                    if name != "members"}
        with pytest.raises(ValueError, match="bounded"):
            bench.benchmark_three_way(members, **controls)


def test_balanced_trial_orders_and_independent_ratio_math(monkeypatch):
    order = []
    zero = np.zeros(bench.RAW_FRAMES, dtype=np.float32)
    samples = {variant: zero.copy() for variant in bench.VARIANTS}
    samples["legacy"][3] = 0.00003
    durations = {"legacy": 0.5, "scalar": 10, "fast": 1}

    def fake(variant, members, seed):
        order.append((variant, members, seed))
        return durations[variant], samples[variant].copy()

    monkeypatch.setattr(bench, "_render_one", fake)
    result = bench.benchmark_three_way((0, 1), seed=5, repeats=3, warmups=0)
    assert [a for a, _, _ in order] == [
        "legacy", "scalar", "fast", "scalar", "fast", "legacy",
        "fast", "legacy", "scalar",
    ]
    assert all(members == (0, 1) and seed == 5 for _, members, seed in order)
    assert result["protocol"] == bench.PROTOCOL
    assert result["latency"]["legacy"]["median_seconds"] == 0.5
    assert result["latency"]["fast"]["median_seconds"] == 1
    assert result["latency"]["scalar"]["median_seconds"] == 10
    assert result["fast_vs_legacy_median_latency_ratio"] == 2
    assert result["scalar_over_fast_median_speedup"] == 10
    assert result["scalar_vs_legacy_median_latency_ratio"] == 20
    assert result["fast_equals_scalar_sha256"] is True
    assert result["maximum_sample_error"]["fast_vs_scalar"] == 0
    assert result["maximum_sample_error"]["fast_vs_legacy"] > 0
    assert result["candidate_used_for_neural_training"] is False
    assert result["audio_files_saved"] is False
    assert result["model_weights_saved"] is False


def test_warmups_rotate_and_are_not_reported_as_measurements(monkeypatch):
    observed = []
    audio = np.ones(bench.RAW_FRAMES, dtype=np.float32)
    def fake(variant, members, seed):
        observed.append(variant)
        return 1, audio.copy()
    monkeypatch.setattr(bench, "_render_one", fake)
    data = bench.benchmark_three_way((0,), seed=3, repeats=1, warmups=2)
    assert observed == [
        "legacy", "scalar", "fast",
        "scalar", "fast", "legacy",
        "legacy", "scalar", "fast",
    ]
    assert data["repeats"] == 1
    assert data["warmups_per_variant"] == 2


def test_repeated_identical_seed_hash_drift_is_rejected(monkeypatch):
    seen = 0
    signal = np.zeros(bench.RAW_FRAMES, dtype=np.float32)
    def fake(variant, members, seed):
        nonlocal seen
        wave = signal.copy()
        if variant == "fast":
            seen += 1
            wave[0] = np.float32(seen * 0.0001)
        return 1.0, wave
    monkeypatch.setattr(bench, "_render_one", fake)
    with pytest.raises(ValueError, match="changed between"):
        bench.benchmark_three_way((0,), seed=2, repeats=2, warmups=0)


def test_render_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown"):
        bench._render_one("never", (0,), 0)


def test_actual_first_party_rain_three_way_smoke_has_finite_timing():
    result = bench.benchmark_three_way((0,), seed=0, repeats=1, warmups=0)
    assert result["frames_24k"] == 36000
    assert len(result["latency"]) == 3
    for variant in bench.VARIANTS:
        assert len(result["latency"][variant]["sha256"]) == 64
        assert math.isfinite(result["latency"][variant]["median_seconds"])
        assert result["latency"][variant]["median_seconds"] > 0
    assert result["maximum_sample_error"]["fast_vs_legacy"] < 2e-5
    assert result["maximum_sample_error"]["scalar_vs_legacy"] < 2e-5
    assert result["fast_equals_scalar_sha256"] is True
