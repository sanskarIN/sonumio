"""DSP benchmark methodology and failure-mode coverage; no model training."""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks import dsp_candidate as bench


def test_benchmark_rejects_invalid_workloads_and_noninteger_flags():
    for options in (
        {"members": (0, 0)}, {"members": [0]}, {"members": (99,)},
        {"seed": True}, {"seed": -1}, {"seed": 2**32},
        {"repeats": 0}, {"repeats": 9}, {"repeats": True},
        {"warmups": -1}, {"warmups": 3}, {"warmups": False},
    ):
        members = options.get("members", (0,))
        kwargs = {key: val for key, val in options.items() if key != "members"}
        with pytest.raises(ValueError, match="configuration"):
            bench.benchmark_case(members, **kwargs)


def test_timing_summary_uses_nearest_rank_p95_and_median():
    summary = bench._summary([4.0, 1.0, 3.0, 2.0])
    assert summary["median_seconds"] == 2.5
    assert summary["minimum_seconds"] == 1
    assert summary["maximum_seconds"] == 4
    assert summary["p95_seconds_nearest_rank"] == 4
    assert summary["median_real_time_factor"] == 2.5 / bench.RAW_DURATION
    for times in ([], [0.0], [float("nan")], [float("inf")], [-1.0]):
        with pytest.raises(ValueError):
            bench._summary(times)


def test_output_sha_is_strict_about_finite_shape_and_dtype():
    signal = np.zeros(bench.RAW_FRAMES, dtype=np.float32)
    digest = bench._output_hash(signal)
    assert len(digest) == 64
    for broken in (
        signal.astype(np.float64),
        signal[:-1],
        np.full(bench.RAW_FRAMES, np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            bench._output_hash(broken)


def test_paired_trials_alternate_order_with_no_benchmark_wav(monkeypatch):
    calls = []
    original = np.zeros(bench.RAW_FRAMES, dtype=np.float32)
    candidate = original.copy()
    candidate[0] = np.float32(0.000001)

    def fake(variant, members, seed):
        calls.append((variant, members, seed))
        return (0.25 if variant == "legacy" else 1.5,
                (original if variant == "legacy" else candidate).copy())

    monkeypatch.setattr(bench, "_one_render", fake)
    report = bench.benchmark_case((0, 1), seed=5, repeats=3, warmups=1)
    assert [c[0] for c in calls] == [
        "legacy", "candidate",
        "legacy", "candidate",
        "candidate", "legacy",
        "legacy", "candidate",
    ]
    assert report["protocol"] == bench.PROTOCOL
    assert report["source_categories"] == ["rain", "thunder"]
    assert report["legacy"]["median_seconds"] == 0.25
    assert report["candidate"]["median_seconds"] == 1.5
    assert report["candidate_to_legacy_median_latency_ratio"] == 6
    assert report["max_different_float32_values"] == 1
    assert report["max_absolute_sample_difference"] > 0
    assert report["audio_files_saved"] is False
    assert report["model_weights_saved"] is False
    assert report["candidate_used_for_training"] is False
    assert report["order_alternated"] is True
    assert len(report["legacy"]["exact_sha256"]) == 64
    assert len(report["candidate"]["exact_sha256"]) == 64


def test_nonreproducible_within_host_candidate_is_rejected(monkeypatch):
    original = np.zeros(bench.RAW_FRAMES, dtype=np.float32)
    count = [0]

    def fake(variant, members, seed):
        arr = original.copy()
        if variant == "candidate":
            count[0] += 1
            arr[5] = count[0] * 0.00001
        return 0.25, arr

    monkeypatch.setattr(bench, "_one_render", fake)
    with pytest.raises(ValueError, match="changed between"):
        bench.benchmark_case((0,), repeats=2, warmups=0)


def test_true_first_party_in_memory_candidate_benchmark_smoke():
    report = bench.benchmark_case((0,), seed=0, repeats=1, warmups=0)
    assert report["legacy"]["median_seconds"] > 0
    assert report["candidate"]["median_seconds"] > 0
    assert math.isfinite(report["candidate_to_legacy_median_latency_ratio"])
    assert report["raw_frames"] == 36000
    assert report["max_absolute_sample_difference"] < 2e-5
    assert report["legacy"]["exact_sha256"] != report["candidate"]["exact_sha256"]
    assert not report["audio_files_saved"]
    assert not report["candidate_used_for_training"]
