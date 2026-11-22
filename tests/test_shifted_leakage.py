"""Shift-aware train/evaluation leakage regression tests (research only)."""
from __future__ import annotations

import numpy as np
import pytest

from research.dataset_manifest import ManifestError
from research.shifted_leakage import ShiftedLeakageGuard


def _random(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.8, 0.8, 2048).astype(np.float32)


def _add(guard, audio, split="train", line=1):
    guard.add(audio, split=split, line=line, piece=1)


@pytest.mark.parametrize("shift", [1, 31, 64, 101, 256])
def test_integer_shifted_copies_are_rejected(shift):
    source = _random(7)
    guard = ShiftedLeakageGuard(frames=2048)
    _add(guard, source)
    with pytest.raises(ManifestError, match=rf"shifted .* offset {shift} frames"):
        _add(guard, np.roll(source, shift), "validation", 2)
    assert guard.indexed_clips == 1
    assert guard.compared_pairs >= 1


def test_shifted_gain_dc_and_polarity_edits_are_rejected():
    source = _random(19)
    guard = ShiftedLeakageGuard(frames=2048)
    _add(guard, source)
    edited = (-0.6 * np.roll(source, 97) + 0.13).astype(np.float32)
    with pytest.raises(ManifestError, match=r"shifted .* offset 97 frames"):
        _add(guard, edited, "test", 2)


def test_shifted_small_noise_edit_is_detected():
    source = _random(20)
    guard = ShiftedLeakageGuard(frames=2048)
    _add(guard, source)
    shifted = (np.roll(source, 42)
               + np.random.default_rng(9).normal(0, 0.0003, 2048)).astype(np.float32)
    with pytest.raises(ManifestError, match="shifted .* overlap"):
        _add(guard, shifted, "validation", 2)


def test_distinct_cross_split_and_same_split_are_allowed():
    guard = ShiftedLeakageGuard(frames=2048)
    _add(guard, _random(0))
    _add(guard, _random(1), "validation", 2)
    # Intra-split duplication is handled by separate count/reporting gates.
    _add(guard, np.roll(_random(0), 41), "train", 3)
    assert guard.indexed_clips == 3
    assert guard.indexed_anchors > 0


def test_lag_beyond_bound_is_not_claimed_detected():
    source = _random(0)
    guard = ShiftedLeakageGuard(frames=2048)
    _add(guard, source)
    _add(guard, np.roll(source, 300), "validation", 2)
    assert guard.indexed_clips == 2


def test_low_complexity_anchor_exclusion_is_explicit():
    t = np.arange(2048, dtype=np.float64) / 8000
    sinusoid = np.sin(2 * np.pi * 321 * t).astype(np.float32)
    guard = ShiftedLeakageGuard(frames=2048)
    _add(guard, sinusoid)
    # Approximately five directional turns per 65 frames: below the
    # high-variation anchor requirement. The gate is not universal.
    assert guard.indexed_anchors == 0
    _add(guard, np.roll(sinusoid, 80), "validation", 2)


def test_wrong_dtype_shape_and_nonfinite_arrays_rejected():
    guard = ShiftedLeakageGuard(frames=2048)
    for invalid in (
        np.zeros(100, dtype=np.float32),
        np.zeros(2048, dtype=np.float64),
        np.full(2048, np.nan, dtype=np.float32),
    ):
        with pytest.raises(ManifestError, match="finite float32"):
            _add(guard, invalid)
    assert guard.indexed_clips == 0


def test_index_bucket_budget_fails_without_partial_insertion():
    source = _random(0)
    guard = ShiftedLeakageGuard(frames=2048, max_bucket_items=1)
    _add(guard, source)
    with pytest.raises(ManifestError, match="posting budget"):
        _add(guard, source)
    assert guard.indexed_clips == 1


def test_candidate_budget_fails_closed():
    original = _random(77)
    guard = ShiftedLeakageGuard(frames=2048, max_candidates=1)
    for i in range(2):
        altered = _random(100 + i)
        # Keep a single high-variation fingerprint, but not whole overlap.
        altered[:65] = original[:65]
        _add(guard, altered, "train", i + 1)
    with pytest.raises(ManifestError, match="candidate budget"):
        _add(guard, original, "validation", 3)
    assert guard.indexed_clips == 2


@pytest.mark.parametrize("options", [
    {"frames": True}, {"frames": 64},
    {"frames": 2048, "max_shift_frames": 0},
    {"frames": 2048, "max_shift_frames": 600},
    {"frames": 2048, "threshold": 1.0},
    {"frames": 2048, "max_candidates": 0},
    {"frames": 2048, "max_bucket_items": 0},
])
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError):
        ShiftedLeakageGuard(**options)


def test_many_train_only_clips_do_not_trigger_full_comparisons():
    guard = ShiftedLeakageGuard(frames=2048)
    for seed in range(500):
        _add(guard, _random(seed), "train", seed + 1)
    assert guard.indexed_clips == 500
    assert guard.compared_pairs == 0
    _add(guard, _random(9999), "validation", 501)
    assert guard.indexed_clips == 501
    assert guard.compared_pairs <= guard.max_candidates
