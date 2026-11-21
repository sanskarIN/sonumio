"""Unit checks for narrow aligned-waveform train/evaluation leakage detection."""
import numpy as np
import pytest

from research.aligned_leakage import AlignedLeakageGuard
from research.dataset_manifest import ManifestError


def random_audio(seed=0):
    return np.random.default_rng(seed).uniform(-0.8, 0.8, 2048).astype(np.float32)


def add(guard, samples, split="train", line=1, piece=1):
    return guard.add(samples, split=split, line=line, piece=piece)


def test_gain_and_dc_edits_are_rejected():
    source = random_audio()
    edited = (source * 0.45 + 0.1).astype(np.float32)
    guard = AlignedLeakageGuard(frames=2048)
    add(guard, source)
    with pytest.raises(ManifestError, match="gain/DC/polarity-invariant"):
        add(guard, edited, "validation", 2)
    assert guard.indexed_clips == 1


def test_polarity_inversion_is_rejected():
    source = random_audio(2)
    guard = AlignedLeakageGuard(frames=2048)
    add(guard, source)
    with pytest.raises(ManifestError, match="correlation"):
        add(guard, -source, "test", 3)


def test_small_noise_edit_is_rejected():
    source = random_audio(7)
    noise = np.random.default_rng(9).normal(0, 0.001, 2048).astype(np.float32)
    guard = AlignedLeakageGuard(frames=2048)
    add(guard, source)
    with pytest.raises(ManifestError, match="overlap"):
        add(guard, source + noise, "validation", 2)


def test_different_recordings_and_same_split_are_allowed():
    guard = AlignedLeakageGuard(frames=2048)
    add(guard, random_audio(0))
    add(guard, random_audio(1), "validation", 2)
    add(guard, random_audio(0), "train", 3)
    assert guard.indexed_clips == 3


def test_time_shift_not_claimed_to_be_detected():
    source = random_audio()
    guard = AlignedLeakageGuard(frames=2048)
    add(guard, source)
    add(guard, np.roll(source, 101), "validation", 2)
    assert guard.indexed_clips == 2


def test_nearly_constant_clip_rejected():
    guard = AlignedLeakageGuard(frames=2048)
    with pytest.raises(ManifestError, match="nearly constant"):
        add(guard, np.ones(2048, dtype=np.float32))


def test_invalid_array_rejected():
    guard = AlignedLeakageGuard(frames=2048)
    for array in (np.zeros(2030, dtype=np.float32),
                  np.zeros(2048, dtype=np.float64),
                  np.full(2048, np.nan, dtype=np.float32)):
        with pytest.raises(ManifestError, match="finite float32"):
            add(guard, array)


def test_bounded_candidate_fanout_fails_closed():
    guard = AlignedLeakageGuard(frames=2048, max_candidates=1)
    source = random_audio(3)
    for i in range(2):
        changed = source.copy()
        changed[(100 + i * 250):(300 + i * 250)] *= -1
        changed[guard._positions] = source[guard._positions]
        add(guard, changed, "train", i + 1)
    with pytest.raises(ManifestError, match="candidate budget"):
        add(guard, source, "validation", 3)


def test_constructor_validates_limits():
    for options in ({"frames": 30}, {"frames": True},
                    {"frames": 2048, "threshold": 1.0},
                    {"frames": 2048, "max_candidates": 0}):
        with pytest.raises(ValueError):
            AlignedLeakageGuard(**options)
