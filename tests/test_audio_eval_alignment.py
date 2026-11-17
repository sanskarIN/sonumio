"""Absolute delay, sample-aligned overlap and limits for acoustic comparisons."""
from __future__ import annotations

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError
from research.audio_eval.alignment import (
    AlignmentResult, align_pair, estimate_lag,
    score_with_overlap, shift_preserving_length,
)
from research.audio_eval.metrics import sample_abs_error


@pytest.fixture
def impulse_train():
    wave = np.zeros(4096, dtype=np.float32)
    wave[[100, 250, 300, 700, 1200, 1990, 2701, 3070]] = [
        .3, -.5, .8, .6, -.9, .7, -.4, .2
    ]
    return make_buffer(wave, 24000)


def test_positive_shift_delays_candidate_by_exact_frames(impulse_train):
    delayed = shift_preserving_length(impulse_train, 23)
    assert delayed.samples[0, 123] == pytest.approx(.3)
    assert delayed.samples[0, 100] == pytest.approx(0)
    assert estimate_lag(impulse_train, delayed, max_lag_frames=50) == 23


def test_negative_shift_advances_candidate_by_exact_frames(impulse_train):
    advanced = shift_preserving_length(impulse_train, -17)
    assert advanced.samples[0, 83] == pytest.approx(.3)
    assert estimate_lag(impulse_train, advanced, max_lag_frames=50) == -17


def test_same_signal_and_near_silence_return_zero_lag(impulse_train):
    assert estimate_lag(impulse_train, impulse_train) == 0
    quiet = make_buffer(np.zeros(2048, dtype=np.float32), 24000)
    assert estimate_lag(quiet, quiet) == 0
    with pytest.raises(AudioContractError):
        estimate_lag(quiet, quiet, max_lag_frames=9999)


def test_aligned_overlap_excludes_introduced_zero_padding(impulse_train):
    delayed = shift_preserving_length(impulse_train, 31)
    aligned = align_pair(impulse_train, delayed, max_lag_frames=64)
    assert aligned.estimated_lag_frames == 31
    assert aligned.applied_shift_frames == -31
    assert aligned.valid_start == 0
    assert aligned.valid_stop == impulse_train.frames - 31
    assert aligned.overlap_frames == impulse_train.frames - 31
    assert aligned.overlap_fraction < 1
    score = score_with_overlap(
        aligned,
        lambda a, b: sample_abs_error(a, b)["rmse"], name="rmse"
    )
    assert score["value"] == pytest.approx(0)
    assert score["estimated_lag_frames"] == 31
    assert score["not_comparable_to_unaligned_full_length_without_disclosure"] is True


def test_disable_alignment_preserves_reference_lag(impulse_train):
    delayed = shift_preserving_length(impulse_train, 15)
    result = align_pair(impulse_train, delayed,
                        apply_alignment=False, max_lag_frames=32)
    assert result.estimated_lag_frames == 15
    assert result.applied_shift_frames == 0
    assert result.valid_start == 0
    assert result.valid_stop == impulse_train.frames
    assert result.report()["time_stretch_applied"] is False


def test_lag_calculation_uses_same_clock_and_channel_topology(impulse_train):
    other_clock = make_buffer(impulse_train.samples, 44100)
    stereo = make_buffer(np.vstack((impulse_train.samples, impulse_train.samples)),
                         24000)
    with pytest.raises(AudioContractError):
        estimate_lag(impulse_train, other_clock)
    with pytest.raises(AudioContractError):
        estimate_lag(impulse_train, stereo)


def test_alignment_invalid_shifts_are_rejected(impulse_train):
    with pytest.raises(AudioContractError):
        shift_preserving_length(impulse_train, 10000)
    with pytest.raises(AudioContractError):
        shift_preserving_length(impulse_train, True)
    with pytest.raises(AudioContractError):
        align_pair(impulse_train, impulse_train, apply_alignment="yes")


def test_alignment_report_is_transparent_about_moving_samples(impulse_train):
    result = align_pair(impulse_train, impulse_train, max_lag_frames=32)
    details = result.report()
    assert details["original_frames"] == impulse_train.frames
    assert details["valid_overlap_fraction"] == 1.0
    assert details["lag_seconds"] == 0.0
    assert details["silent_zero_padding_applied"] is False
