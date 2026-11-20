"""Sample, spectral and multiresolution acoustic comparison calibration tests."""
from __future__ import annotations

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError
from research.audio_eval.metrics import (
    compare_signals, multi_resolution_distance, normalized_correlation,
    sample_abs_error, si_sdr_db, signal_to_error_ratio_db,
    spectral_convergence, spectral_log_distance,
)
from research.audio_eval.reference import (
    AudioReference, EvaluationPair, RightsDeclaration,
)


@pytest.fixture
def reference():
    t = np.arange(8192, dtype=np.float32) / 24000
    return make_buffer((.5 * np.sin(2 * np.pi * 550 * t)).astype(np.float32), 24000)


@pytest.fixture
def pair(reference):
    rights = RightsDeclaration("first-party-synthetic", "first-party-test-fixture",
                               True, False, True)
    source = AudioReference("osc", reference, "oscillator", "test",
                            "nonverbal", "Original sine", rights)
    return EvaluationPair("osc_vs_copy", source, reference, "first-party-osc")


def test_identical_pcm_has_zero_error_and_finite_capped_perfect_scores(reference):
    sample = sample_abs_error(reference.samples, reference.samples)
    assert sample["different_samples"] == 0
    assert sample["max_abs"] == 0
    assert sample["mean_abs"] == 0
    assert sample["rmse"] == 0
    assert sample["sample_count"] == reference.frames
    assert normalized_correlation(reference.samples, reference.samples) == pytest.approx(1)
    assert signal_to_error_ratio_db(reference.samples, reference.samples) == 160
    assert si_sdr_db(reference.samples, reference.samples) == 160


def test_noise_increases_relative_sample_and_spectral_error(reference):
    rng = np.random.default_rng(456)
    perturbed = reference.copy_with(
        reference.samples + rng.normal(0, .02, reference.samples.shape).astype(np.float32)
    )
    clean = sample_abs_error(reference.samples, reference.samples)
    noisy = sample_abs_error(reference.samples, perturbed.samples)
    assert noisy["rmse"] > clean["rmse"]
    assert noisy["different_samples"] > 0
    assert spectral_log_distance(reference, perturbed, n_fft=256,
                                 hop_length=64) > 0
    assert spectral_convergence(reference, perturbed, n_fft=256,
                                hop_length=64) > 0
    assert normalized_correlation(reference.samples, perturbed.samples) < 1


def test_scale_invariant_sdr_ignores_common_constant_gain(reference):
    louder = reference.samples * 2
    assert si_sdr_db(reference.samples, louder) > 120
    assert signal_to_error_ratio_db(reference.samples, louder) < 5
    assert normalized_correlation(reference.samples, louder) == pytest.approx(1)


def test_stereo_correlation_and_mono_shape_are_explicit(reference):
    stereo = np.vstack((reference.samples, -reference.samples))
    assert normalized_correlation(stereo, stereo) == pytest.approx(1)
    assert si_sdr_db(stereo, stereo) > 120
    with pytest.raises(AudioContractError):
        normalized_correlation(stereo, reference.samples)


def test_silent_metrics_are_finite_not_nan():
    silence = np.zeros(2048, dtype=np.float32)
    noise = np.linspace(-.001, .001, 2048, dtype=np.float32)
    assert si_sdr_db(silence, silence) == 0
    assert si_sdr_db(silence, noise) == -160
    assert signal_to_error_ratio_db(silence, silence) == -160
    assert signal_to_error_ratio_db(silence, noise) == -160
    assert normalized_correlation(silence, silence) == 1
    assert normalized_correlation(silence, noise) == 0


def test_multiresolution_report_is_explicit_not_human_quality(reference):
    modified = reference.copy_with(reference.samples * .9)
    report = multi_resolution_distance(reference, modified,
                                       fft_sizes=(128, 256, 512))
    assert len(report["resolutions"]) == 3
    assert report["mean_log_spectral_distance"] > 0
    assert report["mean_spectral_convergence"] > 0
    assert report["not_a_human_perceptual_rating"] is True
    with pytest.raises(AudioContractError):
        multi_resolution_distance(reference, reference, fft_sizes=(256, 256))


def test_complete_pair_report_has_finite_structured_values(pair):
    result = compare_signals(pair, fft_sizes=(128, 256))
    assert result.case_id == "osc_vs_copy"
    assert result.frames == 8192
    assert result.channels == 1
    assert result.sample_rate == 24000
    assert result.same_bytes is True
    assert result.max_sample_absolute_error == 0
    assert result.multiresolution_log_spectral_distance == 0
    assert result.multiresolution_spectral_convergence == 0
    assert result.as_dict()["human_audio_quality_assessed"] is False
    assert result.as_dict()["rights_verified_automatically"] is False
    assert result.as_dict()["model_training_performed"] is False
    assert len(result.reference_sha256) == len(result.candidate_sha256) == 64


def test_pair_report_changes_when_candidate_pcm_changes(pair):
    modified = pair.candidate.mutable_copy()
    modified[0, 100] *= .5
    changed = EvaluationPair(
        "osc_vs_copy", pair.reference,
        pair.candidate.copy_with(modified), "first-party-osc"
    )
    a = compare_signals(pair, fft_sizes=(128,))
    b = compare_signals(changed, fft_sizes=(128,))
    assert a.same_bytes is True
    assert b.same_bytes is False
    assert b.max_sample_absolute_error > 0
    assert b.candidate_sha256 != a.candidate_sha256


def test_metric_mismatch_and_unbounded_settings_fail_closed(pair):
    shorter = pair.candidate.sliced(0, 100)
    with pytest.raises(AudioContractError):
        spectral_log_distance(pair.reference.buffer, shorter)
    with pytest.raises(AudioContractError):
        sample_abs_error(pair.reference.buffer.samples, shorter.samples)
    with pytest.raises(AudioContractError):
        multi_resolution_distance(pair.reference.buffer, pair.candidate,
                                   fft_sizes=(0,))
