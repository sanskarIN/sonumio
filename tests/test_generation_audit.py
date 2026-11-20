"""Numerical generation audit must evaluate actual neural waveform samples safely."""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from research.generation_audit import (
    PROTOCOL, generation_audit, sample_raw_full,
    waveform_statistics, _compare_waveforms, sampling_prior_diagnostic,
)
from research.latent_audio import FRAMES, LatentResearchModel
from research.text_audio import TextWaveDenoiser
from research.wave_diffusion import DiffusionSchedule


def test_wave_statistics_have_physical_bounds_and_reject_invalid_audio():
    silence = waveform_statistics(np.zeros(FRAMES, dtype=np.float32))
    assert silence['rms'] == 0 and silence['peak'] == 0
    assert silence['spectral_centroid_hz'] == 0
    assert silence['low_frequency_energy_fraction'] == 0
    assert silence['spectral_flatness'] == 0
    assert silence['near_silence_fraction'] == 1
    sine = np.sin(2 * np.pi * 300 *
                  np.arange(FRAMES, dtype=np.float32) / 8000).astype(np.float32) * 0.2
    metrics = waveform_statistics(sine)
    assert metrics['sha256'] != silence['sha256']
    assert 290 <= metrics['spectral_centroid_hz'] <= 310
    assert 0 < metrics['rms'] < metrics['peak'] <= 1
    assert 0 <= metrics['low_frequency_energy_fraction'] <= 1
    assert 0 <= metrics['spectral_flatness'] <= 1
    assert 0 <= metrics['clipping_fraction'] <= 1
    for bad in (np.zeros(FRAMES - 1, np.float32),
                np.ones((1, FRAMES), np.float32),
                np.full(FRAMES, np.nan, np.float32),
                np.full(FRAMES, 2.0, np.float32),
                np.zeros(FRAMES, np.int16)):
        with pytest.raises(ValueError):
            waveform_statistics(bad)


def test_same_seed_raw_sampling_is_deterministic_restores_model_mode_and_global_rng():
    torch.manual_seed(93)
    model = TextWaveDenoiser(16)
    schedule = DiffusionSchedule(8)
    model.train()
    before = torch.random.get_rng_state().clone()
    first = sample_raw_full(model, schedule, 'rain', seed=22)
    second = sample_raw_full(model, schedule, 'rain', seed=22)
    third = sample_raw_full(model, schedule, 'rain', seed=23)
    assert model.training
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert first.shape == (FRAMES,)
    assert torch.isfinite(first).all() and (first.abs() <= 1).all()
    with pytest.raises(ValueError):
        sample_raw_full(model, schedule, 'rain', seed=-1)
    with pytest.raises(ValueError):
        sample_raw_full(model, schedule, 'rain', seed=True)
    with pytest.raises(ValueError):
        sample_raw_full(model, schedule, 'rain', seed=1, guidance=float('nan'))


def test_prompt_difference_control_identical_audio_is_not_claimed_as_semantics():
    x = torch.zeros(FRAMES)
    y = torch.ones(FRAMES) * 0.1
    equal = _compare_waveforms(x, x)
    changed = _compare_waveforms(x, y)
    assert equal['mean_absolute_difference'] == 0
    assert equal['exact_same_waveform']
    assert changed['mean_absolute_difference'] == pytest.approx(0.1)
    assert not changed['exact_same_waveform']


def test_actual_generation_audit_has_four_prompts_and_two_arms_no_saved_weights():
    torch.manual_seed(33)
    raw = TextWaveDenoiser(16)
    latent = LatentResearchModel(16)
    schedule = DiffusionSchedule(8)
    raw.train()
    latent.train()
    before = torch.random.get_rng_state().clone()
    report = generation_audit(raw, latent, schedule, seeds=(51,), guidance=1)
    assert raw.training and latent.training
    assert torch.equal(before, torch.random.get_rng_state())
    assert report['protocol'] == PROTOCOL
    assert report['sample_rate'] == 8000
    assert report['duration_seconds'] == FRAMES / 8000
    assert report['seeds'] == [51]
    prior = report['diffusion_sampling_prior']
    assert prior['terminal_alpha_bar'] == pytest.approx(0.9222122, rel=1e-5)
    assert prior['gaussian_prior_mismatch_risk'] is True
    assert prior['clean_signal_coefficient'] > prior['gaussian_noise_coefficient']
    assert report['audio_files_saved'] is False
    assert report['neural_weights_saved'] is False
    assert set(report['prompts_by_category']) == {'rain', 'wind'}
    assert set(report['reference_dsp_metrics']) == {'rain', 'wind'}
    for arm in ('raw', 'latent'):
        samples = report['generated'][arm]
        assert len(samples) == 4
        assert {x['language'] for x in samples} == {'en', 'hi'}
        assert len(report['paired_prompt_contrasts'][arm]) == 4
        for case in samples:
            data = case['stats']
            assert len(data['sha256']) == 64
            assert all(math.isfinite(v) for k, v in data.items() if k != 'sha256')
            assert 0 <= data['peak'] <= 1.0001
        for x in report['paired_prompt_contrasts'][arm]:
            assert x['mean_absolute_difference'] >= 0
            assert -1 <= x['waveform_cosine'] <= 1

    # Repeated calls must produce the same wave hashes, not merely the shape.
    repeated = generation_audit(raw, latent, schedule, seeds=(51,))
    for arm in ('raw', 'latent'):
        assert [r['stats']['sha256'] for r in repeated['generated'][arm]] == [
            r['stats']['sha256'] for r in report['generated'][arm]]
    assert raw.training and latent.training


def test_reject_bad_seed_lists_and_guidance_before_generation():
    raw, latent, schedule = TextWaveDenoiser(16), LatentResearchModel(16), DiffusionSchedule(8)
    for seeds in ((51, 51), (), (True,), (2**32,), [51]):
        with pytest.raises(ValueError):
            generation_audit(raw, latent, schedule, seeds=seeds)
    for guidance in (float('inf'), float('nan'), -1, 5, True):
        with pytest.raises(ValueError):
            generation_audit(raw, latent, schedule, guidance=guidance)



def test_short_linear_diffusion_is_not_close_to_pure_gaussian_prior():
    schedule = DiffusionSchedule(8)
    assessment = sampling_prior_diagnostic(schedule)
    assert assessment['sampling_initialization'] == 'pure_unit_gaussian'
    assert assessment['terminal_alpha_bar'] > 0.9
    assert assessment['clean_signal_coefficient'] > 0.95
    assert assessment['gaussian_noise_coefficient'] < 0.3
    assert assessment['gaussian_prior_mismatch_risk']



def test_cosine_prior_audit_has_nearly_gaussian_terminal_distribution():
    reference = sampling_prior_diagnostic(DiffusionSchedule(8))
    candidate = sampling_prior_diagnostic(
        DiffusionSchedule(8, profile='cosine_zero_terminal_v1'))
    assert reference['gaussian_prior_mismatch_risk'] is True
    assert candidate['gaussian_prior_mismatch_risk'] is False
    assert candidate['terminal_alpha_bar'] < 1e-3
    assert candidate['gaussian_noise_coefficient'] > 0.999
