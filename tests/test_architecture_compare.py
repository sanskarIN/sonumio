"""Controlled same-data raw vs latent DDPM diagnostics and negative controls."""
import json
import math
import pytest

torch = pytest.importorskip('torch')

from research.architecture_compare import (
    PROTOCOL, _audio_hashes, _validate, _validation_indices,
    compare, evaluate_denoiser,
)
from research.latent_audio import FRAMES, LATENT_CHANNELS, LATENT_FRAMES
from research.text_audio import ByteTextTokenizer
from research.wave_diffusion import DiffusionSchedule


class BlindDenoiser(torch.nn.Module):
    def forward(self, noisy, t, tokens):
        return torch.zeros_like(noisy)


@pytest.mark.parametrize('change', [
    {'codec_steps': 0}, {'codec_steps': True},
    {'denoiser_steps': 0}, {'denoiser_steps': 1001},
    {'schedule_steps': 7}, {'schedule_steps': 1000},
    {'channels': 8}, {'channels': 16.0}, {'seed': -1},
    {'seed': 2**32}, {'evaluation_combinations': 1},
    {'evaluation_combinations': 22},
])
def test_invalid_benchmark_configuration_rejected_before_model_allocation(change):
    valid = dict(codec_steps=1, denoiser_steps=1, schedule_steps=8,
                 channels=16, seed=6, evaluation_combinations=2)
    with pytest.raises(ValueError):
        _validate(**{**valid, **change})


def test_holdout_source_grid_is_deterministic_and_has_two_groups():
    assert _validation_indices(2) == [0, 1, 2, 3, 80, 81, 82, 83]
    assert len(_validation_indices(21)) == 84
    assert _validation_indices(21) == list(range(84))


def test_recording_hashes_count_audio_only_once_for_two_languages():
    x = torch.zeros(4, 1, FRAMES)
    x[2:] = 0.25
    assert len(_audio_hashes(x)) == 2
    x[3] = 0.4
    with pytest.raises(ValueError, match='Bilingual captions'):
        _audio_hashes(x)
    with pytest.raises(ValueError, match='Duplicate'):
        _audio_hashes(torch.zeros(4, 1, FRAMES))


@pytest.mark.parametrize('channels,frames', [(1, FRAMES), (LATENT_CHANNELS, LATENT_FRAMES)])
def test_caption_blind_model_scores_exactly_at_chance(channels, frames):
    model = BlindDenoiser()
    model.train()
    audio = torch.randn(8, channels, frames) * 0.03
    tokens = ByteTextTokenizer().encode([
        'rain', 'बारिश', 'wind', 'हवा',
        'forest birds', 'चिड़िया', 'thunder', 'गरज',
    ])
    rng = torch.random.get_rng_state().clone()
    once = evaluate_denoiser(model, DiffusionSchedule(8), audio, tokens, seed=19)
    again = evaluate_denoiser(model, DiffusionSchedule(8), audio, tokens, seed=19)
    assert once == again
    assert model.training
    assert torch.equal(rng, torch.random.get_rng_state())
    assert once['cases_per_noise_level'] == 8
    for level in ('low', 'medium', 'high'):
        metrics = once['noise_levels'][level]
        assert metrics['shuffled_minus_paired_mse'] == 0
        assert metrics['unconditional_minus_paired_mse'] == 0
        assert metrics['paired_improvement_over_zero_fraction'] == 0
        assert metrics['zero_predictor_mse'] > 0


def test_benchmark_rejects_invalid_grouping_and_seed():
    token = ByteTextTokenizer().encode(['rain'] * 8)
    model = BlindDenoiser()
    schedule = DiffusionSchedule(8)
    with pytest.raises(ValueError):
        evaluate_denoiser(model, schedule, torch.ones(4, 1, FRAMES), token[:4])
    with pytest.raises(ValueError):
        evaluate_denoiser(model, schedule, torch.ones(8, 1, 2000), token)
    with pytest.raises(ValueError):
        evaluate_denoiser(model, schedule, torch.ones(8, 1, FRAMES), token, seed=-1)


def test_one_step_architecture_comparison_reports_honest_metrics():
    before = torch.random.get_rng_state().clone()
    report = compare(codec_steps=1, denoiser_steps=1, schedule_steps=8,
                     channels=16, seed=17, evaluation_combinations=2)
    assert torch.equal(before, torch.random.get_rng_state())
    assert report['protocol'] == PROTOCOL
    assert report['seconds_per_clip'] == FRAMES / 8000
    assert report['train_recordings'] == 42
    assert report['train_caption_pairs'] == 84
    assert report['validation_recordings_total'] == 42
    assert report['validation_caption_pairs_total'] == 84
    assert report['evaluated_source_combinations'] == 2
    assert report['evaluated_caption_pairs'] == 8
    assert report['train_eval_exact_byte_overlap'] == 0
    assert report['train_fingerprint'] != report['validation_fingerprint']
    assert report['codec']['trainable_parameters'] > 0
    assert report['raw']['parameters'] > 0 and report['latent']['parameters'] > 0
    assert report['raw']['parameters'] != report['latent']['parameters']
    assert report['codec']['eval_waveform_mae'] >= 0
    assert not report['weights_saved'] and not report['weights_published']
    assert report['model_generation'] is None
    for name in ('raw', 'latent'):
        arm = report[name]
        assert arm['training_seconds'] >= 0
        assert arm['evaluation_seconds'] >= 0
        assert math.isfinite(arm['training_first_mse'])
        assert math.isfinite(arm['training_last_mse'])
        assert set(arm['evaluation']['noise_levels']) == {'low', 'medium', 'high'}
        for band in arm['evaluation']['noise_levels'].values():
            assert all(math.isfinite(value) for value in band.values())
            assert band['zero_predictor_mse'] > 0
    json.dumps(report, allow_nan=False)


def test_exact_audio_overlap_is_rejected_before_training(monkeypatch):
    from research import architecture_compare as module
    original = module.synthetic_bilingual_waves
    def unsafe_loader(seeds):
        return original((0, 1))
    monkeypatch.setattr(module, 'synthetic_bilingual_waves', unsafe_loader)
    with pytest.raises(ValueError, match='Exact audio overlap'):
        compare(codec_steps=1, denoiser_steps=1, evaluation_combinations=2)
