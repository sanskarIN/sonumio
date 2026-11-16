"""Velocity target algebra, stochastic DDPM posterior and strict experiment tests."""
from __future__ import annotations

import copy
import math

import pytest

torch = pytest.importorskip('torch')

from research.velocity_prediction import (
    velocity_target, decode_velocity, posterior_v_step, sample_velocity,
)
from research.velocity_ablation import (
    PROTOCOL, _check_settings, _record_unclipped, _reduce_single,
    run_ablation,
)
from research.wave_diffusion import DiffusionSchedule
from research.latent_audio import FRAMES, LatentResearchModel
from research.text_audio import TextWaveDenoiser


def test_v_target_exactly_inverts_xt_for_all_schedule_timesteps():
    schedule = DiffusionSchedule(8, profile='cosine_zero_terminal_v1')
    clean = torch.randn(8, 1, 64).clamp(-0.5, 0.5)
    noise = torch.randn_like(clean)
    times = torch.arange(8)
    v = velocity_target(clean, noise, times, schedule)
    noisy = schedule.corrupt(clean, times, noise)
    decoded_clean, decoded_eps = decode_velocity(noisy, v, times, schedule)
    assert torch.allclose(decoded_clean, clean, rtol=1e-5, atol=1e-5)
    assert torch.allclose(decoded_eps, noise, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(v).all()


@pytest.mark.parametrize('timestep', [0, 1, 4, 7])
def test_velocity_posterior_mean_matches_forward_ddpm_ground_truth(timestep):
    schedule = DiffusionSchedule(8, profile='cosine_zero_terminal_v1')
    clean = torch.randn(2, 1, 256).clamp(-0.5, 0.5)
    noise = torch.randn_like(clean)
    t = torch.tensor([timestep, timestep])
    x = schedule.corrupt(clean, t, noise)
    v = velocity_target(clean, noise, t, schedule)
    gen = torch.Generator().manual_seed(16)
    result = posterior_v_step(x, v, timestep, schedule, generator=gen)
    if timestep == 0:
        assert torch.allclose(result, clean, atol=1e-5)
    else:
        alpha = schedule.alphas[timestep]
        beta = schedule.betas[timestep]
        bar = schedule.alpha_bars[timestep]
        prev = schedule.alpha_bars[timestep - 1]
        mean = (beta * prev.sqrt() / (1 - bar) * clean
                + alpha.sqrt() * (1 - prev) / (1 - bar) * x)
        sig = (beta * (1 - prev) / (1 - bar)).sqrt()
        draws = torch.randn(x.shape, generator=torch.Generator().manual_seed(16))
        assert torch.allclose(result, mean + sig * draws, atol=1e-5)
    assert torch.isfinite(result).all()


def test_velocity_sampler_is_deterministic_no_clip_and_does_not_touch_global_rng():
    model = TextWaveDenoiser(16)
    schedule = DiffusionSchedule(8, profile='cosine_zero_terminal_v1')
    model.train()
    before = torch.random.get_rng_state().clone()
    x = sample_velocity(model, schedule, 'rain', seed=19)
    y = sample_velocity(model, schedule, 'rain', seed=19)
    z = sample_velocity(model, schedule, 'rain', seed=20)
    assert model.training
    assert torch.equal(before, torch.random.get_rng_state())
    assert x.shape == (FRAMES,) and torch.isfinite(x).all()
    assert torch.equal(x, y) and not torch.equal(x, z)
    assert x.dtype == torch.float32
    with pytest.raises(ValueError):
        sample_velocity(model, schedule, 'rain', seed=-1)
    with pytest.raises(ValueError):
        sample_velocity(model, schedule, 'rain', guidance=float('nan'))
    with pytest.raises(ValueError):
        sample_velocity(model, schedule, 'rain', latent=True)
    with pytest.raises(ValueError):
        sample_velocity(model, schedule, 'rain', latent=True,
                        codec=LatentResearchModel(16).codec, guidance=5)


def test_latent_velocity_sampling_preserves_training_modes_and_expected_shape():
    model = LatentResearchModel(16)
    model.train()
    schedule = DiffusionSchedule(8, profile='cosine_zero_terminal_v1')
    x = sample_velocity(model.denoiser, schedule, 'बारिश', seed=42,
                        latent=True, codec=model.codec)
    assert x.shape == (FRAMES,) and torch.isfinite(x).all()
    assert model.training and model.codec.training and model.denoiser.training


def test_objective_rejects_invalid_triplet_shape_and_nonfinite():
    schedule = DiffusionSchedule(8, profile='cosine_zero_terminal_v1')
    x = torch.randn(2, 1, 32)
    noise = torch.randn_like(x)
    with pytest.raises(ValueError):
        velocity_target(x, noise[:, :, :-1], torch.tensor([0, 1]), schedule)
    with pytest.raises(ValueError):
        velocity_target(x, noise, torch.tensor([-1, 0]), schedule)
    with pytest.raises(ValueError):
        velocity_target(x, noise, torch.tensor([0, 8]), schedule)
    with pytest.raises(ValueError):
        decode_velocity(x, torch.full_like(x, float('nan')),
                        torch.tensor([0, 1]), schedule)


def test_unclipped_generation_health_records_preclamp_range():
    wave = torch.zeros(FRAMES, dtype=torch.float32)
    wave[:FRAMES // 2] = 2.0
    report = _record_unclipped(wave)
    assert report['preclip_peak'] == 2
    assert report['preclip_out_of_range_fraction'] == 0.5
    assert report['postclip']['clipping_fraction'] == 0.5
    assert report['postclip']['peak'] == 1
    with pytest.raises(ValueError):
        _record_unclipped(torch.full((FRAMES,), float('inf')))
    with pytest.raises(ValueError):
        _record_unclipped(torch.ones(FRAMES) * 1e13)


@pytest.mark.parametrize('bad', [
    {'codec_steps': 0}, {'denoiser_steps': 1001},
    {'schedule_steps': 7}, {'channels': 12},
    {'channels': True}, {'codec_steps': 1.0},
])
def test_training_config_guard(bad):
    config = dict(codec_steps=1, denoiser_steps=1, schedule_steps=8, channels=16)
    with pytest.raises(ValueError):
        _check_settings(**(config | bad))


def test_real_paired_training_smoke_and_validation_on_both_arms():
    # Smoke verifies backward updates on actual original DSP, NOT convergence.
    before = torch.random.get_rng_state().clone()
    report = run_ablation(seeds=(17, 29), codec_steps=1,
                          denoiser_steps=1, schedule_steps=8, channels=16)
    assert torch.equal(before, torch.random.get_rng_state())
    assert report['protocol'] == PROTOCOL
    assert report['seeds'] == [17, 29]
    assert report['configuration']['training_recordings'] == 42
    assert report['configuration']['evaluation_recordings'] == 42
    assert report['paired_model_initialization']
    assert report['paired_batch_timesteps_noise_and_condition_dropout']
    assert not report['model_weights_written']
    assert not report['audio_files_written']
    assert not report['release_authorized']
    assert len(report['per_seed']) == 2
    assert len(report['configuration_sha256']) == 64
    for arm in ('raw', 'latent'):
        results = report['within_arm_difference_summary'][arm]
        assert results['preclip_out_of_range_fraction']['count'] == 2
        assert math.isfinite(results['preclip_peak']['mean'])
        assert math.isfinite(results['postclip_clipping_fraction']['mean'])
        assert math.isfinite(results['rain_vs_wind_mae']['mean'])


def test_reject_forged_paired_model_evidence():
    from research.velocity_ablation import _train_single
    record = _train_single(seed=17, codec_steps=1, denoiser_steps=1,
                           schedule_steps=8, channels=16)
    bad = copy.deepcopy(record)
    bad['identical_initial_parameters_by_arm']['raw'] = False
    with pytest.raises(ValueError, match='Invalid/incompatible'):
        _reduce_single(bad, seed=17, steps=1)
    bad = copy.deepcopy(record)
    bad['arms']['latent']['velocity']['samples'][0]['preclip_peak'] = float('nan')
    with pytest.raises(ValueError, match='Invalid objective'):
        _reduce_single(bad, seed=17, steps=1)
    bad = copy.deepcopy(record)
    bad['audio_files_saved'] = True
    with pytest.raises(ValueError, match='Invalid/incompatible'):
        _reduce_single(bad, seed=17, steps=1)
