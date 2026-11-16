"""Schedule ablation must compare paired seeds without publishing model artifacts."""
from __future__ import annotations

import copy
import json
import math

import pytest

torch = pytest.importorskip('torch')

from research.schedule_ablation import (
    PROTOCOL, WAVE_METRICS, _metrics, _validate_study, ablate,
)
from research.multi_seed_study import PROTOCOL as STUDY_PROTOCOL


def _fake_report(profile: str, seeds: tuple[int, ...]) -> dict:
    cosine = profile == 'cosine_zero_terminal_v1'
    records = []
    for seed in seeds:
        waves = []
        for category in ('rain', 'wind'):
            for language in ('en', 'hi'):
                stats = {
                    'sha256': 'a' * 64,
                    'rms': 0.2 if cosine else 0.3,
                    'peak': 0.6 if cosine else 1.0,
                    'clipping_fraction': 0.01 if cosine else 0.15,
                    'near_silence_fraction': 0.0,
                    'spectral_centroid_hz': 1000.0,
                    'low_frequency_energy_fraction': 0.5,
                    'spectral_flatness': 0.3,
                }
                waves.append({'category': category, 'language': language,
                              'stats': stats})
        contrasts = [
            {'language': lang, 'mean_absolute_difference': 0.02,
             'waveform_cosine': 0.9}
            for lang in ('en', 'hi')
        ]
        contrasts.extend([
            {'category': cat, 'language': 'en-vs-hi',
             'mean_absolute_difference': 0.02, 'waveform_cosine': 0.9}
            for cat in ('rain', 'wind')
        ])
        audit = {
            'seeds': [seed + 1500], 'audio_files_saved': False,
            'neural_weights_saved': False,
            'generated': {'raw': copy.deepcopy(waves), 'latent': copy.deepcopy(waves)},
            'paired_prompt_contrasts': {
                'raw': copy.deepcopy(contrasts),
                'latent': copy.deepcopy(contrasts),
            },
        }
        records.append({
            'shared_parameters': {
                'seed': seed, 'schedule_profile': profile,
                'codec_pretraining_steps': 1, 'denoiser_updates_per_arm': 1,
            },
            'train_eval_exact_byte_overlap': 0,
            'raw': {'parameters': 123}, 'latent': {'parameters': 456},
            'codec': {'trainable_parameters': 9},
            'generation_audit': audit,
        })
    return {
        'protocol': STUDY_PROTOCOL,
        'seeds': list(seeds), 'seed_count': len(seeds),
        'generation_audit_enabled': True, 'weights_saved': False,
        'weights_published': False, 'model_generation': None,
        'diffusion_schedule': {
            'profile': profile,
            'terminal_alpha_bar': 0.00003747 if cosine else 0.9222122,
            'gaussian_prior_mismatch_risk': not cosine,
        },
        'shared_data': {
            'data_recipe': 'owned-dsp', 'train_fingerprint': 'a',
            'validation_fingerprint': 'b',
        },
        'individual_runs': records,
        'configuration_sha256': ('c' if cosine else 'd') * 64,
        'mean_and_variability_by_model_and_noise_band': {},
    }


def test_ablation_aggregates_paired_generation_metrics_without_model_selection():
    observed = []
    def runner(**kwargs):
        observed.append(kwargs)
        return _fake_report(kwargs['schedule_profile'], kwargs['seeds'])
    summary = ablate(seeds=(17, 29, 41), codec_steps=1, denoiser_steps=1,
                     evaluation_combinations=2, study_runner=runner)
    assert summary['protocol'] == PROTOCOL
    assert len(observed) == 2
    assert {x['schedule_profile'] for x in observed} == {
        'linear_beta_v1', 'cosine_zero_terminal_v1'}
    assert all(x['generation_probe'] is True for x in observed)
    assert summary['paired_seeds'] == [17, 29, 41]
    assert len(summary['paired_seed_metrics']) == 3
    assert len(summary['configuration_sha256']) == 64
    assert summary['linear_schedule']['terminal_alpha_bar'] > 0.9
    assert summary['cosine_schedule']['terminal_alpha_bar'] < 1e-3
    for arm in ('raw', 'latent'):
        clipping = summary['paired_schedule_difference_summary'][arm]['clipping_fraction']
        assert clipping['count'] == 3
        assert clipping['mean'] == pytest.approx(-0.14)
        assert clipping['negative_runs'] == 3
    assert summary['model_weights_written'] is False
    assert summary['generated_audio_written'] is False
    assert summary['release_authorized'] is False
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize('mutation', [
    lambda r: r.update({'weights_saved': True}),
    lambda r: r.update({'generation_audit_enabled': False}),
    lambda r: r.update({'seeds': [29, 17]}),
    lambda r: r['diffusion_schedule'].update({'profile': 'linear_beta_v1'}),
    lambda r: r['diffusion_schedule'].update({'terminal_alpha_bar': float('nan')}),
    lambda r: r['individual_runs'][0]['generation_audit'].update({'audio_files_saved': True}),
    lambda r: r['individual_runs'][0]['generation_audit']['generated']['raw'][0]['stats'].update(
        {'clipping_fraction': -0.1}),
    lambda r: r['individual_runs'][0]['shared_parameters'].update(
        {'denoiser_updates_per_arm': 2}),
])
def test_reject_uncontrolled_or_unsafe_ablation(mutation):
    def runner(**kwargs):
        sample = _fake_report(kwargs['schedule_profile'], kwargs['seeds'])
        if kwargs['schedule_profile'] == 'cosine_zero_terminal_v1':
            mutation(sample)
        return sample
    with pytest.raises(ValueError):
        ablate(seeds=(17, 29), codec_steps=1, denoiser_steps=1,
               evaluation_combinations=2, study_runner=runner)


def test_failed_or_missing_counterfactual_prompt_is_not_scored():
    x = _fake_report('linear_beta_v1', (17, 29))
    x['individual_runs'][0]['generation_audit']['paired_prompt_contrasts']['raw'] = [
        {'category': 'rain', 'language': 'en-vs-hi',
         'mean_absolute_difference': 0.1, 'waveform_cosine': 0.9}
    ] * 4
    with pytest.raises(ValueError, match='rain-vs-wind'):
        _metrics(x['individual_runs'][0], 'raw')


def test_actual_two_seed_one_step_ablation_runs_without_weights():
    # Tiny integration-only experiment: never treat its quality as a result.
    result = ablate(seeds=(17, 29), codec_steps=1, denoiser_steps=1,
                    schedule_steps=8, channels=16, evaluation_combinations=2)
    assert result['linear_schedule']['terminal_alpha_bar'] > 0.9
    assert result['cosine_schedule']['terminal_alpha_bar'] < 1e-3
    assert len(result['paired_seed_metrics']) == 2
    assert not result['model_weights_written'] and not result['generated_audio_written']
    for arm in ('raw', 'latent'):
        for metric in WAVE_METRICS:
            assert math.isfinite(
                result['paired_schedule_difference_summary'][arm][metric]['mean'])
