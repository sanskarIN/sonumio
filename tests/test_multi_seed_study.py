"""Multi-seed summary contract, negative controls, and reproducibility tests."""
from __future__ import annotations

import copy
import json
import math
from statistics import stdev

import pytest

torch = pytest.importorskip('torch')

from research.multi_seed_study import (
    PROTOCOL, _check_seeds, parse_seeds, run_study,
    summarize_runs, summarize_values, BENCHMARK_METRICS, NOISE_BANDS,
)
from research.architecture_compare import PROTOCOL as SINGLE_RUN_PROTOCOL


def report_for_seed(seed: int) -> dict:
    """Synthetic summaries exercise aggregation without needing to train."""
    def arm(label: str):
        return {
            'parameters': 101 if label == 'raw' else 202,
            'evaluation': {'noise_levels': {
                band: {
                    'paired_mse': 0.9 + seed / 1000,
                    'shuffled_mse': 1.0 + seed / 1000,
                    'unconditional_mse': 1.2 + seed / 1000,
                    'zero_predictor_mse': 1.0,
                    'shuffled_minus_paired_mse': 0.1,
                    'unconditional_minus_paired_mse': 0.3,
                    'paired_improvement_over_zero_fraction': 0.1 - seed / 1000,
                } for band in NOISE_BANDS
            }},
        }
    return {
        'protocol': SINGLE_RUN_PROTOCOL,
        'data_recipe': 'seeded-window-dsp-latent-v2',
        'train_fingerprint': 'a' * 64,
        'validation_fingerprint': 'b' * 64,
        'sample_rate': 8000, 'audio_frames': 8192,
        'evaluated_source_combinations': 4, 'evaluated_caption_pairs': 16,
        'shared_parameters': {
            'seed': seed, 'codec_pretraining_steps': 2,
            'denoiser_updates_per_arm': 4, 'diffusion_schedule_steps': 8,
            'nominal_channels': 16, 'training_batch_size_per_arm': 2,
            'optimizer': 'AdamW',
        },
        'train_eval_exact_byte_overlap': 0,
        'category_disjoint_hard_negatives': True,
        'hard_negative_tokens_sha256': 'c' * 64,
        'codec': {'trainable_parameters': 33, 'eval_waveform_mae': 0.13 + seed / 1000},
        'raw': arm('raw'), 'latent': arm('latent'),
        'model_generation': None, 'weights_saved': False, 'weights_published': False,
    }


def test_seed_parser_is_explicit_and_rejects_duplicates_noncanonical_numbers():
    assert parse_seeds('17,29,41') == (17, 29, 41)
    for raw in ('', '1', '1,1', '-1,2', '01,2', '1, 2',
                '4294967295,1', '2.0,3', '1,,2', '1,2,3,4,5,6,7,8,9'):
        with pytest.raises(ValueError):
            parse_seeds(raw)
    for bad in ((1,), (1, 1), (True, 2), (0.0, 1), (2**32, 1), [1, 2]):
        with pytest.raises(ValueError):
            _check_seeds(bad)


def test_spread_reports_observed_variability_not_fake_confidence():
    result = summarize_values([-1.0, 0.0, 2.0])
    assert result['mean'] == pytest.approx(1 / 3)
    assert result['median'] == 0.0
    assert result['sample_stddev'] == pytest.approx(stdev([-1, 0, 2]))
    assert result['min'] == -1 and result['max'] == 2
    assert (result['negative_runs'], result['zero_runs'], result['positive_runs']) == (1, 1, 1)
    with pytest.raises(ValueError):
        summarize_values([0.1])
    with pytest.raises(ValueError):
        summarize_values([float('nan'), 1.0])
    with pytest.raises(ValueError):
        summarize_values([True, 1.0])


def test_runs_are_aggregated_only_within_each_representation():
    seeds = (17, 29, 41)
    runs = [report_for_seed(seed) for seed in seeds]
    output = summarize_runs(runs, seeds)
    assert output['protocol'] == PROTOCOL
    assert output['seed_count'] == 3
    assert output['configuration_sha256'] == summarize_runs(runs, seeds)['configuration_sha256']
    assert len(output['configuration_sha256']) == 64
    assert output['model_generation'] is None and not output['weights_published']
    assert output['shared_data']['sample_rate'] == 8000
    assert output['individual_runs'] == runs
    for label in ('raw', 'latent'):
        for noise_band in NOISE_BANDS:
            summary = output['mean_and_variability_by_model_and_noise_band'][label][noise_band]
            assert set(summary) == set(BENCHMARK_METRICS)
            assert summary['paired_mse']['count'] == 3
            assert summary['shuffled_minus_paired_mse']['positive_runs'] == 3
            assert summary['unconditional_minus_paired_mse']['positive_runs'] == 3
    assert output['interpretation']['never_compare_cross_domain_absolute_mse']
    assert not output['interpretation']['audio_quality_claim_supported']
    assert not output['interpretation']['model_selection_authorized']
    assert math.isfinite(output['codec_reconstruction_waveform_mae']['mean'])
    json.dumps(output, allow_nan=False)


@pytest.mark.parametrize('mutation', [
    lambda r: r.update({'protocol': 'wrong-format'}),
    lambda r: r.update({'train_fingerprint': 'different'}),
    lambda r: r.update({'evaluated_caption_pairs': 8}),
    lambda r: r['shared_parameters'].update({'denoiser_updates_per_arm': 8}),
    lambda r: r['shared_parameters'].update({'seed': 999}),
    lambda r: r.update({'train_eval_exact_byte_overlap': 1}),
    lambda r: r.update({'weights_saved': True}),
    lambda r: r['raw']['evaluation']['noise_levels']['low'].update({'paired_mse': float('nan')}),
    lambda r: r['latent']['evaluation']['noise_levels']['medium'].pop('zero_predictor_mse'),
])
def test_reject_schema_drift_leakage_or_bad_metrics(mutation):
    seeds = (17, 29)
    runs = [report_for_seed(x) for x in seeds]
    mutation(runs[1])
    with pytest.raises(ValueError):
        summarize_runs(runs, seeds)


def test_reject_seed_count_mismatch():
    with pytest.raises(ValueError, match='count mismatch'):
        summarize_runs([report_for_seed(17)], (17, 29))


def test_injected_runner_is_called_once_per_seed_with_equal_parameters():
    calls = []
    def stub(**options):
        calls.append(options)
        return report_for_seed(options['seed'])
    summary = run_study(seeds=(17, 29, 41), codec_steps=2, denoiser_steps=4,
                        evaluation_combinations=4, runner=stub)
    assert [x['seed'] for x in calls] == [17, 29, 41]
    assert len(set(tuple(sorted((k, v) for k, v in d.items() if k != 'seed'))
                   for d in calls)) == 1
    assert summary['seed_count'] == 3
    assert summary['runtime_environment']['cuda_used'] is False


def test_no_phantom_model_selection_or_weights_with_different_arm_errors():
    runs = [report_for_seed(17), report_for_seed(29)]
    for arm, new_error in (('raw', 0.1), ('latent', 5.0)):
        metrics = runs[1][arm]['evaluation']['noise_levels']['high']
        metrics['paired_mse'] = new_error
        metrics['shuffled_minus_paired_mse'] = metrics['shuffled_mse'] - new_error
        metrics['unconditional_minus_paired_mse'] = metrics['unconditional_mse'] - new_error
        metrics['paired_improvement_over_zero_fraction'] = (
            1 - new_error / metrics['zero_predictor_mse']
        )
    summary = summarize_runs(runs, (17, 29))
    assert summary['mean_and_variability_by_model_and_noise_band']['raw']['high']['paired_mse']['mean'] < (
        summary['mean_and_variability_by_model_and_noise_band']['latent']['high']['paired_mse']['mean']
    )
    assert summary['interpretation']['never_compare_cross_domain_absolute_mse']
    assert not summary['interpretation']['model_selection_authorized']


def test_reject_arithmetically_inconsistent_derived_advantage():
    runs = [report_for_seed(17), report_for_seed(29)]
    runs[1]['raw']['evaluation']['noise_levels']['low']['shuffled_minus_paired_mse'] = 42.0
    with pytest.raises(ValueError, match='Inconsistent derived'):
        summarize_runs(runs, (17, 29))



def fake_generation_audit(seed: int) -> dict:
    samples = [
        {'category': category, 'language': lang,
         'stats': {'sha256': 'a' * 64,
                   'rms': 0.1 + seed / 10000,
                   'peak': 0.4,
                   'clipping_fraction': 0.0,
                   'near_silence_fraction': 0.1,
                   'spectral_centroid_hz': 200.0}}
        for category in ('rain', 'wind') for lang in ('en', 'hi')
    ]
    contrasts = [{'mean_absolute_difference': 0.02} for _ in range(4)]
    return {
        'protocol': 'in-memory-generated-waveform-numerical-audit-v1',
        'seeds': [seed + 1500], 'sample_rate': 8000,
        'duration_seconds': 1.024,
        'audio_files_saved': False, 'neural_weights_saved': False,
        'generated': {'raw': samples, 'latent': samples},
        'paired_prompt_contrasts': {'raw': contrasts, 'latent': contrasts},
    }


def test_multi_seed_generation_audit_is_aggregated_only_with_matching_safe_runs():
    seeds = (17, 29)
    runs = [report_for_seed(s) for s in seeds]
    for seed, run in zip(seeds, runs):
        run['generation_audit'] = fake_generation_audit(seed)
    summary = summarize_runs(runs, seeds)
    assert summary['generation_audit_enabled'] is True
    assert summary['generated_audio_health_by_model']['raw']['rms']['count'] == 2
    assert summary['generated_prompt_waveform_difference_by_model']['latent']['mean'] == 0.02
    assert not summary['interpretation']['audio_quality_claim_supported']
    incomplete = copy.deepcopy(runs)
    incomplete[1]['generation_audit'] = None
    with pytest.raises(ValueError, match='audited and unaudited'):
        summarize_runs(incomplete, seeds)
    unsafe = copy.deepcopy(runs)
    unsafe[1]['generation_audit']['audio_files_saved'] = True
    with pytest.raises(ValueError, match='unsafe generated'):
        summarize_runs(unsafe, seeds)
    invalid = copy.deepcopy(runs)
    invalid[1]['generation_audit']['generated']['raw'][0]['stats']['peak'] = float('nan')
    with pytest.raises(ValueError):
        summarize_runs(invalid, seeds)


def test_generation_audit_flag_is_forwarded_to_runner_only_when_enabled():
    seen = []
    def fake(**options):
        seen.append(options)
        report = report_for_seed(options['seed'])
        if options.get('generation_probe'):
            report['generation_audit'] = fake_generation_audit(options['seed'])
        return report
    output = run_study(seeds=(17, 29), runner=fake, generation_probe=True)
    assert output['generation_audit_enabled']
    assert len(seen) == 2
    assert all(x['generation_probe'] is True for x in seen)
    with pytest.raises(ValueError, match='boolean'):
        run_study(generation_probe=1, runner=fake)
