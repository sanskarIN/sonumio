"""Paired multi-seed cosine-vs-linear diffusion training and audio generation audit.

Runs Sonumio's *same* first-party synthetic corpus, English/Hindi prompts,
random initialization seeds, training update counts, model architectures and
in-memory generated-waveform probes for each schedule. Schedule profile is the
only intended intervention. No WAV, checkpoint, weights, releases or tags.

Scores are numerical diagnostics of deliberately tiny DSP models, NOT
perceptual or real-world evidence and NOT permission to release Sonumio-1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

from research.multi_seed_study import (
    PROTOCOL as STUDY_PROTOCOL, _check_seeds, run_study, summarize_values,
)

PROTOCOL = 'paired-cosine-vs-linear-generated-audio-ablation-v1'
PROFILES = ('linear_beta_v1', 'cosine_zero_terminal_v1')
WAVE_METRICS = ('rms', 'peak', 'clipping_fraction', 'near_silence_fraction',
                'spectral_centroid_hz', 'low_frequency_energy_fraction',
                'spectral_flatness')
CONTRAST_METRICS = ('mean_absolute_difference', 'waveform_cosine')


def _validate_study(study: dict, expected: str, seeds: tuple[int, ...]) -> None:
    if not isinstance(study, dict) or study.get('protocol') != STUDY_PROTOCOL:
        raise ValueError('Unexpected or missing replicated-study protocol')
    if (study.get('seeds') != list(seeds)
            or study.get('seed_count') != len(seeds)
            or study.get('generation_audit_enabled') is not True
            or study.get('model_generation') is not None
            or study.get('weights_saved') is not False
            or study.get('weights_published') is not False):
        raise ValueError('Study seeds, generation controls, or publication flags are unsafe')
    schedule = study.get('diffusion_schedule')
    if not isinstance(schedule, dict) or schedule.get('profile') != expected:
        raise ValueError('Incorrect training/sampling profile in study')
    terminal = schedule.get('terminal_alpha_bar')
    if type(terminal) not in (int, float) or not math.isfinite(terminal) or not 0 < terminal < 1:
        raise ValueError('Invalid terminal alpha bar in study')
    if schedule.get('gaussian_prior_mismatch_risk') is not (terminal > 0.1):
        raise ValueError('Invalid terminal prior risk indicator')
    for seed, record in zip(seeds, study.get('individual_runs', [])):
        if (record.get('shared_parameters', {}).get('seed') != seed
                or record['shared_parameters'].get('schedule_profile') != expected
                or record.get('train_eval_exact_byte_overlap') != 0):
            raise ValueError('Individual run is mismatched or contains split leakage')
        audit = record.get('generation_audit')
        if (not isinstance(audit, dict)
                or audit.get('seeds') != [seed + 1500]
                or audit.get('audio_files_saved') is not False
                or audit.get('neural_weights_saved') is not False):
            raise ValueError('Missing or unsafe generated-audio audit')
        for arm in ('raw', 'latent'):
            if (len(audit.get('generated', {}).get(arm, [])) != 4
                    or len(audit.get('paired_prompt_contrasts', {}).get(arm, [])) != 4):
                raise ValueError('Expected four fixed generated prompts and contrasts per arm')
    if len(study.get('individual_runs', [])) != len(seeds):
        raise ValueError('Incorrect number of completed individual runs')


def _unit_float(value: object, *, name: str, lo: float, hi: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'Nonfinite/non-numeric generated metric: {name}')
    if not lo <= value <= hi:
        raise ValueError(f'Out-of-bounds generated metric: {name}')
    return float(value)


def _metrics(record: dict, arm: str) -> dict[str, float]:
    audit = record['generation_audit']
    result = {}
    for metric in WAVE_METRICS:
        values = [
            _unit_float(row['stats'][metric], name=metric,
                        lo=0, hi=4000 if metric == 'spectral_centroid_hz' else 1.0001)
            for row in audit['generated'][arm]
        ]
        result[metric] = sum(values) / len(values)
    # Report *only distinct sound-category* prompt changes (rain vs wind,
    # for English and Hindi). EN-vs-HI comparison is NOT a class change.
    contrast = [
        row for row in audit['paired_prompt_contrasts'][arm]
        if 'language' in row and row['language'] in ('en', 'hi')
        and 'category' not in row
    ]
    if len(contrast) != 2:
        raise ValueError('Missing fixed English/Hindi rain-vs-wind comparisons')
    for metric in CONTRAST_METRICS:
        values = [
            _unit_float(row[metric], name=metric,
                        lo=-1.0001 if metric == 'waveform_cosine' else 0,
                        hi=1.0001 if metric == 'waveform_cosine' else 2.0001)
            for row in contrast
        ]
        result['rain_vs_wind_' + metric] = sum(values) / len(values)
    return result


def ablate(*, seeds: tuple[int, ...] = (17, 29, 41),
           codec_steps: int = 128, denoiser_steps: int = 256,
           schedule_steps: int = 8, channels: int = 16,
           evaluation_combinations: int = 21,
           study_runner: Callable[..., dict] = run_study) -> dict:
    """Train baseline then candidate with preselected matched seeds and data."""
    _check_seeds(seeds)
    settings = dict(seeds=seeds, codec_steps=codec_steps,
                    denoiser_steps=denoiser_steps,
                    schedule_steps=schedule_steps, channels=channels,
                    evaluation_combinations=evaluation_combinations,
                    generation_probe=True)
    observations = {}
    for profile in PROFILES:
        study = study_runner(**settings, schedule_profile=profile)
        _validate_study(study, profile, seeds)
        observations[profile] = study
    baseline, candidate = [observations[name] for name in PROFILES]
    if baseline.get('shared_data') != candidate.get('shared_data'):
        raise ValueError('Cannot compare schedules with different dataset/caption splits')
    if baseline.get('diffusion_schedule') == candidate.get('diffusion_schedule'):
        raise ValueError('Schedule intervention did not alter terminal noise')

    matched = []
    for base, alt in zip(baseline['individual_runs'], candidate['individual_runs']):
        base_params, alt_params = base['shared_parameters'], alt['shared_parameters']
        remaining_base = {k: v for k, v in base_params.items() if k != 'schedule_profile'}
        remaining_alt = {k: v for k, v in alt_params.items() if k != 'schedule_profile'}
        if remaining_base != remaining_alt:
            raise ValueError('Training hyperparameters differ beyond schedule')
        for arm in ('raw', 'latent'):
            if (base[arm]['parameters'] != alt[arm]['parameters']
                    or base['codec']['trainable_parameters'] !=
                    alt['codec']['trainable_parameters']):
                raise ValueError('Model capacity differs across schedules')
        matched.append({'seed': base_params['seed'], 'raw': {}, 'latent': {}})
        for arm in ('raw', 'latent'):
            left, right = _metrics(base, arm), _metrics(alt, arm)
            if set(left) != set(right):
                raise ValueError('Generated-audio metrics do not match')
            for metric in left:
                matched[-1][arm][metric] = {
                    'linear': left[metric], 'cosine': right[metric],
                    'cosine_minus_linear': right[metric] - left[metric],
                }

    paired_summary = {
        arm: {
            metric: summarize_values([
                record[arm][metric]['cosine_minus_linear'] for record in matched
            ]) for metric in matched[0][arm]
        } for arm in ('raw', 'latent')
    }
    result = {
        'protocol': PROTOCOL,
        'paired_seeds': list(seeds),
        'shared_training_data': baseline['shared_data'],
        'shared_configuration': {
            'codec_steps': codec_steps, 'denoiser_steps': denoiser_steps,
            'schedule_steps': schedule_steps, 'channels': channels,
            'evaluation_combinations': evaluation_combinations,
        },
        'linear_schedule': baseline['diffusion_schedule'],
        'cosine_schedule': candidate['diffusion_schedule'],
        'source_config_sha256': {
            profile: observations[profile]['configuration_sha256']
            for profile in PROFILES
        },
        'paired_seed_metrics': matched,
        'paired_schedule_difference_summary': paired_summary,
        # Retain within-model noise metrics; these cannot select audio quality.
        'within_model_denoising': {
            profile: observations[profile]['mean_and_variability_by_model_and_noise_band']
            for profile in PROFILES
        },
        'model_weights_written': False,
        'generated_audio_written': False,
        'release_authorized': False,
        'limitations': [
            'A paired 3-seed experiment on the same synthetic corpus cannot establish real-world fidelity',
            'Terminal Gaussian-prior proximity alone does not guarantee a valid reverse DDPM',
            'Raw and latent noise prediction errors occupy different representations',
            'The eight-step cosine profile has a large terminal beta and may be numerically unstable',
            'Hard clipping can mask even worse unbounded waveform amplitudes',
            'Prompt-change waveforms are not proof of correct sound categories',
            'No subjective listening, rights-reviewed real audio, FAD or CLAP',
        ],
    }
    result['configuration_sha256'] = hashlib.sha256(
        json.dumps({
            'protocol': PROTOCOL, 'seeds': list(seeds),
            'settings': result['shared_configuration'],
            'source_configs': result['source_config_sha256'],
        }, sort_keys=True).encode('utf-8')
    ).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,29,41', type=str)
    parser.add_argument('--codec-steps', default=128, type=int)
    parser.add_argument('--denoiser-steps', default=256, type=int)
    parser.add_argument('--schedule-steps', default=8, type=int)
    parser.add_argument('--channels', default=16, choices=(16, 24, 32), type=int)
    parser.add_argument('--evaluation-combinations', default=21, type=int)
    parser.add_argument('--output', default=None, type=Path)
    args = parser.parse_args()
    from research.multi_seed_study import parse_seeds
    report = ablate(
        seeds=parse_seeds(args.seeds), codec_steps=args.codec_steps,
        denoiser_steps=args.denoiser_steps, schedule_steps=args.schedule_steps,
        channels=args.channels, evaluation_combinations=args.evaluation_combinations)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized, end='')


if __name__ == '__main__':
    main()
