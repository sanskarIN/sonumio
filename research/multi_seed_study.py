"""Independent-initialization multi-seed study for Sonumio's experimental models.

Runs the existing same-data raw-vs-latent comparison for several prespecified
random seeds. Keeps all per-seed results, aggregates ONLY metrics within each
representation, and tracks variability/positive-control counts. It deliberately
does NOT rank raw against latent MSE, estimate audio quality, publish weights,
or turn exploratory numbers into a release/model-generation criterion.

Example, from repository root:
  python -m research.multi_seed_study --seeds 17,29,41 --codec-steps 32 --denoiser-steps 64 \
    --evaluation-combinations 8 --output checkpoints/private-research/multi-seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from research.architecture_compare import PROTOCOL as SINGLE_RUN_PROTOCOL, compare

PROTOCOL = 'raw-vs-latent-multi-seed-diagnostics-v2'
MIN_SEEDS = 2
MAX_SEEDS = 8
BENCHMARK_METRICS = (
    'paired_mse',
    'shuffled_mse',
    'unconditional_mse',
    'zero_predictor_mse',
    'shuffled_minus_paired_mse',
    'unconditional_minus_paired_mse',
    'paired_improvement_over_zero_fraction',
)
NOISE_BANDS = ('low', 'medium', 'high')


def parse_seeds(value: str) -> tuple[int, ...]:
    """Require a stable explicit seed list; no implicit random sweep designs."""
    if not isinstance(value, str) or not value:
        raise ValueError('Provide comma-separated seeds such as 17,29,41')
    parts = value.split(',')
    if not MIN_SEEDS <= len(parts) <= MAX_SEEDS:
        raise ValueError(f'Use {MIN_SEEDS}..{MAX_SEEDS} distinct seeds')
    if any(not part.isascii() or not part.isdecimal() or part != str(int(part))
           for part in parts):
        raise ValueError('Seeds must be canonical nonnegative decimal integers')
    seeds = tuple(int(part) for part in parts)
    if len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be unique')
    if any(seed >= 2**32 - 4096 for seed in seeds):
        raise ValueError('Seed outside the model comparison safe range')
    return seeds


def _check_seeds(seeds: tuple[int, ...]) -> None:
    if (type(seeds) is not tuple or not MIN_SEEDS <= len(seeds) <= MAX_SEEDS
            or any(type(seed) is not int or not 0 <= seed < 2**32 - 4096
                   for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('Use a tuple of 2..8 unique bounded integer seeds')


def _bounded_float(value: object, description: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'Non-finite or incorrectly typed benchmark metric: {description}')
    return float(value)


def summarize_values(values: list[float]) -> dict:
    """Descriptive statistics; range is the observed spread, NOT a confidence interval."""
    if len(values) < MIN_SEEDS:
        raise ValueError('At least two runs are required for a seed summary')
    numbers = [_bounded_float(x, 'seed statistic') for x in values]
    return {
        'count': len(numbers),
        'mean': float(statistics.fmean(numbers)),
        'median': float(statistics.median(numbers)),
        'sample_stddev': float(statistics.stdev(numbers)),
        'min': float(min(numbers)),
        'max': float(max(numbers)),
        'positive_runs': sum(x > 0 for x in numbers),
        'zero_runs': sum(x == 0 for x in numbers),
        'negative_runs': sum(x < 0 for x in numbers),
    }


def summarize_runs(runs: list[dict], seeds: tuple[int, ...]) -> dict:
    """Reject schema/experiment drift; never pool raw and latent-domain MSE."""
    _check_seeds(seeds)
    if len(runs) != len(seeds):
        raise ValueError('Runs/seeds count mismatch')
    canonical: tuple | None = None
    for seed, run in zip(seeds, runs):
        if not isinstance(run, dict) or run.get('protocol') != SINGLE_RUN_PROTOCOL:
            raise ValueError('A run has an incompatible single-run benchmark protocol')
        params = run.get('shared_parameters')
        if not isinstance(params, dict) or type(params.get('seed')) is not int:
            raise ValueError('A run is missing its explicit random seed')
        if params['seed'] != seed:
            raise ValueError('Seed/run misalignment')
        signature = (
            run.get('data_recipe'), run.get('train_fingerprint'),
            run.get('validation_fingerprint'), run.get('sample_rate'),
            run.get('audio_frames'), run.get('evaluated_source_combinations'),
            run.get('evaluated_caption_pairs'),
            params.get('codec_pretraining_steps'),
            params.get('denoiser_updates_per_arm'),
            params.get('diffusion_schedule_steps'), params.get('nominal_channels'),
            params.get('training_batch_size_per_arm'), params.get('optimizer'),
            run.get('raw', {}).get('parameters'),
            run.get('latent', {}).get('parameters'),
            run.get('codec', {}).get('trainable_parameters'),
            run.get('hard_negative_tokens_sha256'),
        )
        if any(x is None for x in signature):
            raise ValueError('Missing shared configuration or model metadata')
        if canonical is None:
            canonical = signature
        elif signature != canonical:
            raise ValueError('Runs do not share dataset, architecture or training protocol')
        if (run.get('train_eval_exact_byte_overlap') != 0
                or run.get('category_disjoint_hard_negatives') is not True
                or not isinstance(run.get('hard_negative_tokens_sha256'), str)
                or len(run['hard_negative_tokens_sha256']) != 64
                or run.get('weights_saved') is not False
                or run.get('weights_published') is not False
                or run.get('model_generation') is not None):
            raise ValueError('Unsafe run: possible overlap, saved weights or model generation')
        for arm in ('raw', 'latent'):
            details = run.get(arm)
            if not isinstance(details, dict):
                raise ValueError(f'Missing {arm} arm')
            evaluation = details.get('evaluation', {})
            if set(evaluation.get('noise_levels', {})) != set(NOISE_BANDS):
                raise ValueError('Missing or extra evaluation noise band')
            for level in NOISE_BANDS:
                metrics = evaluation['noise_levels'][level]
                if any(metric not in metrics for metric in BENCHMARK_METRICS):
                    raise ValueError('Evaluation metrics are incomplete')
                for metric in BENCHMARK_METRICS:
                    _bounded_float(metrics[metric], f'{arm}/{level}/{metric}')
                if (metrics['zero_predictor_mse'] <= 0
                        or any(metrics[name] < 0 for name in (
                            'paired_mse', 'shuffled_mse', 'unconditional_mse'))):
                    raise ValueError('Invalid noise-prediction MSE')
                derived = {
                    'shuffled_minus_paired_mse':
                        metrics['shuffled_mse'] - metrics['paired_mse'],
                    'unconditional_minus_paired_mse':
                        metrics['unconditional_mse'] - metrics['paired_mse'],
                    'paired_improvement_over_zero_fraction':
                        1 - metrics['paired_mse'] / metrics['zero_predictor_mse'],
                }
                for name, expectation in derived.items():
                    if not math.isclose(metrics[name], expectation,
                                        rel_tol=1e-5, abs_tol=1e-6):
                        raise ValueError('Inconsistent derived noise diagnostic: ' + name)
    aggregated = {}
    for arm in ('raw', 'latent'):
        aggregated[arm] = {
            level: {
                metric: summarize_values([
                    float(run[arm]['evaluation']['noise_levels'][level][metric])
                    for run in runs
                ]) for metric in BENCHMARK_METRICS
            } for level in NOISE_BANDS
        }
    result = {
        'protocol': PROTOCOL,
        'single_run_protocol': SINGLE_RUN_PROTOCOL,
        'seed_count': len(seeds),
        'seeds': list(seeds),
        'shared_data': {
            'data_recipe': canonical[0],
            'train_fingerprint': canonical[1],
            'validation_fingerprint': canonical[2],
            'sample_rate': canonical[3],
            'audio_frames': canonical[4],
            'evaluated_source_combinations': canonical[5],
            'evaluated_caption_pairs': canonical[6],
            'hard_negative_tokens_sha256': canonical[16],
            'category_disjoint_hard_negatives': True,
        },
        'mean_and_variability_by_model_and_noise_band': aggregated,
        'codec_reconstruction_waveform_mae': summarize_values([
            _bounded_float(run['codec']['eval_waveform_mae'], 'codec reconstruction')
            for run in runs
        ]),
        'individual_runs': runs,
        'weights_saved': False,
        'weights_published': False,
        'model_generation': None,
        'interpretation': {
            'shuffled_minus_paired_mse': 'positive means correct text lowers noise error within this representation',
            'unconditional_minus_paired_mse': 'positive means correct text lowers error versus blank text',
            'paired_improvement_over_zero_fraction': 'positive means paired noise predictor beats its own zero-predictor reference',
            'sample_stddev': 'sample standard deviation across independent initialization seeds only',
            'observed_min_max': 'empirical range only, not a confidence interval for other data',
            'never_compare_cross_domain_absolute_mse': True,
            'audio_quality_claim_supported': False,
            'model_selection_authorized': False,
        },
    }
    # Fingerprint the experiment inputs, not system-dependent timings or output
    # noise values. Sorting keys ensures a stable protocol ID across machines.
    fingerprint_material = {
        'protocol': PROTOCOL,
        'seeds': list(seeds),
        'shared_data': result['shared_data'],
        'steps': [canonical[7], canonical[8], canonical[9]],
        'category_disjoint_negative_sha256': canonical[16],
        'channels': canonical[10],
        'model_parameters': [canonical[13], canonical[14], canonical[15]],
    }
    result['configuration_sha256'] = hashlib.sha256(
        json.dumps(fingerprint_material, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return result


def run_study(*, seeds: tuple[int, ...] = (17, 29, 41),
              codec_steps: int = 2, denoiser_steps: int = 2,
              schedule_steps: int = 8, channels: int = 16,
              evaluation_combinations: int = 4,
              runner: Callable[..., dict] = compare) -> dict:
    """Repeat the same research design with distinct model-initialization seeds.

    The optional injected runner exists only for testing the aggregator;
    normal CLI execution always invokes the actual two-model training code.
    """
    _check_seeds(seeds)
    collected = []
    for seed in seeds:
        report = runner(
            codec_steps=codec_steps, denoiser_steps=denoiser_steps,
            schedule_steps=schedule_steps, seed=seed, channels=channels,
            evaluation_combinations=evaluation_combinations,
        )
        collected.append(report)
    summary = summarize_runs(collected, seeds)
    summary['runtime_environment'] = {
        'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'platform': platform.platform(),
        'cpu_threads': torch.get_num_threads(),
        'cuda_used': False,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=parse_seeds, default=(17, 29, 41))
    parser.add_argument('--codec-steps', type=int, default=2)
    parser.add_argument('--denoiser-steps', type=int, default=2)
    parser.add_argument('--schedule-steps', type=int, default=8)
    parser.add_argument('--channels', type=int, default=16, choices=(16, 24, 32))
    parser.add_argument('--evaluation-combinations', type=int, default=4)
    parser.add_argument('--output', type=Path, default=None, help='Optional local JSON report only; no model weights')
    args = parser.parse_args()
    report = run_study(
        seeds=args.seeds, codec_steps=args.codec_steps,
        denoiser_steps=args.denoiser_steps, schedule_steps=args.schedule_steps,
        channels=args.channels, evaluation_combinations=args.evaluation_combinations,
    )
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized, end='')


if __name__ == '__main__':
    main()
