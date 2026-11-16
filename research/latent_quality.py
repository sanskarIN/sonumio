"""Leakage-aware objective diagnostics for Sonumio's experimental latent codec.

Synthetic held-out WAV reconstruction is compared with identity, silence and
a deterministic 9-tap moving-average reference. Scores measure *reconstruction*
on paired waveforms, not the quality of newly synthesized DDPM audio.

Sampling is inspected only for shape, finite values, output energy and clipping;
these tests cannot establish naturalness, audible fidelity, prompt adherence,
copyright safety or similarity to real-world recordings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from research.latent_audio import (
    DATA_RECIPE, FRAMES, LATENT_CHANNELS, SAMPLE_RATE, TRAIN_SEEDS,
    LatentResearchModel, DiffusionSchedule, load_checkpoint,
    render_seeded_waveform, sample_latent,
)
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES

EVAL_PROTOCOL = 'paired-synthetic-codec-quality-v1'
FFT_SIZES = (128, 512, 1024)
METHODS = ('codec', 'silence', 'moving_average', 'identity')
LOSS_METRICS = ('mae', 'rmse', 'log_spectral_l1', 'spectral_convergence')
ALL_METRICS = LOSS_METRICS + ('snr_db',)
MAX_BOOTSTRAPS = 2000


def _audio_array(audio: np.ndarray | torch.Tensor, *, name: str) -> np.ndarray:
    """Strict 1.024-second waveform; do not silently truncate or resample."""
    if isinstance(audio, torch.Tensor):
        if audio.device.type != 'cpu':
            raise ValueError(f'{name} must be a CPU tensor')
        audio = audio.detach().numpy()
    if not isinstance(audio, np.ndarray) or audio.shape != (FRAMES,):
        raise ValueError(f'{name} must contain {FRAMES} mono samples')
    if audio.dtype.kind != 'f' or not np.isfinite(audio).all():
        raise ValueError(f'{name} must be finite floating-point audio')
    if np.max(np.abs(audio)) > 1.0001:
        raise ValueError(f'{name} must be in the [-1, 1] audio range')
    return np.asarray(audio, dtype=np.float64)


def waveform_metrics(reference: np.ndarray | torch.Tensor,
                     reconstruction: np.ndarray | torch.Tensor) -> dict[str, float]:
    """Paired, sample-aligned errors; spectral metrics use Hann/STFT magnitudes.

    The SNR has a finite epsilon and is not scale-invariant. Alignment matters.
    No statistic here is an embedding/FAD/CLAP or subjective listening score.
    """
    target = _audio_array(reference, name='reference')
    measured = _audio_array(reconstruction, name='reconstruction')
    if np.max(np.abs(target)) <= 1e-10:
        raise ValueError('Silence-only reference cannot support this quality benchmark')
    difference = target - measured
    target_energy = float(np.square(target).sum())
    error_energy = float(np.square(difference).sum())
    log_magnitude_error, spectral_convergence = [], []
    for n_fft in FFT_SIZES:
        hop = n_fft // 4
        window = np.hanning(n_fft + 1)[:-1]  # periodic Hann, matches torch.hann_window
        a = np.lib.stride_tricks.sliding_window_view(target, n_fft)[::hop] * window
        b = np.lib.stride_tricks.sliding_window_view(measured, n_fft)[::hop] * window
        reference_mag = np.abs(np.fft.rfft(a, axis=-1))
        candidate_mag = np.abs(np.fft.rfft(b, axis=-1))
        log_magnitude_error.append(float(np.mean(np.abs(np.log1p(reference_mag)
                                                         - np.log1p(candidate_mag)))))
        spectral_convergence.append(float(
            np.linalg.norm(reference_mag - candidate_mag)
            / max(float(np.linalg.norm(reference_mag)), 1e-12)
        ))
    result = {
        'mae': float(np.mean(np.abs(difference))),
        'rmse': float(np.sqrt(np.mean(np.square(difference)))),
        'log_spectral_l1': float(np.mean(log_magnitude_error)),
        'spectral_convergence': float(np.mean(spectral_convergence)),
        'snr_db': float(10 * math.log10((target_energy + 1e-12)
                                       / (error_energy + 1e-12))),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite objective reconstruction statistic')
    return result


def _moving_average(audio: np.ndarray) -> np.ndarray:
    """A fixed conventional DSP comparison, NOT a matched-capacity neural model."""
    result = np.convolve(audio.astype(np.float64), np.ones(9) / 9, mode='same')
    return result.astype(np.float32)


def _summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError('At least one recording is required for a quality summary')
    return {method: {
        metric: float(np.mean([r['metrics'][method][metric] for r in rows]))
        for metric in ALL_METRICS
    } for method in METHODS}


def _bootstrap_codec_vs_reference(rows: list[dict], *, iterations: int,
                                  seed: int) -> dict:
    """Resample sound-combination groups; shared-seed clips are not independent.

    This is an exploratory, DSP-family-bound interval, NOT a population
    confidence interval for real-world audio or a test of model superiority.
    """
    clusters: dict[tuple[int, ...], list[dict]] = {}
    for row in rows:
        clusters.setdefault(row['members'], []).append(row)
    keys = sorted(clusters)
    if len(keys) < 2:
        return {
            'available': False,
            'reason': 'at least two different source-combination groups required',
        }
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, object]] = {}
    for baseline in ('silence', 'moving_average'):
        metric_results = {}
        for metric in LOSS_METRICS:
            by_cluster = np.array([
                np.mean([row['metrics']['codec'][metric] -
                         row['metrics'][baseline][metric] for row in clusters[key]])
                for key in keys
            ], dtype=np.float64)
            selections = rng.integers(0, len(keys), size=(iterations, len(keys)))
            estimates = by_cluster[selections].mean(axis=1)
            metric_results[metric] = {
                'codec_minus_baseline': float(by_cluster.mean()),
                'exploratory_cluster_bootstrap_95pct': [
                    float(v) for v in np.quantile(estimates, [0.025, 0.975])
                ],
                'lower_is_better': True,
            }
        out[baseline] = metric_results
    return {
        'available': True,
        'source_combination_groups': len(keys),
        'iterations': iterations, 'seed': seed,
        'warning': 'DSP source combinations are related; this is not a real-world CI',
        'comparisons': out,
    }


@torch.no_grad()
def evaluate_codec(codec: nn.Module, *,
                   seeds: tuple[int, ...] = (101, 102),
                   combinations: tuple[tuple[int, ...], ...] = SYNTHETIC_COMBINATIONS,
                   bootstrap_iterations: int = 200,
                   bootstrap_seed: int = 911) -> dict:
    """Unique audio recordings rather than one row per English/Hindi caption.

    Detects exact waveform overlap against the fixed training recipe.
    Validation seeds may be changed, but training seeds may not be evaluated.
    """
    if (not seeds or len(seeds) > 8 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or not set(seeds).isdisjoint(TRAIN_SEEDS)):
        raise ValueError('Provide 1..8 unique unsigned evaluation seeds disjoint from training')
    if (not combinations or len(combinations) > len(SYNTHETIC_COMBINATIONS)
            or len(set(combinations)) != len(combinations)
            or any(members not in SYNTHETIC_COMBINATIONS for members in combinations)):
        raise ValueError('Provide unique supported one/two-source combinations')
    if type(bootstrap_iterations) is not int or not 1 <= bootstrap_iterations <= MAX_BOOTSTRAPS:
        raise ValueError('bootstrap_iterations must be in 1..2000')
    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed < 2**32:
        raise ValueError('bootstrap_seed must be an unsigned 32-bit integer')
    try:
        parameter = next(codec.parameters())
    except StopIteration:
        parameter = None
    if parameter is not None and parameter.device.type != 'cpu':
        raise ValueError('Deterministic quality benchmark requires a CPU codec')

    original_mode = codec.training
    codec.eval()
    try:
        training_hashes: set[str] = set()
        for members in combinations:
            for seed in TRAIN_SEEDS:
                waveform = render_seeded_waveform(members, seed)
                training_hashes.add(hashlib.sha256(waveform.tobytes()).hexdigest())

        rows = []
        eval_hashes: set[str] = set()
        fingerprint = hashlib.sha256((EVAL_PROTOCOL + ':' + DATA_RECIPE).encode('ascii'))
        for members in combinations:
            for seed in seeds:
                original = render_seeded_waveform(members, seed)
                digest = hashlib.sha256(original.tobytes()).hexdigest()
                if digest in training_hashes:
                    raise ValueError('Exact training/evaluation audio overlap detected')
                if digest in eval_hashes:
                    raise ValueError('Exact duplicate evaluation waveform detected')
                eval_hashes.add(digest)
                fingerprint.update(bytes(members) + seed.to_bytes(4, 'little'))
                fingerprint.update(original.tobytes())
                result = codec(torch.from_numpy(original).view(1, 1, FRAMES))
                if result.shape != (1, 1, FRAMES):
                    raise ValueError('Codec generated a wrong-sized waveform')
                reconstructed = result[0, 0]
                reference = np.asarray(original, dtype=np.float32)
                candidates = {
                    'codec': reconstructed,
                    'silence': np.zeros(FRAMES, dtype=np.float32),
                    'moving_average': _moving_average(reference),
                    'identity': reference,
                }
                rows.append({
                    'members': members, 'seed': seed,
                    'metrics': {
                        name: waveform_metrics(reference, candidate)
                        for name, candidate in candidates.items()
                    },
                })
    finally:
        codec.train(original_mode)
    return {
        'protocol': EVAL_PROTOCOL, 'data_recipe': DATA_RECIPE,
        'evaluation_sha256': fingerprint.hexdigest(),
        'evaluation_seeds': list(seeds),
        'train_seeds_excluded': list(TRAIN_SEEDS),
        'recordings_checked': len(rows),
        'independent_recordings': len(eval_hashes),
        'same_waveform_for_bilingual_captions_counted_once': True,
        'exact_training_waveform_overlaps': 0,
        'objective_metric_directions': {
            **{name: 'lower' for name in LOSS_METRICS}, 'snr_db': 'higher'
        },
        'overall': _summary(rows),
        'by_source_count': {
            str(size): _summary([row for row in rows if len(row['members']) == size])
            for size in (1, 2) if any(len(row['members']) == size for row in rows)
        },
        'by_category_membership': {
            name: {
                'recordings': sum(index in row['members'] for row in rows),
                'metrics': _summary([row for row in rows if index in row['members']]),
            }
            for index, name in enumerate(CATEGORIES)
            if any(index in row['members'] for row in rows)
        },
        'exploratory_intervals': _bootstrap_codec_vs_reference(
            rows, iterations=bootstrap_iterations, seed=bootstrap_seed),
        'interpretation': (
            'Synthetic paired reconstruction only. Identity is a correctness '
            'ceiling, silence and moving average are untrained DSP controls. '
            'No generated-audio quality, human preference, semantics or '
            'rights provenance conclusions follow from these scores.'
        ),
    }


@torch.no_grad()
def generation_health(model: LatentResearchModel, schedule: DiffusionSchedule, *,
                      prompts: tuple[str, ...] = ('rain', 'बारिश और हवा'),
                      seed: int = 20, guidance: float = 1.0) -> dict:
    """Numerical health of generated samples, without semantic quality claims."""
    if not prompts or len(prompts) > 4:
        raise ValueError('Provide 1..4 evaluation prompts')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    output = []
    for prompt in prompts:
        waveform = sample_latent(model, schedule, prompt, seed=seed, guidance=guidance)
        samples = _audio_array(waveform, name='generated wave')
        output.append({
            'prompt': prompt,
            'sample_sha256': hashlib.sha256(
                samples.astype(np.float32).tobytes()).hexdigest(),
            'frames': FRAMES,
            'rms': float(np.sqrt(np.mean(np.square(samples)))),
            'peak': float(np.max(np.abs(samples))),
            'fraction_at_clipping_boundary': float(np.mean(np.abs(samples) >= 0.999)),
            'fraction_near_silence': float(np.mean(np.abs(samples) < 1e-5)),
        })
    return {
        'sample_rate': SAMPLE_RATE, 'clip_seconds': FRAMES / SAMPLE_RATE,
        'seed': seed, 'guidance': guidance, 'samples': output,
        'warning': 'Numerical sample health is not perceptual quality or text alignment',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--bootstrap-iterations', type=int, default=200)
    parser.add_argument('--bootstrap-seed', type=int, default=911)
    parser.add_argument('--no-generation-sanity', action='store_true')
    parser.add_argument('--output', type=Path, default=None,
                        help='Optional local JSON report; no WAV or weights are published')
    args = parser.parse_args()
    model, schedule, manifest = load_checkpoint(args.checkpoint)
    report = {
        'checkpoint_sha256': manifest['weights_sha256'],
        'training_recipe': (manifest.get('training_report') or {}).get('data_recipe'),
        'codec_evaluation': evaluate_codec(
            model.codec, bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed),
        'generation_health': None if args.no_generation_sanity else generation_health(
            model, schedule),
        'weights_published': False,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized, end='')


if __name__ == '__main__':
    main()
