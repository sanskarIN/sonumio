"""Controlled hard-negative caption retrieval for 1.024s latent diffusion research.

An objective diagnostic of noise prediction, NOT perceptual audio quality.
Matched, wrong and blank captions see exactly the same noisy latent and step.
Research inputs are self-generated; no model weights or datasets are published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

import torch

from research.latent_audio import (
    FRAMES, SAMPLE_RATE, LatentResearchModel, ByteTextTokenizer,
    _downsample_24k_to_8k, dsp_generate, load_checkpoint,
)
from research.text_alignment import (
    CAPTION_VARIANTS, CaptionScores, _caption, _hard_negatives,
)
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import CATEGORIES, DiffusionSchedule

BENCHMARK_VERSION = 'latent-one-second-hard-negatives-v1'
RECORDING_SEEDS = (1001, 1002)
NOISE_SEED = 6331


def _summarize(cases: list[tuple[CaptionScores, int]]) -> dict:
    if not cases:
        raise ValueError('No scored cases for benchmark summary')
    return {
        'cases': len(cases),
        'wrong_caption_comparisons': sum(len(scores.negatives) for scores, _ in cases),
        'pairwise_accuracy': fmean(scores.pairwise_accuracy for scores, _ in cases),
        'top1_retrieval_accuracy': fmean(scores.retrieval_accuracy for scores, _ in cases),
        'top1_uniform_chance': fmean(1 / (1 + len(scores.negatives)) for scores, _ in cases),
        'matched_noise_mse': fmean(scores.matched for scores, _ in cases),
        'wrong_noise_mse': fmean(fmean(scores.negatives) for scores, _ in cases),
        'unconditioned_noise_mse': fmean(scores.null for scores, _ in cases),
        'mean_noise_mse_advantage': fmean(
            fmean(scores.negatives) - scores.matched for scores, _ in cases),
    }


def _noise_mse(model: LatentResearchModel, corrupted: torch.Tensor,
               timestep: torch.Tensor, noise: torch.Tensor,
               token_rows: torch.Tensor) -> list[float]:
    """Measure all candidate captions under identical noisy latent context."""
    output: list[float] = []
    for chunk in token_rows.split(16):
        n = len(chunk)
        predicted = model.denoiser(corrupted.expand(n, -1, -1), timestep.expand(n), chunk)
        error = (predicted - noise).square().flatten(1).mean(1)
        if not torch.isfinite(error).all():
            raise ValueError('Nonfinite benchmark noise error')
        output.extend(float(e) for e in error.cpu())
    return output


@torch.no_grad()
def evaluate(model: LatentResearchModel, schedule: DiffusionSchedule, *,
             seeds: tuple[int, ...] = RECORDING_SEEDS,
             combinations: tuple[tuple[int, ...], ...] = SYNTHETIC_COMBINATIONS,
             noise_seed: int = NOISE_SEED) -> dict:
    """Synthetic-language retrieval, stratified by language/style/noise band.

    Do NOT treat any score as evidence of general text understanding,
    long-form generation, originality or production audio quality.
    """
    if not seeds or len(seeds) > 8 or len(set(seeds)) != len(seeds) or any(
        type(s) is not int or not 0 <= s < 2**32 for s in seeds
    ):
        raise ValueError('Provide 1..8 distinct recording seeds')
    if (not combinations or len(set(combinations)) != len(combinations)
            or any(m not in SYNTHETIC_COMBINATIONS for m in combinations)):
        raise ValueError('Only known unique one/two-source combinations can be tested')
    if type(noise_seed) is not int or not 0 <= noise_seed < 2**32:
        raise ValueError('Noise seed must be a 32-bit unsigned integer')
    device = next(model.parameters()).device
    if device.type != 'cpu' or schedule.alpha_bars.device.type != 'cpu':
        raise ValueError('This reproducible benchmark currently requires CPU')
    generator = torch.Generator().manual_seed(noise_seed)
    timesteps = {'low': schedule.steps // 8, 'medium': schedule.steps // 2,
                 'high': schedule.steps - 1}
    fingerprint = hashlib.sha256(BENCHMARK_VERSION.encode('ascii'))
    tokenizer = ByteTextTokenizer()
    rows: list[tuple[tuple[int, ...], str, str, str, int, CaptionScores]] = []
    was_training = model.training
    model.eval()
    try:
        for members in combinations:
            alternatives = (members, *_hard_negatives(members))
            count = len(alternatives)
            for seed in seeds:
                waveform = dsp_generate(' and '.join(CATEGORIES[i] for i in members),
                                        FRAMES / SAMPLE_RATE, seed).samples
                samples = _downsample_24k_to_8k(waveform)
                if samples.shape != (FRAMES,):
                    raise ValueError('Benchmark waveform is not a 1.024-second clip')
                reference = torch.from_numpy(samples.copy()).view(1, 1, FRAMES)
                latent = model.codec.encode(reference)
                noise = torch.randn(latent.shape, generator=generator)
                captions = [tokenizer.encode([
                    _caption(candidate, language, style, seed) for candidate in alternatives
                ]) for language, style in CAPTION_VARIANTS]
                tokens = torch.cat([torch.zeros_like(captions[0][:1]), *captions])
                fingerprint.update(bytes(members) + seed.to_bytes(4, 'little'))
                fingerprint.update(reference.numpy().tobytes() + tokens.numpy().tobytes())
                for band, step in timesteps.items():
                    t = torch.tensor([step], dtype=torch.long)
                    corrupted = schedule.corrupt(latent, t, noise)
                    unconditional, *losses = _noise_mse(model, corrupted, t, noise, tokens)
                    for i, (language, style) in enumerate(CAPTION_VARIANTS):
                        matched, *negatives = losses[i * count:(i + 1) * count]
                        rows.append((members, language, style, band, seed,
                                     CaptionScores(matched, tuple(negatives), unconditional)))
    finally:
        model.train(was_training)
    metrics = [(scores, seed) for _, _, _, _, seed, scores in rows]
    return {
        'benchmark_version': BENCHMARK_VERSION,
        'dataset_sha256': fingerprint.hexdigest(),
        'recording_seeds': list(seeds), 'noise_seed': noise_seed,
        'unique_rendered_clips': len(seeds) * len(combinations),
        'warnings': ['synthetic-source labels only', 'same DSP family as training',
                     'no perceptual or natural-language quality validation'],
        'overall': _summarize(metrics),
        'by_language_and_wording': {
            f'{language}/{style}': _summarize([
                (scores, seed) for _, lang, wording, _, seed, scores in rows
                if lang == language and wording == style
            ]) for language, style in CAPTION_VARIANTS
        },
        'by_source_count': {
            str(n): _summarize([(scores, seed) for members, _, _, _, seed, scores in rows
                               if len(members) == n]) for n in (1, 2)
            if any(len(members) == n for members in combinations)
        },
        'by_noise_level': {
            band: _summarize([(scores, seed) for _, _, _, level, seed, scores in rows
                              if level == band]) for band in timesteps
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args()
    model, schedule, manifest = load_checkpoint(args.checkpoint)
    report = {
        'checkpoint_sha256': manifest['weights_sha256'],
        'training_recipe': (manifest.get('training_report') or {}).get('data_recipe'),
        'metrics': evaluate(model, schedule),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
