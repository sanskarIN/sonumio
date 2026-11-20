"""Measure caption alignment on held-out synthetic audio with hard negatives."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import fmean

import torch

from research.text_audio import (
    EN_LABELS,
    HI_LABELS,
    SYNTHETIC_COMBINATIONS,
    SYNTHETIC_RECIPE,
    ByteTextTokenizer,
    TextWaveDenoiser,
    _synth_prompts,
    load_checkpoint,
    synthetic_recording,
)
from research.wave_diffusion import DiffusionSchedule


BENCHMARK_VERSION = 'synthetic-caption-retrieval-v1'
RECORDING_SEEDS = (1001, 1002)
NOISE_SEED = 4317
BATCH_SIZE = 16
CAPTION_VARIANTS = (
    ('en', 'training_template'),
    ('hi', 'training_template'),
    ('en', 'heldout_wording'),
    ('hi', 'heldout_wording'),
)


def _hard_negatives(members: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Keep source count fixed and replace exactly one source."""
    sources = set(members)
    return tuple(
        candidate for candidate in SYNTHETIC_COMBINATIONS
        if len(candidate) == len(members) and len(sources.difference(candidate)) == 1
    )


def _caption(members: tuple[int, ...], language: str, style: str, seed: int) -> str:
    if style == 'training_template':
        return _synth_prompts(members, language, seed)
    # Use new framing and reverse the order of two-source descriptions.
    if language == 'en':
        return 'In the background: ' + ' alongside '.join(EN_LABELS[i] for i in reversed(members))
    return 'साथ में ' + ' व '.join(HI_LABELS[i] for i in reversed(members))


@dataclass(frozen=True)
class CaptionScores:
    matched: float
    negatives: tuple[float, ...]
    null: float

    @property
    def pairwise_accuracy(self) -> float:
        return fmean(
            (self.matched < negative) + 0.5 * (self.matched == negative)
            for negative in self.negatives
        )

    @property
    def retrieval_accuracy(self) -> float:
        if self.matched > min(self.negatives):
            return 0.0
        return 1 / (1 + self.negatives.count(self.matched))


@dataclass(frozen=True)
class Observation:
    seed: int
    members: tuple[int, ...]
    language: str
    style: str
    noise_band: str
    scores: CaptionScores


def _summarize(rows: list[Observation]) -> dict[str, int | float]:
    return {
        'cases': len(rows),
        'wrong_caption_comparisons': sum(len(row.scores.negatives) for row in rows),
        'matched_noise_mse': fmean(row.scores.matched for row in rows),
        'wrong_noise_mse': fmean(fmean(row.scores.negatives) for row in rows),
        'null_noise_mse': fmean(row.scores.null for row in rows),
        'mean_mse_advantage': fmean(
            fmean(row.scores.negatives) - row.scores.matched for row in rows
        ),
        'pairwise_accuracy': fmean(row.scores.pairwise_accuracy for row in rows),
        'retrieval_accuracy': fmean(row.scores.retrieval_accuracy for row in rows),
        'retrieval_chance': fmean(1 / (1 + len(row.scores.negatives)) for row in rows),
    }


def _condition_losses(
    model: TextWaveDenoiser, audio: torch.Tensor, noise: torch.Tensor,
    timestep: torch.Tensor, tokens: torch.Tensor,
) -> list[float]:
    errors = []
    for batch in tokens.split(BATCH_SIZE):
        count = len(batch)
        predictions = model(audio.expand(count, -1, -1), timestep.expand(count), batch)
        errors.append((predictions - noise).square().flatten(1).mean(1))
    values = torch.cat(errors)
    if not torch.isfinite(values).all():
        raise ValueError('Alignment benchmark encountered non-finite noise MSE')
    return values.tolist()


@torch.no_grad()
def evaluate(model: TextWaveDenoiser, schedule: DiffusionSchedule) -> dict:
    """Compare each caption with same-language, same-style replacements.

    The model always sees identical corrupted audio and timestep for the
    matched, wrong and null conditions of a case. A higher matched advantage
    or accuracy means better synthetic caption retrieval.
    """
    device = schedule.alpha_bars.device
    generator = torch.Generator().manual_seed(NOISE_SEED)
    steps = {'low': schedule.steps // 8, 'middle': schedule.steps // 2,
             'high': schedule.steps - 1}
    fingerprint = hashlib.sha256(f'{BENCHMARK_VERSION}:{SYNTHETIC_RECIPE}'.encode('ascii'))
    tokenizer = ByteTextTokenizer()
    rows = []
    was_training = model.training
    model.eval()
    try:
        for members in SYNTHETIC_COMBINATIONS:
            candidates = (members, *_hard_negatives(members))
            count = len(candidates)
            for seed in RECORDING_SEEDS:
                clean = synthetic_recording(members, seed).unsqueeze(0)
                noise = torch.randn(clean.shape, generator=generator)
                captions = [tokenizer.encode([
                    _caption(candidate, language, style, seed) for candidate in candidates
                ]) for language, style in CAPTION_VARIANTS]
                tokens = torch.cat([torch.zeros_like(captions[0][:1]), *captions])
                fingerprint.update(bytes(members) + seed.to_bytes(4, 'little'))
                fingerprint.update(clean.numpy().tobytes() + tokens.numpy().tobytes())
                clean, noise, tokens = clean.to(device), noise.to(device), tokens.to(device)
                for band, step in steps.items():
                    timestep = torch.tensor([step], device=device)
                    corrupted = schedule.corrupt(clean, timestep, noise)
                    null, *losses = _condition_losses(model, corrupted, noise, timestep, tokens)
                    for index, (language, style) in enumerate(CAPTION_VARIANTS):
                        matched, *negatives = losses[index * count:(index + 1) * count]
                        rows.append(Observation(seed, members, language, style, band,
                                                CaptionScores(matched, tuple(negatives), null)))
    finally:
        model.train(was_training)

    return {
        'benchmark_version': BENCHMARK_VERSION,
        'synthetic_recipe': SYNTHETIC_RECIPE,
        'dataset_sha256': fingerprint.hexdigest(),
        'recording_seeds': list(RECORDING_SEEDS),
        'noise_seed': NOISE_SEED,
        'timesteps': {
            band: {'step': step, 'noise_variance': float(1 - schedule.alpha_bars[step])}
            for band, step in steps.items()
        },
        'unique_rendered_clips': len(RECORDING_SEEDS) * len(SYNTHETIC_COMBINATIONS),
        'overall': _summarize(rows),
        'by_language_and_style': {
            f'{language}/{style}': _summarize([
                row for row in rows if (row.language, row.style) == (language, style)
            ]) for language, style in CAPTION_VARIANTS
        },
        'by_source_count': {
            str(count): _summarize([row for row in rows if len(row.members) == count])
            for count in (1, 2)
        },
        'by_noise_band': {
            band: _summarize([row for row in rows if row.noise_band == band])
            for band in steps
        },
        'by_recording_seed': {
            str(seed): _summarize([row for row in rows if row.seed == seed])
            for seed in RECORDING_SEEDS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--compare-initialization', action='store_true',
                        help='Score the same architecture with its original random initialization')
    args = parser.parse_args()
    model, schedule, manifest = load_checkpoint(args.checkpoint)
    training = manifest.get('training_report') or {}
    seed = training.get('seed')
    if args.compare_initialization and (type(seed) is not int or not 0 <= seed < 2**32 - 2):
        parser.error('Checkpoint has no valid training seed for the initialization baseline')
    report = {
        'checkpoint': {
            'weights_sha256': manifest['weights_sha256'],
            'training_source': training.get('data_source'),
            'training_recipe': training.get('synthetic_recipe'),
            'training_seed': seed,
            'training_steps': training.get('steps'),
        },
        'trained': evaluate(model, schedule),
    }
    if args.compare_initialization:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            initial = TextWaveDenoiser(model.channels)
        report['initialization'] = evaluate(initial, schedule)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
