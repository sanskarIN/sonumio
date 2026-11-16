"""Controlled same-data comparison of Sonumio's raw and latent text-DDPM prototypes.

Both branches see the same 1.024-second 8kHz DSP recordings, captions, train
indices, diffusion schedule and optimizer-update counts. The latent branch
requires a separately trained frozen codec, so model sizes and computational
budgets differ; raw and latent noise MSE are NOT comparable numbers.

This is a local diagnostic, not perceptual evaluation, a winner selection,
a release gate, or evidence of a Sonumio-1 model. No weights are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.latent_audio import (
    DATA_RECIPE, FRAMES, LATENT_CHANNELS, LATENT_FRAMES, LatentResearchModel,
    TRAIN_SEEDS, VALIDATION_SEEDS, reconstruction_loss, synthetic_bilingual_waves,
)
from research.text_audio import ByteTextTokenizer, TextWaveDenoiser, _synth_prompts
from research.text_audio import text_noise_loss
from research.latent_audio import latent_noise_loss
from research.text_audio import SYNTHETIC_COMBINATIONS
from research.wave_diffusion import DiffusionSchedule, SAMPLE_RATE

PROTOCOL = 'shared-data-raw-vs-latent-text-ddpm-v2'
MAX_OPTIMIZER_STEPS = 1000
MAX_EVAL_COMBINATIONS = len(SYNTHETIC_COMBINATIONS)


def _validate(*, codec_steps: int, denoiser_steps: int, schedule_steps: int,
              seed: int, channels: int, evaluation_combinations: int) -> None:
    if (type(codec_steps) is not int or not 1 <= codec_steps <= MAX_OPTIMIZER_STEPS
            or type(denoiser_steps) is not int
            or not 1 <= denoiser_steps <= MAX_OPTIMIZER_STEPS):
        raise ValueError('Optimizer steps must be integers in 1..1000')
    if type(schedule_steps) is not int or not 8 <= schedule_steps <= 256:
        raise ValueError('Diffusion schedule must be 8..256 steps')
    if type(seed) is not int or not 0 <= seed < 2**32 - 4096:
        raise ValueError('Seed must be a bounded unsigned 32-bit integer')
    if type(channels) is not int or channels not in (16, 24, 32):
        raise ValueError('Model width must be 16, 24 or 32')
    if (type(evaluation_combinations) is not int or
            not 2 <= evaluation_combinations <= MAX_EVAL_COMBINATIONS):
        raise ValueError('Use 2..21 evaluation source combinations')


def _audio_hashes(audio: torch.Tensor) -> set[str]:
    if audio.ndim != 3 or audio.shape[1:] != (1, FRAMES) or len(audio) % 2:
        raise ValueError('Audio must be paired by English/Hindi recording')
    hashes = set()
    for i in range(0, len(audio), 2):
        if not torch.equal(audio[i], audio[i + 1]):
            raise ValueError('Bilingual captions do not share the same recording')
        digest = hashlib.sha256(audio[i].numpy().tobytes()).hexdigest()
        if digest in hashes:
            raise ValueError('Duplicate recording within an evaluation split')
        hashes.add(digest)
    return hashes


def _validation_indices(combinations: int) -> list[int]:
    """Select distinct sound combinations across the one/two-source grid."""
    selected = np.linspace(0, MAX_EVAL_COMBINATIONS - 1, combinations, dtype=int).tolist()
    if len(set(selected)) != combinations:
        raise ValueError('Evaluation source selection produced duplicates')
    # Dataset order: combination, 2 recording seeds, 2 language captions.
    indices = []
    for combination in selected:
        first = combination * len(VALIDATION_SEEDS) * 2
        indices.extend(range(first, first + len(VALIDATION_SEEDS) * 2))
    return indices


def disjoint_negative_tokens(indices: list[int]) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    """Build genuinely wrong audio-category captions with matched length/language.

    A circular prompt shuffle can be semantically wrong only in part:
    combinations such as (rain) and (rain, wind) share a real sound. Here
    neither a singleton nor either member of a two-source caption may be
    present in the recording being scored.
    """
    if not indices or any(type(index) is not int or not 0 <= index < (
        len(SYNTHETIC_COMBINATIONS) * len(VALIDATION_SEEDS) * 2)
        for index in indices
    ):
        raise ValueError('Invalid held-out caption indices')
    captions, negative_members = [], []
    items_per_combination = len(VALIDATION_SEEDS) * 2
    for row in indices:
        combo_index = row // items_per_combination
        members = SYNTHETIC_COMBINATIONS[combo_index]
        seed = VALIDATION_SEEDS[(row // 2) % len(VALIDATION_SEEDS)]
        language = 'en' if row % 2 == 0 else 'hi'
        options = [
            candidate for candidate in SYNTHETIC_COMBINATIONS
            if len(candidate) == len(members) and set(candidate).isdisjoint(members)
        ]
        if not options:
            raise ValueError('No disjoint counterfactual caption exists')
        counterfactual = options[(combo_index + seed) % len(options)]
        captions.append(_synth_prompts(counterfactual, language, seed))
        negative_members.append(counterfactual)
    return ByteTextTokenizer().encode(captions), tuple(negative_members)


@torch.no_grad()
def evaluate_denoiser(model: nn.Module, schedule: DiffusionSchedule,
                      clean: torch.Tensor, tokens: torch.Tensor, *,
                      seed: int = 991,
                      wrong_tokens: torch.Tensor | None = None) -> dict:
    """Compare prompts within ONE representation against identical noisy data.

    A fixed four-example roll replaces each source-combination's captions
    with different source-combination captions. Paired/blank/wrong/zero use
    identical timesteps and target noise. Raw and latent MSE remain distinct
    units, even when the numerical noise distribution is standard normal.
    """
    if (clean.ndim != 3 or clean.shape[0] < 8 or clean.shape[0] % 4
            or clean.shape[1:] not in ((1, FRAMES), (LATENT_CHANNELS, LATENT_FRAMES))
            or tokens.ndim != 2 or tokens.shape[0] != clean.shape[0]
            or not torch.isfinite(clean).all()):
        raise ValueError('Expected at least two 4-example bilingual source groups')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a 32-bit unsigned integer')
    if next(model.parameters(), None) is not None:
        if next(model.parameters()).device.type != 'cpu':
            raise ValueError('CPU benchmark only')
    if tokens.device.type != 'cpu' or clean.device.type != 'cpu':
        raise ValueError('CPU benchmark only')
    rng = torch.Generator().manual_seed(seed)
    if wrong_tokens is not None and (
        wrong_tokens.shape != tokens.shape or wrong_tokens.dtype != tokens.dtype
        or wrong_tokens.device != tokens.device
    ):
        raise ValueError('Counterfactual caption tokens must match the original batch')
    wrong = tokens.roll(4, 0) if wrong_tokens is None else wrong_tokens
    blank = torch.zeros_like(tokens)
    levels = {'low': max(0, schedule.steps // 8),
              'medium': schedule.steps // 2, 'high': schedule.steps - 1}
    was_training = model.training
    model.eval()
    output = {}
    try:
        for name, index in levels.items():
            t = torch.full((len(clean),), index, dtype=torch.long)
            noise = torch.randn(clean.shape, generator=rng)
            noisy = schedule.corrupt(clean, t, noise)
            matched = model(noisy, t, tokens)
            shuffled = model(noisy, t, wrong)
            unconditional = model(noisy, t, blank)
            results = {
                'paired_mse': float(F.mse_loss(matched, noise)),
                'shuffled_mse': float(F.mse_loss(shuffled, noise)),
                'unconditional_mse': float(F.mse_loss(unconditional, noise)),
                'zero_predictor_mse': float(noise.square().mean()),
            }
            if not all(math.isfinite(v) for v in results.values()):
                raise ValueError('Nonfinite model comparison results')
            results['shuffled_minus_paired_mse'] = (
                results['shuffled_mse'] - results['paired_mse'])
            results['unconditional_minus_paired_mse'] = (
                results['unconditional_mse'] - results['paired_mse'])
            results['paired_improvement_over_zero_fraction'] = (
                1 - results['paired_mse'] / results['zero_predictor_mse'])
            output[name] = results
    finally:
        model.train(was_training)
    return {
        'cases_per_noise_level': len(clean),
        'wrong_caption': (
            'category-disjoint same-count bilingual counterfactual'
            if wrong_tokens is not None else
            '4-row circular shuffle (not guaranteed category-disjoint)'
        ),
        'noise_levels': output,
        'warning': 'Noise MSE is domain-specific; these scores are not audio quality',
    }


def compare(*, codec_steps: int = 2, denoiser_steps: int = 2,
            schedule_steps: int = 8, seed: int = 17, channels: int = 16,
            evaluation_combinations: int = 4) -> dict:
    """Train two experimental models and measure diagnostics without weights.

    Shared training pairs, update count, text encoder architecture, schedule,
    optimizer family, batch selection, and 8 kHz 1.024-second target clips.
    Unmatched: parameter counts, receptive fields, codec cost, tensor sizes,
    and compute. This is a diagnostic, not a controlled winning-model trial.
    """
    _validate(codec_steps=codec_steps, denoiser_steps=denoiser_steps,
              schedule_steps=schedule_steps, seed=seed, channels=channels,
              evaluation_combinations=evaluation_combinations)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        started = perf_counter()
        train_set = synthetic_bilingual_waves(TRAIN_SEEDS)
        validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
        train_hashes = _audio_hashes(train_set.waves)
        validation_hashes = _audio_hashes(validation.waves)
        if train_hashes & validation_hashes:
            raise ValueError('Exact audio overlap between training and evaluation')
        indices = _validation_indices(evaluation_combinations)
        selected_waves = validation.waves[indices]
        selected_tokens = validation.tokens[indices]
        negatives, negative_members = disjoint_negative_tokens(indices)
        if negatives.shape != selected_tokens.shape or torch.equal(negatives, selected_tokens):
            raise ValueError('The hard-negative text control is malformed')
        for row, candidate in zip(indices, negative_members):
            source_members = SYNTHETIC_COMBINATIONS[
                row // (len(VALIDATION_SEEDS) * 2)
            ]
            if len(candidate) != len(source_members) or not set(candidate).isdisjoint(source_members):
                raise ValueError('Counterfactual source categories overlap with true audio')
        model = LatentResearchModel(channels)
        codec = model.codec
        raw = TextWaveDenoiser(channels)
        latent = model.denoiser
        schedule = DiffusionSchedule(schedule_steps)
        sample_selector = torch.Generator().manual_seed(seed + 2)
        initial_codec_loss = 0.0
        optimizer_codec = torch.optim.AdamW(codec.parameters(), lr=0.001)
        codec_started = perf_counter()
        unique_training = train_set.waves[::2]
        codec.train()
        for step in range(codec_steps):
            chosen = torch.randint(len(unique_training), (2,), generator=sample_selector)
            batch = unique_training[chosen]
            optimizer_codec.zero_grad(set_to_none=True)
            loss = reconstruction_loss(batch, codec(batch))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite codec training loss')
            if not step:
                initial_codec_loss = float(loss.detach())
            loss.backward()
            nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer_codec.step()
        codec_training_seconds = perf_counter() - codec_started
        codec.eval()
        with torch.no_grad():
            encoded_training = torch.cat([
                codec.encode(piece) for piece in train_set.waves.split(8)
            ])
            encoded_validation = torch.cat([
                codec.encode(piece) for piece in selected_waves.split(8)
            ])
            codec_reconstruction_mae = float(F.l1_loss(
                codec.decode(encoded_validation), selected_waves
            ))
        optimizer_raw = torch.optim.AdamW(raw.parameters(), lr=0.0005)
        optimizer_latent = torch.optim.AdamW(latent.parameters(), lr=0.0005)
        raw_losses, latent_losses = [], []
        raw_train_seconds = 0.0
        latent_train_seconds = 0.0
        for _ in range(denoiser_steps):
            rows = torch.randint(len(train_set.waves), (2,), generator=sample_selector)
            # Same examples, optimizer settings and update count for both arms.
            raw.train()
            started_step = perf_counter()
            optimizer_raw.zero_grad(set_to_none=True)
            loss = text_noise_loss(raw, schedule,
                                   train_set.waves[rows], train_set.tokens[rows])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite raw denoising loss')
            loss.backward()
            nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
            optimizer_raw.step()
            raw_train_seconds += perf_counter() - started_step
            raw_losses.append(float(loss.detach()))
            latent.train()
            started_step = perf_counter()
            optimizer_latent.zero_grad(set_to_none=True)
            loss = latent_noise_loss(latent, schedule,
                                     encoded_training[rows], train_set.tokens[rows])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite latent denoising loss')
            loss.backward()
            nn.utils.clip_grad_norm_(latent.parameters(), 1.0)
            optimizer_latent.step()
            latent_train_seconds += perf_counter() - started_step
            latent_losses.append(float(loss.detach()))
        started_eval = perf_counter()
        raw_eval = evaluate_denoiser(raw, schedule, selected_waves, selected_tokens,
                                     seed=seed + 991, wrong_tokens=negatives)
        raw_eval_seconds = perf_counter() - started_eval
        started_eval = perf_counter()
        latent_eval = evaluate_denoiser(latent, schedule, encoded_validation,
                                        selected_tokens, seed=seed + 991,
                                        wrong_tokens=negatives)
        latent_eval_seconds = perf_counter() - started_eval
        return {
            'protocol': PROTOCOL,
            'data_recipe': DATA_RECIPE,
            'sample_rate': SAMPLE_RATE,
            'audio_frames': FRAMES,
            'seconds_per_clip': FRAMES / SAMPLE_RATE,
            'latent_shape': [LATENT_CHANNELS, LATENT_FRAMES],
            'train_recordings': len(train_hashes),
            'train_caption_pairs': train_set.examples,
            'validation_recordings_total': len(validation_hashes),
            'validation_caption_pairs_total': validation.examples,
            'evaluated_source_combinations': evaluation_combinations,
            'evaluated_caption_pairs': len(selected_tokens),
            'train_fingerprint': train_set.fingerprint,
            'validation_fingerprint': validation.fingerprint,
            'train_eval_exact_byte_overlap': 0,
            'category_disjoint_hard_negatives': True,
            'hard_negative_tokens_sha256': hashlib.sha256(
                negatives.numpy().tobytes()).hexdigest(),
            'shared_parameters': {
                'denoiser_updates_per_arm': denoiser_steps,
                'codec_pretraining_steps': codec_steps,
                'diffusion_schedule_steps': schedule_steps,
                'nominal_channels': channels,
                'training_batch_size_per_arm': 2,
                'seed': seed, 'optimizer': 'AdamW',
            },
            'codec': {
                'trainable_parameters': sum(p.numel() for p in codec.parameters()),
                'first_reconstruction_objective': initial_codec_loss,
                'eval_waveform_mae': codec_reconstruction_mae,
                'training_seconds': round(codec_training_seconds, 4),
            },
            'raw': {
                'parameters': sum(p.numel() for p in raw.parameters()),
                'training_first_mse': raw_losses[0],
                'training_last_mse': raw_losses[-1],
                'training_seconds': round(raw_train_seconds, 4),
                'evaluation_seconds': round(raw_eval_seconds, 4),
                'evaluation': raw_eval,
            },
            'latent': {
                'parameters': sum(p.numel() for p in latent.parameters()),
                'training_first_mse': latent_losses[0],
                'training_last_mse': latent_losses[-1],
                'training_seconds': round(latent_train_seconds, 4),
                'evaluation_seconds': round(latent_eval_seconds, 4),
                'evaluation': latent_eval,
            },
            'weights_saved': False,
            'weights_published': False,
            'model_generation': None,
            'limitations': [
                'Raw and latent noise MSE are on different representations: never rank their absolute values',
                'The branches have unequal parameter counts and compute, and latent requires codec pretraining',
                'Short optimizer runs are smoke tests, not convergence or meaningful prompt alignment',
                'All examples are synthetic DSP; no natural-language or real-world sound generalization',
                'CPU wall times from one short run are not reliable throughput benchmarks',
                'No listening, perceptual audio metrics, or generated-waveform quality comparison',
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec-steps', type=int, default=2)
    parser.add_argument('--denoiser-steps', type=int, default=2)
    parser.add_argument('--schedule-steps', type=int, default=8)
    parser.add_argument('--channels', type=int, default=16, choices=(16, 24, 32))
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--evaluation-combinations', type=int, default=4)
    parser.add_argument('--output', type=Path, default=None,
                        help='Optional local JSON report only; no models/weights are saved')
    args = parser.parse_args()
    report = compare(codec_steps=args.codec_steps, denoiser_steps=args.denoiser_steps,
                     schedule_steps=args.schedule_steps, seed=args.seed,
                     channels=args.channels, evaluation_combinations=args.evaluation_combinations)
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized, end='')


if __name__ == '__main__':
    main()
