"""Matched epsilon vs velocity DDPM research on Sonumio-owned synthetic audio.

This is an intentionally *paired* objective experiment: the epsilon and v
models start from bit-identical model weights; they see precisely the same
batch IDs, timesteps, Gaussian corruptions, and classifier-free text masks.
Both use the SAME cosine diffusion schedule. Only the supervised prediction
target and the mathematically corresponding reverse sampler differ.

All samples and neural weights stay in RAM; report numeric JSON only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.architecture_compare import _audio_hashes
from research.generation_audit import (
    waveform_statistics, _compare_waveforms, PAIRS,
)
from research.latent_audio import (
    FRAMES, LatentResearchModel, TRAIN_SEEDS, VALIDATION_SEEDS,
    synthetic_bilingual_waves, reconstruction_loss, sample_latent,
)
from research.text_audio import TextWaveDenoiser
from research.wave_diffusion import DiffusionSchedule
from research.generation_audit import sample_raw_full
from research.velocity_prediction import velocity_target, sample_velocity
from research.multi_seed_study import _check_seeds, summarize_values

PROTOCOL = 'paired-epsilon-v-objective-cosine-v1'
OBJECTIVES = ('epsilon', 'velocity')
ARMS = ('raw', 'latent')
SAMPLING_SEED_OFFSET = 1500
MAX_STEPS = 1000


def _check_settings(*, codec_steps: int, denoiser_steps: int,
                    schedule_steps: int, channels: int) -> None:
    if (type(codec_steps) is not int or not 1 <= codec_steps <= MAX_STEPS
            or type(denoiser_steps) is not int
            or not 1 <= denoiser_steps <= MAX_STEPS):
        raise ValueError('Codec and denoiser steps must be integers 1..1000')
    if type(schedule_steps) is not int or not 8 <= schedule_steps <= 64:
        raise ValueError('Use 8..64 research diffusion timesteps')
    if type(channels) is not int or channels not in (16, 24, 32):
        raise ValueError('Model width must be 16, 24 or 32')


def _record_unclipped(wave: torch.Tensor, *,
                      decoder_preactivation: torch.Tensor | None = None) -> dict:
    """Expose limiter *input* rather than mistaking a tanh-capped decoder for safety.

    For raw DDPM, this is audio before the final hard clamp. For latent DDPM,
    it is the decoder output BEFORE the codec's internal tanh. Hence the
    latent prelimit out-of-range fraction describes *decoder activation*,
    not final waveform samples above 1; compare objectives within each arm.
    """
    def valid_audio(audio: torch.Tensor) -> np.ndarray:
        if (audio.device.type != 'cpu' or audio.requires_grad
                or audio.ndim != 1 or audio.shape != (FRAMES,)
                or audio.dtype != torch.float32 or not torch.isfinite(audio).all()):
            raise ValueError('Generated signal must be detached finite CPU float32')
        values = audio.numpy().astype(np.float64)
        if float(np.max(np.abs(values))) > 1e12:
            raise ValueError('Generated signal has unstable unbounded amplitude')
        return values

    output = valid_audio(wave)
    if decoder_preactivation is not None:
        internal = valid_audio(decoder_preactivation)
        if not np.allclose(np.tanh(internal), output, atol=2e-6, rtol=2e-6):
            raise ValueError('Codec pre-tanh activation does not reproduce waveform')
        before = internal
        source = 'latent_codec_decoder_before_tanh'
    else:
        before = output
        source = 'raw_ddpm_before_final_clamp'
    report = {
        'prelimiter_signal_source': source,
        'preclip_peak': float(np.abs(before).max()),
        'preclip_rms': float(np.sqrt(np.square(before).mean())),
        'preclip_out_of_range_fraction': float((np.abs(before) > 1).mean()),
        'postclip': waveform_statistics(wave.clamp(-1, 1)),
    }
    if decoder_preactivation is not None:
        report['codec_tanh_saturation_fraction'] = float(
            (np.abs(before) >= np.arctanh(0.999)).mean())
    return report

def _train_single(*, seed: int, codec_steps: int, denoiser_steps: int,
                  schedule_steps: int, channels: int) -> dict:
    """Train both objectives with coupled noise, captions and initial weights."""
    _check_settings(codec_steps=codec_steps, denoiser_steps=denoiser_steps,
                    schedule_steps=schedule_steps, channels=channels)
    if type(seed) is not int or not 0 <= seed < 2**32 - 4096:
        raise ValueError('Invalid initialization seed')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        train = synthetic_bilingual_waves(TRAIN_SEEDS)
        validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
        training_hashes = _audio_hashes(train.waves)
        validation_hashes = _audio_hashes(validation.waves)
        if training_hashes & validation_hashes:
            raise ValueError('Exact data overlap across train/validation')

        schedule = DiffusionSchedule(
            schedule_steps, profile='cosine_zero_terminal_v1')
        base_raw = TextWaveDenoiser(channels)
        base_latent = LatentResearchModel(channels)
        initial = {
            'raw': base_raw,
            'latent': base_latent.denoiser,
        }
        networks = {
            arm: {
                'epsilon': initial[arm],
                'velocity': copy.deepcopy(initial[arm]),
            } for arm in ARMS
        }
        # Test equality, not just shared random_seed declarations.
        initial_matching = {
            arm: all(torch.equal(x, y) for x, y in zip(
                networks[arm]['epsilon'].state_dict().values(),
                networks[arm]['velocity'].state_dict().values()))
            for arm in ARMS
        }
        if not all(initial_matching.values()):
            raise ValueError('Objective model weights do not start identically')
        codec = base_latent.codec
        selector = torch.Generator().manual_seed(seed + 2)
        optimizer_codec = torch.optim.AdamW(codec.parameters(), lr=0.001)
        unique_training = train.waves[::2]
        codec.train()
        for _ in range(codec_steps):
            indices = torch.randint(len(unique_training), (2,), generator=selector)
            batch = unique_training[indices]
            optimizer_codec.zero_grad(set_to_none=True)
            loss = reconstruction_loss(batch, codec(batch))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite codec loss')
            loss.backward()
            nn.utils.clip_grad_norm_(codec.parameters(), 1)
            optimizer_codec.step()
        codec.eval()
        codec.requires_grad_(False)
        with torch.no_grad():
            latent_training = torch.cat([
                codec.encode(chunk) for chunk in train.waves.split(8)
            ])
            reconstruction_errors = []
            for chunk in validation.waves.split(8):
                reconstruction_errors.append(
                    F.l1_loss(codec(chunk), chunk, reduction='sum').item()
                )
            codec_mae = float(sum(reconstruction_errors) / validation.waves.numel())

        optimizers = {
            arm: {
                objective: torch.optim.AdamW(network.parameters(), lr=0.0005)
                for objective, network in models.items()
            } for arm, models in networks.items()
        }
        losses = {arm: {obj: [] for obj in OBJECTIVES} for arm in ARMS}
        for _ in range(denoiser_steps):
            ids = torch.randint(len(train.waves), (2,), generator=selector)
            tokens = train.tokens[ids]
            keep = torch.rand((len(ids),), generator=selector) >= 0.1
            masked_tokens = torch.where(keep[:, None], tokens, torch.zeros_like(tokens))
            for arm in ARMS:
                clean = train.waves[ids] if arm == 'raw' else latent_training[ids]
                timestep = torch.randint(schedule.steps, (len(ids),), generator=selector)
                noise = torch.randn(clean.shape, generator=selector)
                corrupted = schedule.corrupt(clean, timestep, noise)
                targets = {
                    'epsilon': noise,
                    'velocity': velocity_target(clean, noise, timestep, schedule),
                }
                for objective in OBJECTIVES:
                    model = networks[arm][objective]
                    optimizer = optimizers[arm][objective]
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    predicted = model(corrupted, timestep, masked_tokens)
                    error = F.mse_loss(predicted, targets[objective])
                    if not torch.isfinite(error):
                        raise ValueError('Nonfinite denoiser loss')
                    error.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    losses[arm][objective].append(float(error.detach()))

        generated = {}
        sample_seed = seed + SAMPLING_SEED_OFFSET
        for arm in ARMS:
            generated[arm] = {}
            for objective in OBJECTIVES:
                model = networks[arm][objective]
                model.eval()
                audio = {}
                entries = []
                for english, hindi, category in PAIRS:
                    for caption, language in ((english, 'en'), (hindi, 'hi')):
                        if objective == 'epsilon':
                            if arm == 'raw':
                                sampled = sample_raw_full(
                                    model, schedule, caption, seed=sample_seed,
                                    clamp_output=False)
                            else:
                                # Same separately trained codec, before its Tanh.
                                base_latent.denoiser = model
                                sampled = sample_latent(
                                    base_latent, schedule, caption,
                                    seed=sample_seed, clamp_output=False,
                                    return_preactivation=True)
                        else:
                            sampled = sample_velocity(
                                model, schedule, caption, seed=sample_seed,
                                latent=arm == 'latent',
                                codec=codec if arm == 'latent' else None,
                                return_preactivation=arm == 'latent')
                        waveform, preactivation = (
                            sampled if isinstance(sampled, tuple) else (sampled, None))
                        assessment = _record_unclipped(
                            waveform, decoder_preactivation=preactivation)
                        entries.append({
                            'category': category, 'language': language,
                            **assessment,
                        })
                        audio[(category, language)] = waveform.clamp(-1, 1)
                contrasts = [
                    {
                        'language': language,
                        **_compare_waveforms(
                            audio[('rain', language)], audio[('wind', language)]),
                    } for language in ('en', 'hi')
                ]
                generated[arm][objective] = {
                    'samples': entries, 'category_contrasts': contrasts,
                    'first_loss': losses[arm][objective][0],
                    'last_loss': losses[arm][objective][-1],
                    'parameters': sum(p.numel() for p in model.parameters()),
                }

        return {
            'seed': seed,
            'schedule_profile': schedule.profile,
            'terminal_alpha_bar': float(schedule.alpha_bars[-1]),
            'codec_steps': codec_steps,
            'denoiser_steps': denoiser_steps,
            'clip_frames': FRAMES,
            'sample_rate': 8000,
            'training_recordings': len(training_hashes),
            'training_captions': train.examples,
            'validation_recordings': len(validation_hashes),
            'validation_captions': validation.examples,
            'train_fingerprint': train.fingerprint,
            'validation_fingerprint': validation.fingerprint,
            'exact_overlap_count': 0,
            'identical_initial_parameters_by_arm': initial_matching,
            'same_training_examples_timesteps_noise_and_text_masks': True,
            'codec_validation_mae': codec_mae,
            'generation_seed': sample_seed,
            'arms': generated,
            'audio_files_saved': False,
            'model_weights_saved': False,
            'model_generation': None,
        }


def _safe_float(value: object, name: str, *, lower: float = 0,
                upper: float = 1e12) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError('Invalid objective ablation measurement: ' + name)
    return float(value)


def _reduce_single(run: dict, *, seed: int, steps: int) -> dict:
    if (not isinstance(run, dict) or run.get('seed') != seed
            or run.get('schedule_profile') != 'cosine_zero_terminal_v1'
            or run.get('codec_steps') is None
            or run.get('denoiser_steps') != steps
            or run.get('training_recordings') != 42
            or run.get('validation_recordings') != 42
            or run.get('training_captions') != 84
            or run.get('validation_captions') != 84
            or run.get('exact_overlap_count') != 0
            or run.get('audio_files_saved') is not False
            or run.get('model_weights_saved') is not False
            or run.get('model_generation') is not None
            or run.get('identical_initial_parameters_by_arm') != {'raw': True, 'latent': True}
            or run.get('same_training_examples_timesteps_noise_and_text_masks') is not True):
        raise ValueError('Invalid/incompatible paired v-vs-epsilon training evidence')
    abar = _safe_float(run.get('terminal_alpha_bar'), 'terminal abar')
    if not 0 < abar < 0.001:
        raise ValueError('Not the targeted low-terminal-SNR cosine schedule')
    comparison = {}
    for arm in ARMS:
        comparison[arm] = {}
        if set(run['arms'][arm]) != set(OBJECTIVES):
            raise ValueError('Missing trained objective')
        eps, vel = [run['arms'][arm][key] for key in OBJECTIVES]
        if (eps.get('parameters') != vel.get('parameters')
                or len(eps.get('samples', [])) != 4
                or len(vel.get('samples', [])) != 4):
            raise ValueError('Model architecture/sample case mismatch')
        metrics = {}
        for objective, source in (('epsilon', eps), ('velocity', vel)):
            expected_cases = {(category, language) for category in ('rain', 'wind')
                              for language in ('en', 'hi')}
            if {(x['category'], x['language']) for x in source['samples']} != expected_cases:
                raise ValueError('Missing bilingual generation audit cases')
            metrics[objective] = {}
            for key in ('preclip_peak', 'preclip_rms', 'preclip_out_of_range_fraction'):
                vals = [_safe_float(row[key], key) for row in source['samples']]
                if key == 'preclip_out_of_range_fraction' and any(x > 1 for x in vals):
                    raise ValueError('Invalid preclip fraction')
                metrics[objective][key] = sum(vals) / len(vals)
            for key in ('rms', 'peak', 'clipping_fraction',
                        'near_silence_fraction', 'spectral_centroid_hz'):
                values = [_safe_float(row['postclip'][key], key, upper=4000
                                      if key == 'spectral_centroid_hz' else 1.0001)
                          for row in source['samples']]
                metrics[objective]['postclip_' + key] = sum(values) / len(values)
            changes = source.get('category_contrasts', [])
            if (len(changes) != 2 or
                    {x.get('language') for x in changes} != {'en', 'hi'}):
                raise ValueError('Missing shared-noise bilingual category comparisons')
            metrics[objective]['rain_vs_wind_mae'] = sum(
                _safe_float(x.get('mean_absolute_difference'), 'prompt change', upper=2.0001)
                for x in changes
            ) / 2
        if arm == 'latent':
            for objective, source in (('epsilon', eps), ('velocity', vel)):
                samples = source['samples']
                if any(x.get('prelimiter_signal_source') != 'latent_codec_decoder_before_tanh'
                       or 'codec_tanh_saturation_fraction' not in x
                       for x in samples):
                    raise ValueError('Missing pre-tanh codec saturation audit')
                metrics[objective]['codec_tanh_saturation_fraction'] = sum(
                    _safe_float(x['codec_tanh_saturation_fraction'], 'codec tanh saturation',
                                upper=1) for x in samples
                ) / len(samples)
        else:
            for source in (eps, vel):
                if any(x.get('prelimiter_signal_source') != 'raw_ddpm_before_final_clamp'
                       for x in source['samples']):
                    raise ValueError('Missing raw pre-clamp measurement origin')
        for key in metrics['epsilon']:
            a, b = metrics['epsilon'][key], metrics['velocity'][key]
            comparison[arm][key] = {
                'epsilon': a, 'velocity': b, 'velocity_minus_epsilon': b - a,
            }
    return comparison


def run_ablation(*, seeds: tuple[int, ...] = (17, 29, 41),
                 codec_steps: int = 128, denoiser_steps: int = 256,
                 schedule_steps: int = 8, channels: int = 16,
                 runner=_train_single) -> dict:
    _check_seeds(seeds)
    _check_settings(codec_steps=codec_steps, denoiser_steps=denoiser_steps,
                    schedule_steps=schedule_steps, channels=channels)
    runs, results = [], []
    for seed in seeds:
        run = runner(seed=seed, codec_steps=codec_steps,
                     denoiser_steps=denoiser_steps,
                     schedule_steps=schedule_steps, channels=channels)
        runs.append(run)
        results.append({'seed': seed, 'arms': _reduce_single(
            run, seed=seed, steps=denoiser_steps)})
    shared = (
        'train_fingerprint', 'validation_fingerprint', 'sample_rate',
        'clip_frames', 'training_recordings', 'training_captions',
        'validation_recordings', 'validation_captions',
        'codec_steps', 'denoiser_steps', 'schedule_profile', 'terminal_alpha_bar',
    )
    if any(tuple(run.get(key) for key in shared) !=
           tuple(runs[0].get(key) for key in shared) for run in runs[1:]):
        raise ValueError('Training dataset or experiment settings drifted between seeds')
    summary = {
        arm: {
            metric: summarize_values([
                run['arms'][arm][metric]['velocity_minus_epsilon']
                for run in results
            ]) for metric in results[0]['arms'][arm]
        } for arm in ARMS
    }
    report = {
        'protocol': PROTOCOL,
        'seeds': list(seeds),
        'configuration': {
            'codec_steps': codec_steps,
            'denoiser_steps_per_arm_objective': denoiser_steps,
            'diffusion_steps': schedule_steps,
            'channels': channels,
            'schedule_profile': 'cosine_zero_terminal_v1',
            'training_recordings': runs[0]['training_recordings'],
            'evaluation_recordings': runs[0]['validation_recordings'],
            'bilingual_caption_pairs': runs[0]['validation_captions'],
            'train_fingerprint': runs[0]['train_fingerprint'],
            'validation_fingerprint': runs[0]['validation_fingerprint'],
            'terminal_alpha_bar': runs[0]['terminal_alpha_bar'],
        },
        'paired_model_initialization': True,
        'paired_batch_timesteps_noise_and_condition_dropout': True,
        'per_seed': results,
        'within_arm_difference_summary': summary,
        'training_diagnostics': [{
            'seed': run['seed'],
            'codec_heldout_reconstruction_mae': run['codec_validation_mae'],
            'objectives': {
                arm: {
                    name: {
                        'first_loss': run['arms'][arm][name]['first_loss'],
                        'last_loss': run['arms'][arm][name]['last_loss'],
                    } for name in OBJECTIVES
                } for arm in ARMS
            },
        } for run in runs],
        'model_weights_written': False,
        'audio_files_written': False,
        'release_authorized': False,
        'warnings': [
            'MSE against epsilon and MSE against v have different targets; do not compare their values directly',
            'Only Sonumio synthetic DSP recordings, 8 kHz, 1.024 seconds; no real-world generalization',
            'Clamped waveform health is not subjective listening quality or correct prompt sound',
            'A lower pre-clamp amplitude alone does not establish sample quality or semantic understanding',
            'Only three model seeds and short training, not convergence or model-release evidence',
        ],
    }
    report['configuration_sha256'] = hashlib.sha256(
        json.dumps({'protocol': PROTOCOL, 'seeds': list(seeds),
                    'configuration': report['configuration']},
                   sort_keys=True).encode('utf-8')).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='17,29,41')
    parser.add_argument('--codec-steps', type=int, default=128)
    parser.add_argument('--denoiser-steps', type=int, default=256)
    parser.add_argument('--schedule-steps', type=int, default=8)
    parser.add_argument('--channels', type=int, choices=(16, 24, 32), default=16)
    parser.add_argument('--output', type=Path, default=None)
    options = parser.parse_args()
    from research.multi_seed_study import parse_seeds
    report = run_ablation(
        seeds=parse_seeds(options.seeds), codec_steps=options.codec_steps,
        denoiser_steps=options.denoiser_steps,
        schedule_steps=options.schedule_steps, channels=options.channels)
    data = json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + '\n'
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(data, encoding='utf-8')
    print(data, end='')


if __name__ == '__main__':
    main()
