"""Controlled velocity denoiser EMA versus online-weights research.

Train Sonumio raw/latent v-models on original first-party synthetic sounds.
Compare the SAME online training trajectory to its EMA shadow, using identical
held-out corrupted inputs and identical seeded DDIM starting noise. This
experiment does not claim perceptual quality, prompt understanding, nor
original Sonumio-1 model readiness. Nothing is saved except JSON if the user
explicitly redirects standard output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from statistics import mean

import torch
from torch import nn
from torch.nn import functional as F

from research.architecture_compare import _audio_hashes
from research.generation_audit import PAIRS, _compare_waveforms
from research.latent_audio import (
    FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, LatentResearchModel,
    synthetic_bilingual_waves, reconstruction_loss,
)
from research.text_audio import TextWaveDenoiser
from research.velocity_ablation import _record_unclipped
from research.velocity_ddim import sample_velocity_ddim
from research.velocity_prediction import velocity_target
from research.wave_diffusion import DiffusionSchedule
from research.weight_ema import WeightEMA, model_state_sha256

PROTOCOL = "sonumio-velocity-denoiser-ema-paired-study-v1"
PROMPTS = tuple(
    (category, language, caption)
    for en, hi, category in PAIRS
    for language, caption in (("en", en), ("hi", hi))
)


def _settings(*, seed: int, steps: int, codec_steps: int,
              diffusion_steps: int, channels: int, decay: float,
              warmup: bool, sampling_seeds: tuple[int, ...]) -> None:
    if (type(seed) is not int or not 0 <= seed < 2**32 - 8192
            or type(steps) is not int or not 1 <= steps <= 64
            or type(codec_steps) is not int or not 1 <= codec_steps <= 16
            or type(diffusion_steps) is not int or not 8 <= diffusion_steps <= 16
            or type(channels) is not int or channels != 16
            or type(decay) not in (int, float) or not math.isfinite(decay)
            or not 0 <= decay < 1 or type(warmup) is not bool):
        raise ValueError("Invalid bounded CPU velocity EMA experiment settings")
    if (type(sampling_seeds) is not tuple or not 1 <= len(sampling_seeds) <= 2
            or len(set(sampling_seeds)) != len(sampling_seeds)
            or any(type(value) is not int or not 0 <= value < 2**32
                   for value in sampling_seeds)):
        raise ValueError("EMA study requires one or two unique unsigned sampling seeds")


def _validation_mse(source, online: nn.Module, averaged: nn.Module,
                    schedule: DiffusionSchedule, *, seed: int) -> dict:
    """One shared noise/timestep batch per model, all 84 held-out captions."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = {"online": 0.0, "ema": 0.0}
    count = 0
    online.eval()
    averaged.eval()
    with torch.no_grad():
        for offset in range(0, len(source.waves), 4):
            clean = source.waves[offset:offset + 4]
            tokens = source.tokens[offset:offset + len(clean)]
            t = torch.randint(schedule.steps, (len(clean),), generator=generator)
            noise = torch.randn(clean.shape, generator=generator)
            noisy = schedule.corrupt(clean, t, noise)
            target = velocity_target(clean, noise, t, schedule)
            for name, model in (("online", online), ("ema", averaged)):
                predicted = model(noisy, t, tokens)
                output[name] += float(F.mse_loss(
                    predicted, target, reduction="sum",
                ))
            count += target.numel()
    if count <= 0:
        raise ValueError("Empty held-out EMA evaluation")
    mse = {name: error / count for name, error in output.items()}
    if not all(math.isfinite(value) and value >= 0 for value in mse.values()):
        raise ValueError("Nonfinite held-out EMA evaluation loss")
    mse["ema_minus_online"] = mse["ema"] - mse["online"]
    mse["bilingual_caption_examples"] = len(source.waves)
    mse["distinct_audio_recordings"] = len(source.waves) // 2
    mse["shared_noise_and_timesteps"] = True
    return mse


def _generate_comparison(online: nn.Module, averaged: nn.Module,
                         codec: nn.Module | None,
                         schedule: DiffusionSchedule, *,
                         sampling_seeds: tuple[int, ...]) -> list[dict]:
    """Numerical waveform health without writing or publishing generated audio."""
    latent = codec is not None
    records = []
    for seed in sampling_seeds:
        for category, language, caption in PROMPTS:
            generated = {}
            for label, model in (("online", online), ("ema", averaged)):
                result = sample_velocity_ddim(
                    model, schedule, caption, evaluations=schedule.steps,
                    seed=seed, guidance=1, latent=latent, codec=codec,
                    return_preactivation=latent,
                )
                wave, before = (result if isinstance(result, tuple)
                                else (result, None))
                generated[label] = {
                    "health": _record_unclipped(
                        wave, decoder_preactivation=before,
                    ),
                    "waveform": wave.clamp(-1, 1),
                }
            contrast = _compare_waveforms(
                generated["online"]["waveform"], generated["ema"]["waveform"],
            )
            record = {
                "seed": seed, "category": category, "language": language,
                "online_ema_waveform_difference": contrast,
                "prompt_adherence_or_human_quality_measured": False,
            }
            for label in ("online", "ema"):
                health = generated[label]["health"]
                record[label] = {
                    "prelimiter_signal_source": health["prelimiter_signal_source"],
                    "postclip": health["postclip"],
                    "preclip_peak": health["preclip_peak"],
                    "preclip_out_of_range_fraction":
                        health["preclip_out_of_range_fraction"],
                }
            records.append(record)
    return records


def run_study(*, seed: int = 17, steps: int = 4,
              codec_steps: int = 2, diffusion_steps: int = 8,
              channels: int = 16, decay: float = 0.9,
              warmup: bool = False,
              sampling_seeds: tuple[int, ...] = (301,)) -> dict:
    """Train two online v-models, update EMA after each step, evaluate in RAM."""
    _settings(seed=seed, steps=steps, codec_steps=codec_steps,
              diffusion_steps=diffusion_steps, channels=channels,
              decay=decay, warmup=warmup, sampling_seeds=sampling_seeds)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        train = synthetic_bilingual_waves(TRAIN_SEEDS)
        validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
        train_hashes = _audio_hashes(train.waves)
        val_hashes = _audio_hashes(validation.waves)
        if train_hashes & val_hashes or train.fingerprint == validation.fingerprint:
            raise ValueError("Overlapping synthetic train and held-out data")
        schedule = DiffusionSchedule(
            diffusion_steps, profile="cosine_zero_terminal_v1",
        )
        raw = TextWaveDenoiser(channels)
        latent = LatentResearchModel(channels)
        codec = latent.codec
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        optimizer = torch.optim.AdamW(codec.parameters(), lr=0.001)
        unique = train.waves[::2]
        codec.train()
        for _ in range(codec_steps):
            ids = torch.randint(len(unique), (2,), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(unique[ids], codec(unique[ids]))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite EMA study codec loss")
            loss.backward()
            nn.utils.clip_grad_norm_(codec.parameters(), 1)
            optimizer.step()
        codec.eval().requires_grad_(False)
        with torch.no_grad():
            latent_training = torch.cat([
                codec.encode(chunk) for chunk in train.waves.split(8)
            ])
            latent_validation = torch.cat([
                codec.encode(chunk) for chunk in validation.waves.split(8)
            ])
        networks = {"raw": raw, "latent": latent.denoiser}
        shadows = {
            arm: WeightEMA(model, decay=decay, warmup=warmup)
            for arm, model in networks.items()
        }
        optimizers = {
            arm: torch.optim.AdamW(model.parameters(), lr=0.0005)
            for arm, model in networks.items()
        }
        losses = {"raw": [], "latent": []}
        for _ in range(steps):
            ids = torch.randint(len(train.waves), (2,), generator=generator)
            timestep = torch.randint(
                schedule.steps, (len(ids),), generator=generator,
            )
            keep = torch.rand(len(ids), generator=generator) >= 0.1
            tokens = torch.where(
                keep[:, None], train.tokens[ids],
                torch.zeros_like(train.tokens[ids]),
            )
            for arm, model in networks.items():
                clean = train.waves[ids] if arm == "raw" else latent_training[ids]
                noise = torch.randn(clean.shape, generator=generator)
                noisy = schedule.corrupt(clean, timestep, noise)
                target = velocity_target(clean, noise, timestep, schedule)
                model.train()
                optimizers[arm].zero_grad(set_to_none=True)
                prediction = model(noisy, timestep, tokens)
                loss = F.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite EMA study v-loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizers[arm].step()
                shadows[arm].update(model)
                losses[arm].append(float(loss.detach()))
        evaluations, comparisons, states = {}, {}, {}
        for arm, online in networks.items():
            averaged = shadows[arm].snapshot()
            source = validation if arm == "raw" else type(validation)(
                latent_validation, validation.tokens,
                validation.fingerprint, validation.examples,
            )
            online_hash = model_state_sha256(online)
            ema_hash = model_state_sha256(averaged)
            evaluation = _validation_mse(
                source, online, averaged, schedule, seed=seed + 181,
            )
            rows = _generate_comparison(
                online, averaged, codec if arm == "latent" else None,
                schedule, sampling_seeds=sampling_seeds,
            )
            if (model_state_sha256(online) != online_hash
                    or model_state_sha256(averaged) != ema_hash
                    or shadows[arm].state_sha256() != ema_hash):
                raise ValueError("EMA experiment mutated model weights in evaluation")
            evaluations[arm] = evaluation
            comparisons[arm] = rows
            states[arm] = {
                "online_model_sha256": online_hash,
                "ema_model_sha256": ema_hash,
                "ema_updates": shadows[arm].updates,
                "ema_effective_decay_last": shadows[arm].last_effective_decay,
                "same_parameters_through_evaluation": True,
            }
        summary = {
            arm: {
                "heldout_velocity_mse_ema_minus_online":
                    evaluations[arm]["ema_minus_online"],
                "mean_online_clipping_fraction": mean(
                    row["online"]["postclip"]["clipping_fraction"]
                    for row in comparisons[arm]),
                "mean_ema_clipping_fraction": mean(
                    row["ema"]["postclip"]["clipping_fraction"]
                    for row in comparisons[arm]),
                "mean_waveform_difference": mean(
                    row["online_ema_waveform_difference"][
                        "mean_absolute_difference"] for row in comparisons[arm]),
                "generated_pairs_compared": len(comparisons[arm]),
            } for arm in ("raw", "latent")
        }
    config = {
        "seed": seed, "steps": steps, "codec_steps": codec_steps,
        "diffusion_steps": diffusion_steps, "channels": channels,
        "decay": float(decay), "warmup": warmup,
        "sampling_seeds": list(sampling_seeds),
        "prediction_objective": "velocity",
        "sampler": "deterministic_ddim_eta_zero",
        "source": "sonumio_first_party_synthetic_dsp",
    }
    return {
        "protocol": PROTOCOL,
        "configuration": config,
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf8")
        ).hexdigest(),
        "training": {
            "train_fingerprint": train.fingerprint,
            "validation_fingerprint": validation.fingerprint,
            "independent_training_recordings": len(train_hashes),
            "independent_heldout_recordings": len(val_hashes),
            "bilingual_caption_examples_per_split": train.examples,
            "train_heldout_exact_overlap": False,
            "initial_vs_final_training_loss": {
                arm: {"initial": losses[arm][0], "final": losses[arm][-1]}
                for arm in ("raw", "latent")
            },
        },
        "heldout_velocity_objective": evaluations,
        "model_state_integrity": states,
        "in_memory_generated_numerical_comparisons": comparisons,
        "by_arm": summary,
        "online_and_ema_same_training_trajectory": True,
        "heldout_audio_used_for_training": False,
        "no_model_weights_or_audio_saved": True,
        "model_publication_authorized": False,
        "listening_or_semantic_audio_quality_evaluated": False,
        "warning": (
            "EMA is an experimental training-weight average, not a validated "
            "quality improvement. Equal-noise heldout v-MSE and numerical "
            "waveform stats are not listening quality or prompt adherence. "
            "No Sonumio-1 model or weights were published."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--codec-steps", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--sampling-seeds", default="301")
    args = parser.parse_args()
    report = run_study(
        seed=args.seed, steps=args.steps,
        codec_steps=args.codec_steps, diffusion_steps=args.diffusion_steps,
        decay=args.decay, warmup=args.warmup,
        sampling_seeds=tuple(int(x) for x in args.sampling_seeds.split(",")),
    )
    print(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False))


if __name__ == "__main__":
    main()
