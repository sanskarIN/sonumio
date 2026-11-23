"""Matched first-party synthetic study of clean-prefix masked-future training.

Original Sonumio raw v-denoisers are trained from the SAME initial weights
with the SAME caption minibatches, noisy inputs, timesteps and optimizer.
Only the supervised region differs: unknown future samples vs all frames.
Evaluate both on all 84 held-out bilingual captions with identical corruptions.

Reports descriptive numerical metrics only: no generated WAV, model weights,
audio quality claim, speech/music continuation or Sonumio-1 model release.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.architecture_compare import _audio_hashes
from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, synthetic_bilingual_waves
from research.text_audio import TextWaveDenoiser
from research.velocity_prediction import velocity_target
from research.velocity_prefix_continuation import (
    _anchor, _prefix_frames, sample_prefix_velocity_ddim,
)
from research.wave_diffusion import DiffusionSchedule
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-clean-prefix-future-training-paired-study-v1"
ARMS = ("masked_future_loss", "all_frames_loss")
EXAMPLE_IDS = (0, 6)


def _settings(*, seed: int, updates: int, prefix_frames: int,
              diffusion_steps: int, evaluations: int, sample_seed: int) -> None:
    _prefix_frames(prefix_frames)
    if (type(seed) is not int or not 0 <= seed < 2**32 - 10_000
            or type(updates) is not int or not 1 <= updates <= 64
            or type(diffusion_steps) is not int or not 8 <= diffusion_steps <= 16
            or type(evaluations) is not int
            or not 2 <= evaluations <= diffusion_steps
            or type(sample_seed) is not int or not 0 <= sample_seed <= 2**32 - 7):
        raise ValueError("Invalid bounded clean-prefix training study settings")


def _draw(sources, indices: torch.Tensor, schedule: DiffusionSchedule,
          generator: torch.Generator, *, prefix_frames: int):
    clean = sources.waves[indices]
    tokens = sources.tokens[indices]
    timestep = torch.randint(
        schedule.steps, (len(clean),), generator=generator, dtype=torch.long,
    )
    noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
    mixed = schedule.corrupt(clean, timestep, noise)
    visible = _anchor(mixed, clean[..., :prefix_frames])
    target = velocity_target(clean, noise, timestep, schedule)
    return visible, target, tokens, timestep


@torch.no_grad()
def _heldout(models: dict, validation, schedule: DiffusionSchedule,
             *, seed: int, prefix_frames: int) -> dict:
    """Fully paired held-out future-region error over 84 caption examples."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    errors = {name: 0.0 for name in ARMS}
    frame_count = 0
    for model in models.values():
        model.eval()
    for start in range(0, len(validation.waves), 4):
        ids = torch.arange(start, min(start + 4, len(validation.waves)))
        noisy, target, tokens, t = _draw(
            validation, ids, schedule, gen, prefix_frames=prefix_frames,
        )
        for name, model in models.items():
            predicted = model(noisy, t, tokens)
            if (predicted.shape != target.shape
                    or not torch.isfinite(predicted).all()):
                raise ValueError("Nonfinite or incompatible held-out v prediction")
            errors[name] += float(F.mse_loss(
                predicted[..., prefix_frames:],
                target[..., prefix_frames:],
                reduction="sum",
            ))
        frame_count += len(ids) * (FRAMES - prefix_frames)
    results = {name: value / frame_count for name, value in errors.items()}
    if (frame_count != len(validation.waves) * (FRAMES - prefix_frames)
            or len(validation.waves) != 84
            or not all(math.isfinite(v) and v >= 0 for v in results.values())):
        raise ValueError("Held-out future-only evaluation was incomplete")
    results["masked_minus_all"] = (
        results["masked_future_loss"] - results["all_frames_loss"]
    )
    results["heldout_bilingual_caption_examples"] = len(validation.waves)
    results["distinct_audio_recordings"] = len(validation.waves) // 2
    results["scored_future_frames_per_caption"] = FRAMES - prefix_frames
    results["identical_noise_timesteps_tokens_and_inputs"] = True
    return results


def run_study(*, seed: int = 17, updates: int = 4,
              prefix_frames: int = 1024, diffusion_steps: int = 8,
              evaluations: int = 4, sample_seed: int = 301) -> dict:
    _settings(
        seed=seed, updates=updates, prefix_frames=prefix_frames,
        diffusion_steps=diffusion_steps, evaluations=evaluations,
        sample_seed=sample_seed,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        train = synthetic_bilingual_waves(TRAIN_SEEDS)
        validation = synthetic_bilingual_waves(VALIDATION_SEEDS)
        train_hash = _audio_hashes(train.waves)
        val_hash = _audio_hashes(validation.waves)
        if (train_hash & val_hash or len(train_hash) != 42
                or len(val_hash) != 42 or train.examples != 84
                or validation.examples != 84):
            raise ValueError("Source recording overlap or incomplete bilingual dataset")
        schedule = DiffusionSchedule(
            diffusion_steps, profile="cosine_zero_terminal_v1",
        )
        original = TextWaveDenoiser(16)
        models = {
            "masked_future_loss": original,
            "all_frames_loss": copy.deepcopy(original),
        }
        initial = {name: model_state_sha256(model)
                   for name, model in models.items()}
        if initial[ARMS[0]] != initial[ARMS[1]]:
            raise ValueError("Compared research models were initialized differently")
        optimizers = {
            name: torch.optim.AdamW(model.parameters(), lr=5e-4)
            for name, model in models.items()
        }
        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        histories = {name: [] for name in ARMS}
        for _ in range(updates):
            ids = torch.randint(
                len(train.waves), (2,), generator=gen,
            )
            noisy, target, tokens, timestep = _draw(
                train, ids, schedule, gen, prefix_frames=prefix_frames,
            )
            for name in ARMS:
                model = models[name]
                model.train()
                optimizers[name].zero_grad(set_to_none=True)
                predicted = model(noisy, timestep, tokens)
                if predicted.shape != target.shape:
                    raise ValueError("Incorrect clean-prefix research model shape")
                if name == "masked_future_loss":
                    loss = F.mse_loss(
                        predicted[..., prefix_frames:],
                        target[..., prefix_frames:],
                    )
                else:
                    loss = F.mse_loss(predicted, target)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite clean-prefix training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizers[name].step()
                histories[name].append(float(loss.detach()))
        final = {name: model_state_sha256(model)
                 for name, model in models.items()}
        heldout = _heldout(
            models, validation, schedule, seed=seed + 171,
            prefix_frames=prefix_frames,
        )
        results = []
        for index in EXAMPLE_IDS:
            reference = validation.waves[index, 0]
            prefix = reference[:prefix_frames].clone()
            prompt = "rain" if index == 0 else "wind"
            for name in ARMS:
                output, trace = sample_prefix_velocity_ddim(
                    models[name], schedule, prompt, prefix,
                    evaluations=evaluations, seed=sample_seed + index,
                )
                if not trace["exact_original_prefix_preserved"]:
                    raise ValueError("Invalid prefix-conditioned numerical sample")
                results.append({
                    "heldout_example_index": index,
                    "model_variant": name,
                    "sampling_seed": sample_seed + index,
                    "same_input_prefix_for_both_variants": True,
                    "prefix_sha256": trace["prefix_sha256"],
                    "output_sha256": trace["output_sha256"],
                    "first_future_sample_jump": trace["first_future_sample_jump"],
                    "unclipped_peak": trace["unclipped_peak"],
                    "fraction_above_unit_range": trace["fraction_above_unit_range"],
                    "perceptual_or_semantic_continuity_evaluated": False,
                })
        for name in ARMS:
            if model_state_sha256(models[name]) != final[name]:
                raise ValueError("Model parameter state changed during evaluation")
        config = {
            "seed": seed, "updates": updates,
            "prefix_frames": prefix_frames,
            "diffusion_steps": diffusion_steps,
            "evaluations": evaluations,
            "sample_seed": sample_seed,
            "channels": 16,
            "schedule_profile": "cosine_zero_terminal_v1",
            "training_corpus": "sonumio-first-party-synthetic-dsp",
        }
        return {
            "protocol": PROTOCOL,
            "configuration": config,
            "configuration_sha256": hashlib.sha256(json.dumps(
                config, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "data": {
                "independent_training_recordings": len(train_hash),
                "independent_heldout_recordings": len(val_hash),
                "bilingual_caption_examples_per_split": train.examples,
                "train_fingerprint": train.fingerprint,
                "heldout_fingerprint": validation.fingerprint,
                "exact_audio_overlap": False,
                "heldout_audio_used_for_training": False,
            },
            "controls": {
                "same_initial_model_sha256": initial[ARMS[0]],
                "same_training_minibatches_timesteps_noise_and_tokens": True,
                "same_optimizer_type_and_learning_rate": True,
                "only_loss_region_differs": True,
                "same_heldout_noisy_input_for_both_models": True,
                "distinct_trained_weights": final[ARMS[0]] != final[ARMS[1]],
            },
            "trained_model_sha256": final,
            "training_loss_start_end": {
                name: {"initial": histories[name][0], "final": histories[name][-1]}
                for name in ARMS
            },
            "heldout_future_velocity_objective": heldout,
            "prefix_conditioned_numerical_samples": results,
            "all_model_states_unchanged_during_evaluation": True,
            "no_audio_or_weights_saved": True,
            "model_release_authorized": False,
            "human_listening_or_semantic_alignment_measured": False,
            "learned_long_context_or_music_speech_continuity_claim": False,
            "warning": (
                "Only a paired finite synthetic future-prediction study. "
                "Known-prefix construction differs from ordinary v training. "
                "A lower future v-MSE does not establish sounds better, "
                "smoother seams, semantic continuation, speech or music."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--prefix-frames", type=int, default=1024)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=301)
    arguments = parser.parse_args()
    print(json.dumps(run_study(
        seed=arguments.seed, updates=arguments.updates,
        prefix_frames=arguments.prefix_frames,
        diffusion_steps=arguments.diffusion_steps,
        evaluations=arguments.evaluations,
        sample_seed=arguments.sample_seed,
    ), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
