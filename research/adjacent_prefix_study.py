"""Paired clean-history versus hidden-history velocity training research.

The previous and future windows really overlap in ONE longer first-party
synthetic recording. The raw 8192-frame velocity denoiser learns to predict
future-only v from noisy future and a clean audio prefix. A capacity-matched
control receives a zero-valued prefix. Both networks start with bit-identical
weights, get identical training examples, timesteps, target future, noise,
optimizer and loss mask. Only historical information available differs.

At evaluation the context network additionally receives a mismatched
recording prefix, which tests context sensitivity without claiming that
this small model uses context to produce perceptually continuous audio.

This does not create a Sonumio-1 model. No external data or weights, WAVs,
checkpoints, audio bytes or training artifacts are persisted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from statistics import fmean

import torch
from torch import nn
from torch.nn import functional as F

from research.adjacent_audio import (
    AdjacentDataset, RECIPE, adjacent_bilingual_dataset,
    audit_adjacent_split, paired_continuity_probes,
)
from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS
from research.text_audio import TextWaveDenoiser
from research.velocity_ddim import ddim_plan
from research.velocity_prefix_continuation import (
    _anchor, _prefix_frames, sample_prefix_velocity_ddim,
)
from research.velocity_prediction import velocity_target
from research.wave_diffusion import DiffusionSchedule, SAMPLE_RATE
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-adjacent-conditional-velocity-v1"
MODEL_VARIANTS = ("true_history", "hidden_history")
SAMPLE_INDICES = (0, 6, 28, 54)


def _settings(*, model_seed: int, updates: int, prefix_frames: int,
              diffusion_steps: int, evaluations: int,
              generation_seed: int, learning_rate: float,
              batch_size: int, training_seeds: tuple[int, ...],
              heldout_seeds: tuple[int, ...]) -> None:
    _prefix_frames(prefix_frames)
    if (type(model_seed) is not int or not 0 <= model_seed < 2**32 - 512
            or type(updates) is not int or not 1 <= updates <= 128
            or type(diffusion_steps) is not int
            or not 8 <= diffusion_steps <= 32
            or type(evaluations) is not int
            or not 2 <= evaluations <= diffusion_steps
            or type(generation_seed) is not int
            or not 0 <= generation_seed < 2**32 - 128
            or type(learning_rate) not in (float, int)
            or not math.isfinite(learning_rate)
            or not 1e-6 <= learning_rate <= 0.01
            or type(batch_size) is not int or not 1 <= batch_size <= 8):
        raise ValueError("Invalid bounded adjacent continuation training settings")
    if (type(training_seeds) is not tuple or type(heldout_seeds) is not tuple
            or not 1 <= len(training_seeds) <= 8
            or not 1 <= len(heldout_seeds) <= 8
            or len(set(training_seeds)) != len(training_seeds)
            or len(set(heldout_seeds)) != len(heldout_seeds)
            or set(training_seeds) & set(heldout_seeds)
            or not all(type(s) is int and 0 <= s < 2**32
                       for s in training_seeds + heldout_seeds)):
        raise ValueError("Adjacent study must use disjoint recording seeds")


def _draw_batch(
    sources: AdjacentDataset, indices: torch.Tensor,
    schedule: DiffusionSchedule,
    *, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (not isinstance(sources, AdjacentDataset)
            or not isinstance(indices, torch.Tensor)
            or indices.ndim != 1 or len(indices) == 0
            or indices.dtype != torch.long
            or not ((indices >= 0) & (indices < sources.examples)).all()
            or not isinstance(generator, torch.Generator)):
        raise ValueError("Invalid paired audio minibatch indices or local RNG")
    clean = sources.future[indices]
    history = sources.previous[indices, :, -sources.prefix_frames:]
    if not torch.equal(history, clean[..., :sources.prefix_frames]):
        raise ValueError("Training pair lost exact audio adjacency")
    t = torch.randint(
        schedule.steps, (len(clean),),
        generator=generator, dtype=torch.long,
    )
    noise = torch.randn(
        clean.shape, generator=generator, dtype=torch.float32,
    )
    mixed = schedule.corrupt(clean, t, noise)
    target = velocity_target(clean, noise, t, schedule)
    tokens = sources.tokens[indices]
    return mixed, target, tokens, t


def _history_for(
    sources: AdjacentDataset, indices: torch.Tensor,
    *, mode: str,
) -> torch.Tensor:
    known = sources.previous[indices, :, -sources.prefix_frames:]
    if mode == "true_history":
        return known
    if mode == "hidden_history":
        return torch.zeros_like(known)
    if mode == "wrong_history":
        # Each English/Hindi pair occupies consecutive indices. Rotate
        # by TWO to point at the next independent source recording, never
        # the alternative language caption for the SAME source audio.
        other = (indices + 2) % sources.examples
        values = sources.previous[other, :, -sources.prefix_frames:]
        if torch.equal(values, known):
            raise ValueError("Negative-control history cannot equal true history")
        return values
    raise ValueError("Unknown adjacent context experimental condition")


def _future_loss(
    model: nn.Module, mixed: torch.Tensor, target: torch.Tensor,
    timestep: torch.Tensor, tokens: torch.Tensor, *,
    prefix_frames: int,
) -> torch.Tensor:
    predicted = model(mixed, timestep, tokens)
    if (not isinstance(predicted, torch.Tensor)
            or predicted.shape != target.shape
            or predicted.dtype != torch.float32
            or not torch.isfinite(predicted).all()):
        raise ValueError("Invalid or nonfinite adjacent future prediction")
    loss = F.mse_loss(
        predicted[..., prefix_frames:], target[..., prefix_frames:],
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite adjacent future velocity loss")
    return loss


@torch.no_grad()
def _heldout(
    models: dict[str, nn.Module], sources: AdjacentDataset,
    schedule: DiffusionSchedule, *, model_seed: int,
) -> dict:
    """All held-out captions, paired t/epsilon, same future-frame targets.

    Wrong-prefix measurement is a sensitivity test of the same trained
    true-history model, not an additional trained model. It is never
    silently counted as another independent acoustic observation.
    """
    if set(models) != set(MODEL_VARIANTS):
        raise ValueError("Expected exactly two original neural model variants")
    for model in models.values():
        model.eval()
    generator = torch.Generator(device="cpu").manual_seed(model_seed + 171)
    conditions = ("true_history", "hidden_history", "wrong_history")
    errors = {condition: 0.0 for condition in conditions}
    group_errors: dict[str, dict[str, float]] = {}
    count = 0
    for offset in range(0, sources.examples, 4):
        indices = torch.arange(
            offset, min(offset + 4, sources.examples), dtype=torch.long
        )
        noisy, target, tokens, timestep = _draw_batch(
            sources, indices, schedule, generator=generator,
        )
        target_size = (FRAMES - sources.prefix_frames)
        for condition in conditions:
            model = models[
                "hidden_history" if condition == "hidden_history"
                else "true_history"
            ]
            history = _history_for(sources, indices, mode=condition)
            input_audio = _anchor(noisy, history)
            prediction = model(input_audio, timestep, tokens)
            if (prediction.shape != target.shape
                    or not torch.isfinite(prediction).all()):
                raise ValueError("Nonfinite held-out adjacent future prediction")
            per_example = (prediction[..., sources.prefix_frames:]
                           - target[..., sources.prefix_frames:]).square()
            for local_index, source_index in enumerate(indices.tolist()):
                group = sources.group_ids[source_index]
                value = float(per_example[local_index].mean())
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Nonfinite per-recording continuation objective")
                slot = group_errors.setdefault(group, {})
                name = condition + ":" + sources.languages[source_index]
                if name in slot:
                    raise ValueError("Duplicate language within heldout source group")
                slot[name] = value
                errors[condition] += value * target_size
        count += len(indices) * target_size
    expected = sources.examples * (FRAMES - sources.prefix_frames)
    if count != expected or len(group_errors) != sources.group_count:
        raise ValueError("Incomplete or inconsistent heldout objective coverage")
    for group, metrics in group_errors.items():
        for condition in conditions:
            if (condition + ":en" not in metrics
                    or condition + ":hi" not in metrics):
                raise ValueError("Missing full bilingual group score")
    values = {condition: errors[condition] / count for condition in conditions}
    if not all(math.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError("Invalid heldout future prediction score")
    per_recording = {
        group: {
            condition: float(fmean([
                metric[condition + ":en"], metric[condition + ":hi"]
            ])) for condition in conditions
        }
        for group, metric in group_errors.items()
    }
    paired_deltas = [
        scores["hidden_history"] - scores["true_history"]
        for scores in per_recording.values()
    ]
    wrong_deltas = [
        scores["wrong_history"] - scores["true_history"]
        for scores in per_recording.values()
    ]
    return {
        "objective": "future_only_velocity_mse",
        "true_history": values["true_history"],
        "hidden_history": values["hidden_history"],
        "wrong_history": values["wrong_history"],
        "hidden_minus_true": values["hidden_history"] - values["true_history"],
        "wrong_minus_true": values["wrong_history"] - values["true_history"],
        "independent_recordings_scored": sources.group_count,
        "bilingual_caption_examples_scored": sources.examples,
        "scored_future_frames_per_caption": FRAMES - sources.prefix_frames,
        "identical_future_targets_noise_timesteps_and_tokens": True,
        "heldout_prefixes_sampled_from_same_adjacent_sources": True,
        "mismatched_context_rotates_by_acoustic_group": True,
        "group_descriptive_hidden_minus_true": {
            "mean": float(fmean(paired_deltas)),
            "min": float(min(paired_deltas)),
            "max": float(max(paired_deltas)),
        },
        "group_descriptive_wrong_minus_true": {
            "mean": float(fmean(wrong_deltas)),
            "min": float(min(wrong_deltas)),
            "max": float(max(wrong_deltas)),
        },
        "statistical_significance_or_perceptual_quality_measured": False,
    }


@torch.no_grad()
def _generation_probes(
    trained: nn.Module, validation: AdjacentDataset,
    schedule: DiffusionSchedule, *,
    evaluations: int, generation_seed: int,
) -> list[dict]:
    """A few held-out clips, unnormalized and never persisted as WAV."""
    output = []
    for index in SAMPLE_INDICES:
        if index >= validation.examples:
            continue
        caption = validation.captions[index]
        original = validation.future[index, 0]
        previous = validation.previous[index, 0]
        context = previous[-validation.prefix_frames:].clone()
        audio, audit = sample_prefix_velocity_ddim(
            trained, schedule, caption, context,
            evaluations=evaluations, seed=generation_seed + index,
        )
        if (not torch.equal(audio[:validation.prefix_frames], context)
                or not audit["exact_original_prefix_preserved"]):
            raise ValueError("Generated adjacent prefix was not preserved")
        actual_new = original[validation.prefix_frames:]
        generated_new = audio[validation.prefix_frames:]
        residual = (generated_new.double() - actual_new.double())
        match_rmse = float(residual.square().mean().sqrt())
        match_mae = float(residual.abs().mean())
        reference_jump = abs(float(
            original[validation.prefix_frames]
            - original[validation.prefix_frames - 1]
        ))
        rows = {
            "heldout_caption_index": index,
            "recording_group_sha256": validation.group_ids[index],
            "language": validation.languages[index],
            "caption": caption,
            "known_history_sha256": audit["prefix_sha256"],
            "true_future_sha256": hashlib.sha256(
                actual_new.numpy().tobytes()).hexdigest(),
            "sampled_full_clip_sha256": audit["output_sha256"],
            "seed": generation_seed + index,
            "exact_input_prefix_preserved": True,
            "reference_first_future_sample_jump": reference_jump,
            "generated_first_future_sample_jump":
                audit["first_future_sample_jump"],
            "future_sample_aligned_mae": match_mae,
            "future_sample_aligned_rmse": match_rmse,
            "sample_aligned_error_not_a_perceptual_score": True,
            "audible_quality_or_semantic_coherence_assessed": False,
        }
        if any(not math.isfinite(rows[key]) or rows[key] < 0 for key in (
            "reference_first_future_sample_jump",
            "generated_first_future_sample_jump",
            "future_sample_aligned_mae",
            "future_sample_aligned_rmse",
        )):
            raise ValueError("Nonfinite adjacent generation diagnostic")
        output.append(rows)
    return output


def run_study(
    *, model_seed: int = 17, updates: int = 4,
    prefix_frames: int = 1024, diffusion_steps: int = 8,
    evaluations: int = 4, generation_seed: int = 301,
    learning_rate: float = 0.0005, batch_size: int = 2,
    training_seeds: tuple[int, ...] = TRAIN_SEEDS,
    heldout_seeds: tuple[int, ...] = VALIDATION_SEEDS,
) -> dict:
    """Train and evaluate real adjacent-history and no-history neural models."""
    _settings(
        model_seed=model_seed, updates=updates,
        prefix_frames=prefix_frames, diffusion_steps=diffusion_steps,
        evaluations=evaluations, generation_seed=generation_seed,
        learning_rate=learning_rate, batch_size=batch_size,
        training_seeds=training_seeds, heldout_seeds=heldout_seeds,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed)
        training = adjacent_bilingual_dataset(
            training_seeds, prefix_frames=prefix_frames,
        )
        validation = adjacent_bilingual_dataset(
            heldout_seeds, prefix_frames=prefix_frames,
        )
        sources = audit_adjacent_split(training, validation)
        ground_truth = paired_continuity_probes(validation)
        schedule = DiffusionSchedule(
            diffusion_steps, profile="cosine_zero_terminal_v1",
        )
        original = TextWaveDenoiser(16)
        networks = {
            "true_history": original,
            "hidden_history": copy.deepcopy(original),
        }
        initial = {
            name: model_state_sha256(model) for name, model in networks.items()
        }
        if initial["true_history"] != initial["hidden_history"]:
            raise ValueError("Compared models were not initialized identically")
        optimizers = {
            name: torch.optim.AdamW(model.parameters(), lr=learning_rate)
            for name, model in networks.items()
        }
        rng = torch.Generator(device="cpu").manual_seed(model_seed + 1)
        histories = {name: [] for name in MODEL_VARIANTS}
        for _ in range(updates):
            indices = torch.randint(
                training.examples, (batch_size,),
                generator=rng, dtype=torch.long,
            )
            mixed, target, tokens, t = _draw_batch(
                training, indices, schedule, generator=rng,
            )
            for condition in MODEL_VARIANTS:
                network = networks[condition]
                network.train()
                optimizers[condition].zero_grad(set_to_none=True)
                known = _history_for(training, indices, mode=condition)
                conditioned = _anchor(mixed, known)
                loss = _future_loss(
                    network, conditioned, target, t, tokens,
                    prefix_frames=prefix_frames,
                )
                loss.backward()
                gradient = nn.utils.clip_grad_norm_(
                    network.parameters(), max_norm=1.0,
                    error_if_nonfinite=True,
                )
                if not torch.isfinite(gradient):
                    raise ValueError("Nonfinite neural gradient in adjacent study")
                optimizers[condition].step()
                histories[condition].append(float(loss.detach()))
        final = {
            name: model_state_sha256(model) for name, model in networks.items()
        }
        if (final["true_history"] == final["hidden_history"]
                or final["true_history"] == initial["true_history"]):
            raise ValueError("Adjacent training did not produce distinct trained states")
        scores = _heldout(
            networks, validation, schedule, model_seed=model_seed,
        )
        generated = _generation_probes(
            networks["true_history"], validation, schedule,
            evaluations=evaluations, generation_seed=generation_seed,
        )
        if any(model_state_sha256(networks[name]) != final[name]
               for name in MODEL_VARIANTS):
            raise ValueError("Evaluation changed trained model parameters")
        options = {
            "model_seed": model_seed,
            "updates": updates,
            "prefix_frames": prefix_frames,
            "diffusion_steps": diffusion_steps,
            "evaluations": evaluations,
            "generation_seed": generation_seed,
            "learning_rate": float(learning_rate),
            "batch_size": batch_size,
            "training_seeds": list(training_seeds),
            "heldout_seeds": list(heldout_seeds),
            "training_corpus": RECIPE,
            "model_architecture": "original_raw_text_velocity_denoiser_16",
            "schedule": "cosine_zero_terminal_v1",
            "sampler": "clean_prefix_velocity_ddim_eta_zero",
        }
        config_hash = hashlib.sha256(
            json.dumps(
                options, sort_keys=True, ensure_ascii=False,
                allow_nan=False, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "protocol": PROTOCOL,
            "configuration": options,
            "configuration_sha256": config_hash,
            "split": sources,
            "ground_truth_continuity": ground_truth,
            "model_states": {
                "initial_shared_sha256": initial["true_history"],
                "trained": final,
                "distinct_trained_states": True,
                "unchanged_through_evaluation": True,
            },
            "controls": {
                "same_initial_parameters": True,
                "same_model_capacity_optimizer_and_learning_rate": True,
                "identical_minibatches_future_noise_timestep_target_and_tokens": True,
                "future_only_v_objective_for_both_models": True,
                "only_history_observability_differs": True,
                "incorrect_history_is_evaluation_only": True,
                "same_heldout_corruption_for_all_conditions": True,
                "bilingual_captions_share_audio_and_not_independent": True,
            },
            "training_loss": {
                name: {
                    "first": histories[name][0],
                    "last": histories[name][-1],
                    "steps": len(histories[name]),
                }
                for name in MODEL_VARIANTS
            },
            "heldout_future_objective": scores,
            "generation_diagnostics": generated,
            "global_torch_rng_isolated": True,
            "no_audio_or_weights_saved": True,
            "model_release_authorized": False,
            "speech_music_or_general_prompt_quality_established": False,
            "long_context_continuation_established": False,
            "warning": (
                "Continuous first-party source windows provide an exact "
                "adjacent training target, but a tiny nonconverged 8-kHz "
                "model and correlated synthetic families are not evidence "
                "of natural long-form audio or general semantic alignment."
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--prefix-frames", type=int, default=1024)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=4)
    parser.add_argument("--generation-seed", type=int, default=301)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--training-seeds", default="0,1")
    parser.add_argument("--heldout-seeds", default="101,102")
    args = parser.parse_args()
    result = run_study(
        model_seed=args.model_seed, updates=args.updates,
        prefix_frames=args.prefix_frames,
        diffusion_steps=args.diffusion_steps,
        evaluations=args.evaluations,
        generation_seed=args.generation_seed,
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        training_seeds=tuple(
            int(value) for value in args.training_seeds.split(",")
        ),
        heldout_seeds=tuple(
            int(value) for value in args.heldout_seeds.split(",")
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
