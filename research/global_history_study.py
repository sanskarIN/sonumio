"""Train the original full-window history denoiser on actual adjacent DSP.

Two weight-identical GlobalHistoryVelocityDenoiser networks are trained on
the same 8-kHz future windows, captions, timesteps, Gaussian corruption,
future-only v targets and optimizer settings. The TRUE arm sees the
complete preceding 8192-frame recording, the HIDDEN arm sees only zeros.

The held-out WRONG-REMOTE-HISTORY control deliberately changes older
history while replacing its last P samples with the correct prefix.
Therefore correct and wrong histories have exactly the SAME local
overlap, noisy future, t, text and target. Only the remote preceding
history differs, making that contrast a controlled test of nonlocal
context influence (NOT a quality or semantic coherence score).

All data is Sonumio first-party procedural synthetic DSP. No model
weights, WAV, checkpoints, recordings, or audio embeddings are written.
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
    AdjacentDataset, RECIPE as DATA_RECIPE,
    adjacent_bilingual_dataset, audit_adjacent_split,
    paired_continuity_probes,
)
from research.adjacent_prefix_study import _settings, _draw_batch
from research.global_history_denoiser import (
    GlobalHistoryVelocityDenoiser, sample_global_history_ddim,
    PROTOCOL as ARCHITECTURE_PROTOCOL,
)
from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS
from research.velocity_prefix_continuation import _anchor
from research.wave_diffusion import DiffusionSchedule
from research.weight_ema import model_state_sha256

PROTOCOL = "sonumio-global-history-adjacent-training-study-v1"
ARMS = ("full_history", "hidden_history")
CONDITIONS = ("full_history", "hidden_history", "wrong_remote_history")
GENERATION_IDS = (0, 6, 28, 54)


def _history_context(
    data: AdjacentDataset, indices: torch.Tensor, *,
    condition: str,
) -> torch.Tensor:
    if (not isinstance(data, AdjacentDataset)
            or not isinstance(indices, torch.Tensor)
            or indices.ndim != 1 or indices.dtype != torch.long
            or len(indices) == 0
            or not ((indices >= 0) & (indices < data.examples)).all()):
        raise ValueError("Invalid global-history source-group indices")
    true = data.previous[indices]
    if condition == "full_history":
        return true
    if condition == "hidden_history":
        return torch.zeros_like(true)
    if condition == "wrong_remote_history":
        # Dataset places English/Hindi captions consecutively for each
        # recording. Shift by 2 examples to pick the next acoustic group.
        unrelated_indices = (indices + 2) % data.examples
        wrong = data.previous[unrelated_indices].clone()
        width = data.prefix_frames
        wrong[..., -width:] = true[..., -width:]
        if (not torch.equal(wrong[..., -width:], true[..., -width:])
                or torch.equal(wrong, true)):
            raise ValueError("Wrong-remote control must preserve local prefix only")
        return wrong
    raise ValueError("Unknown full-history experimental condition")


def _model_input(
    noisy: torch.Tensor, history: torch.Tensor, *,
    prefix_frames: int,
) -> torch.Tensor:
    if (not isinstance(noisy, torch.Tensor)
            or not isinstance(history, torch.Tensor)
            or noisy.shape != history.shape
            or noisy.ndim != 3 or noisy.shape[1:] != (1, FRAMES)):
        raise ValueError("Global-history model needs paired raw future and history")
    result = _anchor(noisy, history[..., -prefix_frames:])
    if not torch.equal(result[..., :prefix_frames], history[..., -prefix_frames:]):
        raise ValueError("Clean known history was not attached to raw future")
    return result


def _forward_loss(
    model: GlobalHistoryVelocityDenoiser,
    mixed: torch.Tensor, target: torch.Tensor,
    t: torch.Tensor, tokens: torch.Tensor,
    history: torch.Tensor, *, prefix_frames: int,
) -> torch.Tensor:
    prepared = _model_input(
        mixed, history, prefix_frames=prefix_frames,
    )
    prediction = model(
        prepared, t, tokens, history,
        prefix_frames=prefix_frames,
    )
    if (prediction.shape != target.shape
            or not torch.isfinite(prediction).all()):
        raise ValueError("Nonfinite global-history velocity training output")
    loss = F.mse_loss(
        prediction[..., prefix_frames:], target[..., prefix_frames:],
    )
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite global-history future-only v-loss")
    return loss


@torch.no_grad()
def _heldout(models: dict, sources: AdjacentDataset,
             schedule: DiffusionSchedule, *, seed: int) -> dict:
    if set(models) != set(ARMS):
        raise ValueError("Global history study requires exactly two neural models")
    for model in models.values():
        model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed + 181)
    totals = {condition: 0.0 for condition in CONDITIONS}
    group_scores: dict[str, dict] = {}
    items = 0
    for start in range(0, sources.examples, 4):
        indices = torch.arange(
            start, min(start + 4, sources.examples), dtype=torch.long,
        )
        noisy, target, tokens, t = _draw_batch(
            sources, indices, schedule, generator=generator,
        )
        for condition in CONDITIONS:
            neural_model = models[
                "hidden_history" if condition == "hidden_history"
                else "full_history"
            ]
            history = _history_context(
                sources, indices, condition=condition,
            )
            prepared = _model_input(
                noisy, history, prefix_frames=sources.prefix_frames,
            )
            if condition == "wrong_remote_history":
                true = _history_context(
                    sources, indices, condition="full_history",
                )
                # Equal local overlap and noisy input across the two
                # conditions is what isolates nonlocal history influence.
                correct = _model_input(
                    noisy, true, prefix_frames=sources.prefix_frames,
                )
                if not torch.equal(prepared, correct):
                    raise ValueError("Remote-history control changed local future input")
            predicted = neural_model(
                prepared, t, tokens, history,
                prefix_frames=sources.prefix_frames,
            )
            if (predicted.shape != target.shape
                    or not torch.isfinite(predicted).all()):
                raise ValueError("Nonfinite global-history held-out predictions")
            example_errors = (
                predicted[..., sources.prefix_frames:]
                - target[..., sources.prefix_frames:]
            ).square().mean(dim=(1, 2))
            for local_index, dataset_index in enumerate(indices.tolist()):
                error = float(example_errors[local_index])
                if not math.isfinite(error) or error < 0:
                    raise ValueError("Nonfinite future-only heldout v error")
                group = sources.group_ids[dataset_index]
                language = sources.languages[dataset_index]
                entry = group_scores.setdefault(group, {})
                key = condition + ":" + language
                if key in entry:
                    raise ValueError("Bilingual audio source score duplicated")
                entry[key] = error
                totals[condition] += error
        items += len(indices)
    if items != sources.examples or len(group_scores) != sources.group_count:
        raise ValueError("Incomplete held-out global audio-history coverage")
    category_means = {}
    for group, metrics in group_scores.items():
        if set(metrics) != {f"{condition}:{language}"
                           for condition in CONDITIONS
                           for language in ("en", "hi")}:
            raise ValueError("Missing global-history group-language evaluation cells")
        category_means[group] = {
            condition: fmean((
                metrics[condition + ":en"],
                metrics[condition + ":hi"],
            ))
            for condition in CONDITIONS
        }
    average = {name: total / items for name, total in totals.items()}
    if not all(math.isfinite(v) and v >= 0 for v in average.values()):
        raise ValueError("Invalid held-out global history score mean")
    delta_hidden = [
        value["hidden_history"] - value["full_history"]
        for value in category_means.values()
    ]
    delta_wrong = [
        value["wrong_remote_history"] - value["full_history"]
        for value in category_means.values()
    ]
    descriptive = {
        "hidden_minus_full": {
            "mean": fmean(delta_hidden),
            "min": min(delta_hidden),
            "max": max(delta_hidden),
            "positive_recordings": sum(x > 0 for x in delta_hidden),
        },
        "wrong_remote_minus_correct": {
            "mean": fmean(delta_wrong),
            "min": min(delta_wrong),
            "max": max(delta_wrong),
            "positive_recordings": sum(x > 0 for x in delta_wrong),
        },
    }
    return {
        "objective": "future_only_velocity_mse",
        "mse_by_condition": average,
        "group_descriptive_deltas": descriptive,
        "independent_heldout_recordings": sources.group_count,
        "bilingual_caption_examples_scored": sources.examples,
        "future_frames_scored_per_example": FRAMES - sources.prefix_frames,
        "identical_noisy_future_noise_timestep_tokens_for_all_conditions": True,
        "wrong_remote_history_has_exactly_same_known_local_overlap": True,
        "wrong_remote_control_uses_next_acoustic_recording": True,
        "same_language_audio_duplicates_counted_once_as_groups": True,
        "descriptive_deltas_not_statistical_significance": True,
        "human_perception_or_semantic_continuity_measured": False,
    }


@torch.no_grad()
def _generation(
    model: GlobalHistoryVelocityDenoiser,
    dataset: AdjacentDataset,
    schedule: DiffusionSchedule, *,
    seed: int, evaluations: int,
) -> list[dict]:
    output = []
    for index in GENERATION_IDS:
        if index >= dataset.examples:
            continue
        past = dataset.previous[index, 0].clone()
        target_future = dataset.future[index, 0]
        alternative_index = (index + 2) % dataset.examples
        swapped = dataset.previous[alternative_index, 0].clone()
        swapped[-dataset.prefix_frames:] = past[-dataset.prefix_frames:]
        caption = dataset.captions[index]
        correct, real_trace = sample_global_history_ddim(
            model, schedule, caption, past,
            prefix_frames=dataset.prefix_frames,
            evaluations=evaluations, seed=seed + index,
        )
        wrong, wrong_trace = sample_global_history_ddim(
            model, schedule, caption, swapped,
            prefix_frames=dataset.prefix_frames,
            evaluations=evaluations, seed=seed + index,
        )
        overlap = dataset.prefix_frames
        if (not torch.equal(correct[:overlap], past[-overlap:])
                or not torch.equal(wrong[:overlap], past[-overlap:])):
            raise ValueError("Global generated continuity prefix was altered")
        future_error = (correct[overlap:].double()
                        - target_future[overlap:].double())
        remote_delta = (correct[overlap:].double()
                        - wrong[overlap:].double())
        measurement = {
            "heldout_caption_example": index,
            "recording_group_sha256": dataset.group_ids[index],
            "language": dataset.languages[index],
            "caption": caption,
            "sample_seed": seed + index,
            "known_prefix_sha256": hashlib.sha256(
                past[-overlap:].numpy().tobytes()
            ).hexdigest(),
            "correct_clip_sha256": real_trace["generated_sha256"],
            "wrong_remote_clip_sha256": wrong_trace["generated_sha256"],
            "reference_future_sha256": hashlib.sha256(
                target_future[overlap:].numpy().tobytes()
            ).hexdigest(),
            "exact_local_prefix_identity_in_both_outputs": True,
            "generated_first_unknown_sample_jump": real_trace[
                "first_unknown_sample_jump"],
            "true_first_unknown_sample_jump": abs(float(
                target_future[overlap] - target_future[overlap - 1]
            )),
            "sample_aligned_future_mae": float(
                future_error.abs().mean()),
            "sample_aligned_future_rmse": float(
                future_error.square().mean().sqrt()),
            "changed_remote_history_only_future_mae": float(
                remote_delta.abs().mean()),
            "human_listening_quality_measured": False,
        }
        for field in (
            "generated_first_unknown_sample_jump",
            "true_first_unknown_sample_jump",
            "sample_aligned_future_mae",
            "sample_aligned_future_rmse",
            "changed_remote_history_only_future_mae",
        ):
            if not math.isfinite(measurement[field]) or measurement[field] < 0:
                raise ValueError("Invalid finite history generation metric")
        output.append(measurement)
    return output


def run_study(
    *, model_seed: int = 17, updates: int = 4,
    prefix_frames: int = 1024, diffusion_steps: int = 8,
    evaluations: int = 4, generation_seed: int = 301,
    learning_rate: float = 0.0005, batch_size: int = 2,
    training_seeds: tuple[int, ...] = TRAIN_SEEDS,
    heldout_seeds: tuple[int, ...] = VALIDATION_SEEDS,
) -> dict:
    """Original learned history experiment with transparent null controls."""
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
        heldout = adjacent_bilingual_dataset(
            heldout_seeds, prefix_frames=prefix_frames,
        )
        source_audit = audit_adjacent_split(training, heldout)
        ground_truth = paired_continuity_probes(heldout)
        schedule = DiffusionSchedule(
            diffusion_steps, profile="cosine_zero_terminal_v1",
        )
        first = GlobalHistoryVelocityDenoiser(16)
        networks = {
            "full_history": first,
            "hidden_history": copy.deepcopy(first),
        }
        initial = {
            name: model_state_sha256(model) for name, model in networks.items()
        }
        if initial[ARMS[0]] != initial[ARMS[1]]:
            raise ValueError("Global-history model comparison initialized differently")
        optimizer = {
            arm: torch.optim.AdamW(
                networks[arm].parameters(), lr=learning_rate,
            )
            for arm in ARMS
        }
        rng = torch.Generator(device="cpu").manual_seed(model_seed + 1)
        losses = {arm: [] for arm in ARMS}
        encoder_gradient = {arm: [] for arm in ARMS}
        for _ in range(updates):
            indices = torch.randint(
                training.examples, (batch_size,),
                generator=rng, dtype=torch.long,
            )
            noisy, target, tokens, t = _draw_batch(
                training, indices, schedule, generator=rng,
            )
            for arm in ARMS:
                model = networks[arm]
                history = _history_context(
                    training, indices, condition=arm,
                )
                model.train()
                optimizer[arm].zero_grad(set_to_none=True)
                loss = _forward_loss(
                    model, noisy, target, t, tokens, history,
                    prefix_frames=prefix_frames,
                )
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0,
                    error_if_nonfinite=True,
                )
                history_norm = float(sum(
                    parameter.grad.detach().float().square().sum()
                    for parameter in model.history_encoder.parameters()
                    if parameter.grad is not None
                ).sqrt())
                if (not torch.isfinite(grad_norm)
                        or not math.isfinite(history_norm)):
                    raise ValueError("Nonfinite history-encoder gradient")
                optimizer[arm].step()
                losses[arm].append(float(loss.detach()))
                encoder_gradient[arm].append(history_norm)
        trained = {
            arm: model_state_sha256(model) for arm, model in networks.items()
        }
        if (trained["full_history"] == trained["hidden_history"]
                or trained["full_history"] == initial["full_history"]):
            raise ValueError("Global-history model training failed to differentiate states")
        future_scores = _heldout(
            networks, heldout, schedule, seed=model_seed,
        )
        samples = _generation(
            networks["full_history"], heldout, schedule,
            seed=generation_seed, evaluations=evaluations,
        )
        if any(model_state_sha256(networks[arm]) != trained[arm]
               for arm in ARMS):
            raise ValueError("Global-history model changed during heldout evaluation")
        configuration = {
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
            "source": DATA_RECIPE,
            "architecture": ARCHITECTURE_PROTOCOL,
            "objective": "future_only_velocity_mse",
            "schedule": "cosine_zero_terminal_v1",
        }
        digest = hashlib.sha256(json.dumps(
            configuration, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        return {
            "protocol": PROTOCOL,
            "configuration": configuration,
            "configuration_sha256": digest,
            "source_split": source_audit,
            "ground_truth_continuity": ground_truth,
            "neural_model_states": {
                "identical_initial_sha256": initial[ARMS[0]],
                "trained_sha256": trained,
                "distinct_trained_states": True,
                "unchanged_through_evaluation": True,
            },
            "controls": {
                "identical_architecture_and_initial_parameters": True,
                "same_future_target_noisy_samples_timesteps_and_tokens": True,
                "same_optimizers_lr_gradient_clipping_and_objective": True,
                "only_previous_history_information_differs": True,
                "wrong_remote_control_keeps_known_prefix_exact": True,
                "wrong_remote_control_is_evaluation_only": True,
                "bilingual_audio_duplicates_counted_once_per_recording": True,
            },
            "training": {
                arm: {
                    "loss_first": losses[arm][0],
                    "loss_last": losses[arm][-1],
                    "actual_updates": len(losses[arm]),
                    "history_encoder_gradient_norm_mean": fmean(
                        encoder_gradient[arm]),
                    "history_encoder_gradients_checked_finite": True,
                }
                for arm in ARMS
            },
            "heldout_future_objective": future_scores,
            "numerical_generated_future_probes": samples,
            "global_torch_rng_isolated": True,
            "audio_or_model_weights_saved": False,
            "model_publication_authorized": False,
            "speech_music_or_long_context_quality_established": False,
            "warning": (
                "Only short trained original 8-kHz synthetic audio. "
                "A true-versus-hidden-history MSE delta or a wrong-remote "
                "prefix change is numerical evidence, not proof of "
                "perceptual continuity or real-world prompt understanding."
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
    opts = parser.parse_args()
    result = run_study(
        model_seed=opts.model_seed, updates=opts.updates,
        prefix_frames=opts.prefix_frames,
        diffusion_steps=opts.diffusion_steps,
        evaluations=opts.evaluations,
        generation_seed=opts.generation_seed,
        learning_rate=opts.learning_rate,
        batch_size=opts.batch_size,
        training_seeds=tuple(
            int(value) for value in opts.training_seeds.split(",")
        ),
        heldout_seeds=tuple(
            int(value) for value in opts.heldout_seeds.split(",")
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
