"""Independent neural-init replication of actual adjacent full-history training.

Reject configuration drift, source fingerprint drift, duplicate model
weights, omitted recording groups, forged pairing controls and nonfinite
metrics before reporting descriptive across-model effects.

A separate trained seed is a model replicate. The 42 bilingual captions
and multiple generated samples per seed are repeated measurements, not
independent models. No confidence intervals, p-values or quality winner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

from research.global_history_study import (
    PROTOCOL as SINGLE_PROTOCOL, GENERATION_IDS,
    run_study,
)
from research.multi_seed_study import summarize_values
from research.adjacent_prefix_study import _settings

PROTOCOL = "sonumio-replicated-global-history-adjacent-study-v1"
METRICS = ("full_history", "hidden_history", "wrong_remote_history")


def _sha(value) -> bool:
    return (type(value) is str and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _finite(value, *, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Nonfinite/incorrectly typed global-history statistic")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError("Negative or out-of-range global-history statistic")
    return result


def _check_seeds(seeds: tuple[int, ...]) -> None:
    if (type(seeds) is not tuple or not 2 <= len(seeds) <= 4
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 - 512
                   for seed in seeds)):
        raise ValueError("Replication needs 2..4 distinct bounded model seeds")


def _verified_run(
    report: dict, *, seed: int, common: dict,
    expected_source: tuple | None,
) -> dict:
    if type(report) is not dict or report.get("protocol") != SINGLE_PROTOCOL:
        raise ValueError("Wrong or missing original global-history protocol")
    config = report.get("configuration")
    split = report.get("source_split")
    states = report.get("neural_model_states")
    control = report.get("controls")
    heldout = report.get("heldout_future_objective")
    generated = report.get("numerical_generated_future_probes")
    training = report.get("training")
    if (not all(type(x) is dict
                for x in (config, split, states, control, heldout, training))
            or type(generated) is not list):
        raise ValueError("Incomplete source, architecture or objective evidence")
    if (type(config.get("model_seed")) is not int
            or config["model_seed"] != seed
            or any(config.get(key) != value for key, value in common.items())):
        raise ValueError("Global-history run changed non-seed training controls")
    expected = hashlib.sha256(json.dumps(
        config, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if report.get("configuration_sha256") != expected:
        raise ValueError("Global-history configuration fingerprint is invalid")
    _settings(
        model_seed=seed,
        updates=common["updates"],
        prefix_frames=common["prefix_frames"],
        diffusion_steps=common["diffusion_steps"],
        evaluations=common["evaluations"],
        generation_seed=common["generation_seed"],
        learning_rate=common["learning_rate"],
        batch_size=common["batch_size"],
        training_seeds=tuple(common["training_seeds"]),
        heldout_seeds=tuple(common["heldout_seeds"]),
    )
    if (config.get("source") != "sonumio-adjacent-dsp-window-pairs-v1"
            or config.get("architecture") != "sonumio-global-history-v-denoiser-v1"
            or config.get("objective") != "future_only_velocity_mse"
            or config.get("schedule") != "cosine_zero_terminal_v1"
            or report.get("global_torch_rng_isolated") is not True
            or report.get("audio_or_model_weights_saved") is not False
            or report.get("model_publication_authorized") is not False
            or report.get("speech_music_or_long_context_quality_established") is not False):
        raise ValueError("Global-history study made an unauthorized model claim")
    unique_train = 21 * len(common["training_seeds"])
    unique_heldout = 21 * len(common["heldout_seeds"])
    if (split.get("train_independent_recordings") != unique_train
            or split.get("heldout_independent_recordings") != unique_heldout
            or split.get("train_caption_examples") != 2 * unique_train
            or split.get("heldout_caption_examples") != 2 * unique_heldout
            or split.get("source_recording_groups_not_split_across_languages") is not True
            or split.get("exact_audio_source_leakage_detected") is not False
            or split.get("shared_seed_leakage_detected") is not False
            or split.get("exact_sample_overlap_between_neighbors") is not True
            or split.get("independent_acoustic_families_verified") is not False
            or not _sha(split.get("train_fingerprint"))
            or not _sha(split.get("heldout_fingerprint"))):
        raise ValueError("Incomplete adjacent source split evidence")
    fingerprint = (split["train_fingerprint"], split["heldout_fingerprint"])
    if expected_source is not None and fingerprint != expected_source:
        raise ValueError("Source bytes changed between independent training seeds")
    if (control.get("identical_architecture_and_initial_parameters") is not True
            or control.get("same_future_target_noisy_samples_timesteps_and_tokens") is not True
            or control.get("same_optimizers_lr_gradient_clipping_and_objective") is not True
            or control.get("only_previous_history_information_differs") is not True
            or control.get("wrong_remote_control_keeps_known_prefix_exact") is not True
            or control.get("wrong_remote_control_is_evaluation_only") is not True
            or control.get("bilingual_audio_duplicates_counted_once_per_recording") is not True):
        raise ValueError("Paired global history or remote-control integrity was lost")
    trained = states.get("trained_sha256")
    if (type(trained) is not dict
            or set(trained) != {"full_history", "hidden_history"}
            or not _sha(states.get("identical_initial_sha256"))
            or not _sha(trained["full_history"])
            or not _sha(trained["hidden_history"])
            or trained["full_history"] == trained["hidden_history"]
            or states.get("distinct_trained_states") is not True
            or states.get("unchanged_through_evaluation") is not True):
        raise ValueError("Missing independent trained network integrity")
    if (type(heldout.get("mse_by_condition")) is not dict
            or set(heldout["mse_by_condition"]) != set(METRICS)
            or heldout.get("independent_heldout_recordings") != unique_heldout
            or heldout.get("bilingual_caption_examples_scored") != 2 * unique_heldout
            or heldout.get("future_frames_scored_per_example") != 8192 - common["prefix_frames"]
            or heldout.get("identical_noisy_future_noise_timestep_tokens_for_all_conditions") is not True
            or heldout.get("wrong_remote_history_has_exactly_same_known_local_overlap") is not True
            or heldout.get("descriptive_deltas_not_statistical_significance") is not True
            or heldout.get("human_perception_or_semantic_continuity_measured") is not False):
        raise ValueError("Held-out global history evidence is incomplete")
    mse = {name: _finite(value, minimum=0)
           for name, value in heldout["mse_by_condition"].items()}
    summaries = heldout.get("group_descriptive_deltas")
    if (type(summaries) is not dict
            or set(summaries) != {"hidden_minus_full", "wrong_remote_minus_correct"}):
        raise ValueError("Missing grouped global-history effect summaries")
    effects = {
        "hidden_minus_full": mse["hidden_history"] - mse["full_history"],
        "wrong_remote_minus_correct":
            mse["wrong_remote_history"] - mse["full_history"],
    }
    for key, observed in effects.items():
        stats = summaries[key]
        if (type(stats) is not dict
                or not math.isclose(_finite(stats.get("mean")), observed,
                                    rel_tol=1e-5, abs_tol=1e-8)
                or not _finite(stats.get("min")) <= observed
                <= _finite(stats.get("max"))
                or type(stats.get("positive_recordings")) is not int
                or not 0 <= stats["positive_recordings"] <= unique_heldout):
            raise ValueError("Inconsistent grouped held-out comparison")
    if len(generated) != sum(i < 2 * unique_heldout for i in GENERATION_IDS):
        raise ValueError("Generated holdout sample grid is incomplete")
    generated_groups = set()
    sensitivity = []
    for sample in generated:
        idx = sample.get("heldout_caption_example")
        if (type(idx) is not int or idx not in GENERATION_IDS
                or idx >= 2 * unique_heldout
                or idx in generated_groups
                or not _sha(sample.get("correct_clip_sha256"))
                or not _sha(sample.get("wrong_remote_clip_sha256"))
                or not _sha(sample.get("reference_future_sha256"))
                or not _sha(sample.get("known_prefix_sha256"))
                or sample.get("exact_local_prefix_identity_in_both_outputs") is not True
                or sample.get("human_listening_quality_measured") is not False
                or sample.get("sample_seed") != common["generation_seed"] + idx):
            raise ValueError("Invalid generation fingerprint or malformed remote control")
        generated_groups.add(idx)
        sensitivity.append(_finite(
            sample.get("changed_remote_history_only_future_mae"),
            minimum=0,
        ))
        for field in ("sample_aligned_future_mae",
                      "sample_aligned_future_rmse",
                      "true_first_unknown_sample_jump",
                      "generated_first_unknown_sample_jump"):
            _finite(sample.get(field), minimum=0)
    for arm in ("full_history", "hidden_history"):
        trace = training.get(arm)
        if (type(trace) is not dict
                or trace.get("actual_updates") != common["updates"]
                or trace.get("history_encoder_gradients_checked_finite") is not True):
            raise ValueError("History encoder training evidence missing")
        _finite(trace.get("loss_first"), minimum=0)
        _finite(trace.get("loss_last"), minimum=0)
        _finite(trace.get("history_encoder_gradient_norm_mean"), minimum=0)
    return {
        "data_fingerprint": fingerprint,
        "model_initial": states["identical_initial_sha256"],
        "model_trained": trained,
        "heldout_mse": mse,
        "effects": effects,
        "mean_remote_output_sensitivity": sum(sensitivity) / len(sensitivity),
        "configuration_sha256": expected,
    }


def replicate_studies(
    *, model_seeds: tuple[int, ...] = (17, 29, 41),
    updates: int = 4, prefix_frames: int = 1024,
    diffusion_steps: int = 8, evaluations: int = 4,
    generation_seed: int = 301, learning_rate: float = 0.0005,
    batch_size: int = 2,
    training_seeds: tuple[int, ...] = (0, 1),
    heldout_seeds: tuple[int, ...] = (101, 102),
    runner=run_study,
) -> dict:
    _check_seeds(model_seeds)
    if not callable(runner):
        raise ValueError("Replication requires a deterministic study runner")
    common = {
        "updates": updates, "prefix_frames": prefix_frames,
        "diffusion_steps": diffusion_steps, "evaluations": evaluations,
        "generation_seed": generation_seed,
        "learning_rate": float(learning_rate),
        "batch_size": batch_size,
        "training_seeds": list(training_seeds),
        "heldout_seeds": list(heldout_seeds),
    }
    _settings(
        model_seed=model_seeds[0],
        updates=updates, prefix_frames=prefix_frames,
        diffusion_steps=diffusion_steps, evaluations=evaluations,
        generation_seed=generation_seed, learning_rate=learning_rate,
        batch_size=batch_size, training_seeds=training_seeds,
        heldout_seeds=heldout_seeds,
    )
    used_initial = set()
    used_trained = {arm: set() for arm in ("full_history", "hidden_history")}
    identity = None
    records = []
    for seed in model_seeds:
        study = runner(
            model_seed=seed, updates=updates, prefix_frames=prefix_frames,
            diffusion_steps=diffusion_steps, evaluations=evaluations,
            generation_seed=generation_seed,
            learning_rate=learning_rate, batch_size=batch_size,
            training_seeds=training_seeds, heldout_seeds=heldout_seeds,
        )
        observed = _verified_run(
            study, seed=seed, common=common, expected_source=identity,
        )
        if identity is None:
            identity = observed["data_fingerprint"]
        if observed["model_initial"] in used_initial:
            raise ValueError("Distinct model seeds produced identical initial weights")
        used_initial.add(observed["model_initial"])
        for arm in used_trained:
            current = observed["model_trained"][arm]
            if current in used_trained[arm]:
                raise ValueError("Distinct model seeds produced identical trained weights")
            used_trained[arm].add(current)
        records.append({
            "model_seed": seed,
            "configuration_sha256": observed["configuration_sha256"],
            "model_initial_sha256": observed["model_initial"],
            "trained_state_sha256": observed["model_trained"],
            "paired_heldout_mse": observed["heldout_mse"],
            "descriptive_mse_effects": observed["effects"],
            "remote_history_output_sensitivity_mae": observed[
                "mean_remote_output_sensitivity"],
        })
    effects = {
        key: summarize_values([
            row["descriptive_mse_effects"][key] for row in records
        ])
        for key in ("hidden_minus_full", "wrong_remote_minus_correct")
    }
    effects["remote_history_output_sensitivity_mae"] = summarize_values([
        row["remote_history_output_sensitivity_mae"] for row in records
    ])
    config = {
        "model_seeds": list(model_seeds),
        **common,
        "source_train_sha256": identity[0],
        "source_heldout_sha256": identity[1],
        "single_protocol": SINGLE_PROTOCOL,
    }
    return {
        "protocol": PROTOCOL,
        "configuration": config,
        "configuration_sha256": hashlib.sha256(json.dumps(
            config, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "replicated_model_runs": records,
        "model_seeds_counted_as_independent_replicates": len(records),
        "descriptive_between_model_seed_effects": effects,
        "source_recordings_bitwise_matched_within_runner": True,
        "distinct_model_initialization_fingerprints_verified": True,
        "trained_model_weights_distinct_across_seeds": True,
        "same_training_and_evaluation_configuration_across_runs": True,
        "bilingual_captions_and_noise_seeds_not_independent_replicates": True,
        "statistical_significance_or_sound_quality_winner_declared": False,
        "audio_or_model_weights_saved": False,
        "model_publication_authorized": False,
        "warning": (
            "Only a few independent tiny original-model seeds on a narrow "
            "first-party synthetic audio split. Descriptive neural MSE and "
            "remote-context sensitivity do not prove audible quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seeds", default="17,29,41")
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
    result = replicate_studies(
        model_seeds=tuple(int(x) for x in opts.model_seeds.split(",")),
        updates=opts.updates, prefix_frames=opts.prefix_frames,
        diffusion_steps=opts.diffusion_steps, evaluations=opts.evaluations,
        generation_seed=opts.generation_seed,
        learning_rate=opts.learning_rate, batch_size=opts.batch_size,
        training_seeds=tuple(int(x) for x in opts.training_seeds.split(",")),
        heldout_seeds=tuple(int(x) for x in opts.heldout_seeds.split(",")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
