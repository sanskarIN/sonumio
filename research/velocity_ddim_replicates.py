"""Replicated velocity-DDIM sampler study with strict matching and no assets.

Repeats the self-contained synthetic v-training/paired-step experiment for
independent initialization seeds. Reports *descriptive* within-arm numerical
changes, never a statistical confidence interval or an audio quality winner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

from research.multi_seed_study import _check_seeds, summarize_values
from research.velocity_ddim_study import (
    PROTOCOL as SINGLE_PROTOCOL, _settings, run_study,
)

PROTOCOL = "sonumio-replicated-velocity-ddim-study-v1"
METRICS = (
    "mean_clipping_fraction", "mean_rms",
    "mean_out_of_range_fraction", "mean_mae_to_full_step",
)


def _validated_run(run: dict, *, seed: int, common: dict,
                   budgets: tuple[int, ...], sample_seeds: tuple[int, ...],
                   previous_data: tuple | None) -> tuple:
    """Reject forged or incomparable metrics before aggregating seed effects."""
    if type(run) is not dict or run.get("protocol") != SINGLE_PROTOCOL:
        raise ValueError("Unrecognized single-seed sampler study")
    settings = run.get("configuration")
    training = run.get("training")
    stats = run.get("by_arm_budget")
    if (type(settings) is not dict or type(training) is not dict
            or type(stats) is not dict):
        raise ValueError("Incomplete sampler study fields")
    if settings.get("seed") != seed or any(
        settings.get(key) != value for key, value in common.items()
    ):
        raise ValueError("Model training or sampler configuration drifted")
    if (settings.get("budgets") != list(budgets)
            or settings.get("sample_seeds") != list(sample_seeds)
            or settings.get("objective") != "velocity"
            or settings.get("sampler") != "deterministic_ddim_eta_zero"
            or settings.get("training_corpus") != "sonumio-first-party-synthetic-dsp"):
        raise ValueError("Mixed or incompatible sampler configuration")
    if any(run.get(key) is not False for key in (
        "weights_written", "audio_files_written", "release_authorized",
        "heldout_audio_used_for_training",
        "human_listening_or_prompt_semantics_evaluated",
    )):
        raise ValueError("Forbidden model, audio, or rights claim")
    if (run.get("same_starting_gaussian_for_equal_seeds") is not True
            or run.get("same_trained_parameters_across_budgets") is not True
            or training.get("train_and_validation_exact_overlap") is not False):
        raise ValueError("Study does not establish required paired controls")
    if any(type(training.get(field)) is not int for field in (
        "unique_training_waveforms", "unique_heldout_waveforms",
        "heldout_captions_scored_for_velocity",
    )):
        raise ValueError("Missing held-out split coverage")
    if (training["unique_training_waveforms"] != 42
            or training["unique_heldout_waveforms"] != 42
            or training["heldout_captions_scored_for_velocity"] != 84):
        raise ValueError("Incomplete source-group evaluation")
    hashes = (
        training.get("training_fingerprint"),
        training.get("validation_fingerprint"),
        training.get("unique_training_waveforms"),
        training.get("unique_heldout_waveforms"),
    )
    if (any(type(x) is not str or len(x) != 64 for x in hashes[:2])
            or hashes[0] == hashes[1]
            or (previous_data is not None and hashes != previous_data)):
        raise ValueError("Synthetic dataset fingerprint drift across seeds")
    expected_samples = 2 * len(budgets) * len(sample_seeds) * 4
    if type(run.get("sample_count")) is not int or run["sample_count"] != expected_samples:
        raise ValueError("Sampler did not evaluate the complete paired grid")
    if set(stats) != {"raw", "latent"}:
        raise ValueError("Mixed or missing raw/latent arms")
    for arm in ("raw", "latent"):
        table = stats[arm]
        if type(table) is not dict or set(table) != {str(b) for b in budgets}:
            raise ValueError("Incomplete per-budget numerical summary")
        for budget in budgets:
            item = table[str(budget)]
            if (type(item) is not dict
                    or type(item.get("samples")) is not int
                    or item["samples"] != len(sample_seeds) * 4
                    or type(item.get("model_evaluations_per_sample")) is not int
                    or item["model_evaluations_per_sample"] != budget * (
                        1 if common["guidance"] == 1 else 2
                    )):
                raise ValueError("Inconsistent sample counts or model NFEs")
            for name in METRICS:
                value = item.get(name)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Missing/nonfinite sampler metric")
                if (name in ("mean_clipping_fraction", "mean_out_of_range_fraction")
                        and not 0 <= value <= 1):
                    raise ValueError("Impossible waveform clipping fraction")
            if (item["mean_rms"] < 0 or item["mean_mae_to_full_step"] < 0
                    or (budget == budgets[-1]
                        and item["mean_mae_to_full_step"] != 0)):
                raise ValueError("Invalid full-step self-comparison")
    return hashes


def replicate_studies(
    *, seeds: tuple[int, ...] = (17, 29, 41),
    train_steps: int = 8, codec_steps: int = 4,
    schedule_steps: int = 8, channels: int = 16,
    budgets: tuple[int, ...] = (2, 4, 8),
    sample_seeds: tuple[int, ...] = (201, 202),
    guidance: float = 1.0,
    runner=run_study,
) -> dict:
    """Prespecified equal-data, seed-replicated **numeric** experiment."""
    _check_seeds(seeds)
    _settings(
        seed=seeds[0], train_steps=train_steps,
        codec_steps=codec_steps, schedule_steps=schedule_steps,
        channels=channels, budgets=budgets,
        sample_seeds=sample_seeds, guidance=guidance,
    )
    if not callable(runner):
        raise ValueError("Sampler study runner must be callable")
    common = {
        "train_steps": train_steps, "codec_steps": codec_steps,
        "schedule_steps": schedule_steps, "channels": channels,
        "guidance": guidance, "sample_rate": 8000, "frames": 8192,
    }
    runs, data_identity = [], None
    for seed in seeds:
        report = runner(
            seed=seed, train_steps=train_steps, codec_steps=codec_steps,
            schedule_steps=schedule_steps, channels=channels,
            budgets=budgets, sample_seeds=sample_seeds, guidance=guidance,
        )
        data_identity = _validated_run(
            report, seed=seed, common=common, budgets=budgets,
            sample_seeds=sample_seeds, previous_data=data_identity,
        )
        runs.append(report)
    summary = {}
    for arm in ("raw", "latent"):
        summary[arm] = {}
        for budget in budgets:
            measured = {
                metric: summarize_values([
                    run["by_arm_budget"][arm][str(budget)][metric]
                    for run in runs
                ]) for metric in METRICS
            }
            measured["clipping_fraction_reduction_vs_full"] = summarize_values([
                run["by_arm_budget"][arm][str(budgets[-1])]["mean_clipping_fraction"]
                - run["by_arm_budget"][arm][str(budget)]["mean_clipping_fraction"]
                for run in runs
            ])
            summary[arm][str(budget)] = measured
    configuration = {
        "seeds": list(seeds), **common, "budgets": list(budgets),
        "sample_seeds": list(sample_seeds),
        "training_fingerprint": data_identity[0],
        "validation_fingerprint": data_identity[1],
        "training_recordings": data_identity[2],
        "validation_recordings": data_identity[3],
        "study_protocol": SINGLE_PROTOCOL,
    }
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True,
                   separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {
        "protocol": PROTOCOL,
        "configuration": configuration,
        "configuration_sha256": digest,
        "per_seed": [{
            "seed": seed,
            "configuration_sha256": run["configuration_sha256"],
            "training_heldout_velocity_mse": run["training"]["heldout_velocity_mse"],
            "training_codec_reconstruction_mae":
                run["training"]["heldout_codec_waveform_mae"],
            "by_arm_budget": run["by_arm_budget"],
            "sample_count": run["sample_count"],
        } for seed, run in zip(seeds, runs)],
        "within_arm_descriptive_summary": summary,
        "independent_initialization_seeds": len(seeds),
        "same_first_party_dataset_within_execution": True,
        "same_training_protocol_except_model_seed": True,
        "confidence_interval_or_significance_claimed": False,
        "audio_quality_winner_declared": False,
        "semantic_sound_class_accuracy_measured": False,
        "weight_or_audio_files_saved": False,
        "model_publication_authorized": False,
        "warning": (
            "Only small, nonconverged DSP-synthetic 8-kHz models. "
            "Lower clipping or numerical DDIM trajectory error is not proof "
            "of audible quality or prompt adherence; the observed spread "
            "is not a confidence interval. Same-host source hashes do not "
            "establish cross-CPU reproducibility."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="17,29,41")
    parser.add_argument("--train-steps", type=int, default=8)
    parser.add_argument("--codec-steps", type=int, default=4)
    parser.add_argument("--schedule-steps", type=int, default=8)
    parser.add_argument("--budgets", default="2,4,8")
    parser.add_argument("--sample-seeds", default="201,202")
    parser.add_argument("--guidance", type=float, default=1.0)
    args = parser.parse_args()
    result = replicate_studies(
        seeds=tuple(int(x) for x in args.seeds.split(",")),
        train_steps=args.train_steps, codec_steps=args.codec_steps,
        schedule_steps=args.schedule_steps,
        budgets=tuple(int(x) for x in args.budgets.split(",")),
        sample_seeds=tuple(int(x) for x in args.sample_seeds.split(",")),
        guidance=args.guidance,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
