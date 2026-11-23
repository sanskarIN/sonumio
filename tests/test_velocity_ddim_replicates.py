"""Strict multi-initialization sampler research and data-provenance tests."""
from __future__ import annotations

import copy
import math
import json
import hashlib

import pytest

torch = pytest.importorskip("torch")

from research.velocity_ddim_replicates import (
    PROTOCOL, replicate_studies,
)


def fake_run(seed: int, *, train_steps: int, codec_steps: int,
             schedule_steps: int, channels: int,
             budgets: tuple[int, ...], sample_seeds: tuple[int, ...],
             guidance: float):
    parts = {}
    for arm in ("raw", "latent"):
        parts[arm] = {}
        for budget in budgets:
            parts[arm][str(budget)] = {
                "samples": len(sample_seeds) * 4,
                "model_evaluations_per_sample":
                    budget * (1 if guidance == 1 else 2),
                "mean_clipping_fraction": 0.25 if budget == budgets[-1] else 0.05,
                "mean_rms": 0.15 + 0.001 * seed,
                "mean_out_of_range_fraction": 0.2 if budget == budgets[-1] else 0.02,
                "mean_mae_to_full_step": 0 if budget == budgets[-1] else 0.05,
            }
    configuration = {
        "seed": seed, "train_steps": train_steps,
        "codec_steps": codec_steps, "schedule_steps": schedule_steps,
        "channels": channels, "guidance": guidance,
        "sample_rate": 8000, "frames": 8192,
        "budgets": list(budgets), "sample_seeds": list(sample_seeds),
        "objective": "velocity", "sampler": "deterministic_ddim_eta_zero",
        "training_corpus": "sonumio-first-party-synthetic-dsp",
    }
    digest = hashlib.sha256(json.dumps(
        configuration, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "protocol": "sonumio-velocity-sampler-budget-study-v1",
        "configuration": configuration,
        "training": {
            "training_fingerprint": "a" * 64,
            "validation_fingerprint": "b" * 64,
            "unique_training_waveforms": 42,
            "unique_heldout_waveforms": 42,
            "heldout_captions_scored_for_velocity": 84,
            "heldout_velocity_mse": {"raw": 0.9, "latent": 1.1},
            "heldout_codec_waveform_mae": 0.2,
            "train_and_validation_exact_overlap": False,
        },
        "by_arm_budget": parts,
        "sample_count": 2 * len(budgets) * len(sample_seeds) * 4,
        "configuration_sha256": digest,
        "same_starting_gaussian_for_equal_seeds": True,
        "same_trained_parameters_across_budgets": True,
        "heldout_audio_used_for_training": False,
        "weights_written": False, "audio_files_written": False,
        "release_authorized": False,
        "human_listening_or_prompt_semantics_evaluated": False,
    }


def settings():
    return {
        "seeds": (17, 29), "train_steps": 1, "codec_steps": 1,
        "schedule_steps": 8, "channels": 16, "budgets": (2, 8),
        "sample_seeds": (51,), "guidance": 1.0,
    }


def test_validated_two_seed_replicates_report_descriptive_effect_not_quality():
    result = replicate_studies(**settings(), runner=fake_run)
    assert result["protocol"] == PROTOCOL
    assert result["independent_initialization_seeds"] == 2
    assert result["configuration"]["training_recordings"] == 42
    assert result["configuration"]["validation_recordings"] == 42
    assert result["confidence_interval_or_significance_claimed"] is False
    assert result["audio_quality_winner_declared"] is False
    assert result["semantic_sound_class_accuracy_measured"] is False
    assert result["weight_or_audio_files_saved"] is False
    assert result["model_publication_authorized"] is False
    assert len(result["configuration_sha256"]) == 64
    assert len(result["per_seed"]) == 2
    assert result["within_arm_descriptive_summary"]["raw"]["2"][
        "clipping_fraction_reduction_vs_full"]["mean"] == pytest.approx(0.2)
    for arm in ("raw", "latent"):
        assert result["within_arm_descriptive_summary"][arm]["8"][
            "mean_mae_to_full_step"]["mean"] == 0


@pytest.mark.parametrize("corruption", [
    ("protocol", "fake"),
    ("configuration.objective", "epsilon"),
    ("configuration.sample_seeds", [23]),
    ("training.training_fingerprint", "b" * 64),
    ("training.unique_heldout_waveforms", 41),
    ("training.heldout_captions_scored_for_velocity", 4),
    ("weights_written", True),
    ("release_authorized", True),
    ("same_starting_gaussian_for_equal_seeds", False),
    ("by_arm_budget.raw.2.mean_clipping_fraction", float("nan")),
    ("by_arm_budget.raw.2.mean_clipping_fraction", 1.2),
    ("by_arm_budget.raw.8.mean_mae_to_full_step", 0.3),
    ("by_arm_budget.latent.8.model_evaluations_per_sample", 10),
    ("sample_count", 4),
])
def test_replicates_fail_closed_on_forged_or_incomparable_reports(corruption):
    key, value = corruption

    def corrupted(**kwargs):
        output = copy.deepcopy(fake_run(**kwargs))
        item = output
        tokens = key.split(".")
        for token in tokens[:-1]:
            item = item[token]
        item[tokens[-1]] = value
        return output

    with pytest.raises(ValueError):
        replicate_studies(**settings(), runner=corrupted)


@pytest.mark.parametrize("override", [
    {"seeds": (17,)},
    {"seeds": (17, 17)},
    {"seeds": (17, True)},
    {"budgets": (2, 4)},
    {"sample_seeds": (51, 51)},
    {"guidance": 5},
    {"train_steps": 0},
])
def test_bad_replicate_settings_rejected_without_training(override):
    with pytest.raises(ValueError):
        replicate_studies(**(settings() | override), runner=fake_run)


def test_real_two_initializations_train_with_same_data_and_preserve_global_rng():
    before = torch.random.get_rng_state().clone()
    report = replicate_studies(**settings())
    assert torch.equal(before, torch.random.get_rng_state())
    assert report["independent_initialization_seeds"] == 2
    assert [run["seed"] for run in report["per_seed"]] == [17, 29]
    assert all(
        math.isfinite(run["training_heldout_velocity_mse"]["raw"])
        for run in report["per_seed"]
    )
    assert report["configuration"]["training_fingerprint"] != report[
        "configuration"]["validation_fingerprint"]
    assert report["weight_or_audio_files_saved"] is False
