"""Adversarial and real-training tests for paired EMA-decay inference studies."""
from __future__ import annotations

import copy
import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.velocity_ema_decay_ablation import (
    PROTOCOL, _verify_run, _validate_settings, run_ablation,
)
from research.velocity_ema_study import run_study


def options() -> dict:
    return {
        "seeds": (17, 29), "decays": (0.0, 0.5),
        "steps": 1, "codec_steps": 1,
        "diffusion_steps": 8, "channels": 16,
        "warmup": False, "sampling_seeds": (301,),
    }


def verify_options() -> dict:
    return {
        "seed": 17, "decay": 0.0, "steps": 1,
        "codec_steps": 1, "diffusion_steps": 8,
        "channels": 16, "warmup": False,
        "sampling_seeds": (301,),
    }


@pytest.fixture(scope="module")
def zero_reference() -> dict:
    return run_study(**verify_options())


@pytest.mark.parametrize("overrides", [
    {"seeds": ()}, {"seeds": (17,)}, {"seeds": (17, 17)},
    {"seeds": (17, True)}, {"seeds": (17, 2**32)},
    {"decays": (0.5, 0.9)}, {"decays": (0.0,)},
    {"decays": (0.0, 0.0)}, {"decays": (0.0, 0.9, 0.5)},
    {"decays": (0.0, True)}, {"decays": (0.0, float("nan"))},
    {"decays": (0.0, 1.0)}, {"decays": (0.0, -0.01)},
    {"steps": 0}, {"steps": True}, {"steps": 65},
    {"codec_steps": 0}, {"codec_steps": 17},
    {"diffusion_steps": 7}, {"channels": 32},
    {"sampling_seeds": ()}, {"sampling_seeds": (2, 2)},
    {"sampling_seeds": (True,)}, {"warmup": "yes"},
])
def test_reject_invalid_ablation_settings_before_running(overrides):
    with pytest.raises(ValueError):
        _validate_settings(**(options() | overrides))


def test_correct_zero_decay_reference_has_full_source_and_null_control(zero_reference):
    checked = _verify_run(zero_reference, **verify_options())
    assert len(checked["config_sha"]) == 64
    assert checked["data_fingerprint"][0] != checked["data_fingerprint"][1]
    for arm in ("raw", "latent"):
        assert checked["outcomes"][arm]["mse_delta"] == 0.0
        assert checked["outcomes"][arm]["clipping_delta"] == 0.0
        assert checked["outcomes"][arm]["online_sha"] == checked["outcomes"][arm]["ema_sha"]
        assert len(checked["online_rows"][arm]) == 4


@pytest.mark.parametrize(("path", "value"), [
    (("protocol",), "wrong-protocol"),
    (("configuration", "seed"), 18),
    (("configuration", "decay"), 0.5),
    (("configuration_sha256",), "0" * 64),
    (("training", "independent_training_recordings"), 41),
    (("training", "train_fingerprint"), "garbage"),
    (("training", "train_heldout_exact_overlap"), True),
    (("heldout_velocity_objective", "raw", "bilingual_caption_examples"), 4),
    (("heldout_velocity_objective", "latent", "shared_noise_and_timesteps"), False),
    (("heldout_velocity_objective", "raw", "ema"), float("nan")),
    (("heldout_velocity_objective", "latent", "ema_minus_online"), 1.0),
    (("model_state_integrity", "raw", "ema_updates"), 0),
    (("model_state_integrity", "raw", "online_model_sha256"), "zz" * 32),
    (("no_model_weights_or_audio_saved",), False),
    (("model_publication_authorized",), True),
    (("listening_or_semantic_audio_quality_evaluated",), True),
    (("by_arm", "raw", "generated_pairs_compared"), 1),
    (("by_arm", "raw", "mean_ema_clipping_fraction"), 0.75),
])
def test_reject_malformed_or_misleading_ema_evidence(zero_reference, path, value):
    forged = copy.deepcopy(zero_reference)
    item = forged
    for key in path[:-1]:
        item = item[key]
    item[path[-1]] = value
    with pytest.raises(ValueError):
        _verify_run(forged, **verify_options())


@pytest.mark.parametrize(("path", "value"), [
    (("seed",), 100),
    (("language",), "fr"),
    (("prompt_adherence_or_human_quality_measured",), True),
    (("online", "prelimiter_signal_source"), "unknown"),
    (("online", "postclip", "sha256"), "a" * 8),
    (("online", "postclip", "rms"), float("nan")),
    (("online", "postclip", "clipping_fraction"), 1.3),
    (("ema", "postclip", "sha256"), "f" * 64),
    (("online_ema_waveform_difference", "exact_same_waveform"), False),
    (("online_ema_waveform_difference", "mean_absolute_difference"), 0.2),
])
def test_zero_decay_comparison_detects_forged_waveform_records(
    zero_reference, path, value,
):
    forged = copy.deepcopy(zero_reference)
    row = forged["in_memory_generated_numerical_comparisons"]["raw"][0]
    item = row
    for key in path[:-1]:
        item = item[key]
    item[path[-1]] = value
    with pytest.raises(ValueError):
        _verify_run(forged, **verify_options())


def test_complete_real_two_model_seed_two_decay_ablation():
    state_before = torch.random.get_rng_state().clone()
    result = run_ablation(**options())
    assert torch.equal(state_before, torch.random.get_rng_state())
    assert result["protocol"] == PROTOCOL
    assert result["total_matched_runs"] == 4
    assert result["configuration"]["model_seeds"] == [17, 29]
    assert result["configuration"]["ema_decays"] == [0.0, 0.5]
    assert result["same_source_data_across_runs_verified"] is True
    assert result["identical_online_trajectory_within_model_seed_verified"] is True
    assert result["independent_online_model_fingerprints_across_seeds_verified"] is True
    assert result["zero_decay_null_control_passed"] is True
    assert result["no_weights_or_audio_saved"] is True
    assert result["model_release_authorized"] is False
    assert result["perceptual_quality_or_prompt_adherence_measured"] is False
    assert result["statistical_significance_or_quality_winner_declared"] is False
    assert len(result["configuration_sha256"]) == 64
    assert len(result["per_model_seed_and_decay"]) == 4
    encoded = json.dumps(result, allow_nan=False)
    assert len(encoded) < 80000
    for arm in ("raw", "latent"):
        summary = result["within_arm_descriptive_summary"][arm]
        assert summary["0.0"]["heldout_mse_delta_ema_minus_online"]["mean"] == 0
        assert summary["0.0"]["generated_clipping_delta_ema_minus_online"]["mean"] == 0
        assert summary["0.0"]["waveform_mae_between_online_and_ema"]["mean"] == 0
        assert summary["0.5"]["heldout_mse_delta_ema_minus_online"]["count"] == 2
        assert 0 <= summary["0.5"]["ema_reduced_heldout_error_in_models"] <= 2
        assert 0 <= summary["0.5"]["ema_reduced_clipping_in_models"] <= 2
        for record in result["per_model_seed_and_decay"]:
            candidate = record["arms"][arm]
            assert math.isfinite(candidate["mse_delta"])
            assert math.isfinite(candidate["clipping_delta"])
            assert 0 <= candidate["online_clipping"] <= 1
            assert 0 <= candidate["ema_clipping"] <= 1


def test_identical_online_trajectory_gate_rejects_drift_across_decay(
    zero_reference,
):
    # Return valid but separately produced model-seed records. Corrupt the
    # evidence hash on the second decay to simulate a hidden training change.
    cached = {}
    for seed in (17, 29):
        for decay in (0.0, 0.5):
            cached[(seed, decay)] = run_study(**(
                verify_options() | {"seed": seed, "decay": decay}
            ))

    def changed_online(seed, decay, **kwargs):
        result = copy.deepcopy(cached[(seed, decay)])
        if decay != 0:
            result["model_state_integrity"]["raw"]["online_model_sha256"] = "1" * 64
        return result

    with pytest.raises(ValueError, match="Changing EMA decay changed"):
        run_ablation(**options(), runner=changed_online)
