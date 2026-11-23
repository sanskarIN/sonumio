"""Bounded end-to-end synthetic velocity-DDIM study regression tests."""
from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from research.velocity_ddim_study import (
    PROTOCOL, _settings, _summary, run_study,
)


def _default():
    return {
        "seed": 11, "train_steps": 1, "codec_steps": 1,
        "schedule_steps": 8, "channels": 16, "budgets": (2, 8),
        "sample_seeds": (31,), "guidance": 1.0,
    }


@pytest.mark.parametrize("overrides", [
    {"seed": -1},
    {"seed": True},
    {"seed": 2**32},
    {"train_steps": 0},
    {"train_steps": True},
    {"train_steps": 65},
    {"codec_steps": 0},
    {"codec_steps": 65},
    {"schedule_steps": 7},
    {"schedule_steps": 33},
    {"channels": 32},
    {"budgets": (2,)},
    {"budgets": (8, 2)},
    {"budgets": (2, 2, 8)},
    {"budgets": (2, 4)},
    {"budgets": (True, 8)},
    {"budgets": (2.0, 8)},
    {"budgets": (2, 4, 5, 6, 8)},
    {"sample_seeds": ()},
    {"sample_seeds": (2, 2)},
    {"sample_seeds": (1, 2, 3, 4)},
    {"sample_seeds": (True,)},
    {"sample_seeds": (2**32,)},
    {"guidance": float("inf")},
    {"guidance": True},
    {"guidance": -1},
    {"guidance": 5},
])
def test_invalid_budgets_and_training_settings_fail_before_training(overrides):
    with pytest.raises(ValueError):
        _settings(**(_default() | overrides))


def test_real_synthetic_velocity_training_and_paired_inference_has_safe_report():
    before = torch.random.get_rng_state().clone()
    settings = _default()
    report = run_study(**settings)
    assert torch.equal(before, torch.random.get_rng_state())
    assert report["protocol"] == PROTOCOL
    assert report["configuration"]["objective"] == "velocity"
    assert report["configuration"]["sampler"] == "deterministic_ddim_eta_zero"
    assert report["configuration"]["training_corpus"] == "sonumio-first-party-synthetic-dsp"
    assert report["configuration"]["frames"] == 8192
    assert report["training"]["unique_training_waveforms"] == 42
    assert report["training"]["unique_heldout_waveforms"] == 42
    assert report["training"]["paired_bilingual_captions_per_split"] == 84
    assert report["training"]["train_and_validation_exact_overlap"] is False
    assert report["training"]["training_fingerprint"] != report["training"]["validation_fingerprint"]
    assert report["sample_count"] == 16  # 2 arms x 2 budgets x 4 bilingual prompts
    assert len(report["samples"]) == 16
    assert len(report["prompt_change_controls"]) == 8
    assert len(report["configuration_sha256"]) == 64
    assert report["same_starting_gaussian_for_equal_seeds"]
    assert report["same_trained_parameters_across_budgets"]
    assert report["heldout_audio_used_for_training"] is False
    assert report["weights_written"] is False
    assert report["audio_files_written"] is False
    assert report["release_authorized"] is False
    assert report["human_listening_or_prompt_semantics_evaluated"] is False
    assert all(math.isfinite(v) for v in report["training"]["heldout_velocity_mse"].values())
    # Numeric report must contain NO binary waveform or model-parameter blobs.
    encoded = json.dumps(report, allow_nan=False, ensure_ascii=False)
    assert "model_weights" not in encoded
    assert "base64" not in encoded
    assert len(encoded) < 200_000
    for arm in ("raw", "latent"):
        assert set(report["by_arm_budget"][arm]) == {"2", "8"}
        assert report["by_arm_budget"][arm]["2"]["model_evaluations_per_sample"] == 2
        assert report["by_arm_budget"][arm]["8"]["model_evaluations_per_sample"] == 8
        assert report["by_arm_budget"][arm]["8"]["mean_mae_to_full_step"] == 0
        for record in report["samples"]:
            if record["arm"] != arm:
                continue
            assert len(record["postclamp_sha256"]) == 64
            assert math.isfinite(record["postclamp_rms"])
            assert math.isfinite(record["peak_before_external_limiter"])
            assert math.isfinite(record["mae_to_full_step_ddim"])
            assert 0 <= record["postclamp_clipping_fraction"] <= 1
            assert 0 <= record["out_of_range_fraction"] <= 1
            if arm == "latent":
                assert 0 <= record["latent_pretanh_saturation_fraction"] <= 1
            else:
                assert "latent_pretanh_saturation_fraction" not in record
            if record["budget"] == 8:
                assert record["identical_to_full_step"] is True
                assert record["mae_to_full_step_ddim"] == 0.0
    assert all(
        control["semantic_sound_recognition_evaluated"] is False
        for control in report["prompt_change_controls"]
    )


def test_budget_summary_rejects_empty_or_forged_data():
    with pytest.raises(IndexError):
        _summary([], budgets=(2, 8))
