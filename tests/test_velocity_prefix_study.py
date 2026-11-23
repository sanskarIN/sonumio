"""Actual synthetic masked-future training research controls for Sonumio."""
from __future__ import annotations

import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.velocity_prefix_study import PROTOCOL, _settings, run_study


def tiny() -> dict:
    return {
        "seed": 17, "updates": 1,
        "prefix_frames": 512, "diffusion_steps": 8,
        "evaluations": 2, "sample_seed": 301,
    }


@pytest.mark.parametrize("change", [
    {"seed": True}, {"seed": -1}, {"seed": 2**32},
    {"updates": 0}, {"updates": 65}, {"updates": True},
    {"prefix_frames": 0}, {"prefix_frames": 129},
    {"prefix_frames": 8192}, {"prefix_frames": True},
    {"diffusion_steps": 7}, {"diffusion_steps": 17},
    {"evaluations": 1}, {"evaluations": 9},
    {"evaluations": True}, {"sample_seed": -1},
    {"sample_seed": True}, {"sample_seed": 2**32},
])
def test_invalid_study_settings_rejected_without_training(change):
    with pytest.raises(ValueError):
        _settings(**(tiny() | change))


def test_real_controlled_future_only_synthetic_training_and_heldout():
    before_rng = torch.random.get_rng_state().clone()
    result = run_study(**tiny())
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert result["protocol"] == PROTOCOL
    assert len(result["configuration_sha256"]) == 64
    assert result["configuration"]["prefix_frames"] == 512
    assert result["data"]["independent_training_recordings"] == 42
    assert result["data"]["independent_heldout_recordings"] == 42
    assert result["data"]["bilingual_caption_examples_per_split"] == 84
    assert result["data"]["exact_audio_overlap"] is False
    assert result["data"]["heldout_audio_used_for_training"] is False
    assert result["data"]["train_fingerprint"] != result["data"]["heldout_fingerprint"]
    controls = result["controls"]
    assert controls["same_training_minibatches_timesteps_noise_and_tokens"] is True
    assert controls["same_optimizer_type_and_learning_rate"] is True
    assert controls["only_loss_region_differs"] is True
    assert controls["same_heldout_noisy_input_for_both_models"] is True
    assert controls["distinct_trained_weights"] is True
    assert len(controls["same_initial_model_sha256"]) == 64
    for arm in ("masked_future_loss", "all_frames_loss"):
        assert len(result["trained_model_sha256"][arm]) == 64
        training = result["training_loss_start_end"][arm]
        assert math.isfinite(training["initial"])
        assert math.isfinite(training["final"])
        assert training["initial"] >= 0 and training["final"] >= 0
    heldout = result["heldout_future_velocity_objective"]
    assert heldout["heldout_bilingual_caption_examples"] == 84
    assert heldout["distinct_audio_recordings"] == 42
    assert heldout["scored_future_frames_per_caption"] == 8192 - 512
    assert heldout["identical_noise_timesteps_tokens_and_inputs"] is True
    assert heldout["masked_minus_all"] == pytest.approx(
        heldout["masked_future_loss"] - heldout["all_frames_loss"]
    )
    assert all(math.isfinite(heldout[arm]) and heldout[arm] >= 0
               for arm in ("masked_future_loss", "all_frames_loss"))
    rows = result["prefix_conditioned_numerical_samples"]
    assert len(rows) == 4
    for row in rows:
        assert row["heldout_example_index"] in (0, 6)
        assert row["model_variant"] in ("masked_future_loss", "all_frames_loss")
        assert row["same_input_prefix_for_both_variants"] is True
        assert len(row["prefix_sha256"]) == len(row["output_sha256"]) == 64
        assert math.isfinite(row["first_future_sample_jump"])
        assert math.isfinite(row["unclipped_peak"])
        assert 0 <= row["fraction_above_unit_range"] <= 1
        assert row["perceptual_or_semantic_continuity_evaluated"] is False
    assert result["all_model_states_unchanged_during_evaluation"] is True
    assert result["no_audio_or_weights_saved"] is True
    assert result["model_release_authorized"] is False
    assert result["human_listening_or_semantic_alignment_measured"] is False
    assert result["learned_long_context_or_music_speech_continuity_claim"] is False
    output = json.dumps(result, allow_nan=False, ensure_ascii=False)
    assert len(output) < 15000
    assert "base64" not in output
