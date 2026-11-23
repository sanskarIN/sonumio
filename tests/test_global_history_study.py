"""Original full-history neural training, remote-context controls and checks."""
from __future__ import annotations

import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.adjacent_audio import adjacent_bilingual_dataset
from research.adjacent_prefix_study import _draw_batch
from research.global_history_denoiser import GlobalHistoryVelocityDenoiser
from research.global_history_study import (
    PROTOCOL, ARMS, _history_context, _model_input,
    _forward_loss, run_study,
)
from research.wave_diffusion import DiffusionSchedule
from research.latent_audio import FRAMES


def options() -> dict:
    return dict(
        model_seed=17, updates=1, prefix_frames=512,
        diffusion_steps=8, evaluations=2,
        generation_seed=301, learning_rate=0.0005,
        batch_size=2,
        training_seeds=(0,), heldout_seeds=(101,),
    )


@pytest.fixture(scope="module")
def study():
    return run_study(**options())


@pytest.fixture(scope="module")
def audio():
    return adjacent_bilingual_dataset((101,), prefix_frames=512)


def test_real_global_history_training_data_and_model_provenance(study):
    assert study["protocol"] == PROTOCOL
    config = study["configuration"]
    assert config["model_seed"] == 17
    assert config["updates"] == 1
    assert config["prefix_frames"] == 512
    assert config["training_seeds"] == [0]
    assert config["heldout_seeds"] == [101]
    assert len(study["configuration_sha256"]) == 64
    source = study["source_split"]
    assert source["train_independent_recordings"] == 21
    assert source["heldout_independent_recordings"] == 21
    assert source["train_caption_examples"] == 42
    assert source["heldout_caption_examples"] == 42
    assert source["exact_sample_overlap_between_neighbors"] is True
    assert source["exact_audio_source_leakage_detected"] is False
    assert source["source_recording_groups_not_split_across_languages"] is True
    assert source["independent_acoustic_families_verified"] is False
    assert source["train_fingerprint"] != source["heldout_fingerprint"]
    assert study["ground_truth_continuity"]["recordings_counted_once"] == 21
    model = study["neural_model_states"]
    assert len(model["identical_initial_sha256"]) == 64
    assert model["distinct_trained_states"] is True
    assert model["unchanged_through_evaluation"] is True
    assert set(model["trained_sha256"]) == set(ARMS)
    assert all(len(v) == 64 for v in model["trained_sha256"].values())
    assert model["trained_sha256"]["full_history"] != model[
        "trained_sha256"]["hidden_history"]
    assert all(v != model["identical_initial_sha256"]
               for v in model["trained_sha256"].values())


def test_actual_neural_history_training_controls_and_gradients(study):
    controls = study["controls"]
    assert controls["identical_architecture_and_initial_parameters"] is True
    assert controls["same_future_target_noisy_samples_timesteps_and_tokens"] is True
    assert controls["same_optimizers_lr_gradient_clipping_and_objective"] is True
    assert controls["only_previous_history_information_differs"] is True
    assert controls["wrong_remote_control_keeps_known_prefix_exact"] is True
    assert controls["wrong_remote_control_is_evaluation_only"] is True
    assert controls["bilingual_audio_duplicates_counted_once_per_recording"] is True
    for arm in ARMS:
        metrics = study["training"][arm]
        assert metrics["actual_updates"] == 1
        assert metrics["history_encoder_gradients_checked_finite"] is True
        assert math.isfinite(metrics["loss_first"]) and metrics["loss_first"] > 0
        assert math.isfinite(metrics["loss_last"]) and metrics["loss_last"] > 0
        assert math.isfinite(metrics["history_encoder_gradient_norm_mean"])
        assert metrics["history_encoder_gradient_norm_mean"] >= 0


def test_wrong_remote_context_preserves_local_overlap_but_changes_earlier_history(audio):
    indices = torch.tensor([0, 2], dtype=torch.long)
    correct = _history_context(audio, indices, condition="full_history")
    hidden = _history_context(audio, indices, condition="hidden_history")
    wrong = _history_context(audio, indices, condition="wrong_remote_history")
    assert correct.shape == hidden.shape == wrong.shape == (2, 1, FRAMES)
    assert torch.count_nonzero(hidden) == 0
    assert not torch.equal(correct, wrong)
    assert torch.equal(correct[..., -512:], wrong[..., -512:])
    assert not torch.equal(correct[..., :-512], wrong[..., :-512])
    mixed = torch.randn(2, 1, FRAMES)
    actual = _model_input(mixed, correct, prefix_frames=512)
    swapped = _model_input(mixed, wrong, prefix_frames=512)
    null = _model_input(mixed, hidden, prefix_frames=512)
    assert torch.equal(actual, swapped)
    assert not torch.equal(actual, null)
    assert torch.equal(actual[..., :512], correct[..., -512:])
    with pytest.raises(ValueError):
        _history_context(audio, indices, condition="invalid")


def test_trainable_global_history_encoder_has_real_future_loss_gradients(audio):
    torch.manual_seed(51)
    model = GlobalHistoryVelocityDenoiser(16)
    ids = torch.tensor([0, 2], dtype=torch.long)
    schedule = DiffusionSchedule(8)
    before = torch.random.get_rng_state().clone()
    mixed, target, tokens, t = _draw_batch(
        audio, ids, schedule,
        generator=torch.Generator(device="cpu").manual_seed(22),
    )
    context = _history_context(audio, ids, condition="full_history")
    future_loss = _forward_loss(
        model, mixed, target, t, tokens, context,
        prefix_frames=512,
    )
    assert torch.isfinite(future_loss) and future_loss > 0
    future_loss.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in model.history_encoder.parameters()
    )
    assert torch.equal(before, torch.random.get_rng_state())


def test_true_hidden_and_wrong_remote_share_heldout_future_and_noise(study):
    stats = study["heldout_future_objective"]
    assert stats["objective"] == "future_only_velocity_mse"
    assert stats["independent_heldout_recordings"] == 21
    assert stats["bilingual_caption_examples_scored"] == 42
    assert stats["future_frames_scored_per_example"] == FRAMES - 512
    assert stats["identical_noisy_future_noise_timestep_tokens_for_all_conditions"] is True
    assert stats["wrong_remote_history_has_exactly_same_known_local_overlap"] is True
    assert stats["wrong_remote_control_uses_next_acoustic_recording"] is True
    assert stats["same_language_audio_duplicates_counted_once_as_groups"] is True
    assert stats["descriptive_deltas_not_statistical_significance"] is True
    assert stats["human_perception_or_semantic_continuity_measured"] is False
    errors = stats["mse_by_condition"]
    assert set(errors) == {"full_history", "hidden_history", "wrong_remote_history"}
    assert all(math.isfinite(v) and v >= 0 for v in errors.values())
    for key, diff in (
        ("hidden_minus_full", errors["hidden_history"] - errors["full_history"]),
        ("wrong_remote_minus_correct",
         errors["wrong_remote_history"] - errors["full_history"]),
    ):
        desc = stats["group_descriptive_deltas"][key]
        assert desc["mean"] == pytest.approx(diff, rel=1e-5, abs=1e-8)
        assert desc["min"] <= desc["mean"] <= desc["max"]
        assert 0 <= desc["positive_recordings"] <= 21


def test_same_seed_generated_future_controls_have_exact_prefix_and_numeric_quality_limits(study):
    generated = study["numerical_generated_future_probes"]
    assert [row["heldout_caption_example"] for row in generated] == [0, 6, 28]
    for record in generated:
        assert record["language"] == "en"
        assert record["sample_seed"] == 301 + record["heldout_caption_example"]
        for field in (
            "recording_group_sha256", "known_prefix_sha256",
            "correct_clip_sha256", "wrong_remote_clip_sha256",
            "reference_future_sha256",
        ):
            assert len(record[field]) == 64
        assert record["exact_local_prefix_identity_in_both_outputs"] is True
        assert record["human_listening_quality_measured"] is False
        for name in (
            "generated_first_unknown_sample_jump",
            "true_first_unknown_sample_jump",
            "sample_aligned_future_mae",
            "sample_aligned_future_rmse",
            "changed_remote_history_only_future_mae",
        ):
            assert math.isfinite(record[name]) and record[name] >= 0


def test_study_no_weights_published_and_serialized_only_numeric_metadata(study):
    assert study["global_torch_rng_isolated"] is True
    assert study["audio_or_model_weights_saved"] is False
    assert study["model_publication_authorized"] is False
    assert study["speech_music_or_long_context_quality_established"] is False
    serialized = json.dumps(study, allow_nan=False, ensure_ascii=False)
    assert len(serialized) < 50_000
    assert "base64" not in serialized


def test_training_and_evaluation_preserve_global_random_rng():
    before = torch.random.get_rng_state().clone()
    run_study(**options())
    assert torch.equal(before, torch.random.get_rng_state())
