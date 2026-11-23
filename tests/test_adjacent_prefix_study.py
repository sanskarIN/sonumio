"""Tests for the real adjacent-history versus hidden-history training pilot."""
from __future__ import annotations

import hashlib
import json
import math

import pytest
torch = pytest.importorskip("torch")
from torch import nn

from research.adjacent_audio import (
    adjacent_bilingual_dataset, audit_adjacent_split,
)
from research.adjacent_prefix_study import (
    PROTOCOL, _draw_batch, _future_loss, _heldout,
    _history_for, _settings, run_study,
)
from research.latent_audio import FRAMES
from research.text_audio import TextWaveDenoiser
from research.wave_diffusion import DiffusionSchedule


def minimal_settings() -> dict:
    return {
        "model_seed": 17,
        "updates": 1,
        "prefix_frames": 512,
        "diffusion_steps": 8,
        "evaluations": 2,
        "generation_seed": 301,
        "learning_rate": 0.0005,
        "batch_size": 2,
        "training_seeds": (0,),
        "heldout_seeds": (101,),
    }


@pytest.fixture(scope="module")
def study():
    return run_study(**minimal_settings())


@pytest.fixture(scope="module")
def sources():
    return adjacent_bilingual_dataset((101,), prefix_frames=512)


@pytest.mark.parametrize("bad", [
    {"model_seed": True}, {"model_seed": -1},
    {"model_seed": 2**32},
    {"updates": 0}, {"updates": True}, {"updates": 129},
    {"prefix_frames": 127}, {"prefix_frames": 4097},
    {"prefix_frames": 511}, {"prefix_frames": True},
    {"diffusion_steps": 7}, {"diffusion_steps": 33},
    {"diffusion_steps": True},
    {"evaluations": 1}, {"evaluations": 9},
    {"evaluations": True},
    {"generation_seed": True},
    {"generation_seed": -1},
    {"generation_seed": 2**32},
    {"learning_rate": 0}, {"learning_rate": -0.5},
    {"learning_rate": 1}, {"learning_rate": float("inf")},
    {"learning_rate": float("nan")}, {"learning_rate": True},
    {"batch_size": 0}, {"batch_size": 9}, {"batch_size": True},
    {"training_seeds": (0, 0)},
    {"training_seeds": ()},
    {"heldout_seeds": ()},
    {"heldout_seeds": (101, 101)},
    {"heldout_seeds": (0,)},
    {"training_seeds": (True,)},
    {"training_seeds": ("0",)},
    {"training_seeds": 0},
])
def test_reject_invalid_training_configurations_before_sampling(bad):
    with pytest.raises(ValueError):
        _settings(**(minimal_settings() | bad))


def test_real_training_produces_strict_numeric_nonpublication_report(study):
    report = study
    assert report["protocol"] == PROTOCOL
    assert len(report["configuration_sha256"]) == 64
    assert report["configuration"]["training_seeds"] == [0]
    assert report["configuration"]["heldout_seeds"] == [101]
    assert report["configuration"]["prefix_frames"] == 512
    assert report["split"]["train_independent_recordings"] == 21
    assert report["split"]["heldout_independent_recordings"] == 21
    assert report["split"]["train_caption_examples"] == 42
    assert report["split"]["heldout_caption_examples"] == 42
    assert report["split"]["exact_sample_overlap_between_neighbors"] is True
    assert report["split"]["independent_acoustic_families_verified"] is False
    assert report["split"]["exact_audio_source_leakage_detected"] is False
    assert report["ground_truth_continuity"]["recordings_counted_once"] == 21
    assert report["ground_truth_continuity"]["caption_examples"] == 42
    assert report["controls"]["same_initial_parameters"] is True
    assert report["controls"]["same_model_capacity_optimizer_and_learning_rate"] is True
    assert report["controls"][
        "identical_minibatches_future_noise_timestep_target_and_tokens"
    ] is True
    assert report["controls"]["future_only_v_objective_for_both_models"] is True
    assert report["controls"]["only_history_observability_differs"] is True
    assert report["controls"]["incorrect_history_is_evaluation_only"] is True
    assert report["controls"]["same_heldout_corruption_for_all_conditions"] is True
    assert report["controls"]["bilingual_captions_share_audio_and_not_independent"]
    state = report["model_states"]
    assert state["distinct_trained_states"] is True
    assert state["unchanged_through_evaluation"] is True
    assert len(state["initial_shared_sha256"]) == 64
    assert state["trained"]["true_history"] != state["trained"]["hidden_history"]
    for arm in ("true_history", "hidden_history"):
        assert len(state["trained"][arm]) == 64
        assert state["trained"][arm] != state["initial_shared_sha256"]
        train = report["training_loss"][arm]
        assert train["steps"] == 1
        assert math.isfinite(train["first"]) and train["first"] > 0
        assert math.isfinite(train["last"]) and train["last"] > 0
    assert report["global_torch_rng_isolated"] is True
    assert report["no_audio_or_weights_saved"] is True
    assert report["model_release_authorized"] is False
    assert report["speech_music_or_general_prompt_quality_established"] is False
    assert report["long_context_continuation_established"] is False
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
    assert len(encoded) < 35_000
    assert "base64" not in encoded


def test_heldout_noise_is_paired_across_models_and_wrong_history(study):
    metrics = study["heldout_future_objective"]
    assert metrics["objective"] == "future_only_velocity_mse"
    assert metrics["independent_recordings_scored"] == 21
    assert metrics["bilingual_caption_examples_scored"] == 42
    assert metrics["scored_future_frames_per_caption"] == FRAMES - 512
    assert metrics["identical_future_targets_noise_timesteps_and_tokens"] is True
    assert metrics["heldout_prefixes_sampled_from_same_adjacent_sources"] is True
    assert metrics["mismatched_context_rotates_by_acoustic_group"] is True
    assert metrics["statistical_significance_or_perceptual_quality_measured"] is False
    for key in ("true_history", "hidden_history", "wrong_history"):
        assert math.isfinite(metrics[key]) and metrics[key] > 0
    assert metrics["hidden_minus_true"] == pytest.approx(
        metrics["hidden_history"] - metrics["true_history"]
    )
    assert metrics["wrong_minus_true"] == pytest.approx(
        metrics["wrong_history"] - metrics["true_history"]
    )
    for summary in ("group_descriptive_hidden_minus_true",
                    "group_descriptive_wrong_minus_true"):
        stats = metrics[summary]
        assert all(math.isfinite(stats[key]) for key in ("mean", "min", "max"))
        assert stats["min"] <= stats["mean"] <= stats["max"]


def test_heldout_generation_is_sample_aligned_not_claimed_perceptual(study):
    rows = study["generation_diagnostics"]
    assert len(rows) == 4
    assert [item["heldout_caption_index"] for item in rows] == [0, 6, 28, 54][:4] or [
        item["heldout_caption_index"] for item in rows
    ] == [0, 6, 28]
    for record in rows:
        assert record["heldout_caption_index"] in (0, 6, 28, 54)
        assert record["language"] in ("en", "hi")
        assert len(record["recording_group_sha256"]) == 64
        assert len(record["known_history_sha256"]) == 64
        assert len(record["true_future_sha256"]) == 64
        assert len(record["sampled_full_clip_sha256"]) == 64
        assert record["exact_input_prefix_preserved"] is True
        assert record["sample_aligned_error_not_a_perceptual_score"] is True
        assert record["audible_quality_or_semantic_coherence_assessed"] is False
        for key in (
            "reference_first_future_sample_jump",
            "generated_first_future_sample_jump",
            "future_sample_aligned_mae",
            "future_sample_aligned_rmse",
        ):
            assert math.isfinite(record[key]) and record[key] >= 0


def test_true_and_hidden_context_construct_different_inputs(sources):
    idx = torch.tensor([0, 2], dtype=torch.long)
    actual = sources.previous[idx, :, -512:]
    correct = _history_for(sources, idx, mode="true_history")
    hidden = _history_for(sources, idx, mode="hidden_history")
    rotated = _history_for(sources, idx, mode="wrong_history")
    assert torch.equal(actual, correct)
    assert torch.count_nonzero(hidden) == 0
    assert correct.shape == hidden.shape == rotated.shape == (2, 1, 512)
    assert not torch.equal(correct, hidden)
    assert not torch.equal(correct, rotated)
    with pytest.raises(ValueError):
        _history_for(sources, idx, mode="invented")


def test_draw_batch_preserves_future_target_and_noise(sources):
    from research.velocity_prediction import velocity_target
    from research.velocity_prefix_continuation import _anchor

    schedule = DiffusionSchedule(8)
    idx = torch.tensor([0, 2], dtype=torch.long)
    gen1 = torch.Generator(device="cpu").manual_seed(712)
    gen2 = torch.Generator(device="cpu").manual_seed(712)
    first = _draw_batch(sources, idx, schedule, generator=gen1)
    repeated = _draw_batch(sources, idx, schedule, generator=gen2)
    assert all(torch.equal(a, b) for a, b in zip(first, repeated))
    mixed, target, tokens, timestep = first
    assert mixed.shape == target.shape == (2, 1, FRAMES)
    assert tokens.shape == (2, 130)
    assert timestep.shape == (2,)
    assert torch.isfinite(mixed).all() and torch.isfinite(target).all()
    assert torch.equal(
        _anchor(mixed, _history_for(sources, idx, mode="true_history"))[:, :, :512],
        sources.future[idx, :, :512],
    )
    assert not torch.equal(
        _anchor(mixed, _history_for(sources, idx, mode="hidden_history"))[:, :, :512],
        sources.future[idx, :, :512],
    )


def test_future_loss_matches_manual_future_only_objective_and_gradients(sources):
    from research.velocity_prefix_continuation import _anchor
    schedule = DiffusionSchedule(8)
    torch.manual_seed(812)
    network = TextWaveDenoiser(16)
    idx = torch.tensor([0, 2], dtype=torch.long)
    mixed, target, tokens, t = _draw_batch(
        sources, idx, schedule,
        generator=torch.Generator(device="cpu").manual_seed(99),
    )
    known = _history_for(sources, idx, mode="true_history")
    visible = _anchor(mixed, known)
    loss = _future_loss(
        network, visible, target, t, tokens, prefix_frames=512,
    )
    assert math.isfinite(loss.item()) and loss.item() > 0
    with torch.no_grad():
        predicted = network(visible, t, tokens)
        expected = (predicted[..., 512:] - target[..., 512:]).square().mean()
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in network.parameters()
    )


def test_incompatible_heldout_models_refused(sources):
    with pytest.raises(ValueError, match="two"):
        _heldout({"wrong": nn.Linear(1, 1)}, sources, DiffusionSchedule(8),
                 model_seed=17)


@pytest.mark.parametrize("change", [
    {"training_seeds": (0, 1), "heldout_seeds": (1,)},
    {"training_seeds": (True,)},
    {"heldout_seeds": (2**32,)},
    {"training_seeds": [0]},
    {"heldout_seeds": "101"},
])
def test_study_rejects_source_split_overlap_and_malformed_seeds(change):
    with pytest.raises(ValueError):
        _settings(**(minimal_settings() | change))
