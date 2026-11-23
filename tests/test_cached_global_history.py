"""Exact cached-vs-uncached original history-DDIM waveform parity tests."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math

import pytest
torch = pytest.importorskip("torch")

from research.cached_global_history import (
    PROTOCOL, INFERENCE_PROTOCOL, PreparedHistory,
    prepare_history, sample_cached_global_history_ddim, parity_check,
)
from research.global_history_denoiser import (
    GlobalHistoryVelocityDenoiser, sample_global_history_ddim,
)
from research.latent_audio import FRAMES
from research.wave_diffusion import DiffusionSchedule
from research.weight_ema import model_state_sha256


def history():
    t = torch.arange(FRAMES, dtype=torch.float32) / 8000.0
    return (
        0.08 * torch.sin(2 * math.pi * 175 * t)
        + 0.04 * torch.cos(2 * math.pi * 383 * t)
    ).float()


@pytest.mark.parametrize("guidance", [1.0, 0.0, 1.5, 4.0])
@pytest.mark.parametrize("budget", [2, 3, 8])
def test_exact_uncached_and_prepared_history_ddim_parity_for_bounded_budgets(
    guidance, budget,
):
    torch.manual_seed(17)
    network = GlobalHistoryVelocityDenoiser(16).train()
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    prior = history()
    original_rng = torch.random.get_rng_state().clone()
    state = model_state_sha256(network)
    baseline, _ = sample_global_history_ddim(
        network, schedule, "बारिश", prior,
        prefix_frames=512, evaluations=budget,
        seed=123, guidance=guidance,
    )
    cached, report = sample_cached_global_history_ddim(
        network, schedule, "बारिश", prior,
        prefix_frames=512, evaluations=budget,
        seed=123, guidance=guidance,
    )
    assert torch.equal(baseline, cached)
    assert report["protocol"] == INFERENCE_PROTOCOL
    assert report["timestep_evaluations"] == budget
    assert report["history_encoder_evaluations_for_this_call"] == 1
    assert report["expected_uncached_history_encoder_evaluations"] == budget * (
        1 if guidance == 1 else 2
    )
    assert report["total_denoiser_model_evaluations"] == report[
        "expected_uncached_history_encoder_evaluations"]
    assert report["history_encoder_calls_avoided"] == (
        report["expected_uncached_history_encoder_evaluations"] - 1
    )
    assert report["text_timestep_and_residual_network_still_evaluated_every_step"]
    assert torch.equal(cached[:512], prior[-512:])
    assert report["known_prefix_preserved_exactly"]
    assert report["normalization_or_clipping_applied"] is False
    assert report["audio_embeddings_or_model_weights_written"] is False
    assert report["model_publication_authorized"] is False
    assert report["latency_speedup_or_audio_quality_claim"] is False
    assert report["source_sha256"] == hashlib.sha256(
        prior.numpy().tobytes()
    ).hexdigest()
    assert report["waveform_sha256"] == hashlib.sha256(
        cached.numpy().tobytes()
    ).hexdigest()
    assert model_state_sha256(network) == state
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert network.training


def test_prepared_history_reuse_avoids_all_encoder_invocations():
    torch.manual_seed(34)
    model = GlobalHistoryVelocityDenoiser(16).train()
    sample = history()
    schedule = DiffusionSchedule(8)
    counter = [0]

    def record_forward(module, inputs, output):
        counter[0] += 1

    hook = model.history_encoder.register_forward_hook(record_forward)
    try:
        prepared = prepare_history(model, sample)
        assert counter[0] == 1
        first, info = sample_cached_global_history_ddim(
            model, schedule, "rain", sample,
            prefix_frames=1024, evaluations=4,
            seed=12, guidance=1.5,
            prepared_history=prepared,
        )
        assert counter[0] == 1
        second, info2 = sample_cached_global_history_ddim(
            model, schedule, "wind", sample,
            prefix_frames=1024, evaluations=4,
            seed=13, guidance=1.5,
            prepared_history=prepared,
        )
        assert counter[0] == 1
        assert not torch.equal(first, second)
        assert info["history_encoder_evaluations_for_this_call"] == 0
        assert info["history_encoder_calls_avoided"] == 8
        assert info["cache_created_during_call"] is False
        assert info2["cache"]["source_sha256"] == info["cache"]["source_sha256"]
        assert info["cache"]["protocol"] == PROTOCOL
        assert info["cache"]["audio_or_embeddings_exported"] is False
        assert info["cache"]["model_weights_saved"] is False
    finally:
        hook.remove()


def test_true_uncached_encoder_invocation_count_matches_cfg_budget():
    model = GlobalHistoryVelocityDenoiser(16)
    counter = [0]
    handle = model.history_encoder.register_forward_hook(
        lambda module, inputs, output: counter.__setitem__(0, counter[0] + 1)
    )
    try:
        for guidance, budget in ((1.0, 3), (1.5, 4)):
            prior = history()
            previous = counter[0]
            sample_global_history_ddim(
                model, DiffusionSchedule(8), "rain", prior,
                prefix_frames=512, evaluations=budget,
                seed=51, guidance=guidance,
            )
            assert counter[0] - previous == budget * (
                1 if guidance == 1 else 2
            )
    finally:
        handle.remove()


def test_prepared_history_model_and_audio_hashes_are_validated():
    model = GlobalHistoryVelocityDenoiser(16)
    first = history()
    prepared = prepare_history(model, first)
    assert prepared.protocol == PROTOCOL
    assert prepared.source_frames == FRAMES
    assert prepared.model_channels == 16
    assert prepared.source_sha256 == hashlib.sha256(
        first.numpy().tobytes()
    ).hexdigest()
    assert prepared.model_sha256 == model_state_sha256(model)
    assert len(prepared.embedding_sha256) == 64
    assert prepared.embedding.shape == (1, 16)
    assert prepared.embedding.dtype == torch.float32
    assert torch.isfinite(prepared.embedding).all()
    assert prepared.embedding.requires_grad is False
    assert torch.equal(prepared.validate(model, first), prepared.embedding)
    described = prepared.describe()
    assert described["source_frames"] == FRAMES
    assert described["embedding_dimension"] == 16
    assert described["audio_or_embeddings_exported"] is False
    assert described["model_publication_authorized"] is False


@pytest.mark.parametrize("change", [
    "content", "model", "embedding", "source-digest",
    "model-digest", "feature-digest", "channels", "protocol",
])
def test_mutated_or_wrongly_associated_prepared_feature_is_rejected(change):
    torch.manual_seed(59)
    model = GlobalHistoryVelocityDenoiser(16)
    source = history()
    cache = prepare_history(model, source)
    if change == "content":
        other = source.clone()
        other[0] += 0.001
        args = (model, other)
    elif change == "model":
        other_model = GlobalHistoryVelocityDenoiser(16)
        args = (other_model, source)
    elif change == "embedding":
        with torch.no_grad():
            cache.embedding[0, 0] += 0.01
        args = (model, source)
    elif change == "source-digest":
        cache = replace(cache, source_sha256="0" * 64)
        args = (model, source)
    elif change == "model-digest":
        cache = replace(cache, model_sha256="0" * 64)
        args = (model, source)
    elif change == "feature-digest":
        cache = replace(cache, embedding_sha256="0" * 64)
        args = (model, source)
    elif change == "channels":
        cache = replace(cache, model_channels=24)
        args = (model, source)
    else:
        cache = replace(cache, protocol="invalid")
        args = (model, source)
    with pytest.raises(ValueError, match="Stale"):
        cache.validate(*args)


def test_one_optimizer_update_invalidates_previous_model_feature():
    torch.manual_seed(71)
    model = GlobalHistoryVelocityDenoiser(16)
    source = history()
    cache = prepare_history(model, source)
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    with pytest.raises(ValueError, match="Stale"):
        cache.validate(model, source)


@pytest.mark.parametrize("invalid_source", [
    torch.zeros(FRAMES - 1),
    torch.zeros(FRAMES + 1),
    torch.zeros(FRAMES, dtype=torch.float64),
    torch.full((FRAMES,), float("nan")),
    torch.full((FRAMES,), float("inf")),
    torch.ones(FRAMES) * 1e13,
    torch.zeros(FRAMES, requires_grad=True),
])
def test_preparing_invalid_audio_is_rejected_before_encoder(invalid_source):
    model = GlobalHistoryVelocityDenoiser(16)
    with pytest.raises(ValueError):
        prepare_history(model, invalid_source)


@pytest.mark.parametrize("overrides", [
    {"seed": -1}, {"seed": True}, {"seed": 2**32},
    {"guidance": -1}, {"guidance": 4.1},
    {"guidance": True}, {"guidance": float("nan")},
    {"prefix_frames": 0}, {"prefix_frames": 127},
    {"prefix_frames": 513}, {"prefix_frames": True},
    {"evaluations": 1}, {"evaluations": 9},
    {"evaluations": True}, {"caption": ""},
    {"prepared_history": {"embedding": "fake"}},
])
def test_cached_sampler_rejects_bad_configuration(overrides):
    network = GlobalHistoryVelocityDenoiser(16)
    options = dict(
        model=network,
        schedule=DiffusionSchedule(8),
        caption="rain", prior_history=history(),
        prefix_frames=512, evaluations=3,
        seed=17, guidance=1.0,
    )
    with pytest.raises(ValueError):
        sample_cached_global_history_ddim(**(options | overrides))


def test_prepared_feature_validation_failure_does_not_advance_rng():
    torch.manual_seed(89)
    model = GlobalHistoryVelocityDenoiser(16)
    input_audio = history()
    cache = prepare_history(model, input_audio)
    state_before = torch.random.get_rng_state().clone()
    invalid = replace(cache, embedding_sha256="a" * 64)
    with pytest.raises(ValueError):
        sample_cached_global_history_ddim(
            model, DiffusionSchedule(8), "wind",
            input_audio, prefix_frames=512,
            evaluations=3, seed=17,
            prepared_history=invalid,
        )
    assert torch.equal(state_before, torch.random.get_rng_state())


@pytest.mark.parametrize("guidance,budget", [
    (1.0, 2), (1.0, 8), (1.5, 2), (2.0, 4),
])
def test_parity_check_refuses_any_difference_and_reports_verified_hashes(
    guidance, budget,
):
    torch.manual_seed(91)
    network = GlobalHistoryVelocityDenoiser(16)
    report = parity_check(
        network, DiffusionSchedule(8), "हवा", history(),
        prefix_frames=512, evaluations=budget, seed=301,
        guidance=guidance,
    )
    assert report["protocol"] == INFERENCE_PROTOCOL
    assert report["exact_bitwise_sampler_parity"] is True
    assert report["uncached_sha256"] == report["cached_sha256"]
    assert report["history_encoder_calls_avoided"] == budget * (
        1 if guidance == 1 else 2
    ) - 1
    assert report["guidance"] == guidance
    assert report["evaluations"] == budget
    assert report["seed"] == 301
    assert report["no_weights_embeddings_or_audio_exported"] is True
    assert report["perceptual_quality_or_real_world_latency_established"] is False
