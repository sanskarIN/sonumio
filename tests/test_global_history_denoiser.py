"""Full-window neural history encoder and clean-prefix v-DDIM integration tests."""
from __future__ import annotations

import hashlib
import math

import pytest
torch = pytest.importorskip("torch")
from torch import nn

from research.global_history_denoiser import (
    PROTOCOL, INFERENCE_PROTOCOL, SEQUENCE_PROTOCOL,
    FRAMES, _context_inputs, HistoryEncoder, GlobalHistoryVelocityDenoiser,
    masked_history_velocity_loss, sample_global_history_ddim,
    extend_global_history_sequence,
)
from research.text_audio import ByteTextTokenizer
from research.wave_diffusion import DiffusionSchedule
from research.weight_ema import model_state_sha256


def _history() -> torch.Tensor:
    x = torch.arange(FRAMES, dtype=torch.float32)
    return (0.1 * torch.sin(x * 2 * math.pi / 157)
            + 0.03 * torch.sin(x * 2 * math.pi / 607)).float()


def _clean_pair(batch_size=2, prefix_frames=1024):
    history = _history().view(1, 1, FRAMES).repeat(batch_size, 1, 1)
    future_only = torch.arange(
        FRAMES - prefix_frames, dtype=torch.float32
    )
    future_only = (
        0.09 * torch.cos(future_only * 2 * math.pi / 313)
    ).view(1, 1, -1).repeat(batch_size, 1, 1)
    future = torch.cat((history[:, :, -prefix_frames:], future_only), dim=-1)
    return history, future


@pytest.mark.parametrize("channels", [16, 24, 32])
def test_neural_full_window_context_encoder_shapes_grads(channels):
    torch.manual_seed(100)
    module = HistoryEncoder(channels)
    signal = _history().view(1, 1, FRAMES).repeat(2, 1, 1)
    result = module(signal)
    assert result.shape == (2, channels)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in module.parameters()
    )


def test_neural_history_influences_distant_future_without_changing_local_overlap():
    """Proof of architecture's nonlocal conditioning, not learned audio quality."""
    torch.manual_seed(101)
    network = GlobalHistoryVelocityDenoiser(16).eval()
    base = torch.zeros(1, 1, FRAMES)
    alternate = base.clone()
    alternate[:, :, :6000] = (
        0.8 * torch.sin(torch.arange(6000).float() / 13)
    )
    # Last 1024 frames remain IDENTICAL, including the clean anchored prefix.
    assert torch.equal(base[..., -1024:], alternate[..., -1024:])
    input_noise = torch.zeros(1, 1, FRAMES)
    tokens = ByteTextTokenizer().encode(["rain"])
    step = torch.tensor([4], dtype=torch.long)
    with torch.no_grad():
        a = network(input_noise, step, tokens, base, prefix_frames=1024)
        b = network(input_noise, step, tokens, alternate, prefix_frames=1024)
    assert a.shape == b.shape == (1, 1, FRAMES)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.mean(torch.abs(a[..., 7168:] - b[..., 7168:])) > 1e-8


@pytest.mark.parametrize("channels", [16, 24, 32])
def test_full_history_model_is_velocity_predictor_over_raw_8192_frames(channels):
    torch.manual_seed(7)
    network = GlobalHistoryVelocityDenoiser(channels)
    history, future = _clean_pair(batch_size=1, prefix_frames=512)
    tokens = ByteTextTokenizer().encode(["बारिश"])
    t = torch.tensor([3], dtype=torch.long)
    result = network(
        future, t, tokens, history, prefix_frames=512,
    )
    assert result.shape == (1, 1, FRAMES)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_masked_future_loss_is_differentiable_through_explicit_history_encoder():
    torch.manual_seed(17)
    network = GlobalHistoryVelocityDenoiser(16)
    history, future = _clean_pair(batch_size=2, prefix_frames=1024)
    tokens = ByteTextTokenizer().encode(["rain", "बारिश"])
    random_before = torch.random.get_rng_state().clone()
    loss = masked_history_velocity_loss(
        network, DiffusionSchedule(8), future, tokens, history,
        prefix_frames=1024,
        generator=torch.Generator(device="cpu").manual_seed(471),
    )
    assert torch.isfinite(loss) and loss.ndim == 0 and loss.item() > 0
    assert torch.equal(random_before, torch.random.get_rng_state())
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in network.history_encoder.parameters()
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in network.text_encoder.parameters()
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in network.residual.parameters()
    )


def test_loss_is_reproducible_given_identical_model_parameters_and_generator():
    torch.manual_seed(18)
    net = GlobalHistoryVelocityDenoiser(16).eval()
    history, future = _clean_pair(batch_size=2, prefix_frames=512)
    tokens = ByteTextTokenizer().encode(["rain", "wind"])
    schedule = DiffusionSchedule(8)
    before = torch.random.get_rng_state().clone()
    left = masked_history_velocity_loss(
        net, schedule, future, tokens, history, prefix_frames=512,
        generator=torch.Generator(device="cpu").manual_seed(31),
    )
    right = masked_history_velocity_loss(
        net, schedule, future, tokens, history, prefix_frames=512,
        generator=torch.Generator(device="cpu").manual_seed(31),
    )
    assert torch.equal(left, right)
    assert torch.equal(before, torch.random.get_rng_state())


def test_sample_global_history_inference_seeds_models_modes_are_reproducible():
    torch.manual_seed(20)
    network = GlobalHistoryVelocityDenoiser(16).train()
    prior = _history()
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    before_mode = network.training
    before_rng = torch.random.get_rng_state().clone()
    before_state = model_state_sha256(network)
    config = dict(
        prefix_frames=1024, evaluations=3, seed=41,
        guidance=1.5,
    )
    one, report = sample_global_history_ddim(
        network, schedule, "rain", prior, **config,
    )
    again, repeated = sample_global_history_ddim(
        network, schedule, "rain", prior, **config,
    )
    changed, _ = sample_global_history_ddim(
        network, schedule, "rain", prior,
        **(config | {"seed": 42}),
    )
    assert torch.equal(one, again)
    assert not torch.equal(one, changed)
    assert report == repeated
    assert one.shape == (FRAMES,)
    assert torch.isfinite(one).all()
    assert torch.equal(one[:1024], prior[-1024:])
    assert report["protocol"] == INFERENCE_PROTOCOL
    assert report["conditioning"] == "explicit_global_previous_8192_frame_encoder"
    assert report["requires_matching_global_history_future_v_training"] is True
    assert report["sample_rate"] == 8000
    assert report["prior_frames"] == FRAMES
    assert report["prefix_frames"] == 1024
    assert report["timestep_evaluations"] == 3
    assert report["model_evaluations"] == 6
    assert report["known_prefix_exactly_preserved"] is True
    assert report["normalization_or_clipping_applied"] is False
    assert report["model_parameters_unchanged"] is True
    assert report["global_rng_unchanged"] is True
    assert report["audio_or_weights_saved"] is False
    assert report["publication_authorized"] is False
    assert report["listening_or_long_context_quality_measured"] is False
    assert report["previous_audio_sha256"] == hashlib.sha256(
        prior.numpy().tobytes()).hexdigest()
    assert report["generated_sha256"] == hashlib.sha256(
        one.numpy().tobytes()).hexdigest()
    assert math.isfinite(report["first_unknown_sample_jump"])
    assert math.isfinite(report["peak_before_clamp"])
    assert 0 <= report["fraction_outside_unit_audio_range"] <= 1
    assert before_mode == network.training
    assert before_state == model_state_sha256(network)
    assert torch.equal(before_rng, torch.random.get_rng_state())


def test_repeated_global_history_sequence_uses_full_previous_generated_clip():
    torch.manual_seed(21)
    model = GlobalHistoryVelocityDenoiser(16)
    before = model_state_sha256(model)
    original = _history()
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    config = dict(
        segments=3, prefix_frames=512,
        evaluations=2, seed=90, guidance=1.0,
    )
    a, report = extend_global_history_sequence(
        model, schedule, original, ("rain", "हवा"), **config,
    )
    b, repeated = extend_global_history_sequence(
        model, schedule, original, ("rain", "हवा"), **config,
    )
    assert torch.equal(a, b)
    assert report == repeated
    assert a.shape == (FRAMES + 2 * (FRAMES - 512),)
    assert torch.equal(a[:FRAMES], original)
    assert report["protocol"] == SEQUENCE_PROTOCOL
    assert report["training_protocol"] == PROTOCOL
    assert report["source_initial_clip_supplied"] is True
    assert report["segments"] == 3
    assert report["prefix_frames"] == 512
    assert report["frames"] == a.numel()
    assert report["duration_seconds"] == a.numel() / 8000
    assert report["continuation_seed_base"] == 90
    assert [x["seed"] for x in report["joins"]] == [91, 92]
    assert [x["future_caption"] for x in report["joins"]] == ["rain", "हवा"]
    assert len(report["joins"]) == 2
    assert all(x["known_overlap_preserved"] for x in report["joins"])
    assert report["model_state_preserved"] is True
    assert report["rng_preserved"] is True
    assert report["model_weights_or_audio_written"] is False
    assert report["publication_authorized"] is False
    assert report["long_audio_quality_or_semantics_measured"] is False
    assert model_state_sha256(model) == before


@pytest.mark.parametrize("bad_channels", [
    0, -1, 8, 17, 48, True, 16.0, "16", None,
])
def test_bad_model_capacity_rejected(bad_channels):
    with pytest.raises(ValueError):
        HistoryEncoder(bad_channels)
    with pytest.raises(ValueError):
        GlobalHistoryVelocityDenoiser(bad_channels)


@pytest.mark.parametrize("history", [
    torch.zeros(FRAMES - 1),
    torch.zeros(FRAMES + 1),
    torch.zeros(FRAMES, dtype=torch.float64),
    torch.zeros((1, FRAMES)),
    torch.full((FRAMES,), float("nan")),
    torch.full((FRAMES,), float("inf")),
    torch.full((FRAMES,), 1e13),
    torch.zeros(FRAMES, requires_grad=True),
])
def test_bad_sampling_history_refused_before_neural_work(history):
    model = GlobalHistoryVelocityDenoiser(16)
    with pytest.raises(ValueError):
        sample_global_history_ddim(
            model, DiffusionSchedule(8), "wind", history,
            prefix_frames=512, evaluations=2, seed=42,
        )


@pytest.mark.parametrize("change", [
    {"prefix_frames": 0}, {"prefix_frames": 129},
    {"prefix_frames": 4097}, {"prefix_frames": 512.0},
    {"evaluations": 1}, {"evaluations": 9},
    {"evaluations": True}, {"seed": -1},
    {"seed": True}, {"seed": 2**32},
    {"guidance": -1}, {"guidance": 4.01},
    {"guidance": float("nan")}, {"guidance": True},
    {"caption": ""},
])
def test_invalid_inference_config_rejected(change):
    model = GlobalHistoryVelocityDenoiser(16)
    opts = dict(
        network=model, schedule=DiffusionSchedule(8),
        caption="wind", prior_history=_history(),
        prefix_frames=512, evaluations=3, seed=19,
    )
    with pytest.raises(ValueError):
        sample_global_history_ddim(**(opts | change))


@pytest.mark.parametrize("change", [
    {"segments": 1}, {"segments": 9}, {"segments": True},
    {"prefix_frames": 128.0}, {"prefix_frames": 0},
    {"seed": -1}, {"seed": 2**32 - 1},
    {"captions": ()}, {"captions": ("rain",)},
    {"captions": ("rain", "wind", "storm")},
    {"captions": ("", "wind")},
    {"initial_clip": torch.zeros(FRAMES - 1)},
    {"initial_clip": torch.full((FRAMES,), float("inf"))},
])
def test_invalid_sequence_args_rejected(change):
    model = GlobalHistoryVelocityDenoiser(16)
    args = dict(
        network=model, schedule=DiffusionSchedule(8),
        initial_clip=_history(), captions=("rain", "wind"),
        segments=3, prefix_frames=512,
        evaluations=2, seed=10,
    )
    with pytest.raises(ValueError):
        extend_global_history_sequence(**(args | change))


def test_history_must_come_from_true_future_prefix_for_training():
    net = GlobalHistoryVelocityDenoiser(16)
    previous, future = _clean_pair(batch_size=1, prefix_frames=512)
    future = future.clone()
    future[..., :512] = 0
    with pytest.raises(ValueError, match="contiguous"):
        masked_history_velocity_loss(
            net, DiffusionSchedule(8), future,
            ByteTextTokenizer().encode(["rain"]),
            previous, prefix_frames=512,
            generator=torch.Generator(device="cpu").manual_seed(12),
        )


def test_malformed_network_forward_rejects_false_prefix():
    net = GlobalHistoryVelocityDenoiser(16)
    history, future = _clean_pair(batch_size=1, prefix_frames=512)
    future = future.clone()
    future[..., 0] += 0.5
    with pytest.raises(ValueError, match="prefix"):
        net(
            future, torch.tensor([4]),
            ByteTextTokenizer().encode(["wind"]),
            history, prefix_frames=512,
        )


def test_history_encoder_rejects_nonfinite_or_malformed_data():
    model = HistoryEncoder(16)
    for invalid in (
        torch.ones(1, 1, FRAMES - 1),
        torch.full((1, 1, FRAMES), float("nan")),
        torch.full((1, 1, FRAMES), float("inf")),
        torch.ones(1, 1, FRAMES, dtype=torch.float64),
        torch.ones(1, 1, FRAMES) * 1e13,
    ):
        with pytest.raises(ValueError):
            model(invalid)
