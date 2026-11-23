"""Mathematical oracle and real raw-model regression tests for prefix v-DDIM."""
from __future__ import annotations

import hashlib
import math

import pytest
torch = pytest.importorskip("torch")
from torch import nn

from research.latent_audio import FRAMES
from research.text_audio import ByteTextTokenizer, TextWaveDenoiser
from research.velocity_prefix_continuation import (
    PROTOCOL, SEQUENCE_PROTOCOL, masked_future_velocity_loss,
    sample_prefix_velocity_ddim, sample_continuation_sequence,
)
from research.wave_diffusion import DiffusionSchedule
from research.weight_ema import model_state_sha256


class ExactFutureOracle(nn.Module):
    """Oracle predicts v that reconstructs ONE specified future x0 exactly."""

    def __init__(self, clean: torch.Tensor, schedule: DiffusionSchedule):
        super().__init__()
        self.register_buffer("target", clean.view(1, 1, FRAMES).clone())
        self.theta = nn.Parameter(torch.zeros(()))
        self.schedule = schedule

    def forward(self, current, timestep, tokens):
        bar = self.schedule.alpha_bars[timestep].view(-1, 1, 1)
        return (
            bar.sqrt() * current - self.target - 0 * self.theta
        ) / (1 - bar).sqrt()


@pytest.mark.parametrize("profile", [
    "linear_beta_v1", "cosine_zero_terminal_v1",
])
@pytest.mark.parametrize("steps", [2, 3, 8])
def test_oracle_v_model_reconstructs_known_future_with_exact_prefix(profile, steps):
    schedule = DiffusionSchedule(8, profile=profile)
    target = 0.2 * torch.sin(
        torch.arange(FRAMES, dtype=torch.float32) * (2 * math.pi / 200),
    )
    known = target[:512].clone()
    oracle = ExactFutureOracle(target, schedule)
    before = model_state_sha256(oracle)
    result, report = sample_prefix_velocity_ddim(
        oracle, schedule, "rain", known, evaluations=steps,
        seed=17, guidance=1.0,
    )
    assert torch.equal(result[:512], known)
    assert torch.max(torch.abs(result[512:] - target[512:])) < 0.0001
    assert report["protocol"] == PROTOCOL
    assert report["exact_original_prefix_preserved"] is True
    assert report["requires_matching_masked_future_training"] is True
    assert report["neural_evaluations"] == steps
    assert report["model_publication_authorized"] is False
    assert model_state_sha256(oracle) == before


def test_loss_trains_future_only_using_real_cpu_text_denoiser():
    torch.manual_seed(19)
    model = TextWaveDenoiser(16)
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    clean = (torch.randn(2, 1, FRAMES) * 0.1).float()
    tokens = ByteTextTokenizer().encode(("rain", "बारिश"))
    rng_before = torch.random.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(121)
    loss = masked_future_velocity_loss(
        model, schedule, clean, tokens, prefix_frames=1024,
        generator=generator,
    )
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_training_does_not_leak_loss_from_prefix_region():
    class KnownPrediction(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, noisy, timestep, tokens):
            return self.scale * noisy

    data = torch.zeros((1, 1, FRAMES), dtype=torch.float32)
    model = KnownPrediction()
    gen = torch.Generator(device="cpu").manual_seed(11)
    loss = masked_future_velocity_loss(
        model, DiffusionSchedule(8), data,
        ByteTextTokenizer().encode(["rain"]),
        prefix_frames=1024, generator=gen,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_raw_neural_prefix_and_multiclip_continuation_are_deterministic():
    torch.manual_seed(23)
    model = TextWaveDenoiser(16)
    model.train()
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    prior_rng = torch.random.get_rng_state().clone()
    prior_hash = model_state_sha256(model)
    prefix = torch.linspace(-0.05, 0.05, 512)
    a, one = sample_prefix_velocity_ddim(
        model, schedule, "बारिश", prefix, evaluations=3, seed=31,
        guidance=1.5,
    )
    b, second = sample_prefix_velocity_ddim(
        model, schedule, "बारिश", prefix, evaluations=3, seed=31,
        guidance=1.5,
    )
    c, _ = sample_prefix_velocity_ddim(
        model, schedule, "बारिश", prefix, evaluations=3, seed=32,
        guidance=1.5,
    )
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.equal(a[:512], prefix)
    assert one == second
    assert one["neural_evaluations"] == 6
    assert one["prefix_sha256"] == hashlib.sha256(
        prefix.numpy().tobytes()).hexdigest()
    assert 0 <= one["fraction_above_unit_range"] <= 1
    assert one["no_model_weights_or_audio_saved"] is True
    initial_clip = (0.03 * torch.cos(
        torch.arange(FRAMES) * (2 * math.pi / 300)
    )).float()
    config = dict(
        initial_clip=initial_clip,
        segments=3, prefix_frames=512,
        evaluations=2, seed=33, guidance=1,
    )
    continuous, report = sample_continuation_sequence(
        model, schedule, ("rain", "बारिश", "wind"), **config,
    )
    again, same_report = sample_continuation_sequence(
        model, schedule, ("rain", "बारिश", "wind"), **config,
    )
    assert torch.equal(continuous, again)
    assert report == same_report
    assert continuous.shape == (FRAMES + 2 * (FRAMES - 512),)
    assert report["output_frames"] == continuous.numel()
    assert torch.equal(continuous[:FRAMES], initial_clip)
    assert report["source_initial_clip_supplied_by_caller"] is True
    assert report["initial_clip_sha256"] == hashlib.sha256(
        initial_clip.numpy().tobytes()).hexdigest()
    assert report["protocol"] == SEQUENCE_PROTOCOL
    assert report["trained_long_context_model"] is False
    assert len(report["joins"]) == 2
    assert [x["seed"] for x in report["joins"]] == [34, 35]
    assert all(x["exact_prefix_preserved"] for x in report["joins"])
    assert report["model_state_preserved"] is True
    assert report["global_rng_preserved"] is True
    assert report["no_model_weights_or_audio_saved"] is True
    assert report["listening_quality_or_prompt_alignment_measured"] is False
    assert torch.equal(prior_rng, torch.random.get_rng_state())
    assert model_state_sha256(model) == prior_hash
    assert model.training


@pytest.mark.parametrize("prefix", [
    torch.zeros(127), torch.zeros(129), torch.zeros(4097),
    torch.zeros(512, dtype=torch.float64),
    torch.full((512,), float("nan")),
    torch.full((512,), float("inf")),
    torch.zeros(512, requires_grad=True),
    torch.ones(512) * 1e13,
    torch.zeros(1, 512),
])
def test_bad_prefix_rejected(prefix):
    denoiser = TextWaveDenoiser(16)
    with pytest.raises(ValueError):
        sample_prefix_velocity_ddim(
            denoiser, DiffusionSchedule(8), "rain",
            prefix, evaluations=2,
        )


@pytest.mark.parametrize("config", [
    {"evaluations": 1}, {"evaluations": 9},
    {"evaluations": True}, {"seed": -1},
    {"seed": 2**32}, {"seed": True},
    {"guidance": float("nan")}, {"guidance": -1.0},
    {"guidance": 5}, {"caption": "   "},
])
def test_sampler_bad_settings_rejected(config):
    model = TextWaveDenoiser(16)
    opts = dict(
        denoiser=model, schedule=DiffusionSchedule(8),
        caption="rain", prefix=torch.zeros(512),
        evaluations=3, seed=17, guidance=1,
    )
    with pytest.raises(ValueError):
        sample_prefix_velocity_ddim(**(opts | config))


@pytest.mark.parametrize("config", [
    {"segments": 1}, {"segments": 9},
    {"segments": True}, {"prefix_frames": 127},
    {"prefix_frames": 513}, {"seed": -1},
    {"seed": 2**32 - 1},
    {"captions": ("rain",)}, {"captions": ("rain", "")},
])
def test_sequence_bad_settings_rejected(config):
    model = TextWaveDenoiser(16)
    opts = dict(
        denoiser=model, schedule=DiffusionSchedule(8),
        captions=("rain", "wind"), segments=2,
        initial_clip=torch.zeros(FRAMES),
        prefix_frames=512, evaluations=2, seed=1,
    )
    with pytest.raises(ValueError):
        sample_continuation_sequence(**(opts | config))


def test_needle_in_known_prefix_is_preserved_exactly():
    model = TextWaveDenoiser(16)
    prefix = torch.zeros(1024)
    prefix[10] = 0.625
    prefix[1000] = -0.875
    result, _ = sample_prefix_velocity_ddim(
        model, DiffusionSchedule(8), "wind",
        prefix, evaluations=2, seed=64,
    )
    assert torch.equal(result[:1024], prefix)
    assert result[10] == 0.625 and result[1000] == -0.875


def test_nonfinite_model_output_rejected_without_rng_leakage():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))

        def forward(self, x, timestep, tokens):
            return torch.full_like(x, float("nan"))

    model = Broken()
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        sample_prefix_velocity_ddim(
            model, DiffusionSchedule(8), "wind", torch.zeros(512),
            evaluations=2, seed=17,
        )
    assert torch.equal(rng_before, torch.random.get_rng_state())


@pytest.mark.parametrize("initial", [
    torch.zeros(FRAMES - 1),
    torch.zeros(FRAMES, dtype=torch.float64),
    torch.full((FRAMES,), float("nan")),
    torch.full((FRAMES,), 1e13),
])
def test_continuation_rejects_invalid_lead_clip(initial):
    model = TextWaveDenoiser(16)
    with pytest.raises(ValueError):
        sample_continuation_sequence(
            model, DiffusionSchedule(8), "wind",
            initial_clip=initial, segments=2,
            prefix_frames=512, evaluations=2,
        )
