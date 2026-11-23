"""Mathematical and integration tests for the independent v-DDIM sampler."""
from __future__ import annotations

import copy
import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from research.velocity_ddim import (
    DDIMPlan, ddim_plan, velocity_ddim_transition, sample_velocity_ddim,
)
from research.velocity_prediction import velocity_target
from research.wave_diffusion import DiffusionSchedule
from research.latent_audio import FRAMES, LatentResearchModel
from research.text_audio import TextWaveDenoiser


@pytest.mark.parametrize(("steps", "budget"), [
    (8, 2), (8, 3), (8, 4), (8, 8),
    (16, 2), (16, 5), (16, 16),
    (32, 3), (32, 7), (32, 31),
    (256, 2), (256, 127), (256, 256),
])
def test_plan_is_strictly_descending_deterministic_and_exact_at_full_steps(steps, budget):
    schedule = DiffusionSchedule(steps, profile="cosine_zero_terminal_v1")
    first = ddim_plan(schedule, evaluations=budget)
    assert first == ddim_plan(schedule, evaluations=budget)
    assert len(first.timesteps) == budget
    assert first.timesteps[0] == steps - 1 and first.timesteps[-1] == 0
    assert len(set(first.timesteps)) == budget
    assert all(x > y for x, y in zip(first.timesteps, first.timesteps[1:]))
    if budget == steps:
        assert first.timesteps == tuple(range(steps - 1, -1, -1))
    report = first.as_dict()
    assert report["model_evaluations"] == budget
    assert report["denoising_transitions"] == budget
    assert report["publication_authorized"] is False
    assert report["sampler"] == "deterministic_ddim_eta_zero"
    assert first.as_dict(guidance=2)["model_evaluations"] == 2 * budget


@pytest.mark.parametrize(("steps", "budget", "expected"), [
    (8, 2, (7, 0)),
    (8, 3, (7, 4, 0)),
    (8, 4, (7, 5, 2, 0)),
    (16, 5, (15, 11, 8, 4, 0)),
])
def test_step_rounding_protocol_is_frozen(steps, budget, expected):
    plan = ddim_plan(DiffusionSchedule(steps), evaluations=budget)
    assert plan.timesteps == expected


@pytest.mark.parametrize("kwargs", [
    {"evaluations": 0}, {"evaluations": 1},
    {"evaluations": True}, {"evaluations": 2.0},
    {"evaluations": 9}, {"evaluations": -1},
])
def test_plan_rejects_invalid_evaluation_budgets(kwargs):
    with pytest.raises(ValueError):
        ddim_plan(DiffusionSchedule(8), **kwargs)


def test_plan_rejects_forged_order_or_profile():
    with pytest.raises(ValueError):
        DDIMPlan(8, 3, (7, 7, 0), "linear_beta_v1")
    with pytest.raises(ValueError):
        DDIMPlan(8, 3, (7, 4, 0), "not-a-trained-schedule")
    with pytest.raises(ValueError):
        DDIMPlan(8, 3, (7, 4, -1), "linear_beta_v1")
    with pytest.raises(ValueError):
        ddim_plan(DiffusionSchedule(8), evaluations=8).as_dict(guidance=float("inf"))


@pytest.mark.parametrize("steps", [8, 16, 32])
@pytest.mark.parametrize("budget", [2, 3, 4])
def test_ideal_velocity_oracle_reconstructs_same_clean_wave_through_skipped_steps(
    steps, budget,
):
    """x0/noise remain consistent across an arbitrary valid DDIM step subset."""
    schedule = DiffusionSchedule(steps, profile="cosine_zero_terminal_v1")
    torch.manual_seed(71)
    clean = torch.randn(2, 1, 96, dtype=torch.float32) * 0.25
    noise = torch.randn_like(clean)
    plan = ddim_plan(schedule, evaluations=budget)
    first = plan.timesteps[0]
    initial_step = torch.tensor([first, first], dtype=torch.long)
    current = schedule.corrupt(clean, initial_step, noise)
    for i, timestep in enumerate(plan.timesteps):
        previous = plan.timesteps[i + 1] if i + 1 < budget else -1
        t = torch.tensor([timestep, timestep], dtype=torch.long)
        velocity = velocity_target(clean, noise, t, schedule)
        current = velocity_ddim_transition(
            current, velocity, timestep, previous, schedule,
        )
        if previous >= 0:
            expected = schedule.corrupt(
                clean, torch.tensor([previous, previous]), noise,
            )
        else:
            expected = clean
        assert torch.allclose(current, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("t", "prev"), [
    (0, 0), (4, 5), (-1, -1), (8, 0), (4, -2), (True, -1),
])
def test_reverse_transition_rejects_invalid_order(t, prev):
    schedule = DiffusionSchedule(8)
    tensor = torch.zeros(1, 1, 32)
    with pytest.raises(ValueError):
        velocity_ddim_transition(tensor, tensor, t, prev, schedule)


def test_transition_rejects_nonfinite_and_mismatched_tensors():
    schedule = DiffusionSchedule(8)
    x = torch.zeros(2, 1, 32)
    with pytest.raises(ValueError):
        velocity_ddim_transition(x, x[:, :, :-1], 7, 0, schedule)
    with pytest.raises(ValueError):
        velocity_ddim_transition(x, torch.full_like(x, float("inf")), 7, 0, schedule)


class CountingDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.calls: list[int] = []

    def forward(self, audio, timestep, tokens):
        self.calls.append(int(timestep[0]))
        return self.scale * torch.tanh(audio)


def test_sampler_evaluation_count_and_classifier_free_guidance():
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    model = CountingDenoiser()
    x = sample_velocity_ddim(
        model, schedule, "rain", evaluations=3, seed=5, guidance=1,
    )
    assert x.shape == (FRAMES,)
    assert model.calls == [7, 4, 0]
    model.calls.clear()
    y = sample_velocity_ddim(
        model, schedule, "rain", evaluations=3, seed=5, guidance=2,
    )
    assert y.shape == (FRAMES,)
    assert model.calls == [7, 7, 4, 4, 0, 0]


def test_real_raw_denoiser_has_repeatable_and_seed_sensitive_inference():
    torch.manual_seed(100)
    model = TextWaveDenoiser(channels=16)
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    before = torch.random.get_rng_state().clone()
    model.train()
    x = sample_velocity_ddim(model, schedule, "rain", evaluations=3, seed=19)
    again = sample_velocity_ddim(model, schedule, "rain", evaluations=3, seed=19)
    other = sample_velocity_ddim(model, schedule, "rain", evaluations=3, seed=20)
    full = sample_velocity_ddim(model, schedule, "rain", evaluations=8, seed=19)
    assert x.shape == (FRAMES,) and x.dtype == torch.float32
    assert torch.equal(x, again) and not torch.equal(x, other)
    assert not torch.equal(x, full)
    assert torch.isfinite(x).all()
    assert model.training
    assert torch.equal(before, torch.random.get_rng_state())


def test_latent_decoder_preactivations_are_reportable_without_clipping():
    torch.manual_seed(104)
    model = LatentResearchModel(16)
    model.denoiser.train()
    model.codec.train()
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    before = torch.random.get_rng_state().clone()
    wave, pre = sample_velocity_ddim(
        model.denoiser, schedule, "बारिश", evaluations=4, seed=10,
        latent=True, codec=model.codec, return_preactivation=True,
    )
    assert wave.shape == pre.shape == (FRAMES,)
    assert torch.allclose(wave, torch.tanh(pre), rtol=1e-6, atol=1e-6)
    assert torch.isfinite(pre).all()
    assert model.denoiser.training and model.codec.training
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("bad", [
    {"seed": -1}, {"seed": 2**32}, {"seed": True},
    {"guidance": -0.1}, {"guidance": 4.1},
    {"guidance": float("nan")}, {"guidance": True},
    {"latent": True}, {"latent": "yes"},
    {"return_preactivation": True}, {"evaluations": 100},
])
def test_sampler_rejects_invalid_input_before_running_model(bad):
    model = CountingDenoiser()
    schedule = DiffusionSchedule(8)
    with pytest.raises(ValueError):
        sample_velocity_ddim(model, schedule, "rain", evaluations=3,
                             **({k: v for k, v in bad.items()} if "evaluations" not in bad else bad))
    assert model.calls == []


def test_unchanged_existing_velocity_sampler_and_no_output_files(tmp_path):
    from research.velocity_prediction import sample_velocity
    model = CountingDenoiser()
    schedule = DiffusionSchedule(8)
    assert not list(tmp_path.iterdir())
    result = sample_velocity_ddim(model, schedule, "wind", seed=1, evaluations=2)
    legacy = sample_velocity(model, schedule, "wind", seed=1)
    assert result.shape == legacy.shape == (FRAMES,)
    assert not list(tmp_path.iterdir())
