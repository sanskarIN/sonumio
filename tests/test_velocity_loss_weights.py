"""Numerical objective tests and real paired training regression."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from research.velocity_loss_weights import timestep_weights, velocity_loss
from research.velocity_prediction import decode_velocity
from research.wave_diffusion import DiffusionSchedule
from research.device_velocity_training import TrainConfig, BatchPlanner, train_model
from research.text_audio import TextWaveDenoiser, ByteTextTokenizer
from research.latent_audio import FRAMES


@pytest.mark.parametrize("gamma", [0.1, 1, 5, 100])
def test_min_snr_matches_definition(gamma):
    a = DiffusionSchedule(32, profile="cosine_zero_terminal_v1").alpha_bars
    snr = a.double() / (1-a.double())
    expected = torch.minimum(snr, torch.full_like(snr, gamma)) / (1+snr)
    torch.testing.assert_close(timestep_weights(a, "min_snr", gamma).double(), expected,
                               atol=1e-7, rtol=1e-6)


def test_clean_weight_exactly_matches_clean_mse():
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    t = torch.tensor([0, 4, 7])
    noisy = torch.randn(3, 1, 128)
    target = torch.randn_like(noisy)
    pred = torch.randn_like(noisy)
    a, _ = decode_velocity(noisy, pred, t, schedule)
    b, _ = decode_velocity(noisy, target, t, schedule)
    raw, weighted = velocity_loss(pred, target, t, timestep_weights(schedule.alpha_bars, "clean_mse"))
    torch.testing.assert_close(weighted, (a-b).square().mean(dim=(1,2)), atol=1e-6, rtol=1e-5)
    assert torch.equal(raw, (pred-target).square().mean(dim=(1,2)))


@pytest.mark.parametrize("policy,gamma", [("unknown",5), ("min_snr",True), ("min_snr",0),
                                         ("uniform",float("nan"))])
def test_invalid_config(policy,gamma):
    with pytest.raises(ValueError):
        TrainConfig(loss_policy=policy, min_snr_gamma=gamma)


@pytest.mark.parametrize("policy", ["uniform","min_snr","clean_mse"])
def test_partition_independent_weighted_gradient(policy):
    a = DiffusionSchedule(8, profile="cosine_zero_terminal_v1").alpha_bars
    weights = timestep_weights(a, policy)
    x = torch.randn(5,1,16)
    y = torch.randn_like(x)
    t = torch.tensor([0,1,4,6,7])
    p = torch.tensor(0.7, requires_grad=True)
    q = p.detach().clone().requires_grad_()
    _, loss = velocity_loss(x*p,y,t,weights)
    loss.mean().backward()
    for s in (slice(0,2), slice(2,4), slice(4,5)):
        _, piece = velocity_loss(x[s]*q,y[s],t[s],weights)
        (piece.sum()/5).backward()
    torch.testing.assert_close(p.grad,q.grad)


def test_real_objective_ablation_same_inputs():
    rng = torch.Generator().manual_seed(11)
    waves = torch.randn(6,1,FRAMES,generator=rng)*0.1
    tokens = ByteTextTokenizer().encode(["rain","wind","ocean","forest","ambient","thunder"])
    cfg = TrainConfig(steps=2,batch_size=3,microbatch_size=2)
    reports = []
    with torch.random.fork_rng(devices=[]):
        for policy in ("uniform","min_snr","clean_mse"):
            torch.manual_seed(17)
            reports.append(train_model(TextWaveDenoiser(16),waves,tokens,replace(cfg,loss_policy=policy)))
    assert len({r["initial_state_sha256"] for r in reports}) == 1
    assert len({r["batch_plan"]["input_trace_sha256"] for r in reports}) == 1
    assert len({r["trained_state_sha256"] for r in reports}) == 3
    assert reports[0]["mean_weighted_objective"] == pytest.approx(reports[0]["mean_velocity_mse"])
    assert all(r["mean_weighted_objective"] <= r["mean_velocity_mse"] + 1e-7 for r in reports)
