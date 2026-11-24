"""Learning-rate endpoints, optimizer update semantics and failure cleanup."""
from dataclasses import replace
import math

import pytest

torch = pytest.importorskip("torch")

from research.device_learning_rate import LearningRatePlan, set_learning_rate
from research.device_velocity_training import TrainConfig, train_model
from research.latent_audio import FRAMES
from research.text_audio import TextWaveDenoiser, ByteTextTokenizer


@pytest.mark.parametrize("updates,warmup", [(2,0),(5,1),(8,2),(64,8)])
def test_warmup_cosine_endpoints_and_monotonic_decay(updates,warmup):
    plan = LearningRatePlan(.001,updates,"warmup_cosine",warmup,.1)
    rates = [plan.rate(i) for i in range(updates)]
    assert rates[warmup] == pytest.approx(.001)
    assert rates[-1] == pytest.approx(.0001)
    assert all(a <= b for a,b in zip(rates[:warmup],rates[1:warmup]))
    assert all(a >= b for a,b in zip(rates[warmup:],rates[warmup+1:]))
    assert plan.describe()["per_update_learning_rates"] == rates
    assert plan.describe()["mean_learning_rate"] == pytest.approx(math.fsum(rates)/updates)


@pytest.mark.parametrize("kwargs", [
    {"base_rate":float("nan")},{"updates":0},{"updates":True},
    {"policy":"unknown"},{"warmup_updates":1},{"minimum_ratio":-1},
    {"policy":"warmup_cosine","warmup_updates":3},
])
def test_invalid_learning_rate_configuration(kwargs):
    with pytest.raises(ValueError):
        LearningRatePlan(**({"base_rate":.001,"updates":4}|kwargs))


def test_constant_rate_and_invalid_indices():
    plan = LearningRatePlan(.0005,4)
    assert [plan.rate(i) for i in range(4)] == [.0005]*4
    for index in (-1,4,True,1.0):
        with pytest.raises(ValueError):
            plan.rate(index)


def test_every_optimizer_group_receives_same_logical_rate():
    x = torch.nn.Parameter(torch.ones(1))
    y = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([{"params":[x]},{"params":[y]}],lr=.001)
    plan = LearningRatePlan(.001,4,"warmup_cosine",1,.1)
    for i in range(4):
        rate = set_learning_rate(opt,plan,i)
        assert all(group["lr"] == rate for group in opt.param_groups)


def corpus():
    waves = torch.randn(4,1,FRAMES,generator=torch.Generator().manual_seed(42))*.1
    tokens = ByteTextTokenizer().encode(["rain","wind","forest","ocean"])
    return waves,tokens


def test_real_training_schedule_invariant_to_microbatch_count():
    waves,tokens = corpus()
    cfg = TrainConfig(steps=4,batch_size=3,microbatch_size=2,
                      learning_rate_policy="warmup_cosine",warmup_updates=1)
    reports = []
    with torch.random.fork_rng(devices=[]):
        for microbatch in (2,1):
            torch.manual_seed(17)
            reports.append(train_model(TextWaveDenoiser(16),waves,tokens,
                                      replace(cfg,microbatch_size=microbatch)))
    for row in reports:
        assert row["applied_learning_rates"] == cfg.rate_plan().describe()["per_update_learning_rates"]
        assert row["optimizer_updates"] == 4
    assert reports[0]["batch_plan"] == reports[1]["batch_plan"]
    assert reports[0]["microbatch_backward_calls"] == 8
    assert reports[1]["microbatch_backward_calls"] == 12


def test_model_cleanup_on_optimizer_initialization_failure(monkeypatch):
    waves,tokens = corpus()
    net = TextWaveDenoiser(16).train()
    def fail(*args,**kwargs):
        raise RuntimeError("Injected optimizer setup failure")
    monkeypatch.setattr(torch.optim,"AdamW",fail)
    with pytest.raises(RuntimeError,match="Injected optimizer"):
        train_model(net,waves,tokens,TrainConfig(steps=1))
    assert not net.training
    assert all(parameter.device.type == "cpu" for parameter in net.parameters())
