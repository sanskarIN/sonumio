"""Device selection, stochastic input plans and accumulation regression tests."""
from dataclasses import replace
import json
import math

import pytest

torch = pytest.importorskip("torch")

from research.device_velocity_training import (
    BatchPlanner, TrainConfig, model_fingerprint, resolve_device,
    train_model, training_update, validate_corpus,
)
from research.device_velocity_study import compare_parameters, run_study
from research.latent_audio import FRAMES
from research.text_audio import ByteTextTokenizer, TextWaveDenoiser
from research.wave_diffusion import DiffusionSchedule


def corpus():
    generator = torch.Generator(device="cpu").manual_seed(123)
    waves = torch.randn((6, 1, FRAMES), generator=generator) * 0.1
    tokens = ByteTextTokenizer().encode(["rain", "wind", "forest", "ocean", "thunder", "ambient"])
    return waves, tokens


def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        return TextWaveDenoiser(16)


@pytest.mark.parametrize("kwargs", [
    {"steps": True}, {"steps": 0}, {"steps": 4097}, {"batch_size": 33},
    {"microbatch_size": 0}, {"microbatch_size": 9}, {"batch_size": 1},
    {"schedule_steps": 7}, {"seed": -1}, {"learning_rate": float("nan")},
    {"max_grad_norm": 0}, {"condition_dropout": 1.1}, {"device": None},
])
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


@pytest.mark.parametrize("request_name", ["mps", "gpu", "CUDA", "cuda:-1", "cuda:01", "cuda:1000"])
def test_reject_invalid_device_names(request_name):
    with pytest.raises(ValueError, match="Device must be"):
        resolve_device(request_name)


def test_no_silent_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda")


def test_cuda_index_check(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("cuda") == torch.device("cuda:0")
    assert resolve_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda:2")


def test_plan_independent_of_microbatch_and_device():
    config = TrainConfig(steps=3, batch_size=5, microbatch_size=2)
    a = BatchPlanner(config, 6)
    b = BatchPlanner(replace(config, microbatch_size=3, device="cuda:0"), 6)
    global_rng = torch.random.get_rng_state().clone()
    for _ in range(config.steps):
        left, right = a.next(), b.next()
        for name in ("indices", "timesteps", "keep_caption", "noise"):
            assert torch.equal(getattr(left, name), getattr(right, name))
        assert [left.indices[s].numel() for s in left.slices(2)] == [2, 2, 1]
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert a.describe() == b.describe()
    assert sum(a.describe()["timestep_histogram"]) == 15
    assert a.describe()["caption_rows_drawn"] == 15
    with pytest.raises(ValueError, match="exhausted"):
        a.next()


def test_full_and_null_caption_controls():
    for rate, expected in ((0, 6), (1, 0)):
        cfg = TrainConfig(steps=2, batch_size=3, microbatch_size=1, condition_dropout=rate)
        p = BatchPlanner(cfg, 6)
        p.next()
        p.next()
        assert p.describe()["retained_caption_rows"] == expected


def test_corpus_validation():
    waves, tokens = corpus()
    validate_corpus(waves, tokens)
    for x, y in ((waves[:, :, :-1], tokens), (waves.double(), tokens),
                 (waves, tokens[:, :-1]), (waves, tokens.float()),
                 (waves, torch.full_like(tokens, 999)),
                 (torch.full_like(waves, float("nan")), tokens)):
        with pytest.raises(ValueError, match="waveform and UTF-8"):
            validate_corpus(x, y)


def test_accumulation_matches_full_batch_gradient_with_remainder():
    waves, tokens = corpus()
    cfg = TrainConfig(steps=1, batch_size=5, microbatch_size=5, max_grad_norm=100)
    left, right = model(), model()
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    plan = BatchPlanner(cfg, len(waves)).next()
    opt_a = torch.optim.SGD(left.parameters(), lr=0.001)
    opt_b = torch.optim.SGD(right.parameters(), lr=0.001)
    a = training_update(left, opt_a, schedule, waves, tokens, plan, cfg, torch.device("cpu"))
    b = training_update(right, opt_b, schedule, waves, tokens, plan,
                        replace(cfg, microbatch_size=2), torch.device("cpu"))
    assert a["microbatches"] == 1
    assert b["microbatches"] == 3
    assert a["velocity_mse"] == pytest.approx(b["velocity_mse"], rel=1e-5, abs=1e-7)
    assert a["preclip_gradient_norm"] == pytest.approx(b["preclip_gradient_norm"], rel=2e-4)
    assert compare_parameters(left, right)["maximum_absolute_parameter_difference"] < 1e-6
    assert all(p.grad is None for p in left.parameters())
    assert all(p.grad is None for p in right.parameters())


def test_nonfinite_update_does_not_step_optimizer():
    waves, tokens = corpus()
    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
        def forward(self, audio, timestep, caption):
            return audio * self.weight * float("nan")
    net = BadModel()
    cfg = TrainConfig(steps=1, batch_size=3, microbatch_size=2)
    opt = torch.optim.AdamW(net.parameters())
    plan = BatchPlanner(cfg, len(waves)).next()
    before = model_fingerprint(net)
    with pytest.raises(ValueError, match="Nonfinite"):
        training_update(net, opt, DiffusionSchedule(8), waves, tokens, plan, cfg, torch.device("cpu"))
    assert not opt.state
    assert model_fingerprint(net) == before
    assert net.weight.grad is None


def test_real_adamw_training_returns_cpu_state_and_trace():
    waves, tokens = corpus()
    net = model()
    before = model_fingerprint(net)
    cfg = TrainConfig(steps=2, batch_size=3, microbatch_size=2)
    result = train_model(net, waves, tokens, cfg)
    assert result["device"] == result["evaluation_device"] == "cpu"
    assert result["optimizer_updates"] == 2
    assert result["microbatch_backward_calls"] == 4
    assert result["initial_state_sha256"] == before
    assert result["trained_state_sha256"] == model_fingerprint(net)
    assert result["trained_state_sha256"] != before
    assert not net.training and next(net.parameters()).device.type == "cpu"
    assert result["batch_plan"]["caption_rows_drawn"] == 6
    assert math.isfinite(result["mean_velocity_mse"])
    json.dumps(result, allow_nan=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_actual_cuda_train_smoke():
    waves, tokens = corpus()
    cpu = train_model(model(), waves, tokens, TrainConfig(steps=1, batch_size=3, microbatch_size=2))
    net = model()
    gpu = train_model(net, waves, tokens,
                      TrainConfig(steps=1, batch_size=3, microbatch_size=2, device="cuda"))
    assert gpu["device"] == "cuda:0"
    assert gpu["cuda_device_name"]
    assert cpu["initial_state_sha256"] == gpu["initial_state_sha256"]
    assert cpu["batch_plan"] == gpu["batch_plan"]
    assert next(net.parameters()).device.type == "cpu"
    assert math.isfinite(gpu["mean_velocity_mse"])


def test_actual_paired_study_controls():
    cfg = TrainConfig(steps=1, batch_size=3, microbatch_size=2)
    before = torch.random.get_rng_state().clone()
    report = run_study(config=cfg, alternative_microbatch=1, evaluations=2)
    assert torch.equal(before, torch.random.get_rng_state())
    assert report["paired_initialization_and_inputs"]
    assert report["unique_training_recordings"] == report["unique_validation_recordings"] == 42
    assert not report["exact_split_overlap"]
    assert not report["cuda_training_executed"]
    assert not report["weights_or_audio_saved"]
    assert set(report["arms"]) == {"reference_cpu", "alternative_cpu"}
    for arm in report["arms"].values():
        assert arm["heldout"]["caption_rows_per_timestep"] == 84
        if report["reference_probe"]["gate"]["passed"]:
            assert len(arm["generated"]["cases"]) == 12
            assert arm["generated"]["null_caption_invariant"]
        else:
            assert arm["generated"] is None
    json.dumps(report, allow_nan=False)
