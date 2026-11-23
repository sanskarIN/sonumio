"""Independent EMA numeric/atomicity tests; research-only CPU weights."""
import pytest
import torch
from torch import nn

from research.weight_ema import WeightEMA, model_state_sha256


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))


def test_numeric_update_and_integer_buffer_copy():
    source = SimpleNet()
    ema = WeightEMA(source, decay=0.8)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([4.0, 8.0, 12.0]))
        source.counter.add_(1)
    assert ema.update(source) == pytest.approx(0.8)
    shadow = ema.snapshot()
    assert torch.allclose(shadow.weight, torch.tensor([1.6, 3.2, 4.8]))
    assert shadow.counter.item() == 1
    assert not shadow.weight.requires_grad and not shadow.training
    assert ema.updates == 1
    assert len(ema.state_sha256()) == 64
    assert ema.describe()["checkpoint_written"] is False
    assert ema.describe()["model_weights_published"] is False


def test_warmup_formula_and_identical_weights_do_not_drift():
    source = SimpleNet()
    ema = WeightEMA(source, decay=0.9, warmup=True)
    assert ema.update(source) == pytest.approx(0.1)
    assert ema.update(source) == pytest.approx(2 / 11)
    assert torch.equal(ema.snapshot().weight, source.weight)


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), -0.1, 1.0, True, "0.9", None,
])
def test_invalid_decay_rejected(bad):
    with pytest.raises(ValueError):
        WeightEMA(SimpleNet(), decay=bad)


def test_mismatched_structure_or_dtype_does_not_change_shadow():
    source = SimpleNet()
    ema = WeightEMA(source)
    original = ema.state_sha256()
    with pytest.raises(ValueError, match="names"):
        ema.update(nn.Linear(2, 2))
    with pytest.raises(ValueError, match="CPU float32"):
        ema.update(SimpleNet().double())
    assert ema.updates == 0
    assert ema.state_sha256() == original


def test_nonfinite_input_cannot_partially_mutate_shadow():
    source = SimpleNet()
    ema = WeightEMA(source)
    before = ema.state_sha256()
    with torch.no_grad():
        source.weight[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ema.update(source)
    assert ema.state_sha256() == before and ema.updates == 0


def test_shadow_snapshot_has_no_write_access_to_live_ema():
    source = SimpleNet()
    ema = WeightEMA(source)
    independent = ema.snapshot()
    initial = ema.state_sha256()
    with torch.no_grad():
        independent.weight.fill_(123)
    assert ema.state_sha256() == initial
    assert not torch.equal(independent.weight, ema.snapshot().weight)


def test_update_preserves_global_rng_and_online_training_mode():
    source = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 3))
    source.train()
    prior_rng = torch.random.get_rng_state().clone()
    ema = WeightEMA(source)
    before = model_state_sha256(source)
    with torch.no_grad():
        source[0].weight.add_(1)
    ema.update(source)
    assert source.training
    assert torch.equal(torch.random.get_rng_state(), prior_rng)
    assert model_state_sha256(source) != before
    assert all(not p.requires_grad for p in ema.snapshot().parameters())


def test_empty_modules_rejected():
    with pytest.raises(ValueError, match="parameterized"):
        WeightEMA(nn.ReLU())


def test_reject_integer_or_bool_decay_and_nonfinite_source():
    for bad in (-0.1, 1, True):
        with pytest.raises(ValueError):
            WeightEMA(SimpleNet(), decay=bad)
    source = SimpleNet()
    with torch.no_grad():
        source.weight.fill_(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        WeightEMA(source)


def test_atomic_rejection_on_candidate_overflow():
    source = SimpleNet()
    with torch.no_grad():
        source.weight.fill_(-3.0e38)
    ema = WeightEMA(source, decay=0.5)
    with torch.no_grad():
        source.weight.fill_(3.0e38)
    # Both finite inputs: float32 convex interpolation must not overflow;
    # update should produce finite values, never a half-updated state.
    ema.update(source)
    assert torch.isfinite(ema.snapshot().weight).all()
