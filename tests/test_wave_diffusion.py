"""Research tests are optional because standard installations do not need PyTorch."""
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from research.wave_diffusion import (
    CATEGORIES, TinyWaveDenoiser, DiffusionSchedule, CLIP_FRAMES,
    dataset, training_step, train, load_checkpoint, save_checkpoint,
)


def test_dataset_is_deterministic_and_split_by_seed():
    a, labels, fingerprint = dataset((1, 2))
    b, other, check = dataset((1, 2))
    c, _, other_fingerprint = dataset((100,))
    assert a.shape == (12, 1, CLIP_FRAMES)
    assert torch.equal(a, b) and torch.equal(labels, other)
    assert fingerprint == check and fingerprint != other_fingerprint
    assert len(CATEGORIES) == 6 and c.shape[0] == 6
    assert torch.isfinite(a).all()


def test_training_backward_and_unconditional_path():
    torch.manual_seed(22)
    model, schedule = TinyWaveDenoiser(channels=16), DiffusionSchedule(8)
    sample = torch.rand(3, 1, 256) - 0.5
    labels = torch.tensor([0, 1, 2])
    loss = training_step(model, schedule, sample, labels, drop_condition=1)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(sample[:, 0], labels, labels)


def test_sampling_determinism_and_validation():
    torch.manual_seed(33)
    model, schedule = TinyWaveDenoiser(channels=16), DiffusionSchedule(8)
    first = schedule.sample(model, 0, seed=2, frames=256)
    second = schedule.sample(model, 0, seed=2, frames=256)
    different = schedule.sample(model, 0, seed=3, frames=256)
    assert first.shape == (256,)
    assert torch.equal(first, second) and not torch.equal(first, different)
    assert torch.isfinite(first).all() and first.abs().max() <= 1
    with pytest.raises(ValueError):
        schedule.sample(model, -1)
    with pytest.raises(ValueError):
        schedule.sample(model, 0, guidance=float('nan'))


def test_checkpoint_integrity_and_loading(tmp_path: Path):
    torch.manual_seed(5)
    model = TinyWaveDenoiser(channels=16)
    schedule = DiffusionSchedule(8)
    save_checkpoint(tmp_path, model, schedule, {"sample": True})
    restored, diffusion, metadata = load_checkpoint(tmp_path)
    assert diffusion.steps == 8 and metadata["weights_publication_authorized"] is False
    assert set(model.state_dict()) == set(restored.state_dict())
    assert all(torch.equal(model.state_dict()[k], restored.state_dict()[k]) for k in model.state_dict())
    with (tmp_path / "weights.npz").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(tmp_path)


def test_train_short_smoke(tmp_path: Path):
    report = train(steps=2, seed=8, channels=16, diffusion_steps=8, output=tmp_path)
    assert np.isfinite(report["heldout_noise_mse"])
    assert report["parameter_count"] > 1_000
    assert report["weights_published"] is False
    assert (tmp_path / "manifest.json").exists()
