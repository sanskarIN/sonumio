"""Optional research tests skip on installations without PyTorch."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from research.spectral_smoke import TinySpectralPredictor, create_synthetic_dataset, train

def test_synthetic_data_has_stable_provenance():
    categories, features, digest = create_synthetic_dataset((1,))
    assert categories.shape == (6,)
    assert features.shape == (6, 96)
    assert len(digest) == 64
    assert torch.isfinite(features).all()
    _, _, digest_again = create_synthetic_dataset((1,))
    assert digest_again == digest

def test_tiny_training_reduces_loss_and_checkpoint_is_pickle_free(tmp_path):
    report = train(steps=30, seed=14, output=tmp_path)
    assert report["final_train_mse"] < report["initial_mse"]
    assert report["heldout_seed_mse"] >= 0
    assert report["original_sonumio_model_generation"] is None
    with np.load(tmp_path / "tiny_spectral_weights.npz", allow_pickle=False) as weights:
        assert "embedding.weight" in weights.files
        assert weights["embedding.weight"].shape == (6, 24)
    assert (tmp_path / "metrics.json").exists()
