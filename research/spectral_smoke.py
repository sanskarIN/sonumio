"""Original neural-training smoke test using only Sonumio synthetic audio.

Learns average log-frequency energy per sound class. NOT a waveform generator,
NOT a candidate Sonumio-1. This isolates training, evaluation and safe
checkpointing without needing external recordings.

From repository root:
  python -m pip install -e '.[research]'
  python -m research.spectral_smoke --steps 40
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from sonumio.engines.procedural import generate as synthesize

PROMPT_CLASSES = ("rain", "thunder", "wind", "forest", "ocean", "ambient")
N_FFT = 512
N_BINS = 96

class TinySpectralPredictor(nn.Module):
    """Class-conditioned spectrum regression; it cannot generate audio."""

    def __init__(self, classes: int = len(PROMPT_CLASSES), bins: int = N_BINS):
        super().__init__()
        self.embedding = nn.Embedding(classes, 24)
        self.network = nn.Sequential(nn.Linear(24, 96), nn.GELU(),
                                     nn.Linear(96, bins), nn.Softplus())

    def forward(self, category_ids: torch.Tensor) -> torch.Tensor:
        return self.network(self.embedding(category_ids))

def create_synthetic_dataset(seeds: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Build rights-clear targets from Sonumio's own DSP; no downloaded data."""
    features, labels = [], []
    fingerprint = hashlib.sha256()
    for class_id, prompt in enumerate(PROMPT_CLASSES):
        for seed in seeds:
            samples = synthesize(prompt, 0.5, seed).samples
            fingerprint.update(prompt.encode("utf-8"))
            fingerprint.update(seed.to_bytes(4, "little"))
            fingerprint.update(samples.tobytes())
            wave = torch.from_numpy(samples)
            spec = torch.stft(wave, n_fft=N_FFT, hop_length=128,
                              window=torch.hann_window(N_FFT), return_complex=True)
            energy = torch.log1p(spec.abs()[:N_BINS].mean(dim=1))
            features.append(energy)
            labels.append(class_id)
    return torch.tensor(labels, dtype=torch.long), torch.stack(features), fingerprint.hexdigest()

def train(steps: int = 40, seed: int = 7, output: Path | None = None) -> dict:
    """Train and evaluate held-out synthetic seeds on CPU, with safe NPZ weights."""
    if not 1 <= steps <= 5_000:
        raise ValueError("steps must be between 1 and 5,000")
    torch.manual_seed(seed)
    np.random.seed(seed)
    labels, targets, dataset_sha256 = create_synthetic_dataset()
    test_labels, test_targets, test_sha256 = create_synthetic_dataset((100, 101))
    model = TinySpectralPredictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.0)
    start = perf_counter()
    model.train()
    with torch.no_grad():
        initial_loss = float(nn.functional.mse_loss(model(labels), targets))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(labels), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_loss = float(nn.functional.mse_loss(model(labels), targets))
        eval_loss = float(nn.functional.mse_loss(model(test_labels), test_targets))
    report = {
        "purpose": "neural training smoke test, NOT a generative audio model",
        "original_sonumio_model_generation": None,
        "architecture": "category-conditioned spectrum regressor",
        "dataset_provenance": "Synthetic audio from Sonumio procedural DSP (no outside recordings)",
        "seed": seed, "steps": steps,
        "training_examples": len(labels), "evaluation_examples": len(test_labels),
        "train_dataset_sha256": dataset_sha256, "eval_dataset_sha256": test_sha256,
        "initial_mse": initial_loss, "final_train_mse": train_loss,
        "heldout_seed_mse": eval_loss, "wall_seconds": round(perf_counter() - start, 3),
        "model_weights_published": False,
    }
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
        checkpoint = output / "tiny_spectral_weights.npz"
        np.savez_compressed(checkpoint, **arrays)
        report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/spectral-smoke"))
    args = parser.parse_args()
    print(json.dumps(train(args.steps, args.seed, args.output), indent=2))

if __name__ == "__main__":
    main()
