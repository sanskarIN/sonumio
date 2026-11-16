"""Experimental *generative* waveform diffusion for synthetic Sonumio sounds.

An original, tiny class-conditioned DDPM research baseline, NOT Sonumio-1.
Only Sonumio-generated DSP examples enter this dataset. It is deliberately
limited to 0.256-second, 8-kHz clips, six categories, and no open-vocabulary
text understanding. Avoid treating its output as production audio.

Usage:
    python -m research.wave_diffusion train --steps 50 --output checkpoints/wave-diffusion
    python -m research.wave_diffusion sample --checkpoint checkpoints/wave-diffusion --prompt rain --output test.wav

Checkpoint files are numeric-only .npz and JSON, loaded without pickle.
Never publish checkpoints without explicit authorization and rights review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from research.checkpoint_archive import validate_numeric_archive
from sonumio.audio import encode_wav
from sonumio.engines.procedural import generate as dsp_generate

CATEGORIES = ("rain", "thunder", "wind", "forest", "ocean", "ambient")
SAMPLE_RATE = 8_000
CLIP_FRAMES = 2_048
DEFAULT_DIFFUSION_STEPS = 32


class ConditionedResidual(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.conv = nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation)
        self.condition_projection = nn.Linear(channels, channels * 2)
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition_projection(condition).unsqueeze(-1).chunk(2, dim=1)
        h = F.silu(self.norm(x))
        h = self.conv(h)
        h = F.silu(h * (1 + scale) + shift)
        return (x + self.out(h)) * (2 ** -0.5)


class TinyWaveDenoiser(nn.Module):
    """Class/time-conditioned 1D denoiser, no pretrained or external weights."""

    def __init__(self, channels: int = 32, classes: int = len(CATEGORIES)):
        super().__init__()
        if channels % 8 or channels < 8:
            raise ValueError("channels must be a multiple of 8")
        self.classes = classes
        self.channels = channels
        self.embedding = nn.Embedding(classes + 1, channels)  # null class for CFG
        self.time_mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.SiLU(),
                                      nn.Linear(channels * 2, channels))
        self.input = nn.Conv1d(1, channels, 5, padding=2)
        self.layers = nn.ModuleList(ConditionedResidual(channels, d) for d in (1, 2, 4, 8, 16, 1))
        self.output = nn.Sequential(nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv1d(channels, 1, 3, padding=1))

    def forward(self, audio: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [batch, 1, frames]")
        if timestep.ndim != 1 or timestep.numel() != audio.shape[0]:
            raise ValueError("timestep must have shape [batch]")
        if label.ndim == 1:
            if label.shape != timestep.shape:
                raise ValueError("class label must have shape [batch]")
            label_features = self.embedding(label)
        elif label.ndim == 2:
            if label.shape != (audio.shape[0], self.classes):
                raise ValueError("multi-label conditions must have shape [batch, classes]")
            if not torch.isfinite(label).all() or not ((label == 0) | (label == 1)).all():
                raise ValueError("multi-label conditions must contain only finite binary values")
            count = label.sum(dim=1, keepdim=True)
            if (count > 2).any():
                raise ValueError("a maximum of two sound categories is supported")
            combined = (label.to(self.embedding.weight.dtype) @ self.embedding.weight[:self.classes]) / count.clamp(min=1).sqrt()
            label_features = torch.where(count > 0, combined, self.embedding.weight[self.classes].unsqueeze(0))
        else:
            raise ValueError("invalid sound-condition tensor")
        half = self.channels // 2
        frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=audio.device) / max(half - 1, 1))
        angles = timestep.float()[:, None] * frequencies[None]
        time_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        condition = label_features + self.time_mlp(time_features)
        h = self.input(audio)
        for layer in self.layers:
            h = layer(h, condition)
        return self.output(h)


class DiffusionSchedule(nn.Module):
    """Linear-beta DDPM with buffered, device-aware statistics."""

    def __init__(self, steps: int = DEFAULT_DIFFUSION_STEPS):
        super().__init__()
        if not 8 <= steps <= 1_000:
            raise ValueError("diffusion steps must be between 8 and 1000")
        self.steps = steps
        betas = torch.linspace(0.0001, 0.02, steps, dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def corrupt(self, clean: torch.Tensor, step: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bars[step].view(-1, 1, 1)
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: TinyWaveDenoiser, category: int, *, seed: int = 0,
               guidance: float = 1.0, frames: int = CLIP_FRAMES) -> torch.Tensor:
        if not 0 <= category < model.classes:
            raise ValueError("Unknown audio category")
        if not 0 <= guidance <= 4 or not math.isfinite(guidance):
            raise ValueError("guidance must be finite and in [0, 4]")
        if not 256 <= frames <= 32_768 or frames % 16:
            raise ValueError("frames must be a multiple of 16, 256..32768")
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((1, 1, frames), generator=generator, device=device)
        model.eval()
        for index in range(self.steps - 1, -1, -1):
            t = torch.tensor([index], device=device, dtype=torch.long)
            predicted = model(x, t, torch.tensor([category], device=device))
            if guidance != 1:
                unconditioned = model(x, t, torch.tensor([model.classes], device=device))
                predicted = unconditioned + guidance * (predicted - unconditioned)
            alpha, bar, beta = self.alphas[index], self.alpha_bars[index], self.betas[index]
            mean = (x - beta / torch.sqrt(1 - bar) * predicted) / torch.sqrt(alpha)
            if index:
                prev_bar = self.alpha_bars[index - 1]
                posterior_var = beta * (1 - prev_bar) / (1 - bar)
                x = mean + posterior_var.sqrt() * torch.randn(x.shape, device=device, generator=generator)
            else:
                x = mean
        return x.clamp(-1, 1).cpu()[0, 0]


def _downsample_24k_to_8k(data: np.ndarray) -> np.ndarray:
    """Simple anti-aliasing FIR for synthetic experiments, not a studio resampler."""
    taps = np.arange(-24, 25, dtype=np.float64)
    kernel = np.sinc(taps / 3) * np.hamming(len(taps))
    kernel /= kernel.sum()
    filtered = np.convolve(data.astype(np.float64), kernel, mode="same")
    return filtered[::3].astype(np.float32)


def dataset(seeds: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[torch.Tensor, torch.Tensor, str]:
    clips, classes = [], []
    fingerprint = hashlib.sha256()
    for category, prompt in enumerate(CATEGORIES):
        for seed in seeds:
            wave = dsp_generate(prompt, 0.5, seed).samples
            mono = _downsample_24k_to_8k(wave)[:CLIP_FRAMES]
            fingerprint.update(prompt.encode("utf-8") + seed.to_bytes(4, "little") + mono.tobytes())
            clips.append(torch.from_numpy(mono.copy()).view(1, -1))
            classes.append(category)
    return torch.stack(clips), torch.tensor(classes, dtype=torch.long), fingerprint.hexdigest()


def training_step(model: TinyWaveDenoiser, schedule: DiffusionSchedule, x: torch.Tensor,
                  categories: torch.Tensor, *, drop_condition: float = 0.1) -> torch.Tensor:
    steps = torch.randint(0, schedule.steps, (len(x),), device=x.device)
    noise = torch.randn_like(x)
    dropped = torch.rand(len(x), device=x.device) < drop_condition
    if categories.ndim == 2:
        labels = torch.where(dropped[:, None], torch.zeros_like(categories), categories)
    else:
        labels = torch.where(dropped, model.classes, categories)
    return F.mse_loss(model(schedule.corrupt(x, steps, noise), steps, labels), noise)


def save_checkpoint(path: Path, model: TinyWaveDenoiser, schedule: DiffusionSchedule, report: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    # No Python object deserialization; reader uses allow_pickle=False.
    np.savez_compressed(path / "weights.npz", **arrays)
    manifest = {
        "format": "sonumio-synthetic-diffusion-experiment-1", "channels": model.channels,
        "classes": list(CATEGORIES), "diffusion_steps": schedule.steps, "sample_rate": SAMPLE_RATE,
        "clip_frames": CLIP_FRAMES, "training_report": report,
        "weights_sha256": hashlib.sha256((path / "weights.npz").read_bytes()).hexdigest(),
        "not_original_sonumio_1": True, "weights_publication_authorized": False,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path) -> tuple[TinyWaveDenoiser, DiffusionSchedule, dict]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "sonumio-synthetic-diffusion-experiment-1" or manifest.get("classes") != list(CATEGORIES):
        raise ValueError("Unsupported checkpoint manifest or mismatched prompt categories")
    if manifest.get("sample_rate") != SAMPLE_RATE or manifest.get("clip_frames") != CLIP_FRAMES:
        raise ValueError("Incompatible waveform layout")
    channels, steps = manifest.get("channels"), manifest.get("diffusion_steps")
    if (type(channels) is not int or channels not in range(8, 65, 8)
            or type(steps) is not int or not 8 <= steps <= 1000):
        raise ValueError("Unsupported waveform diffusion checkpoint configuration")
    weight_file = path / "weights.npz"
    if weight_file.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Checkpoint exceeds research size limit")
    checkpoint = weight_file.read_bytes()
    if hashlib.sha256(checkpoint).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    model = TinyWaveDenoiser(channels=channels)
    schedule = DiffusionSchedule(steps)
    expected = model.state_dict()
    validate_numeric_archive(weight_file, expected)
    with np.load(weight_file, allow_pickle=False) as weights:
        if set(weights.files) != set(expected):
            raise ValueError("Unexpected checkpoint tensor names")
        safe = {}
        for name, value in expected.items():
            arr = weights[name]
            if arr.shape != tuple(value.shape) or not np.isfinite(arr).all() or arr.dtype != np.float32:
                raise ValueError(f"Invalid checkpoint tensor: {name}")
            safe[name] = torch.from_numpy(arr.copy())
        model.load_state_dict(safe, strict=True)
    return model.eval(), schedule, manifest


def train(*, steps: int = 50, seed: int = 9, output: Path | None = None,
          diffusion_steps: int = DEFAULT_DIFFUSION_STEPS, channels: int = 32,
          manifest: Path | None = None, audio_root: Path | None = None,
          clips_per_asset: int = 2, mixed_synthetic: bool = False) -> dict:
    if not 1 <= steps <= 50_000:
        raise ValueError("training steps must be 1..50000")
    if not 0 <= seed < 2 ** 32:
        raise ValueError("seed must be in [0, 2**32)")
    torch.manual_seed(seed)
    model, schedule = TinyWaveDenoiser(channels), DiffusionSchedule(diffusion_steps)
    if (manifest is None) != (audio_root is None):
        raise ValueError("--manifest and --audio-root must be provided together")
    if mixed_synthetic and manifest is not None:
        raise ValueError("mixed synthetic conditioning cannot be combined with real-recording manifest")
    ingestion_report = None
    if mixed_synthetic:
        from research.multi_condition import mixed_dataset
        x, labels, fingerprint = mixed_dataset()
        eval_x, eval_labels, eval_fingerprint = mixed_dataset((101, 102))
    elif manifest is not None:
        from research.audio_dataset import load_reviewed_splits
        loaded, ingestion_report = load_reviewed_splits(manifest, audio_root, clips_per_asset=clips_per_asset)
        x = torch.from_numpy(loaded["train"].clips)
        labels = torch.from_numpy(loaded["train"].labels)
        eval_x = torch.from_numpy(loaded["validation"].clips)
        eval_labels = torch.from_numpy(loaded["validation"].labels)
        fingerprint = loaded["train"].fingerprint
        eval_fingerprint = loaded["validation"].fingerprint
    else:
        x, labels, fingerprint = dataset()
        eval_x, eval_labels, eval_fingerprint = dataset((101, 102))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    started = perf_counter()
    losses = []
    model.train()
    for _ in range(steps):
        batch = torch.randint(0, len(x), (8,))
        optimizer.zero_grad(set_to_none=True)
        loss = training_step(model, schedule, x[batch], labels[batch])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        heldout = float(training_step(model, schedule, eval_x, eval_labels, drop_condition=0.0))
    report = {
        "purpose": "experimental conditional waveform DDPM trained exclusively on DSP samples",
        "model_generation": None, "not_production_ready": True,
        "training_examples": len(x), "heldout_examples": len(eval_x),
        "data_source": ("sonumio-mixed-synthetic-dsp" if mixed_synthetic else
                        "rights-declared-local-wav" if manifest is not None else "sonumio-synthetic-dsp"),
        "conditioning_mode": "multi-label" if mixed_synthetic else "single-class",
        "data_ingestion": ingestion_report,
        "training_fingerprint": fingerprint, "heldout_fingerprint": eval_fingerprint,
        "steps": steps, "seed": seed, "initial_step_noise_mse": losses[0],
        "last_step_noise_mse": losses[-1], "heldout_noise_mse": heldout,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "weights_published": False,
    }
    if output is not None:
        save_checkpoint(output, model, schedule, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--steps", type=int, default=50)
    training.add_argument("--seed", type=int, default=9)
    training.add_argument("--output", type=Path, default=Path("checkpoints/wave-diffusion"))
    training.add_argument("--manifest", type=Path, help="Rights-reviewed JSONL manifest")
    training.add_argument("--audio-root", type=Path, help="Local directory containing manifest WAV paths")
    training.add_argument("--clips-per-asset", type=int, default=2)
    training.add_argument("--mixed-synthetic", action="store_true", help="Experimental one/two-category DSP composition training")
    sampling = commands.add_parser("sample")
    sampling.add_argument("--checkpoint", required=True, type=Path)
    sampling.add_argument("--prompt", required=True, help="Single-category or supported two-sound prompt")
    sampling.add_argument("--seed", type=int, default=10)
    sampling.add_argument("--guidance", type=float, default=1.0)
    sampling.add_argument("--output", type=Path, default=Path("outputs/research-wave.wav"))
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(steps=args.steps, seed=args.seed, output=args.output,
                               manifest=args.manifest, audio_root=args.audio_root,
                               clips_per_asset=args.clips_per_asset,
                               mixed_synthetic=args.mixed_synthetic), indent=2))
    else:
        model, schedule, manifest = load_checkpoint(args.checkpoint)
        mode = manifest.get("training_report", {}).get("conditioning_mode", "single-class")
        if mode == "multi-label":
            from research.multi_condition import sample_mixed
            wave = sample_mixed(schedule, model, args.prompt, seed=args.seed, guidance=args.guidance)
        elif args.prompt in CATEGORIES:
            wave = schedule.sample(model, CATEGORIES.index(args.prompt), seed=args.seed,
                                   guidance=args.guidance)
        else:
            raise ValueError("This checkpoint only supports one of: " + ", ".join(CATEGORIES))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encode_wav(wave.numpy(), SAMPLE_RATE))
        print(f"Experimental 8 kHz, {len(wave) / SAMPLE_RATE:.3f}s sample: {args.output}")


if __name__ == "__main__":
    main()
