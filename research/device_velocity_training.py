"""Device-explicit training of the original experimental waveform v-model.

CPU-generated stochastic inputs make batch plans independent of training
device and microbatch partition. Floating-point execution is not promised
bitwise identical across hardware. This module never saves weights or audio.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re

import torch
from torch import nn

from research.latent_audio import FRAMES
from research.text_audio import MAX_TOKENS, BYTE_VOCAB
from research.velocity_prediction import velocity_target
from research.velocity_loss_weights import validate_policy, timestep_weights, velocity_loss, describe_weights
from research.wave_diffusion import DiffusionSchedule

MAX_BATCH = 32
MAX_MICROBATCH = 8
MAX_ROWS = 4096


def resolve_device(request: str) -> torch.device:
    """No silent CUDA-to-CPU fallback, and no arbitrary device selectors."""
    if request == "cpu":
        return torch.device("cpu")
    if not isinstance(request, str) or not re.fullmatch(r"cuda(?::(?:0|[1-9][0-9]{0,2}))?", request):
        raise ValueError("Device must be cpu, cuda, or cuda:<index>")
    index = int(request.split(":")[1]) if ":" in request else 0
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise RuntimeError("Requested CUDA device is not available")
    return torch.device("cuda", index)


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 64
    batch_size: int = 8
    microbatch_size: int = 2
    schedule_steps: int = 8
    learning_rate: float = 5e-4
    max_grad_norm: float = 1.0
    condition_dropout: float = 0.1
    seed: int = 17
    device: str = "cpu"
    loss_policy: str = "uniform"
    min_snr_gamma: float = 5.0

    def __post_init__(self):
        checks = ((self.steps, 1, 4096), (self.batch_size, 1, MAX_BATCH),
                  (self.microbatch_size, 1, MAX_MICROBATCH), (self.schedule_steps, 8, 256),
                  (self.seed, 0, 2**32 - 8192))
        if any(type(value) is not int or not lo <= value <= hi for value, lo, hi in checks):
            raise ValueError("Invalid bounded velocity training configuration")
        if self.microbatch_size > self.batch_size:
            raise ValueError("Microbatch must not exceed logical batch")
        for value, lo, hi in ((self.learning_rate, 1e-6, 0.01), (self.max_grad_norm, 0.01, 100),
                              (self.condition_dropout, 0, 1)):
            if type(value) not in (float, int) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError("Invalid optimizer or condition dropout setting")
        validate_policy(self.loss_policy, self.min_snr_gamma)
        if not isinstance(self.device, str):
            raise ValueError("Device must be a string")


def validate_corpus(waves: torch.Tensor, tokens: torch.Tensor):
    if (not isinstance(waves, torch.Tensor) or waves.ndim != 3
            or waves.shape[1:] != (1, FRAMES) or not 1 <= len(waves) <= MAX_ROWS
            or waves.device.type != "cpu" or waves.dtype != torch.float32
            or waves.requires_grad or not torch.isfinite(waves).all()
            or not isinstance(tokens, torch.Tensor) or tokens.shape != (len(waves), MAX_TOKENS)
            or tokens.device.type != "cpu" or tokens.dtype != torch.long
            or (tokens < 0).any() or (tokens >= BYTE_VOCAB).any()):
        raise ValueError("Expected bounded CPU waveform and UTF-8 caption corpus")


@dataclass(frozen=True)
class PlannedBatch:
    indices: torch.Tensor
    timesteps: torch.Tensor
    keep_caption: torch.Tensor
    noise: torch.Tensor

    def slices(self, maximum: int):
        if type(maximum) is not int or not 1 <= maximum <= MAX_MICROBATCH:
            raise ValueError("Invalid microbatch size")
        for start in range(0, len(self.indices), maximum):
            yield slice(start, min(start + maximum, len(self.indices)))


class BatchPlanner:
    """Materialize one logical batch at a time; hash complete CPU inputs."""

    def __init__(self, config: TrainConfig, rows: int):
        if not isinstance(config, TrainConfig) or type(rows) is not int or not 1 <= rows <= MAX_ROWS:
            raise ValueError("Invalid batch planning inputs")
        self.config = config
        self.rows = rows
        self.rng = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        self.trace = hashlib.sha256(b"sonumio-device-velocity-plan-v1\0")
        self.draws = 0
        self.kept = 0
        self.histogram = torch.zeros(config.schedule_steps, dtype=torch.long)

    def next(self) -> PlannedBatch:
        if self.draws >= self.config.steps:
            raise ValueError("Logical training plan exhausted")
        n = self.config.batch_size
        indices = torch.randint(self.rows, (n,), generator=self.rng)
        timesteps = torch.randint(self.config.schedule_steps, (n,), generator=self.rng)
        keep = torch.rand(n, generator=self.rng) >= self.config.condition_dropout
        noise = torch.randn((n, 1, FRAMES), generator=self.rng, dtype=torch.float32)
        for value in (indices, timesteps, keep, noise):
            self.trace.update(value.numpy().tobytes())
        self.histogram += torch.bincount(timesteps, minlength=self.config.schedule_steps)
        self.draws += 1
        self.kept += int(keep.sum())
        return PlannedBatch(indices, timesteps, keep, noise)

    def describe(self) -> dict:
        return {
            "logical_batches": self.draws,
            "caption_rows_drawn": self.draws * self.config.batch_size,
            "retained_caption_rows": self.kept,
            "timestep_histogram": self.histogram.tolist(),
            "input_trace_sha256": self.trace.hexdigest(),
            "rng_device": "cpu",
        }


def model_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256(b"sonumio-device-velocity-model-v1\0")
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Expected finite float32 model parameters and buffers")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def training_update(model, optimizer, schedule, waves, tokens, plan, config, device):
    """One logical optimizer update; normalize over the entire logical batch."""
    if len(plan.indices) != config.batch_size:
        raise ValueError("Logical batch does not match configuration")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    weighted_total = 0.0
    weights = timestep_weights(schedule.alpha_bars, config.loss_policy, config.min_snr_gamma)
    microbatches = 0
    try:
        for part in plan.slices(config.microbatch_size):
            indices = plan.indices[part]
            clean = waves[indices].to(device)
            step = plan.timesteps[part].to(device)
            keep = plan.keep_caption[part].to(device)
            noise = plan.noise[part].to(device)
            caption = tokens[indices].to(device)
            caption = torch.where(keep[:, None], caption, torch.zeros_like(caption))
            target = velocity_target(clean, noise, step, schedule)
            prediction = model(schedule.corrupt(clean, step, noise), step, caption)
            if prediction.shape != target.shape:
                raise ValueError("Denoiser output shape mismatch")
            # Weight each microbatch by its actual size, including a remainder.
            # Summing independent gradients gives the logical-batch mean.
            raw, weighted = velocity_loss(prediction, target, step, weights)
            loss = weighted.sum() / config.batch_size
            loss.backward()
            total += float(raw.detach().sum().cpu()) / config.batch_size
            weighted_total += float(loss.detach().cpu())
            microbatches += 1
        norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                        error_if_nonfinite=True)
        optimizer.step()
        if not math.isfinite(total) or not math.isfinite(weighted_total) or not torch.isfinite(norm):
            raise ValueError("Nonfinite training update diagnostics")
    finally:
        # Prevent stale partial gradients after a failed or successful update.
        optimizer.zero_grad(set_to_none=True)
    return {"velocity_mse": total, "weighted_objective": weighted_total, "preclip_gradient_norm": float(norm.detach().cpu()),
            "microbatches": microbatches}


def train_model(model: nn.Module, waves, tokens, config: TrainConfig):
    """Return a trained CPU model and numeric report; CUDA inference is not assumed."""
    validate_corpus(waves, tokens)
    if not isinstance(config, TrainConfig):
        raise ValueError("Expected validated training configuration")
    device = resolve_device(config.device)
    initial = model_fingerprint(model)
    model.to(device)
    schedule = DiffusionSchedule(config.schedule_steps, profile="cosine_zero_terminal_v1").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, foreach=False)
    planner = BatchPlanner(config, len(waves))
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    try:
        for _ in range(config.steps):
            history.append(training_update(model, optimizer, schedule, waves, tokens,
                                           planner.next(), config, device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        # The existing research DDIM sampler is CPU-only.
        model.to("cpu").eval()
    losses = [row["velocity_mse"] for row in history]
    return {
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda if device.type == "cuda" else None,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_dtype": "float32", "gradient_accumulation": True,
        "loss_policy": config.loss_policy, "min_snr_gamma": config.min_snr_gamma,
        "timestep_weights": describe_weights(timestep_weights(schedule.alpha_bars, config.loss_policy, config.min_snr_gamma)),
        "mean_weighted_objective": sum(row["weighted_objective"] for row in history) / len(history),
        "logical_batch_size": config.batch_size, "microbatch_size": config.microbatch_size,
        "optimizer_updates": len(history),
        "microbatch_backward_calls": sum(row["microbatches"] for row in history),
        "first_velocity_mse": losses[0], "last_velocity_mse": losses[-1],
        "mean_velocity_mse": sum(losses) / len(losses),
        "maximum_preclip_gradient_norm": max(row["preclip_gradient_norm"] for row in history),
        "clipped_updates": sum(row["preclip_gradient_norm"] > config.max_grad_norm for row in history),
        "initial_state_sha256": initial, "trained_state_sha256": model_fingerprint(model),
        "batch_plan": planner.describe(), "evaluation_device": "cpu",
        "cross_device_bitwise_equivalence_claimed": False,
        "weights_or_audio_saved": False,
    }
