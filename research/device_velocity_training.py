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
from research.semantic_training import SemanticHead, semantic_loss, semantic_diagnostics, validate_semantic_weight, validate_targets
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
    timestep_sampling: str = "uniform"
    caption_sampling: str = "independent"
    semantic_weight: float = 0.0

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
        validate_semantic_weight(self.semantic_weight)
        if self.timestep_sampling not in ("uniform", "stratified") or self.caption_sampling not in ("independent", "bilingual_pair"):
            raise ValueError("Unknown training sampling policy")
        if self.caption_sampling == "bilingual_pair" and self.batch_size % 2:
            raise ValueError("Bilingual paired sampling requires an even logical batch")
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
        if config.caption_sampling == "bilingual_pair" and rows % 2:
            raise ValueError("Bilingual sampling requires adjacent recording pairs")
        self.rng = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        self.timestep_rng = torch.Generator(device="cpu").manual_seed(config.seed + 4096)
        self.pending_timesteps = torch.empty(0, dtype=torch.long)
        self.trace = hashlib.sha256(b"sonumio-device-velocity-plan-v1\0")
        self.draws = 0
        self.kept = 0
        self.histogram = torch.zeros(config.schedule_steps, dtype=torch.long)
        self.row_histogram = torch.zeros(rows, dtype=torch.long)

    def _stratified_timesteps(self, count):
        # Consume complete random permutations, carrying the remainder across
        # logical batches. Every completed cycle visits every timestep once.
        parts = []
        while count:
            if not len(self.pending_timesteps):
                self.pending_timesteps = torch.randperm(self.config.schedule_steps, generator=self.timestep_rng)
            take = min(count, len(self.pending_timesteps))
            parts.append(self.pending_timesteps[:take])
            self.pending_timesteps = self.pending_timesteps[take:]
            count -= take
        return torch.cat(parts)

    def next(self) -> PlannedBatch:
        if self.draws >= self.config.steps:
            raise ValueError("Logical training plan exhausted")
        n = self.config.batch_size
        paired = self.config.caption_sampling == "bilingual_pair"
        units = n // 2 if paired else n
        selected = torch.randint(self.rows // 2 if paired else self.rows, (units,), generator=self.rng)
        # Draw the original uniform timestep stream even when it is replaced
        # by stratification, so index/dropout/noise draws remain paired.
        uniform = torch.randint(self.config.schedule_steps, (units,), generator=self.rng)
        timesteps = self._stratified_timesteps(units) if self.config.timestep_sampling == "stratified" else uniform
        keep = torch.rand(units, generator=self.rng) >= self.config.condition_dropout
        noise = torch.randn((units, 1, FRAMES), generator=self.rng, dtype=torch.float32)
        if paired:
            indices = torch.stack((selected * 2, selected * 2 + 1), dim=1).flatten()
            timesteps = timesteps.repeat_interleave(2)
            keep = keep.repeat_interleave(2)
            noise = noise.repeat_interleave(2, dim=0)
        else:
            indices = selected
        for value in (indices, timesteps, keep, noise):
            self.trace.update(value.numpy().tobytes())
        self.histogram += torch.bincount(timesteps, minlength=self.config.schedule_steps)
        self.row_histogram += torch.bincount(indices, minlength=self.rows)
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
            "timestep_sampling": self.config.timestep_sampling,
            "caption_sampling": self.config.caption_sampling,
            "distinct_caption_rows_seen": int((self.row_histogram > 0).sum()),
            "minimum_caption_row_draws": int(self.row_histogram.min()),
            "maximum_caption_row_draws": int(self.row_histogram.max()),
            "timestep_count_spread": int(self.histogram.max() - self.histogram.min()),
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


def training_update(model, optimizer, schedule, waves, tokens, plan, config, device,
                    *, semantic_head=None, semantic_targets=None):
    """One logical optimizer update; normalize over the entire logical batch."""
    if len(plan.indices) != config.batch_size:
        raise ValueError("Logical batch does not match configuration")
    if bool(config.semantic_weight) != (semantic_head is not None and semantic_targets is not None):
        raise ValueError("Semantic training requires explicit auxiliary head and targets")
    model.train()
    if semantic_head is not None:
        semantic_head.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    weighted_total = 0.0
    semantic_total = 0.0
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
            objective = weighted.sum() / config.batch_size
            auxiliary = (semantic_loss(model.text_encoder, semantic_head, caption,
                                       semantic_targets[indices].to(device), keep) / config.batch_size
                         if semantic_head is not None else objective.new_zeros(()))
            loss = objective + config.semantic_weight * auxiliary
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite combined objective")
            loss.backward()
            total += float(raw.detach().sum().cpu()) / config.batch_size
            weighted_total += float(objective.detach().cpu())
            semantic_total += float(auxiliary.detach().cpu())
            microbatches += 1
        parameters = list(model.parameters())
        if semantic_head is not None:
            parameters += list(semantic_head.parameters())
        norm = nn.utils.clip_grad_norm_(parameters, config.max_grad_norm,
                                        error_if_nonfinite=True)
        optimizer.step()
        if not math.isfinite(total) or not math.isfinite(weighted_total) or not torch.isfinite(norm):
            raise ValueError("Nonfinite training update diagnostics")
    finally:
        # Prevent stale partial gradients after a failed or successful update.
        optimizer.zero_grad(set_to_none=True)
    return {"velocity_mse": total, "weighted_objective": weighted_total,
            "semantic_auxiliary": semantic_total, "preclip_gradient_norm": float(norm.detach().cpu()),
            "microbatches": microbatches}


def train_model(model: nn.Module, waves, tokens, config: TrainConfig,
                *, semantic_targets=None, semantic_validation=None):
    """Return a trained CPU model and numeric report; CUDA inference is not assumed."""
    validate_corpus(waves, tokens)
    if not isinstance(config, TrainConfig):
        raise ValueError("Expected validated training configuration")
    if config.caption_sampling == "bilingual_pair":
        if (len(waves) % 2 or not torch.equal(waves[::2], waves[1::2])
                or (tokens[::2] == tokens[1::2]).all(dim=1).any()
                or (tokens == 0).all(dim=1).any()):
            raise ValueError("Bilingual paired sampling requires equal adjacent waveforms and distinct non-null captions")
    if config.semantic_weight:
        validate_targets(semantic_targets, len(tokens))
        if not hasattr(model, "text_encoder") or not hasattr(model, "channels"):
            raise ValueError("Semantic training requires the text waveform denoiser")
        if semantic_validation is not None:
            val_tokens, val_targets = semantic_validation
            if (val_tokens.ndim != 2 or val_tokens.shape[1] != MAX_TOKENS or val_tokens.dtype != torch.long
                    or val_tokens.device.type != "cpu" or not 1 <= len(val_tokens) <= MAX_ROWS
                    or (val_tokens < 0).any() or (val_tokens >= BYTE_VOCAB).any()):
                raise ValueError("Invalid semantic validation caption batch")
            validate_targets(val_targets, len(val_tokens))
    elif semantic_targets is not None or semantic_validation is not None:
        raise ValueError("Semantic targets supplied with disabled objective")
    device = resolve_device(config.device)
    initial = model_fingerprint(model)
    head = None
    semantic_report = None
    if config.semantic_weight:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed + 6144)
            head = SemanticHead(model.channels)
        head = head.to(device)
    model.to(device)
    schedule = DiffusionSchedule(config.schedule_steps, profile="cosine_zero_terminal_v1").to(device)
    parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, foreach=False)
    if head is not None:
        semantic_report = {"initial_training": semantic_diagnostics(model.text_encoder, head, tokens, semantic_targets),
                           "initial_validation": semantic_diagnostics(model.text_encoder, head, *semantic_validation)
                           if semantic_validation is not None else None}
    planner = BatchPlanner(config, len(waves))
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    try:
        for _ in range(config.steps):
            history.append(training_update(model, optimizer, schedule, waves, tokens,
                                           planner.next(), config, device,
                                           semantic_head=head, semantic_targets=semantic_targets))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        # The existing research DDIM sampler is CPU-only.
        model.to("cpu").eval()
        if head is not None:
            head.to("cpu").eval()
    if head is not None:
        semantic_report["final_training"] = semantic_diagnostics(model.text_encoder, head, tokens, semantic_targets)
        semantic_report["final_validation"] = (semantic_diagnostics(model.text_encoder, head, *semantic_validation)
                                               if semantic_validation is not None else None)
        semantic_report["head_retained_for_inference"] = False
    losses = [row["velocity_mse"] for row in history]
    return {
        "device": str(device), "torch_version": str(torch.__version__),
        "cuda_runtime_version": torch.version.cuda if device.type == "cuda" else None,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "training_dtype": "float32", "gradient_accumulation": True,
        "loss_policy": config.loss_policy, "min_snr_gamma": config.min_snr_gamma,
        "semantic_weight": config.semantic_weight, "semantic_diagnostics": semantic_report,
        "mean_semantic_auxiliary": sum(row["semantic_auxiliary"] for row in history) / len(history),
        "mean_combined_objective": sum(row["weighted_objective"] + config.semantic_weight * row["semantic_auxiliary"] for row in history) / len(history),
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
