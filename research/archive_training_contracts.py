"""Strict configuration and progress contracts for private CPU archive training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import platform

import numpy as np
import torch

from research.archive_sampler import MAX_DRAWS
from research.corpus_archive_format import ArchiveError
from research.wave_diffusion import DiffusionSchedule

MAX_TRAINING_STEPS = 50_000
OPTIMIZER_PROTOCOL = "sonumio-cpu-adamw-v1"


@dataclass(frozen=True, slots=True)
class ArchiveTrainingConfig:
    seed: int = 9
    steps: int = 50
    batch_size: int = 8
    channels: int = 32
    diffusion_steps: int = 32
    diffusion_profile: str = "linear_beta_v1"
    validation_seed: int = 101
    validation_repeats: int = 2
    validation_batch_size: int = 8

    def __post_init__(self):
        bounds = (
            (self.seed, 0, 2**32-1), (self.steps, 1, MAX_TRAINING_STEPS),
            (self.batch_size, 1, 64), (self.channels, 8, 64),
            (self.diffusion_steps, 8, 1000), (self.validation_seed, 0, 2**32-1),
            (self.validation_repeats, 1, 4), (self.validation_batch_size, 1, 64),
        )
        if any(type(x) is not int or not low <= x <= high for x, low, high in bounds):
            raise ArchiveError("Invalid archive training configuration")
        if self.channels % 8 or self.steps*self.batch_size > MAX_DRAWS:
            raise ArchiveError("Archive training channel or draw budget exceeded")
        if type(self.diffusion_profile) is not str or self.diffusion_profile not in DiffusionSchedule.PROFILES:
            raise ArchiveError("Unknown archive diffusion profile")

    def trajectory_identity(self) -> dict:
        """The target total step count may grow; every other setting is pinned."""
        result = asdict(self)
        del result["steps"]
        return result


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    completed_steps: int = 0
    first_loss: float | None = None
    last_loss: float | None = None
    loss_total: float = 0.0

    def __post_init__(self):
        if type(self.completed_steps) is not int or not 0 <= self.completed_steps <= MAX_TRAINING_STEPS:
            raise ArchiveError("Invalid completed training step count")
        if not self.completed_steps:
            if self.first_loss is not None or self.last_loss is not None or self.loss_total != 0:
                raise ArchiveError("Empty progress must not contain training losses")
        else:
            values = (self.first_loss, self.last_loss, self.loss_total)
            if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
                raise ArchiveError("Invalid finite nonnegative training loss summary")
            if self.loss_total < max(self.first_loss, self.last_loss):
                raise ArchiveError("Training loss total is inconsistent")
            if self.completed_steps == 1 and not self.first_loss == self.last_loss == self.loss_total:
                raise ArchiveError("Single-step training losses disagree")

    def advance(self, loss: float) -> "TrainingProgress":
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ArchiveError("Invalid training step loss")
        return TrainingProgress(
            self.completed_steps+1, float(loss) if self.completed_steps == 0 else self.first_loss,
            float(loss), self.loss_total + float(loss),
        )

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def parse(cls, value: object) -> "TrainingProgress":
        if not isinstance(value, dict) or set(value) != {
            "completed_steps", "first_loss", "last_loss", "loss_total",
        }:
            raise ArchiveError("Invalid training progress fields")
        return cls(**value)


def optimizer_identity() -> dict:
    return {
        "protocol": OPTIMIZER_PROTOCOL, "lr": .001, "betas": [.9, .999],
        "eps": 1e-8, "weight_decay": .01, "amsgrad": False,
        "foreach": False, "fused": False, "maximize": False,
        "capturable": False, "differentiable": False,
    }


def make_archive_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    options = optimizer_identity()
    del options["protocol"]
    options["betas"] = tuple(options["betas"])
    return torch.optim.AdamW(model.parameters(), **options)


def require_optimizer_contract(
    model: torch.nn.Module, optimizer: torch.optim.AdamW,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ArchiveError("Expected a single-group archive AdamW optimizer")
    parameters = tuple(model.named_parameters())
    group = optimizer.param_groups[0]
    actual = group["params"]
    if len(actual) != len(parameters) or any(
        observed is not wanted for observed, (_, wanted) in zip(actual, parameters, strict=True)
    ):
        raise ArchiveError("Optimizer parameters do not match the archive model")
    options = optimizer_identity()
    del options["protocol"]
    options["betas"] = tuple(options["betas"])
    for key, expected in options.items():
        observed = group.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise ArchiveError("Archive AdamW settings changed: " + key)
    return parameters


def runtime_identity() -> dict:
    """Pin a conservative same-environment resume policy.

    This is not a proof of identical CPU hardware or cross-platform numerical
    reproducibility. Exact continuation is tested in the same environment.
    """
    if torch.get_default_dtype() != torch.float32:
        raise ArchiveError("Archive training requires the default float32 dtype")
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": "float32",
        "device": "cpu",
    }
