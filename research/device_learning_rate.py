"""Bounded per-optimizer-update learning-rate policies.

The default is constant, preserving the existing experimental trajectory.
Warmup/cosine uses integer logical update indices, not microbatch counts.
The schedule is computed without global RNG or optimizer mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

POLICIES = ("constant", "warmup_cosine")


@dataclass(frozen=True)
class LearningRatePlan:
    base_rate: float
    updates: int
    policy: str = "constant"
    warmup_updates: int = 0
    minimum_ratio: float = 0.1

    def __post_init__(self):
        if (type(self.base_rate) not in (int, float) or not math.isfinite(self.base_rate)
                or not 1e-6 <= self.base_rate <= .01
                or type(self.updates) is not int or not 1 <= self.updates <= 4096
                or type(self.policy) is not str or self.policy not in POLICIES
                or type(self.warmup_updates) is not int or not 0 <= self.warmup_updates < self.updates
                or type(self.minimum_ratio) not in (int, float)
                or not math.isfinite(self.minimum_ratio) or not 0 <= self.minimum_ratio <= 1):
            raise ValueError("Invalid learning-rate schedule")
        if self.policy == "constant" and self.warmup_updates != 0:
            raise ValueError("Constant learning rate does not support warmup")
        if self.policy == "warmup_cosine" and self.updates - self.warmup_updates < 2:
            raise ValueError("Cosine decay needs at least two post-warmup updates")

    def rate(self, index: int) -> float:
        if type(index) is not int or not 0 <= index < self.updates:
            raise ValueError("Learning-rate update index out of range")
        if self.policy == "constant":
            return float(self.base_rate)
        if index < self.warmup_updates:
            return float(self.base_rate * (index + 1) / self.warmup_updates)
        progress = (index - self.warmup_updates) / (self.updates - self.warmup_updates - 1)
        cosine = .5 * (1 + math.cos(math.pi * progress))
        return float(self.base_rate * (self.minimum_ratio + (1 - self.minimum_ratio) * cosine))

    def describe(self) -> dict:
        values = [self.rate(index) for index in range(self.updates)]
        return {
            "policy": self.policy, "base_learning_rate": float(self.base_rate),
            "warmup_updates": self.warmup_updates, "minimum_ratio": float(self.minimum_ratio),
            "logical_optimizer_updates": self.updates,
            "first_learning_rate": values[0], "last_learning_rate": values[-1],
            "minimum_learning_rate": min(values), "maximum_learning_rate": max(values),
            "mean_learning_rate": math.fsum(values) / len(values),
            "per_update_learning_rates": values,
        }


def set_learning_rate(optimizer, plan: LearningRatePlan, update: int) -> float:
    """Apply once before a logical optimizer update, never per microbatch."""
    rate = plan.rate(update)
    if not optimizer.param_groups:
        raise ValueError("Optimizer has no parameter groups")
    for group in optimizer.param_groups:
        group["lr"] = rate
    return rate
