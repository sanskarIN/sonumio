"""Bounded in-memory EMA for Sonumio's experimental CPU neural denoisers.

For an online parameter theta, shadow <- decay * shadow + (1-decay) * theta.
Neither weights nor audio are persisted. This is not a quality guarantee.

Validate ALL tensors and candidate updates before changing ANY shadow tensor:
one nonfinite parameter, incompatible state or overflow must fail closed.
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    if not isinstance(module, nn.Module):
        raise ValueError("EMA source must be a PyTorch module")
    values = module.state_dict()
    if not values:
        raise ValueError("EMA requires a parameterized neural model")
    for name, tensor in values.items():
        if (not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or (tensor.is_floating_point() and tensor.dtype != torch.float32)
                or (not tensor.is_floating_point() and tensor.dtype not in
                    (torch.int8, torch.uint8, torch.int16, torch.int32,
                     torch.int64, torch.bool))
                or (tensor.is_floating_point() and not torch.isfinite(tensor).all())):
            raise ValueError("EMA requires finite CPU float32 model state and integral buffers")
    return values


def _structure_match(a: dict[str, torch.Tensor],
                     b: dict[str, torch.Tensor]) -> None:
    if set(a) != set(b):
        raise ValueError("EMA model state names changed")
    for name in a:
        if (a[name].shape != b[name].shape or a[name].dtype != b[name].dtype
                or a[name].device != b[name].device):
            raise ValueError("EMA model state layout changed at " + name)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    """One-way digest of model names/shapes/types/bytes, no weight publication."""
    digest = hashlib.sha256(b"sonumio-neural-ema-state-v1\0")
    for name, tensor in sorted(state.items()):
        data = tensor.detach().contiguous().numpy()
        digest.update(name.encode("utf8") + b"\0")
        digest.update(str(tuple(data.shape)).encode("ascii") + b"\0")
        digest.update(str(data.dtype).encode("ascii") + b"\0")
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


class WeightEMA:
    """Strictly isolated, frozen EMA shadow with optional gradual warmup."""

    def __init__(self, source: nn.Module, *, decay: float = 0.99,
                 warmup: bool = False) -> None:
        if (type(decay) not in (float, int) or not math.isfinite(decay)
                or not 0 <= decay < 1):
            raise ValueError("EMA decay must be finite in [0,1)")
        if type(warmup) is not bool:
            raise ValueError("EMA warmup must be boolean")
        original = _state(source)
        self.decay = float(decay)
        self.warmup = warmup
        self.updates = 0
        self.last_effective_decay: float | None = None
        self._shadow = copy.deepcopy(source).eval()
        self._shadow.requires_grad_(False)
        _structure_match(original, _state(self._shadow))
        self.initial_sha256 = _state_hash(original)

    @torch.no_grad()
    def update(self, source: nn.Module) -> float:
        """All-or-nothing no-grad update with no modification of online weights."""
        online = _state(source)
        target = _state(self._shadow)
        _structure_match(online, target)
        decay = (min(self.decay, (1 + self.updates) / (10 + self.updates))
                 if self.warmup else self.decay)
        replacements: dict[str, torch.Tensor] = {}
        for name, current in online.items():
            if current.is_floating_point():
                candidate = torch.lerp(current.detach(), target[name], decay)
                if not torch.isfinite(candidate).all():
                    raise ValueError("EMA arithmetic produced nonfinite state")
            else:
                candidate = current.detach().clone()
            replacements[name] = candidate
        for name, candidate in replacements.items():
            target[name].copy_(candidate)
        self._shadow.eval()
        self.updates += 1
        self.last_effective_decay = decay
        return decay

    def snapshot(self) -> nn.Module:
        """Return an independent frozen evaluation copy, not the live shadow."""
        result = copy.deepcopy(self._shadow).eval()
        result.requires_grad_(False)
        return result

    def state_sha256(self) -> str:
        return _state_hash(_state(self._shadow))

    def describe(self) -> dict:
        return {
            "protocol": "sonumio-in-memory-ema-v1",
            "decay": self.decay, "warmup": self.warmup,
            "updates": self.updates,
            "last_effective_decay": self.last_effective_decay,
            "initial_model_sha256": self.initial_sha256,
            "shadow_model_sha256": self.state_sha256(),
            "checkpoint_written": False,
            "model_weights_published": False,
            "model_generation": None,
        }


def model_state_sha256(source: nn.Module) -> str:
    """One-way digest of an existing model's current in-memory state."""
    return _state_hash(_state(source))
