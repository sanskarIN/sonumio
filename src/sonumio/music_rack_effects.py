"""Strictly allowlisted, deterministic stereo music insert processors.

The rack is offline and preserves the score's frame count. It does not load
plugins, external impulse responses, executable expressions or model weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .audio import SAMPLE_RATE
from .errors import GenerationError
from .audio_core.contracts import make_buffer
from .audio_core.filters import FilterSpec, design_biquad, biquad_filter
from .audio_core.dynamics import CompressorConfig, GateConfig, compress, noise_gate
from .audio_core.effects import DelayConfig, ModulationConfig, delay, tremolo
from .audio_core.numerics import saturate

MAX_STAGE_PEAK = 32.
MAX_CHAIN = 6
MAX_EFFECT_FRAMES = 60 * SAMPLE_RATE

# Each entry specifies default, lower/upper bounds or allowed values.
SCHEMAS = {
    "gain": {"db": (0., -36., 18.)},
    "width": {"width": (1., 0., 2.)},
    "eq": {
        "kind": ("peaking", ("lowpass", "highpass", "bandpass", "notch",
                              "peaking", "lowshelf", "highshelf")),
        "cutoff_hz": (1000., 20., 11000.),
        "q": (.7071067811865476, .1, 10.),
        "gain_db": (0., -18., 18.),
    },
    "compressor": {
        "threshold_dbfs": (-18., -60., 0.),
        "ratio": (4., 1., 20.),
        "attack_seconds": (.01, 0., .2),
        "release_seconds": (.1, 0., 2.),
        "knee_db": (6., 0., 18.),
        "makeup_db": (0., -18., 12.),
    },
    "gate": {
        "open_threshold_dbfs": (-42., -90., 0.),
        "close_threshold_dbfs": (-50., -100., 0.),
        "attack_seconds": (.005, 0., .2),
        "release_seconds": (.08, 0., 2.),
        "hold_seconds": (.02, 0., .5),
        "floor_db": (-90., -120., 0.),
    },
    "delay": {
        "delay_seconds": (.24, .001, 2.),
        "feedback": (.25, 0., .8),
        "taps": (3, 1, 8),
        "ping_pong": (False,),
    },
    "tremolo": {
        "rate_hz": (1., .05, 20.),
        "depth": (.5, 0., 1.),
        "waveform": ("sine", ("sine", "triangle")),
        "phase": (0., -2 * math.pi, 2 * math.pi),
    },
    "saturation": {
        "threshold": (1., .05, 4.),
        "mode": ("tanh", ("tanh", "cubic")),
    },
}

# Cost units are conservative scheduling estimates, not elapsed-time promises.
COST = {"gain": 1, "width": 1, "eq": 8, "compressor": 8, "gate": 12,
        "delay": 2, "tremolo": 2, "saturation": 2}


def _real(value, low, high, label):
    if type(value) not in (int, float) or not low <= value <= high:
        raise GenerationError("Invalid rack " + label)
    return float(value)


def _parameter(value, rule, label):
    default = rule[0]
    if type(default) is bool:
        if type(value) is not bool:
            raise GenerationError("Invalid rack " + label)
        return value
    if type(default) is str:
        if type(value) is not str or value not in rule[1]:
            raise GenerationError("Invalid rack " + label)
        return value
    if type(default) is int:
        if type(value) is not int or not rule[1] <= value <= rule[2]:
            raise GenerationError("Invalid rack " + label)
        return value
    return _real(value, rule[1], rule[2], label)


def _normalize(kind, values):
    if type(kind) is not str or kind not in SCHEMAS:
        raise GenerationError("Unknown music rack processor")
    if type(values) is not dict or not set(values) <= set(SCHEMAS[kind]):
        raise GenerationError("Unexpected rack processor parameters")
    result = {key: _parameter(values.get(key, rule[0]), rule, key)
              for key, rule in SCHEMAS[kind].items()}
    if kind == "eq":
        # Shelves use Q as slope; the tighter range avoids marginal poles.
        if result["kind"] in ("lowshelf", "highshelf") and result["q"] > 1:
            raise GenerationError("Shelf slope must not exceed one")
        if result["kind"] not in ("peaking", "lowshelf", "highshelf") and result["gain_db"] != 0:
            raise GenerationError("Gain is not used by this filter kind")
        design_biquad(FilterSpec(SAMPLE_RATE, **result))
    elif kind == "gate":
        GateConfig(**result)
    elif kind == "compressor":
        CompressorConfig(**result)
    elif kind == "delay":
        DelayConfig(**result, wet=1.)
    elif kind == "tremolo":
        ModulationConfig(**result, wet=1.)
    return tuple(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class Insert:
    kind: str
    parameters: tuple = ()
    wet: float = 1.
    bypass: bool = False

    def __post_init__(self):
        if (type(self.parameters) is not tuple
                or len(self.parameters) > 12
                or any(type(row) is not tuple or len(row) != 2
                       or type(row[0]) is not str for row in self.parameters)):
            raise GenerationError("Expected bounded rack parameter tuple")
        values = dict(self.parameters)
        if len(values) != len(self.parameters):
            raise GenerationError("Duplicate rack parameter")
        object.__setattr__(self, "parameters", _normalize(self.kind, values))
        object.__setattr__(self, "wet", _real(self.wet, 0., 1., "wet"))
        if type(self.bypass) is not bool:
            raise GenerationError("Invalid rack bypass")

    @classmethod
    def create(cls, kind, /, *, wet=1., bypass=False, **parameters):
        return cls(kind, tuple(parameters.items()), wet, bypass)

    @property
    def enabled(self):
        return not self.bypass and self.wet != 0.

    @property
    def cost(self):
        if not self.enabled:
            return 0
        if self.kind == "delay":
            return COST[self.kind] + dict(self.parameters)["taps"]
        return COST[self.kind]

    def as_dict(self):
        return {"kind": self.kind, "parameters": dict(self.parameters),
                "wet": self.wet, "bypass": self.bypass}


def validate_chain(chain):
    if (type(chain) is not tuple or len(chain) > MAX_CHAIN
            or any(not isinstance(item, Insert) for item in chain)):
        raise GenerationError("Invalid bounded music rack chain")


def _checked(samples):
    if (not isinstance(samples, np.ndarray)
            or samples.dtype not in (np.float32, np.float64)
            or samples.ndim != 2 or samples.shape[0] != 2
            or not 1 <= samples.shape[1] <= MAX_EFFECT_FRAMES
            or not np.isfinite(samples).all()
            or np.max(np.abs(samples)) > MAX_STAGE_PEAK):
        raise GenerationError("Invalid bounded rack stereo signal")
    return samples


def stage_digest(samples):
    samples = _checked(samples)
    h = hashlib.sha256(b"sonumio-rack-stage-f64le-v1\0")
    h.update(samples.shape[1].to_bytes(8, "little"))
    h.update(np.ascontiguousarray(samples, dtype="<f8").tobytes())
    return h.hexdigest()


def stage_stats(samples):
    samples = _checked(samples)
    data = samples.astype(np.float64, copy=False)
    return {"peak": float(np.max(np.abs(data))),
            "rms": float(np.sqrt(np.mean(data ** 2))),
            "sha256": stage_digest(data)}


def _effect(source, insert):
    values = dict(insert.parameters)
    if insert.kind == "gain":
        return source.astype(np.float64) * 10 ** (values["db"] / 20.)
    if insert.kind == "width":
        width = values["width"]
        if width == 1.:
            return source
        left, right = source.astype(np.float64)
        mid = (left + right) / 2.
        side = (left - right) / 2. * width
        return np.stack((mid + side, mid - side))
    audio = make_buffer(source, SAMPLE_RATE)
    if insert.kind == "eq":
        return biquad_filter(audio, FilterSpec(SAMPLE_RATE, **values)).samples
    if insert.kind == "compressor":
        return compress(audio, CompressorConfig(**values)).samples
    if insert.kind == "gate":
        return noise_gate(audio, GateConfig(**values)).samples
    if insert.kind == "delay":
        # The core delay returns dry when no tap fits. Here the wet branch
        # is explicitly silent in that case, preserving send semantics.
        if round(values["delay_seconds"] * SAMPLE_RATE) >= audio.frames:
            return np.zeros_like(source)
        return delay(audio, DelayConfig(**values, wet=1.)).samples
    if insert.kind == "tremolo":
        return tremolo(audio, ModulationConfig(**values, wet=1.)).samples
    if insert.kind == "saturation":
        return saturate(audio.samples, **values)
    raise GenerationError("Unknown rack processor")


def apply_insert(samples, insert):
    source = _checked(samples)
    if not isinstance(insert, Insert):
        raise GenerationError("Expected music rack insert")
    if not insert.enabled:
        return source
    wet = _checked(_effect(source, insert))
    if wet.shape != source.shape:
        raise GenerationError("Music rack changed score duration")
    if insert.wet == 1.:
        return wet
    return _checked(source.astype(np.float64) * (1. - insert.wet)
                    + wet.astype(np.float64) * insert.wet)


def apply_chain(samples, chain):
    validate_chain(chain)
    current = _checked(samples)
    stages = []
    for index, insert in enumerate(chain):
        before = stage_stats(current)
        current = apply_insert(current, insert)
        after = stage_stats(current)
        stages.append({"index": index, "insert": insert.as_dict(),
                       "enabled": insert.enabled, "input": before,
                       "output": after})
    return current, stages
