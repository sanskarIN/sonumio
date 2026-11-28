"""Declarative allowlisted offline audio processing DAG for local experiments.

A graph is data, not executable source: no Python evaluation, plugins,
network endpoints, arbitrary import names, file paths, shell commands or
models. Strict topological validation, bounded nodes and fixed sample
clocks reduce accidental denial-of-service and hidden generation changes.
This graph engine is opt-in; the existing Sonumio API never invokes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import AudioBuffer, finite_float
from .errors import AudioContractError, AudioLimitError
from .hashing import canonical_json_bytes, digest_audio
from .numerics import db_to_gain, mix_accumulate, saturate
from .segments import reverse, trim, pad, concatenate, crossfade
from .dynamics import (
    CompressorConfig, compress, GateConfig, noise_gate,
    peak_limit, peak_normalize, rms_normalize
)
from .filters import FilterSpec, biquad_filter, dc_block
from .effects import DelayConfig, delay, ModulationConfig, tremolo
from .automation import (
    AutomationCurve, Keyframe, apply_gain_automation
)
from .windows import envelope
from .multiband import CrossoverPlan, multiband_compress

_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
OPS: dict[str, tuple[int, int]] = {
    "source": (0, 0),
    "gain": (1, 1),
    "reverse": (1, 1),
    "trim": (1, 1),
    "pad": (1, 1),
    "fade": (1, 1),
    "softclip": (1, 1),
    "biquad": (1, 1),
    "dc_block": (1, 1),
    "compress": (1, 1),
    "multiband_compress": (1, 1),
    "noise_gate": (1, 1),
    "peak_limit": (1, 1),
    "peak_normalize": (1, 1),
    "rms_normalize": (1, 1),
    "delay": (1, 1),
    "tremolo": (1, 1),
    "automate": (1, 1),
    "mix": (2, 16),
    "concat": (2, 16),
    "crossfade": (2, 2),
}

_PARAMETERS: dict[str, frozenset[str]] = {
    "source": frozenset({"source_id"}),
    "gain": frozenset({"db"}),
    "reverse": frozenset(),
    "trim": frozenset({"start", "stop"}),
    "pad": frozenset({"before", "after", "mode"}),
    "fade": frozenset({"fade_in", "fade_out", "curve"}),
    "softclip": frozenset({"mode", "threshold"}),
    "biquad": frozenset({"kind", "cutoff_hz", "q", "gain_db", "passes"}),
    "dc_block": frozenset({"pole"}),
    "compress": frozenset({
        "threshold_dbfs", "ratio", "attack_seconds", "release_seconds",
        "knee_db", "makeup_db", "link_channels",
    }),
    "multiband_compress": frozenset({
        "cutoffs_hz", "taps", "window", "thresholds_dbfs", "ratios",
        "makeup_db", "attack_seconds", "release_seconds",
        "knee_db", "link_channels", "wet", "output_gain_db",
    }),
    "noise_gate": frozenset({
        "open_threshold_dbfs", "close_threshold_dbfs", "attack_seconds",
        "release_seconds", "hold_seconds", "floor_db",
    }),
    "peak_limit": frozenset({"ceiling_dbfs", "release_seconds"}),
    "peak_normalize": frozenset({"target_dbfs", "allow_boost"}),
    "rms_normalize": frozenset({"target_dbfs", "max_gain_db"}),
    "delay": frozenset({"delay_seconds", "feedback", "wet", "taps", "ping_pong"}),
    "tremolo": frozenset({"rate_hz", "depth", "wet", "waveform", "phase"}),
    "automate": frozenset({"keys", "unit"}),
    "mix": frozenset({"gains_db", "normalize"}),
    "concat": frozenset(),
    "crossfade": frozenset({"frames", "law"}),
}


def _simple_parameter(value: Any, *, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if value is None or type(value) in (bool, str, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, (tuple, list)):
        return len(value) <= 64 and all(
            _simple_parameter(v, depth=depth + 1) for v in value
        )
    return False


def _freeze_parameter(value):
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter(x) for x in value)
    return value


def _export_parameter(value):
    if isinstance(value, tuple):
        return [_export_parameter(x) for x in value]
    return value


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    operation: str
    inputs: tuple[str, ...] = ()
    parameters: dict[str, Any] | None = None

    def __post_init__(self):
        if not isinstance(self.node_id, str) or not _ID.fullmatch(self.node_id):
            raise AudioContractError("Invalid declarative node ID")
        if self.operation not in OPS:
            raise AudioContractError("Non-allowlisted audio graph operation")
        if not isinstance(self.inputs, tuple) or not all(
            isinstance(x, str) and _ID.fullmatch(x) for x in self.inputs
        ):
            raise AudioContractError("Node references must be validated IDs")
        low, high = OPS[self.operation]
        if not low <= len(self.inputs) <= high:
            raise AudioContractError("Wrong number of signal graph inputs")
        if len(set(self.inputs)) != len(self.inputs):
            raise AudioContractError("Duplicate graph input reference")
        params = {} if self.parameters is None else self.parameters
        if (type(params) is not dict or len(params) > 20
                or not set(params).issubset(_PARAMETERS[self.operation])
                or not all(_simple_parameter(v) for v in params.values())):
            raise AudioContractError("Unsupported or nonfinite graph parameters")
        # Retain no references to input dictionaries or nested keyframe lists.
        # A frozen dataclass containing a mutable dict is NOT really frozen.
        object.__setattr__(self, "parameters", MappingProxyType({
            key: _freeze_parameter(value) for key, value in params.items()
        }))


@dataclass(frozen=True, slots=True)
class SignalGraph:
    """Nodes must appear in topological order. Exactly one sink is selected."""

    nodes: tuple[GraphNode, ...]
    output: str

    def __post_init__(self):
        if not isinstance(self.nodes, tuple) or not 1 <= len(self.nodes) <= 96:
            raise AudioLimitError("Signal graph needs 1..96 nodes")
        if any(not isinstance(x, GraphNode) for x in self.nodes):
            raise AudioContractError("Expected GraphNode entries")
        names = [node.node_id for node in self.nodes]
        if len(set(names)) != len(names):
            raise AudioContractError("Duplicate processing node IDs")
        known = set()
        for node in self.nodes:
            if any(x not in known for x in node.inputs):
                raise AudioContractError("Graph references future/missing node or cycle")
            known.add(node.node_id)
        if self.output not in known:
            raise AudioContractError("Unknown graph output sink")
        # All source nodes must have a bound source_id; no file path is allowed.
        for node in self.nodes:
            if node.operation == "source" and not node.parameters.get("source_id"):
                raise AudioContractError("Source node needs explicit source_id")
        canonical_json_bytes(self.as_dict())

    def as_dict(self) -> dict:
        return {
            "version": 1,
            "nodes": [
                {"id": n.node_id, "operation": n.operation,
                 "inputs": list(n.inputs), "parameters": {
                    key: _export_parameter(value)
                    for key, value in n.parameters.items()
                 }}
                for n in self.nodes
            ],
            "output": self.output,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()

    @classmethod
    def from_dict(cls, raw: dict) -> "SignalGraph":
        if (type(raw) is not dict or set(raw) != {"version", "nodes", "output"}
                or raw["version"] != 1 or not isinstance(raw["nodes"], list)):
            raise AudioContractError("Invalid versioned audio-graph object")
        nodes = []
        for obj in raw["nodes"]:
            if not isinstance(obj, dict) or set(obj) != {
                "id", "operation", "inputs", "parameters"
            }:
                raise AudioContractError("Invalid signal node schema")
            if not isinstance(obj["inputs"], list):
                raise AudioContractError("Signal node input list required")
            nodes.append(GraphNode(obj["id"], obj["operation"],
                                   tuple(obj["inputs"]), obj["parameters"]))
        return cls(tuple(nodes), raw["output"])


@dataclass(frozen=True, slots=True)
class NodeExecution:
    node_id: str
    operation: str
    frames: int
    channels: int
    sample_rate: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GraphResult:
    audio: AudioBuffer
    graph_sha256: str
    trace: tuple[NodeExecution, ...]
    source_modification: bool = False
    neural_model_inference: bool = False

    def report(self) -> dict:
        return {
            "graph_sha256": self.graph_sha256,
            "result_audio_sha256": digest_audio(self.audio),
            "source_modification": self.source_modification,
            "neural_model_inference": self.neural_model_inference,
            "nodes": [{
                "node_id": x.node_id, "operation": x.operation,
                "sample_rate": x.sample_rate, "channels": x.channels,
                "frames": x.frames, "sha256": x.sha256
            } for x in self.trace],
        }


def _execute(node: GraphNode, inputs: list[AudioBuffer],
             sources: Mapping[str, AudioBuffer]) -> AudioBuffer:
    op, p = node.operation, node.parameters
    if op == "source":
        identifier = p["source_id"]
        if not isinstance(identifier, str) or identifier not in sources:
            raise AudioContractError("Missing declared in-memory source_id")
        source = sources[identifier]
        if not isinstance(source, AudioBuffer):
            raise AudioContractError("Graph sources must be AudioBuffer objects")
        return source
    head = inputs[0]
    if op == "gain":
        db = finite_float(p.get("db", 0), "gain db", minimum=-120, maximum=48)
        return head.copy_with((head.samples.astype(np.float64) *
                               db_to_gain(db)).astype(np.float32))
    if op == "reverse":
        return reverse(head)
    if op == "trim":
        return trim(head, p.get("start", 0), p.get("stop"))
    if op == "pad":
        return pad(head, before=p.get("before", 0),
                   after=p.get("after", 0), mode=p.get("mode", "zeros"))
    if op == "fade":
        f = envelope(head.frames, fade_in=p.get("fade_in", 0),
                     fade_out=p.get("fade_out", 0), curve=p.get("curve", "cosine"))
        return head.copy_with((head.samples * f[None, :]).astype(np.float32))
    if op == "softclip":
        return head.copy_with(saturate(head.samples, threshold=p.get("threshold", 1),
                                       mode=p.get("mode", "tanh")))
    if op == "biquad":
        values = {k: v for k, v in p.items() if k not in ("passes",)}
        return biquad_filter(head, FilterSpec(head.sample_rate, **values),
                             passes=p.get("passes", 1))
    if op == "dc_block":
        return head.copy_with(dc_block(head.samples, pole=p.get("pole", .995)))
    if op == "compress":
        return compress(head, CompressorConfig(**p))
    if op == "multiband_compress":
        cutoffs = p.get("cutoffs_hz", (180., 2500.))
        if not isinstance(cutoffs, (tuple, list)):
            raise AudioContractError("Multiband cutoffs must be an ordered array")
        plan = CrossoverPlan(
            head.sample_rate, tuple(cutoffs),
            taps=p.get("taps", 257), window=p.get("window", "blackman"),
        )
        thresholds = p.get("thresholds_dbfs", (-18.,) * plan.bands)
        ratios = p.get("ratios", (4.,) * plan.bands)
        makeup = p.get("makeup_db", (0.,) * plan.bands)
        for name, values in (
            ("thresholds_dbfs", thresholds), ("ratios", ratios), ("makeup_db", makeup),
        ):
            if not isinstance(values, (tuple, list)) or len(values) != plan.bands:
                raise AudioContractError(name + " must match crossover band count")
        configs = tuple(
            CompressorConfig(
                threshold_dbfs=thresholds[i], ratio=ratios[i],
                makeup_db=makeup[i],
                attack_seconds=p.get("attack_seconds", .01),
                release_seconds=p.get("release_seconds", .1),
                knee_db=p.get("knee_db", 6),
                link_channels=p.get("link_channels", True),
            )
            for i in range(plan.bands)
        )
        return multiband_compress(
            head, plan, configs, wet=p.get("wet", 1.),
            output_gain_db=p.get("output_gain_db", 0.),
        ).audio
    if op == "noise_gate":
        return noise_gate(head, GateConfig(**p))
    if op == "peak_limit":
        return peak_limit(head, **p)
    if op == "peak_normalize":
        return peak_normalize(head, **p)
    if op == "rms_normalize":
        return rms_normalize(head, **p)
    if op == "delay":
        return delay(head, DelayConfig(**p))
    if op == "tremolo":
        return tremolo(head, ModulationConfig(**p))
    if op == "automate":
        raw_keys = p.get("keys")
        if not isinstance(raw_keys, (list, tuple)) or not 1 <= len(raw_keys) <= 64:
            raise AudioContractError("Automation graph requires bounded keys")
        keys = []
        for obj in raw_keys:
            if not isinstance(obj, (list, tuple)) or len(obj) not in (2, 3):
                raise AudioContractError("Automation graph key invalid")
            keys.append(Keyframe(*obj))
        return apply_gain_automation(head, AutomationCurve(tuple(keys)),
                                     gain_unit=p.get("unit", "linear"))
    if op in ("mix", "concat"):
        if any(x.sample_rate != head.sample_rate or
               x.channels != head.channels for x in inputs):
            raise AudioContractError("Graph mix/concat source clocks differ")
        if op == "concat":
            return concatenate(inputs)
        gains = p.get("gains_db", [0.0] * len(inputs))
        if not isinstance(gains, (tuple, list)):
            raise AudioContractError("mix gains require list")
        values = mix_accumulate([x.samples for x in inputs],
                                gains_db=gains, normalize=p.get("normalize", "none"))
        return head.copy_with(values, source_note="declarative mix")
    if op == "crossfade":
        if inputs[0].sample_rate != inputs[1].sample_rate:
            raise AudioContractError("Crossfade graph clocks differ")
        return crossfade(inputs[0], inputs[1], frames=p.get("frames"),
                         law=p.get("law", "equal_power"))
    raise AudioContractError("Graph operation not implemented in allowlist")


def execute_graph(graph: SignalGraph, sources: Mapping[str, AudioBuffer],
                  *, max_total_frames: int = 4_000_000) -> GraphResult:
    if not isinstance(graph, SignalGraph) or not isinstance(sources, Mapping):
        raise AudioContractError("Expected declarative graph and in-memory sources")
    if type(max_total_frames) is not int or not 1 <= max_total_frames <= 100_000_000:
        raise AudioContractError("Invalid total graph frame budget")
    if len(sources) > 64:
        raise AudioLimitError("At most 64 in-memory sources")
    cache: dict[str, AudioBuffer] = {}
    trace = []
    total = 0
    for node in graph.nodes:
        inputs = [cache[k] for k in node.inputs]
        audio = _execute(node, inputs, sources)
        total += audio.frames * audio.channels
        if total > max_total_frames:
            raise AudioLimitError("Processing graph exceeds total materialized frame cap")
        cache[node.node_id] = audio
        trace.append(NodeExecution(
            node.node_id, node.operation, audio.frames, audio.channels,
            audio.sample_rate, digest_audio(audio),
        ))
    return GraphResult(cache[graph.output], graph.sha256, tuple(trace))


def graph_safety_report(graph: SignalGraph, sources: Mapping[str, AudioBuffer]
                        ) -> dict:
    if not isinstance(graph, SignalGraph):
        raise AudioContractError("Expected SignalGraph")
    names = [n.parameters["source_id"] for n in graph.nodes
             if n.operation == "source"]
    return {
        "graph_sha256": graph.sha256,
        "declared_sources": names,
        "unresolved_sources": [s for s in names if s not in sources],
        "nodes": len(graph.nodes),
        "executable_code_allowed": False,
        "file_or_network_access_allowed": False,
        "model_inference_allowed": False,
        "requires_human_rights_review_before_training": True,
    }
