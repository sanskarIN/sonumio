"""Explicit, versioned and tamper-detectable acoustic evaluation plans.

Plans require a prespecified metric, source split, random seed, baseline
engine and provenance policy before analysis. They do not authenticate
recording rights, discover a held-out corpus or substitute for blind
human listening. Original Sonumio-1 checkpoints and public model weights
are out of scope; there is no automatic training or model ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
import re
from typing import Any

from sonumio.audio_core.errors import AudioContractError
from sonumio.audio_core.hashing import canonical_json_bytes

_METRIC_DIRECTIONS = {
    "sample_rmse": "lower",
    "max_sample_absolute_error": "lower",
    "mean_sample_absolute_error": "lower",
    "signal_error_ratio_db": "higher",
    "si_sdr_db": "higher",
    "normalized_correlation": "higher",
    "multiresolution_log_spectral_distance": "lower",
    "multiresolution_spectral_convergence": "lower",
}
_ALLOWED_SPLITS = frozenset({"validation", "test", "stress"})
_ENGINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


@dataclass(frozen=True, slots=True)
class MetricRule:
    """One objective with natural direction and optional acceptance threshold."""

    name: str
    accepted_boundary: float | None = None
    weight: float = 1.0

    def __post_init__(self):
        if self.name not in _METRIC_DIRECTIONS:
            raise AudioContractError("Metric is outside prespecified acoustic metric catalog")
        if (self.accepted_boundary is not None and
                (type(self.accepted_boundary) not in (float, int)
                 or not math.isfinite(self.accepted_boundary))):
            raise AudioContractError("Metric threshold must be finite")
        if type(self.weight) not in (int, float) or not math.isfinite(
            self.weight
        ) or not 0 < self.weight <= 100:
            raise AudioContractError("Metric weight must be positive and finite")

    @property
    def direction(self) -> str:
        return _METRIC_DIRECTIONS[self.name]

    def meets_threshold(self, value: float) -> bool | None:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise AudioContractError("Nonfinite observed evaluation metric")
        if self.accepted_boundary is None:
            return None
        if self.direction == "higher":
            return value >= self.accepted_boundary
        return value <= self.accepted_boundary

    def naturalized_difference(self, candidate: float, baseline: float) -> float:
        """Positive means the candidate is better under this one metric."""
        if any(type(x) not in (float, int) or not math.isfinite(x)
               for x in (candidate, baseline)):
            raise AudioContractError("Nonfinite paired acoustic metric scores")
        change = candidate - baseline
        return change if self.direction == "higher" else -change


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """Fixed evaluation protocol, passed around as immutable data."""

    plan_id: str
    split: str
    metric: MetricRule
    candidate_engine: str
    fft_sizes: tuple[int, ...] = (256, 512, 1024)
    minimum_groups: int = 2
    bootstrap_iterations: int = 1000
    confidence: float = .95
    seed: int = 0
    source_rights_review_required: bool = True
    human_listening_required_before_release: bool = True
    notes: str = ""

    def __post_init__(self):
        if not isinstance(self.plan_id, str) or _ENGINE.fullmatch(self.plan_id) is None:
            raise AudioContractError("Invalid predeclared acoustic plan id")
        if self.split not in _ALLOWED_SPLITS:
            raise AudioContractError("Cannot report trained acoustic cases as evaluation")
        if not isinstance(self.metric, MetricRule):
            raise AudioContractError("EvaluationPlan needs a MetricRule")
        if not isinstance(self.candidate_engine, str) or _ENGINE.fullmatch(
            self.candidate_engine
        ) is None:
            raise AudioContractError("Missing explicit candidate engine identity")
        if (not isinstance(self.fft_sizes, tuple) or
                not 1 <= len(self.fft_sizes) <= 5 or
                len(set(self.fft_sizes)) != len(self.fft_sizes) or
                any(type(n) is not int or not 32 <= n <= 8192
                    for n in self.fft_sizes)):
            raise AudioContractError("Invalid bounded spectral resolution list")
        if type(self.minimum_groups) is not int or not 2 <= self.minimum_groups <= 10000:
            raise AudioContractError("Evaluation requires two or more independent groups")
        if type(self.bootstrap_iterations) is not int or not 100 <= self.bootstrap_iterations <= 100_000:
            raise AudioContractError("Invalid group bootstrap iteration budget")
        if type(self.confidence) not in (int, float) or not math.isfinite(
            self.confidence
        ) or not .5 <= self.confidence < 1:
            raise AudioContractError("Invalid confidence level")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise AudioContractError("Prespecified random seed is required")
        if self.source_rights_review_required is not True:
            raise AudioContractError("Cannot disable rights review in a final evaluation plan")
        if self.human_listening_required_before_release is not True:
            raise AudioContractError("Cannot waive listening review from a numerical plan")
        if not isinstance(self.notes, str) or len(self.notes) > 4000:
            raise AudioContractError("Invalid evaluation plan notes")

    def as_dict(self) -> dict:
        value = asdict(self)
        value["protocol"] = "sonumio-prespecified-acoustic-evaluation-v1"
        value["metric"]["direction"] = self.metric.direction
        value["automatically_licensed"] = False
        value["publication_authorized"] = False
        return value

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationPlan":
        if not isinstance(data, dict) or set(data) != {
            "plan_id", "split", "metric", "candidate_engine", "fft_sizes",
            "minimum_groups", "bootstrap_iterations", "confidence", "seed",
            "source_rights_review_required",
            "human_listening_required_before_release", "notes",
            "protocol", "automatically_licensed", "publication_authorized",
        }:
            raise AudioContractError("Invalid versioned evaluation plan schema")
        if (data["protocol"] != "sonumio-prespecified-acoustic-evaluation-v1"
                or data["automatically_licensed"] is not False
                or data["publication_authorized"] is not False):
            raise AudioContractError("Invalid evaluation plan provenance claims")
        metric_data = data["metric"]
        if (not isinstance(metric_data, dict)
                or set(metric_data) != {
                    "name", "accepted_boundary", "weight", "direction"
                }):
            raise AudioContractError("Invalid prespecified metric metadata")
        metric = MetricRule(
            metric_data["name"], metric_data["accepted_boundary"],
            metric_data["weight"],
        )
        if metric_data["direction"] != metric.direction:
            raise AudioContractError("Reported metric direction was altered")
        return cls(
            plan_id=data["plan_id"],
            split=data["split"], metric=metric,
            candidate_engine=data["candidate_engine"],
            fft_sizes=tuple(data["fft_sizes"]),
            minimum_groups=data["minimum_groups"],
            bootstrap_iterations=data["bootstrap_iterations"],
            confidence=data["confidence"], seed=data["seed"],
            source_rights_review_required=data["source_rights_review_required"],
            human_listening_required_before_release=data["human_listening_required_before_release"],
            notes=data["notes"],
        )


def metric_catalog() -> tuple[dict, ...]:
    """Public immutable-safe objective catalog for evaluation interfaces."""
    return tuple({
        "name": name,
        "direction": direction,
        "requires_human_interpretation": True,
        "trained_embedding_model_required": False,
    } for name, direction in sorted(_METRIC_DIRECTIONS.items()))


def preregistration_report(plan: EvaluationPlan) -> dict:
    if not isinstance(plan, EvaluationPlan):
        raise AudioContractError("Expected EvaluationPlan")
    return {
        "plan_sha256": plan.sha256,
        "plan_id": plan.plan_id,
        "metric": plan.metric.name,
        "direction": plan.metric.direction,
        "split": plan.split,
        "candidate_engine": plan.candidate_engine,
        "independent_groups_required": plan.minimum_groups,
        "human_rights_review_still_required": True,
        "human_listening_before_model_publication_required": True,
        "trained_audio_model_available": False,
        "audio_files_exported": False,
    }
