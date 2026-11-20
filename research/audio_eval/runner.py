"""Execute prespecified metric evaluation over finite in-memory references.

The runner computes transparent per-case acoustic metrics and estimates
uncertainty at the independent source group level. It NEVER tunes a model,
uses test labels in training, uploads private audio, authenticates rights or
pretends that one spectral metric establishes human listening quality.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from sonumio.audio_core.errors import AudioContractError
from sonumio.audio_core.hashing import canonical_json_bytes
from .metrics import MetricSet, compare_signals
from .protocol import EvaluationPlan, MetricRule
from .reference import EvaluationPair, EvaluationSet
from .statistics import (
    grouped_bootstrap, group_effect_report, holm_adjust,
)


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    case_id: str
    asset_id: str
    group_id: str
    split: str
    candidate_engine: str
    metric: str
    score: float
    meets_threshold: bool | None
    candidate_sha256: str
    reference_sha256: str
    metric_protocol: str = "sonumio-numeric-audio-eval-v1"

    def __post_init__(self):
        if type(self.score) not in (int, float) or not math.isfinite(self.score):
            raise AudioContractError("Evaluation row must contain finite metric")
        if self.meets_threshold is not None and type(self.meets_threshold) is not bool:
            raise AudioContractError("Metric threshold decision must be explicit bool/none")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    plan_sha256: str
    dataset_sha256: str
    split: str
    rows: tuple[EvaluationRow, ...]
    metric: str
    independent_groups: int
    group_bootstrap: dict
    protocol: str = "sonumio-acoustic-numeric-benchmark-v1"

    def __post_init__(self):
        if not self.rows:
            raise AudioContractError("Cannot report an empty evaluation")
        if self.independent_groups < 2:
            raise AudioContractError("Cannot claim uncertainty for one independent group")
        if len({r.case_id for r in self.rows}) != len(self.rows):
            raise AudioContractError("Duplicate scored evaluation cases")
        if any(r.metric != self.metric or r.split != self.split for r in self.rows):
            raise AudioContractError("Metric or split changed during evaluation")

    @property
    def mean_metric(self) -> float:
        # Group-equal weighting; not raw clip-count weighting.
        scores = defaultdict(list)
        for row in self.rows:
            scores[row.group_id].append(row.score)
        return float(np.mean([
            float(np.mean(scores[group])) for group in sorted(scores)
        ]))

    @property
    def all_thresholds_pass(self) -> bool | None:
        decisions = [r.meets_threshold for r in self.rows]
        if any(x is None for x in decisions):
            return None
        return all(decisions)

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "plan_sha256": self.plan_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split": self.split,
            "metric": self.metric,
            "independent_groups": self.independent_groups,
            "cases": len(self.rows),
            "mean_metric_group_weighted": self.mean_metric,
            "all_case_thresholds_pass": self.all_thresholds_pass,
            "group_bootstrap": dict(self.group_bootstrap),
            "rows": [r.as_dict() for r in self.rows],
            "automated_audio_rights_authentication": False,
            "automatic_model_release_approval": False,
            "perceptual_audio_quality_assessed": False,
            "training_or_checkpoint_written": False,
            "audio_files_saved": False,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


def run_evaluation(samples: EvaluationSet,
                   plan: EvaluationPlan) -> EvaluationResult:
    if not isinstance(samples, EvaluationSet) or not isinstance(plan, EvaluationPlan):
        raise AudioContractError("Expected EvaluationSet and preregistered EvaluationPlan")
    if not samples.strict_rights:
        raise AudioContractError("Evaluation plan refuses unchecked rights declarations")
    selection = tuple(
        pair for pair in samples.pairs
        if pair.split == plan.split and pair.candidate_engine == plan.candidate_engine
    )
    if not selection:
        raise AudioContractError("No recordings match planned split and candidate engine")
    unique_groups = {p.group_id for p in selection}
    if len(unique_groups) < plan.minimum_groups:
        raise AudioContractError("Evaluation has too few independent source groups")
    if len(selection) > 5000:
        raise AudioContractError("Too many scored cases for bounded CPU evaluation")
    rows: list[EvaluationRow] = []
    metric = plan.metric
    for pair in sorted(selection, key=lambda p: p.case_id):
        values = compare_signals(pair, fft_sizes=plan.fft_sizes)
        score = float(getattr(values, metric.name))
        rows.append(EvaluationRow(
            case_id=pair.case_id,
            asset_id=pair.reference.asset_id,
            group_id=pair.reference.group_id,
            split=pair.split,
            candidate_engine=pair.candidate_engine,
            metric=metric.name,
            score=score,
            meets_threshold=metric.meets_threshold(score),
            candidate_sha256=values.candidate_sha256,
            reference_sha256=values.reference_sha256,
        ))
    scores = [r.score for r in rows]
    group_ids = [r.group_id for r in rows]
    interval = grouped_bootstrap(
        scores, group_ids, confidence=plan.confidence,
        iterations=plan.bootstrap_iterations, seed=plan.seed,
    ).as_dict()
    return EvaluationResult(
        plan_sha256=plan.sha256,
        dataset_sha256=samples.fingerprint,
        split=plan.split,
        rows=tuple(rows),
        metric=metric.name,
        independent_groups=len(unique_groups),
        group_bootstrap=interval,
    )


def compare_engine_reports(first: EvaluationResult,
                           second: EvaluationResult, *,
                           metric_rule: MetricRule,
                           seed: int = 0,
                           bootstrap_iterations: int = 1000,
                           permutations: int = 1000) -> dict:
    """Preplanned paired A/B contrast, matching cases and original references."""
    if not isinstance(first, EvaluationResult) or not isinstance(
        second, EvaluationResult
    ) or not isinstance(metric_rule, MetricRule):
        raise AudioContractError("Expected two audio metric reports and a direction rule")
    if first.split != second.split or first.metric != second.metric or (
        first.metric != metric_rule.name
    ):
        raise AudioContractError("Cannot compare different metric/split definitions")
    a = {r.case_id: r for r in first.rows}
    b = {r.case_id: r for r in second.rows}
    if a.keys() != b.keys():
        raise AudioContractError("A/B models must score identical case identities")
    for key in a:
        x, y = a[key], b[key]
        if (x.reference_sha256 != y.reference_sha256
                or x.asset_id != y.asset_id or x.group_id != y.group_id):
            raise AudioContractError("A/B reference source or group was replaced")
    names = sorted(a)
    values_a = [a[k].score for k in names]
    values_b = [b[k].score for k in names]
    groups = [a[k].group_id for k in names]
    report = group_effect_report(
        values_a, values_b, groups,
        seed=seed, bootstrap_iterations=bootstrap_iterations,
        permutations=permutations,
    )
    # group_effect_report assumes candidate-minus-reference; orient the
    # same numerical effect according to the preregistered natural direction.
    direction = 1 if metric_rule.direction == "higher" else -1
    raw_difference = report["interval"]["estimate"]
    return {
        "protocol": "sonumio-registered-paired-audio-ab-v1",
        "metric": first.metric,
        "improvement_direction": metric_rule.direction,
        "mean_naturalized_improvement": direction * raw_difference,
        "mean_raw_difference": raw_difference,
        "uncertainty": report,
        "case_count": len(names),
        "independent_groups": len(set(groups)),
        "source_cases_identical": True,
        "reference_bytes_identical": True,
        "automatic_audio_quality_winner_declared": False,
        "multiple_comparison_family_declared": False,
        "human_listening_mandatory_for_release": True,
    }


def numeric_benchmark_summary(results: Sequence[EvaluationResult]) -> dict:
    if not isinstance(results, (list, tuple)) or not 1 <= len(results) <= 32:
        raise AudioContractError("Expected bounded evaluated metric collection")
    if any(not isinstance(x, EvaluationResult) for x in results):
        raise AudioContractError("All comparison inputs must be evaluated")
    return {
        "protocol": "sonumio-numeric-benchmark-summary-v1",
        "reports": [
            {
                "report_sha256": x.sha256,
                "split": x.split,
                "metric": x.metric,
                "cases": len(x.rows),
                "independent_groups": x.independent_groups,
                "mean_group_weighted_metric": x.mean_metric,
                "all_case_thresholds_pass": x.all_thresholds_pass,
            } for x in results
        ],
        "metric_direction_comparability_required": True,
        "not_cross_metric_rankable": True,
        "no_human_mos_inferred": True,
        "no_model_release_authorized": True,
    }
