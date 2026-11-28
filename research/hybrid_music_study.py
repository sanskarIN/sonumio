"""Replicated end-to-end comparison of local hybrid musical arrangements.

Training is restricted to the predeclared synthetic note grid. Evaluation
uses newly composed scores and excludes exact training control identities
from all learned and sine substitutions. The same score and mix serve all
arms. This is reconstruction research, not open-vocabulary generation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import statistics

from research.instrument_harmonic_model import initialize_model
from research.instrument_harmonic_study import StudyPlan, train_model
from research.instrument_note_data import build_corpus, training_schedule, INSTRUMENTS
from research.hybrid_music_render import (
    compare_hybrid_score, describe_routes, route_score, MAX_EVENT_SAMPLES,
)
from sonumio.arrangement import ArrangementPlan, compose_arrangement, arrangement_digest
from sonumio.music_mix import MixPlan

PROTOCOL = "sonumio-hybrid-arrangement-reconstruction-study-v1"
ARMS = ("fixed_sine", "initial", "trained", "wrong_instrument")
MAX_STUDY_RENDER_SAMPLES = 180_000_000


@dataclass(frozen=True, slots=True)
class ScoreCase:
    style: str
    mode: str
    root: int
    bpm: int
    seed: int
    swing: float = 0.

    def __post_init__(self):
        self.to_plan()

    def to_plan(self):
        return ArrangementPlan(style=self.style, mode=self.mode, root=self.root,
                               bpm=self.bpm, bars=4, seed=self.seed, swing=self.swing)


DEFAULT_CASES = (
    ScoreCase("calm", "major", 53, 160, 101),
    ScoreCase("cinematic", "minor", 65, 160, 109),
    ScoreCase("rhythmic", "major", 62, 152, 127, .1),
    ScoreCase("rhythmic", "minor", 57, 160, 131, .15),
)


@dataclass(frozen=True, slots=True)
class HybridStudyPlan:
    training: StudyPlan = StudyPlan()
    cases: tuple[ScoreCase, ...] = DEFAULT_CASES

    def __post_init__(self):
        if not isinstance(self.training, StudyPlan):
            raise ValueError("Expected bounded instrument training plan")
        if (type(self.cases) is not tuple or not 1 <= len(self.cases) <= 8
                or any(not isinstance(case, ScoreCase) for case in self.cases)
                or len(set(self.cases)) != len(self.cases)):
            raise ValueError("Invalid unique score case panel")
        # Every comparison renders both the reference and the candidate.
        upper = len(self.cases) * MAX_EVENT_SAMPLES * 2 * (
            1 + len(self.training.model_seeds) * (2 + len(INSTRUMENTS) - 1)
        )
        # Use actual bounded score event counts instead of the much looser
        # per-score ceiling when checking the prespecified total budget.
        if upper > MAX_STUDY_RENDER_SAMPLES:
            counts = sum(describe_routes(route_score(compose_arrangement(case.to_plan())))
                         ["event_samples"] for case in self.cases)
            actual_bound = counts * 2 * (1 + len(self.training.model_seeds) * (2 + len(INSTRUMENTS) - 1))
            if actual_bound > MAX_STUDY_RENDER_SAMPLES:
                raise ValueError("Hybrid reconstruction study exceeds render budget")


def prepare_panel(plan, excluded):
    if not isinstance(plan, HybridStudyPlan):
        raise ValueError("Expected HybridStudyPlan")
    rows = []
    known_scores = set()
    for case in plan.cases:
        score = compose_arrangement(case.to_plan())
        identity = arrangement_digest(score)
        if identity in known_scores:
            raise ValueError("Duplicate generated evaluation score")
        known_scores.add(identity)
        coverage = describe_routes(route_score(score, excluded=excluded))
        if coverage["eligible_events"] == 0:
            raise ValueError("Evaluation score has no eligible non-training notes")
        rows.append((case, score, coverage))
    return tuple(rows)


def _comparison(score, excluded, arm, model=None, offset=1):
    return compare_hybrid_score(score, excluded=excluded, arm=arm,
                                model=model, wrong_offset=offset, mix=MixPlan())


def _metric(report):
    value = report["mix_comparison"]["relative_wave_mse"]
    if value is None:
        raise ValueError("Cannot compare a silent reference mix")
    return value


def summarize_case(baseline, initial, trained, wrong):
    reports = (baseline, initial, trained, *wrong)
    if len(wrong) != len(INSTRUMENTS) - 1:
        raise ValueError("Wrong instrument control panel is incomplete")
    if len({r["reference_wav_sha256"] for r in reports}) != 1:
        raise ValueError("Hybrid comparison reference waveform changed")
    for field in ("score_sha256", "eligible_events", "eligible_samples", "event_samples"):
        if len({r["routing"][field] for r in reports}) != 1:
            raise ValueError("Hybrid comparison routing changed across arms")
    if tuple(r["routing"]["wrong_offset"] for r in wrong) != tuple(range(1, len(INSTRUMENTS))):
        raise ValueError("Wrong instrument controls are not exhaustive")
    if trained["routing"]["model_sha256"] != wrong[0]["routing"]["model_sha256"] or any(
        row["routing"]["model_sha256"] != trained["routing"]["model_sha256"] for row in wrong
    ):
        raise ValueError("Wrong-label controls used different trained weights")
    errors = {
        "fixed_sine": _metric(baseline),
        "initial": _metric(initial),
        "trained": _metric(trained),
        "wrong_instrument": statistics.fmean(_metric(report) for report in wrong),
    }
    return {
        "relative_mix_wave_mse": errors,
        "initial_minus_trained": errors["initial"] - errors["trained"],
        "sine_minus_trained": errors["fixed_sine"] - errors["trained"],
        "wrong_minus_matched": errors["wrong_instrument"] - errors["trained"],
    }


def run_study(plan=None):
    plan = HybridStudyPlan() if plan is None else plan
    if not isinstance(plan, HybridStudyPlan):
        raise ValueError("Expected HybridStudyPlan")
    corpus = build_corpus(plan.training.grid)
    training_rows = corpus.panel("train")
    excluded = frozenset(row.spec.identity for row in training_rows)
    panel = prepare_panel(plan, excluded)
    schedule, schedule_digest = training_schedule(training_rows, updates=plan.training.updates,
                                                 batch_size=plan.training.batch_size,
                                                 seed=plan.training.sampler_seed)
    baselines = tuple(_comparison(score, excluded, "fixed_sine") for _, score, _ in panel)
    replicates = []
    for seed in plan.training.model_seeds:
        model = initialize_model(plan.training.width, seed)
        initial = tuple(_comparison(score, excluded, "matched", model) for _, score, _ in panel)
        training = train_model(model, corpus, plan.training, schedule)
        results = []
        for index, (case, score, coverage) in enumerate(panel):
            trained = _comparison(score, excluded, "matched", model)
            wrong = tuple(_comparison(score, excluded, "wrong_instrument", model, offset)
                          for offset in range(1, len(INSTRUMENTS)))
            result = summarize_case(baselines[index], initial[index], trained, wrong)
            results.append({
                "case": asdict(case), "coverage": coverage,
                "score_sha256": arrangement_digest(score),
                "initial": initial[index], "trained": trained, "wrong_instrument": wrong,
                "summary": result,
            })
        means = {name: statistics.fmean(row["summary"][name] for row in results)
                 for name in ("initial_minus_trained", "sine_minus_trained", "wrong_minus_matched")}
        replicates.append({"model_seed": seed, "training": training, "cases": results,
                           "mean_paired_effect": means})
    means = {}
    for name in ("initial_minus_trained", "sine_minus_trained", "wrong_minus_matched"):
        values = [r["mean_paired_effect"][name] for r in replicates]
        means[name] = {"mean": statistics.fmean(values), "minimum": min(values),
                       "maximum": max(values), "positive_replicates": sum(v > 0 for v in values)}
    return {
        "protocol": PROTOCOL, "plan": asdict(plan),
        "source": corpus.report(), "training_schedule_sha256": schedule_digest,
        "exact_training_control_identities_excluded": len(excluded),
        "fixed_sine_baselines": baselines,
        "replicates": replicates, "replicate_summary": means,
        "score_cases": len(panel), "wrong_instrument_conditions": len(INSTRUMENTS) - 1,
        "generated_audio_written": False, "checkpoint_written": False,
        "model_weights_published": False, "perceptual_quality_established": False,
        "external_recordings_used": False,
        "evaluation_used_for_training": False,
        "scope": "reconstruction of first-party synthetic musical arrangements",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=48)
    args = parser.parse_args(argv)
    try:
        plan = HybridStudyPlan(training=StudyPlan(width=args.width, updates=args.updates,
                                                   batch_size=args.batch_size))
        result = run_study(plan)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Hybrid music reconstruction failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
