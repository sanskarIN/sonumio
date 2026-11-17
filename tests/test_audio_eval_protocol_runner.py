"""Preregistered acoustic evaluation plans, grouped scores and fair A/B tests."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError
from research.audio_eval.protocol import (
    EvaluationPlan, MetricRule, metric_catalog, preregistration_report,
)
from research.audio_eval.runner import (
    EvaluationResult, EvaluationRow, run_evaluation,
    compare_engine_reports, numeric_benchmark_summary,
)
from research.audio_eval.reference import (
    AudioReference, EvaluationPair, EvaluationSet, RightsDeclaration,
)


@pytest.fixture
def rights():
    return RightsDeclaration(
        "first-party-synthetic", "first-party-test-fixture",
        True, False, True,
    )


@pytest.fixture
def dataset(rights):
    cases = []
    for i in range(4):
        t = np.arange(2048, dtype=np.float64) / 24000
        wave = (.2 * np.sin(2 * np.pi * (300 + 80*i) * t)).astype(np.float32)
        audio = make_buffer(wave, 24000)
        corrupted = audio.copy_with(audio.samples * (.99 - .04 * i))
        ref = AudioReference(
            f"asset_{i}", audio, f"group_{i}", "test",
            "nonverbal", f"Original {300+80*i} Hz sine", rights,
        )
        cases.append(EvaluationPair(
            f"case_{i}", ref, corrupted,
            "experimental-local-dsp",
        ))
    return EvaluationSet(tuple(cases))


@pytest.fixture
def plan():
    return EvaluationPlan(
        plan_id="study_synthetic_v1",
        split="test",
        metric=MetricRule("sample_rmse", accepted_boundary=.05),
        candidate_engine="experimental-local-dsp",
        fft_sizes=(128, 256), bootstrap_iterations=100,
        minimum_groups=2, confidence=.95, seed=123,
    )


def test_catalog_contains_explicit_directions_no_neural_backend():
    catalog = metric_catalog()
    assert len(catalog) >= 8
    by_name = {m["name"]: m for m in catalog}
    assert by_name["sample_rmse"]["direction"] == "lower"
    assert by_name["si_sdr_db"]["direction"] == "higher"
    assert all(x["trained_embedding_model_required"] is False for x in catalog)
    assert all(x["requires_human_interpretation"] is True for x in catalog)


def test_metric_rule_thresholds_and_naturalized_improvement():
    low = MetricRule("sample_rmse", accepted_boundary=.1)
    high = MetricRule("normalized_correlation", accepted_boundary=.9)
    assert low.meets_threshold(.05) is True
    assert low.meets_threshold(.2) is False
    assert high.meets_threshold(.95) is True
    assert high.meets_threshold(.7) is False
    assert MetricRule("sample_rmse").meets_threshold(.5) is None
    assert low.naturalized_difference(.05, .08) == pytest.approx(.03)
    assert high.naturalized_difference(.95, .9) == pytest.approx(.05)


def test_metric_rule_rejects_unknown_metric_nonfinite_and_negative_weight():
    with pytest.raises(AudioContractError):
        MetricRule("internet_rating")
    with pytest.raises(AudioContractError):
        MetricRule("sample_rmse", accepted_boundary=float("nan"))
    with pytest.raises(AudioContractError):
        MetricRule("sample_rmse", weight=0)
    with pytest.raises(AudioContractError):
        MetricRule("sample_rmse").meets_threshold(float("nan"))


def test_plan_is_canonical_and_rejects_tampered_metric_direction(plan):
    raw = plan.as_dict()
    assert raw["protocol"] == "sonumio-prespecified-acoustic-evaluation-v1"
    assert raw["automatically_licensed"] is False
    assert raw["publication_authorized"] is False
    restored = EvaluationPlan.from_dict(json.loads(json.dumps(raw)))
    assert restored.sha256 == plan.sha256
    altered = json.loads(json.dumps(raw))
    altered["metric"]["direction"] = "higher"
    with pytest.raises(AudioContractError):
        EvaluationPlan.from_dict(altered)
    altered = json.loads(json.dumps(raw))
    altered["publication_authorized"] = True
    with pytest.raises(AudioContractError):
        EvaluationPlan.from_dict(altered)


def test_evaluation_plan_rejects_training_as_holdout_and_weak_rights(plan):
    with pytest.raises(AudioContractError):
        replace(plan, split="train")
    with pytest.raises(AudioContractError):
        replace(plan, minimum_groups=1)
    with pytest.raises(AudioContractError):
        replace(plan, source_rights_review_required=False)
    with pytest.raises(AudioContractError):
        replace(plan, human_listening_required_before_release=False)
    with pytest.raises(AudioContractError):
        replace(plan, seed=-1)


def test_plan_reports_reproducible_provenance_without_release_claim(plan):
    info = preregistration_report(plan)
    assert info["plan_sha256"] == plan.sha256
    assert info["metric"] == "sample_rmse"
    assert info["direction"] == "lower"
    assert info["human_rights_review_still_required"] is True
    assert info["human_listening_before_model_publication_required"] is True
    assert info["trained_audio_model_available"] is False
    assert info["audio_files_exported"] is False


def test_evaluation_executes_prespecified_source_group_scoped_stats(dataset, plan):
    report = run_evaluation(dataset, plan)
    assert isinstance(report, EvaluationResult)
    assert report.split == "test"
    assert report.metric == "sample_rmse"
    assert report.independent_groups == 4
    assert len(report.rows) == 4
    assert report.all_thresholds_pass is True
    assert report.mean_metric > 0
    assert report.mean_metric < .05
    assert len(report.sha256) == 64
    assert report.group_bootstrap["groups"] == 4
    assert report.group_bootstrap["resampling_unit"] == "source_group"
    assert report.as_dict()["perceptual_audio_quality_assessed"] is False
    assert report.as_dict()["automatic_model_release_approval"] is False
    assert report.as_dict()["audio_files_saved"] is False


def test_same_dataset_and_seed_have_bitwise_equal_reports(dataset, plan):
    a = run_evaluation(dataset, plan)
    b = run_evaluation(dataset, plan)
    assert a.sha256 == b.sha256
    assert a.as_dict() == b.as_dict()
    assert a.dataset_sha256 == dataset.fingerprint
    assert [x.case_id for x in a.rows] == [
        "case_0", "case_1", "case_2", "case_3"
    ]


def test_predeclared_metric_direction_changes_gate_interpretation(dataset, plan):
    strict = replace(plan, metric=MetricRule(
        "sample_rmse", accepted_boundary=0
    ))
    report = run_evaluation(dataset, strict)
    assert report.all_thresholds_pass is False
    assert report.mean_metric > 0
    ungated = run_evaluation(dataset, replace(
        plan, metric=MetricRule("sample_rmse")
    ))
    assert ungated.all_thresholds_pass is None


def test_insufficient_groups_and_engine_selection_fail_closed(dataset, plan):
    with pytest.raises(AudioContractError, match="independent"):
        run_evaluation(dataset, replace(plan, minimum_groups=5))
    with pytest.raises(AudioContractError, match="No recordings"):
        run_evaluation(dataset, replace(plan, candidate_engine="other"))
    with pytest.raises(AudioContractError, match="No recordings"):
        run_evaluation(dataset, replace(plan, split="validation"))


def test_evaluation_refuses_non_strict_rights_even_for_synthetic(dataset, plan):
    unsafe = EvaluationSet(dataset.pairs, strict_rights=False)
    with pytest.raises(AudioContractError, match="rights"):
        run_evaluation(unsafe, plan)


def test_naturalized_paired_comparison_requires_same_reference_cases(dataset, plan):
    a = run_evaluation(dataset, plan)
    # Another candidate engine with the exact same references but
    # explicitly different PCM samples and no forged model credentials.
    new_pairs = tuple(
        EvaluationPair(
            x.case_id, x.reference,
            x.reference.buffer.copy_with(x.reference.buffer.samples * .999),
            "another-local-dsp",
        )
        for x in dataset.pairs
    )
    b_plan = replace(plan, candidate_engine="another-local-dsp")
    b = run_evaluation(EvaluationSet(new_pairs), b_plan)
    result = compare_engine_reports(
        a, b, metric_rule=plan.metric,
        permutations=100, bootstrap_iterations=100, seed=4,
    )
    assert result["source_cases_identical"] is True
    assert result["reference_bytes_identical"] is True
    assert result["case_count"] == 4
    assert result["independent_groups"] == 4
    assert result["mean_naturalized_improvement"] > 0
    assert result["human_listening_mandatory_for_release"] is True
    assert result["automatic_audio_quality_winner_declared"] is False
    assert result["multiple_comparison_family_declared"] is False


def test_paired_comparison_rejects_missing_cases_and_metric_mismatch(dataset, plan):
    a = run_evaluation(dataset, plan)
    b = run_evaluation(dataset, replace(
        plan, metric=MetricRule("normalized_correlation")
    ))
    with pytest.raises(AudioContractError):
        compare_engine_reports(a, b, metric_rule=plan.metric)
    # A subset contains only 3 groups, still a valid independent analysis,
    # but it must not be compared against the 4-case full source set.
    subset = EvaluationSet(dataset.pairs[:-1])
    c = run_evaluation(subset, plan)
    with pytest.raises(AudioContractError, match="identical case"):
        compare_engine_reports(a, c, metric_rule=plan.metric)


def test_benchmark_summary_never_ranks_incomparable_metrics(dataset, plan):
    a = run_evaluation(dataset, plan)
    b = run_evaluation(dataset, replace(
        plan, metric=MetricRule("si_sdr_db")
    ))
    overview = numeric_benchmark_summary([a, b])
    assert len(overview["reports"]) == 2
    assert overview["not_cross_metric_rankable"] is True
    assert overview["no_human_mos_inferred"] is True
    assert overview["no_model_release_authorized"] is True
    with pytest.raises(AudioContractError):
        numeric_benchmark_summary([])


def test_result_rejects_duplicate_cases_and_unbound_metrics():
    row = EvaluationRow(
        "case", "asset", "group", "test",
        "engine", "sample_rmse", 0.1, True, "a" * 64, "b" * 64,
    )
    with pytest.raises(AudioContractError):
        EvaluationResult("a"*64, "b"*64, "test",
                         (row, row), "sample_rmse", 2, {})
    with pytest.raises(AudioContractError):
        EvaluationResult("a"*64, "b"*64, "test",
                         (row,), "si_sdr_db", 2, {})
    with pytest.raises(AudioContractError):
        EvaluationRow("a", "b", "g", "test", "e", "sample_rmse",
                      float("nan"), True, "a"*64, "b"*64)
