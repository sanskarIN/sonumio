"""Group-preserving acoustic metric confidence intervals and p-value tests."""
from __future__ import annotations

import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError
from research.audio_eval.statistics import (
    BootstrapInterval, PairedComparison, group_effect_report,
    grouped_bootstrap, holm_adjust, paired_sign_flip_test,
)


@pytest.fixture
def correlated_groups():
    groups = ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3
    before = [1, 1.1, 0.9, 2, 2.1, 1.9, 3, 3.1, 2.9, 4, 4.1, 3.9]
    after = [x + .5 for x in before]
    return before, after, groups


def test_bootstrap_resamples_groups_not_individual_waveforms(correlated_groups):
    before, after, groups = correlated_groups
    interval = grouped_bootstrap(
        (np.asarray(after) - np.asarray(before)).tolist(),
        groups, confidence=.95, iterations=1000, seed=42
    )
    assert isinstance(interval, BootstrapInterval)
    assert interval.samples == 12
    assert interval.groups == 4
    assert interval.estimate == pytest.approx(.5, abs=1e-8)
    assert interval.lower == pytest.approx(.5, abs=1e-8)
    assert interval.upper == pytest.approx(.5, abs=1e-8)
    assert interval.resampling_unit == "source_group"


def test_bootstrap_determinism_and_nonparametric_interval(correlated_groups):
    before, after, groups = correlated_groups
    values = (np.asarray(after) - np.asarray(before)).tolist()
    a = grouped_bootstrap(values, groups, iterations=1000, seed=1)
    b = grouped_bootstrap(values, groups, iterations=1000, seed=1)
    assert a == b
    assert a.lower <= a.estimate <= a.upper
    assert a.as_dict()["seed"] == 1
    assert a.as_dict()["resampling_unit"] == "source_group"


def test_paired_sign_flip_reports_finite_nonzero_p(correlated_groups):
    before, after, groups = correlated_groups
    row = paired_sign_flip_test(
        after, before, groups, permutations=1000, seed=15
    )
    assert isinstance(row, PairedComparison)
    assert row.difference == pytest.approx(.5, abs=1e-8)
    assert 0 < row.p_value <= 1
    assert row.groups == 4
    assert row.direction == "two-sided"
    assert row.report_protocol.endswith("-v1")


def test_directional_tests_and_grouped_summary(correlated_groups):
    before, after, groups = correlated_groups
    greater = paired_sign_flip_test(after, before, groups, permutations=1000,
                                    direction="a-greater", seed=44)
    lesser = paired_sign_flip_test(after, before, groups, permutations=1000,
                                   direction="b-greater", seed=44)
    assert greater.p_value < lesser.p_value
    summary = group_effect_report(before, after, groups,
                                  bootstrap_iterations=500,
                                  permutations=500, seed=15)
    assert summary["direction"] == "candidate-minus-reference"
    assert summary["group_pseudoreplication_guard"] is True
    assert summary["independent_groups"] == 4
    assert summary["measurements"] == 12
    assert summary["is_human_listening_quality_score"] is False
    assert summary["multiple_comparisons_adjusted"] is False


def test_holm_adjustment_is_monotonic_with_order_and_bounded():
    values = [.03, .001, .04, .4]
    adjusted = holm_adjust(values)
    assert len(adjusted) == len(values)
    assert adjusted[1] <= adjusted[0]
    assert adjusted[0] <= adjusted[2]
    assert adjusted[2] <= adjusted[3]
    assert all(0 <= x <= 1 for x in adjusted)
    assert all(a >= b for a, b in zip(adjusted, values))


def test_invalid_group_metadata_and_fake_independent_sample_are_rejected():
    with pytest.raises(AudioContractError):
        grouped_bootstrap([1, 2], ["one", "one"], iterations=100)
    with pytest.raises(AudioContractError):
        grouped_bootstrap([1, 2], ["one"], iterations=100)
    with pytest.raises(AudioContractError):
        grouped_bootstrap([1, float("nan")], ["a", "b"], iterations=100)
    with pytest.raises(AudioContractError):
        grouped_bootstrap([1, 2], ["a", "b"], iterations=1)
    with pytest.raises(AudioContractError):
        paired_sign_flip_test([1, 2], [1], ["a", "b"])
    with pytest.raises(AudioContractError):
        holm_adjust([1.5, .2])
    with pytest.raises(AudioContractError):
        paired_sign_flip_test([1, 2], [0, 0], ["a", "b"],
                              direction="unregistered")


def test_group_effect_is_not_mistaken_for_perceptual_rating(correlated_groups):
    before, after, groups = correlated_groups
    summary = group_effect_report(before, after, groups,
                                  bootstrap_iterations=100,
                                  permutations=100, seed=8)
    assert summary["external_dataset_rights_verified"] is False
    assert "Intervals assume" in summary["caution"]
    assert "not" in summary["caution"].lower()
