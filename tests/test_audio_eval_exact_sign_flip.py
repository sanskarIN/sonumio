"""Exact small-group sign-flip tests and bounded large-group fallback."""
from __future__ import annotations

import pytest

from sonumio.audio_core.errors import AudioContractError
from research.audio_eval.statistics import (
    EXACT_SIGN_FLIP_MAX_GROUPS,
    group_effect_report, paired_sign_flip_test,
)


def test_four_group_exact_two_sided_and_directional_tail_probabilities():
    a, b, groups = [1.] * 4, [0.] * 4, ["a", "b", "c", "d"]
    two = paired_sign_flip_test(a, b, groups, permutations=100, seed=4)
    higher = paired_sign_flip_test(a, b, groups, direction="a-greater",
                                   permutations=100, seed=19)
    lower = paired_sign_flip_test(a, b, groups, direction="b-greater",
                                  permutations=100, seed=33)
    assert two.method == higher.method == lower.method == "exact"
    assert two.p_value == pytest.approx(2 / 16)
    assert higher.p_value == pytest.approx(1 / 16)
    assert lower.p_value == pytest.approx(1)
    assert two.permutations == higher.permutations == 16
    assert two.requested_permutations == 100
    assert two.groups == 4
    assert two.resampling_unit == "source_group"


def test_exact_is_seed_and_budget_independent_but_checks_seed():
    aa = [3., 5., -1., 2., 0.5]
    bb = [0.] * len(aa)
    gg = list("abcde")
    first = paired_sign_flip_test(aa, bb, gg, permutations=100, seed=0)
    other = paired_sign_flip_test(aa, bb, gg, permutations=20000, seed=123)
    assert first.p_value == other.p_value
    assert first.difference == other.difference
    assert first.permutations == other.permutations == 32
    assert first.method == other.method == "exact"
    with pytest.raises(AudioContractError, match="seed"):
        paired_sign_flip_test(aa, bb, gg, seed=True)


def test_repeated_recordings_do_not_create_fake_independent_sign_flips():
    one = paired_sign_flip_test([1, 1], [0, 0], ["a", "b"], permutations=100)
    repeated = paired_sign_flip_test([1] * 10, [0] * 10,
                                     ["a"] * 5 + ["b"] * 5, permutations=100)
    assert one.groups == repeated.groups == 2
    assert one.permutations == repeated.permutations == 4
    assert one.p_value == repeated.p_value == 0.5


def test_zero_and_tied_effects_have_valid_full_tail_mass():
    all_zero = paired_sign_flip_test([2, 2, 2], [2, 2, 2], list("abc"),
                                      permutations=100)
    assert all_zero.p_value == 1.0
    for direction in ("a-greater", "b-greater"):
        assert paired_sign_flip_test([2, 2, 2], [2, 2, 2], list("abc"),
                                     permutations=100, direction=direction).p_value == 1.0


def test_max_exact_groups_has_discrete_p_floor_and_all_assignments():
    n = EXACT_SIGN_FLIP_MAX_GROUPS
    result = paired_sign_flip_test([1.] * n, [0.] * n,
                                   [str(k) for k in range(n)],
                                   permutations=100, seed=10)
    assert result.groups == n
    assert result.permutations == 2 ** n
    assert result.method == "exact"
    assert result.p_value == pytest.approx(2 / (2 ** n))


def test_above_limit_uses_seeded_bounded_monte_carlo_with_plus_one():
    n = EXACT_SIGN_FLIP_MAX_GROUPS + 1
    a, b, groups = [1.] * n, [0.] * n, [str(k) for k in range(n)]
    one = paired_sign_flip_test(a, b, groups, permutations=100, seed=2)
    same = paired_sign_flip_test(a, b, groups, permutations=100, seed=2)
    assert one == same
    assert one.method == "monte_carlo"
    assert one.permutations == one.requested_permutations == 100
    assert 1 / 101 <= one.p_value <= 1
    with pytest.raises(AudioContractError, match="permutations"):
        paired_sign_flip_test(a, b, groups, permutations=99)


def test_effect_report_explains_the_method_and_does_not_imply_human_quality():
    report = group_effect_report(
        [0., 0., 0., 0.], [0.25] * 4, list("abcd"),
        bootstrap_iterations=100, permutations=100, seed=9,
    )
    assert report["protocol"].endswith("-v2")
    assert report["sign_flip"]["method"] == "exact"
    assert report["sign_flip"]["permutations"] == 16
    assert report["sign_flip"]["requested_permutations"] == 100
    assert "discrete resolution floor" in report["caution"]
    assert report["is_human_listening_quality_score"] is False
    assert report["multiple_comparisons_adjusted"] is False
