"""Exact rational clock semantics and bounded materialized time partitions."""
from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

import sonumio.audio_core.timebase as timebase
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.timebase import (
    TimeGrid, frames_from_seconds, split_intervals,
)


@pytest.mark.parametrize("sample_rate", [8_000, 24_000, 44_100, 192_000])
def test_rational_values_near_half_frame_do_not_collapse_to_decimal_ties(sample_rate):
    half = Fraction(1, 2 * sample_rate)
    epsilon = Fraction(1, 10**40)
    assert frames_from_seconds(half - epsilon, sample_rate) == 0
    assert frames_from_seconds(half, sample_rate) == 0
    assert frames_from_seconds(half + epsilon, sample_rate) == 1

    one_and_half = Fraction(3, 2 * sample_rate)
    assert frames_from_seconds(one_and_half - epsilon, sample_rate) == 1
    assert frames_from_seconds(one_and_half, sample_rate) == 2
    assert frames_from_seconds(one_and_half + epsilon, sample_rate) == 2


@pytest.mark.parametrize("frames", [0, 1, 2, 5, 16, 257, 2048])
def test_exact_fraction_roundtrip_of_frame_positions(frames):
    rate = 44_100
    assert frames_from_seconds(Fraction(frames, rate), rate) == frames


def test_fraction_boundary_enforces_duration_and_zero_policy():
    with pytest.raises(AudioContractError, match="zero"):
        frames_from_seconds(Fraction(1, 10**12), 8000, allow_zero=False)
    with pytest.raises(AudioLimitError, match="bounded frame"):
        frames_from_seconds(Fraction(-1, 2), 8000)
    with pytest.raises(AudioLimitError, match="bounded frame"):
        frames_from_seconds(Fraction(12000001, 100000), 8000)
    assert frames_from_seconds(Fraction(120), 8000) == 960000


def test_fraction_precision_budget_guards_extremely_large_denominators():
    with pytest.raises(AudioLimitError, match="precision"):
        frames_from_seconds(Fraction(1, 2**5000), 8000)


def test_decimal_and_float_behavior_remains_documented():
    assert frames_from_seconds(Decimal("0.0000625"), 8000) == 0
    assert frames_from_seconds(Decimal("0.0001875"), 8000) == 2
    assert frames_from_seconds(0.0001875, 8000) == 2
    with pytest.raises(AudioContractError):
        frames_from_seconds(True, 8000)


def test_decimal_values_near_half_frame_are_also_rounded_exactly():
    half = Decimal("0.0000625")
    tiny = Decimal("1e-40")
    # Form the input value in a high-precision context: default Decimal
    # arithmetic would already erase tiny differences before our function.
    with localcontext() as ctx:
        ctx.prec = 80
        below, above = half - tiny, half + tiny
    assert frames_from_seconds(below, 8000) == 0
    assert frames_from_seconds(half, 8000) == 0
    assert frames_from_seconds(above, 8000) == 1
    # Constructing these values via a Decimal arithmetic operation would
    # itself round under the default context; use exact decimal literals.
    assert frames_from_seconds(
        Decimal("0.0000625000000000000000000000000000000001"), 8000
    ) == 1
    assert frames_from_seconds(
        Decimal("0.0000624999999999999999999999999999999999"), 8000
    ) == 0


def test_huge_decimal_precision_is_explicitly_bounded():
    with pytest.raises(AudioLimitError, match="precision"):
        frames_from_seconds(Decimal("1e-1200"), 8000)


def test_materialized_grid_preflights_large_count_before_allocation(monkeypatch):
    monkeypatch.setattr(timebase, "MAX_PARTITION_INTERVALS", 3)
    grid = TimeGrid(8000, 13)
    with pytest.raises(AudioLimitError, match="partition"):
        grid.partition(4)
    assert [(v.start, v.stop) for v in grid.partition(5)] == [
        (0, 5), (5, 10), (10, 13),
    ]
    assert grid.partition(13)[0].frames == 13
    for invalid in (0, -1, True, 0.5, "4"):
        with pytest.raises(AudioContractError, match="chunk size"):
            grid.partition(invalid)


def test_large_timegrid_can_still_be_iterated_lazily():
    # An extreme partition is rejected when materializing a tuple. Clients
    # can still stream chunks without keeping all their interval objects.
    grid = TimeGrid(192000, 192000 * 120)
    with pytest.raises(AudioLimitError, match="interval cap"):
        grid.partition(1)
    iterator = split_intervals(grid.frames, grid.sample_rate, 1)
    first = [next(iterator) for _ in range(4)]
    assert [(x.start, x.stop) for x in first] == [
        (0, 1), (1, 2), (2, 3), (3, 4),
    ]
