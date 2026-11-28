"""Bridge geometry and policy tests run without optional old arrangement API."""
from __future__ import annotations

import pytest

from sonumio.midi_workstation.arrangement_bridge import (
    ArrangementBridgePolicy, ArrangementRoute, _score_tick, _score_pan,
)
from sonumio.midi_workstation.arrangement_bridge_cli import _route
from sonumio.midi_workstation.contracts import MidiWorkstationError


def test_all_eighth_grid_endpoints_roundtrip_at_480_ppqn():
    for target in range(0,513):
        assert _score_tick(target*60,480,512)==target


def test_all_eighth_grid_endpoints_roundtrip_at_nonstandard_ppqn():
    for target in range(0,256):
        assert _score_tick(target*125,1000,256)==target


@pytest.mark.parametrize("tick",[1,2,3,59,61,62,119,121,479,481,7199])
def test_fails_closed_for_any_off_grid_tick(tick):
    with pytest.raises(MidiWorkstationError,match="off arrangement eighth-beat grid"):
        _score_tick(tick,480,128)


@pytest.mark.parametrize("tick",[-1,7681,9999999])
def test_outside_score_grid_rejected(tick):
    with pytest.raises(MidiWorkstationError):
        _score_tick(tick,480,128)


def test_uneven_grid_ppqn_cannot_round_to_nearby_eighth():
    assert _score_tick(480,480,128)==8
    with pytest.raises(MidiWorkstationError):
        _score_tick(479,480,128)


@pytest.mark.parametrize("pan,expected",[(0,-1.),(1,-63/64),(63,-1/64),(64,0.),
                                         (65,1/63),(126,62/63),(127,1.)])
def test_seven_bit_pan_maps_exact_center_and_extremes(pan,expected):
    assert _score_pan(pan)==expected


def test_seven_bit_pan_is_monotone():
    values=[_score_pan(x) for x in range(128)]
    assert values==sorted(values)
    assert len(set(values))==128


@pytest.mark.parametrize("bad",["", "track with space", "../../etc/passwd", "x@y", "a:b"])
def test_invalid_track_ids_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        ArrangementRoute(bad,"lead")


@pytest.mark.parametrize("bad",["snare-bass", "unlisted", "guitar", "piano", 5, None])
def test_invalid_target_instruments_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        ArrangementRoute("melody",bad)


@pytest.mark.parametrize("bad",[-25,25,-100,100,1.5,True,False,"1",None])
def test_invalid_transposition_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        ArrangementRoute("melody","lead",bad)


@pytest.mark.parametrize("v",[-24,-12,-1,0,1,12,24])
def test_valid_transposition_bounds(v):
    assert ArrangementRoute("melody","lead",v).transpose==v


def test_policy_requires_unique_sorted_routes():
    policy=ArrangementBridgePolicy((ArrangementRoute("a","keys"),
                                    ArrangementRoute("b","lead")))
    assert len(policy.routes)==2
    with pytest.raises(MidiWorkstationError):
        ArrangementBridgePolicy((ArrangementRoute("b","lead"),
                                 ArrangementRoute("a","keys")))
    with pytest.raises(MidiWorkstationError):
        ArrangementBridgePolicy((ArrangementRoute("a","keys"),
                                 ArrangementRoute("a","lead")))


@pytest.mark.parametrize("value",[0,4097,1.1,True,None])
def test_policy_budget_validation(value):
    with pytest.raises(MidiWorkstationError):
        ArrangementBridgePolicy((ArrangementRoute("a","keys"),),maximum_events=value)


@pytest.mark.parametrize("value",[0,1,None,"true"])
def test_policy_flag_validation(value):
    with pytest.raises(MidiWorkstationError):
        ArrangementBridgePolicy((ArrangementRoute("a","keys"),),include_muted=value)
    with pytest.raises(MidiWorkstationError):
        ArrangementBridgePolicy((ArrangementRoute("a","keys"),),preserve_note_pan=value)


@pytest.mark.parametrize("argument,expected",[
    ("a:lead",("a","lead",0)),
    ("track-1:keys:0",("track-1","keys",0)),
    ("track_2:bass:-12",("track_2","bass",-12)),
    ("x:pluck:+12",("x","pluck",12)),
])
def test_route_parser(argument,expected):
    obj=_route(argument)
    assert (obj.source_track_id,obj.target_instrument,obj.transpose)==expected


@pytest.mark.parametrize("value",[
    "",":lead","a:","a:lead:","a:lead:1.5","a:lead:1:extra",
    "a:lead: ","a:lead:25","a:lead:-25","a:not-real",
])
def test_invalid_route_parser(value):
    with pytest.raises(MidiWorkstationError):
        _route(value)


def test_route_rejects_non_strings():
    for value in (None,123,True,(),{}):
        with pytest.raises(MidiWorkstationError):
            _route(value)
