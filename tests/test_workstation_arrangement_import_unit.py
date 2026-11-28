"""Unit-scale contracts of reverse arrangement conversion without legacy API."""
from __future__ import annotations

import pytest

from sonumio.midi_workstation.arrangement_import import (
    LegacyInstrumentRoute, LegacyImportPolicy, _round_nearest_positive,
    _velocity, _pan,
)
from sonumio.midi_workstation.arrangement_import_cli import _route
from sonumio.midi_workstation.contracts import MidiWorkstationError


def test_velocity_exact_and_range():
    assert _velocity(1.)==(127,0.)
    assert _velocity(1/127.)==(1,0.)
    value, error=_velocity(.5)
    assert value==64
    assert 0<error<1/127
    assert _velocity(.001)[0]==1


def test_velocity_quantization_monotone():
    sequence=[_velocity(i/1000)[0] for i in range(1,1001)]
    assert sequence==sorted(sequence)
    assert min(sequence)==1 and max(sequence)==127


@pytest.mark.parametrize("v,expected",[(0.,64),(1.,127),(-1.,0),(.5,96),(-.5,32)])
def test_pan_endpoints_and_midpoints(v,expected):
    actual,error=_pan(v)
    assert actual==expected
    assert error<.01


def test_pan_never_discontinuous_when_crossing_center():
    vals=[_pan((i-1000)/1000)[0] for i in range(2001)]
    assert vals==sorted(vals)
    assert _pan(0.)==(64,0.)


@pytest.mark.parametrize("x,expect",[(0.,0),(.49,0),(.50,1),(.51,1),
                                     (1.49,1),(1.5,2),(2.5,3)])
def test_round_half_up_boundary(x,expect):
    assert _round_nearest_positive(x)==expect


@pytest.mark.parametrize("x",[-.01,float('nan'),float('inf'),-float('inf')])
def test_round_rejects_negative_and_nonfinite(x):
    with pytest.raises(MidiWorkstationError):
        _round_nearest_positive(x)


def test_reverse_routes_require_percussion_channel_9():
    assert LegacyInstrumentRoute("kick","kick-track",9).channel==9
    assert LegacyInstrumentRoute("lead","lead-track",0).channel==0
    with pytest.raises(MidiWorkstationError,match="percussion channel"):
        LegacyInstrumentRoute("kick","kick-track",0)
    with pytest.raises(MidiWorkstationError,match="percussion channel"):
        LegacyInstrumentRoute("keys","keys-track",9)


@pytest.mark.parametrize("value",["", "unknown", "rhodes", "piano", 8, None])
def test_reverse_unknown_instrument_rejected(value):
    with pytest.raises(MidiWorkstationError):
        LegacyInstrumentRoute(value,"track",0)


@pytest.mark.parametrize("value",["", "a b", "a:b", "../../etc", 0, True])
def test_reverse_invalid_track_ids_rejected(value):
    with pytest.raises(MidiWorkstationError):
        LegacyInstrumentRoute("lead",value,0)


@pytest.mark.parametrize("value",[True,False,16,-1,1.5,None])
def test_reverse_channel_validation(value):
    with pytest.raises(MidiWorkstationError):
        LegacyInstrumentRoute("lead","track",value)


@pytest.mark.parametrize("value",[True,False,128,-1,1.0,None])
def test_reverse_program_validation(value):
    with pytest.raises(MidiWorkstationError):
        LegacyInstrumentRoute("lead","track",0,program=value)


@pytest.mark.parametrize("value",[True,False,128,-1,1.0,None])
def test_reverse_bank_validation(value):
    with pytest.raises(MidiWorkstationError):
        LegacyInstrumentRoute("lead","track",0,bank_msb=value)
    with pytest.raises(MidiWorkstationError):
        LegacyInstrumentRoute("lead","track",0,bank_lsb=value)


def test_reverse_policy_routes_canonical_and_unique():
    a=LegacyInstrumentRoute("keys","keyed",0)
    b=LegacyInstrumentRoute("lead","solo",1)
    assert LegacyImportPolicy((a,b)).routes==(a,b)
    with pytest.raises(MidiWorkstationError,match="unique and sorted"):
        LegacyImportPolicy((b,a))
    with pytest.raises(MidiWorkstationError,match="unique and sorted"):
        LegacyImportPolicy((a,a))
    with pytest.raises(MidiWorkstationError,match="unique and sorted"):
        LegacyImportPolicy((a,LegacyInstrumentRoute("lead","keyed",1)))


@pytest.mark.parametrize("ppqn",[0,1,7,9,479,481,32768,True,1.0,None])
def test_reverse_requires_valid_divisible_ppqn(ppqn):
    with pytest.raises(MidiWorkstationError):
        LegacyImportPolicy((LegacyInstrumentRoute("lead","track",0),),ppqn=ppqn)


@pytest.mark.parametrize("ppqn",[8,24,96,480,960,32760])
def test_reverse_accepts_exact_ppqn_clock(ppqn):
    assert LegacyImportPolicy((LegacyInstrumentRoute("lead","track",0),),ppqn=ppqn).ppqn==ppqn


@pytest.mark.parametrize("flag",[0,1,None,"yes",(),[]])
def test_reverse_flags_must_be_bool(flag):
    route=(LegacyInstrumentRoute("lead","track",0),)
    for keyword in ("allow_velocity_quantization","allow_pan_quantization","allow_swing_loss"):
        with pytest.raises(MidiWorkstationError):
            LegacyImportPolicy(route,**{keyword:flag})


@pytest.mark.parametrize("max_events",[0,4097,1.0,True,None])
def test_reverse_bounded_event_budget(max_events):
    with pytest.raises(MidiWorkstationError):
        LegacyImportPolicy((LegacyInstrumentRoute("lead","track",0),),maximum_events=max_events)


@pytest.mark.parametrize("value,expected",[
    ("lead:melody:0:10",("lead","melody",0,10,0,0)),
    ("keys:keys-1:3:11:2",("keys","keys-1",3,11,2,0)),
    ("kick:drums:9:0:0:0",("kick","drums",9,0,0,0)),
])
def test_reverse_cli_route_parser(value,expected):
    actual=_route(value)
    assert (actual.instrument,actual.track_id,actual.channel,actual.program,
            actual.bank_msb,actual.bank_lsb)==expected


@pytest.mark.parametrize("value",[
    "", "lead", "lead:track:0", "lead:track:0:0:",
    "lead:track:0:0:0:0:0", "lead:track:a:0", "lead:track:0:1.2",
    "bad:track:0:0", "lead:track:9:0", "lead:track:0:128",
])
def test_reverse_cli_bad_routes(value):
    with pytest.raises(MidiWorkstationError):
        _route(value)
