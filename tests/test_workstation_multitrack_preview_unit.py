"""Multitrack patch and geometry contracts without optional audio-core install."""
from __future__ import annotations

from dataclasses import replace
import pytest

from sonumio.midi_workstation.contracts import MidiWorkstationError,MidiLimitError
from sonumio.midi_workstation.multitrack_preview import (
    TrackPatch, MultiTrackPreviewSettings, _peak_polyphony,
)
from sonumio.midi_workstation.multitrack_preview_cli import _patch


def p(name="a",**kwargs):
    return TrackPatch(name,**kwargs)


def settings(*patches,**kwargs):
    return MultiTrackPreviewSettings(tuple(patches or (p(),)),**kwargs)


@pytest.mark.parametrize("waveform",["sine","saw","square","triangle"])
def test_all_original_waveforms_supported(waveform):
    assert p(waveform=waveform).waveform==waveform


@pytest.mark.parametrize("bad",[None,1,True,"fm","sample","pluck","noise",""])
def test_unknown_waveforms_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        p(waveform=bad)


@pytest.mark.parametrize("bad",["", "a b", "a:b", "../x", 0,None,True])
def test_invalid_stable_track_ids_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        p(bad)


@pytest.mark.parametrize("bad",[float('nan'),float('inf'),-1.,1.01,None,True,"1",0j])
def test_gain_must_be_finite_and_bounded(bad):
    with pytest.raises(MidiWorkstationError):
        p(gain=bad)


@pytest.mark.parametrize("size",[0,1,2,255,257,300,1023,8193,1.0,None,True])
def test_table_size_is_power_of_two_in_bounds(size):
    with pytest.raises(MidiWorkstationError):
        p(table_size=size)


@pytest.mark.parametrize("size",[256,512,1024,2048,4096,8192])
def test_table_size_power_of_two_supported(size):
    assert p(table_size=size).table_size==size


@pytest.mark.parametrize("bad",[0,256,257,-1,1.,True,None])
def test_harmonic_work_budget_enforced(bad):
    with pytest.raises(MidiWorkstationError):
        p(table_size=256,max_harmonics=bad)


@pytest.mark.parametrize("field",["attack_frames","decay_frames","release_frames"])
@pytest.mark.parametrize("value",[-1,1000001,1.1,True,None])
def test_envelope_frame_fields_reject_invalid_values(field,value):
    with pytest.raises(MidiWorkstationError):
        p(**{field:value})


@pytest.mark.parametrize("value",[-.01,1.01,True,None,float('nan')])
def test_sustain_must_be_bounded_finite(value):
    with pytest.raises(MidiWorkstationError):
        p(sustain_level=value)


def test_patches_must_be_unique_and_sorted():
    assert [x.track_id for x in settings(p("a"),p("b")).patches]==["a","b"]
    with pytest.raises(MidiWorkstationError):
        settings(p("b"),p("a"))
    with pytest.raises(MidiWorkstationError):
        settings(p("a"),p("a"))


@pytest.mark.parametrize("value",[0,3,8,192001,8000.0,True,None])
def test_sample_rate_geometry_bounded(value):
    with pytest.raises(MidiWorkstationError):
        settings(p(),sample_rate=value)


@pytest.mark.parametrize("value",[0,3,-1,2.0,True,None])
def test_channel_geometry_bounded(value):
    with pytest.raises(MidiWorkstationError):
        settings(p(),channels=value)


@pytest.mark.parametrize("value",[0,1,None,"yes"])
def test_behavior_flags_must_be_boolean(value):
    with pytest.raises(MidiWorkstationError):
        settings(p(),include_muted=value)
    with pytest.raises(MidiWorkstationError):
        settings(p(),allow_skip_percussion=value)


@pytest.mark.parametrize("value",["", "linear", "limiter", 1, None])
def test_normalization_explicit_modes_only(value):
    with pytest.raises(MidiWorkstationError):
        settings(p(),normalization=value)


@pytest.mark.parametrize("value",[0.,1.1,-1.,None,float('nan'),True])
def test_peak_level_must_be_valid(value):
    with pytest.raises(MidiWorkstationError):
        settings(p(),maximum_peak=value)


@pytest.mark.parametrize("value",[0,2049,None,True,1.0])
def test_maximum_notes_resource_budget(value):
    with pytest.raises(MidiWorkstationError):
        settings(p(),maximum_notes=value)


@pytest.mark.parametrize("value",[0,14000001,None,True,1.0])
def test_maximum_work_resource_budget(value):
    with pytest.raises(MidiWorkstationError):
        settings(p(),maximum_work_samples=value)


@pytest.mark.parametrize("raw,expected",[
    ("melody:sine:0.25",("melody","sine",.25)),
    ("kick:square:0",("kick","square",0.)),
    ("pad:triangle:1",("pad","triangle",1.)),
])
def test_cli_patch_parser(raw,expected):
    output=_patch(raw)
    assert (output.track_id,output.waveform,output.gain)==expected


@pytest.mark.parametrize("raw",[
    "", "id", "id:sine", "id:sine:", "id:sine:a", "id:sine:nan",
    "id:sine:inf", "id:sine:2", "id:sine:-.1", "id:sine:.2:extra",
    "id:unknown:.1", "bad id:sine:.1", 3, None,
])
def test_cli_invalid_patch_syntax(raw):
    with pytest.raises(MidiWorkstationError):
        _patch(raw)


class _Note:
    def __init__(self,start,stop,track="a"):
        self.start_frame=start
        self.stop_frame=stop
        self.track_id=track


class _Schedule:
    def __init__(self,notes):
        self.notes=notes


def test_peak_polyphony_counts_tail_and_adjacent_release_order():
    schedule=_Schedule([_Note(0,100),_Note(100,200)])
    assert _peak_polyphony(schedule,{"a":p(release_frames=0)})==1
    assert _peak_polyphony(schedule,{"a":p(release_frames=1)})==2


def test_peak_polyphony_rejects_more_than_sixty_four_notes():
    schedule=_Schedule([_Note(0,100) for _ in range(65)])
    with pytest.raises(MidiLimitError,match="64 simultaneous"):
        _peak_polyphony(schedule,{"a":p()})


def test_peak_polyphony_ignores_skipped_percussion_tracks():
    schedule=_Schedule([_Note(0,100,"drum"),_Note(0,100,"a")])
    assert _peak_polyphony(schedule,{"a":p()})==1


def test_many_sequential_notes_have_one_polyphonic_voice():
    schedule=_Schedule([_Note(i*100,(i+1)*100) for i in range(1000)])
    assert _peak_polyphony(schedule,{"a":p(release_frames=0)})==1
