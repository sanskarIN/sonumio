"""Instrument and stereo stem properties with analytic controls."""
from dataclasses import replace
import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan, Event, TRACKS, compose_arrangement
from sonumio.music_instruments import render_event, smooth_edges
from sonumio.music_mix import MixPlan, render_stems, mix_stems, room_response
from sonumio.music_analysis import analyze_audio
from sonumio.errors import GenerationError


@pytest.fixture(scope="module")
def stems():
    return render_stems(compose_arrangement(ArrangementPlan(bars=4,bpm=160,seed=29)))


@pytest.mark.parametrize("track",TRACKS)
@pytest.mark.parametrize("pitch",[36,60,96])
def test_instrument_bounded_repeatable(track,pitch):
    event=Event(track,pitch,0,4,.7,0)
    a=render_event(event,4800,17)
    b=render_event(event,4800,17)
    assert a.dtype==np.float64 and a.shape==(4800,)
    assert np.array_equal(a,b)
    assert np.isfinite(a).all() and np.max(np.abs(a))<=1
    assert np.sqrt(np.mean(a*a))>.0001
    assert a[0]==0 and abs(a[-1])<1e-12


@pytest.mark.parametrize("track",["kick","snare","hat"])
def test_percussion_noise_stream_is_event_local(track):
    event=Event(track,42,8,4,.7,0)
    a=render_event(event,4800,17)
    render_event(replace(event,tick=16),4800,17)
    assert np.array_equal(a,render_event(event,4800,17))
    assert not np.array_equal(a,render_event(event,4800,18))
    assert not np.array_equal(a,render_event(replace(event,tick=16),4800,17))


def test_pitched_fundamental_frequency():
    event=Event("bass",69,0,8,.7,0)
    samples=render_event(event,24000,0)
    magnitude=np.abs(np.fft.rfft(samples*np.hanning(len(samples))))
    assert abs(np.argmax(magnitude)-440)<=1


@pytest.mark.parametrize("length",[2,3,4,100,1000])
def test_edge_envelopes_are_finite_and_bounded(length):
    a=smooth_edges(length,240,1200)
    assert a.shape==(length,)
    assert a[0]==0 and abs(a[-1])<1e-12
    assert np.all((a>=0)&(a<=1))


def test_stems_are_independent_and_read_only(stems):
    assert tuple(stems)==TRACKS
    assert all(s.dtype==np.float32 and s.shape[0]==2 for s in stems.values())
    assert all(not s.flags.writeable for s in stems.values())
    assert all(np.isfinite(s).all() for s in stems.values())
    assert np.max(np.abs(stems["lead"]))>0
    assert not np.array_equal(stems["lead"],stems["pad"])


def test_solo_matches_independent_linear_mix(stems):
    plan=MixPlan(gains=(.1,)*8,solo=("lead",),room=0,peak_ceiling=.99)
    a,report=mix_stems(stems,plan)
    expected=(stems["lead"].astype(np.float64)*.1*smooth_edges(a.shape[1],240,1200)).astype(np.float32)
    assert report["master_attenuation"]==1
    np.testing.assert_array_equal(a,expected)
    assert report["tracks"]["lead"]["active"]
    assert not report["tracks"]["pad"]["active"]


def test_mute_all_is_exact_silence(stems):
    audio,report=mix_stems(stems,MixPlan(mute=TRACKS))
    assert not np.any(audio)
    assert report["master_attenuation"]==1


def test_mix_does_not_mutate_stems_and_master_only_attenuates(stems):
    before={k:s.copy() for k,s in stems.items()}
    a,report=mix_stems(stems,MixPlan(gains=(2.,)*8,peak_ceiling=.2))
    assert np.max(np.abs(a))<=.20000002
    assert 0<report["master_attenuation"]<=1
    assert all(np.array_equal(s,before[k]) for k,s in stems.items())
    assert report["metrics"]["clipping_fraction"]==0


def test_room_response_has_only_prescribed_finite_reflections():
    dry=np.zeros((2,10000),dtype=np.float64)
    dry[0,0]=1
    wet=room_response(dry,1.)
    assert wet[0,0]==1
    assert wet[0,round(.029*24000)]==.23
    assert wet[1,round(.043*24000)]==.19
    assert np.count_nonzero(wet)==7
    assert np.array_equal(room_response(dry,0),dry)
    assert np.count_nonzero(dry)==1


def test_stereo_signal_oracles():
    t=np.arange(24000)/24000
    x=(.3*np.sin(2*np.pi*440*t)).astype(np.float32)
    same=analyze_audio(np.stack((x,x)))
    opposite=analyze_audio(np.stack((x,-x)))
    zero=analyze_audio(np.zeros((2,24001),dtype=np.float32))
    assert same["centered_stereo_correlation"]==pytest.approx(1)
    assert same["mono_mix_rms_retention"]==pytest.approx(1)
    assert opposite["centered_stereo_correlation"]==pytest.approx(-1)
    assert opposite["mono_mix_rms_retention"]==0
    assert zero["centered_stereo_correlation"] is None
    assert zero["mono_mix_rms_retention"] is None
    assert zero["quiet_half_second_windows"]==3
    assert sum(w["frames"] for w in zero["timeline"])==24001


@pytest.mark.parametrize("kwargs",[
    {"gains":(1.,)},{"gains":(float("nan"),)*8},{"solo":("lead","lead")},
    {"mute":("invalid",)},{"mute":("lead",),"solo":("lead",)},
    {"room":1.1},{"peak_ceiling":0},
])
def test_invalid_mix_plan(kwargs):
    with pytest.raises(GenerationError):
        MixPlan(**kwargs)
