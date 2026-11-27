"""Analytic synthesis, reproducibility and strict bounds."""
import json
import numpy as np
import pytest
from dataclasses import replace

from sonumio.audio_core.contracts import make_buffer, AudioLayout
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.granular import (
    GranularSettings, grain_schedule, grain_window, synthesize_grains,
    _read_grain, _map_indices,
)


def test_schedule_is_seeded_and_independent():
    cfg=GranularSettings(2048,grain_frames=256,hop_frames=128,
        position_jitter=.2,pitch_jitter_semitones=3,reverse_probability=.4,
        amplitude_jitter=.2,time_jitter_frames=16,stereo_spread=.5,seed=17)
    a=grain_schedule(1024,cfg)
    assert a==grain_schedule(1024,cfg)
    assert a!=grain_schedule(1024,replace(cfg,seed=18))
    b=grain_schedule(1024,replace(cfg,amplitude_jitter=0))
    assert [(g.source_center,g.rate,g.pan,g.center_frame) for g in a]==[
        (g.source_center,g.rate,g.pan,g.center_frame) for g in b]
    assert all(0<=g.source_center<=1023 for g in a)
    assert all(abs(g.center_frame-g.index*128)<=16 for g in a)
    assert all(.4<=g.amplitude<=.6 for g in a)


@pytest.mark.parametrize("name",["hann","triangle","tukey"])
@pytest.mark.parametrize("frames",[8,17,256])
def test_grain_windows(name,frames):
    w=grain_window(frames,name)
    assert w.shape==(frames,)
    assert w[0]==w[-1]==0
    assert np.all((w>=0)&(w<=1))
    np.testing.assert_allclose(w,w[::-1],atol=1e-14)


@pytest.mark.parametrize("boundary",["wrap","reflect","zero"])
@pytest.mark.parametrize("rate",[-4,-1,.25,1,4])
def test_dc_interpolation_interior(boundary,rate):
    x=np.ones((2,256),dtype=np.float32)*.25
    p=np.linspace(100,150,23)
    a=_read_grain(x,p,rate,boundary)
    np.testing.assert_allclose(a,.25,atol=1e-13)


@pytest.mark.parametrize("boundary",["wrap","reflect"])
def test_dc_preservation_at_edges(boundary):
    x=np.ones((1,5),dtype=np.float32)*.3
    a=_read_grain(x,np.arange(-10,12)+.3,2.,boundary)
    np.testing.assert_allclose(a,float(x[0,0]),atol=1e-13)


def test_zero_extension_does_not_invent_dc():
    x=np.ones((1,16),dtype=np.float32)
    a=_read_grain(x,np.array([-100.,100.]),1.,"zero")
    np.testing.assert_array_equal(a,np.zeros((1,2)))


def test_integer_positions_reproduce_source_at_unity():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(2,32)).astype(np.float32)
    a=_read_grain(x,np.arange(32,dtype=float),1.,"wrap")
    np.testing.assert_allclose(a,x,atol=1e-13)


def test_high_rate_filter_suppresses_alias_energy():
    t=np.arange(4096)
    low=np.sin(2*np.pi*.03*t).astype(np.float32)[None,:]
    high=np.sin(2*np.pi*.35*t).astype(np.float32)[None,:]
    positions=np.arange(200,3800,4,dtype=float)
    a=_read_grain(low,positions,4,"wrap")
    b=_read_grain(high,positions,4,"wrap")
    assert np.sqrt(np.mean(a*a))>.6
    assert np.sqrt(np.mean(b*b))<.02


@pytest.mark.parametrize("boundary",["wrap","reflect","zero"])
def test_exact_geometry_reproducibility_and_source_immutability(boundary):
    rng=np.random.default_rng(10)
    x=rng.normal(size=(2,2048)).astype(np.float32)*.1
    layout=AudioLayout(2,"stereo",("custom L","custom R"))
    audio=make_buffer(x,24000,layout=layout)
    cfg=GranularSettings(3001,grain_frames=256,hop_frames=91,
                         boundary=boundary,pitch_jitter_semitones=4,seed=2)
    a=synthesize_grains(audio,cfg)
    b=synthesize_grains(audio,cfg)
    assert a.audio.frames==3001 and a.audio.spec.layout==layout
    np.testing.assert_array_equal(a.audio.samples,b.audio.samples)
    np.testing.assert_array_equal(audio.samples,x)
    assert a.report==b.report
    json.dumps(a.report.as_dict(),allow_nan=False)


def test_dense_overlap_and_silence():
    audio=make_buffer(np.ones(128,dtype=np.float32)*.25,8000)
    cfg=GranularSettings(300,grain_frames=64,hop_frames=8,amplitude=1,
                         position_jitter=0,peak_ceiling=None)
    result=synthesize_grains(audio,cfg)
    assert result.report.maximum_overlap_weight>1
    assert result.report.output_peak<=.25000001
    assert result.report.uncovered_frames==0
    silence=make_buffer(np.zeros(128,dtype=np.float32),8000)
    assert not np.any(synthesize_grains(silence,cfg).audio.samples)


def test_sparse_grains_have_explicit_gaps():
    audio=make_buffer(np.ones(100,dtype=np.float32),8000)
    cfg=GranularSettings(512,grain_frames=32,hop_frames=128)
    result=synthesize_grains(audio,cfg)
    assert result.report.uncovered_frames>0
    assert result.audio.samples[0,64]==0


def test_mono_to_stereo_and_linked_antiphase():
    x=np.sin(np.arange(512)*.2).astype(np.float32)
    cfg=GranularSettings(400,grain_frames=64,hop_frames=32,stereo_spread=.8)
    mono=synthesize_grains(make_buffer(x,8000),cfg)
    assert mono.audio.channels==2
    assert not np.array_equal(mono.audio.samples[0],mono.audio.samples[1])
    stereo=synthesize_grains(make_buffer(np.stack((x,-x)),8000),
                             replace(cfg,stereo_spread=0))
    np.testing.assert_array_equal(stereo.audio.samples[0],-stereo.audio.samples[1])


def test_single_frame_source_and_output():
    source=make_buffer(np.array([.25],dtype=np.float32),8000)
    cfg=GranularSettings(1,grain_frames=8,hop_frames=8,boundary="reflect")
    assert synthesize_grains(source,cfg).audio.frames==1


def test_explicit_peak_policy():
    source=make_buffer(np.ones(128,dtype=np.float32),8000)
    cfg=GranularSettings(200,grain_frames=64,hop_frames=16,amplitude=2,
                         peak_ceiling=.5)
    result=synthesize_grains(source,cfg)
    assert result.report.pre_limit_peak>1
    assert result.report.attenuation<1
    assert result.report.output_peak<=.5000001


@pytest.mark.parametrize("kwargs",[
    {"output_frames":0},{"grain_frames":7},{"hop_frames":True},
    {"position_jitter":float("nan")},{"pitch_semitones":24,"pitch_jitter_semitones":1},
    {"time_jitter_frames":129},{"seed":-1},{"reverse_probability":1.1},
    {"boundary":"invalid"},{"window":"invalid"},{"normalization":"invalid"},
    {"peak_ceiling":0},{"stereo_spread":True},
])
def test_settings_validation(kwargs):
    fields={"output_frames":1000}
    fields.update(kwargs)
    with pytest.raises(AudioContractError):
        GranularSettings(**fields)


def test_work_limits_and_multichannel_guard(monkeypatch):
    import sonumio.audio_core.granular as module
    audio=make_buffer(np.ones(128,dtype=np.float32),8000)
    monkeypatch.setattr(module,"MAX_GRAIN_PRODUCTS",1)
    with pytest.raises(AudioLimitError):
        synthesize_grains(audio,GranularSettings(100))
    with pytest.raises(AudioContractError):
        synthesize_grains(make_buffer(np.ones((3,128),dtype=np.float32),8000),
                          GranularSettings(100,stereo_spread=.5))
