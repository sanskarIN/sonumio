"""Nature sound primitives, stereo scene controls and exact clocks."""
from dataclasses import replace
import numpy as np
import pytest

from sonumio.nature_synthesis import NatureVoice,render_voice,KINDS,SAMPLE_RATE,_noise
from sonumio.nature_scene import (
    AutomationPoint,NatureLayer,NatureScene,preset,render_layer,render_scene,
    generate_scene_audio,scene_digest,_curve,
)
from sonumio.nature_analysis import describe,compare
from sonumio.errors import GenerationError


@pytest.mark.parametrize("kind",KINDS)
def test_original_nature_signal_contract(kind):
    voice=NatureVoice(kind,.65,.5,.5,17)
    a=render_voice(voice,2*SAMPLE_RATE)
    b=render_voice(voice,2*SAMPLE_RATE)
    other=render_voice(replace(voice,seed=29),2*SAMPLE_RATE)
    assert a.shape==(2*SAMPLE_RATE,) and a.dtype==np.float32
    assert np.array_equal(a,b)
    assert not np.array_equal(a,other)
    assert np.isfinite(a).all() and np.max(np.abs(a))<=.800001
    assert np.sqrt(np.mean(a*a))>1e-5
    assert a[0]==0 and abs(a[-1])<1e-10
    assert not np.any(render_voice(replace(voice,intensity=0),2*SAMPLE_RATE))
    assert describe(a)["overrange_fraction"]==0


@pytest.mark.parametrize("kind",KINDS)
def test_shortest_supported_clip(kind):
    audio=render_voice(NatureVoice(kind,seed=41),SAMPLE_RATE//10)
    assert audio.shape==(SAMPLE_RATE//10,)
    assert np.isfinite(audio).all()


@pytest.mark.parametrize("settings",[
    {"kind":None},{"kind":"unknown"},{"intensity":-1},{"intensity":True},
    {"texture":float("nan")},{"motion":1.01},{"seed":2**32},
])
def test_voice_validation(settings):
    values={"kind":"rain"}
    values.update(settings)
    with pytest.raises(GenerationError):
        NatureVoice(**values)


def test_band_shaped_noise_fft_oracle():
    frames=24000
    audio=_noise(np.random.default_rng(17),frames,500,2500)
    spectrum=np.abs(np.fft.rfft(audio))
    frequency=np.fft.rfftfreq(frames,1/SAMPLE_RATE)
    assert abs(np.mean(audio))<1e-12
    assert np.sqrt(np.mean(audio*audio))==pytest.approx(1.)
    assert spectrum[(frequency<500)|(frequency>2500)].max()<1e-8


def test_rain_texture_shifts_spectral_center():
    voice=NatureVoice("rain",.65,.1,.5,17)
    a=render_voice(voice,2*SAMPLE_RATE)
    b=render_voice(replace(voice,texture=.9),2*SAMPLE_RATE)
    result=compare(a,b)
    assert result["centroid_delta_hz"]>0


def test_thunder_has_delayed_attack():
    voice=NatureVoice("thunder",.7,.5,.5,17)
    y=render_voice(voice,4*SAMPLE_RATE)
    assert not np.any(y[:int(.22*4*SAMPLE_RATE)-1])
    assert np.any(y[int(1.2*SAMPLE_RATE):])


def test_automation_interpolates_absolute_frame_positions():
    points=(AutomationPoint(0,-12.),AutomationPoint(3,0.),AutomationPoint(6,-6.))
    np.testing.assert_array_equal(_curve(points,8),[-12,-8,-4,0,-2,-4,-6,-6])


def test_width_zero_preserves_equal_power_mono_center_and_hard_pan():
    voice=NatureVoice("wind",.6,seed=17)
    layer=NatureLayer(voice,0,SAMPLE_RATE,width=0,fade_in=0,fade_out=0,
                      gain=(AutomationPoint(0,0.),))
    mono=render_voice(voice,SAMPLE_RATE).astype(np.float64)
    center=render_layer(layer)
    np.testing.assert_allclose(center[0],mono/np.sqrt(2),atol=1e-12)
    np.testing.assert_allclose(center[1],mono/np.sqrt(2),atol=1e-12)
    left=render_layer(replace(layer,pan=(AutomationPoint(0,-1.),)))
    right=render_layer(replace(layer,pan=(AutomationPoint(0,1.),)))
    np.testing.assert_allclose(left[0],mono,atol=1e-12)
    np.testing.assert_allclose(right[1],mono,atol=1e-12)
    assert not np.any(left[1])
    assert np.max(np.abs(right[0]))<1e-12


def test_layer_placement_and_mix_preserve_silent_prefix():
    voice=NatureVoice("stream",.7,seed=13)
    layer=NatureLayer(voice,12000,24000,width=.7)
    scene=NatureScene(48000,(layer,),room=0)
    a,report=render_scene(scene)
    assert a.shape==(2,48000)
    assert not np.any(a[:,:12000]) and not np.any(a[:,36000:])
    assert np.any(a[:,12000:36000])
    assert report["scene_sha256"]==scene_digest(scene)
    assert report["layers"][0]["start"]==12000
    assert report["layers"][0]["frames"]==24000


@pytest.mark.parametrize("name",["forest","storm","coast","creek","campfire"])
def test_preset_stereo_wav_replay_and_peak(name):
    scene=preset(name,2.,17)
    a,ra=render_scene(scene)
    b,rb=render_scene(scene)
    assert np.array_equal(a,b) and ra==rb
    assert a.shape==(2,48000)
    assert not np.array_equal(a[0],a[1])
    assert np.max(np.abs(a))<=.900001
    assert ra["metrics"]["clipping_fraction"]==0
    wav,report=generate_scene_audio(scene)
    assert wav[:4]==b"RIFF" and len(wav)==44+48000*4
    assert report["frames"]==48000
    assert not report["learned_model"]


def test_master_attenuation_only():
    voice=NatureVoice("rain",1.,1.,1.,17)
    layer=NatureLayer(voice,0,24000,gain=(AutomationPoint(0,12.),))
    scene=NatureScene(24000,(layer,layer,layer),peak_ceiling=.1,room=1.)
    audio,report=render_scene(scene)
    assert np.max(np.abs(audio))<=.10000001
    assert report["master_attenuation"]<1
    quiet=NatureScene(24000,(replace(layer,voice=replace(voice,intensity=0)),))
    audio,report=render_scene(quiet)
    assert not np.any(audio) and report["master_attenuation"]==1


@pytest.mark.parametrize("settings",[
    {"start":-1},{"frames":100},{"width":1.1},{"fade_in":24000},
    {"gain":(AutomationPoint(1,0),)},
    {"pan":(AutomationPoint(0,2),)},
    {"gain":(AutomationPoint(0,0),AutomationPoint(0,1))},
    {"pan":(AutomationPoint(0,0),AutomationPoint(24000,1))},
])
def test_invalid_layer_geometry(settings):
    values={"voice":NatureVoice("rain"),"start":0,"frames":24000}
    values.update(settings)
    with pytest.raises(GenerationError):
        NatureLayer(**values)


def test_scene_budget_and_out_of_bounds():
    layer=NatureLayer(NatureVoice("rain"),0,24000)
    with pytest.raises(GenerationError):
        NatureScene(12000,(layer,))
    with pytest.raises(GenerationError):
        NatureScene(24000,(layer,)*13)
    long=NatureLayer(NatureVoice("rain"),0,30*SAMPLE_RATE)
    with pytest.raises(GenerationError):
        NatureScene(30*SAMPLE_RATE,(long,)*7)
