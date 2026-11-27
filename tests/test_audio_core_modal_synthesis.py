"""Procedural contact sound geometry, spectrum and deterministic events."""
import json
from dataclasses import replace
import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.modal_synthesis import (
    ResonanceMode,Resonator,ContactEvent,ContactScene,MATERIALS,EXCITATIONS,
    material_resonator,excitation_impulses,render_contact_scene,
)


@pytest.mark.parametrize("material",MATERIALS)
def test_material_profiles_render_original_audio(material):
    resonator=material_resonator(material)
    event=ContactEvent(200,4000,resonator,seed=12)
    scene=ContactScene(24000,5000,(event,))
    a=render_contact_scene(scene)
    b=render_contact_scene(scene)
    assert a.audio.frames==5000 and a.audio.channels==2
    assert a.report.output_peak>0
    assert a.report.output_peak<=.9000001
    assert a.report.scene_sha256==scene.sha256
    np.testing.assert_array_equal(a.audio.samples,b.audio.samples)
    assert np.all(a.audio.samples[:,:200]==0)
    assert np.all(a.audio.samples[:,4200:]==0)
    assert np.all(a.audio.samples[:,4199]==0)
    json.dumps(a.report.as_dict(),allow_nan=False)


@pytest.mark.parametrize("excitation",EXCITATIONS)
def test_excitation_schedule(excitation):
    event=ContactEvent(0,8000,material_resonator("wood"),
                       excitation=excitation,seed=9)
    a=excitation_impulses(event,8000)
    assert a==excitation_impulses(event,8000)
    assert all(0<=position<8000 and strength>0 for position,strength in a)
    assert [p for p,_ in a]==sorted(set(p for p,_ in a))
    if excitation in ("rattle","scrape","step"):
        assert len(a)>1
        assert a!=excitation_impulses(replace(event,seed=10),8000)


def test_single_mode_frequency_and_decay():
    resonator=Resonator((ResonanceMode(400,.2,1),),contact_noise=0)
    event=ContactEvent(0,8000,resonator,fade_out_frames=0)
    x=render_contact_scene(ContactScene(8000,8000,(event,),channels=1)).audio.samples[0]
    frequency=np.argmax(np.abs(np.fft.rfft(x*np.hanning(len(x)))))*8000/len(x)
    assert abs(frequency-400)<=1
    a=np.sqrt(np.mean(x[800:1600]**2))
    b=np.sqrt(np.mean(x[2400:3200]**2))
    assert b/a==pytest.approx(np.exp(-1),rel=.01)


def test_pitch_transposition_and_size():
    resonator=Resonator((ResonanceMode(400,.2,1),),contact_noise=0)
    event=ContactEvent(0,8000,resonator,pitch_semitones=12)
    x=render_contact_scene(ContactScene(8000,8000,(event,),channels=1)).audio.samples[0]
    frequency=np.argmax(np.abs(np.fft.rfft(x*np.hanning(len(x)))))
    assert abs(frequency-800)<=1
    a=material_resonator("wood",size=1)
    b=material_resonator("wood",size=2,damping=2)
    assert b.modes[0].frequency_hz==a.modes[0].frequency_hz/2
    assert b.modes[0].decay_seconds==a.modes[0].decay_seconds/2


def test_nyquist_guard_omits_modes_and_rejects_empty():
    profile=Resonator((ResonanceMode(200,.1,1),ResonanceMode(4000,.1,.5)))
    event=ContactEvent(0,1000,profile)
    out=render_contact_scene(ContactScene(8000,1000,(event,)))
    assert out.report.event_reports[0].active_modes==1
    assert out.report.event_reports[0].omitted_modes==1
    event=replace(event,resonator=Resonator((ResonanceMode(4000,.1,1),)))
    with pytest.raises(AudioContractError,match="No resonance"):
        render_contact_scene(ContactScene(8000,1000,(event,)))


def test_hard_pan_and_event_mix():
    a=ContactEvent(0,2000,material_resonator("wood"),pan=-1)
    b=replace(a,start_frame=1000,pan=1,seed=2)
    scene=ContactScene(8000,3000,(a,b),peak_ceiling=None)
    result=render_contact_scene(scene)
    assert np.all(result.audio.samples[1,:1000]==0)
    assert np.all(result.audio.samples[0,2000:]==0)
    solo_a=render_contact_scene(replace(scene,events=(a,)))
    solo_b=render_contact_scene(replace(scene,events=(b,)))
    np.testing.assert_allclose(result.audio.samples,solo_a.audio.samples+solo_b.audio.samples,atol=1e-7)


def test_peak_ceiling_and_silence():
    event=ContactEvent(0,1000,material_resonator("metal"),strength=1)
    raw=render_contact_scene(ContactScene(8000,1000,(event,)*8,peak_ceiling=None))
    capped=render_contact_scene(ContactScene(8000,1000,(event,)*8,peak_ceiling=.3))
    assert raw.report.output_peak>.3
    assert capped.report.attenuation<1
    assert capped.report.output_peak<=.3000001
    silent=render_contact_scene(ContactScene(8000,1000,(replace(event,strength=0),)))
    assert not np.any(silent.audio.samples)


@pytest.mark.parametrize("kwargs",[
    {"frequency_hz":True},{"frequency_hz":10},{"decay_seconds":0},
    {"amplitude":float("nan")},{"amplitude":2},
])
def test_mode_validation(kwargs):
    data=dict(frequency_hz=400,decay_seconds=.1,amplitude=1)
    data.update(kwargs)
    with pytest.raises(AudioContractError):
        ResonanceMode(**data)


def test_resonator_validation():
    m=ResonanceMode(200,.1,1)
    for modes in ((),[m],(m,m),(ResonanceMode(200,.1,0),)):
        with pytest.raises(AudioContractError):
            Resonator(modes)


@pytest.mark.parametrize("kwargs",[
    {"start_frame":-1},{"frames":7},{"strength":True},{"hardness":2},
    {"pitch_semitones":13},{"pan":-2},{"seed":-1},{"fade_out_frames":1001},
    {"excitation":"unknown"},
])
def test_contact_validation(kwargs):
    data=dict(start_frame=0,frames=1000,resonator=material_resonator("wood"))
    data.update(kwargs)
    with pytest.raises(AudioContractError):
        ContactEvent(**data)


def test_scene_geometry_and_limits(monkeypatch):
    import sonumio.audio_core.modal_synthesis as module
    event=ContactEvent(0,1000,material_resonator("wood"))
    with pytest.raises(AudioContractError):
        ContactScene(8000,999,(event,))
    with pytest.raises(AudioContractError):
        ContactScene(8000,1000,(replace(event,pan=.3),),channels=1)
    scene=ContactScene(8000,1000,(event,))
    monkeypatch.setattr(module,"MAX_MODE_PRODUCTS",1)
    with pytest.raises(AudioLimitError,match="mode-product"):
        render_contact_scene(scene)
