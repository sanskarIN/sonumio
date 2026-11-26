"""Exact soundtrack placement, sidechain causality and end-to-end export."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
import wave

from sonumio.soundtrack import (
    Placement,Ducking,SoundtrackPlan,place,ducking_envelope,combine,
    soundtrack_digest,render_soundtrack,generate_soundtrack,CONTROL_HOP,
)
from sonumio.arrangement import ArrangementPlan,compose_arrangement
from sonumio.nature_scene import preset
from sonumio.music_score_io import encode_score
from sonumio.nature_scene_io import encode_scene
from sonumio.soundtrack_cli import main
from sonumio.errors import GenerationError


def stereo(frames,value=.1):
    return np.full((2,frames),value,dtype=np.float32)


def test_exact_source_slice_destination_and_gain():
    src=np.stack((np.arange(10),-np.arange(10))).astype(np.float32)/10
    settings=Placement(2,3,4,0,0,0)
    result=place(src,settings,12)
    expected=np.zeros((2,12))
    expected[:,3:7]=src[:,2:6]
    np.testing.assert_array_equal(result,expected)
    gained=place(src,replace(settings,gain_db=-6),12)
    np.testing.assert_allclose(gained,expected*10**(-6/20),atol=1e-12)


def test_source_is_not_mutated_and_fades_are_exact():
    source=stereo(2400)
    before=source.copy()
    result=place(source,Placement(0,10,2000,0,200,400),3000)
    assert np.array_equal(source,before)
    assert not np.any(result[:,:10])
    assert result[0,10]==0 and abs(result[0,2009])<1e-12
    assert not np.any(result[:,2010:])
    assert result[0,209]==pytest.approx(.1)
    assert result[0,1610]==pytest.approx(.1)


@pytest.mark.parametrize("placement",[
    Placement(10,0,10,0,0,0),Placement(0,10,10,0,0,0),
    Placement(0,0,10,0,6,5),Placement(10,0,None,0,0,0),
])
def test_reject_outside_geometry_or_overlap(placement):
    with pytest.raises(GenerationError):
        placement.resolve(10,10)


@pytest.mark.parametrize("kwargs",[
    {"source_start":-1},{"destination_start":True},{"frames":0},
    {"gain_db":float("nan")},{"fade_in":-1},
])
def test_invalid_placement(kwargs):
    with pytest.raises(GenerationError):
        Placement(**kwargs)


def test_disabled_ducking_and_ratio_one_are_identity():
    x=stereo(24000,.5)
    for config in (Ducking(),Ducking(True,ratio=1),Ducking(True,max_reduction_db=0)):
        gain,report=ducking_envelope(x,config)
        assert np.array_equal(gain,np.ones(24000))
        assert report["max_reduction_db"]==0


def test_ducking_is_causal_and_attack_release_bounded():
    x=stereo(24000*3,0)
    x[:,24000:48000]=.5
    config=Ducking(True,-30,4,12,.05,.2)
    gain,report=ducking_envelope(x,config)
    assert np.all(gain[:24000+CONTROL_HOP]==1)
    assert gain[36000]<.5
    assert gain[48000]<.5
    assert gain[-1]>.95
    assert np.min(gain)>=10**(-12/20)-1e-12
    assert 0<report["mean_reduction_db"]<report["max_reduction_db"]<=12+1e-12
    other=x.copy()
    other[:,48000:]=.7
    changed,_=ducking_envelope(other,config)
    np.testing.assert_array_equal(gain[:48000+CONTROL_HOP],changed[:48000+CONTROL_HOP])


def test_silent_sidechain_does_not_duck():
    gain,report=ducking_envelope(stereo(24001,0),Ducking(True))
    assert np.array_equal(gain,np.ones(24001))
    assert report["max_reduction_db"]==0


@pytest.mark.parametrize("kwargs",[
    {"enabled":1},{"threshold_dbfs":-100},{"ratio":.9},{"max_reduction_db":31},
    {"attack_seconds":0},{"release_seconds":float("inf")},
])
def test_invalid_ducking(kwargs):
    with pytest.raises(GenerationError):
        Ducking(**kwargs)


def test_linear_combine_and_attenuation_only():
    music=stereo(2400,.2)
    ambience=stereo(2400,.1)
    p=Placement(0,0,None,0,0,0)
    plan=SoundtrackPlan(2400,p,p,peak_ceiling=.9)
    output,report=combine(music,ambience,plan)
    np.testing.assert_allclose(output,.3,atol=1e-7)
    assert report["master_attenuation"]==1
    louder,report=combine(music,ambience,replace(plan,peak_ceiling=.1))
    np.testing.assert_allclose(louder,.1,atol=1e-7)
    assert report["master_attenuation"]==pytest.approx(1/3,rel=1e-6)
    assert report["metrics"]["clipping_fraction"]==0


def test_ducking_changes_only_ambience_before_mastering():
    music=stereo(24000,.3)
    ambience=stereo(24000,.1)
    p=Placement(0,0,None,0,0,0)
    plan=SoundtrackPlan(24000,p,p)
    a,ra=combine(music,ambience,plan)
    b,rb=combine(music,ambience,replace(plan,ducking=Ducking(True)))
    assert ra["music_placed"]==rb["music_placed"]
    assert ra["ambience_placed"]==rb["ambience_placed"]
    assert rb["ambience_after_ducking"]["rms"]<ra["ambience_after_ducking"]["rms"]
    assert np.all(b<=a)
    assert np.all(b>=music-1e-7)


def sources():
    score=compose_arrangement(ArrangementPlan(bars=4,bpm=160,seed=17))
    scene=preset("forest",2.,29)
    plan=SoundtrackPlan(48000,Placement(12000,0,48000,-3,240,480),
                        Placement(0,0,48000,-6,240,480),ducking=Ducking(True))
    return score,scene,plan


def test_complete_original_soundtrack_repeatability():
    score,scene,plan=sources()
    a,ra=render_soundtrack(score,scene,plan)
    b,rb=render_soundtrack(score,scene,plan)
    assert np.array_equal(a,b)
    assert ra==rb
    assert ra["soundtrack_sha256"]==soundtrack_digest(score,scene,plan)
    assert ra["soundtrack_sha256"]!=soundtrack_digest(score,scene,replace(plan,peak_ceiling=.8))
    assert a.shape==(2,48000) and np.max(np.abs(a))<=.900001
    assert not ra["learned_model"]
    assert not ra["external_recordings_used"]
    wav,report=generate_soundtrack(score,scene,plan)
    assert report["wav_sha256"]==hashlib.sha256(wav).hexdigest()


def test_invalid_geometry_rejected_before_render(monkeypatch):
    import sonumio.soundtrack as module
    score,scene,plan=sources()
    def unexpected(*args):
        raise AssertionError("Invalid placement reached renderer")
    monkeypatch.setattr(module,"render_stems",unexpected)
    with pytest.raises(GenerationError):
        render_soundtrack(score,scene,replace(plan,frames=1000))


def test_cli_pins_source_and_output_digests(tmp_path,capsys):
    score,scene,plan=sources()
    sp=tmp_path/"music.json"
    npth=tmp_path/"nature.json"
    out=tmp_path/"mix.wav"
    sp.write_bytes(encode_score(score))
    npth.write_bytes(encode_scene(scene))
    args=[str(sp),str(npth),"--frames","48000","--music-source-start","12000",
          "--music-frames","48000","--ambience-frames","48000","--duck",
          "--output",str(out)]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert report["output_sha256"]==hashlib.sha256(out.read_bytes()).hexdigest()
    with wave.open(str(out),"rb") as wav:
        assert wav.getnframes()==48000 and wav.getnchannels()==2 and wav.getframerate()==24000
    original=out.read_bytes()
    with pytest.raises(SystemExit):
        main(args+["--overwrite","--expected-output-sha256","0"*64])
    assert out.read_bytes()==original
    with pytest.raises(SystemExit):
        main(args+["--overwrite","--expected-score-sha256","0"*64])
    assert out.read_bytes()==original
