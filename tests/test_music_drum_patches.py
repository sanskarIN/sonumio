"""Independent modal phase oracles, patch budgets and track routing tests."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan,Event,TRACKS,PITCHED,compose_arrangement,arrangement_digest
from sonumio.audio import SAMPLE_RATE
from sonumio.errors import GenerationError
from sonumio.music_automation import AutomationPlan,generate_automated_music
from sonumio.music_mix import render_stems,generate_arrangement_audio
from sonumio.music_patch_synthesis import NoiseLayer
from sonumio.music_patch_presets import preset_bank
from sonumio.music_patch_bank import render_patched_stems,encode_patch_bank
from sonumio.music_drum_patches import (
    DrumMode,DrumNoise,DrumPatch,phase_cycles,render_drum_patch,DRUMS,
)
from sonumio.music_drum_bank import (
    DrumRoute,DrumBank,encode_drum_bank,decode_drum_bank,describe_combined_banks,
    render_combined_stems,generate_combined_music,MAX_BYTES,
)
from sonumio.music_drum_presets import PRESETS,drum_preset,preset_drum_bank
from sonumio.music_drum_cli import main
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4,bpm=160,seed=17))


@pytest.mark.parametrize("start,end",[(100.,50.),(50.,100.),(100.,100.)])
def test_analytic_phase_matches_integrated_frequency(start,end):
    mode=DrumMode(start,end,.04,.8,.3,.25)
    t=np.arange(12000,dtype=np.float64)/SAMPLE_RATE
    phase=phase_cycles(t,mode)
    assert phase[0]==.25
    midpoint=(t[1:]+t[:-1])/2
    expected=end+(start-end)*np.exp(-midpoint/.04)
    np.testing.assert_allclose(np.diff(phase)*SAMPLE_RATE,expected,atol=1e-5,rtol=1e-6)


def test_static_mode_matches_explicit_damped_sine():
    mode=DrumMode(120.,120.,.03,.8,.1,.25)
    patch=DrumPatch((mode,),attack_frames=1,release_frames=1)
    event=Event("kick",36,0,4,.5,0.)
    t=np.arange(2400)/SAMPLE_RATE
    expected=.4*np.sin(2*np.pi*(120*t+.25))*np.exp(-t/.1)
    expected[0]=expected[-1]=0.
    np.testing.assert_allclose(render_drum_patch(event,len(t),17,patch),expected,atol=1e-13,rtol=0)


@pytest.mark.parametrize("track,name",[(t,n) for t,names in PRESETS.items() for n in names])
def test_presets_sound_and_repeat_deterministically(track,name):
    patch=drum_preset(track,name)
    event=Event(track,{"kick":36,"snare":38,"hat":42}[track],8,4,.7,0.)
    actual=render_drum_patch(event,4096,17,patch)
    np.testing.assert_array_equal(actual,render_drum_patch(event,4096,17,patch))
    assert np.any(actual)
    assert actual[0]==actual[-1]==0
    assert np.max(np.abs(actual))<=.700001
    assert not np.array_equal(actual,render_drum_patch(event,4096,18,patch))


def test_noise_only_patch_and_short_event_bounds():
    patch=DrumPatch((),(DrumNoise(NoiseLayer(.8,"low",8,.1),2),))
    event=Event("hat",42,0,1,1.,0.)
    for frames in (2,3,10,2048):
        result=render_drum_patch(event,frames,17,patch)
        assert result.shape==(frames,)
        assert result[0]==result[-1]==0
        assert np.max(np.abs(result))<=.8


def test_noise_layers_are_independent_of_other_events():
    patch=drum_preset("snare","snap")
    event=Event("snare",38,8,4,.5,0.)
    a=render_drum_patch(event,2048,17,patch)
    render_drum_patch(replace(event,tick=32),2048,17,patch)
    np.testing.assert_array_equal(a,render_drum_patch(event,2048,17,patch))
    assert not np.array_equal(a,render_drum_patch(replace(event,tick=9),2048,17,patch))


@pytest.mark.parametrize("factory",[
    lambda:DrumMode(0,100,.03,.5,.1),lambda:DrumMode(20000,100,.03,.5,.1),
    lambda:DrumMode(100,50,0,.5,.1),lambda:DrumMode(100,50,.03,True,.1),
    lambda:DrumMode(100,50,.03,.5,float("nan")),
    lambda:DrumNoise(NoiseLayer()),lambda:DrumPatch(()),
    lambda:DrumPatch((DrumMode(100,50,.03,.8,.1),), (DrumNoise(NoiseLayer(.5)),)),
    lambda:DrumPatch((DrumMode(100,50,.03,.5,.1),),attack_frames=0),
    lambda:DrumRoute("lead",drum_preset("kick","round")),
])
def test_invalid_controls(factory):
    with pytest.raises(GenerationError):
        factory()


def test_empty_drum_bank_preserves_original_and_automated_mix(score):
    bank=DrumBank(arrangement_digest(score))
    a=render_stems(score)
    b=render_combined_stems(score,bank)
    for track in TRACKS:
        np.testing.assert_array_equal(a[track],b[track])
    assert generate_combined_music(score,bank)[0]==generate_arrangement_audio(score)[0]
    auto=AutomationPlan(())
    assert generate_combined_music(score,bank,mix=auto)[0]==generate_automated_music(score,auto)[0]


def test_combined_banks_preserve_unassigned_and_independent_tracks(score):
    drums=preset_drum_bank(score,{"kick":"deep","hat":"soft"})
    pitched=preset_bank(score,{"keys":"glass"})
    original=render_stems(score)
    pitch_only=render_patched_stems(score,pitched)
    drum_only=render_combined_stems(score,drums)
    combined=render_combined_stems(score,drums,pitched)
    for track in TRACKS:
        np.testing.assert_array_equal(combined[track],pitch_only[track] if track in PITCHED else drum_only[track])
        if track not in ("keys","kick","hat"):
            np.testing.assert_array_equal(combined[track],original[track])
    for track in ("keys","kick","hat"):
        assert not np.array_equal(combined[track],original[track])
    report=describe_combined_banks(score,drums,pitched)
    assert report["cost"]==report["drums"]["cost"]+report["pitched"]["cost"]


def test_bank_document_roundtrip_and_strict_fields(score):
    bank=preset_drum_bank(score,{"kick":"round","hat":"metal"})
    data=encode_drum_bank(bank)
    assert decode_drum_bank(data,expected_sha256=bank.sha256)==bank
    raw=json.loads(data)
    raw["routes"][0]["patch"]["modes"][0]["path"]="file.wav"
    with pytest.raises(GenerationError):
        decode_drum_bank(json.dumps(raw).encode())
    with pytest.raises(GenerationError):
        decode_drum_bank(data,expected_sha256="0"*64)


@pytest.mark.parametrize("data",[b"",b"[]",b"{}",b"\xff",b"x"*(MAX_BYTES+1),
                                  b'{"format":1,"format":2}'])
def test_invalid_document(data):
    with pytest.raises(GenerationError):
        decode_drum_bank(data)


def test_stale_source_and_budget_fail_before_synthesis(score,monkeypatch):
    bank=preset_drum_bank(score,{"kick":"round"})
    with pytest.raises(GenerationError,match="SHA256"):
        render_combined_stems(score,replace(bank,score_sha256="0"*64))
    import sonumio.music_drum_bank as module
    monkeypatch.setattr(module,"MAX_COMBINED_COST",1)
    def forbidden(*_):
        pytest.fail("Synthesis started before combined budget verification")
    monkeypatch.setattr(module,"_render_stems",forbidden)
    with pytest.raises(GenerationError,match="budget"):
        render_combined_stems(score,bank)


def test_cli_presets_and_combined_render(score,tmp_path,capsys):
    source=tmp_path/"score.json"
    drums=tmp_path/"drums.json"
    pitched=tmp_path/"pitched.json"
    output=tmp_path/"render.wav"
    source.write_bytes(encode_score(score))
    pitched.write_bytes(encode_patch_bank(preset_bank(score,{"keys":"glass"})))
    assert main(["presets",str(source),"--assign","kick=deep","--assign","hat=metal"])==0
    drums.write_text(capsys.readouterr().out)
    args=[str(source),str(drums),"--pitched",str(pitched)]
    assert main(["inspect",*args])==0
    expected=json.loads(capsys.readouterr().out)
    assert main(["render",*args,"--output",str(output)])==0
    report=json.loads(capsys.readouterr().out)
    assert report["cost"]==expected["cost"]
    assert report["output_sha256"]==hashlib.sha256(output.read_bytes()).hexdigest()
    assert main(["render",*args,"--output",str(output)])==2
    assert main(["render",*args,"--output",str(source),"--overwrite"])==2
