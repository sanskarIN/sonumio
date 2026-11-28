"""Sound-design patch validation, signal oracles and arrangement integration."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan,Event,TRACKS,compose_arrangement,arrangement_digest
from sonumio.audio import SAMPLE_RATE
from sonumio.errors import GenerationError
from sonumio.music_automation import AutomationPlan,MixKey,MixLane,generate_automated_music
from sonumio.music_mix import render_stems,generate_arrangement_audio
from sonumio.music_patch_synthesis import (
    Partial,OscillatorBank,Envelope,NoiseLayer,InstrumentPatch,
    envelope_samples,partial_frequency,active_partials,patch_cost,render_patch,
)
from sonumio.music_patch_bank import (
    PatchBank,PatchRoute,encode_patch_bank,decode_patch_bank,describe_patch_bank,
    render_patched_stems,generate_patched_music,MAX_BYTES,
)
from sonumio.music_patch_presets import NAMES,preset,preset_bank
from sonumio.music_patch_cli import main
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4,bpm=160,seed=17,swing=.2))


def sine_patch():
    return InstrumentPatch((OscillatorBank((Partial(1,1.),)),),Envelope(1,0,1.,1))


def test_pure_sine_matches_independent_absolute_clock():
    event=Event("lead",69,8,4,.5,0.)
    frames=2400
    result=render_patch(event,frames,17,sine_patch())
    expected=.5*np.sin(2*np.pi*440*np.arange(frames)/SAMPLE_RATE)
    expected[0]=expected[-1]=0.
    np.testing.assert_allclose(result,expected,atol=1e-13,rtol=0)


@pytest.mark.parametrize("frames",[2,3,17,120,12000,96000])
def test_envelope_zero_edges_and_bounded_short_notes(frames):
    env=envelope_samples(Envelope(3600,2400,.8,5400),frames)
    assert env.shape==(frames,)
    assert env[0]==env[-1]==0.
    assert np.isfinite(env).all()
    assert 0<=env.min()<=env.max()<=1.


def test_sustain_and_release_absolute_geometry():
    env=envelope_samples(Envelope(10,20,.4,30),100)
    assert env[10]==pytest.approx(1.)
    assert env[30]==pytest.approx(.4)
    assert env[69]==pytest.approx(.4)
    assert env[-1]==0
    assert all(a>=b for a,b in zip(env[69:],env[70:]))


def test_detune_octave_and_inharmonicity():
    partial=Partial(3,.4)
    bank=OscillatorBank((partial,),detune_cents=100.,octave=1,inharmonicity=.001)
    expected=440.*2**(1+100/1200)*3*np.sqrt(1+.001*9)
    assert partial_frequency(69,bank,partial)==pytest.approx(expected)


def test_frequency_guard_removes_high_harmonics_without_renormalizing():
    low=Partial(1,.25)
    high=Partial(16,.75)
    patch=InstrumentPatch((OscillatorBank((low,high),octave=1),),Envelope(1,0,1.,1))
    event=Event("lead",84,0,4,1.,0.)
    rows=active_partials(event.pitch,patch.banks[0])
    assert [p.harmonic for p,_ in rows]==[1]
    actual=render_patch(event,2048,17,patch)
    assert np.max(np.abs(actual))<=.250001


def test_no_representable_partial_rejected():
    patch=InstrumentPatch((OscillatorBank((Partial(16,1.),),octave=2),))
    with pytest.raises(GenerationError,match="representable"):
        render_patch(Event("lead",96,0,1,.5,0.),2048,17,patch)


def test_velocity_exponent_is_explicit():
    event=Event("lead",69,0,4,.5,0.)
    patch=replace(sine_patch(),velocity_exponent=2.)
    a=render_patch(event,2400,17,patch)
    b=render_patch(replace(event,velocity=1.),2400,17,patch)
    np.testing.assert_array_equal(a,b*.25)


@pytest.mark.parametrize("color",["white","low","high"])
def test_seeded_noise_is_order_independent_and_bounded(color):
    patch=InstrumentPatch((OscillatorBank((Partial(1,1.),),gain=.6),),
                          Envelope(1,0,1.,1),NoiseLayer(.4,color,8,.2))
    event=Event("lead",69,8,4,.8,0.)
    a=render_patch(event,4096,17,patch)
    render_patch(replace(event,tick=16),4096,17,patch)
    b=render_patch(event,4096,17,patch)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,render_patch(event,4096,18,patch))
    assert not np.array_equal(a,render_patch(replace(event,tick=9),4096,17,patch))
    assert np.max(np.abs(a))<=.8+1e-12


@pytest.mark.parametrize("name",NAMES)
def test_all_presets_are_valid_and_different_from_plain_sine(name):
    patch=preset(name)
    event=Event("lead",69,0,4,.7,0.)
    a=render_patch(event,4096,17,patch)
    b=render_patch(event,4096,17,sine_patch())
    assert np.any(a)
    assert np.max(np.abs(a))<=1
    assert not np.allclose(a,b)


@pytest.mark.parametrize("factory",[
    lambda:Partial(0,.1),lambda:Partial(True,.1),lambda:Partial(1,1.1),
    lambda:Partial(1,.5,.001),lambda:Partial(1,.5,phase_cycles=float("nan")),
    lambda:OscillatorBank((Partial(1,.6),Partial(2,.6))),
    lambda:OscillatorBank((Partial(2,.3),Partial(1,.3))),
    lambda:OscillatorBank((Partial(1,.2),Partial(1,.3))),
    lambda:OscillatorBank((Partial(1,1.),),octave=3),
    lambda:Envelope(0),lambda:Envelope(sustain=1.1),lambda:NoiseLayer(color="external"),
    lambda:InstrumentPatch((OscillatorBank((Partial(1,1.),)),),noise=NoiseLayer(.5)),
])
def test_invalid_patch_controls(factory):
    with pytest.raises(GenerationError):
        factory()


def test_empty_bank_has_exact_original_stem_and_wav_parity(score):
    bank=PatchBank(arrangement_digest(score))
    original=render_stems(score)
    actual=render_patched_stems(score,bank)
    for track in TRACKS:
        np.testing.assert_array_equal(actual[track],original[track])
    wav,_=generate_patched_music(score,bank)
    assert wav==generate_arrangement_audio(score)[0]
    auto=AutomationPlan((MixLane("lead","gain",(MixKey(0,.5),)),))
    assert generate_patched_music(score,bank,auto)[0]==generate_automated_music(score,auto)[0]


def test_routed_patch_changes_only_selected_stem(score):
    bank=preset_bank(score,{"lead":"reed"})
    old=render_stems(score)
    actual=render_patched_stems(score,bank)
    for track in TRACKS:
        if track=="lead":
            assert not np.array_equal(actual[track],old[track])
        else:
            np.testing.assert_array_equal(actual[track],old[track])
        assert not actual[track].flags.writeable
    wav,report=generate_patched_music(score,bank)
    assert report["patch_bank_sha256"]==bank.sha256
    assert report["wav_sha256"]==hashlib.sha256(wav).hexdigest()
    assert report["custom_note_events"]==sum(e.track=="lead" for e in score.events)


def test_canonical_patch_bank_and_strict_nested_json(score):
    bank=preset_bank(score,{"lead":"reed","keys":"glass"})
    assert decode_patch_bank(encode_patch_bank(bank),expected_sha256=bank.sha256)==bank
    raw=json.loads(encode_patch_bank(bank))
    raw["routes"][0]["patch"]["banks"][0]["plugin_path"]="example.so"
    with pytest.raises(GenerationError):
        decode_patch_bank(json.dumps(raw).encode())
    with pytest.raises(GenerationError):
        decode_patch_bank(encode_patch_bank(bank),expected_sha256="0"*64)


@pytest.mark.parametrize("raw",[b"",b"[]",b"{}",b"\xff",b"x"*(MAX_BYTES+1),
                                 b'{"format":1,"format":2}'])
def test_invalid_bank_document(raw):
    with pytest.raises(GenerationError):
        decode_patch_bank(raw)


def test_route_validation_and_stale_score(score):
    route=PatchRoute("lead",sine_patch())
    with pytest.raises(GenerationError):
        PatchRoute("kick",sine_patch())
    with pytest.raises(GenerationError):
        PatchBank(arrangement_digest(score),(route,route))
    with pytest.raises(GenerationError,match="SHA256"):
        describe_patch_bank(score,PatchBank("0"*64))


def test_budget_rejected_before_render(score,monkeypatch):
    bank=preset_bank(score,{"lead":"reed"})
    import sonumio.music_patch_bank as module
    monkeypatch.setattr(module,"MAX_COST",1)
    def forbidden(*_):
        pytest.fail("synthesis started before checking patch budget")
    monkeypatch.setattr(module,"_render_stems",forbidden)
    with pytest.raises(GenerationError,match="budget"):
        render_patched_stems(score,bank)


def test_cli_presets_inspection_and_atomic_render(score,tmp_path,capsys):
    source=tmp_path/"score.json"
    patches=tmp_path/"patches.json"
    target=tmp_path/"out.wav"
    source.write_bytes(encode_score(score))
    assert main(["presets",str(source),"--assign","lead=reed","--assign","keys=glass"])==0
    patches.write_text(capsys.readouterr().out)
    bank=decode_patch_bank(patches.read_bytes())
    assert main(["inspect",str(source),str(patches)])==0
    assert json.loads(capsys.readouterr().out)["patch_bank_sha256"]==bank.sha256
    args=["render",str(source),str(patches),"--output",str(target)]
    assert main(args+["--expected-patches-sha256","0"*64])==2
    assert not target.exists()
    assert main(args+["--expected-patches-sha256",bank.sha256])==0
    report=json.loads(capsys.readouterr().out)
    assert report["output_sha256"]==hashlib.sha256(target.read_bytes()).hexdigest()
    assert main(args)==2
    assert main(["render",str(source),str(patches),"--output",str(source),"--overwrite"])==2
