"""Complete sound recipes: independent engine parity and strict source binding."""
from dataclasses import replace
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan, TRACKS, compose_arrangement
from sonumio.audio import SAMPLE_RATE, encode_wav_channels
from sonumio.errors import GenerationError
from sonumio.music_project import create_project
from sonumio.music_automation import AutomationPlan, MixLane, MixKey, mix_automated_stems
from sonumio.music_performance import Performance, Expression, render_performance_stems
from sonumio.music_patch_bank import PatchBank, encode_patch_bank
from sonumio.music_patch_presets import preset_bank
from sonumio.music_drum_bank import DrumBank, encode_drum_bank, render_combined_stems
from sonumio.music_drum_presets import preset_drum_bank
from sonumio.music_rack import RackPlan, TrackRack, mix_rack_stems
from sonumio.music_rack_effects import Insert
from sonumio.music_render_recipe import RenderRecipe, encode_recipe, encode_receipt, _canonical, _sha
from sonumio.music_render_bundle import create_bundle, verify_bundle
from sonumio.music_sound_recipe import (
    SoundRecipe, compile_sound, describe_sound, encode_sound_recipe, decode_sound_recipe,
    render_sound_stems, render_sound, verify_sound, MAX_BYTES,
)
from sonumio.music_sound_bundle import create_sound_bundle, verify_sound_bundle
from sonumio.music_sound_recipe_cli import main


@pytest.fixture(scope="module")
def project():
    score=compose_arrangement(ArrangementPlan(bars=4,bpm=160,seed=17))
    auto=AutomationPlan((MixLane("lead","gain",(MixKey(0,.6),MixKey(64,1.))),))
    return create_project("Custom sound",score,auto)


@pytest.fixture(scope="module")
def banks(project):
    return (preset_bank(project.score,{"lead":"reed","keys":"glass"}),
            preset_drum_bank(project.score,{"kick":"deep","hat":"metal"}))


@pytest.fixture(scope="module")
def performance(project):
    i=next(i for i,e in enumerate(project.score.events) if e.track=="lead")
    return Performance(project.score_sha256,(Expression(i,velocity_scale=.65,pan_delta=.1),))


@pytest.fixture(scope="module")
def rack(project):
    return RackPlan(project.automation,tracks=(TrackRack("lead",(Insert.create("gain",db=2.),)),))


def test_component_roundtrip_and_identity(project,banks,performance,rack):
    recipe=SoundRecipe(RenderRecipe(project,performance,rack),*banks)
    data=encode_sound_recipe(recipe)
    assert decode_sound_recipe(data,expected_sha256=recipe.sha256)==recipe
    assert recipe.identity["project_sha256"]==project.sha256
    assert recipe.identity["performance_sha256"]==performance.sha256
    assert recipe.identity["pitched_bank_sha256"]==banks[0].sha256
    assert recipe.identity["drum_bank_sha256"]==banks[1].sha256
    assert recipe.sha256!=SoundRecipe(RenderRecipe(project,performance),*banks).sha256
    assert recipe.sha256!=SoundRecipe(RenderRecipe(project,performance,rack),banks[0]).sha256


@pytest.mark.parametrize("with_rack",[False,True])
def test_without_performance_matches_independent_custom_engines(project,banks,rack,with_rack):
    base=RenderRecipe(project,rack=rack if with_rack else None)
    recipe=SoundRecipe(base,*banks)
    stems=render_combined_stems(project.score,banks[1],banks[0])
    actual=render_sound_stems(recipe)
    for track in TRACKS:
        np.testing.assert_array_equal(actual[track],stems[track])
    audio,_=(mix_rack_stems(project.score,stems,rack) if with_rack
             else mix_automated_stems(project.score,stems,project.automation))
    wav,report=render_sound(recipe)
    assert wav==encode_wav_channels(audio,SAMPLE_RATE)
    assert report["wav_sha256"]==hashlib.sha256(wav).hexdigest()
    assert verify_sound(recipe,wav,report,replay=True)["replayed"]


def test_performance_unassigned_tracks_match_original_overlay(project,banks,performance):
    recipe=SoundRecipe(RenderRecipe(project,performance),*banks)
    actual=render_sound_stems(recipe)
    baseline=render_performance_stems(project.score,performance)
    for track in TRACKS:
        if track not in ("lead","keys","kick","hat"):
            np.testing.assert_array_equal(actual[track],baseline[track])
        assert not actual[track].flags.writeable
    assert not np.array_equal(actual["lead"],baseline["lead"])
    identity=SoundRecipe(RenderRecipe(project,Performance(project.score_sha256)),*banks)
    no_overlay=SoundRecipe(RenderRecipe(project),*banks)
    for track in TRACKS:
        np.testing.assert_array_equal(render_sound_stems(identity)[track],
                                      render_sound_stems(no_overlay)[track])


def test_cost_uses_actual_performed_duration_before_synthesis(project,banks):
    i=next(i for i,e in enumerate(project.score.events) if e.track=="lead" and e.length>=2)
    performance=Performance(project.score_sha256,(Expression(i,end_delta=-10),))
    original=compile_sound(SoundRecipe(RenderRecipe(project),*banks))
    shortened=compile_sound(SoundRecipe(RenderRecipe(project,performance),*banks))
    assert shortened.pitched_cost<original.pitched_cost
    assert shortened.drum_cost==original.drum_cost
    assert sum(shortened.track_frames)==sum(original.track_frames)-10


def test_cost_failure_precedes_stem_allocation(project,banks,monkeypatch):
    import sonumio.music_sound_recipe as module
    monkeypatch.setattr(module,"MAX_TOTAL_COST",1)
    with pytest.raises(GenerationError,match="budget"):
        SoundRecipe(RenderRecipe(project),*banks)


def test_stale_bank_and_empty_recipe_rejected(project,banks):
    with pytest.raises(GenerationError,match="SHA256"):
        SoundRecipe(RenderRecipe(project),replace(banks[0],score_sha256="0"*64))
    with pytest.raises(GenerationError,match="SHA256"):
        SoundRecipe(RenderRecipe(project),drums=replace(banks[1],score_sha256="0"*64))
    with pytest.raises(GenerationError,match="at least one"):
        SoundRecipe(RenderRecipe(project),PatchBank(project.score_sha256),DrumBank(project.score_sha256))


@pytest.mark.parametrize("raw",[b"",b"{}",b"[]",b"\xff",b"x"*(MAX_BYTES+1),b'{"format":1,"format":2}'])
def test_invalid_sound_recipe(raw):
    with pytest.raises(GenerationError):
        decode_sound_recipe(raw)


def test_nested_patch_tampering_and_noncanonical_rejected(project,banks):
    recipe=SoundRecipe(RenderRecipe(project),*banks)
    data=encode_sound_recipe(recipe)
    raw=json.loads(data)
    raw["pitched"]["routes"][0]["patch"]["velocity_exponent"]=.9
    with pytest.raises(GenerationError):
        decode_sound_recipe(_canonical(raw))
    raw=json.loads(data)
    raw["identity"]["pitched_bank_sha256"]="0"*64
    with pytest.raises(GenerationError):
        decode_sound_recipe(_canonical(raw))
    with pytest.raises(GenerationError):
        decode_sound_recipe(data,expected_sha256="0"*64)


def test_receipt_hash_and_replay_have_distinct_meanings(project,banks,performance,rack):
    recipe=SoundRecipe(RenderRecipe(project,performance,rack),*banks)
    wav,report=render_sound(recipe)
    altered=wav[:-1]+bytes([wav[-1]^1])
    with pytest.raises(GenerationError,match="checksum"):
        verify_sound(recipe,altered,report)
    # A self-consistent replacement WAV is not provenance. Only replay
    # detects a forged receipt with an updated digest.
    replacement=dict(report,wav_sha256=_sha(altered))
    replacement.pop("receipt_sha256")
    replacement["receipt_sha256"]=_sha(_canonical(replacement))
    encode_receipt(replacement)
    assert verify_sound(recipe,altered,replacement)["authenticated"] is False
    with pytest.raises(GenerationError,match="replay"):
        verify_sound(recipe,altered,replacement,replay=True)


@pytest.fixture(scope="module")
def bundle(project,banks,performance,rack):
    recipe=SoundRecipe(RenderRecipe(project,performance,rack),*banks)
    return recipe,create_sound_bundle(recipe)


def test_sound_bundle_exact_replay(bundle):
    recipe,data=bundle
    actual,wav,receipt,report=verify_sound_bundle(data,expected_sha256=recipe.sha256,replay=True)
    assert actual==recipe
    assert report["replayed"]
    assert report["wav_sha256"]==hashlib.sha256(wav).hexdigest()
    assert report["receipt_sha256"]==receipt["receipt_sha256"]
    assert report["bundle_sha256"]==hashlib.sha256(data).hexdigest()


def test_bundle_tamper_and_v1_isolation(bundle,project):
    _,data=bundle
    with pytest.raises(GenerationError):
        verify_sound_bundle(data+b"extra")
    with pytest.raises(GenerationError):
        verify_bundle(data)
    old=create_bundle(RenderRecipe(project))
    assert verify_bundle(old,replay=True)[3]["replayed"]
    with pytest.raises(GenerationError):
        verify_sound_bundle(old)
    target=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src,zipfile.ZipFile(target,"w") as out:
        for info in src.infolist():
            payload=src.read(info)
            if info.filename=="audio.wav":
                payload=payload[:-1]+bytes([payload[-1]^1])
            out.writestr(info,payload)
    with pytest.raises(GenerationError):
        verify_sound_bundle(target.getvalue())


def test_complete_recipe_cli_atomic_publication(project,banks,tmp_path,capsys):
    base=tmp_path/"base.json"
    pitched=tmp_path/"pitched.json"
    drums=tmp_path/"drums.json"
    sound=tmp_path/"sound.json"
    archive=tmp_path/"render.zip"
    wav=tmp_path/"audio.wav"
    base.write_bytes(encode_recipe(RenderRecipe(project)))
    pitched.write_bytes(encode_patch_bank(banks[0]))
    drums.write_bytes(encode_drum_bank(banks[1]))
    assert main(["create",str(base),"--pitched",str(pitched),"--drums",str(drums),"--output",str(sound)])==0
    planned=json.loads(capsys.readouterr().out)
    assert not planned["rendered"]
    assert main(["bundle",str(sound),"--output",str(archive)])==0
    bundled=json.loads(capsys.readouterr().out)
    assert bundled["sound_recipe_sha256"]==planned["sound_recipe_sha256"]
    assert main(["verify",str(archive),"--replay"])==0
    assert json.loads(capsys.readouterr().out)["replayed"]
    assert main(["extract-wav",str(archive),"--output",str(wav)])==0
    assert hashlib.sha256(wav.read_bytes()).hexdigest()==bundled["wav_sha256"]
    assert main(["extract-wav",str(archive),"--output",str(wav)])==2
    assert main(["create",str(base),"--pitched",str(pitched),"--output",str(base),"--overwrite"])==2
