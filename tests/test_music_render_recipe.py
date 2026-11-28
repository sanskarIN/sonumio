"""Integrated render receipts, independent engine parity and archive rejection."""
from dataclasses import replace
import hashlib
import io
import json
import stat
import zipfile

import pytest

from sonumio.arrangement import ArrangementPlan, arrangement_digest, compose_arrangement
from sonumio.audio import SAMPLE_RATE, encode_wav_channels
from sonumio.errors import GenerationError
from sonumio.music_automation import AutomationPlan, MixKey, MixLane
from sonumio.music_performance import Expression, Performance, generate_performance_music, render_performance_stems
from sonumio.music_project import create_project, encode_project, render_project
from sonumio.music_rack import RackPlan, TrackRack, generate_rack_music, mix_rack_stems
from sonumio.music_rack_effects import Insert
from sonumio.music_render_recipe import (
    RenderRecipe, encode_recipe, decode_recipe, describe_recipe, render_recipe,
    encode_receipt, decode_receipt, verify_render, MAX_RECIPE_BYTES,
)
from sonumio.music_render_bundle import (
    create_bundle, encode_bundle, verify_bundle, MEMBERS, MAX_BUNDLE_BYTES,
)
from sonumio.music_render_wav import inspect_pcm16
from sonumio.music_render_recipe_cli import main


@pytest.fixture(scope="module")
def project():
    score = compose_arrangement(ArrangementPlan(bars=4, bpm=160, seed=42))
    auto = AutomationPlan((MixLane("lead", "gain", (MixKey(0, .6), MixKey(64, 1.))),))
    return create_project("Integrated render", score, auto)


@pytest.fixture(scope="module")
def performance(project):
    index = next(i for i,e in enumerate(project.score.events) if e.track == "lead")
    return Performance(project.score_sha256, (Expression(index, velocity_scale=.8),))


@pytest.fixture(scope="module")
def rack(project):
    return RackPlan(project.automation,
                    tracks=(TrackRack("lead", (Insert.create("gain", db=3.),)),),
                    master=(Insert.create("width", width=.8),))


@pytest.mark.parametrize("mode", ["project", "performance", "rack", "performance-rack"])
def test_all_four_render_paths_match_independent_engines(project, performance, rack, mode):
    recipe = RenderRecipe(project, performance if "performance" in mode else None,
                          rack if "rack" in mode else None)
    wav, report = render_recipe(recipe)
    if mode == "project":
        expected, _ = render_project(project)
    elif mode == "performance":
        expected, _ = generate_performance_music(project.score, performance, project.automation)
    elif mode == "rack":
        expected, _ = generate_rack_music(project.score, rack)
    else:
        stems = render_performance_stems(project.score, performance)
        audio, _ = mix_rack_stems(project.score, stems, rack)
        expected = encode_wav_channels(audio, SAMPLE_RATE)
    assert wav == expected
    assert report["engine"] == mode
    assert report["rendered"] is True
    assert report["wav_sha256"] == hashlib.sha256(wav).hexdigest()
    assert decode_recipe(encode_recipe(recipe)) == recipe
    assert decode_receipt(encode_receipt(report)) == report
    verified = verify_render(recipe, wav, report, replay=True)
    assert verified["replayed"] is True and verified["authenticated"] is False


def test_reject_mismatching_project_automation_and_stale_expression(project):
    with pytest.raises(GenerationError, match="automation"):
        RenderRecipe(project, rack=RackPlan())
    with pytest.raises(GenerationError, match="SHA256"):
        RenderRecipe(project, performance=Performance("0"*64))


def test_recipe_digest_pins_and_nested_canonicality(project, performance):
    recipe = RenderRecipe(project, performance)
    data = encode_recipe(recipe)
    assert decode_recipe(data, expected_sha256=recipe.sha256) == recipe
    with pytest.raises(GenerationError, match="identity"):
        decode_recipe(data, expected_sha256="0"*64)
    raw = json.loads(data)
    raw["identity"]["performance_sha256"] = None
    with pytest.raises(GenerationError, match="identity"):
        decode_recipe(json.dumps(raw).encode())
    raw = json.loads(data)
    raw["performance"]["expressions"][0]["velocity_scale"] = .9
    with pytest.raises(GenerationError, match="identity"):
        decode_recipe(json.dumps(raw).encode())


@pytest.mark.parametrize("data", [b"", b"[]", b"\xff", b"{}", b"x"*(MAX_RECIPE_BYTES+1),
                                  b'{"format":1,"format":2}'])
def test_malformed_recipe(data):
    with pytest.raises(GenerationError):
        decode_recipe(data)


def test_wav_header_is_verified_independently(project):
    recipe = RenderRecipe(project)
    wav, report = render_recipe(recipe)
    pcm = inspect_pcm16(wav, report["frames"])
    assert pcm["frames"] == report["frames"]
    assert len(pcm["sum_squares"]) == 2
    assert pcm["clipped_samples"] == [0,0]
    altered = bytearray(wav)
    altered[22] = 1
    with pytest.raises(GenerationError, match="header"):
        inspect_pcm16(bytes(altered), report["frames"])


def test_receipt_and_audio_mutation_rejected(project):
    recipe = RenderRecipe(project)
    wav, report = render_recipe(recipe)
    with pytest.raises(GenerationError, match="digest"):
        verify_render(recipe, wav, {**report, "engine": "rack"})
    with pytest.raises(GenerationError, match="checksum"):
        verify_render(recipe, wav[:-1]+bytes([wav[-1]^1]), report)
    with pytest.raises(GenerationError, match="recipe"):
        verify_render(RenderRecipe(project, Performance(project.score_sha256)), wav, report)


@pytest.fixture(scope="module")
def bundle(project, performance, rack):
    recipe = RenderRecipe(project, performance, rack)
    return recipe, create_bundle(recipe)


def test_bundle_round_trip_and_exact_replay(bundle):
    recipe, data = bundle
    decoded, wav, report, verified = verify_bundle(data, replay=True, expected_recipe_sha256=recipe.sha256)
    assert decoded == recipe
    assert verified["members_verified"] == list(MEMBERS)
    assert verified["bundle_sha256"] == hashlib.sha256(data).hexdigest()
    assert verified["replayed"] is True
    assert encode_bundle(decoded, wav, report) == data


def altered_archive(data, action):
    target = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(target,"w") as out:
        for info in source.infolist():
            payload = source.read(info)
            if action == "missing" and info.filename == "audio.wav":
                continue
            if action == "tampered" and info.filename == "audio.wav":
                payload = payload[:-1]+bytes([payload[-1]^1])
            if action == "compressed" and info.filename == "audio.wav":
                info.compress_type = zipfile.ZIP_DEFLATED
            if action == "symlink" and info.filename == "audio.wav":
                info.external_attr = (stat.S_IFLNK | 0o777)<<16
            out.writestr(info,payload)
        if action == "extra":
            out.writestr("../outside.wav",b"bad")
        if action == "duplicate":
            with pytest.warns(UserWarning):
                out.writestr("recipe.json",b"bad")
    return target.getvalue()


@pytest.mark.parametrize("action", ["missing","tampered","compressed","symlink","extra","duplicate"])
def test_archive_member_attacks_rejected(bundle, action):
    with pytest.raises(GenerationError):
        verify_bundle(altered_archive(bundle[1], action))


def test_archive_trailing_bytes_and_bad_pin(bundle):
    recipe, data = bundle
    with pytest.raises(GenerationError, match="canonical"):
        verify_bundle(data+b"trailing")
    with pytest.raises(GenerationError):
        verify_bundle(data, expected_recipe_sha256="0"*64)
    with pytest.raises(GenerationError):
        verify_bundle(b"x"*(MAX_BUNDLE_BYTES+1))


def test_cli_create_bundle_verify_extract(project, tmp_path, capsys):
    source = tmp_path/"source.sonumio"
    recipe = tmp_path/"recipe.json"
    archive = tmp_path/"render.zip"
    output = tmp_path/"audio.wav"
    source.write_bytes(encode_project(project))
    assert main(["create",str(source),"--output",str(recipe)]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert not planned["rendered"]
    assert main(["bundle",str(recipe),"--output",str(archive)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["recipe_sha256"] == planned["recipe_sha256"]
    assert main(["verify",str(archive),"--replay"]) == 0
    assert json.loads(capsys.readouterr().out)["replayed"]
    assert main(["extract-wav",str(archive),"--output",str(output)]) == 0
    assert hashlib.sha256(output.read_bytes()).hexdigest() == receipt["wav_sha256"]
    assert main(["extract-wav",str(archive),"--output",str(output)]) == 2
    assert main(["create",str(source),"--output",str(source),"--overwrite"]) == 2
