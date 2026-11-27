"""Primary CLI routing for the new original DSP generation tools."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sonumio.cli import main
from sonumio.audio_core import (
    AssetRecipe, SoundPackPlan, ContactPattern, pattern_scene,
    SpatialPlan, SpatialPlacement, Trajectory, Receiver, Position,
    SourceBinding, create_pack_bytes, verify_pack_bytes, select_asset,
    make_buffer,
)
from sonumio.audio_core.hashing import canonical_json_bytes
from sonumio.audio_core.waveio import encode_wav_bytes, read_wav_file


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _plan(path, obj):
    path.write_bytes(canonical_json_bytes(obj))
    return path


def test_main_contact_routes_and_preserves_geometry(tmp_path, capsys):
    output = tmp_path / "contacts.wav"
    assert main(["contact-sound", "--material", "wood", "--count", "2",
                 "--event-ms", "80", "--interval-ms", "150",
                 "--sample-rate", "8000", "--output", str(output)]) == 0
    report = _json(capsys)
    audio = read_wav_file(output)
    assert audio.channels == 2
    assert audio.sample_rate == 8000
    assert audio.frames == 1840
    assert report["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["render"]["model_inference"] is False


def test_main_spatial_routes_and_named_source_pins(tmp_path, capsys):
    source = tmp_path / "tone.wav"
    samples = (.2 * np.sin(np.arange(800) * .2)).astype(np.float32)
    source.write_bytes(encode_wav_bytes(make_buffer(samples, 8000)))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = _plan(tmp_path / "scene.json", {
        "sample_rate": 8000,
        "placements": [{"source_id": "tone", "trajectory": {
            "keys": [{"frame": 0, "position": {"x": 0, "y": 2}}]}}],
        "receiver": {"spacing_m": 0, "sound_speed": 320},
    })
    output = tmp_path / "spatial.wav"
    args = ["spatial-scene", "--plan", str(plan), "--source", "tone=" + str(source),
            "--source-sha256", "tone=" + digest, "--output", str(output)]
    assert main(args) == 0
    report = _json(capsys)
    assert report["sources"]["tone"]["source_sha256"] == digest
    assert report["render"]["frames"] == 850
    assert read_wav_file(output).channels == 2
    assert all(not p["truncated"] for p in report["render"]["paths"])


def test_main_library_full_workflow(tmp_path, capsys):
    plan = _plan(tmp_path / "library.json", {
        "pack_id": "demo", "seed": 7,
        "recipes": [{"asset_id": "rain", "frames": 2400, "variants": 2,
                     "tags": ["rain"]}],
    })
    archive = tmp_path / "library.zip"
    assert main(["sound-pack-plan", "--plan", str(plan)]) == 0
    assert _json(capsys)["asset_count"] == 2
    assert main(["sound-pack", "--plan", str(plan), "--output", str(archive)]) == 0
    created = _json(capsys)
    digest = created["archive_sha256"]
    assert main(["sound-pack-verify", str(archive), "--expected-sha256", digest,
                 "--replay"]) == 0
    assert _json(capsys)["replayed_assets"] == 2
    assert main(["sound-pack-list", str(archive), "--tag", "rain"]) == 0
    assert _json(capsys)["matched_assets"] == 2
    output = tmp_path / "rain.wav"
    assert main(["sound-pack-select", str(archive), "--asset", "rain",
                 "--variant", "2", "--expected-sha256", digest,
                 "--output", str(output)]) == 0
    selected = _json(capsys)
    assert selected["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert read_wav_file(output).frames == 2400


def test_primary_cli_invalid_library_plan_does_not_publish(tmp_path, capsys):
    plan = _plan(tmp_path / "bad.json", {"pack_id": "bad", "recipes": []})
    output = tmp_path / "bad.zip"
    assert main(["sound-pack", "--plan", str(plan), "--output", str(output)]) == 2
    assert "recipe" in capsys.readouterr().err
    assert not output.exists()


def test_contact_json_alias_protects_source(tmp_path, capsys):
    plan = _plan(tmp_path / "contact.json", {"pattern": {"count": 1}})
    target = tmp_path / "alias.wav"
    try:
        os.link(plan, target)
    except OSError:
        pytest.skip("hardlinks unavailable")
    original = plan.read_bytes()
    assert main(["contact-sound", "--scene", str(plan), "--output",
                 str(target), "--overwrite"]) == 2
    assert "file identity" in capsys.readouterr().err
    assert plan.read_bytes() == target.read_bytes() == original


def test_public_api_generation_exports():
    config = SoundPackPlan("demo", (AssetRecipe("rain", 2400),))
    raw, receipt = create_pack_bytes(config)
    assert verify_pack_bytes(raw).archive_sha256 == receipt.archive_sha256
    assert select_asset(raw, "rain").record["frames"] == 2400
    assert len(pattern_scene(ContactPattern(count=1)).events) == 1
    p = SpatialPlan(8000, (SpatialPlacement("sound", Trajectory.stationary(0, 0)),),
                    receiver=Receiver(Position(0, 0)))
    assert p.source_ids == ("sound",)


def test_cli_help_lists_new_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("contact-sound", "spatial-scene", "sound-pack",
                    "sound-pack-plan", "sound-pack-verify", "sound-pack-list",
                    "sound-pack-select"):
        assert command in help_text
