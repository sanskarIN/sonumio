"""Canonical sound pack archive integrity, replay and atomic publication."""
from dataclasses import asdict,replace
import hashlib
import io
import json
import os
import zipfile
import numpy as np
import pytest

from sonumio.audio_core.asset_pack import (
    AssetRecipe,SoundPackPlan,generate_pack_members,plan_jobs,inspect_asset,build_manifest,
)
from sonumio.audio_core.asset_archive import (
    create_pack_bytes,verify_pack_bytes,verify_pack_file,publish_pack_bytes,
    export_sound_pack,load_pack_plan,_canonical_archive,
)
from sonumio.audio_core.asset_pack_cli import main
from sonumio.audio_core.errors import AudioContractError,AudioIntegrityError,AudioLimitError
from sonumio.audio_core.hashing import canonical_json_bytes
from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.waveio import encode_wav_bytes


def plan():
    return SoundPackPlan("rain-library",(
        AssetRecipe("rain",4801,variants=2,tags=("ambient","rain")),
        AssetRecipe("stream",4800,nature_kind="stream"),
    ),seed=9)


@pytest.mark.parametrize("bits",[8,16,24,32])
def test_canonical_bytes_reproducible_and_replay(bits):
    cfg=replace(plan(),output_bits=bits,dither=True)
    a,first=create_pack_bytes(cfg)
    b,second=create_pack_bytes(cfg)
    assert a==b and first==second
    assert first.archive_sha256==hashlib.sha256(a).hexdigest()
    assert first.archive_bytes==len(a)
    assert first.manifest["plan_sha256"]==cfg.sha256
    assert first.replayed_assets==0
    result=verify_pack_bytes(a,expected_sha256=first.archive_sha256,replay=True)
    assert result.replayed_assets==cfg.asset_count
    assert result.as_dict()["recipe_replay_verified"]
    assert not result.as_dict()["signature_verified"]
    with zipfile.ZipFile(io.BytesIO(a)) as archive:
        assert archive.namelist()==["manifest.json",*[j.member for j in plan_jobs(cfg)]]
        assert all(i.compress_type==zipfile.ZIP_STORED for i in archive.infolist())


def test_forged_but_self_consistent_source_claim_needs_replay():
    cfg=plan()
    members,_=generate_pack_members(cfg)
    job=plan_jobs(cfg)[0]
    # A valid, non-silent, correctly sized WAV from an unrelated signal.
    samples=(.2*np.sin(np.arange(job.recipe.frames)*.13)).astype(np.float32)
    members[job.member]=encode_wav_bytes(make_buffer(samples,24000))
    records=tuple(inspect_asset(j,members[j.member],cfg.output_bits) for j in plan_jobs(cfg))
    manifest=build_manifest(cfg,records)
    raw=_canonical_archive(canonical_json_bytes(manifest),members)
    assert verify_pack_bytes(raw).replayed_assets==0
    with pytest.raises(AudioIntegrityError,match="replay mismatch"):
        verify_pack_bytes(raw,replay=True)


def test_tampered_manifest_and_payload_are_rejected():
    cfg=plan()
    members,manifest=generate_pack_members(cfg)
    manifest["assets"][0]["signal"]["sum_codes"]+=1
    raw=_canonical_archive(canonical_json_bytes(manifest),members)
    with pytest.raises(AudioIntegrityError,match="manifest"):
        verify_pack_bytes(raw)
    members,manifest=generate_pack_members(cfg)
    name=next(iter(members))
    changed=bytearray(members[name])
    changed[44]^=1
    members[name]=bytes(changed)
    raw=_canonical_archive(canonical_json_bytes(manifest),members)
    with pytest.raises(AudioIntegrityError,match="manifest"):
        verify_pack_bytes(raw)


def test_wrong_digest_and_trailing_bytes():
    raw,_=create_pack_bytes(plan())
    with pytest.raises(AudioIntegrityError,match="SHA256"):
        verify_pack_bytes(raw,expected_sha256="0"*64)
    with pytest.raises(AudioIntegrityError,match="canonical"):
        verify_pack_bytes(raw+b"extra")


def test_missing_extra_and_unsafe_names_are_rejected():
    members,manifest=generate_pack_members(plan())
    payload=canonical_json_bytes(manifest)
    missing=dict(members)
    del missing[next(iter(missing))]
    extra=dict(members)
    extra["../unwanted.wav"]=next(iter(members.values()))
    for values in (missing,extra):
        raw=_canonical_archive(payload,values)
        with pytest.raises(AudioContractError):
            verify_pack_bytes(raw)


def test_compressed_and_duplicate_entries_are_rejected():
    members,manifest=generate_pack_members(plan())
    output=io.BytesIO()
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json",canonical_json_bytes(manifest))
        for name,data in members.items():
            archive.writestr(name,data)
    with pytest.raises(AudioContractError,match="stored"):
        verify_pack_bytes(output.getvalue())
    output=io.BytesIO()
    with zipfile.ZipFile(output,"w") as archive:
        archive.writestr("manifest.json",canonical_json_bytes(manifest))
        with pytest.warns(UserWarning,match="Duplicate"):
            archive.writestr("manifest.json",b"{}")
        for name,data in members.items():
            archive.writestr(name,data)
    with pytest.raises(AudioContractError,match="Duplicate"):
        verify_pack_bytes(output.getvalue())


def test_alternate_metadata_and_member_order_are_rejected():
    members,manifest=generate_pack_members(plan())
    output=io.BytesIO()
    with zipfile.ZipFile(output,"w") as archive:
        # Default ZipInfo timestamps are not the canonical fixed date.
        archive.writestr("manifest.json",canonical_json_bytes(manifest))
        for name,data in members.items():
            archive.writestr(name,data)
    with pytest.raises(AudioIntegrityError,match="canonical"):
        verify_pack_bytes(output.getvalue())
    output=io.BytesIO()
    with zipfile.ZipFile(output,"w") as archive:
        archive.writestr("manifest.json",canonical_json_bytes(manifest))
        for name in reversed(list(members)):
            archive.writestr(name,members[name])
    with pytest.raises(AudioIntegrityError,match="members"):
        verify_pack_bytes(output.getvalue())


def test_archive_cap_and_malformed_zip(monkeypatch):
    import sonumio.audio_core.asset_archive as module
    with pytest.raises(AudioContractError):
        verify_pack_bytes(b"not a zip")
    raw,_=create_pack_bytes(plan())
    monkeypatch.setattr(module,"MAX_ARCHIVE_BYTES",1)
    with pytest.raises(AudioLimitError):
        verify_pack_bytes(raw)


def test_atomic_export_and_pinned_file_verification(tmp_path):
    path=tmp_path/"library.zip"
    a=export_sound_pack(plan(),path)
    assert path.read_bytes()
    assert verify_pack_file(path,expected_sha256=a.verified.archive_sha256)==a.verified
    assert verify_pack_file(path,replay=True).replayed_assets==3
    assert not list(tmp_path.glob("*.partial"))
    before=path.read_bytes()
    with pytest.raises(AudioContractError,match="overwrite"):
        export_sound_pack(plan(),path)
    with pytest.raises(AudioIntegrityError,match="SHA256"):
        export_sound_pack(plan(),path,overwrite=True,expected_sha256="0"*64)
    assert path.read_bytes()==before
    b=export_sound_pack(replace(plan(),seed=10),path,overwrite=True)
    assert b.verified.archive_sha256!=a.verified.archive_sha256


def test_invalid_archive_never_creates_destination_parent(tmp_path):
    target=tmp_path/"missing"/"bad.zip"
    with pytest.raises(AudioContractError):
        publish_pack_bytes(b"invalid",target,create_parent=True)
    assert not target.parent.exists()
    raw,_=create_pack_bytes(plan())
    with pytest.raises(AudioIntegrityError):
        publish_pack_bytes(raw,target,create_parent=True,expected_sha256="0"*64)
    assert not target.parent.exists()


def test_symlink_destination_and_input_are_rejected(tmp_path):
    raw,_=create_pack_bytes(plan())
    original=tmp_path/"original.zip"
    original.write_bytes(raw)
    alias=tmp_path/"alias.zip"
    try:
        alias.symlink_to(original)
    except OSError:
        pytest.skip("symlink privileges unavailable")
    with pytest.raises(AudioContractError):
        verify_pack_file(alias)
    with pytest.raises(AudioContractError):
        publish_pack_bytes(raw,alias,overwrite=True)
    assert original.read_bytes()==raw


def test_plan_snapshot_and_cli_roundtrip(tmp_path,capsys):
    cfg=plan()
    plan_path=tmp_path/"plan.json"
    plan_path.write_bytes(canonical_json_bytes(asdict(cfg)))
    assert load_pack_plan(plan_path)==cfg
    out=tmp_path/"library.zip"
    assert main(["sound-pack-plan","--plan",str(plan_path)])==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["asset_count"]==3
    assert not out.exists()
    assert main(["sound-pack","--plan",str(plan_path),"--output",str(out)])==0
    created=json.loads(capsys.readouterr().out)
    assert main(["sound-pack-verify",str(out),"--expected-sha256",created["archive_sha256"],"--replay"])==0
    result=json.loads(capsys.readouterr().out)
    assert result["recipe_replay_verified"]
    assert result["archive_sha256"]==created["archive_sha256"]


def test_cli_bad_input_no_output(tmp_path,capsys):
    plan_path=tmp_path/"plan.json"
    plan_path.write_text('{"pack_id":"broken","recipes":[]}')
    out=tmp_path/"bad.zip"
    assert main(["sound-pack","--plan",str(plan_path),"--output",str(out)])==2
    assert "recipe" in capsys.readouterr().err
    assert not out.exists()
