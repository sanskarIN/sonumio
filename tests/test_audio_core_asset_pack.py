"""Deterministic original asset plans, integer diagnostics and quality gates."""
from dataclasses import asdict,replace
import hashlib
import json
import numpy as np
import pytest

from sonumio.audio_core.asset_pack import (
    AssetRecipe,PackQuality,SoundPackPlan,parse_pack_plan,plan_jobs,render_asset,
    inspect_asset,integer_signal,generate_pack_members,build_manifest,duplicate_groups,
    wav_size,
)
from sonumio.audio_core.errors import AudioContractError,AudioIntegrityError,AudioLimitError
from sonumio.audio_core.hashing import canonical_json_bytes


def plan():
    return SoundPackPlan("forest-effects",(
        AssetRecipe("rain",4801,variants=3,tags=("ambient","rain")),
        AssetRecipe("wind",6000,variants=2,nature_kind="wind"),
    ),seed=42)


def test_job_seed_stability_when_other_recipes_are_added():
    original=plan()
    a=plan_jobs(original)
    expanded=replace(original,recipes=(AssetRecipe("birds",4800,nature_kind="birds"),*original.recipes))
    b=plan_jobs(expanded)
    assert a==b[1:]
    assert all(j.dither_seed is None for j in a)
    dithered=plan_jobs(replace(original,dither=True))
    assert [j.seed for j in a]==[j.seed for j in dithered]
    assert all(j.seed!=j.dither_seed for j in dithered)
    assert len({j.seed for j in a})==5
    assert [j.member for j in a]==["audio/rain-001.wav","audio/rain-002.wav",
        "audio/rain-003.wav","audio/wind-001.wav","audio/wind-002.wav"]


@pytest.mark.parametrize("bits",[8,16,24,32])
def test_exact_integer_signal_no_overflow(bits):
    high=(1<<(bits-1))-1
    low=-(1<<(bits-1))
    values=[low,high,0,1,-1,low,high]*12000
    result=integer_signal(np.array(values,dtype=np.int64)[None,:],bits)
    assert result.samples==len(values)
    assert result.minimum_code==low and result.maximum_code==high
    assert result.peak_code==-low
    assert result.sum_codes==sum(values)
    assert result.sum_squared_codes==sum(v*v for v in values)
    assert result.positive_rails==result.negative_rails==24000
    assert result.nonzero_samples==72000
    assert result.rms(bits)==pytest.approx((sum(v*v for v in values)/len(values))**.5/-low)


@pytest.mark.parametrize("bits",[8,16,24,32])
def test_pack_geometry_and_repeatability(bits):
    cfg=replace(plan(),output_bits=bits,dither=True)
    a,manifest=generate_pack_members(cfg)
    b,second=generate_pack_members(cfg)
    assert a==b and manifest==second
    assert manifest["asset_count"]==5
    assert manifest["total_frames"]==cfg.total_frames
    assert manifest["payload_bytes"]==cfg.payload_bytes==sum(map(len,a.values()))
    assert manifest["plan_sha256"]==cfg.sha256
    assert len({r["wav_sha256"] for r in manifest["assets"]})==5
    for job,record in zip(plan_jobs(cfg),manifest["assets"],strict=True):
        assert len(a[job.member])==wav_size(job.recipe.frames,bits)
        assert record["wav_sha256"]==hashlib.sha256(a[job.member]).hexdigest()
        assert record==inspect_asset(job,a[job.member],bits).as_dict()
    json.dumps(manifest,allow_nan=False)


def test_prompt_source_is_original_and_exact():
    recipe=AssetRecipe("drone",12001,source="procedural",nature_kind=None,prompt="ambient")
    cfg=SoundPackPlan("drone-pack",(recipe,))
    members,manifest=generate_pack_members(cfg)
    assert manifest["assets"][0]["frames"]==12001
    assert manifest["assets"][0]["source"]=="procedural"
    assert not manifest["model_inference"]
    assert not manifest["recording_sources_used"]


def test_exact_duplicate_policy():
    recipe=AssetRecipe("drone",12000,variants=2,source="procedural",nature_kind=None,prompt="ambient")
    cfg=SoundPackPlan("drone-pack",(recipe,))
    members,manifest=generate_pack_members(cfg)
    assert len(set(members.values()))==1
    assert manifest["duplicate_groups"]==(("audio/drone-001.wav","audio/drone-002.wav"),)
    with pytest.raises(AudioIntegrityError,match="Identical"):
        generate_pack_members(replace(cfg,quality=PackQuality(allow_duplicate_audio=False)))


def test_silence_quality_gate_and_explicit_allowance():
    cfg=SoundPackPlan("silent",(AssetRecipe("silent",2400,intensity=0),))
    with pytest.raises(AudioIntegrityError,match="RMS"):
        generate_pack_members(cfg)
    _,manifest=generate_pack_members(replace(cfg,quality=PackQuality(minimum_rms=0)))
    assert manifest["assets"][0]["signal"]["sum_squared_codes"]==0


def test_bad_encoded_geometry_and_pad():
    cfg=replace(plan(),output_bits=8)
    job=plan_jobs(cfg)[0]
    wav=render_asset(job,cfg)
    assert wav[-1]==0
    with pytest.raises(AudioIntegrityError):
        inspect_asset(job,wav[:-1],8)
    with pytest.raises(AudioContractError):
        inspect_asset(job,wav[:-1]+b"\x01",8)


def test_plan_roundtrip():
    cfg=plan()
    decoded=parse_pack_plan(canonical_json_bytes(asdict(cfg)))
    assert decoded==cfg and decoded.sha256==cfg.sha256


@pytest.mark.parametrize("raw",[
    b'{}',b'{"pack_id":"p","recipes":[]}',
    b'{"pack_id":"p","pack_id":"q","recipes":[]}',
    b'{"pack_id":"p","recipes":[{"asset_id":"../x","frames":2400}]}',
    b'{"pack_id":"p","recipes":[{"asset_id":"con","frames":2400}]}',
    b'{"pack_id":"p","recipes":[{"asset_id":"x","frames":2400,"tags":"abc"}]}',
    b'{"pack_id":"p","recipes":[{"asset_id":"x","frames":2400,"intensity":NaN}]}',
    b'{"pack_id":"p","recipes":[{"asset_id":"x","frames":2400,"unknown":1}]}',
])
def test_invalid_plan_json(raw):
    with pytest.raises((AudioContractError,TypeError,ValueError)):
        parse_pack_plan(raw)


def test_recipe_and_total_budget_validation():
    with pytest.raises(AudioContractError):
        AssetRecipe("rain",True)
    with pytest.raises(AudioContractError):
        AssetRecipe("rain",2400,tags=("z","a"))
    with pytest.raises(AudioContractError):
        AssetRecipe("rain",2400,source="procedural",nature_kind=None,prompt="rain")
    with pytest.raises(ValueError):
        AssetRecipe("rain",12000,source="procedural",nature_kind=None,prompt="clone voice rain")
    with pytest.raises(AudioContractError):
        SoundPackPlan("pack",(AssetRecipe("z",2400),AssetRecipe("a",2400)))
    with pytest.raises(AudioLimitError):
        SoundPackPlan("pack",(AssetRecipe("long",360000,variants=24),))


def test_manifest_rejects_forged_identity():
    cfg=plan()
    jobs=plan_jobs(cfg)
    records=tuple(inspect_asset(j,render_asset(j,cfg),cfg.output_bits) for j in jobs)
    with pytest.raises(AudioIntegrityError):
        build_manifest(cfg,(replace(records[0],seed=0),*records[1:]))
    assert not duplicate_groups(records)
