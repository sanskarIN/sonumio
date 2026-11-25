"""Stage boundary equivalence, private checkpoint failure and no-clobber tests."""
from dataclasses import replace
import hashlib
import json
import struct

import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_latent_study import run_archive_latent
from research.archive_latent_handoff import HEADER,MAGIC,MAX_HEADER,decode_handoff
from research.crop_codec import CropAudioCodec
from research.corpus_archive_format import ArchiveError,canonical_json
from test_archive_latent import small_plan
from test_archive_training import fixture,build


@pytest.fixture(scope="module")
def stage(tmp_path_factory):
    root=tmp_path_factory.mktemp("codec-handoff")
    archive=root/"fixture.scorpus"
    result=build(archive,fixture())
    plan=small_plan()
    ckpt=root/"codec.ckpt"
    rng=torch.get_rng_state().clone()
    first=run_archive_latent(archive,expected_sha256=result.archive_sha256,plan=plan)
    only=run_archive_latent(archive,expected_sha256=result.archive_sha256,plan=plan,
                            codec_out=ckpt,codec_only=True)
    assert torch.equal(rng,torch.get_rng_state())
    pin=only["codec_handoff"]["receipt"]["sha256"]
    return archive,result.archive_sha256,plan,ckpt,pin,first,only


def test_codec_stage_replay_exact_two_stage_equivalence(stage,monkeypatch):
    archive,sha,plan,ckpt,pin,first,only=stage
    def forbidden(*args,**kwargs):
        pytest.fail("Restored stage must not train codec again")
    monkeypatch.setattr("research.archive_latent_study.train_codec",forbidden)
    rng=torch.get_rng_state().clone()
    resumed=run_archive_latent(archive,expected_sha256=sha,plan=plan,codec_in=ckpt,codec_sha256=pin)
    assert torch.equal(rng,torch.get_rng_state())
    assert only["denoiser_steps_executed"]==0
    assert only["codec"]==first["codec"]==resumed["codec"]
    assert resumed["denoiser"]==first["denoiser"]
    assert resumed["validation"]==first["validation"]
    assert resumed["generation"]==first["generation"]
    assert resumed["codec_handoff"]["codec_optimizer_steps_executed"]==0
    assert resumed["codec_handoff"]["mode"]=="restored"
    assert resumed["codec_handoff"]["receipt"]["sha256"]==pin
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==pin
    assert not resumed["audio_or_weights_written"]
    assert "private-train-asset-" not in json.dumps(only)
    assert str(ckpt) not in json.dumps(only)


def test_handoff_can_change_denoiser_budget_not_codec(stage):
    archive,sha,plan,ckpt,pin,_,_=stage
    changed=replace(plan,diffusion_updates=1)
    report=run_archive_latent(archive,expected_sha256=sha,plan=changed,codec_in=ckpt,codec_sha256=pin)
    assert report["denoiser"]["training"]["steps"]==1
    assert report["codec_handoff"]["codec_optimizer_steps_executed"]==0
    with pytest.raises(ArchiveError,match="identity differs"):
        run_archive_latent(archive,expected_sha256=sha,plan=replace(plan,bits=4),
                           codec_in=ckpt,codec_sha256=pin)


def test_wrong_pin_and_no_clobber(stage):
    archive,sha,plan,ckpt,pin,_,_=stage
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        run_archive_latent(archive,expected_sha256=sha,plan=plan,codec_in=ckpt,codec_sha256="0"*64)
    with pytest.raises(ArchiveError,match="already exists"):
        run_archive_latent(archive,expected_sha256=sha,plan=plan,codec_out=ckpt,codec_only=True)
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==pin
    assert set(ckpt.parent.iterdir())=={ckpt,archive}


@pytest.mark.parametrize("change",["calibration","sampler","identity","trained","publication"])
def test_rehashed_inconsistent_metadata_fails(stage,tmp_path,change):
    archive,sha,plan,ckpt,_,_,_=stage
    blob=ckpt.read_bytes()
    magic,n,m=HEADER.unpack_from(blob)
    metadata=json.loads(blob[HEADER.size:HEADER.size+n])
    payload=blob[HEADER.size+n:]
    if change=="calibration": metadata["calibration_sha256"]="0"*64
    elif change=="sampler": metadata["training"]["sampler"]["draws"]+=1
    elif change=="identity": metadata["identity"]["runtime"]["threads"]+=1
    elif change=="trained": metadata["trained_codec_sha256"]="0"*64
    else: metadata["publication_authorized"]=True
    header=canonical_json(metadata)
    changed=HEADER.pack(magic,len(header),m)+header+payload
    path=tmp_path/"tampered.ckpt"
    path.write_bytes(changed)
    with pytest.raises(ArchiveError):
        run_archive_latent(archive,expected_sha256=sha,plan=plan,codec_in=path,
                           codec_sha256=hashlib.sha256(changed).hexdigest())


@pytest.mark.parametrize("variant",["magic","oversized_header","truncated","trailing"])
def test_envelope_bounds(stage,variant):
    blob=stage[3].read_bytes()
    if variant=="magic": blob=b"INVALID!"+blob[8:]
    elif variant=="oversized_header": blob=HEADER.pack(MAGIC,MAX_HEADER+1,1)+blob[HEADER.size:]
    elif variant=="truncated": blob=blob[:-1]
    else: blob+=b"x"
    with pytest.raises(ArchiveError):
        decode_handoff(blob,CropAudioCodec())


@pytest.mark.parametrize("options",[
    {"codec_only":True},{"codec_in":"x.ckpt"},{"codec_sha256":"0"*64},
    {"codec_out":"x.ckpt","codec_in":"y.ckpt","codec_sha256":"0"*64},
])
def test_invalid_options_fail_before_io(options):
    with pytest.raises(ArchiveError,match="options"):
        run_archive_latent("missing.scorpus",expected_sha256="0"*64,plan=small_plan(),**options)
