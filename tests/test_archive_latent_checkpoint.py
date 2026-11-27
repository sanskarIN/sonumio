"""Exact private latent continuation, tampering and output policy."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_latent_study import run_archive_latent,main
from research.archive_latent_checkpoint import tensor_specs,validate_losses
from research.archive_checkpoint_format import decode_checkpoint,encode_checkpoint
from research.crop_latent import CropLatentDenoiser
from research.corpus_archive_format import ArchiveError
from test_archive_latent import small_plan
from test_archive_training import fixture,build


@pytest.fixture(scope="module")
def checkpoint_case(tmp_path_factory):
    directory=tmp_path_factory.mktemp("latent-optimizer")
    archive=directory/"fixture.scorpus"
    sha=build(archive,fixture()).archive_sha256
    plan=small_plan(diffusion_updates=4)
    ckpt=directory/"step2.stcheckpoint"
    before=torch.get_rng_state().clone()
    prefix=run_archive_latent(archive,expected_sha256=sha,
          plan=replace(plan,diffusion_updates=2),latent_out=ckpt)
    reference=run_archive_latent(archive,expected_sha256=sha,plan=plan)
    assert torch.equal(before,torch.get_rng_state())
    receipt=prefix["latent_checkpoint"]["receipt"]
    assert receipt["completed_steps"]==2
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==receipt["sha256"]
    return archive,sha,plan,ckpt,receipt["sha256"],reference


def test_exact_optimizer_continuation(checkpoint_case,tmp_path):
    archive,sha,plan,ckpt,pin,reference=checkpoint_case
    final=tmp_path/"step4.stcheckpoint"
    before=torch.get_rng_state().clone()
    actual=run_archive_latent(archive,expected_sha256=sha,plan=plan,
                              latent_in=ckpt,latent_sha256=pin,latent_out=final)
    assert torch.equal(before,torch.get_rng_state())
    for key in ("codec","denoiser","validation","generation"):
        assert actual[key]==reference[key]
    assert actual["latent_checkpoint"]["restored_steps"]==2
    assert actual["latent_checkpoint"]["executed_steps"]==2
    assert actual["latent_checkpoint"]["receipt"]["completed_steps"]==4
    assert actual["audio_or_weights_written"]
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==pin
    assert set(tmp_path.iterdir())=={final}
    assert str(final) not in json.dumps(actual)


def test_same_step_restore_has_no_optimizer_updates(checkpoint_case):
    archive,sha,plan,ckpt,pin,_=checkpoint_case
    actual=run_archive_latent(archive,expected_sha256=sha,
          plan=replace(plan,diffusion_updates=2),latent_in=ckpt,latent_sha256=pin)
    assert actual["latent_checkpoint"]["executed_steps"]==0
    assert not actual["audio_or_weights_written"]


@pytest.mark.parametrize("change",["width","bits","seed","calibration_mode","schedule_steps"])
def test_incompatible_training_trajectory(checkpoint_case,change):
    archive,sha,plan,ckpt,pin,_=checkpoint_case
    values={"width":32,"bits":4,"seed":18,"calibration_mode":"clip_weighted","schedule_steps":16}
    with pytest.raises(ArchiveError):
        run_archive_latent(archive,expected_sha256=sha,plan=replace(plan,**{change:values[change]}),
                           latent_in=ckpt,latent_sha256=pin)


def test_bad_digest_and_no_clobber(checkpoint_case):
    archive,sha,plan,ckpt,pin,_=checkpoint_case
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        run_archive_latent(archive,expected_sha256=sha,plan=plan,latent_in=ckpt,latent_sha256="0"*64)
    with pytest.raises(ArchiveError,match="already exists"):
        run_archive_latent(archive,expected_sha256=sha,plan=plan,latent_out=ckpt)
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==pin


@pytest.mark.parametrize("change",["step","second_moment","trace","losses","overflow_loss","runtime","model","publication"])
def test_rehashed_inconsistent_checkpoint_rejected(checkpoint_case,tmp_path,change):
    archive,sha,plan,ckpt,_,_=checkpoint_case
    with torch.random.fork_rng(devices=[]):
        specs=tensor_specs(CropLatentDenoiser(2048,plan.width))
    metadata,arrays=decode_checkpoint(ckpt.read_bytes(),specs)
    if change=="step":
        key=next(k for k in arrays if k.endswith("/step"))
        arrays[key]=np.array(1,dtype=np.float32)
    elif change=="second_moment":
        key=next(k for k in arrays if k.endswith("/exp_avg_sq"))
        arrays[key]=-np.ones_like(arrays[key])
    elif change=="trace": metadata["progress"]["trace_sha256"]="0"*64
    elif change=="losses": metadata["progress"]["losses"]=[]
    elif change=="overflow_loss": metadata["progress"]["losses"][0]=10**400
    elif change=="runtime": metadata["identity"]["runtime"]["num_threads"]+=1
    elif change=="model": metadata["trained_model_sha256"]="0"*64
    else: metadata["publication_authorized"]=True
    blob=encode_checkpoint(metadata,arrays,specs)
    path=tmp_path/"bad.stcheckpoint"
    path.write_bytes(blob)
    with pytest.raises(ArchiveError):
        run_archive_latent(archive,expected_sha256=sha,plan=plan,latent_in=path,
                           latent_sha256=hashlib.sha256(blob).hexdigest())


def test_checkpoint_can_use_codec_handoff(checkpoint_case,tmp_path):
    archive,sha,plan,ckpt,pin,reference=checkpoint_case
    codec=tmp_path/"codec.ckpt"
    first=run_archive_latent(archive,expected_sha256=sha,plan=plan,
                             codec_out=codec,codec_only=True)
    actual=run_archive_latent(archive,expected_sha256=sha,plan=plan,codec_in=codec,
         codec_sha256=first["codec_handoff"]["receipt"]["sha256"],
         latent_in=ckpt,latent_sha256=pin)
    assert actual["codec_handoff"]["codec_optimizer_steps_executed"]==0
    assert actual["denoiser"]==reference["denoiser"]
    assert actual["validation"]==reference["validation"]


@pytest.mark.parametrize("options",[
    {"latent_in":"x.stcheckpoint"},{"latent_sha256":"0"*64},
    {"codec_only":True,"codec_out":"x.ckpt","latent_out":"y.stcheckpoint"},
])
def test_invalid_options_before_archive_access(options):
    with pytest.raises(ArchiveError,match="latent checkpoint options"):
        run_archive_latent("missing.scorpus",expected_sha256="0"*64,plan=small_plan(),**options)


@pytest.mark.parametrize("losses", [[], [True], [float("nan")], [float("inf")], [-1], [10**400], [0]*5])
def test_loss_history_rejects_invalid_numeric_values(losses):
    with pytest.raises(ArchiveError,match="Invalid latent checkpoint loss history"):
        validate_losses(losses,small_plan(diffusion_updates=4))


def test_loss_history_accepts_finite_nonnegative_values():
    validate_losses([0,0.5,1e-20,1e200],small_plan(diffusion_updates=4))
