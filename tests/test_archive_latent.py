"""Private archive latent training and split-isolated aggregate diagnostics."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_latent_access import AuditedCropAccess,counter_generator
from research.archive_latent_study import ArchiveLatentPlan,run_archive_latent,main
from research.archive_readiness import audit_archive
from research.archive_sampler import CounterGroupSampler
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError
from test_archive_training import fixture,build


def small_plan(**kwargs):
    return replace(ArchiveLatentPlan(codec_steps=1,diffusion_updates=2,batch_size=2,width=16,
                                    schedule_steps=8,reverse_steps=2,validation_repeats=1),**kwargs)


@pytest.fixture
def archive(tmp_path):
    path=tmp_path/"fixture.scorpus"
    result=build(path,fixture())
    return path,result.archive_sha256


def test_bounded_access_audited_indices_and_scan(archive):
    path,sha=archive
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        access=AuditedCropAccess(reader,ready)
        assert access.count("train")==len(ready.indices("train"))
        sampler=CounterGroupSampler(ready,seed=17)
        indices,wave,labels=access.sample(sampler,4)
        assert wave.shape==(4,1,2048)
        assert labels.shape==(4,)
        assert set(indices.tolist())<=set(ready.indices("train"))
        scanned=[]
        for rows,batch in access.scan("validation",3):
            assert 1<=len(batch)<=3
            scanned.extend(row.index for row in rows)
        assert tuple(scanned)==ready.indices("validation")
        with pytest.raises(ArchiveError):
            list(access.scan("test",2))
        with pytest.raises(ArchiveError):
            access.sample(sampler,17)


def test_eligibility_tampering_rejected(archive):
    path,sha=archive
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        changed=replace(ready,eligible={**ready.eligible,"train":ready.eligible["train"][1:]})
        with pytest.raises(ArchiveError,match="fingerprint changed"):
            AuditedCropAccess(reader,changed)


def test_counter_rng_domains_and_no_global_state():
    before=torch.get_rng_state().clone()
    a=torch.randn(16,generator=counter_generator(7,b"noise",0))
    b=torch.randn(16,generator=counter_generator(7,b"noise",0))
    c=torch.randn(16,generator=counter_generator(7,b"time",0))
    assert torch.equal(a,b) and not torch.equal(a,c)
    assert torch.equal(before,torch.get_rng_state())


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path=tmp_path_factory.mktemp("private-archive-latent")/"fixture.scorpus"
    result=build(path,fixture())
    plan=small_plan()
    before=torch.get_rng_state().clone()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    report=run_archive_latent(path,expected_sha256=result.archive_sha256,plan=plan)
    assert torch.equal(before,torch.get_rng_state())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    return path,result.archive_sha256,plan,report


def test_actual_two_stage_training_and_aggregate_validation(completed):
    path,sha,plan,report=completed
    assert report["archive_sha256"]==sha
    assert report["training_examples"]==6
    assert report["codec"]["training"]["sampler"]["draws"]==2
    assert report["denoiser"]["training"]["sampler"]["draws"]==4
    assert report["codec"]["initial_sha256"]!=report["codec"]["trained_sha256"]
    assert report["denoiser"]["initial_sha256"]!=report["denoiser"]["trained_sha256"]
    assert report["codec"]["calibration"]["training_clips"]==report["eligible_training_clips"]
    assert report["codec"]["frozen_during_diffusion"]
    assert report["calibration_train_only"] and not report["latent_corpus_materialized"]
    validation=report["validation"]
    assert validation["clip_weighted"]["cases"]==report["eligible_validation_clips"]*3
    assert validation["group_macro"]["recording_groups"]==6
    assert set(validation["by_timestep"])=={"early","middle","late"}
    assert report["validation_prediction_rows"]==validation["clip_weighted"]["cases"]*7
    assert report["generation"]["null_control_verified"]
    assert len(report["generation"]["rows"])==36
    assert not report["test_split_scored"]
    assert not report["audio_or_weights_written"]
    assert not report["perceptual_quality_established"]
    assert set(path.parent.iterdir())=={path}
    encoded=json.dumps(report,allow_nan=False)
    for split in ("train","validation","test"):
        assert f"private-{split}-asset-" not in encoded
        assert f"private-{split}-group-" not in encoded
    assert str(path) not in encoded


def test_same_runtime_replay_and_only_training_optimizer_batches(completed,monkeypatch):
    path,sha,plan,previous=completed
    calls=[]
    original=CorpusArchiveReader.read_train_batch
    def watched(reader,indices):
        calls.append(tuple(indices))
        return original(reader,indices)
    monkeypatch.setattr(CorpusArchiveReader,"read_train_batch",watched)
    actual=run_archive_latent(path,expected_sha256=sha,plan=plan)
    assert len(calls)==plan.codec_steps+plan.diffusion_updates
    assert all(len(batch)==plan.batch_size for batch in calls)
    assert actual==previous


def test_validation_scan_batch_partition_invariance(completed):
    path,sha,plan,previous=completed
    actual=run_archive_latent(path,expected_sha256=sha,plan=replace(plan,validation_batch_size=1))
    assert actual["validation"]==previous["validation"]
    assert actual["codec"]==previous["codec"]
    assert actual["denoiser"]==previous["denoiser"]
    assert actual["generation"]==previous["generation"]


def test_wrong_external_pin_stops_training(completed,monkeypatch):
    path,_,plan,_=completed
    def forbidden(*args,**kwargs):
        pytest.fail("Unverified archive must not reach training")
    monkeypatch.setattr("research.archive_latent_study.train_codec",forbidden)
    with pytest.raises(ArchiveError,match="SHA256 mismatch"):
        run_archive_latent(path,expected_sha256="0"*64,plan=plan)


@pytest.mark.parametrize("changes",[
    {"seed":True},{"codec_steps":0},{"diffusion_updates":0},{"bits":3},
    {"width":8},{"batch_size":17},{"reverse_steps":9},{"validation_repeats":5},
    {"generation_seeds":(1,1)},{"codec_steps":500,"diffusion_updates":2000,"batch_size":16,
                               "generation_seeds":tuple(range(8)),"schedule_steps":64,"reverse_steps":64},
])
def test_plan_rejects_invalid_workloads(changes):
    with pytest.raises(ValueError):
        small_plan(**changes)


def test_cli_aggregate_report(completed,capsys):
    path,sha,_,_=completed
    assert main([str(path),"--expected-sha256",sha,"--codec-steps","1","--diffusion-updates","1",
                 "--batch-size","2","--width","16","--schedule-steps","8","--reverse-steps","2",
                 "--validation-repeats","1"])==0
    output=capsys.readouterr().out
    report=json.loads(output)
    assert report["protocol"]=="sonumio-private-archive-latent-study-v1"
    assert str(path) not in output
    assert not report["audio_or_weights_written"]


def test_validation_budget_preflight():
    plan=small_plan(validation_repeats=4)
    with pytest.raises(ArchiveError,match="validation workload"):
        plan.validation_predictions(4096)
    assert plan.validation_predictions(100)==8400


def test_generation_budget_preflight():
    with pytest.raises(ArchiveError,match="workload exceeds budget"):
        small_plan(generation_seeds=tuple(range(8)),schedule_steps=64,reverse_steps=64)
