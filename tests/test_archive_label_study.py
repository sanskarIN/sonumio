"""Paired experimental-design, negative-control and multi-seed tests."""
from dataclasses import replace
import copy
import hashlib
import json

import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_study_plan import StudyPlan,corruption_batch,update_trace,parse_seeds
from research.archive_label_study import run_study,summarize_runs,describe
from research.corpus_archive_format import ArchiveError
from research.corpus_archive import CorpusArchiveReader
from research.wave_diffusion import CATEGORIES
from test_archive_training import fixture,build


def small_plan(**kwargs):
    return replace(StudyPlan(seeds=(17,29),steps=2,batch_size=3,channels=8,
                             diffusion_steps=8,assay_repeats=1,assay_batch_size=8),**kwargs)


@pytest.mark.parametrize("changes",[
    {"seeds":(1,)},{"seeds":(1,1)},{"seeds":(True,2)},{"seeds":[1,2]},
    {"seeds":(1,2**32)},{"steps":0},{"batch_size":33},{"channels":9},
    {"drop_condition":float("nan")},{"drop_condition":True},{"drop_condition":.6},
    {"diffusion_profile":"unknown"},{"assay_repeats":0},{"assay_batch_size":17},
    {"steps":2000,"batch_size":32},
])
def test_plan_bounds(changes):
    with pytest.raises(ArchiveError):
        small_plan(**changes)


@pytest.mark.parametrize("raw",["","01,2","1, 2","-1,2","1.0,2","1,,2","1,２"])
def test_seed_parser_rejects_noncanonical(raw):
    with pytest.raises(ArchiveError):
        parse_seeds(raw)


def test_workload_and_plan_fingerprint():
    plan=small_plan()
    assert plan.train_examples==24
    assert plan.assay_predictions(10)==2*3*10*3*7
    assert plan.fingerprint("a"*64,"b"*64,"c"*64)==plan.fingerprint("a"*64,"b"*64,"c"*64)
    assert plan.fingerprint("a"*64,"b"*64,"c"*64)!=replace(plan,steps=3).fingerprint("a"*64,"b"*64,"c"*64)
    with pytest.raises(ArchiveError):
        replace(plan,seeds=tuple(range(8)),assay_repeats=4).assay_predictions(4096)


def test_corruption_shared_and_domain_separated():
    plan=small_plan(batch_size=32)
    labels=torch.arange(32)%6
    before=torch.get_rng_state().clone()
    first=corruption_batch(plan,17,0,labels)
    second=corruption_batch(plan,17,0,labels)
    changed=corruption_batch(plan,17,0,(labels+1)%6)
    for field in ("noise","steps","dropped","random_label"):
        assert torch.equal(getattr(first,field),getattr(second,field))
        assert torch.equal(getattr(first,field),getattr(changed,field))
    assert torch.equal(first.matched[~first.dropped],labels[~first.dropped])
    assert (first.matched[first.dropped]==6).all()
    assert (first.random_label[first.dropped]==6).all()
    assert not torch.equal(first.noise,corruption_batch(plan,29,0,labels).noise)
    assert not torch.equal(first.noise,corruption_batch(plan,17,1,labels).noise)
    assert torch.equal(before,torch.get_rng_state())
    a,b=hashlib.sha256(),hashlib.sha256()
    update_trace(a,list(range(32)),first)
    update_trace(b,list(range(32)),second)
    assert a.hexdigest()==b.hexdigest()


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    path=tmp_path_factory.mktemp("paired-archive")/"fixture.scorpus"
    result=build(path,fixture())
    plan=small_plan()
    before=torch.get_rng_state().clone()
    report=run_study(path,expected_sha256=result.archive_sha256,plan=plan)
    assert torch.equal(before,torch.get_rng_state())
    return path,result.archive_sha256,plan,report


def test_paired_study_controls_and_seed_summary(completed):
    path,sha,plan,report=completed
    assert report["archive_sha256"]==sha
    assert report["training_examples_across_arms"]==24
    assert len(report["runs"])==2
    assert not report["test_split_scored"]
    assert not report["weights_or_audio_files_written"]
    assert not report["statistical_significance_claimed"]
    hashes=[]
    traces=[]
    for run in report["runs"]:
        assert run["shared_null_examples"]+run["non_null_examples"]==6
        assert sum(run["true_non_null_label_counts"].values())==run["non_null_examples"]
        assert sum(run["random_non_null_label_counts"].values())==run["non_null_examples"]
        assert run["random_label_accidental_matches"]<=run["non_null_examples"]
        init=run["initialization"]
        hashes.append(init["model_sha256"])
        traces.append(init["noise_trace_sha256"])
        for arm in run["arms"].values():
            assert arm["initial_model_sha256"]==init["model_sha256"]
            assert arm["trained_model_sha256"]!=init["model_sha256"]
            assert arm["assay"]["noise_trace_sha256"]==init["noise_trace_sha256"]
            assert arm["completed_steps"]==2
    assert hashes[0]!=hashes[1]
    assert traces[0]==traces[1]
    values=[run["arms"]["matched"]["assay"]["group_macro"]["matched_mse"]-
            run["arms"]["random_label"]["assay"]["group_macro"]["matched_mse"]
            for run in report["runs"]]
    assert report["paired_seed_summaries"]["group_macro"]["matched_mse"]["matched_minus_random_label"]==describe(values)
    assert not list(path.parent.glob("*.stcheckpoint"))
    assert "private-train-asset-" not in json.dumps(report)


def test_study_reproducible_and_fetches_each_training_batch_once(completed,monkeypatch):
    path,sha,plan,previous=completed
    real=CorpusArchiveReader.read_train_batch
    calls=[]
    def track(reader,indices):
        calls.append(len(indices))
        return real(reader,indices)
    monkeypatch.setattr(CorpusArchiveReader,"read_train_batch",track)
    again=run_study(path,expected_sha256=sha,plan=plan)
    assert calls==[3]*4
    assert again["runs"]==previous["runs"]
    assert again["paired_seed_summaries"]==previous["paired_seed_summaries"]


@pytest.mark.parametrize("change",["seed","trace","model","step","coverage","profile"])
def test_summary_rejects_incompatible_runs(completed,change):
    _,_,plan,report=completed
    runs=copy.deepcopy(report["runs"])
    row=runs[1]
    arm=row["arms"]["matched"]
    if change=="seed": row["seed"]=False
    elif change=="trace": arm["assay"]["noise_trace_sha256"]="0"*64
    elif change=="model": arm["initial_model_sha256"]="0"*64
    elif change=="step": arm["completed_steps"]+=1
    elif change=="coverage": arm["assay"]["clip_weighted"]["cases"]+=1
    elif change=="profile": arm["assay"]["diffusion_profile"]="linear_beta_v1"
    with pytest.raises(ArchiveError):
        summarize_runs(runs,plan)


def test_wrong_archive_pin_fails_before_training(completed,monkeypatch):
    path,_,plan,_=completed
    def fail(*args,**kwargs):
        pytest.fail("unverified archive must not train")
    monkeypatch.setattr("research.archive_label_study._single_run",fail)
    with pytest.raises(ArchiveError,match="SHA256 mismatch"):
        run_study(path,expected_sha256="0"*64,plan=plan)


def test_descriptive_summary_not_inference():
    result=describe([-1.,0.,2.])
    assert result["mean"]==pytest.approx(1/3)
    assert result["positive"]==result["negative"]==result["zero"]==1
    with pytest.raises(ArchiveError):
        describe([0.,float("nan")])
