"""Integration tests for multi-seed paired generated-signal research."""
import copy
import json
from dataclasses import replace

import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_generated_assay import GenerationConfig
from research.archive_generated_study import preflight_generation,summarize_generated_study,_summarize_seed_values
from research.archive_label_study import run_study
from research.corpus_archive_format import ArchiveError
from test_archive_training import fixture,build
from test_archive_label_study import small_plan


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    path=tmp_path_factory.mktemp("generated-study")/"fixture.scorpus"
    built=build(path,fixture())
    plan=small_plan(steps=1)
    config=GenerationConfig(seeds=(701,709),reverse_steps=2,guidances=(0.,1.))
    rng=torch.get_rng_state().clone()
    report=run_study(path,expected_sha256=built.archive_sha256,plan=plan,generation=config)
    assert torch.equal(rng,torch.get_rng_state())
    return path,built.archive_sha256,plan,config,report


def test_integrated_study_uses_identical_generation_reference_and_noise(result):
    path,sha,plan,config,report=result
    study=report["generated_signal_study"]
    assert report["generation_forward_calls"]==preflight_generation(plan,config)==288
    assert study["total_forward_calls"]==288
    assert study["reference_panel_and_noise_trace_shared"]
    references=[]
    traces=[]
    for run in report["runs"]:
        reports=[run["initial_generation"],*[arm["generation"] for arm in run["arms"].values()]]
        for item in reports:
            references.append(item["reference_sha256"])
            traces.append(item["initial_noise_trace_sha256"])
            assert not item["perceptual_quality_established"]
            assert len(item["cases"])==24
    assert len(set(references))==len(set(traces))==1
    summary=study["paired_seed_summaries"]["matched_minus_random_label"]["1.0/all"]["strict_target_win"]
    assert summary["declared_training_seeds"]==summary["defined_training_seeds"]==2
    assert summary["all_seed_values_defined"]
    assert study["statistical_significance_claimed"] is False
    assert not list(path.parent.glob("*.wav"))
    assert "private-train-asset-" not in json.dumps(report)


def test_generation_study_no_posthoc_missing_value_imputation():
    empty=_summarize_seed_values([None,None])
    assert empty["mean"] is None and empty["sample_stddev"] is None
    assert empty["missing_training_seeds"]==2
    partial=_summarize_seed_values([None,1.])
    assert partial["mean"]==1 and partial["sample_stddev"] is None
    assert not partial["all_seed_values_defined"]
    assert partial["missing_training_seeds"]==1


@pytest.mark.parametrize("change",["noise","reference","model","case_order","null","missing"])
def test_generated_study_rejects_unpaired_panels(result,change):
    _,_,plan,config,report=result
    runs=copy.deepcopy(report["runs"])
    arm=runs[1]["arms"]["matched"]["generation"]
    if change=="noise": arm["initial_noise_trace_sha256"]="0"*64
    elif change=="reference": arm["reference_sha256"]="0"*64
    elif change=="model": arm["model_sha256"]="0"*64
    elif change=="case_order": arm["cases"][0],arm["cases"][1]=arm["cases"][1],arm["cases"][0]
    elif change=="null": arm["null_output_sha256_by_seed"][0]="0"*64
    else: arm["cases"].pop()
    with pytest.raises(ArchiveError):
        summarize_generated_study(runs,plan,config)


def test_budget_and_schedule_preflight():
    with pytest.raises(ArchiveError,match="exceeds diffusion schedule"):
        preflight_generation(small_plan(),GenerationConfig(reverse_steps=9))
    plan=small_plan(seeds=tuple(range(8)),diffusion_steps=64)
    config=GenerationConfig(seeds=tuple(range(8)),reverse_steps=64,guidances=(0.,1.,2.,3.))
    with pytest.raises(ArchiveError,match="forward-call budget"):
        preflight_generation(plan,config)
