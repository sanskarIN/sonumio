"""Real paired precision replicates and fail-closed cross-domain comparisons."""
from dataclasses import replace
import copy
import json

import pytest

torch=pytest.importorskip("torch")
from research.crop_latent_study import LatentStudyPlan
from research.crop_latent_replicates import ReplicatePlan,run_replicates,summarize_replicates
from research.crop_codec_packet import CropCodecError


def plan():
    base=LatentStudyPlan(codec_steps=1,diffusion_updates=1,batch_size=2,width=16,
                         schedule_steps=8,reverse_steps=2,training_seeds=(0,),
                         validation_seeds=(101,),generation_seeds=(701,709))
    return ReplicatePlan((17,29),(4,),base)


@pytest.fixture(scope="module")
def completed():
    design=plan()
    before=torch.get_rng_state().clone()
    report=run_replicates(design)
    assert torch.equal(before,torch.get_rng_state())
    return design,report


def test_replicates_pair_training_and_evaluate_waveform_domain(completed):
    design,report=completed
    assert report["training_examples"]==16
    assert report["generation_forward_calls"]==384
    result=report["paired_comparison"]
    assert result["shared_data_and_evaluation_noise_verified"]
    assert result["shared_within_seed_training_traces_verified"]
    assert result["distinct_initializations_across_seeds_verified"]
    assert result["latent_velocity_mse_compared_across_precision_arms"] is False
    summary=result["waveform_domain_seed_summaries"]["q4"]["decoded_x0"]["all"]["waveform_l1"]
    assert summary["seeds"]==2
    for run in report["runs"]:
        a,b=run["arms"]["float"],run["arms"]["q4"]
        assert a["codec"]["initial_sha256"]==b["codec"]["initial_sha256"]
        assert a["denoiser"]["initial_sha256"]==b["denoiser"]["initial_sha256"]
        assert a["validation"]["trace_sha256"]==b["validation"]["trace_sha256"]
        assert a["generation"]["noise_trace_sha256"]==b["generation"]["noise_trace_sha256"]
    assert not report["perceptual_quality_established"]
    assert not report["model_publication_authorized"]
    assert "velocity_mse" not in json.dumps(result["waveform_domain_seed_summaries"])
    json.dumps(report,allow_nan=False)


def test_summary_roundtrip_through_json(completed):
    design,report=completed
    restored=json.loads(json.dumps(report["runs"]))
    assert summarize_replicates(restored,design)==report["paired_comparison"]


@pytest.mark.parametrize("change",["seed","data","noise","init","steps","null","missing","order","config"])
def test_replicate_summary_refuses_incompatible_reports(completed,change):
    design,report=completed
    runs=copy.deepcopy(report["runs"])
    row=runs[1]
    arm=row["arms"]["q4"]
    if change=="seed": row["seed"]=1
    elif change=="data": arm["training_data_sha256"]="0"*64
    elif change=="noise": arm["validation"]["trace_sha256"]="0"*64
    elif change=="init": arm["denoiser"]["initial_sha256"]="0"*64
    elif change=="steps": arm["denoiser"]["training"]["training_trace_sha256"]="0"*64
    elif change=="null": arm["generation"]["rows"][3]["waveform_sha256"]="0"*64
    elif change=="missing": arm["validation"]["rows"].pop()
    elif change=="order": arm["generation"]["rows"].reverse()
    else: arm["configuration"]["codec_steps"]=99
    with pytest.raises(CropCodecError):
        summarize_replicates(runs,design)


def test_plan_caps_and_pins():
    design=plan()
    assert design.fingerprint()==plan().fingerprint()
    assert replace(design,precision_bits=(8,)).fingerprint()!=design.fingerprint()
    for settings in ({"model_seeds":(1,)},{"model_seeds":(1,1)},{"precision_bits":(3,)},
                     {"precision_bits":(4,4)}):
        with pytest.raises(CropCodecError):
            replace(design,**settings)
    large=replace(design.base,codec_steps=500,diffusion_updates=2000,batch_size=16)
    with pytest.raises(CropCodecError):
        ReplicatePlan(tuple(range(8)),(2,4,6,8),large)
