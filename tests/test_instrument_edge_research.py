"""Sustained instrument edge coordinate and factorial protocol regression tests."""
from dataclasses import replace
import json

import pytest

torch=pytest.importorskip("torch")

from research.instrument_edge_data import DISTANCES,build_schedule,describe_schedule
from research.instrument_edge_model import (
    ATTACK_LIMITS,RELEASE_LIMITS,absolute_edge_frames,initialize_edges,
    envelope_error,edge_training_loss,waveform_diagnostics,edge_parameters,
)
from research.instrument_edge_study import (
    ARMS,PRIMARY,StudyPlan,evaluate_note,evaluate_panel,run_study,
)
from research.instrument_long_data import LongGrid,build_corpus,region_span,REGIONS
from research.instrument_long_model import render_complete
from research.weight_ema import model_state_sha256


@pytest.fixture(scope="module")
def grid():
    return LongGrid(training_pitches=(60,),evaluation_pitches=(66,),
                    training_velocities=(500,),evaluation_velocities=(750,),
                    training_frames=(12000,),evaluation_frames=(16000,))


@pytest.fixture(scope="module")
def corpus(grid):
    return build_corpus(grid)


def controls():
    return (torch.tensor([0,1,2,3,4],dtype=torch.long),
            torch.tensor([48,54,60,66,72],dtype=torch.long),
            torch.tensor([350,500,600,750,900],dtype=torch.long))


@pytest.mark.parametrize("limits",[ATTACK_LIMITS,RELEASE_LIMITS])
def test_absolute_edge_mapping_is_monotonic_bounded_and_differentiable(limits):
    unit=torch.tensor([.0005,.125,.25,.375,.4995],requires_grad=True)
    out=absolute_edge_frames(unit,96000,limits)
    assert out.shape==unit.shape
    assert out[0].item()==pytest.approx(limits[0])
    assert out[-1].item()==pytest.approx(limits[1],rel=1e-5)
    assert bool((out[1:]>out[:-1]).all())
    out.sum().backward()
    assert bool(torch.isfinite(unit.grad).all())
    # The boundary clamp may legitimately saturate at either endpoint.
    # Strictly positive gradients are required throughout the interior.
    assert bool((unit.grad[1:-1]>0).all())
    assert bool((unit.grad[[0,-1]]>=0).all())
    short=absolute_edge_frames(unit.detach(),2048,limits)
    assert float(short.max())<1024


@pytest.mark.parametrize("fractions,frames",[
    (torch.tensor([float("nan")]),12000),(torch.tensor([0.]),12000),
    (torch.tensor([.6]),12000),(torch.tensor([.2],dtype=torch.float64),12000),
    (torch.tensor([.2]),1024),(torch.tensor([.2]),96001),
])
def test_absolute_edge_rejects_bad_inputs(fractions,frames):
    with pytest.raises(ValueError):
        absolute_edge_frames(fractions,frames,ATTACK_LIMITS)


def test_factorial_coordinates_share_exact_trainable_state():
    initial=torch.random.get_rng_state().clone()
    a=initialize_edges(32,17,"fraction")
    b=initialize_edges(32,17,"absolute")
    assert torch.equal(torch.random.get_rng_state(),initial)
    assert model_state_sha256(a)==model_state_sha256(b)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert not torch.equal(a.render_window(*controls(),12000,0,2048),
                           b.render_window(*controls(),12000,0,2048))


@pytest.mark.parametrize("coordinate",["fraction","absolute"])
def test_window_partition_replays_at_actual_note_edges(coordinate):
    model=initialize_edges(32,17,coordinate)
    args=tuple(x[:1] for x in controls())
    full=torch.cat([x for _,x in render_complete(model,*args,24000,block=4096)],dim=1)
    split=torch.cat([x for _,x in render_complete(model,*args,24000,block=997)],dim=1)
    torch.testing.assert_close(full,split,rtol=0,atol=1e-6)
    assert float(full[0,0])==0 and float(full[0,-1])==0
    assert float(full.abs().max())<=1.000001
    params=edge_parameters(model,*args,24000)
    assert 0<float(params["attack_frames"][0])<12000
    assert 0<float(params["release_frames"][0])<12000


def test_edge_distance_schedule_is_deterministic_and_covers_multiple_offsets(corpus):
    rows=corpus.get("train")
    a=build_schedule(rows,updates=24,batch_size=2,seed=17)
    b=build_schedule(rows,updates=24,batch_size=2,seed=17)
    assert a==b and a.sha256==b.sha256
    assert a.sha256!=build_schedule(rows,updates=24,batch_size=2,seed=29).sha256
    report=describe_schedule(a,rows)
    for region in REGIONS:
        assert report["by_region"][region]["batches"]==8
        assert report["by_region"][region]["samples"]==8*2*2048
    for region in ("attack","release"):
        assert 0 in report["by_region"][region]["distances"]
        assert len(report["by_region"][region]["distances"])>1
    for batch,distance in zip(a.batches,a.distances,strict=True):
        lo,hi=region_span(batch.frames,batch.region)
        assert lo<=batch.start<=hi-batch.count
        if batch.region=="attack":
            assert batch.start-lo==distance
        if batch.region=="release":
            assert hi-batch.count-batch.start==distance
        assert all(rows[i].note.frames==batch.frames for i in batch.indices)


def test_envelope_error_is_phase_insensitive_and_handles_partial_blocks():
    t=torch.arange(2051,dtype=torch.float32)/24000
    x=(.4*torch.sin(2*torch.pi*440*t))[None,:]
    assert float(envelope_error(x,x)[0])==0
    assert float(envelope_error(-x,x)[0])==0
    assert float(envelope_error(.5*x,x)[0])>0
    silent=torch.zeros_like(x)
    assert float(envelope_error(silent,silent)[0])==0
    assert bool(torch.isfinite(envelope_error(x,silent)).all())
    pred=(x*.7).detach().requires_grad_()
    envelope_error(pred,x).mean().backward()
    assert bool(torch.isfinite(pred.grad).all())
    assert bool((pred.grad!=0).any())


@pytest.mark.parametrize("coordinate,objective",ARMS)
def test_each_factorial_arm_has_finite_gradients(coordinate,objective):
    model=initialize_edges(32,17,coordinate)
    args=tuple(x[:2] for x in controls())
    target=torch.sin(torch.arange(2048,dtype=torch.float32)[None,:]*.1).repeat(2,1)*.2
    predicted=model.render_window(*args,12000,9952,2048)
    loss=edge_training_loss(predicted,target,objective)
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())


def test_near_silent_region_diagnostics_are_finite():
    x=torch.zeros((1,2048),dtype=torch.float32)
    y=torch.full_like(x,.001)
    metrics=waveform_diagnostics(y,x)
    assert all(bool(torch.isfinite(v).all()) for v in metrics.values())
    assert float(metrics["absolute_wave_mse"][0])==pytest.approx(1e-6)
    assert float(metrics["reference_rms"][0])==0


def test_full_note_evaluation_and_mode_preservation(corpus):
    model=initialize_edges(32,17,"absolute")
    model.train()
    state=model_state_sha256(model)
    row=corpus.get("joint")[0]
    report=evaluate_note(row,model)
    assert set(report["regions"])=={"full","attack","sustain","release","first2048","last2048"}
    assert report["parameters"]["release_frames"]>0
    panel=evaluate_panel(corpus.get("joint"),model)
    assert panel["notes"]==5
    assert model.training and model_state_sha256(model)==state
    json.dumps(panel,allow_nan=False)


def test_prescribed_small_factorial_experiment_really_updates_all_arms(grid):
    plan=StudyPlan(grid=grid,seeds=(17,29),width=32,updates=3,batch_size=2,panels=("joint",))
    result=run_study(plan)
    assert len(result["replicates"])==2
    for replica in result["replicates"]:
        assert set(replica["training"])=={a+"-"+b for a,b in ARMS}
        for training in replica["training"].values():
            assert training["updates"]==3
            assert training["initial_sha256"]==replica["initial_sha256"]
            assert training["trained_sha256"]!=training["initial_sha256"]
            assert training["schedule_sha256"]==result["schedule"]["sha256"]
            assert training["trainable_parameters"]==replica["trainable_parameters"]
        assert replica["wrong_primary"]["joint"]["wrong_labels_per_note"]==4
    assert "baseline_minus_primary" in result["summary"]["joint"]["release"]["envelope_error"]
    assert result["evaluation_used_for_training"] is False
    assert result["weights_published"] is False
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("changes",[
    {"seeds":(17,)},{"seeds":(17,17)},{"width":24},{"updates":0},
    {"panels":("train",)},{"panels":("joint","duration")},
])
def test_invalid_study_configuration(grid,changes):
    with pytest.raises(ValueError):
        replace(StudyPlan(grid=grid),**changes)
