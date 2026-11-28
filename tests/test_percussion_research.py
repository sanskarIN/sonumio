"""Verify stochastic percussion training, feature oracles and isolation."""
from dataclasses import replace
import json

import numpy as np
import pytest

torch=pytest.importorskip("torch")

from research.percussion_data import (
    DRUMS,PANELS,DrumSpec,DrumGrid,DrumExample,DrumCorpus,
    build_corpus,render_source,excitation,collate,schedule,evaluation_batches,
)
from research.percussion_model import initialize,broadband_control,features,feature_errors,objective
from research.percussion_study import StudyPlan,train,run_study,evaluate_panel
from research.weight_ema import model_state_sha256


def small_grid():
    return DrumGrid(training_seeds=(7,),evaluation_seeds=(103,),
                    training_velocities=(500,),evaluation_velocities=(750,),
                    training_frames=(4096,),evaluation_frames=(6144,))


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(small_grid())


def test_source_replay_and_seed_isolation(corpus):
    train=corpus.get("train")
    hashes={row.sha256 for row in train}
    for row in train:
        assert render_source(row.spec)==row
        assert not row.samples.flags.writeable
    for panel in PANELS[1:]:
        rows=corpus.get(panel)
        assert {row.spec.seed for row in rows}=={103}
        assert not hashes&{row.sha256 for row in rows}
        assert {row.spec.drum for row in rows}==set(DRUMS)
    assert build_corpus(small_grid()).report()==corpus.report()


@pytest.mark.parametrize("change",[
    {"evaluation_seeds":(7,)},{"evaluation_velocities":(500,)},
    {"evaluation_frames":(4096,)},{"training_seeds":(7,7)},
    {"training_velocities":(0,)},{"training_frames":(8193,)},
])
def test_invalid_or_overlapping_grid(change):
    with pytest.raises(ValueError):
        replace(small_grid(),**change)


def test_waveform_leakage_is_rejected(corpus):
    panels=list(corpus.panels)
    rows=list(panels[1])
    rows[0]=DrumExample(rows[0].spec,panels[0][0].raw)
    panels[1]=tuple(rows)
    with pytest.raises(ValueError,match="PCM overlaps"):
        DrumCorpus(corpus.grid,tuple(panels))


def test_independent_excitation_has_disjoint_domains_and_replicas():
    spec=DrumSpec("hat",500,4096,7)
    a=excitation(spec,"train",0)
    assert a.shape==(4098,) and a.dtype==np.float32
    assert np.max(np.abs(a))<=1.
    np.testing.assert_array_equal(a,excitation(spec,"train",0))
    assert not np.array_equal(a,excitation(spec,"evaluation",0))
    assert not np.array_equal(a,excitation(spec,"train",1))
    assert not np.array_equal(a,excitation(replace(spec,seed=8),"train",0))


def test_batch_and_schedule_replay(corpus):
    rows=corpus.get("train")
    a,digest=schedule(rows,updates=5,batch_size=2,seed=11)
    b,second=schedule(rows,updates=5,batch_size=2,seed=11)
    assert a==b and digest==second
    seen=tuple(row for batch in evaluation_batches(rows,2) for row in batch)
    assert seen==rows
    batch=collate(rows,domain="train",replica=0)
    assert batch.targets.shape==(3,4096)
    assert batch.excitation.shape==(3,4098)
    assert batch.labels.tolist()==[0,1,2]


def test_synthesis_is_bounded_and_stochastic(corpus):
    rows=corpus.get("train")
    batch=collate(rows,domain="evaluation",replica=0)
    model=initialize(32,17)
    a=model(batch.labels,batch.velocities,batch.frames,batch.excitation)
    b=model(batch.labels,batch.velocities,batch.frames,batch.excitation)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert a.shape==batch.targets.shape
    assert torch.count_nonzero(a[:,0])==0 and torch.count_nonzero(a[:,-1])==0
    assert torch.all(a.abs().amax(dim=1)<=batch.velocities.float()/1000.+1e-6)
    other=collate(rows,domain="evaluation",replica=1)
    c=model(batch.labels,batch.velocities,batch.frames,other.excitation)
    assert not torch.equal(a,c)
    d=model((batch.labels+1)%3,batch.velocities,batch.frames,batch.excitation)
    assert not torch.equal(a,d)


def test_invalid_model_inputs(corpus):
    batch=collate(corpus.get("train"),domain="train",replica=0)
    model=initialize(32,17)
    args=[batch.labels,batch.velocities,batch.frames,batch.excitation]
    for i,value in [(0,torch.tensor([0,1,3])),(1,torch.tensor([0,500,500])),
                    (2,1024),(3,batch.excitation[:,:-1]),(3,batch.excitation*2)]:
        changed=args.copy()
        changed[i]=value
        with pytest.raises(ValueError):
            model(*changed)


def test_feature_oracles_and_phase_invariance():
    t=torch.arange(4096,dtype=torch.float32)/24000
    low=(.3*torch.sin(2*torch.pi*100*t))[None,:]
    high=(.3*torch.sin(2*torch.pi*7000*t))[None,:]
    assert features(low)["centroid"].item()<features(high)["centroid"].item()
    assert all(torch.count_nonzero(x)==0 for x in feature_errors(low,low).values())
    assert all(torch.count_nonzero(x)==0 for x in feature_errors(-low,low).values())
    assert objective(high,low)>0
    assert objective(low*.5,low)>0


def test_actual_gradient_and_label_independent_baseline(corpus):
    batch=collate(corpus.get("train"),domain="train",replica=0)
    model=initialize(32,17)
    predicted=model(batch.labels,batch.velocities,batch.frames,batch.excitation)
    loss=objective(predicted,batch.targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad)>0 for p in model.parameters())
    a=broadband_control(batch.labels,batch.velocities,batch.frames,batch.excitation)
    b=broadband_control((batch.labels+1)%3,batch.velocities,batch.frames,batch.excitation)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_training_and_wrong_label_controls(corpus):
    plan=StudyPlan(grid=corpus.grid,seeds=(17,29),width=32,updates=3,batch_size=2,
                   evaluation_batch_size=3,excitation_replicates=2)
    model=initialize(32,17)
    batches,digest=schedule(corpus.get("train"),updates=3,batch_size=2,seed=plan.sampler_seed)
    before=model_state_sha256(model)
    with pytest.raises(ValueError,match="schedule"):
        train(model,corpus,plan,tuple(reversed(batches)))
    assert before==model_state_sha256(model)
    result=train(model,corpus,plan,batches)
    assert result["trained_sha256"]!=before and result["schedule_sha256"]==digest
    state=model_state_sha256(model)
    wrong=evaluate_panel(corpus.get("joint"),plan,model=model,wrong=True)
    assert wrong["conditions_per_note"]==2 and wrong["excitation_replicates"]==2
    assert set(wrong["by_drum"])==set(DRUMS)
    assert model.training and model_state_sha256(model)==state


def test_replicated_study_executes_all_arms():
    plan=StudyPlan(grid=small_grid(),seeds=(17,29),width=32,updates=3,batch_size=2,
                   evaluation_batch_size=3,excitation_replicates=2)
    result=run_study(plan)
    assert len(result["results"])==2
    assert result["independent_excitation"]
    assert result["source_phase_matching"] is False
    assert result["evaluation_used_for_training"] is False
    for row in result["results"]:
        assert row["training"]["updates"]==3
        assert set(row["evaluation"])=={"broadband","initial","trained","wrong"}
        assert set(row["paired"])==set(PANELS[1:])
    json.dumps(result,allow_nan=False)
