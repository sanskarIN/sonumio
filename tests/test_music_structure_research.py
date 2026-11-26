"""Joint bar structure: vocabulary, causal factorization and sampling tests."""
from dataclasses import replace
import json
import math
import numpy as np
import pytest

torch=pytest.importorskip("torch")

from sonumio.arrangement import ArrangementPlan, compose_arrangement
from sonumio.music_analysis import analyze_score
from research.music_structure_data import (
    BarSequence, PATTERNS, CHORD_COUNT, RHYTHM_COUNT, CHORD_BOS, RHYTHM_BOS,
    extract, schedule, fingerprint, allowed_chords, allowed_rhythms,
    corpus, split_corpus, reconstruct,
)
from research.music_structure_model import (
    StructureGRU, CHORD_PAD, RHYTHM_PAD, collate, joint_loss, sample_structure,
)
from research.music_structure_baselines import StructureCounts, evaluate_counts, BASELINES
from research.music_structure_study import StructurePlan, likelihood, generated_metrics, run_study
from research.weight_ema import model_state_sha256


def example(bars=8,style="cinematic"):
    plan=ArrangementPlan(style=style,bars=bars,bpm=120,seed=17)
    return plan,extract(compose_arrangement(plan))


@pytest.mark.parametrize("style",["calm","cinematic","rhythmic"])
@pytest.mark.parametrize("bars",[4,6,7,8,12,16])
def test_original_structure_extract_and_reconstruct(style,bars):
    plan,seq=example(bars,style)
    assert seq.contexts==schedule(plan)
    assert len(seq.chords)==len(seq.rhythms)==bars
    assert seq.chords[-2:]==(4,0)
    assert seq.rhythms[-1]!=0
    score=reconstruct(plan,seq)
    assert extract(score)==seq
    assert analyze_score(score)["final_lead_tonic"]
    stats=analyze_score(score)
    assert stats["strong_beat_chord_hits"]==stats["strong_beat_notes"]
    for event in score.events:
        assert event.tick+event.length<=plan.ticks


def test_all_rhythm_tokens_render_and_final_silence_rejected():
    plan,seq=example(8)
    rhythms=tuple(range(8))
    changed=BarSequence(seq.chords,rhythms,seq.contexts)
    assert extract(reconstruct(plan,changed))==changed
    score=reconstruct(plan,changed)
    assert not any(e.track=="lead" and e.tick<32 for e in score.events)
    with pytest.raises(ValueError):
        reconstruct(plan,replace(changed,rhythms=rhythms[:-1]+(0,)))
    with pytest.raises(ValueError):
        reconstruct(plan,replace(changed,chords=seq.chords[:-1]+(2,)))


def test_structure_digests_and_novel_partition():
    a,b=corpus((1,2),(6,8))
    c,d=corpus((1,2),(6,8))
    assert a==c and b==d
    train,dev,novel,receipt=split_corpus((0,1,2),(101,102),(6,8))
    seen={fingerprint(s,tokens_only=True) for _,s in train}
    assert all(fingerprint(s,tokens_only=True) not in seen for _,s in novel)
    assert receipt["development_exact_token_overlap_rows"]+len(novel)==len(dev)
    assert receipt["novel_development_rows"]==len(novel)
    assert not receipt["source_seed_overlap"]
    with pytest.raises(ValueError):
        split_corpus((0,1),(1,2))


def test_padded_joint_history_does_not_leak_targets():
    _,a=example(4)
    _,b=example(8)
    pc,pr,ctx,tc,tr=collate([a,b])
    for i,seq in enumerate((a,b)):
        n=len(seq.chords)
        assert pc[i,0]==CHORD_BOS and pr[i,0]==RHYTHM_BOS
        assert pc[i,1:n].tolist()==list(seq.chords[:-1])
        assert pr[i,1:n].tolist()==list(seq.rhythms[:-1])
        assert tc[i,:n].tolist()==list(seq.chords)
        assert tr[i,:n].tolist()==list(seq.rhythms)
        assert (pc[i,n:]==CHORD_PAD).all()
        assert (pr[i,n:]==RHYTHM_PAD).all()
        assert (tc[i,n:]==-100).all() and (tr[i,n:]==-100).all()


def test_chord_head_is_independent_of_current_target_chord():
    _,seq=example()
    pc,pr,ctx,tc,tr=collate([seq])
    model=StructureGRU(32).eval()
    with torch.no_grad():
        c,r,_=model(pc,pr,ctx,tc)
        other_c,other_r,_=model(pc,pr,ctx,(tc+1)%CHORD_COUNT)
        torch.testing.assert_close(c,other_c,rtol=0,atol=0)
        assert not torch.equal(r,other_r)
        altered=pc.clone()
        altered[:,4:]=0
        future_c,_,_=model(altered,pr,ctx,tc)
        torch.testing.assert_close(c[:,:4],future_c[:,:4],rtol=0,atol=0)


def test_recurrent_incremental_equivalence():
    _,seq=example()
    pc,pr,ctx,tc,tr=collate([seq])
    model=StructureGRU(32).eval()
    with torch.no_grad():
        c,r,_=model(pc,pr,ctx,tc)
        cs,rs=[],[]
        hidden=None
        for i in range(len(seq.chords)):
            state,chord,hidden=model.encode(pc[:,i:i+1],pr[:,i:i+1],ctx[:,i:i+1],hidden)
            cs.append(chord)
            rs.append(model.rhythm_logits(state,tc[:,i:i+1]))
        torch.testing.assert_close(torch.cat(cs,1),c,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(torch.cat(rs,1),r,rtol=1e-5,atol=2e-6)


def test_joint_loss_and_padding_gradient():
    model=StructureGRU(32)
    _,a=example(4)
    _,b=example(8)
    pc,pr,ctx,tc,tr=collate([a,b])
    c,r,_=model(pc,pr,ctx,tc.clamp_min(0))
    loss,lc,lr=joint_loss(c,r,tc,tr)
    torch.testing.assert_close(loss,lc+lr)
    new_c=c.detach().clone()
    new_r=r.detach().clone()
    new_c[tc==-100]=123
    new_r[tr==-100]=-123
    torch.testing.assert_close(loss.detach(),joint_loss(new_c,new_r,tc,tr)[0],rtol=0,atol=0)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_sampling_uses_context_only_and_preserves_rng_and_weights():
    model=StructureGRU(32).train()
    plan,seq=example()
    before=model_state_sha256(model)
    rng=torch.get_rng_state().clone()
    a=sample_structure(model,seq.contexts,17)
    b=sample_structure(model,schedule(plan),17)
    assert a==b and model.training
    assert torch.equal(rng,torch.get_rng_state())
    assert model_state_sha256(model)==before
    assert generated_metrics(a,set())["all_rules_valid"]
    assert extract(reconstruct(plan,a))==a


def test_uniform_joint_likelihood_oracle():
    seqs=[example(4)[1],example(8)[1]]
    model=StructureCounts(seqs)
    expected=likelihood(seqs,evaluate_counts(model,seqs,"uniform"))
    assert expected["chord_nll"]==pytest.approx(math.log(CHORD_COUNT))
    assert expected["rhythm_nll"]==pytest.approx(math.log(RHYTHM_COUNT))
    assert expected["joint_nll"]==pytest.approx(math.log(CHORD_COUNT*RHYTHM_COUNT))
    assert expected["bars"]==12
    assert likelihood([],()) is None
    for kind in BASELINES:
        for a,b in evaluate_counts(model,seqs,kind):
            assert (a>0).all() and (b>0).all()
            np.testing.assert_allclose(a.sum(1),1)
            np.testing.assert_allclose(b.sum(1),1)


@pytest.mark.parametrize("settings",[
    {"model_seeds":(1,)},{"development_seeds":(0,)},{"updates":0},
    {"bar_counts":(6,6)},{"bar_counts":(3,)},{"batch_size":True},{"width":12},
])
def test_plan_validation(settings):
    with pytest.raises(ValueError):
        StructurePlan(**settings)


def test_structure_training_and_generated_waveform_smoke():
    plan=StructurePlan(model_seeds=(17,29),training_seeds=(0,1,2),
                       development_seeds=(101,102),bar_counts=(6,8),
                       updates=2,batch_size=2,width=32,sample_seeds=(701,))
    before=torch.get_rng_state().clone()
    result=run_study(plan)
    assert torch.equal(before,torch.get_rng_state())
    assert result["training_examples"]==8
    assert result["unmasked_likelihood"]
    assert not result["perceptual_quality_established"]
    assert not result["independent_test_scored"]
    for run in result["runs"]:
        assert run["trained_sha256"]!=run["initial_sha256"]
        assert run["generation"]["samples"]==6
        assert all(row["masked"]["all_rules_valid"] for row in run["generation"]["rows"])
        render=run["generation"]["render"]
        assert render["bytes"]==44+render["frames"]*4
        assert render["metrics"]["clipping_fraction"]==0
    json.dumps(result,allow_nan=False)
