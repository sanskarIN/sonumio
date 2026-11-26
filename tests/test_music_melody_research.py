"""Original melody dataset, recurrent model and independent likelihood controls."""
from dataclasses import replace
import math
import json
import numpy as np
import pytest

torch=pytest.importorskip("torch")

from sonumio.arrangement import ArrangementPlan, compose_arrangement
from research.music_melody_data import (
    MelodySequence, extract, fingerprint, valid_tokens, replace_melody, split_corpus,
    collate, BOS, PAD, PITCH_COUNT,
)
from research.music_melody_model import MelodyGRU, token_loss, generate_tokens
from research.music_melody_baselines import MelodyCounts, evaluate_counts
from research.music_melody_study import (
    MelodyPlan, likelihood_metrics, free_running_metrics, run_study,
)
from research.weight_ema import model_state_sha256


def sequences():
    return [extract(compose_arrangement(ArrangementPlan(style=style,bars=8,seed=17)))
            for style in ("calm","rhythmic")]


def test_relative_pitch_and_conditioning_contract():
    score=compose_arrangement(ArrangementPlan(bars=8,seed=17))
    seq=extract(score)
    assert len(seq.pitches)==sum(e.track=="lead" for e in score.events)
    assert all(p in valid_tokens(c,final=i==len(seq.pitches)-1)
               for i,(p,c) in enumerate(zip(seq.pitches,seq.contexts,strict=True)))
    assert replace_melody(score,seq.pitches)==score
    assert fingerprint(seq)==fingerprint(extract(score))
    bad=list(seq.pitches)
    bad[-1]=1
    with pytest.raises(ValueError):
        replace_melody(score,tuple(bad))


def test_corpus_source_and_exact_relative_pitch_split():
    train,dev,receipt=split_corpus((0,1,2),(101,102,103),bars=8)
    assert receipt["training_source_rows"]==18
    assert receipt["development_source_rows"]==18
    assert len(train)==receipt["training_unique_rows"]
    assert len(dev)==receipt["development_retained_rows"]
    assert not ({fingerprint(s,pitches_only=True) for _,s in train}
                &{fingerprint(s,pitches_only=True) for _,s in dev})
    with pytest.raises(ValueError):
        split_corpus((0,1),(1,2))


def test_variable_length_collation_and_no_target_leakage():
    a,b=sequences()
    assert len(a.pitches)!=len(b.pitches)
    previous,context,targets=collate([a,b])
    for i,seq in enumerate((a,b)):
        n=len(seq.pitches)
        assert previous[i,0]==BOS
        assert previous[i,1:n].tolist()==list(seq.pitches[:-1])
        assert targets[i,:n].tolist()==list(seq.pitches)
        assert torch.all(previous[i,n:]==PAD)
        assert torch.all(targets[i,n:]==-100)
        assert context[i,:n].tolist()==[list(c) for c in seq.contexts]


def test_gru_causal_logits_and_incremental_equivalence():
    model=MelodyGRU(32).eval()
    seq=sequences()[0]
    prev,ctx,_=collate([seq])
    with torch.no_grad():
        all_logits,_=model(prev,ctx)
        hidden=None
        incremental=[]
        for i in range(prev.shape[1]):
            logits,hidden=model(prev[:,i:i+1],ctx[:,i:i+1],hidden)
            incremental.append(logits)
        torch.testing.assert_close(all_logits,torch.cat(incremental,1),rtol=1e-5,atol=2e-6)
        changed=prev.clone()
        changed[:,5:]=0
        other,_=model(changed,ctx)
        torch.testing.assert_close(all_logits[:,:5],other[:,:5],rtol=0,atol=0)


def test_token_loss_ignores_padding_and_has_finite_gradients():
    model=MelodyGRU(32)
    prev,ctx,targets=collate(sequences())
    logits,_=model(prev,ctx)
    loss=token_loss(logits,targets)
    altered=logits.detach().clone()
    altered[targets==-100]=123
    torch.testing.assert_close(loss.detach(),token_loss(altered,targets),rtol=0,atol=0)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_free_running_sampling_never_reads_target_pitch_and_preserves_model():
    model=MelodyGRU(32).train()
    sequence=sequences()[0]
    altered=MelodySequence(tuple(0 for _ in sequence.pitches),sequence.contexts)
    before=model_state_sha256(model)
    rng=torch.get_rng_state().clone()
    a=generate_tokens(model,sequence,19)
    assert a==generate_tokens(model,altered,19)
    assert a==generate_tokens(model,sequence,19)
    assert torch.equal(rng,torch.get_rng_state())
    assert model.training
    assert model_state_sha256(model)==before
    assert free_running_metrics(sequence,a)["valid_token_fraction"]==1
    assert a[-1]%12==0


def test_uniform_likelihood_matches_analytic_entropy():
    seq=sequences()
    counts=MelodyCounts(seq)
    report=likelihood_metrics(seq,evaluate_counts(counts,seq,"uniform"))
    assert report["mean_nll"]==pytest.approx(math.log(PITCH_COUNT))
    assert report["perplexity"]==pytest.approx(PITCH_COUNT)
    assert report["tokens"]==sum(len(s.pitches) for s in seq)
    for kind in ("unigram","conditional_bigram"):
        rows=evaluate_counts(counts,seq,kind)
        assert all(np.allclose(r.sum(1),1) and np.all(r>0) for r in rows)


@pytest.mark.parametrize("kwargs",[
    {"model_seeds":(17,)},{"training_seeds":(101,)},{"width":12},
    {"updates":0},{"batch_size":True},{"sample_seeds":(1,1)},{"bars":17},
])
def test_invalid_study_plan(kwargs):
    with pytest.raises(ValueError):
        MelodyPlan(**kwargs)


def test_melody_training_and_actual_audio_render_smoke():
    plan=MelodyPlan(model_seeds=(17,29),training_seeds=(0,1,2),
                    development_seeds=(101,102,103),updates=2,batch_size=2,
                    width=32,bars=8,sample_seeds=(701,))
    before=torch.get_rng_state().clone()
    report=run_study(plan)
    assert torch.equal(before,torch.get_rng_state())
    assert report["train_examples"]==8
    assert report["unmasked_likelihood_evaluation"]
    assert report["mask_constraints_are_not_learned"]
    assert not report["perceptual_quality_established"]
    for run in report["runs"]:
        assert run["initial_model_sha256"]!=run["final_model_sha256"]
        assert run["generation"]["mean_masked_valid_token_fraction"]==1
        assert run["generation"]["samples"]==6
        rendered=run["generation"]["first_generated_score_render"]
        assert rendered["channels"]==2 and rendered["sample_rate"]==24000
        assert rendered["wav_bytes"]==44+rendered["frames"]*4
        assert rendered["metrics"]["clipping_fraction"]==0
    json.dumps(report,allow_nan=False)
