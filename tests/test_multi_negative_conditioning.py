"""Multi-negative conditioning regression tests."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from research.counterfactual_conditioning import CaptionInterventions
from research.device_velocity_training import TrainConfig, train_model
from research.latent_audio import FRAMES, TRAIN_SEEDS
from research.multi_negative_conditioning import (
    matched_margin_sum, negative_indices, training_margin_sum, validate_margin,
)
from research.text_audio import ByteTextTokenizer, TextWaveDenoiser, SYNTHETIC_COMBINATIONS, _synth_prompts


def table_and_tokens():
    tokens = ByteTextTokenizer().encode([
        _synth_prompts(members, language, seed)
        for members in SYNTHETIC_COMBINATIONS
        for seed in TRAIN_SEEDS for language in ("en", "hi")
    ])
    return CaptionInterventions.build(tokens, TRAIN_SEEDS), tokens


def test_negative_selection_distinct_disjoint_and_preserves_strata():
    table, tokens = table_and_tokens()
    ids = torch.arange(len(tokens))
    for round_index in range(5):
        negatives = negative_indices(table, ids, round_index=round_index, count=4)
        assert negatives.shape == (len(tokens), 4)
        for row, selected in enumerate(negatives.tolist()):
            assert len(set(selected)) == 4
            for other in selected:
                assert not set(table.members[row]) & set(table.members[other])
                assert len(table.members[row]) == len(table.members[other])
                assert table.seeds[row] == table.seeds[other]
                assert table.languages[row] == table.languages[other]
    assert torch.equal(negative_indices(table, ids, round_index=1, count=2),
                       negative_indices(table, ids, round_index=1, count=2))


@pytest.mark.parametrize("weight,margin,count", [
    (-1,.05,2), (3,.05,2), (1,0,2), (1,.51,2),
    (1,float("nan"),2), (1,.05,0), (1,.05,5), (1,.05,True),
])
def test_invalid_margin_settings(weight, margin, count):
    with pytest.raises(ValueError):
        validate_margin(weight,margin,count)


def test_margin_uses_lowest_error_and_detaches_negatives():
    matched = torch.tensor([[[2.]], [[1.]]], requires_grad=True)
    target = torch.zeros_like(matched)
    errors = torch.tensor([[3.,1.], [4.,3.]],requires_grad=True)
    loss = matched_margin_sum(matched,target,errors,margin=.1)
    assert loss.item() == pytest.approx(3.1)
    loss.backward()
    torch.testing.assert_close(matched.grad,torch.tensor([[[4.]],[[0.]]]))
    assert errors.grad is None


def test_partition_equivalence_for_detached_margin():
    target = torch.zeros(5,1,4)
    x = torch.arange(20,dtype=torch.float32).reshape(5,1,4)/10
    negatives = torch.ones(5,2)*.5
    p = torch.tensor(1.,requires_grad=True)
    q = p.detach().clone().requires_grad_()
    (matched_margin_sum(x*p,target,negatives,margin=.05)/5).backward()
    for s in (slice(0,2),slice(2,4),slice(4,5)):
        (matched_margin_sum(x[s]*q,target[s],negatives[s],margin=.05)/5).backward()
    torch.testing.assert_close(p.grad,q.grad)


def test_no_model_calls_for_all_dropped_captions():
    class NeverCalled:
        def __call__(self,*args):
            raise AssertionError("Dropped caption invoked wrong branch")
    a = torch.zeros(3,1,8)
    result = training_margin_sum(NeverCalled(),a,torch.zeros(3,dtype=torch.long),
                                 a,a,torch.zeros(3,dtype=torch.bool),
                                 torch.zeros(3,2,130,dtype=torch.long),margin=.05)
    assert result.item() == 0


def test_real_multi_negative_training_has_same_input_plan():
    _, tokens = table_and_tokens()
    waves = torch.randn(len(tokens)//2,1,FRAMES,generator=torch.Generator().manual_seed(31))*0.1
    waves = waves.repeat_interleave(2,dim=0)
    cfg = TrainConfig(steps=2,batch_size=3,microbatch_size=2,condition_dropout=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        a = train_model(TextWaveDenoiser(16),waves,tokens,cfg)
        torch.manual_seed(17)
        b = train_model(TextWaveDenoiser(16),waves,tokens,
                        replace(cfg,margin_weight=.5),intervention_seeds=TRAIN_SEEDS)
    assert a["batch_plan"] == b["batch_plan"]
    assert a["initial_state_sha256"] == b["initial_state_sha256"]
    assert a["mean_caption_margin"] == 0
    assert b["mean_caption_margin"] >= 0
    assert b["mean_combined_objective"] == pytest.approx(
        b["mean_weighted_objective"] + .5*b["mean_caption_margin"])
    with pytest.raises(ValueError):
        train_model(TextWaveDenoiser(16),waves,tokens,replace(cfg,margin_weight=.5))
