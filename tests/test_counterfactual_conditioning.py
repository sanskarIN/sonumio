import json

import pytest

torch = pytest.importorskip("torch")

from research import counterfactual_conditioning as cc
from research import counterfactual_conditioning_study as study
from research.text_audio import ByteTextTokenizer, SYNTHETIC_COMBINATIONS, _synth_prompts


def tokens_for(seeds):
    return ByteTextTokenizer().encode([
        _synth_prompts(members, language, seed)
        for members in SYNTHETIC_COMBINATIONS
        for seed in seeds
        for language in ("en", "hi")
    ])


def test_interventions_preserve_language_cardinality_and_template():
    seeds = (0, 1)
    tokens = tokens_for(seeds)
    table = cc.CaptionInterventions.build(tokens, seeds)
    indices = torch.arange(len(tokens))
    rng_before = torch.random.get_rng_state().clone()
    for round_index in range(6):
        wrong = table.wrong_indices(indices, round_index=round_index)
        assert not torch.equal(tokens, tokens[wrong])
        for i, j in enumerate(wrong.tolist()):
            assert table.languages[i] == table.languages[j]
            assert table.seeds[i] == table.seeds[j]
            assert len(table.members[i]) == len(table.members[j])
            assert not set(table.members[i]).intersection(table.members[j])
            assert not torch.equal(tokens[i], tokens[j])
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_reordered_captions_fail_closed():
    tokens = tokens_for((0, 1))
    with pytest.raises(ValueError, match="Caption order"):
        cc.CaptionInterventions.build(tokens.roll(1, dims=0), (0, 1))


@pytest.mark.parametrize("seeds", [(), (1, 1), (True,), (-1,), [1]])
def test_invalid_intervention_seeds(seeds):
    with pytest.raises(ValueError, match="recording seeds"):
        cc.CaptionInterventions.build(torch.zeros((1, 130), dtype=torch.long), seeds)


def test_relative_penalty_only_gradients_matched_branch():
    matched = torch.ones((2, 1, 4), requires_grad=True)
    wrong = torch.ones_like(matched, requires_grad=True)
    target = torch.zeros_like(matched)
    loss = cc.relative_caption_penalty(matched, wrong, target)
    assert loss.item() == pytest.approx(cc.RELATIVE_MARGIN)
    loss.backward()
    assert torch.all(matched.grad > 0)
    assert wrong.grad is None


def test_margin_satisfied_no_auxiliary_gradient():
    matched = torch.full((1, 1, 4), 0.5, requires_grad=True)
    wrong = torch.ones_like(matched, requires_grad=True)
    loss = cc.relative_caption_penalty(matched, wrong, torch.zeros_like(matched))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(matched.grad) == 0
    assert wrong.grad is None


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, noisy, timestep, tokens):
        self.calls.append((noisy.clone(), timestep.clone(), tokens.clone(), torch.is_grad_enabled()))
        return torch.ones_like(noisy)


def test_training_penalty_keeps_only_conditioned_rows():
    model = RecordingModel()
    noisy = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    timestep = torch.tensor([2, 6])
    matched = torch.ones_like(noisy, requires_grad=True)
    target = torch.zeros_like(noisy)
    tokens = torch.tensor([[1, 2], [3, 4]])
    loss = cc.training_penalty(model, noisy, timestep, matched, target,
                               torch.tensor([False, True]), tokens)
    x, t, c, grad_enabled = model.calls[0]
    assert torch.equal(x, noisy[1:])
    assert torch.equal(t, timestep[1:])
    assert torch.equal(c, tokens[1:])
    assert not grad_enabled
    loss.backward()
    assert torch.count_nonzero(matched.grad[0]) == 0
    assert torch.count_nonzero(matched.grad[1]) == 4


def test_all_dropped_rows_skip_negative_forward():
    model = RecordingModel()
    noisy = torch.zeros((2, 1, 4))
    penalty = cc.training_penalty(model, noisy, torch.zeros(2, dtype=torch.long),
                                  noisy, noisy, torch.zeros(2, dtype=torch.bool),
                                  torch.zeros((2, 130), dtype=torch.long))
    assert penalty.item() == 0
    assert model.calls == []


@pytest.mark.parametrize("weight", [True, -1, 2.1, float("nan"), float("inf")])
def test_invalid_weight(weight):
    with pytest.raises(ValueError, match="Conditioning weight"):
        cc.validate_weight(weight)


@pytest.mark.parametrize("kwargs", [
    {"weight": 0}, {"train_steps": 0}, {"evaluations": 9},
    {"sampling_seeds": (301, 301)}, {"seed": True},
])
def test_invalid_study_configuration(kwargs):
    with pytest.raises(ValueError, match="bounded paired conditioning"):
        study.run_study(**kwargs)


def test_real_paired_training_and_generation_preserve_controls():
    original_rng = torch.random.get_rng_state().clone()
    report = study.run_study(train_steps=2, codec_steps=1, evaluations=2)
    assert torch.equal(original_rng, torch.random.get_rng_state())
    assert report["same_initial_parameters"]
    assert report["same_training_input_trace"]
    assert report["same_frozen_codec"]
    assert report["retained_caption_examples"] > 0
    assert report["generated_scores_available"]
    assert not report["semantic_quality_established"]
    assert not report["weights_or_audio_saved"]
    baseline = report["arms"]["baseline"]
    treatment = report["arms"]["treatment"]
    assert baseline["training"]["conditioning_weight"] == 0
    assert treatment["training"]["conditioning_weight"] == 0.5
    assert all(v == 0 for v in baseline["training"]["mean_auxiliary_caption_penalty"].values())
    for result in (baseline, treatment):
        assert result["generated_case_count"] == 24
        assert len(result["null_guidance_checks"]) == 2
        for arm in ("raw", "latent"):
            heldout = result["teacher_forced"][arm]
            assert heldout["unique_recordings"] == 42
            assert heldout["caption_cases_per_timestep"] == 84
            assert heldout["same_noise_for_bilingual_pairs"]
            assert not heldout["independent_caption_cases"]
            assert set(heldout["by_timestep_and_caption_stratum"]) == {"0", "4", "7"}
            for strata in heldout["by_timestep_and_caption_stratum"].values():
                assert sum(row["caption_cases"] for row in strata.values()) == 84
                assert strata["en_1_source"]["caption_cases"] == 12
                assert strata["hi_2_source"]["caption_cases"] == 30
                assert all(0 <= row["matched_beats_wrong"] <= row["caption_cases"]
                           for row in strata.values())
    json.dumps(report, allow_nan=False)
