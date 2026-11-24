"""Semantic auxiliary loss, strict source labels and actual optimizer tests."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from research.semantic_training import (
    SemanticHead, semantic_loss, semantic_diagnostics, synthetic_source_targets,
    validate_targets, validate_semantic_weight,
)
from research.device_velocity_training import TrainConfig, train_model
from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS
from research.text_audio import (
    ByteTextTokenizer, TextWaveDenoiser, TextEncoder, SYNTHETIC_COMBINATIONS, _synth_prompts,
)


def canonical_tokens(seeds):
    return ByteTextTokenizer().encode([
        _synth_prompts(members, language, seed)
        for members in SYNTHETIC_COMBINATIONS for seed in seeds for language in ("en", "hi")
    ])


def test_canonical_targets_preserve_membership_and_language():
    tokens = canonical_tokens(TRAIN_SEEDS)
    targets = synthetic_source_targets(tokens, TRAIN_SEEDS)
    assert targets.shape == (84, 6)
    assert torch.equal(targets[::2], targets[1::2])
    expected = [members for members in SYNTHETIC_COMBINATIONS for _ in range(4)]
    for row, members in zip(targets, expected, strict=True):
        assert tuple(row.nonzero().flatten().tolist()) == members
    with pytest.raises(ValueError, match="Caption order"):
        synthetic_source_targets(tokens.flip(0), TRAIN_SEEDS)


@pytest.mark.parametrize("weight", [-1, 2.1, float("nan"), True, "1"])
def test_invalid_weights(weight):
    with pytest.raises(ValueError):
        validate_semantic_weight(weight)


@pytest.mark.parametrize("targets", [
    torch.zeros(2, 6), torch.ones(2, 6), torch.full((2,6),0.5),
    torch.ones(2,5), torch.full((2,6),float("nan")),
])
def test_invalid_targets(targets):
    with pytest.raises(ValueError):
        validate_targets(targets,2)


def test_dropped_captions_have_no_auxiliary_gradient():
    encoder = TextEncoder(16)
    head = SemanticHead(16)
    tokens = ByteTextTokenizer().encode(["rain","wind","forest"])
    targets = torch.eye(6)[:3]
    keep = torch.tensor([True,False,True])
    loss = semantic_loss(encoder,head,tokens,targets,keep)
    changed = tokens.clone()
    changed[1] = tokens[0]
    targets_changed = targets.clone()
    targets_changed[1] = targets[0]
    torch.testing.assert_close(loss,semantic_loss(encoder,head,changed,targets_changed,keep))
    loss.backward()
    assert head.projection.weight.grad is not None
    assert encoder.tokens.weight.grad is not None
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    zero = semantic_loss(encoder,head,tokens,targets,torch.zeros(3,dtype=torch.bool))
    assert zero.item() == 0 and not zero.requires_grad
    assert all(p.grad is None for p in head.parameters())


def test_uneven_microbatch_auxiliary_matches_full_logical_batch():
    encoder = TextEncoder(16)
    head = SemanticHead(16)
    tokens = ByteTextTokenizer().encode(["rain","wind","forest","ocean","thunder"])
    targets = torch.eye(6)[:5]
    keep = torch.tensor([True,False,True,True,False])
    (semantic_loss(encoder,head,tokens,targets,keep)/5).backward()
    full = head.projection.weight.grad.clone()
    encoder.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    for s in (slice(0,2),slice(2,4),slice(4,5)):
        piece = semantic_loss(encoder,head,tokens[s],targets[s],keep[s])
        if piece.requires_grad:
            (piece/5).backward()
    torch.testing.assert_close(head.projection.weight.grad,full,atol=1e-6,rtol=1e-5)


def test_diagnostics_restore_training_modes_and_report_micro_f1():
    encoder = TextEncoder(16).train()
    head = SemanticHead(16).eval()
    with torch.no_grad():
        head.projection.weight.zero_()
        head.projection.bias.fill_(-1)
        head.projection.bias[0] = 1
    tokens = ByteTextTokenizer().encode(["rain","wind"])
    targets = torch.eye(6)[:2]
    report = semantic_diagnostics(encoder,head,tokens,targets)
    assert report["exact_membership_matches"] == 1
    assert report["micro_f1"] == 0.5
    assert report["bit_accuracy"] == pytest.approx(10/12)
    assert encoder.training and not head.training
    assert not report["generated_audio_semantics_established"]


def test_real_auxiliary_training_keeps_inference_layout_and_plan():
    tokens = ByteTextTokenizer().encode(["rain","wind","forest","ocean","thunder","ambient"])
    waves = torch.randn(6,1,FRAMES,generator=torch.Generator().manual_seed(31))*0.1
    targets = torch.eye(6)
    cfg = TrainConfig(steps=2,batch_size=3,microbatch_size=2,condition_dropout=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        a = TextWaveDenoiser(16)
        baseline = train_model(a,waves,tokens,cfg)
        torch.manual_seed(17)
        b = TextWaveDenoiser(16)
        treatment = train_model(b,waves,tokens,replace(cfg,semantic_weight=0.5),
                                semantic_targets=targets,semantic_validation=(tokens,targets))
    assert baseline["batch_plan"] == treatment["batch_plan"]
    assert baseline["initial_state_sha256"] == treatment["initial_state_sha256"]
    assert baseline["trained_state_sha256"] != treatment["trained_state_sha256"]
    assert a.state_dict().keys() == b.state_dict().keys()
    assert baseline["semantic_diagnostics"] is None
    assert treatment["mean_semantic_auxiliary"] > 0
    assert treatment["mean_combined_objective"] > treatment["mean_weighted_objective"]
    assert not treatment["semantic_diagnostics"]["head_retained_for_inference"]
    assert treatment["semantic_diagnostics"]["final_validation"]["caption_rows"] == 6
    with pytest.raises(ValueError):
        train_model(b,waves,tokens,cfg,semantic_targets=targets)
    with pytest.raises(ValueError):
        train_model(b,waves,tokens,replace(cfg,semantic_weight=0.5))
