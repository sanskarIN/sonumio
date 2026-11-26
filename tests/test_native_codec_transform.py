"""Native block codec analytical reconstruction and controlled training tests."""
import math
import json
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.native_codec import FRAMES, STRIDE, LATENT_FRAMES, LATENT_CHANNELS
from research.native_codec_transform import (
    NativeBlockCodec, dct_basis, initial_basis, projection_diagnostics,
)
from research.native_codec_transform_study import run_study
from research.native_codec_convergence import ConvergencePlan, phase_components


@pytest.mark.parametrize("kind",["random_orthogonal","frequency_stratified"])
def test_basis_orthonormal_and_deterministic(kind):
    a=initial_basis(kind,17)
    assert a.shape==(LATENT_CHANNELS,STRIDE)
    torch.testing.assert_close(a@a.T,torch.eye(LATENT_CHANNELS),atol=2e-6,rtol=0)
    assert torch.equal(a,initial_basis(kind,17))
    if kind=="random_orthogonal":
        assert not torch.equal(a,initial_basis(kind,29))


def test_dct_exact_projection_on_retained_basis_and_rejected_bin():
    model=NativeBlockCodec()
    kept=dct_basis()[12].repeat(LATENT_FRAMES).reshape(1,1,FRAMES)
    latent=model.encode(kept)
    assert latent.shape==(1,LATENT_CHANNELS,LATENT_FRAMES)
    torch.testing.assert_close(model.decode_raw(latent),kept,atol=2e-7,rtol=1e-5)
    n=torch.arange(STRIDE,dtype=torch.float64)
    rejected=(math.sqrt(2/STRIDE)*torch.cos(math.pi*(n+.5)*13/STRIDE)).float()
    rejected=rejected.repeat(LATENT_FRAMES).reshape(1,1,FRAMES)
    assert model.encode(rejected).abs().max()<1e-7
    assert model.decode_raw(model.encode(rejected)).abs().max()<1e-6


def test_decoder_is_independent_of_prior_encode_calls():
    model=NativeBlockCodec()
    latent=torch.zeros(1,LATENT_CHANNELS,LATENT_FRAMES)
    first=model.decode_raw(latent)
    model.encode(torch.ones(1,1,FRAMES)*.5)
    assert torch.equal(first,model.decode_raw(latent))
    assert torch.count_nonzero(first)==0


def test_transform_gradient_and_parameter_count():
    model=NativeBlockCodec()
    assert sum(p.numel() for p in model.parameters())==2*STRIDE*LATENT_CHANNELS
    x=torch.rand(2,1,FRAMES)*.4-.2
    before=model.analysis.detach().clone()
    loss=phase_components(x,model(x))["normalized_waveform_mse"].mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.optim.AdamW(model.parameters(),lr=.001).step()
    assert not torch.equal(before,model.analysis)
    assert projection_diagnostics(model)["analysis_gram_rmse"]>=0


@pytest.mark.parametrize("indices",[(0,)*16,tuple(range(15)),tuple(range(15))+(192,)])
def test_invalid_dct_indices(indices):
    with pytest.raises(ValueError):
        dct_basis(indices)


def test_paired_architecture_study_smoke():
    plan=ConvergencePlan(model_seeds=(17,29),training_seeds=(0,),
                         validation_seeds=(201,),updates=2,batch_size=2,checkpoints=(1,2))
    before=torch.get_rng_state().clone()
    report=run_study(plan)
    assert torch.equal(before,torch.get_rng_state())
    assert report["train_examples"]==24
    assert report["all_arms_share_batches_and_objective"]
    assert not report["all_arms_parameter_matched"]
    assert report["block_arms_share_architecture"]
    a,b=report["runs"]
    assert a["arms"]["frequency_stratified"]["initial_model_sha256"]==b["arms"]["frequency_stratified"]["initial_model_sha256"]
    assert a["arms"]["random_orthogonal"]["initial_model_sha256"]!=b["arms"]["random_orthogonal"]["initial_model_sha256"]
    for run in report["runs"]:
        assert run["arms"]["frequency_stratified"]["parameters"]==run["arms"]["random_orthogonal"]["parameters"]
        for entry in run["arms"].values():
            assert entry["snapshots"][-1]["evaluation"]["model_sha256"]!=entry["initial_model_sha256"]
    assert not report["independent_test_scored"]
    json.dumps(report,allow_nan=False)
