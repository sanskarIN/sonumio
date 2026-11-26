"""Training-only frequency subspace and anchored optimization regression tests."""
import json
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.native_codec import FRAMES, STRIDE, LATENT_CHANNELS, LATENT_FRAMES
from research.native_codec_frequency_basis import (
    BasisPlan, full_dct, fit_frequency_basis, initialized_codec, anchored_penalty,
    SPLIT_BIN,
)
from research.native_codec_frequency_study import FrequencyStudyPlan, run_study
from research.weight_ema import model_state_sha256


def synthetic_panel():
    basis=full_dct()
    rows=[]
    for lo,hi in [(2,83),(4,105),(7,149),(8,170)]:
        # Equal-block signals with nonzero energy in both bands.
        block=(.3*basis[lo]+.1*basis[hi]).float()
        rows.append(block.repeat(LATENT_FRAMES).reshape(1,FRAMES))
    return torch.stack(rows)


def test_complete_dct_is_orthonormal():
    basis=full_dct()
    torch.testing.assert_close(basis@basis.T,torch.eye(STRIDE,dtype=torch.float64),
                               rtol=0,atol=1e-12)
    assert SPLIT_BIN==72


@pytest.mark.parametrize("high_rank",[4,8])
def test_fitted_basis_is_orthogonal_and_band_separated(high_rank):
    data=synthetic_panel()
    before=data.clone()
    basis,report=fit_frequency_basis(data,BasisPlan(high_rank))
    assert basis.shape==(LATENT_CHANNELS,STRIDE)
    assert torch.equal(before,data)
    torch.testing.assert_close(basis@basis.T,torch.eye(LATENT_CHANNELS),
                               rtol=0,atol=3e-6)
    spectrum=basis.double()@full_dct().T
    assert spectrum[:LATENT_CHANNELS-high_rank,SPLIT_BIN:].abs().max()<1e-7
    assert spectrum[LATENT_CHANNELS-high_rank:,:SPLIT_BIN].abs().max()<1e-7
    assert not report["validation_used"] and not report["center_vector_stored"]
    assert report["retained_low_energy"]==pytest.approx(report["low_subspace_energy"],rel=1e-6)
    assert report["retained_high_energy"]==pytest.approx(report["high_subspace_energy"],rel=1e-6)
    model=initialized_codec(basis)
    torch.testing.assert_close(model.decode_raw(model.encode(data)),data,rtol=1e-5,atol=1e-7)


def test_fit_is_equal_clip_power_weighted():
    source=synthetic_panel()
    scaled=source*torch.tensor([1.,.5,2.,.7]).view(-1,1,1)
    # Degenerate eigenvalues may rotate; compare retained projectors
    # on the nondegenerate observed training directions instead.
    a,ra=fit_frequency_basis(source,BasisPlan(4))
    b,rb=fit_frequency_basis(scaled,BasisPlan(4))
    for name in ("low_subspace_energy","high_subspace_energy","retained_low_energy","retained_high_energy"):
        assert ra[name]==pytest.approx(rb[name],rel=1e-6)
    torch.testing.assert_close(initialized_codec(a).decode_raw(initialized_codec(a).encode(source)),
                               initialized_codec(b).decode_raw(initialized_codec(b).encode(source)),
                               rtol=1e-5,atol=1e-7)


def test_anchoring_penalty_zero_at_start_and_differentiable_after_update():
    anchor,_=fit_frequency_basis(synthetic_panel())
    model=initialized_codec(anchor)
    initial=anchored_penalty(model,anchor)
    assert initial["anchor_distance"].item()==0
    assert initial["analysis_synthesis_disagreement"].item()==0
    assert initial["orthogonality_error"].item()<1e-10
    with torch.no_grad():
        model.analysis[0,0]+=.1
        model.synthesis[0,1]-=.1
    changed=anchored_penalty(model,anchor)
    assert changed["anchor_distance"]>0
    assert changed["analysis_synthesis_disagreement"]>0
    sum(changed.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert anchor.grad is None


@pytest.mark.parametrize("rank",[True,0,16,17,4.0])
def test_invalid_rank(rank):
    with pytest.raises(ValueError):
        BasisPlan(rank)


def test_reject_untrusted_basis_and_invalid_training():
    with pytest.raises(ValueError):
        initialized_codec(torch.ones(LATENT_CHANNELS,STRIDE))
    with pytest.raises(ValueError):
        fit_frequency_basis(torch.zeros(37,1,FRAMES))
    with pytest.raises(ValueError):
        fit_frequency_basis(torch.full((1,1,FRAMES),float("nan")))


@pytest.mark.parametrize("kwargs",[
    {"order_seeds":(1,)},{"development_seeds":(0,)},{"checkpoints":(64,)},
    {"updates":257},{"batch_size":True},{"checkpoints":(32,32,128)}
])
def test_frequency_plan_validation(kwargs):
    with pytest.raises(ValueError):
        FrequencyStudyPlan(**kwargs)


def test_frequency_study_smoke():
    plan=FrequencyStudyPlan(order_seeds=(17,29),training_seeds=(0,),
                            development_seeds=(301,),updates=2,batch_size=2,checkpoints=(1,2))
    rng=torch.get_rng_state().clone()
    report=run_study(plan)
    assert torch.equal(rng,torch.get_rng_state())
    assert report["train_examples"]==16
    assert report["training_clips"]==report["development_clips"]==6
    assert not report["development_used_for_basis_fit"]
    assert not report["independent_test_scored"]
    assert report["shared_initialization_and_batches"]
    assert report["order_seeds_are_not_independent_initializations"]
    assert set(report["fixed"])=={"analytic_dct","fitted_4high","fitted_8high"}
    for run in report["runs"]:
        assert run["initial_model_sha256"]==report["fixed"]["fitted_4high"]["model_sha256"]
        assert set(run["arms"])=={"unregularized","anchored"}
        for snapshots in run["arms"].values():
            assert [r["step"] for r in snapshots]==[1,2]
            assert snapshots[-1]["evaluation"]["model_sha256"]!=run["initial_model_sha256"]
    assert len(report["summary"]["anchored_minus_unregularized"])==2
    json.dumps(report,allow_nan=False)
