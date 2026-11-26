"""Native-band codec geometry, high-frequency controls and paired training."""
import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.native_codec import (
    Native24kCodec, FRAMES, SAMPLE_RATE, LATENT_FRAMES, LATENT_CHANNELS,
    polyphase_pack, polyphase_unpack, native_metrics, native_loss,
)
from research.native_codec_study import (
    StudyPlan, dataset, lowband_control, batch_plan, run_study,
)
from research.wave_diffusion import CATEGORIES
from research.weight_ema import model_state_sha256


def tone(hz):
    return (.2*torch.sin(2*math.pi*hz*torch.arange(FRAMES,dtype=torch.float64)
                        /SAMPLE_RATE)).float().reshape(1,1,FRAMES)


def test_polyphase_is_exact_rearrangement_and_gradients():
    x=torch.linspace(-1,1,FRAMES).reshape(1,1,FRAMES).requires_grad_()
    p=polyphase_pack(x)
    assert p.shape==(1,3,FRAMES//3)
    torch.testing.assert_close(p[0,1],x[0,0,1::3],rtol=0,atol=0)
    y=polyphase_unpack(p)
    assert torch.equal(x,y)
    y.sum().backward()
    assert torch.equal(x.grad,torch.ones_like(x))


def test_native_codec_geometry_and_update():
    model=Native24kCodec()
    source=torch.cat((tone(1000),tone(7000)))
    latent=model.encode(source)
    assert latent.shape==(2,LATENT_CHANNELS,LATENT_FRAMES)
    assert latent.abs().max()<=1
    output=model(source)
    assert output.shape==source.shape and output.abs().max()<=1
    old=model_state_sha256(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    loss=native_loss(source,output,.5)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert model_state_sha256(model)!=old


def test_identity_and_silence_finite_gradients():
    x=tone(7000)
    y=x.clone().requires_grad_()
    stats=native_metrics(x,y)
    for key in ("waveform_mae","waveform_mse","log_spectral_mae",
                "spectral_convergence","highband_magnitude_error"):
        assert stats[key].item()==pytest.approx(0,abs=1e-8)
    native_loss(x,y,.5).backward()
    assert torch.isfinite(y.grad).all()
    zero=torch.zeros_like(x)
    restored=zero.clone().requires_grad_()
    loss=native_loss(zero,restored,.5)
    loss.backward()
    assert loss.item()==pytest.approx(0,abs=1e-8)
    assert torch.isfinite(restored.grad).all()


def test_resampling_control_removes_highband_not_lowband():
    high=tone(7000)
    low=tone(1000)
    high_metrics=native_metrics(high,lowband_control(high))
    low_metrics=native_metrics(low,lowband_control(low))
    assert high_metrics["source_highband_fraction"].item()>.99
    assert low_metrics["source_highband_fraction"].item()<.01
    assert high_metrics["highband_magnitude_error"].item()>.9
    assert low_metrics["waveform_mae"].item()<.005
    assert lowband_control(high).square().mean()<high.square().mean()*.01


@pytest.mark.parametrize("invalid", [
    torch.zeros(1,1,FRAMES-1),torch.zeros(1,2,FRAMES),
    torch.zeros(1,1,FRAMES,dtype=torch.float64),
    torch.full((1,1,FRAMES),float("nan")),
])
def test_bad_input_rejected(invalid):
    with pytest.raises(ValueError):
        Native24kCodec().encode(invalid)


@pytest.mark.parametrize("settings", [
    {"model_seeds":(1,)},{"model_seeds":(1,1)},{"training_seeds":(101,)},
    {"updates":0},{"updates":True},{"batch_size":9},
    {"model_seeds":(1,2,3,4),"updates":256,"batch_size":8},
])
def test_plan_rejects_invalid_budgets(settings):
    with pytest.raises(ValueError):
        StudyPlan(**settings)


def test_original_rate_sources_and_seed_disjointness():
    a,labels,hashes,digest=dataset((0,1))
    b,_,other,other_digest=dataset((101,))
    assert a.shape==(12,1,FRAMES) and b.shape==(6,1,FRAMES)
    assert labels==tuple(i for i in range(6) for _ in range(2))
    assert len(set(hashes))==len(hashes)
    assert not set(hashes)&set(other)
    assert digest!=other_digest
    replay,_,_,same=dataset((0,1))
    assert torch.equal(a,replay) and same==digest


def test_batch_trace_reproducible():
    plan=StudyPlan(updates=5,batch_size=4)
    a,trace=batch_plan(plan,9,6)
    b,other=batch_plan(plan,9,6)
    assert a==b and trace==other
    assert all(0<=x<6 for row in a for x in row)
    assert trace!=batch_plan(plan,10,6)[1]


def test_paired_native_training_smoke():
    plan=StudyPlan(model_seeds=(17,29),training_seeds=(0,),
                   validation_seeds=(101,),updates=2,batch_size=2)
    rng=torch.get_rng_state().clone()
    report=run_study(plan)
    assert torch.equal(torch.get_rng_state(),rng)
    assert report["sample_rate"]==24000
    assert report["clip_seconds"]==.512
    assert report["latent_scalar_compression"]==12
    assert report["training_examples"]==16
    assert report["matched_initialization_batches_and_architecture"]
    for run in report["runs"]:
        assert run["initial"]["model_sha256"]==run["initial_model_sha256"]
        assert set(run["arms"])=={"base","highband"}
        for arm in run["arms"].values():
            assert arm["trained"]["model_sha256"]!=run["initial_model_sha256"]
            assert set(arm["trained"]["by_category"])==set(CATEGORIES)
    for key,summary in report["paired_seed_deltas"].items():
        expected=[r["arms"]["highband"]["trained"]["mean"][key]-
                  r["arms"]["base"]["trained"]["mean"][key] for r in report["runs"]]
        assert summary["per_seed_deltas"]==expected
    assert not report["perceptual_quality_established"]
    assert report["files_written"]==0
    json.dumps(report,allow_nan=False)


def test_band_fractions_always_in_unit_interval():
    x=torch.cat((tone(1000),tone(7000),torch.zeros_like(tone(1000))))
    metrics=native_metrics(x,x)
    for key in ("source_highband_fraction","restored_highband_fraction"):
        assert torch.all((metrics[key]>=0)&(metrics[key]<=1))
    assert metrics["source_highband_fraction"][2].item()==0


def test_smoothed_residual_norm_is_exact_zero_monotone_and_differentiable():
    from research.native_codec import _smooth_residual_norm, EPS
    energy=torch.tensor([0.,1e-20,1e-12,1e-8,1e-4,1.,100.],
                        dtype=torch.float32,requires_grad=True)
    actual=_smooth_residual_norm(energy)
    oracle=energy.detach().double()/(torch.sqrt(energy.detach().double()+EPS)
                                     +math.sqrt(EPS))
    assert actual[0].item()==0.
    assert torch.all(actual[1:]>actual[:-1])
    torch.testing.assert_close(actual.double(),oracle,rtol=2e-7,atol=0)
    actual.sum().backward()
    assert torch.isfinite(energy.grad).all()
    assert torch.all(energy.grad>0)


@pytest.mark.parametrize("amplitude",[0.,1e-7,1e-4,.2])
def test_identical_native_audio_has_exact_zero_loss(amplitude):
    x=tone(7000)*(amplitude/.2)
    y=x.clone().requires_grad_()
    loss=native_loss(x,y,.5)
    assert loss.item()==0.
    loss.backward()
    assert torch.isfinite(y.grad).all()
    assert torch.count_nonzero(y.grad).item()==0
