"""Frozen codec calibration, crop latent v-objective and end-to-end study."""
from dataclasses import replace
import json
import numpy as np
import pytest

torch=pytest.importorskip("torch")
from research.crop_codec import CropAudioCodec
from research.crop_codec_packet import CropCodecError,code_shape
from research.crop_latent import (
    CropLatentCalibration,LatentMomentAccumulator,fit_calibration,restore_for_decoder,
    CropLatentDenoiser,velocity_loss,sample_latent,reverse_indices,encode_native,
)
from research.crop_latent_study import LatentStudyPlan,run_study
from research.wave_diffusion import DiffusionSchedule
from research.weight_ema import model_state_sha256


def test_calibration_moments_partition_invariant():
    values=torch.linspace(-1,1,8*16*32).reshape(8,16,32)
    one=LatentMomentAccumulator(2048)
    one.add(values)
    many=LatentMomentAccumulator(2048)
    for batch in values.split(3):
        many.add(batch)
    a,b=one.finish(),many.finish()
    np.testing.assert_allclose(a.mean,b.mean,rtol=0,atol=1e-15)
    np.testing.assert_allclose(a.std,b.std,rtol=0,atol=1e-15)
    expected=values.double().mean((0,2)).numpy()
    np.testing.assert_allclose(a.mean,expected,rtol=0,atol=1e-15)
    restored=a.restore(a.normalize(values))
    torch.testing.assert_close(restored,values,rtol=1e-5,atol=1e-7)


def test_constant_channels_floor_and_explicit_projection():
    accumulator=LatentMomentAccumulator(2048)
    accumulator.add(torch.zeros(2,16,32))
    cal=accumulator.finish()
    assert cal.report()["floored_channels"]==16
    assert cal.normalize(torch.zeros(1,16,32)).abs().max()==0
    raw=torch.full((1,16,32),200.)
    projected=restore_for_decoder(raw,cal,bits=4)
    assert projected.diagnostics["native_peak_before_projection"]==2.
    assert projected.diagnostics["native_outside_bound_fraction"]==1.
    assert projected.diagnostics["clamp_mse"]==1.
    assert torch.equal(projected.latent,torch.ones_like(raw))


def test_calibration_requires_frozen_codec_and_preserves_state():
    model=CropAudioCodec()
    values=torch.rand(3,1,2048)*.4-.2
    with pytest.raises(CropCodecError,match="frozen"):
        fit_calibration(model,[values])
    model.eval().requires_grad_(False)
    before=model_state_sha256(model)
    result=fit_calibration(model,[values[:1],values[1:]],bits=4)
    assert result.clips==3
    assert before==model_state_sha256(model)
    native=encode_native(model,values,4)
    torch.testing.assert_close(result.restore(result.normalize(native)),native,atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize("frames",[2048,8192])
def test_latent_velocity_gradient_and_layout(frames):
    model=CropLatentDenoiser(frames,width=16)
    schedule=DiffusionSchedule(8,profile="cosine_zero_terminal_v1")
    values=torch.randn((2,*code_shape(frames)))
    noise=torch.randn_like(values)
    loss=velocity_loss(model,schedule,values,torch.tensor([0,7]),noise,torch.tensor([1,6]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad)>0 for p in model.parameters())


def test_reverse_sampling_exact_null_and_global_rng():
    model=CropLatentDenoiser(width=16)
    schedule=DiffusionSchedule(8,profile="cosine_zero_terminal_v1")
    noise=torch.randn(1,16,32)
    before=noise.clone()
    rng=torch.get_rng_state().clone()
    state=model_state_sha256(model)
    a=sample_latent(model,schedule,noise,0,guidance=0,budget=4)
    b=sample_latent(model,schedule,noise,5,guidance=0,budget=4)
    assert torch.equal(a,b)
    assert torch.equal(noise,before)
    assert torch.equal(rng,torch.get_rng_state())
    assert model.training and model_state_sha256(model)==state
    assert reverse_indices(8,4)==(7,4,2,0)


def test_zero_velocity_reverse_is_finite_at_cosine_terminal():
    model=CropLatentDenoiser(width=16)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    schedule=DiffusionSchedule(8,profile="cosine_zero_terminal_v1")
    noise=torch.ones(1,16,32)
    actual=sample_latent(model,schedule,noise,0,budget=8)
    assert torch.isfinite(actual).all()
    assert actual.abs().max()<=1


@pytest.mark.parametrize("settings",[
    {"seed":True},{"bits":3},{"width":8},{"reverse_steps":33},{"codec_steps":0},
    {"diffusion_updates":0},{"training_seeds":(101,)},{"generation_seeds":(1,1)},
])
def test_plan_bounds(settings):
    with pytest.raises(CropCodecError):
        replace(LatentStudyPlan(),**settings)


@pytest.fixture(scope="module")
def study():
    plan=LatentStudyPlan(codec_steps=1,diffusion_updates=2,batch_size=2,width=16,
                         schedule_steps=8,reverse_steps=2,training_seeds=(0,),
                         validation_seeds=(101,),generation_seeds=(701,709))
    before=torch.get_rng_state().clone()
    report=run_study(plan)
    assert torch.equal(before,torch.get_rng_state())
    return plan,report


def test_end_to_end_two_stage_training_and_generation(study):
    plan,report=study
    assert report["training_clips"]==report["validation_clips"]==6
    assert report["codec"]["calibration"]["training_clips"]==6
    assert report["codec"]["frozen_during_diffusion"]
    assert report["codec"]["initial_sha256"]!=report["codec"]["trained_sha256"]
    assert report["denoiser"]["initial_sha256"]!=report["denoiser"]["trained_sha256"]
    assert report["denoiser"]["objective"]=="velocity"
    assert report["validation"]["clip_timestep_cases"]==18
    assert report["generation"]["null_control_verified"]
    assert len(report["generation"]["rows"])==36
    assert report["no_model_weights_or_audio_written"]
    assert report["original_sonumio_1"] is False
    assert report["perceptual_quality_established"] is False
    json.dumps(report,allow_nan=False)


def test_study_same_runtime_replay(study):
    plan,report=study
    assert run_study(plan)==report
