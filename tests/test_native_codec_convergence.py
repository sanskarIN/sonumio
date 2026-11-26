"""Phase-sensitive codec objective and controlled convergence regression tests."""
import json
import math
import torch
import pytest

from research.native_codec import FRAMES, SAMPLE_RATE, Native24kCodec, native_loss
from research.native_codec_convergence import (
    ARMS, ConvergencePlan, phase_components, objective, score, summarize, run_study,
)
from research.native_codec_study import dataset
from research.wave_diffusion import CATEGORIES


def tone(hz=7000, amplitude=.2):
    t=torch.arange(FRAMES,dtype=torch.float64)/SAMPLE_RATE
    return (amplitude*torch.sin(2*math.pi*hz*t)).float().reshape(1,1,FRAMES)


@pytest.mark.parametrize("amplitude",[0.,1e-7,1e-4,.2])
@pytest.mark.parametrize("arm",ARMS)
def test_exact_identity_loss_and_finite_gradients(amplitude,arm):
    source=tone(amplitude=amplitude)
    restored=source.clone().requires_grad_()
    loss=objective(source,restored,arm)
    assert loss.item()==0
    loss.backward()
    assert torch.isfinite(restored.grad).all()
    assert torch.count_nonzero(restored.grad)==0


def test_phase_oracle_detects_polarity_even_with_equal_magnitudes():
    source=tone()
    values=phase_components(source,-source)
    assert values["normalized_waveform_mse"].item()==pytest.approx(4.,rel=1e-6)
    assert values["complex_spectral_error"].item()==pytest.approx(4.,rel=1e-6)
    assert values["complex_highband_error"].item()>3.99
    assert native_loss(source,-source,.5)>0
    assert objective(source,-source,"balanced_phase")>objective(source,-source,"balanced_wave")


@pytest.mark.parametrize("hz",[1000,7000])
def test_silence_control_analytic(hz):
    source=tone(hz)
    stats=phase_components(source,torch.zeros_like(source))
    assert stats["normalized_waveform_mse"].item()==pytest.approx(1.,rel=1e-6)
    assert stats["complex_spectral_error"].item()==pytest.approx(1.,rel=1e-6)
    if hz==1000:
        assert stats["complex_highband_error"].item()<.001
    else:
        assert stats["complex_highband_error"].item()>.99


def test_amplitude_normalization_does_not_reward_quiet_errors():
    high=tone(amplitude=.2)
    low=tone(amplitude=.02)
    a=phase_components(high,.5*high)
    b=phase_components(low,.5*low)
    for key in ("normalized_waveform_mse","complex_spectral_error","complex_highband_error"):
        torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-8)


def test_original_pilot_objective_reproduced():
    x=tone()
    y=.7*x
    torch.testing.assert_close(objective(x,y,"pilot_highband"),native_loss(x,y,.5),
                               rtol=0,atol=0)


def test_control_scoring_distinguishes_silence_and_identity():
    waves=torch.cat([tone(500+i*1000) for i in range(len(CATEGORIES))])
    labels=tuple(range(len(CATEGORIES)))
    silence=score(waves,labels,control=torch.zeros_like)
    identity=score(waves,labels,control=lambda x:x)
    assert silence["silence_wins"]==0 and silence["silence_ties"]==6
    assert silence["median_relative_mse_to_silence"]==1
    assert identity["silence_wins"]==6 and identity["silence_ties"]==0
    assert identity["positive_snr_clips"]==6
    assert identity["median_relative_mse_to_silence"]==0
    assert identity["mean"]["complex_highband_error"]==0


@pytest.mark.parametrize("settings",[
    {"checkpoints":(32,64)},{"checkpoints":(32,32,128)},
    {"checkpoints":(64,32,128)},{"checkpoints":(0,128)},
    {"checkpoints":[32,128]},{"updates":257},
    {"batch_size":True},{"validation_seeds":(0,)},
    {"model_seeds":(1,1)},{"model_seeds":(1,2,3,4),"updates":256,"batch_size":8,
                           "checkpoints":(256,)},
])
def test_plan_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        ConvergencePlan(**settings)


def test_development_gate_requires_all_conditions():
    plan=ConvergencePlan(model_seeds=(17,29),updates=1,checkpoints=(1,))
    silence={"mean":{"complex_highband_error":.2}}
    def evaluation(snr,wins,high):
        return {"mean":{"snr_db":snr,"complex_highband_error":high},
                "silence_wins":wins,"clips":12}
    good=evaluation(2.,9,.1)
    runs=[{"seed":seed,"arms":{arm:[{"step":1,"evaluation":good}] for arm in ARMS}}
          for seed in plan.model_seeds]
    assert all(x["development_reconstruction_gate_passed"] for x in summarize(runs,plan,silence).values())
    for bad in (evaluation(-.1,9,.1),evaluation(2.,7,.1),evaluation(2.,9,.2)):
        runs[0]["arms"]["balanced_phase"][0]["evaluation"]=bad
        assert not summarize(runs,plan,silence)["balanced_phase"]["development_reconstruction_gate_passed"]


def test_convergence_paired_smoke():
    plan=ConvergencePlan(model_seeds=(17,29),training_seeds=(0,),
                         validation_seeds=(201,),updates=2,batch_size=2,checkpoints=(1,2))
    rng=torch.get_rng_state().clone()
    report=run_study(plan)
    assert torch.equal(torch.get_rng_state(),rng)
    assert report["train_examples"]==24
    assert report["training_clips"]==report["development_clips"]==6
    assert report["matched_architecture_initialization_and_batches"]
    assert not report["independent_test_scored"]
    assert not report["perceptual_quality_established"]
    for run in report["runs"]:
        assert run["initial"]["model_sha256"]==run["initial_model_sha256"]
        assert tuple(run["arms"])==ARMS
        for snapshots in run["arms"].values():
            assert [p["step"] for p in snapshots]==[1,2]
            assert snapshots[-1]["evaluation"]["model_sha256"]!=run["initial_model_sha256"]
            assert 0<=snapshots[-1]["evaluation"]["silence_wins"]<=6
    json.dumps(report,allow_nan=False)
