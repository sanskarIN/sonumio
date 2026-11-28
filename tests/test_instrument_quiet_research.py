"""Quiet-region objectives: exact baseline parity and actual matched training."""
from dataclasses import replace
import json

import pytest

torch=pytest.importorskip("torch")

from research.instrument_harmonic_model import instrument_loss
from research.instrument_long_data import LongGrid
from research.instrument_quiet_loss import (
    ARMS, POWER_FLOOR, loss_components, relative_gradient_bound, energy_diagnostics,
)
from research.instrument_quiet_study import StudyPlan,run_study


def pair():
    t=torch.arange(2048,dtype=torch.float32)/24000
    wave=torch.sin(2*torch.pi*440*t)
    return torch.stack((wave*.1,wave*.0001)),torch.stack((wave*.2,wave*.0002))


def test_baseline_matches_original_exact_loss():
    predicted,target=pair()
    actual,_=loss_components(predicted,target,"baseline")
    torch.testing.assert_close(actual,instrument_loss(predicted,target),rtol=0,atol=0)


def test_floor_reduces_quiet_relative_multiplier_without_changing_loud():
    _,target=pair()
    base=relative_gradient_bound(target,1e-8)
    floor=relative_gradient_bound(target,POWER_FLOOR)
    assert base[0]==floor[0]
    assert base[1]>floor[1]
    assert floor[1].item()==pytest.approx(1/POWER_FLOOR)


@pytest.mark.parametrize("arm",ARMS)
def test_silent_targets_and_gradients_are_finite(arm):
    predicted,target=pair()
    predicted=predicted.detach().requires_grad_()
    target[1]=0
    loss,parts=loss_components(predicted,target,arm)
    loss.backward()
    assert bool(torch.isfinite(predicted.grad).all())
    assert all(bool(torch.isfinite(value)) for value in parts.values())
    assert parts["floor_fraction"].item()==.5


def test_floor_changes_only_waveform_normalization():
    predicted,target=pair()
    baseline,b=loss_components(predicted,target,"baseline")
    floor,f=loss_components(predicted,target,"floor")
    extended,e=loss_components(predicted,target,"floor-envelope")
    assert floor<baseline
    assert f["spectral"]==b["spectral"]==e["spectral"]
    assert f["absolute_mse"]==b["absolute_mse"]==e["absolute_mse"]
    assert f["reference_power"]==b["reference_power"]
    assert extended>floor
    torch.testing.assert_close(extended-floor,.35*e["envelope"])


def test_diagnostics_keep_absolute_and_original_relative_error():
    predicted,target=pair()
    result=energy_diagnostics(predicted,target)
    assert result["relative_mse"][1]>result["floored_mse"][1]
    assert result["relative_mse"][0]==result["floored_mse"][0]
    assert result["absolute_mse"][0]>result["absolute_mse"][1]
    assert result["rms_difference"][0]>result["rms_difference"][1]


@pytest.mark.parametrize("arm",["",None,True,"unknown"])
def test_invalid_arm(arm):
    predicted,target=pair()
    with pytest.raises(ValueError):
        loss_components(predicted,target,arm)


def test_prespecified_matched_study_updates_every_arm():
    grid=LongGrid(training_pitches=(60,),evaluation_pitches=(66,),
                  training_velocities=(500,),evaluation_velocities=(750,),
                  training_frames=(12000,),evaluation_frames=(16000,))
    plan=StudyPlan(grid=grid,seeds=(17,29),width=32,updates=3,batch_size=2,panels=("joint",))
    result=run_study(plan)
    assert len(result["replicates"])==2
    assert result["power_floor"]==POWER_FLOOR
    assert result["same_architecture"] is True
    for replica in result["replicates"]:
        assert set(replica["trained"])==set(ARMS)
        for report in replica["training"].values():
            assert report["initial_sha256"]==replica["initial_sha256"]
            assert report["trained_sha256"]!=report["initial_sha256"]
            assert report["schedule_sha256"]==result["schedule"]["sha256"]
            assert report["updates"]==3
            assert 0<=report["mean"]["gradient_clipped"]<=1
        assert replica["wrong"]["joint"]["wrong_labels_per_note"]==4
    assert "absolute_wave_mse" in result["summary"]["joint"]["release"]
    assert result["evaluation_used_for_training"] is False
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("changes",[
    {"seeds":(17,)},{"seeds":(17,17)},{"width":24},{"updates":0},
    {"panels":("train",)},{"batch_size":0},
])
def test_invalid_study(changes):
    with pytest.raises(ValueError):
        replace(StudyPlan(),**changes)
