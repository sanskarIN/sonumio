"""Explicit unweighted energy gates, exact windows and stream boundaries."""
import json
import math
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.gated_energy import (
    GatedEnergySettings, GatedEnergyAccumulator, measure_gated_energy,
    summarize_energy,
)


@pytest.mark.parametrize("amplitude", [.01,.1,.25,.5])
@pytest.mark.parametrize("channels", [1,2,6])
def test_constant_energy_dbfs(amplitude,channels):
    audio=make_buffer(np.full((channels,4001),amplitude,dtype=np.float32),8000)
    result=measure_gated_energy(audio,packet_frames=31)
    expected=20*math.log10(amplitude)
    assert result.overall_rms_dbfs==pytest.approx(expected,abs=1e-6)
    assert result.gated_rms_dbfs==pytest.approx(expected,abs=1e-6)
    assert result.relative_pass_windows==result.full_windows
    assert result.omitted_tail_frames==1
    assert not result.as_dict()["lufs_measured"]


def test_two_stage_gating_linear_energy():
    settings=GatedEnergySettings(absolute_gate_dbfs=-50,relative_gate_db=-10)
    powers=np.array([0,1e-6,1e-4,.01,.09])
    absolute,relative,absolute_db,threshold,gated=summarize_energy(powers,settings=settings)
    assert absolute==3 and relative==2
    average=(.0001+.01+.09)/3
    assert absolute_db==pytest.approx(10*math.log10(average))
    assert threshold==pytest.approx(10*math.log10(average)-10)
    assert gated==pytest.approx(10*math.log10(.05))


def test_silence_has_no_fabricated_db_value():
    audio=make_buffer(np.zeros(5000,dtype=np.float32),8000)
    result=measure_gated_energy(audio)
    assert result.overall_rms_dbfs is None
    assert result.gated_rms_dbfs is None
    assert result.absolute_pass_windows==result.relative_pass_windows==0
    assert result.active_window_fraction==0
    json.dumps(result.as_dict(),allow_nan=False)


@pytest.mark.parametrize("frames", [1,19,3199,3200,3201,3999,4000,4001])
def test_exact_window_geometry(frames):
    audio=make_buffer(np.ones(frames,dtype=np.float32)*.2,8000)
    r=measure_gated_energy(audio,packet_frames=13)
    full=0 if frames<3200 else 1+(frames-3200)//800
    assert r.full_windows==full
    assert r.short_window_used==(full==0)
    assert r.omitted_tail_frames==(0 if full==0 else frames-(3200+(full-1)*800))
    assert r.absolute_pass_windows==max(1,full)


@pytest.mark.parametrize("packet_frames",[1,17,513,8192])
def test_partition_matches_offline_window_reference(packet_frames):
    rng=np.random.default_rng(56)
    x=rng.uniform(-.7,.7,size=(2,10001)).astype(np.float32)
    audio=make_buffer(x,8000)
    cfg=GatedEnergySettings(window_ms=150,hop_ms=31)
    a=measure_gated_energy(audio,settings=cfg,packet_frames=packet_frames)
    b=measure_gated_energy(audio,settings=cfg,packet_frames=8192)
    assert a.gated_rms_dbfs==b.gated_rms_dbfs
    assert a.maximum_window_dbfs==b.maximum_window_dbfs
    assert a.overall_rms_dbfs==pytest.approx(b.overall_rms_dbfs,abs=1e-12)
    window,hop=cfg.geometry(8000)
    energy=np.mean(x.astype(np.float64)**2,axis=0)
    powers=np.array([energy[i:i+window].mean() for i in range(0,len(energy)-window+1,hop)])
    expected=summarize_energy(powers,settings=cfg)
    assert a.relative_pass_windows==expected[1]
    assert a.gated_rms_dbfs==expected[-1]


def test_antiphase_not_cancelled():
    t=np.arange(8000)/8000
    wave=(.2*np.sin(2*np.pi*100*t)).astype(np.float32)
    a=measure_gated_energy(make_buffer(wave,8000))
    b=measure_gated_energy(make_buffer(np.stack((wave,-wave)),8000))
    assert a.gated_rms_dbfs==pytest.approx(b.gated_rms_dbfs,abs=1e-12)


def test_accumulator_state_and_retention():
    acc=GatedEnergyAccumulator(8000,1,expected_frames=4000)
    for start in range(0,4000,23):
        acc.update(FramePacket(8000,start,np.full(min(23,4000-start),.2,dtype=np.float32)))
        assert acc.retained_frames<acc.window
    assert acc.finish().frames==4000
    with pytest.raises(AudioContractError):
        acc.finish()


def test_packet_contracts():
    acc=GatedEnergyAccumulator(8000,1,expected_frames=20)
    x=np.ones(10,dtype=np.float32)
    with pytest.raises(AudioContractError):
        acc.update(FramePacket(8000,1,x))
    with pytest.raises(AudioContractError):
        acc.update(FramePacket(24000,0,x))
    acc.update(FramePacket(8000,0,x))
    with pytest.raises(AudioContractError):
        acc.finish()
    acc.update(FramePacket(8000,10,x))
    assert acc.finish().frames==20


@pytest.mark.parametrize("kwargs",[
    {"window_ms":True},{"window_ms":49},{"hop_ms":0},{"hop_ms":401},
    {"absolute_gate_dbfs":-121},{"relative_gate_db":1},{"relative_gate_db":float("nan")},
])
def test_invalid_settings(kwargs):
    with pytest.raises(AudioContractError):
        GatedEnergySettings(**kwargs)


def test_work_cap(monkeypatch):
    import sonumio.audio_core.gated_energy as module
    monkeypatch.setattr(module,"MAX_WINDOW_PRODUCTS",1)
    with pytest.raises(AudioLimitError,match="work"):
        GatedEnergyAccumulator(8000,1,expected_frames=4000)
    acc=GatedEnergyAccumulator(8000,1)
    with pytest.raises(AudioLimitError,match="work"):
        acc.update(FramePacket(8000,0,np.zeros(4000,dtype=np.float32)))
    assert acc.frames_seen==0
