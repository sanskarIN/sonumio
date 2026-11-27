"""Gain staging and estimated peak guard with analytic and adverse PCM."""
import json
import math
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.interpolated_peak import PeakEstimateSettings
from sonumio.audio_core.gated_energy import GatedEnergySettings
from sonumio.audio_core.mastering import (
    MasteringSettings, MasteringMeasurement, measure_mastering, master_audio,
)


def signal(amplitude=.2,frames=12000,frequency=200,rate=24000):
    t=np.arange(frames)/rate
    return make_buffer((amplitude*np.sin(2*np.pi*frequency*t)).astype(np.float32),rate)


def test_static_rms_target_and_peak_guard():
    audio=signal()
    cfg=MasteringSettings(target_rms_dbfs=-24,max_boost_db=12)
    result=master_audio(audio,cfg)
    assert result.report.output.energy.gated_rms_dbfs==pytest.approx(-24,abs=1e-5)
    assert result.report.gated_target_error_db==pytest.approx(0,abs=1e-5)
    assert result.report.output.peak.estimated_peak<=cfg.ceiling
    assert result.report.limiter is None
    assert result.audio.spec==audio.spec
    assert not result.report.as_dict()["lufs_measured"]


def test_no_boost_by_default_and_explicit_boost_cap():
    source=signal(.01)
    report=master_audio(source,MasteringSettings(target_rms_dbfs=-12)).report
    assert report.requested_gain_db>0
    assert report.initial_gain_db==0
    result=master_audio(source,MasteringSettings(target_rms_dbfs=-12,max_boost_db=6))
    assert result.report.initial_gain_db==6
    assert result.report.output.energy.gated_rms_dbfs-result.report.source.energy.gated_rms_dbfs==pytest.approx(6,abs=1e-5)


def test_static_peak_priority_can_prevent_target():
    source=signal(.08,frames=24000)
    x=source.samples.copy()
    x[0,12000]=1.
    source=make_buffer(x,24000)
    cfg=MasteringSettings(target_rms_dbfs=-12,max_boost_db=24,estimated_peak_ceiling_dbfs=-3)
    result=master_audio(source,cfg)
    assert result.report.static_peak_guard_db<0
    assert result.report.output.peak.estimated_peak<=cfg.ceiling
    assert result.report.gated_target_error_db<0


def test_dynamic_mode_retains_more_background_than_static_for_sparse_peak():
    t=np.arange(24000)/24000
    x=(.05*np.sin(2*np.pi*200*t)).astype(np.float32)
    x[12000]=1.5
    audio=make_buffer(np.stack((x,-x)),24000)
    base=dict(target_rms_dbfs=-15,max_boost_db=24,estimated_peak_ceiling_dbfs=-1)
    static=master_audio(audio,MasteringSettings(**base))
    dynamic=master_audio(audio,MasteringSettings(**base,mode="lookahead"))
    assert dynamic.report.limiter is not None
    assert dynamic.report.limiter.attenuated_frames>0
    assert dynamic.report.output.energy.gated_rms_dbfs>static.report.output.energy.gated_rms_dbfs
    assert dynamic.report.output.peak.estimated_peak<=10**(-1/20)
    np.testing.assert_array_equal(dynamic.audio.samples[0],-dynamic.audio.samples[1])


@pytest.mark.parametrize("mode",["static","lookahead"])
def test_estimated_intersample_ceiling_on_quarter_rate_wave(mode):
    t=np.arange(4096)
    source=make_buffer((.98*np.sin(np.pi*t/2+np.pi/4)).astype(np.float32),48000)
    cfg=MasteringSettings(target_rms_dbfs=None,mode=mode,estimated_peak_ceiling_dbfs=-3)
    result=master_audio(source,cfg)
    assert result.report.source.peak.estimated_peak>source.samples.max()
    assert result.report.output.peak.estimated_peak<=cfg.ceiling
    assert result.report.static_peak_guard_db<0 or result.report.final_peak_guard_db<0


@pytest.mark.parametrize("mode",["static","lookahead"])
def test_silent_input_no_artificial_boost(mode):
    source=make_buffer(np.zeros(1024,dtype=np.float32),24000)
    result=master_audio(source,MasteringSettings(mode=mode,max_boost_db=30))
    assert not result.report.gated_signal_present
    assert result.report.requested_gain_db==0
    assert result.report.gated_target_error_db is None
    assert np.count_nonzero(result.audio.samples)==0
    json.dumps(result.report.as_dict(),allow_nan=False)


def test_packet_partition_invariance_of_pipeline():
    rng=np.random.default_rng(72)
    source=make_buffer(rng.normal(size=(2,12000)).astype(np.float32)*.1,24000)
    from dataclasses import replace
    cfg=MasteringSettings(target_rms_dbfs=-16,max_boost_db=6)
    a=master_audio(source,cfg)
    b=master_audio(source,replace(cfg,packet_frames=37))
    np.testing.assert_array_equal(a.audio.samples,b.audio.samples)
    assert a.report.output.peak.estimated_peak==pytest.approx(b.report.output.peak.estimated_peak,abs=1e-12)
    assert a.report.output.energy.gated_rms_dbfs==b.report.output.energy.gated_rms_dbfs


def test_joint_measurement_failed_state():
    m=MasteringMeasurement(8000,1,expected_frames=4)
    with pytest.raises(AudioContractError):
        m.update(FramePacket(8000,1,np.zeros(2,dtype=np.float32)))
    assert m.state=="failed"
    with pytest.raises(AudioContractError):
        m.finish()


def test_joint_measurement_finished_state():
    m=MasteringMeasurement(8000,1,expected_frames=4)
    m.update(FramePacket(8000,0,np.zeros(4,dtype=np.float32)))
    assert m.finish().peak.frames==4
    assert m.state=="finished"
    with pytest.raises(AudioContractError):
        m.finish()


@pytest.mark.parametrize("kwargs",[
    {"target_rms_dbfs":True},{"target_rms_dbfs":-2},{"max_boost_db":31},
    {"max_boost_db":float("nan")},{"mode":"unknown"},{"peak":object()},
    {"energy":object()},{"estimated_peak_ceiling_dbfs":1},{"lookahead_ms":-1},
    {"packet_frames":0},
])
def test_invalid_mastering_settings(kwargs):
    with pytest.raises(AudioContractError):
        MasteringSettings(**kwargs)


def test_mastering_budget(monkeypatch):
    import sonumio.audio_core.mastering as module
    monkeypatch.setattr(module,"MAX_MASTERING_PCM_BYTES",1)
    with pytest.raises(AudioLimitError,match="budget"):
        master_audio(signal())
