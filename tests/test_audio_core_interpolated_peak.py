"""Finite-sinc peak estimates against direct references and packet partitions."""
import json
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.interpolated_peak import (
    PeakEstimateSettings, InterpolatedPeakAccumulator, interpolation_kernels,
    estimate_interpolated_peak,
)


def brute_force(samples, settings):
    kernels = interpolation_kernels(settings)
    channels, frames = samples.shape
    offsets = np.arange(-settings.radius+1, settings.radius+1)
    values = []
    for channel in range(channels):
        best = (0., 0, 0, 0.)
        for frame in range(frames):
            candidates = [float(samples[channel, frame])]
            if frame < frames-1:
                positions = frame+offsets
                tap_values = np.zeros(len(positions))
                valid = (positions >= 0)&(positions < frames)
                tap_values[valid] = samples[channel, positions[valid]]
                candidates.extend(float(np.dot(tap_values,k)) for k in kernels)
            for phase, value in enumerate(candidates):
                candidate = (abs(value),frame,phase,value)
                if (-candidate[0],frame,phase) < (-best[0],best[1],best[2]):
                    best = candidate
        values.append(best)
    return values


@pytest.mark.parametrize("factor", [2,4,8])
@pytest.mark.parametrize("frames", [1,2,7,23,129])
def test_direct_reference(factor, frames):
    rng = np.random.default_rng(9)
    samples = rng.uniform(-.8,.8,size=(2,frames)).astype(np.float32)
    settings = PeakEstimateSettings(factor=factor, radius=8)
    audio = make_buffer(samples, 24000)
    report = estimate_interpolated_peak(audio, settings=settings, packet_frames=3)
    expected = brute_force(samples, settings)
    for actual, reference in zip(report.channel_peaks, expected):
        assert actual.amplitude == pytest.approx(reference[0], abs=1e-13)
        assert actual.signed_value == pytest.approx(reference[3], abs=1e-13)
        assert (actual.frame, actual.phase) == reference[1:3]
    assert report.estimated_peak >= report.sample_peak
    assert report.products == settings.work(2,frames)
    json.dumps(report.as_dict(), allow_nan=False)


@pytest.mark.parametrize("packet_frames", [1,7,31,256,8192])
def test_partition_invariance(packet_frames):
    rng = np.random.default_rng(40)
    audio = make_buffer(rng.uniform(-.5,.5,size=(2,1025)).astype(np.float32), 48000)
    settings = PeakEstimateSettings(factor=4,radius=16)
    a = estimate_interpolated_peak(audio, settings=settings,packet_frames=packet_frames)
    b = estimate_interpolated_peak(audio, settings=settings,packet_frames=8192)
    assert a.estimated_peak == pytest.approx(b.estimated_peak, abs=1e-13)
    assert a.strongest.frame == b.strongest.frame and a.strongest.phase == b.strongest.phase
    assert a.sample_peak == b.sample_peak
    assert a.products == b.products


def test_high_frequency_intersample_overshoot():
    # Quarter-rate sine sampled at 45-degree phase: integer samples are
    # all near 0.707 of the underlying amplitude, while midpoints recover it.
    t = np.arange(2048)
    wave = (.9*np.sin(np.pi*t/2+np.pi/4)).astype(np.float32)
    audio = make_buffer(wave, 48000)
    result = estimate_interpolated_peak(audio, settings=PeakEstimateSettings(factor=4,radius=24))
    assert result.sample_peak < .65
    assert result.estimated_peak > .87
    assert result.overshoot_db > 2.5
    assert not result.as_dict()["certified_true_peak"]


def test_silence_and_ties():
    audio = make_buffer(np.zeros((2,41),dtype=np.float32),8000)
    report = estimate_interpolated_peak(audio)
    assert report.estimated_peak_dbfs is None
    assert report.overshoot_db is None
    assert (report.strongest.channel,report.strongest.frame,report.strongest.phase)==(0,0,0)
    json.dumps(report.as_dict(),allow_nan=False)


def test_kernel_dc_normalization():
    for factor in (2,4,8):
        kernels = interpolation_kernels(PeakEstimateSettings(factor=factor))
        assert len(kernels)==factor-1
        for kernel in kernels:
            assert kernel.shape==(32,)
            assert float(kernel.sum())==pytest.approx(1,abs=1e-14)


def test_retained_halo_and_finalization():
    acc = InterpolatedPeakAccumulator(24000,1,expected_frames=1001)
    for start in range(0,1001,19):
        block=np.ones(min(19,1001-start),dtype=np.float32)*.1
        acc.update(FramePacket(24000,start,block))
        assert acc.retained_frames <= 2*acc.settings.radius
    result=acc.finish()
    assert result.frames==1001 and acc.retained_frames==0
    with pytest.raises(AudioContractError):
        acc.finish()
    with pytest.raises(AudioContractError):
        acc.update(FramePacket(24000,1001,np.ones(1,dtype=np.float32)))


def test_packet_contract_and_incomplete_finish():
    acc=InterpolatedPeakAccumulator(24000,1,expected_frames=10)
    block=np.ones(5,dtype=np.float32)
    with pytest.raises(AudioContractError):
        acc.update(FramePacket(24000,1,block))
    with pytest.raises(AudioContractError):
        acc.update(FramePacket(8000,0,block))
    acc.update(FramePacket(24000,0,block))
    with pytest.raises(AudioContractError):
        acc.finish()
    acc.update(FramePacket(24000,5,block))
    assert acc.finish().frames==10


@pytest.mark.parametrize("kwargs", [
    {"factor":True},{"factor":3},{"radius":3},{"radius":65},
    {"beta":float("nan")},{"beta":True},{"beta":21},
])
def test_invalid_settings(kwargs):
    with pytest.raises(AudioContractError):
        PeakEstimateSettings(**kwargs)


def test_work_cap(monkeypatch):
    import sonumio.audio_core.interpolated_peak as module
    monkeypatch.setattr(module,"MAX_PRODUCTS",100)
    with pytest.raises(AudioLimitError,match="work"):
        InterpolatedPeakAccumulator(24000,2,expected_frames=100)
    acc=InterpolatedPeakAccumulator(24000,2)
    with pytest.raises(AudioLimitError,match="work"):
        acc.update(FramePacket(24000,0,np.zeros((2,100),dtype=np.float32)))
    assert acc.frames_seen==0
