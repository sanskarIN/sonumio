"""Ratio-dependent sinc bandwidth, bounded scratch and finite PCM regressions."""
import numpy as np
import pytest

from sonumio.audio_core import resample as rs
from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError


def tone(rate, hz, frames):
    return np.sin(2 * np.pi * hz * np.arange(frames) / rate).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


@pytest.mark.parametrize("window", ["kaiser", "hann", "blackman"])
def test_extreme_downsampling_preserves_passband_and_rejects_alias(window):
    rate, target = 192000, 8000
    source = np.stack([tone(rate, 3000, 24000), tone(rate, 5000, 24000)])
    result = rs.sinc_resample(
        source, rate, target, settings=rs.ResampleSettings(target, window=window)
    )
    assert result.shape == (2, 1000)
    interior = result[:, 64:-64]
    assert rms(interior[0]) == pytest.approx(2 ** -0.5, rel=.02)
    assert rms(interior[1]) < .01
    assert result.dtype == np.float32


@pytest.mark.parametrize("source_rate,target", [(48000, 8000), (44100, 16000), (8000, 48000)])
def test_constant_and_channel_isolation_even_at_edges(source_rate, target):
    data = np.stack([np.full(1001, .25), np.full(1001, -.5)]).astype(np.float32)
    result = rs.sinc_resample(data, source_rate, target)
    assert result.shape == (2, rs.output_frames(1001, source_rate, target))
    np.testing.assert_allclose(result[0], .25, atol=1e-6)
    np.testing.assert_allclose(result[1], -.5, atol=1e-6)


def test_kernel_cell_budget_and_batch_partition_invariance(monkeypatch):
    source = np.random.default_rng(9).normal(0, .1, (2, 5000)).astype(np.float32)
    real_window = rs._window
    shapes = []

    def tracked(distances, *args):
        shapes.append(distances.shape)
        assert distances.size <= rs._MAX_KERNEL_CELLS
        return real_window(distances, *args)

    monkeypatch.setattr(rs, "_window", tracked)
    small = rs.sinc_resample(source, 192000, 8000, settings=rs.ResampleSettings(
        8000, lobes=128, chunk_frames=64
    ))
    large = rs.sinc_resample(source, 192000, 8000, settings=rs.ResampleSettings(
        8000, lobes=128, chunk_frames=65536
    ))
    assert len(shapes) > 2
    assert max(rows for rows, _ in shapes) < 64
    np.testing.assert_array_equal(small, large)


def test_fractional_ratio_chunk_independence():
    data = np.random.default_rng(11).normal(0, .1, 3000).astype(np.float32)
    a = rs.sinc_resample(data, 44100, 16000, settings=rs.ResampleSettings(
        16000, chunk_frames=64
    ))
    b = rs.sinc_resample(data, 44100, 16000, settings=rs.ResampleSettings(
        16000, chunk_frames=8192
    ))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", [rs.linear_resample, rs.sinc_resample])
@pytest.mark.parametrize("target", [8000, 16000])
def test_unrepresentable_input_rejected_even_for_same_clock(method, target):
    data = np.array([0, 1e100], dtype=np.float64)
    with pytest.raises(AudioContractError, match="finite float32"):
        method(data, 8000, target)


def test_sinc_overshoot_is_rejected_without_returning_infinity():
    limit = np.finfo(np.float32).max
    data = np.array([-limit, limit, limit, -limit], dtype=np.float32)
    with pytest.raises(AudioContractError, match="finite float32"):
        rs.sinc_resample(data, 8000, 32000)


@pytest.mark.parametrize("method", [rs.linear_resample, rs.sinc_resample])
def test_working_allocation_is_validated_before_widening(monkeypatch, method):
    import sonumio.audio_core.contracts as contracts
    original = contracts.validate_allocation
    def bounded(channels, frames, **kwargs):
        return original(channels, frames, **{**kwargs, "max_bytes": 48})
    monkeypatch.setattr(rs, "validate_allocation", bounded)
    with pytest.raises(AudioLimitError):
        method(np.zeros(8, dtype=np.float32), 8000, 16000)


def test_linear_partition_is_invariant_and_alias_policy_remains_explicit(monkeypatch):
    data = np.random.default_rng(2).normal(0, .1, (2, 400)).astype(np.float32)
    reference = rs.linear_resample(data, 8000, 44100)
    monkeypatch.setattr(rs, "_LINEAR_BATCH_FRAMES", 37)
    np.testing.assert_array_equal(rs.linear_resample(data, 8000, 44100), reference)
    with pytest.raises(AudioContractError, match="aliases"):
        rs.linear_resample(data, 44100, 8000)
    assert rs.linear_resample(data, 44100, 8000, allow_aliasing=True).shape == (
        2, rs.output_frames(400, 44100, 8000)
    )


def test_same_clock_report_does_not_claim_filtering():
    audio = make_buffer(np.arange(8, dtype=np.float32) / 8, 8000)
    settings = rs.ResampleSettings(8000)
    converted = rs.resample_audio(audio, settings)
    np.testing.assert_array_equal(audio.samples, converted.samples)
    assert rs.resample_report(audio, converted, settings)["anti_alias_filter_applied"] is False


def test_invalid_settings_rejected_with_contract_error():
    with pytest.raises(AudioContractError):
        rs.sinc_resample(np.zeros(32, dtype=np.float32), 8000, 16000, settings={})
