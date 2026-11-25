"""Regression coverage for numerical PCM contracts across NumPy versions."""
import math
import numpy as np
import pytest

from sonumio.audio_core import contracts, numerics
from sonumio.audio_core.contracts import (
    AudioBuffer, AudioLayout, AudioSpec, as_channel_first, check_audio,
    finite_float, make_buffer,
)
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket


@pytest.mark.parametrize("value", [10**1000, -(10**1000), float("nan"), float("inf"), True])
def test_finite_scalar_error_taxonomy(value):
    with pytest.raises(AudioContractError, match="finite number"):
        finite_float(value, "test")


@pytest.mark.parametrize("sign", [-1, 1])
def test_finite_double_cannot_poison_float32_buffers(sign):
    samples = np.array([0, sign * 1e100], dtype=np.float64)
    check_audio(samples)  # Valid double, but not valid canonical float32.
    with np.errstate(over="raise"):
        for convert in (
            lambda: as_channel_first(samples),
            lambda: make_buffer(samples, 8000),
            lambda: AudioBuffer(samples, AudioSpec(8000, AudioLayout.for_channels(1), 2)),
            lambda: FramePacket(8000, 0, samples),
        ):
            with pytest.raises(AudioContractError, match="finite float32"):
                convert()
    np.testing.assert_array_equal(as_channel_first(samples, float64=True)[0], samples)


def test_float32_limit_and_overrange_are_not_clipped():
    limit = float(np.finfo(np.float32).max)
    data = np.array([-limit, -2, 0, 2, limit], dtype=np.float64)
    out = as_channel_first(data)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[0], data.astype(np.float32))


def test_copy_false_is_opportunistic_across_numpy_versions():
    source = np.arange(20, dtype=np.float32).reshape(2, 10)
    shared = as_channel_first(source, copy=False)
    assert np.shares_memory(source, shared)
    converted = as_channel_first(source, copy=False, float64=True)
    assert converted.dtype == np.float64
    assert not np.shares_memory(source, converted)
    sliced = source[:, ::2]
    contiguous = as_channel_first(sliced, copy=False)
    assert contiguous.flags.c_contiguous
    np.testing.assert_array_equal(contiguous, sliced)
    transposed = as_channel_first(source.T.copy(), channels_last=True, copy=False)
    assert transposed.flags.c_contiguous
    np.testing.assert_array_equal(transposed, source)


def test_ownership_preserved_after_opportunistic_conversion():
    source = np.arange(12, dtype=np.float32)
    buf = make_buffer(source, 8000)
    source[:] = -3
    assert buf.samples[0, 8] == 8
    assert not np.shares_memory(source, buf.samples)
    with pytest.raises(ValueError):
        buf.samples.setflags(write=True)


def test_target_allocation_validated_before_widening(monkeypatch):
    source = np.zeros(8, dtype=np.float32)
    buf = make_buffer(source, 8000)
    actual = contracts.validate_allocation
    def small_budget(channels, frames, **kwargs):
        return actual(channels, frames, **{**kwargs, "max_bytes": 48})
    monkeypatch.setattr(contracts, "validate_allocation", small_budget)
    with pytest.raises(AudioLimitError):
        as_channel_first(source, float64=True)
    with pytest.raises(AudioLimitError):
        buf.mutable_copy(dtype="float64")


def test_invalid_explicit_layout_is_contract_error():
    with pytest.raises(AudioContractError, match="AudioLayout"):
        make_buffer(np.zeros(4, dtype=np.float32), 8000, layout={})


@pytest.mark.parametrize("scale", [1e300, 1e-300, 1., 1e-30])
def test_rms_handles_extreme_amplitudes_without_squaring_overflow(scale):
    samples = np.array([[scale, -scale, 0, 0], [0, 0, 0, 0]], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        assert numerics.stable_peak(samples) == scale
        assert numerics.stable_rms(samples) / scale == pytest.approx(.5, rel=1e-14)
        np.testing.assert_allclose(
            numerics.stable_rms(samples, per_channel=True) / scale,
            [math.sqrt(.5), 0], rtol=1e-14, atol=0,
        )


def test_max_finite_and_silence_rms():
    limit = np.finfo(np.float64).max
    assert numerics.stable_rms(np.array([limit, -limit])) == limit
    assert numerics.stable_rms(np.zeros((2, 11), dtype=np.float32)) == 0


def test_rms_noncontiguous_partition_matches_reference(monkeypatch):
    samples = np.random.default_rng(12).normal(size=(3, 200))[:, ::2]
    reference = np.sqrt(np.mean(samples**2, axis=1))
    monkeypatch.setattr(numerics, "_ARITHMETIC_BATCH_FRAMES", 7)
    np.testing.assert_allclose(numerics.stable_rms(samples, per_channel=True),
                               reference, rtol=1e-14)
    assert numerics.stable_rms(samples) == pytest.approx(np.sqrt(np.mean(samples**2)))


def test_mixing_is_batched_without_widening_whole_sources(monkeypatch):
    a = np.arange(40, dtype=np.float32).reshape(2, 20)
    b = np.ones((2, 12), dtype=np.float64)
    reference = numerics.mix_accumulate([a, b], gains_db=[0, 6], cap_frames=17)
    monkeypatch.setattr(numerics, "_ARITHMETIC_BATCH_FRAMES", 3)
    result = numerics.mix_accumulate([a, b], gains_db=[0, 6], cap_frames=17)
    np.testing.assert_array_equal(result, reference)
    expected = a[:, :17].astype(np.float64)
    expected[:, :12] += 10**(6/20)
    np.testing.assert_array_equal(result, expected.astype(np.float32))
    assert a[0, 0] == 0 and b[0, 0] == 1


def test_many_aliasing_sources_mix_without_retained_widened_copies():
    source = np.ones((2, 120), dtype=np.float32)
    result = numerics.mix_accumulate([source] * 64, normalize="equal_power")
    np.testing.assert_array_equal(result, np.full((2, 120), 8, dtype=np.float32))


def test_mix_rejects_float64_overflow_and_float32_narrowing():
    huge = np.full(8, np.finfo(np.float64).max)
    with pytest.raises(AudioContractError, match="finite float64"):
        numerics.mix_accumulate([huge, huge])
    with pytest.raises(AudioContractError, match="finite float32"):
        numerics.mix_accumulate([np.full(8, 1e100)])
    np.testing.assert_array_equal(
        numerics.mix_accumulate([np.full(8, 1e100)], normalize="peak"),
        np.ones((1, 8), dtype=np.float32),
    )
