"""Bounded PCM arithmetic, float32 quantization and exact windows."""
from __future__ import annotations

import math
import numpy as np
import pytest

from sonumio.audio_core.numerics import (
    Decibel, balance_stereo, db_to_gain, dbfs, equal_power_ramps,
    finite_clamp, gain_to_db, lerp, linear_ramp, mid_side_decode,
    mid_side_encode, mix_accumulate, saturate, stable_peak, stable_rms,
)
from sonumio.audio_core.windows import (
    blackman, envelope, hamming, hann, overlap_envelope, sqrt_hann,
    tukey, validate_cola, weighted_overlap_add, window_bank,
)
from sonumio.audio_core.dither import (
    QuantizationSettings, quantize_pcm, pack_pcm, unpack_pcm,
    dequantize_pcm, encode_pcm_bytes, decode_pcm_bytes,
)
from sonumio.audio_core.errors import AudioContractError, AudioFormatError


def test_decibel_conversions_and_silence_floor():
    assert db_to_gain(0) == 1.0
    assert db_to_gain(20) == pytest.approx(10)
    assert db_to_gain(-6) == pytest.approx(.501187, rel=1e-5)
    assert gain_to_db(1) == 0
    assert gain_to_db(0) == -300
    assert dbfs(.1) == pytest.approx(-20)
    assert Decibel(-3).linear == pytest.approx(db_to_gain(-3))
    assert (Decibel(-3) + Decibel(3)).value == 0
    for bad in (float("nan"), float("inf"), -400):
        with pytest.raises(AudioContractError):
            db_to_gain(bad)


def test_explicit_peak_rms_and_multichannel_weight():
    wave = np.array([[1, 0, -1, 0], [0, 0, 0, 0]], dtype=np.float32)
    assert stable_peak(wave) == 1
    np.testing.assert_allclose(stable_peak(wave, per_channel=True), [1, 0])
    assert stable_rms(wave) == pytest.approx(math.sqrt(.25))
    np.testing.assert_allclose(stable_rms(wave, per_channel=True),
                               [math.sqrt(.5), 0])


@pytest.mark.parametrize("kind", ["hard", "tanh", "cubic"])
def test_saturation_returns_finite_float32_bounds(kind):
    source = np.array([-9, -1, 0, 1, 9], dtype=np.float32)
    result = saturate(source, mode=kind, threshold=.8)
    assert result.dtype == np.float32
    assert result.shape == (1, 5)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 1.2
    assert result[0, 2] == pytest.approx(0)


def test_mix_explicit_gains_and_length_padding():
    a = np.ones(8, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    mixed = mix_accumulate([a, b], gains_db=[0, -6.020599913])
    assert mixed.shape == (1, 8)
    np.testing.assert_allclose(mixed[0, :4], 1.5, atol=1e-6)
    np.testing.assert_allclose(mixed[0, 4:], 1.0, atol=1e-6)
    limited = mix_accumulate([a, a], normalize="peak")
    assert np.max(limited) == 1
    with pytest.raises(AudioContractError):
        mix_accumulate([a, np.ones((2, 4), dtype=np.float32)])


def test_equal_power_fade_and_lerp_bounds():
    a, b = equal_power_ramps(128)
    np.testing.assert_allclose(a*a + b*b, 1, atol=1e-12)
    assert a[0] == pytest.approx(1) and b[-1] == pytest.approx(1)
    x = np.array([0, 2], dtype=np.float32)
    y = np.array([2, 0], dtype=np.float32)
    np.testing.assert_array_equal(lerp(x, y, .5), [1, 1])
    with pytest.raises(AudioContractError):
        lerp(x, y, 2)
    with pytest.raises(AudioContractError):
        equal_power_ramps(1)
    np.testing.assert_allclose(linear_ramp(5), np.linspace(0, 1, 5))


def test_mid_side_is_reversible_and_balance_is_bounded():
    stereo = np.array([[.5, 1., -.5], [.1, -.5, .5]], dtype=np.float32)
    mid, side = mid_side_encode(stereo)
    recovered = mid_side_decode(mid, side)
    np.testing.assert_allclose(recovered, stereo, atol=1e-7)
    fully_left = balance_stereo(stereo, -1)
    np.testing.assert_allclose(fully_left[1], 0, atol=1e-7)
    fully_right = balance_stereo(stereo, 1)
    np.testing.assert_allclose(fully_right[0], 0, atol=1e-7)


def test_window_endpoints_symmetry_and_periodic_variance():
    for maker in (hann, hamming, blackman):
        data = maker(128)
        np.testing.assert_allclose(data, data[::-1], atol=1e-12)
        assert np.isfinite(data).all()
        assert len(data) == 128
    assert hann(1)[0] == 1
    assert hann(128)[0] == 0
    assert hann(128)[-1] == 0
    assert hann(128, periodic=True)[-1] > 0
    assert hamming(128)[0] > 0
    assert np.all(sqrt_hann(64) >= 0)
    np.testing.assert_allclose(tukey(64, alpha=0), 1)
    np.testing.assert_allclose(tukey(64, alpha=1), hann(64))


def test_window_bank_returns_mutable_copies_not_cached_shared_buffers():
    x = window_bank("hann", 100, periodic=True)
    y = window_bank("hann", 100, periodic=True)
    x[:] = -99
    assert np.min(y) >= 0
    with pytest.raises(AudioContractError):
        window_bank("evil-code", 100)


@pytest.mark.parametrize("curve", ["linear", "cosine", "smoothstep", "equal_power"])
def test_envelope_fades_are_bounded_and_dont_multiply_overlaps(curve):
    e = envelope(100, fade_in=20, fade_out=20, curve=curve)
    assert e[0] == pytest.approx(0)
    assert e[-1] == pytest.approx(0)
    assert np.all((e >= -1e-12) & (e <= 1 + 1e-12))
    np.testing.assert_allclose(e[30:70], 1, atol=1e-12)
    with pytest.raises(AudioContractError):
        envelope(20, fade_in=15, fade_out=15)


def test_cola_hann_and_non_constant_overlap():
    window = hann(128, periodic=True)
    assert validate_cola(window, 64)
    env = overlap_envelope(window, 64)
    assert np.max(env) == pytest.approx(np.min(env), abs=1e-9)
    assert not validate_cola(window, 80)


def test_weighted_overlap_add_recovers_nonzero_window_interiors():
    n = 128
    window = hann(n, periodic=True)
    # Frame arrays already analysis-windowed; OLA removes the second window
    source = np.ones(256, dtype=np.float32)
    frames = [
        (source[pos:pos+n] * window).astype(np.float32)
        for pos in range(0, 256 - n + 1, 64)
    ]
    output = weighted_overlap_add(frames, hop=64, window=window)
    assert output.shape == (256,)
    np.testing.assert_allclose(output[64:192], 1, atol=1e-5)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_pcm_integer_quantization_and_dequantization(bits):
    source = np.array([[-1, -.5, 0, .5, 1]], dtype=np.float32)
    settings = QuantizationSettings(bits=bits)
    integers = quantize_pcm(source, settings)
    assert integers.dtype == np.int64
    assert int(integers[0, 0]) == -(1 << (bits - 1))
    assert int(integers[0, -1]) == (1 << (bits - 1)) - 1
    payload = pack_pcm(integers, bits=bits)
    assert len(payload) == source.size * bits // 8
    recovered = unpack_pcm(payload, bits=bits, channels=1)
    np.testing.assert_array_equal(recovered, integers)
    decoded = dequantize_pcm(recovered, bits=bits)
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, source, atol=2 / (1 << (bits - 1)))


def test_tpdf_dither_requires_seed_and_is_byte_deterministic():
    with pytest.raises(AudioContractError):
        QuantizationSettings(bits=16, dither=True)
    a = QuantizationSettings(bits=16, dither=True, seed=15)
    b = QuantizationSettings(bits=16, dither=True, seed=16)
    audio = np.linspace(-.8, .8, 200, dtype=np.float32)
    assert encode_pcm_bytes(audio, settings=a) == encode_pcm_bytes(audio, settings=a)
    assert encode_pcm_bytes(audio, settings=a) != encode_pcm_bytes(audio, settings=b)


def test_strict_pcm_shape_and_integer_underflow_rejected():
    with pytest.raises(AudioFormatError):
        QuantizationSettings(bits=20)
    with pytest.raises(AudioContractError):
        unpack_pcm(b"\x00\x01\x02", bits=16, channels=1)
    with pytest.raises(AudioContractError):
        dequantize_pcm(np.array([35000], dtype=np.int64), bits=16)
    with pytest.raises(AudioContractError):
        pack_pcm(np.array([-40000], dtype=np.int64), bits=16)
    with pytest.raises(AudioContractError):
        finite_clamp(0, 2, 1)
