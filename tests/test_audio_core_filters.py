"""Stable bounded frequency filtering and stereo-linked dynamics."""
from __future__ import annotations

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.filters import (
    Biquad, BiquadState, FilterSpec, biquad_filter, dc_block,
    design_biquad, design_fir_highpass, design_fir_lowpass, fir_filter,
)
from sonumio.audio_core.dynamics import (
    CompressorConfig, GateConfig, compress, dynamic_range_report,
    envelope_follower, noise_gate, peak_limit, peak_normalize, rms_normalize,
)
from sonumio.audio_core.numerics import stable_peak, stable_rms
from sonumio.audio_core.errors import AudioContractError


@pytest.fixture
def tones():
    sr = 24000
    time = np.arange(6000, dtype=np.float32) / sr
    low = .4 * np.sin(2 * np.pi * 100 * time)
    high = .4 * np.sin(2 * np.pi * 5000 * time)
    return make_buffer(np.stack((low + high, low - high)), sr)


@pytest.mark.parametrize("kind", [
    "lowpass", "highpass", "bandpass", "notch",
    "allpass", "peaking", "lowshelf", "highshelf"
])
def test_rbq_biquad_types_are_stable_and_finite(kind):
    config = FilterSpec(24000, 1000, kind, q=.707, gain_db=6)
    coefficients = design_biquad(config)
    assert isinstance(coefficients, Biquad)
    source = np.ones((2, 300), dtype=np.float32) * .1
    state = BiquadState(coefficients, 2)
    result = state.process(source)
    assert result.shape == source.shape
    assert np.isfinite(result).all()


def test_biquad_chunk_state_is_equivalent_to_full_causal_pass(tones):
    spec = FilterSpec(tones.sample_rate, 900, "lowpass")
    coefficients = design_biquad(spec)
    first = BiquadState(coefficients, 2)
    entire = first.process(tones.samples)
    second = BiquadState(coefficients, 2)
    a = second.process(tones.samples[:, :3000])
    b = second.process(tones.samples[:, 3000:])
    np.testing.assert_allclose(np.concatenate((a, b), axis=1),
                               entire, atol=1e-6)
    second.reset()
    fresh = second.process(tones.samples)
    np.testing.assert_allclose(fresh, entire, atol=1e-6)


def test_lowpass_attenuates_treble_and_highpass_attenuates_bass(tones):
    low = biquad_filter(tones, FilterSpec(24000, 500, "lowpass"), passes=2)
    high = biquad_filter(tones, FilterSpec(24000, 1000, "highpass"), passes=2)
    t = np.arange(tones.frames) / tones.sample_rate
    bass = np.sin(2 * np.pi * 100 * t)
    treble = np.sin(2 * np.pi * 5000 * t)
    assert abs(np.dot(low.samples[0], bass)) > abs(np.dot(low.samples[0], treble))
    assert abs(np.dot(high.samples[0], treble)) > abs(np.dot(high.samples[0], bass))
    assert tones.sample_rate == 24000


def test_fir_filters_have_expected_dc_response_and_shape():
    low = design_fir_lowpass(24000, 2000, taps=129)
    high = design_fir_highpass(24000, 2000, taps=129)
    assert len(low) == len(high) == 129
    assert sum(low) == pytest.approx(1, abs=1e-10)
    assert sum(high) == pytest.approx(0, abs=1e-10)
    ones = np.ones((1, 1000), dtype=np.float32)
    lp = fir_filter(ones, low)
    hp = fir_filter(ones, high)
    assert lp.shape == hp.shape == ones.shape
    np.testing.assert_allclose(lp[:, 100:-100], 1, atol=1e-5)
    np.testing.assert_allclose(hp[:, 100:-100], 0, atol=1e-5)


def test_fir_invalid_filters_and_invalid_biquad_denominator_rejected():
    with pytest.raises(AudioContractError):
        design_fir_lowpass(24000, 13000)
    with pytest.raises(AudioContractError):
        design_fir_lowpass(24000, 1000, taps=128)
    with pytest.raises(AudioContractError):
        Biquad(1, 0, 0, 0, 1.1)
    with pytest.raises(AudioContractError):
        FilterSpec(24000, 12000, "lowpass")
    with pytest.raises(AudioContractError):
        fir_filter(np.ones(10, dtype=np.float32), np.ones(100), mode="valid")


def test_dc_block_reduces_steady_offset_after_transient():
    signal = np.ones(1024, dtype=np.float32) * .5
    removed = dc_block(signal, pole=.98)
    assert removed.shape == (1, 1024)
    assert removed[0, 0] > .49
    assert abs(removed[0, -1]) < 1e-6


def test_linked_envelope_follower_tracks_louder_channel():
    left = np.ones(2400, dtype=np.float32) * .8
    right = np.ones(2400, dtype=np.float32) * .2
    src = np.stack((left, right))
    linked = envelope_follower(src, 24000, attack_seconds=0,
                               release_seconds=0, link_channels=True)
    unlinked = envelope_follower(src, 24000, attack_seconds=0,
                                 release_seconds=0, link_channels=False)
    assert linked.shape == (1, 2400)
    assert unlinked.shape == (2, 2400)
    np.testing.assert_allclose(linked, .8, atol=1e-6)
    np.testing.assert_allclose(unlinked[1], .2, atol=1e-6)


def test_compressor_reduces_above_threshold_steady_state():
    source = make_buffer(np.ones((2, 2400), dtype=np.float32) * .8, 24000)
    config = CompressorConfig(threshold_dbfs=-18, ratio=6,
                              attack_seconds=.001, release_seconds=.1,
                              knee_db=0, makeup_db=0)
    out = compress(source, config)
    assert out.samples.shape == source.samples.shape
    assert out.samples[0, -1] < .3
    np.testing.assert_array_equal(out.samples[0], out.samples[1])
    assert np.all(np.isfinite(out.samples))


def test_peak_and_rms_normalization_are_independent_operations():
    buffer = make_buffer(np.array([.1, -.1, .2, -.2], dtype=np.float32), 24000)
    peak = peak_normalize(buffer, target_dbfs=-6)
    assert stable_peak(peak.samples) == pytest.approx(10**(-6/20), rel=1e-5)
    rms = rms_normalize(buffer, target_dbfs=-20)
    assert stable_rms(rms.samples) == pytest.approx(.1, rel=1e-5)
    no_boost = peak_normalize(buffer, target_dbfs=-6, allow_boost=False)
    np.testing.assert_allclose(no_boost.samples, buffer.samples)


def test_limiter_never_exceeds_sample_ceiling():
    signal = np.array([[.1, 2, .1, -3, .2]], dtype=np.float32)
    audio = make_buffer(signal, 24000)
    limited = peak_limit(audio, ceiling_dbfs=-1)
    assert stable_peak(limited.samples) <= 10 ** (-1 / 20) + 1e-6
    assert np.all(np.isfinite(limited.samples))
    assert stable_peak(audio.samples) == 3


def test_hysteresis_gate_and_metrics_are_finite():
    signal = np.concatenate((
        np.zeros(2048), np.ones(2048) * .5, np.zeros(2048)
    )).astype(np.float32)
    audio = make_buffer(signal, 24000)
    gated = noise_gate(audio, GateConfig())
    assert gated.frames == audio.frames
    assert np.isfinite(gated.samples).all()
    report = dynamic_range_report(gated)
    assert report["peak"] >= 0
    assert report["rms"] >= 0
    assert report["noise_floor_estimating_method"] == "none"


def test_invalid_dynamic_configs_do_not_accept_nan_or_inverted_gate():
    with pytest.raises(AudioContractError):
        CompressorConfig(ratio=0.5)
    with pytest.raises(AudioContractError):
        CompressorConfig(attack_seconds=float("nan"))
    with pytest.raises(AudioContractError):
        GateConfig(open_threshold_dbfs=-50, close_threshold_dbfs=-40)
    with pytest.raises(AudioContractError):
        envelope_follower(np.ones(20, dtype=np.float32), 24000, detector="invalid")
