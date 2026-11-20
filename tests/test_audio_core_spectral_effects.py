"""Model-free spectral EQ, noise masks, tilt and heuristic HPSS regression tests."""
from __future__ import annotations

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.stft import StftConfig, stft
from sonumio.audio_core.spectral_effects import (
    SpectralGateOptions, SpectralBand, estimate_noise_profile,
    spectral_gate_mask, spectral_gate, spectral_eq_mask, spectral_eq,
    spectral_tilt, median_harmonic_percussive_masks,
    heuristic_harmonic_percussive, spectral_operation_report,
)
from sonumio.audio_core.errors import AudioContractError


@pytest.fixture
def tone():
    sr = 24000
    t = np.arange(8192, dtype=np.float64) / sr
    wave = .25 * np.sin(2 * np.pi * 440 * t)
    wave *= np.sin(np.linspace(0, np.pi, len(t))) ** 2
    return make_buffer(wave.astype(np.float32), sr)


@pytest.fixture
def noisy():
    rng = np.random.default_rng(2026)
    noise = rng.normal(0, .01, 8192)
    t = np.arange(8192, dtype=np.float64) / 24000
    tone = .3 * np.sin(2 * np.pi * 800 * t)
    # First 2048 frames contain only noise; rest are noise+tone.
    source = noise.copy()
    source[2048:] += tone[2048:]
    return make_buffer(source.astype(np.float32), 24000)


@pytest.mark.parametrize("statistic", ["median", "mean", "maximum"])
def test_explicit_noise_region_returns_nonnegative_channel_bin_profile(noisy, statistic):
    spec = stft(noisy, StftConfig(n_fft=256, hop_length=64))
    profile = estimate_noise_profile(spec, start_frame=0, stop_frame=8,
                                     statistic=statistic)
    assert profile.shape == (noisy.channels, 129)
    assert profile.dtype == np.float32
    assert np.all(np.isfinite(profile))
    assert np.min(profile) >= 0


def test_profile_requires_valid_frame_range_and_estimator(noisy):
    spec = stft(noisy, StftConfig(n_fft=256, hop_length=64))
    with pytest.raises(AudioContractError):
        estimate_noise_profile(spec, start_frame=8, stop_frame=8)
    with pytest.raises(AudioContractError):
        estimate_noise_profile(spec, start_frame=-1, stop_frame=10)
    with pytest.raises(AudioContractError):
        estimate_noise_profile(spec, start_frame=0, stop_frame=100000)
    with pytest.raises(AudioContractError):
        estimate_noise_profile(spec, statistic="machine-learning")


def test_spectral_gate_mask_is_bounded_and_reproducible(noisy):
    spec = stft(noisy, StftConfig(n_fft=256, hop_length=64))
    profile = estimate_noise_profile(spec, start_frame=0, stop_frame=8)
    params = SpectralGateOptions(
        threshold_ratio=1.5, floor_gain=.1, exponent=1.5,
        temporal_smooth=3, frequency_smooth=3,
    )
    a = spectral_gate_mask(spec, profile, options=params)
    b = spectral_gate_mask(spec, profile, options=params)
    np.testing.assert_array_equal(a, b)
    assert a.shape == spec.spectrum.shape
    assert np.min(a) >= .1 - 1e-5
    assert np.max(a) <= 1
    assert a.dtype == np.float32


def test_untouched_original_and_declared_noise_interval_gate(noisy):
    raw = noisy.samples.copy()
    config = StftConfig(256, 64)
    result = spectral_gate(
        noisy, noise_start_seconds=0, noise_stop_seconds=.06,
        config=config, options=SpectralGateOptions(threshold_ratio=2)
    )
    assert result.frames == noisy.frames
    assert result.channels == noisy.channels
    assert result.sample_rate == noisy.sample_rate
    assert result.samples.dtype == np.float32
    assert np.isfinite(result.samples).all()
    np.testing.assert_array_equal(noisy.samples, raw)
    assert not np.array_equal(result.samples, noisy.samples)
    report = spectral_operation_report(noisy, result, "noise gate")
    assert report["neural_model_inference"] is False
    assert report["perceptual_enhancement_claimed"] is False
    assert report["generator_or_training_data_modified"] is False


def test_noise_gate_requires_nonempty_bounded_quiet_reference(tone):
    with pytest.raises(AudioContractError):
        spectral_gate(tone, noise_start_seconds=.1, noise_stop_seconds=.1)
    with pytest.raises(AudioContractError):
        spectral_gate(tone, noise_start_seconds=-1)
    with pytest.raises(AudioContractError):
        spectral_gate(tone, noise_stop_seconds=100)
    with pytest.raises(AudioContractError):
        SpectralGateOptions(threshold_ratio=-1)
    with pytest.raises(AudioContractError):
        SpectralGateOptions(temporal_smooth=2)
    with pytest.raises(AudioContractError):
        SpectralGateOptions(floor_gain=1.1)


def test_identity_frequency_eq_is_nearly_transparent(tone):
    config = StftConfig(256, 64)
    bands = (SpectralBand(0, 0), SpectralBand(12000, 0))
    result = spectral_eq(tone, bands, config=config)
    np.testing.assert_allclose(result.samples, tone.samples, atol=1e-5)
    assert result.frames == tone.frames
    assert np.isfinite(result.samples).all()


def test_eq_frequency_points_are_ordered_distinct_and_within_nyquist(tone):
    spec = stft(tone, StftConfig(256, 64))
    with pytest.raises(AudioContractError):
        spectral_eq_mask(spec, (SpectralBand(400, 0), SpectralBand(100, 0)))
    with pytest.raises(AudioContractError):
        spectral_eq_mask(spec, (SpectralBand(400, 0), SpectralBand(400, 1)))
    with pytest.raises(AudioContractError):
        spectral_eq_mask(spec, (SpectralBand(1, 0), SpectralBand(13000, 2)))
    with pytest.raises(AudioContractError):
        SpectralBand(100, float("nan"))


def test_selected_tone_band_attenuation_lowers_pcm_rms(tone):
    config = StftConfig(512, 128)
    result = spectral_eq(tone, (
        SpectralBand(0, -18),
        SpectralBand(440, -18),
        SpectralBand(1000, 0),
        SpectralBand(12000, 0),
    ), config=config)
    assert float(np.sqrt(np.mean(result.samples ** 2))) < (
        .3 * float(np.sqrt(np.mean(tone.samples ** 2)))
    )


def test_tilt_has_bounded_deterministic_pcm_and_zero_slope_identity(tone):
    config = StftConfig(512, 128)
    flat = spectral_tilt(tone, db_per_octave=0,
                         pivot_hz=1000, config=config)
    np.testing.assert_allclose(flat.samples, tone.samples, atol=1e-5)
    tilted = spectral_tilt(tone, db_per_octave=-6,
                           pivot_hz=1000, config=config)
    assert tilted.frames == tone.frames
    assert np.isfinite(tilted.samples).all()
    assert not np.array_equal(tilted.samples, tone.samples)
    with pytest.raises(AudioContractError):
        spectral_tilt(tone, db_per_octave=100)


def test_hpss_soft_masks_sum_to_one_and_have_valid_geometry(tone):
    spec = stft(tone, StftConfig(256, 64))
    harmonic, percussive = median_harmonic_percussive_masks(
        spec, time_filter=5, frequency_filter=5
    )
    assert harmonic.shape == percussive.shape == spec.spectrum.shape
    assert np.min(harmonic) >= 0 and np.max(harmonic) <= 1
    assert np.min(percussive) >= 0 and np.max(percussive) <= 1
    np.testing.assert_allclose(harmonic + percussive, 1, atol=1e-6)


def test_heuristic_spectral_stems_sum_back_to_source_to_numerical_tolerance(tone):
    h, p = heuristic_harmonic_percussive(
        tone, config=StftConfig(256, 64),
        time_filter=5, frequency_filter=5
    )
    assert h.frames == p.frames == tone.frames
    np.testing.assert_allclose(h.samples + p.samples, tone.samples, atol=1e-5)
    assert np.isfinite(h.samples).all() and np.isfinite(p.samples).all()


def test_hpss_rejects_unsupported_window_and_exponent(tone):
    spec = stft(tone, StftConfig(256, 64))
    with pytest.raises(AudioContractError):
        median_harmonic_percussive_masks(spec, time_filter=2)
    with pytest.raises(AudioContractError):
        median_harmonic_percussive_masks(spec, frequency_filter=32)
    with pytest.raises(AudioContractError):
        median_harmonic_percussive_masks(spec, mask_power=0)


def test_noise_profile_shape_and_finite_values_cannot_be_forged(tone):
    spec = stft(tone, StftConfig(256, 64))
    for bad in (
        np.zeros((2, 129), dtype=np.float32),
        np.ones((1, 129), dtype=np.float64) * np.nan,
        np.ones((1, 129), dtype=np.float32) * -1,
        np.ones((1, 129), dtype=np.int32),
    ):
        with pytest.raises(AudioContractError):
            spectral_gate_mask(spec, bad)
