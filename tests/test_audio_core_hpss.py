"""Numerical reference checks for complementary median-filter STFT separation."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.hpss import (
    HpssSettings, HpssGeometry, SeparationResult, _median_axis,
    _mask_from_estimates, separate_spectrogram, split_harmonic_percussive,
)
from sonumio.audio_core.stft import StftConfig, stft


def mixture(*, rate=16000, frames=8192, channels=1):
    n = np.arange(frames)
    sinusoid = .14 * np.sin(2 * np.pi * 400 * n / rate)
    transient = np.zeros(frames)
    transient[1700] = .8
    transient[3700] = -.65
    transient[5700] = .9
    wave = (sinusoid + transient).astype(np.float32)
    return make_buffer(np.tile(wave, (channels, 1)), rate)


def plan(**kwargs):
    values = dict(n_fft=512, hop_frames=128,
                  harmonic_median_frames=13, percussive_median_bins=13)
    values.update(kwargs)
    return HpssSettings(**values)


@pytest.mark.parametrize("channels", [1, 2, 3])
@pytest.mark.parametrize("frames", [129, 512, 2048, 8192])
def test_harmonic_plus_percussive_reconstructs_input(channels, frames):
    audio = mixture(channels=channels, frames=frames)
    result = split_harmonic_percussive(audio, plan())
    assert isinstance(result, SeparationResult)
    assert result.harmonic.spec == audio.spec
    assert result.percussive.spec == audio.spec
    combined = (
        result.harmonic.samples.astype(np.float64)
        + result.percussive.samples.astype(np.float64)
    )
    np.testing.assert_allclose(combined, audio.samples, atol=3e-6)
    assert result.stats.reconstruction_peak_error < 3e-6
    assert result.stats.reconstruction_rms_error < 1e-6
    assert result.stats.frames == frames
    assert result.stats.channels == channels
    assert not result.harmonic.samples.flags.writeable


def test_nonpercussive_sinusoid_favors_harmonic_component():
    rate = 16000
    n = np.arange(16000)
    source = make_buffer(
        (.3 * np.sin(2 * np.pi * 500 * n / rate)).astype(np.float32), rate,
    )
    result = split_harmonic_percussive(source, plan())
    # Omit transient edges in the objective energy comparison.
    harmonic = result.harmonic.samples[0, 2000:13000]
    percussive = result.percussive.samples[0, 2000:13000]
    assert float(np.sum(harmonic ** 2)) > 8 * float(np.sum(percussive ** 2))


def test_rare_impulses_favor_percussive_component():
    source = np.zeros(16000, dtype=np.float32)
    source[2000] = 1.
    source[8000] = -.8
    source[14000] = .7
    result = split_harmonic_percussive(
        make_buffer(source, 16000), plan(),
    )
    harmonic = float(np.sum(result.harmonic.samples.astype(float) ** 2))
    percussive = float(np.sum(result.percussive.samples.astype(float) ** 2))
    assert percussive > harmonic * 4


@pytest.mark.parametrize("power", [1., 2., 4., 8.])
def test_soft_masks_partition_unity_even_when_signal_is_silent(power):
    options = plan(mask_power=power)
    h = np.array([[[0., .1, .3, 2., .5]]])
    p = np.array([[[0., .6, .7, .1, .5]]])
    hmask, pmask = _mask_from_estimates(h, p, options)
    np.testing.assert_array_equal(hmask + pmask, 1.)
    assert hmask[0, 0, 0] == .5
    assert pmask[0, 0, 0] == .5
    assert np.min(hmask) >= 0
    assert np.max(hmask) <= 1


def test_median_works_on_each_independent_axis():
    magnitude = np.zeros((2, 7, 9), dtype=np.float64)
    magnitude[:, 2, :] = 2
    magnitude[:, :, 3] += 3
    frequency = _median_axis(magnitude, 3, axis=1)
    temporal = _median_axis(magnitude, 3, axis=2)
    assert frequency.shape == magnitude.shape
    assert temporal.shape == magnitude.shape
    assert frequency[0, 2, 4] == 0
    assert temporal[0, 2, 4] == 2


def test_masks_are_readonly_and_stft_spectra_remain_unmodified():
    audio = mixture()
    cfg = StftConfig(n_fft=512, hop_length=128)
    spec = stft(audio, cfg)
    original = spec.spectrum.copy()
    a, b, mask = separate_spectrogram(spec, plan())
    assert mask.shape == spec.spectrum.shape
    assert not mask.flags.writeable
    np.testing.assert_allclose(
        a.spectrum + b.spectrum, original, rtol=1e-13, atol=1e-12,
    )
    np.testing.assert_array_equal(spec.spectrum, original)
    with pytest.raises(ValueError):
        mask.setflags(write=True)


def test_stereo_layout_and_source_samples_preserved():
    rng = np.random.default_rng(99)
    source = make_buffer(
        rng.normal(0, .1, (2, 2500)).astype(np.float32),
        16000, layout=AudioLayout(2, "discrete", ("dialogue", "room")),
    )
    original = source.samples.copy()
    result = split_harmonic_percussive(source, plan())
    assert result.harmonic.spec.layout == source.spec.layout
    assert result.percussive.spec.layout == source.spec.layout
    np.testing.assert_array_equal(source.samples, original)


def test_no_model_or_perfect_source_separation_claims():
    report = split_harmonic_percussive(mixture(), plan()).stats.as_dict()
    assert report["protocol"] == "sonumio-median-hpss-v1"
    assert report["model_inference"] is False
    assert report["perfect_instrument_isolation_claimed"] is False
    assert report["source_recording_modified"] is False
    assert report["perceptual_quality_established"] is False


def test_silent_audio_returns_two_finite_silent_tracks():
    source = make_buffer(np.zeros((2, 1024), dtype=np.float32), 16000)
    result = split_harmonic_percussive(source, plan())
    np.testing.assert_array_equal(result.harmonic.samples, 0)
    np.testing.assert_array_equal(result.percussive.samples, 0)
    assert result.stats.reconstruction_peak_error == 0
    assert np.isfinite(result.stats.harmonic_energy_fraction)


@pytest.mark.parametrize("bad", [
    dict(n_fft=63), dict(n_fft=8193), dict(hop_frames=0),
    dict(n_fft=True), dict(hop_frames=512, n_fft=512),
    dict(harmonic_median_frames=2), dict(harmonic_median_frames=102),
    dict(percussive_median_bins=0), dict(percussive_median_bins=12),
    dict(mask_power=0), dict(mask_power=9),
    dict(margin_harmonic=-1), dict(margin_percussive=20),
    dict(minimum_magnitude=0), dict(minimum_magnitude=float("nan")),
])
def test_invalid_settings_rejected(bad):
    with pytest.raises(AudioContractError):
        HpssSettings(**bad)


def test_mismatched_fft_geometry_rejected():
    source = mixture()
    spec = stft(source, StftConfig(n_fft=256, hop_length=64))
    with pytest.raises(AudioContractError):
        separate_spectrogram(spec, plan())


@pytest.mark.parametrize("bad", [
    np.zeros((2, 2), dtype=np.float64),
    np.zeros((1, 7, 9), dtype=np.int32),
    np.full((1, 7, 9), np.nan, dtype=np.float64),
    -np.ones((1, 7, 9), dtype=np.float64),
])
def test_invalid_median_inputs_rejected(bad):
    with pytest.raises(AudioContractError):
        _median_axis(bad, 3, axis=2)


def test_resource_guard_fails_before_median_allocation():
    with pytest.raises(AudioLimitError):
        HpssGeometry(8, 4097, 5000, 101, 101)
    with pytest.raises(AudioLimitError):
        HpssGeometry(4, 2049, 5000, 51, 51)


def test_invalid_soft_mask_input_fails_closed():
    options = plan()
    valid = np.zeros((1, 4, 7), dtype=np.float64)
    with pytest.raises(AudioContractError):
        _mask_from_estimates(valid, np.ones((1, 3, 7)), options)
    with pytest.raises(AudioContractError):
        _mask_from_estimates(valid, valid.astype(np.float32), options)
    with pytest.raises(AudioContractError):
        _mask_from_estimates(valid, np.full_like(valid, np.nan), options)
