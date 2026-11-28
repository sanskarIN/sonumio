"""Regularized spectral suppression profiles, gain bounds and source invariants."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.stft import StftConfig, stft
from sonumio.audio_core.wiener_suppression import (
    WienerSettings, NoiseProfile, estimate_wiener_profile,
    wiener_gain, suppress_wiener,
)


def audio_with_stationary_noise(rate=16000, frames=8000, channels=1, seed=71):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, .035, (channels, frames))
    sine = .3 * np.sin(2 * np.pi * 440 * np.arange(frames) / rate)
    noise[:, round(.12 * rate):] += sine[None, round(.12 * rate):]
    return make_buffer(noise.astype(np.float32), rate)


def options(**changes):
    values = dict(noise_stop_seconds=.1, n_fft=512, hop_frames=128,
                  minimum_gain=.05, oversubtraction=1.5)
    values.update(changes)
    return WienerSettings(**values)


def test_all_pass_with_zero_noise_profile_is_unity_gain():
    signal = audio_with_stationary_noise()
    spec = stft(signal, StftConfig(n_fft=512, hop_length=128))
    profile = NoiseProfile(
        np.zeros((1, spec.bins), dtype=np.float64),
        spec.sample_rate, spec.config.n_fft, 0, 4,
    )
    gain = wiener_gain(spec, profile, options())
    # Nonzero STFT energy should map to exactly unit amplitude.
    positive = np.abs(spec.spectrum) > 1e-10
    np.testing.assert_allclose(gain[positive], 1, atol=1e-12)
    assert np.all(gain >= .05)
    assert np.all(gain <= 1)


def test_detected_noise_profile_and_gain_are_bounded():
    signal = audio_with_stationary_noise()
    cfg = StftConfig(n_fft=512, hop_length=128)
    spec = stft(signal, cfg)
    profile = estimate_wiener_profile(spec, options())
    assert profile.power.shape == (1, 257)
    assert np.isfinite(profile.power).all()
    assert np.all(profile.power >= 0)
    assert not profile.power.flags.writeable
    with pytest.raises(ValueError):
        profile.power.setflags(write=True)
    gain = wiener_gain(spec, profile, options())
    assert gain.shape == spec.spectrum.shape
    assert np.min(gain) >= .05 - 1e-12
    assert np.max(gain) <= 1 + 1e-12
    assert np.any(gain < 1)


@pytest.mark.parametrize("wet", [0., .25, .5, 1.])
def test_spectral_suppression_keeps_clock_shape_and_wet_bypass(wet):
    signal = audio_with_stationary_noise(channels=2)
    result = suppress_wiener(signal, options(wet=wet))
    assert result.audio.spec == signal.spec
    assert np.isfinite(result.audio.samples).all()
    assert not result.audio.samples.flags.writeable
    assert result.report.sample_rate == signal.sample_rate
    assert result.report.frames == signal.frames
    assert result.report.channels == signal.channels
    if wet == 0:
        assert result.audio is signal
        np.testing.assert_array_equal(result.audio.samples, signal.samples)
    assert result.report.as_dict()["model_inference"] is False
    assert result.report.as_dict()["perceptual_quality_established"] is False


def test_profiled_noise_region_is_reduced():
    signal = audio_with_stationary_noise()
    restored = suppress_wiener(signal, options()).audio
    # The declared first 0.1 s contains only stationary noise.
    original_noise = signal.samples[0, 256:1200]
    after_noise = restored.samples[0, 256:1200]
    rms_before = float(np.sqrt(np.mean(original_noise.astype(float) ** 2)))
    rms_after = float(np.sqrt(np.mean(after_noise.astype(float) ** 2)))
    assert rms_after < rms_before * .8
    assert rms_after > 0


def test_tonal_content_not_entirely_erased():
    signal = audio_with_stationary_noise()
    result = suppress_wiener(signal, options())
    original = signal.samples[0, 4000:7000]
    processed = result.audio.samples[0, 4000:7000]
    t = np.arange(4000, 7000)
    reference = np.sin(2 * np.pi * 440 * t / signal.sample_rate)
    amplitude_before = abs(float(np.dot(original, reference)))
    amplitude_after = abs(float(np.dot(processed, reference)))
    assert amplitude_after > amplitude_before * .5


def test_stereo_profile_linking_requires_explicit_option():
    audio = audio_with_stationary_noise(channels=2)
    spec = stft(audio, StftConfig(n_fft=512, hop_length=128))
    local = estimate_wiener_profile(spec, options())
    linked = estimate_wiener_profile(
        spec, options(channel_linked_profile=True),
    )
    assert local.power.shape == linked.power.shape
    np.testing.assert_array_equal(linked.power[0], linked.power[1])
    assert np.all(linked.power >= local.power)


def test_deterministic_output_and_unmodified_input():
    signal = audio_with_stationary_noise(channels=2)
    before = signal.samples.copy()
    first = suppress_wiener(signal, options())
    second = suppress_wiener(signal, options())
    np.testing.assert_array_equal(first.audio.samples, second.audio.samples)
    assert first.report == second.report
    np.testing.assert_array_equal(signal.samples, before)


@pytest.mark.parametrize("bad", [
    dict(n_fft=63), dict(n_fft=8193), dict(n_fft=True),
    dict(hop_frames=0), dict(hop_frames=512, n_fft=512),
    dict(noise_start_seconds=-1), dict(noise_stop_seconds=0),
    dict(noise_start_seconds=.2, noise_stop_seconds=.1),
    dict(noise_statistic="unknown"),
    dict(oversubtraction=0), dict(oversubtraction=9),
    dict(minimum_gain=-1), dict(minimum_gain=2),
    dict(smoothing_time=2), dict(smoothing_time=16),
    dict(smoothing_frequency=0), dict(smoothing_frequency=4),
    dict(channel_linked_profile=1), dict(wet=2), dict(wet=float("nan")),
])
def test_invalid_settings_rejected(bad):
    with pytest.raises(AudioContractError):
        WienerSettings(**bad)


def test_noise_interval_must_fit_original_source():
    signal = audio_with_stationary_noise()
    with pytest.raises(AudioContractError):
        suppress_wiener(signal, options(noise_stop_seconds=20.))


def test_profile_bin_and_channel_mismatch_rejected():
    audio = audio_with_stationary_noise()
    spec = stft(audio, StftConfig(n_fft=512, hop_length=128))
    bad = NoiseProfile(np.ones((1, 513), dtype=np.float64),
                       16000, 1024, 0, 4)
    with pytest.raises(AudioContractError):
        wiener_gain(spec, bad, options())


def test_malformed_profile_rejected():
    with pytest.raises(AudioContractError):
        NoiseProfile(np.array([[np.nan]], dtype=np.float64), 16000, 0, 0, 4)
    with pytest.raises(AudioContractError):
        NoiseProfile(np.ones((1, 257), dtype=np.float32), 16000, 512, 0, 4)
    with pytest.raises(AudioContractError):
        NoiseProfile(np.ones((1, 257), dtype=np.float64), 16000, 512, 4, 1)


def test_undersized_reference_interval_is_still_bounded():
    audio = audio_with_stationary_noise()
    result = suppress_wiener(audio, options(noise_stop_seconds=.001))
    assert result.profile.source_stop_frame > result.profile.source_start_frame


def test_direct_noncanonical_audio_argument_is_rejected():
    with pytest.raises(AudioContractError):
        suppress_wiener(None)
    signal = audio_with_stationary_noise()
    with pytest.raises(AudioContractError):
        suppress_wiener(signal, None)
