"""Invertible bounded STFT, spectral features and mask-validation tests."""
from __future__ import annotations

import math
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.stft import (
    StftConfig, Spectrogram, complex_mask, istft,
    spectrogram_report, stft,
)
from sonumio.audio_core.features import (
    FeatureSummary, clipping_fraction, crest_factor_db, frame_energy,
    hz_to_mel, mel_spectrogram, mel_to_hz, silence_fraction, spectral_bandwidth,
    spectral_centroid, spectral_flatness, spectral_flux, spectral_rolloff,
    summarize_audio, triangular_mel_bank, zero_crossing_rate,
)
from sonumio.audio_core.errors import AudioContractError


@pytest.fixture
def signal():
    rate = 24000
    t = np.arange(12345, dtype=np.float32) / rate
    left = .35 * np.sin(2 * np.pi * 200 * t)
    right = .15 * np.sin(2 * np.pi * 1500 * t)
    return make_buffer(np.stack((left, right)), rate)


@pytest.mark.parametrize("nfft,hop", [(128, 32), (256, 64), (512, 128), (1024, 512)])
def test_centered_hann_stft_istft_recovers_float_pcm(signal, nfft, hop):
    cfg = StftConfig(n_fft=nfft, hop_length=hop, center=True, pad_end=True)
    spectrum = stft(signal, cfg)
    assert spectrum.channels == 2
    assert spectrum.bins == nfft // 2 + 1
    assert spectrum.spectrum.dtype == np.complex128
    assert spectrum.spectrum.flags.writeable is False
    restored = istft(spectrum)
    assert restored.frames == signal.frames
    assert restored.channels == signal.channels
    np.testing.assert_allclose(restored.samples, signal.samples, atol=1e-5)


def test_spectral_array_backing_cannot_reenable_numpy_write_flag(signal):
    spec = stft(signal, StftConfig(256, 64))
    assert spec.spectrum.flags.writeable is False
    with pytest.raises(ValueError):
        spec.spectrum.setflags(write=True)
    cloned = spec.spectrum.copy()
    cloned.fill(0)
    assert np.max(np.abs(spec.spectrum)) > 0


def test_spectrogram_frequency_grid_and_magnitude_power_shape(signal):
    spec = stft(signal, StftConfig(n_fft=512, hop_length=128))
    assert spec.frequencies[0] == 0
    assert spec.frequencies[-1] == 12000
    assert spec.magnitude().shape == spec.spectrum.shape
    assert np.all(spec.power() >= 0)
    assert np.all(np.isfinite(spec.phase()))
    assert len(spec.times) == spec.frames
    report = spectrogram_report(spec)
    assert report["sample_rate"] == 24000
    assert report["source_frames"] == 12345
    assert report["channels"] == 2
    assert report["bins"] == 257
    assert report["model_inference"] is False


def test_frequency_features_reveal_different_channel_tones(signal):
    spec = stft(signal, StftConfig(n_fft=1024, hop_length=256))
    c = np.median(spectral_centroid(spec), axis=1)
    assert c[0] < c[1]
    assert 100 < c[0] < 450
    assert 1100 < c[1] < 1800
    bandwidth = spectral_bandwidth(spec)
    assert bandwidth.shape == (2, spec.frames)
    assert np.all(bandwidth >= 0)
    rolloff = spectral_rolloff(spec)
    assert rolloff.shape == (2, spec.frames)
    assert np.max(rolloff) <= 12000
    flatness = spectral_flatness(spec)
    assert flatness.shape == (2, spec.frames)
    assert np.all((flatness >= 0) & (flatness <= 1))
    assert frame_energy(spec).shape == (2, spec.frames)
    assert spectral_flux(spec).shape == (2, spec.frames)


def test_silent_spectrum_features_are_finite():
    audio = make_buffer(np.zeros(1024, dtype=np.float32), 24000)
    spec = stft(audio, StftConfig(n_fft=256, hop_length=128))
    for value in (
        spectral_centroid(spec), spectral_bandwidth(spec),
        spectral_flatness(spec), spectral_flux(spec),
        spectral_rolloff(spec), frame_energy(spec),
    ):
        assert np.isfinite(value).all()
    assert zero_crossing_rate(audio.samples, per_channel=False) == 0
    assert silence_fraction(audio.samples) == 1
    assert clipping_fraction(audio.samples) == 0


def test_complex_mask_requires_finite_safe_gain_and_shape(signal):
    spec = stft(signal, StftConfig(n_fft=256, hop_length=64))
    masked = complex_mask(spec, np.ones(spec.spectrum.shape[1:]) * .5)
    assert masked.spectrum.shape == spec.spectrum.shape
    np.testing.assert_allclose(masked.spectrum, .5 * spec.spectrum)
    with pytest.raises(AudioContractError):
        complex_mask(spec, np.ones((2, 3)))
    with pytest.raises(AudioContractError):
        complex_mask(spec, np.full(spec.spectrum.shape, float("nan")))
    with pytest.raises(AudioContractError):
        complex_mask(spec, np.ones(spec.spectrum.shape) * 9)


def test_invalid_fft_bounds_and_nonfinite_spectra_rejected(signal):
    for nfft, hop in ((1, 1), (8193, 256), (256, 0), (256, 257)):
        with pytest.raises(AudioContractError):
            StftConfig(nfft, hop)
    spec = stft(signal)
    corrupt = spec.spectrum.copy()
    corrupt[0, 0, 0] = complex(float("nan"), 0)
    with pytest.raises(AudioContractError):
        Spectrogram(corrupt, 24000, signal.frames, spec.config)


def test_feature_summary_is_named_metrics_not_lufs(signal):
    summary = summarize_audio(signal, config=StftConfig(256, 64))
    assert isinstance(summary, FeatureSummary)
    as_dict = summary.as_dict()
    assert as_dict["sample_rate"] == 24000
    assert as_dict["channels"] == 2
    assert "lufs" not in as_dict
    assert as_dict["clipping_fraction"] == 0
    assert as_dict["rms_dbfs"] < as_dict["peak_dbfs"]
    assert as_dict["crest_db"] > 0


def test_mel_hz_roundtrips_and_mel_spectrogram_geometry(signal):
    frequencies = np.array([0, 100, 500, 4000, 10000], dtype=np.float64)
    np.testing.assert_allclose(mel_to_hz(hz_to_mel(frequencies)),
                               frequencies, atol=1e-9)
    spec = stft(signal, StftConfig(512, 128))
    bank = triangular_mel_bank(24000, 512, bands=40)
    assert bank.shape == (40, 257)
    assert np.all(bank >= 0)
    output = mel_spectrogram(spec, bands=40)
    assert output.shape == (2, 40, spec.frames)
    assert np.all(output >= 0)
    with pytest.raises(AudioContractError):
        triangular_mel_bank(24000, 512, bands=0)


def test_clipping_and_silence_are_explicit_signal_thresholds():
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    assert clipping_fraction(x) == 4 / 5
    assert silence_fraction(x, floor_dbfs=-20) == 1 / 5
    assert crest_factor_db(x) > 0
    with pytest.raises(AudioContractError):
        silence_fraction(x, floor_dbfs=20)
