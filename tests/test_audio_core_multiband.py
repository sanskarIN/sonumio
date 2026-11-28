"""Reference tests for complementary FIR crossovers and band dynamics."""
import dataclasses
import math

import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.dynamics import CompressorConfig
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.multiband import (
    CrossoverPlan, CrossoverGeometry, BandLevel, MultibandResult,
    split_bands, recombine_bands, multiband_compress,
)


def source(*, rate=16000, frames=353, channels=2):
    rng = np.random.default_rng(2027)
    audio = rng.normal(0, .18, (channels, frames)).astype(np.float32)
    return make_buffer(audio, rate)


def unity_settings(bands):
    return tuple(CompressorConfig(ratio=1, knee_db=0, attack_seconds=0,
                                   release_seconds=0)
                 for _ in range(bands))


@pytest.mark.parametrize("cutoffs", [
    (200.,), (180., 2500.), (100., 800., 5000.),
    (100., 400., 1600., 3500., 6000.),
])
@pytest.mark.parametrize("taps", [65, 129, 257])
@pytest.mark.parametrize("window", ["blackman", "hann", "hamming"])
def test_complementary_kernels_form_centered_unit_impulse(cutoffs, taps, window):
    plan = CrossoverPlan(16000, cutoffs, taps, window)
    kernels = plan.kernels()
    assert kernels.shape == (len(cutoffs) + 1, taps)
    assert kernels.dtype == np.float64
    assert not kernels.flags.writeable
    reference = np.zeros(taps)
    reference[plan.delay_frames] = 1
    np.testing.assert_allclose(kernels.sum(axis=0), reference, atol=1e-12)
    with pytest.raises(ValueError):
        kernels.setflags(write=True)


@pytest.mark.parametrize("frames", [1, 13, 64, 65, 256, 257, 353, 997])
@pytest.mark.parametrize("channels", [1, 2, 5])
@pytest.mark.parametrize("cutoffs", [(1000.,), (180., 2500.)])
def test_band_sum_is_sample_aligned_input(frames, channels, cutoffs):
    audio = source(frames=frames, channels=channels)
    plan = CrossoverPlan(audio.sample_rate, cutoffs, 65)
    bands = split_bands(audio, plan)
    assert len(bands) == plan.bands
    assert all(b.spec == audio.spec for b in bands)
    assert all(not b.samples.flags.writeable for b in bands)
    result = recombine_bands(bands)
    np.testing.assert_allclose(result.samples, audio.samples,
                               rtol=1e-5, atol=5e-7)
    assert result.sample_rate == audio.sample_rate
    assert result.channels == audio.channels
    assert result.frames == audio.frames


@pytest.mark.parametrize("frequency,expected_band", [
    (80., 0), (800., 1), (5000., 2),
])
def test_sinusoidal_energy_routed_to_expected_band(frequency, expected_band):
    rate = 16000
    frames = 16000
    t = np.arange(frames) / rate
    # Integer-cycle sinusoids avoid uncertain leakage in the main interval.
    audio = make_buffer((.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32),
                        rate)
    plan = CrossoverPlan(rate, (250., 2500.), 257)
    bands = split_bands(audio, plan)
    # Exclude edge transients so frequency responses dominate the measurement.
    energies = [
        float(np.mean(b.samples[0, 500:-500].astype(np.float64) ** 2))
        for b in bands
    ]
    assert int(np.argmax(energies)) == expected_band
    assert energies[expected_band] > 10 * max(
        e for i, e in enumerate(energies) if i != expected_band
    )


@pytest.mark.parametrize("wet", [0., .25, .75, 1.])
@pytest.mark.parametrize("gain_db", [-6., 0., 3.])
def test_unity_multiband_reconstructs_with_declared_gain(wet, gain_db):
    audio = source()
    plan = CrossoverPlan(audio.sample_rate, (150., 2800.), 129)
    outcome = multiband_compress(
        audio, plan, unity_settings(plan.bands), wet=wet, output_gain_db=gain_db,
    )
    assert isinstance(outcome, MultibandResult)
    assert outcome.audio.spec == audio.spec
    factor = 10 ** (gain_db / 20)
    np.testing.assert_allclose(outcome.audio.samples, audio.samples * factor,
                               rtol=3e-5, atol=5e-7)
    assert len(outcome.report.bands) == plan.bands
    assert outcome.report.wet == wet
    assert outcome.report.output_gain_db == gain_db
    assert outcome.report.group_delay_frames == plan.delay_frames
    report = outcome.report.as_dict()
    assert report["protocol"] == "sonumio-fir-multiband-v1"
    assert report["model_inference"] is False
    assert report["automatic_clipping_prevention"] is False
    assert len(report["bands"]) == plan.bands


def test_wet_zero_is_exact_identity_reference():
    audio = source()
    plan = CrossoverPlan(audio.sample_rate, (160., 2000.), 129)
    compressors = tuple(CompressorConfig(threshold_dbfs=-50, ratio=20)
                        for _ in range(plan.bands))
    result = multiband_compress(audio, plan, compressors, wet=0)
    assert result.audio is audio
    np.testing.assert_array_equal(result.audio.samples, audio.samples)


def test_band_compression_does_not_change_layout_or_clock():
    rate = 16000
    waveform = np.zeros((2, 4000), dtype=np.float32)
    waveform[0] = .8 * np.sin(2 * np.pi * 120 * np.arange(4000) / rate)
    waveform[1] = .3 * np.sin(2 * np.pi * 2500 * np.arange(4000) / rate)
    layout = AudioLayout(2, "discrete", ("vocal", "instrument"))
    audio = make_buffer(waveform, rate, layout=layout)
    plan = CrossoverPlan(rate, (400., 1400.), 257)
    configs = tuple(CompressorConfig(threshold_dbfs=-35, ratio=10,
                                     attack_seconds=0, release_seconds=0, knee_db=0)
                    for _ in range(plan.bands))
    result = multiband_compress(audio, plan, configs, wet=1)
    assert result.audio.spec == audio.spec
    assert float(np.max(np.abs(result.audio.samples))) < .8
    assert len(result.report.bands) == 3
    assert all(isinstance(b, BandLevel) for b in result.report.bands)
    assert all(b.output_rms <= b.input_rms + 1e-8 for b in result.report.bands)


@pytest.mark.parametrize("value", [
    (), (100., 100.), (500., 300.), (0.,), (-20.,), (8000.,),
    (100., float("nan")), (float("inf"),), tuple(range(100, 700, 100)),
])
def test_invalid_cutoff_lists_rejected(value):
    with pytest.raises(AudioContractError):
        CrossoverPlan(16000, value)


@pytest.mark.parametrize("taps", [0, 1, 2, 64, 128, 4096, True, 100000])
def test_invalid_taps_rejected(taps):
    with pytest.raises(AudioContractError):
        CrossoverPlan(16000, taps=taps)


def test_invalid_window_and_sample_clock():
    with pytest.raises(AudioContractError):
        CrossoverPlan(8000, window="kaiser")
    plan = CrossoverPlan(16000)
    with pytest.raises(AudioContractError):
        split_bands(source(rate=8000), plan)
    with pytest.raises(AudioContractError):
        multiband_compress(source(rate=8000), plan, unity_settings(plan.bands))


def test_invalid_multiband_configurations():
    audio = source()
    plan = CrossoverPlan(16000, taps=65)
    valid = unity_settings(plan.bands)
    for invalid in ((), valid[:1], list(valid), (None,) * plan.bands):
        with pytest.raises(AudioContractError):
            multiband_compress(audio, plan, invalid)
    for invalid in (True, -1, 1.5, float("nan")):
        with pytest.raises(AudioContractError):
            multiband_compress(audio, plan, valid, wet=invalid)
    for invalid in (-61, 25, float("inf")):
        with pytest.raises(AudioContractError):
            multiband_compress(audio, plan, valid, output_gain_db=invalid)


def test_band_geometry_validation_and_fft_resource_bounds():
    g = CrossoverGeometry(2, 257, 3, 65)
    assert g.fft_size == 512
    assert g.bins == 257
    with pytest.raises(AudioLimitError):
        CrossoverGeometry(8, 1_000_000, 6, 4095)
    with pytest.raises(AudioContractError):
        CrossoverGeometry(9, 10, 2, 5)
    with pytest.raises(AudioContractError):
        CrossoverGeometry(1, 0, 2, 5)


def test_band_recombination_rejects_mixed_clock_and_layout():
    a = source(rate=8000)
    b = source(rate=16000)
    with pytest.raises(AudioContractError):
        recombine_bands((a, b))
    with pytest.raises(AudioContractError):
        recombine_bands((a,))
    with pytest.raises(AudioContractError):
        recombine_bands([a, a])
    with pytest.raises(AudioContractError):
        recombine_bands((a, None))
    c = make_buffer(a.samples, a.sample_rate,
                    layout=AudioLayout(2, "discrete", ("L", "R")))
    with pytest.raises(AudioContractError):
        recombine_bands((a, c))


def test_no_mutation_of_caller_pcm():
    audio = source()
    before = audio.samples.copy()
    plan = CrossoverPlan(16000, taps=65)
    split_bands(audio, plan)
    multiband_compress(audio, plan, unity_settings(plan.bands))
    np.testing.assert_array_equal(audio.samples, before)
