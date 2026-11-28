"""Deterministic sweep stimulus, linear response recovery and strict bounds."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.matrix_convolution import convolve_matrix
from sonumio.audio_core.sweep_identification import (
    SweepPlan, IdentificationSettings, IdentificationGeometry,
    render_log_sweep, identify_linear_response,
)


def default_stimulus():
    return render_log_sweep(SweepPlan(
        sample_rate=8000, frames=4096,
        start_hz=70., end_hz=3100., fade_frames=64,
    ))


def response(source, kernels, delay=0):
    length = len(source.samples[0]) + kernels.shape[1] - 1 + delay
    channels = kernels.shape[0]
    out = np.zeros((channels, length), dtype=np.float64)
    for i in range(channels):
        filtered = np.convolve(source.samples[0].astype(np.float64), kernels[i])
        out[i, delay:delay + len(filtered)] = filtered
    return make_buffer(out.astype(np.float32), source.sample_rate)


@pytest.mark.parametrize("delay", [0, 13, 87])
@pytest.mark.parametrize("channels", [1, 2, 3])
def test_known_short_fir_is_recovered(delay, channels):
    excitation = default_stimulus()
    known = np.zeros((channels, 64), dtype=np.float64)
    for i in range(channels):
        known[i, 0] = 1 - i * .1
        known[i, 12] = .3 + i * .025
        known[i, 43] = -.12
    capture = response(excitation, known, delay=delay)
    settings = IdentificationSettings(
        taps=64, latency_frames=delay,
        regularization=1e-12, block_frames=64,
    )
    result = identify_linear_response(excitation, capture, settings)
    assert result.impulse.coefficients.shape == (channels, 1, 64)
    assert result.report.latency_frames == delay
    assert result.report.channels == channels
    assert len(result.report.channels_report) == channels
    np.testing.assert_allclose(result.impulse.coefficients[:, 0, :],
                               known, atol=.02, rtol=.04)
    assert all(report.relative_fit_rms < .03
               for report in result.report.channels_report)
    assert all(report.peak_tap == 0 for report in result.report.channels_report)
    assert result.report.as_dict()["measured_acoustics_certified"] is False


def test_sweep_has_declared_frequency_trend():
    plan = SweepPlan(sample_rate=8000, frames=8000,
                     start_hz=100, end_hz=3000, amplitude=.4, fade_frames=32)
    audio = render_log_sweep(plan)
    assert audio.frames == 8000
    assert audio.channels == 1
    assert audio.sample_rate == 8000
    assert float(np.max(np.abs(audio.samples))) <= .400001
    assert abs(float(audio.samples[0, 0])) < 1e-10
    assert not audio.samples.flags.writeable
    # Approximate zero crossing density grows as instantaneous frequency.
    front = audio.samples[0, 100:2100]
    back = audio.samples[0, 5900:7900]
    front_cross = np.count_nonzero(front[1:] * front[:-1] < 0)
    back_cross = np.count_nonzero(back[1:] * back[:-1] < 0)
    assert back_cross > front_cross * 4


@pytest.mark.parametrize("rate", [8000, 16000, 48000])
def test_sweep_deterministic_across_independent_renders(rate):
    p = SweepPlan(
        sample_rate=rate, frames=rate,
        start_hz=80, end_hz=min(7000, rate * .4),
    )
    a, b = render_log_sweep(p), render_log_sweep(p)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_empty_delay_and_impulse_fits_identity():
    source = default_stimulus()
    capture = make_buffer(source.samples, 8000)
    out = identify_linear_response(
        source, capture, IdentificationSettings(taps=1, regularization=1e-14,
                                                block_frames=64),
    )
    assert out.impulse.coefficients.shape == (1, 1, 1)
    assert out.impulse.coefficients[0, 0, 0] == pytest.approx(1, abs=.01)


def test_impulse_is_compatible_with_matrix_convolution():
    excitation = default_stimulus()
    impulse = np.zeros((1, 32))
    impulse[0, 0] = 1
    impulse[0, 7] = .5
    capture = response(excitation, impulse)
    identified = identify_linear_response(
        excitation, capture, IdentificationSettings(taps=32, block_frames=64,
                                                    regularization=1e-12),
    )
    reproduced = convolve_matrix(excitation, identified.impulse)
    np.testing.assert_allclose(reproduced.samples, capture.samples,
                               atol=.015, rtol=.02)


def test_zero_energy_source_rejected():
    source = make_buffer(np.zeros(1000, dtype=np.float32), 8000)
    capture = make_buffer(np.zeros(1200, dtype=np.float32), 8000)
    with pytest.raises(AudioContractError, match="energy too low"):
        identify_linear_response(source, capture,
                                 IdentificationSettings(taps=20))


def test_disagreeing_sample_clocks_and_channels_rejected():
    source = default_stimulus()
    capture = make_buffer(np.zeros(5000, dtype=np.float32), 16000)
    with pytest.raises(AudioContractError):
        identify_linear_response(source, capture, IdentificationSettings(taps=20))
    with pytest.raises(AudioContractError):
        identify_linear_response(
            make_buffer(np.zeros((2, 4096), dtype=np.float32), 8000),
            make_buffer(np.zeros(5000, dtype=np.float32), 8000),
            IdentificationSettings(taps=20),
        )


def test_capture_must_exceed_source_and_requested_latency():
    source = default_stimulus()
    capture = make_buffer(np.zeros(4000, dtype=np.float32), 8000)
    with pytest.raises(AudioContractError):
        identify_linear_response(source, capture,
                                 IdentificationSettings(taps=20))
    capture = make_buffer(np.zeros(4100, dtype=np.float32), 8000)
    with pytest.raises(AudioContractError):
        identify_linear_response(source, capture,
                                 IdentificationSettings(taps=20, latency_frames=4090))


@pytest.mark.parametrize("invalid", [
    dict(frames=1), dict(frames=256, start_hz=3000, end_hz=100),
    dict(frames=256, end_hz=6000), dict(frames=256, fade_frames=65),
    dict(amplitude=0), dict(amplitude=1.5), dict(amplitude=float("nan")),
    dict(start_hz=-20), dict(end_hz=float("inf")),
    dict(fade_frames=-1), dict(fade_frames=True),
])
def test_invalid_sweep_plan_rejected(invalid):
    args = dict(sample_rate=8000, frames=8000, start_hz=50, end_hz=3000)
    args.update(invalid)
    with pytest.raises(AudioContractError):
        SweepPlan(**args)


@pytest.mark.parametrize("invalid", [
    dict(taps=0), dict(taps=65537),
    dict(regularization=0), dict(regularization=.2),
    dict(latency_frames=-1), dict(block_frames=32),
    dict(block_frames=123), dict(block_frames=True),
    dict(minimum_excitation_rms=0), dict(minimum_excitation_rms=2),
])
def test_invalid_identification_settings_rejected(invalid):
    with pytest.raises(AudioContractError):
        IdentificationSettings(**invalid)


def test_resource_geometry_rejects_unbounded_fft():
    with pytest.raises(AudioLimitError):
        IdentificationGeometry(8, 2_000_000, 3_000_000, 65536)


def test_source_immutability_and_repeatable_measurement():
    source = default_stimulus()
    known = np.zeros((1, 20))
    known[0, 0], known[0, 3] = .8, -.25
    capture = response(source, known)
    before_x = source.samples.copy()
    before_y = capture.samples.copy()
    settings = IdentificationSettings(taps=20, block_frames=64,
                                      regularization=1e-12)
    first = identify_linear_response(source, capture, settings)
    second = identify_linear_response(source, capture, settings)
    np.testing.assert_array_equal(first.impulse.coefficients,
                                  second.impulse.coefficients)
    assert first.report == second.report
    np.testing.assert_array_equal(source.samples, before_x)
    np.testing.assert_array_equal(capture.samples, before_y)
    assert not first.impulse.coefficients.flags.writeable


def test_low_level_configuration_type_safety():
    source = default_stimulus()
    capture = response(source, np.ones((1, 16)) * .05)
    with pytest.raises(AudioContractError):
        identify_linear_response(source, capture, None)
    with pytest.raises(AudioContractError):
        render_log_sweep(None)
    with pytest.raises(AudioContractError):
        identify_linear_response(source, capture,
                                 IdentificationSettings(taps=1000, latency_frames=5000))
