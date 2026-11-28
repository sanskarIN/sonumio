"""GCC-PHAT signed time delays, channel selection and non-destructive shifting."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.gcc_phat_alignment import (
    AlignmentSettings, AlignmentGeometry, AlignmentReport,
    estimate_delay, shift_integer_frames, align_observed_to_reference,
)


def pair(lag=71, frames=4096, channels=1, seed=42):
    rng = np.random.default_rng(seed)
    source = rng.normal(0, .3, frames).astype(np.float32)
    observed = np.zeros(frames, dtype=np.float32)
    if lag >= 0:
        observed[lag:] = source[:frames - lag]
    else:
        observed[:frames + lag] = source[-lag:]
    if channels == 1:
        return make_buffer(source, 16000), make_buffer(observed, 16000)
    reference = np.stack([source] + [source * .4] * (channels - 1))
    captured = np.stack([observed] + [observed * .6] * (channels - 1))
    return make_buffer(reference, 16000), make_buffer(captured, 16000)


def config(**updates):
    values = dict(maximum_lag_frames=512)
    values.update(updates)
    return AlignmentSettings(**values)


@pytest.mark.parametrize("lag", [-300, -123, -1, 0, 1, 35, 127, 300])
@pytest.mark.parametrize("phat_exponent", [0., .5, 1.])
def test_known_positive_and_negative_delay_recovered(lag, phat_exponent):
    x, y = pair(lag)
    result = estimate_delay(x, y, config(phat_exponent=phat_exponent))
    assert result.hypothesis is not None
    assert result.hypothesis.lag_frames == lag
    assert abs(result.hypothesis.fractional_lag_frames - lag) < .5
    assert result.hypothesis.seconds == pytest.approx(
        result.hypothesis.fractional_lag_frames / 16000,
    )
    assert result.sample_rate == 16000
    assert result.hypothesis.peak_ratio > 1


def test_hypothesis_compensates_capture_to_reference_clock():
    x, y = pair(89)
    result = estimate_delay(x, y, config())
    aligned = align_observed_to_reference(y, result)
    np.testing.assert_array_equal(aligned.samples[0, :-89], x.samples[0, :-89])
    np.testing.assert_array_equal(aligned.samples[0, -89:], 0)
    assert aligned.spec == y.spec


@pytest.mark.parametrize("shift", [-120, -1, 0, 1, 120, 4500])
def test_integer_shift_exact_samples_and_output_geometry(shift):
    audio = make_buffer(np.arange(1, 501, dtype=np.float32), 16000)
    result = shift_integer_frames(audio, shift)
    assert result.spec == audio.spec
    expected = np.zeros(500, dtype=np.float32)
    if abs(shift) < 500:
        if shift >= 0:
            expected[shift:] = audio.samples[0, :500 - shift]
        else:
            expected[:500 + shift] = audio.samples[0, -shift:]
    np.testing.assert_array_equal(result.samples[0], expected)
    assert not result.samples.flags.writeable


def test_selected_channels_not_implicitly_downmixed():
    x, y = pair(51, channels=2)
    opts = config(reference_channel=1, observed_channel=1)
    result = estimate_delay(x, y, opts)
    assert result.hypothesis.lag_frames == 51
    assert result.reference_channel == 1
    assert result.observed_channel == 1


def test_zero_energy_must_return_unknown_not_fabricated_lag():
    x = make_buffer(np.zeros(1024, dtype=np.float32), 8000)
    y = make_buffer(np.zeros(1024, dtype=np.float32), 8000)
    result = estimate_delay(x, y, config())
    assert result.hypothesis is None
    assert result.reference_rms == 0
    assert result.observed_rms == 0
    with pytest.raises(AudioContractError):
        align_observed_to_reference(y, result)


def test_report_exposes_uncertainty_without_physical_certification():
    x, y = pair(49)
    report = estimate_delay(x, y, config()).as_dict()
    assert report["protocol"] == "sonumio-gcc-phat-alignment-v1"
    assert report["model_inference"] is False
    assert report["clock_drift_estimated"] is False
    assert report["physical_propagation_delay_certified"] is False
    assert report["automatic_resampling"] is False
    assert report["hypothesis"]["lag_frames"] == 49


def test_source_immutable_even_after_alignment():
    x, y = pair(85, channels=2)
    old_x, old_y = x.samples.copy(), y.samples.copy()
    result = estimate_delay(x, y, config())
    aligned = align_observed_to_reference(y, result)
    np.testing.assert_array_equal(x.samples, old_x)
    np.testing.assert_array_equal(y.samples, old_y)
    assert aligned.sample_rate == x.sample_rate
    assert aligned.channels == 2


def test_custom_layout_preserved_through_integer_shift():
    audio = make_buffer(
        np.arange(2 * 500, dtype=np.float32).reshape(2, 500), 16000,
        layout=AudioLayout(2, "discrete", ("near", "far")),
    )
    shifted = shift_integer_frames(audio, -80)
    assert shifted.spec.layout == audio.spec.layout
    np.testing.assert_array_equal(shifted.samples[:, :420], audio.samples[:, 80:])


def test_identical_replays_are_numerically_deterministic():
    x, y = pair(-38, seed=77)
    a = estimate_delay(x, y, config())
    b = estimate_delay(x, y, config())
    assert a == b
    aligned_a = align_observed_to_reference(y, a)
    aligned_b = align_observed_to_reference(y, b)
    np.testing.assert_array_equal(aligned_a.samples, aligned_b.samples)


def test_mismatched_sample_clocks_and_channel_indices_rejected():
    x, y = pair(50)
    changed = make_buffer(y.samples, 8000)
    with pytest.raises(AudioContractError):
        estimate_delay(x, changed, config())
    with pytest.raises(AudioContractError):
        estimate_delay(x, y, config(reference_channel=2))
    with pytest.raises(AudioContractError):
        estimate_delay(x, y, config(observed_channel=2))


@pytest.mark.parametrize("bad", [
    dict(maximum_lag_frames=0), dict(maximum_lag_frames=100001),
    dict(maximum_lag_frames=True), dict(reference_channel=-1),
    dict(observed_channel=8), dict(remove_dc=0),
    dict(taper=1), dict(refinement=0), dict(phat_exponent=-1),
    dict(phat_exponent=2), dict(minimum_signal_rms=-1),
    dict(minimum_peak_score=2), dict(phat_exponent=float("nan")),
])
def test_invalid_alignment_settings_rejected(bad):
    with pytest.raises(AudioContractError):
        AlignmentSettings(**bad)


def test_lag_bound_must_fit_signals():
    x, y = pair(3, frames=80)
    with pytest.raises(AudioContractError):
        estimate_delay(x, y, config(maximum_lag_frames=90))


def test_geometry_resource_budget_is_independent_of_audio_allocation():
    with pytest.raises(AudioLimitError):
        AlignmentGeometry(500000, 500000, 100, 8, 8)


@pytest.mark.parametrize("lag", [-100001, 100001, 2.5, True])
def test_invalid_integer_shifts_rejected(lag):
    x, _ = pair(0)
    with pytest.raises(AudioContractError):
        shift_integer_frames(x, lag)


def test_untrusted_report_or_unrelated_clock_is_rejected():
    x, y = pair(20)
    actual = estimate_delay(x, y, config())
    with pytest.raises(AudioContractError):
        align_observed_to_reference(y, None)
    unrelated = make_buffer(y.samples, 8000)
    with pytest.raises(AudioContractError):
        align_observed_to_reference(unrelated, actual)
