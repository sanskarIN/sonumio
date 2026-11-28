"""Sparse impulse restoration: unchanged anchors, limits and explicit masks."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.click_repair import (
    ClickRepairOptions, ClickDetection, detect_clicks, repair_clicks,
    _cubic, _linear,
)


def quiet(frames=4000, channels=1):
    return make_buffer(np.zeros((channels, frames), dtype=np.float32), 16000)


def corrupted(frames=4000, frames_with_clicks=(300, 1800), channels=1):
    data = np.zeros((channels, frames), dtype=np.float32)
    for channel in range(channels):
        for i, position in enumerate(frames_with_clicks):
            data[channel, position] = (-1 if i % 2 else 1) * (0.8 - channel * 0.1)
    return make_buffer(data, 16000)


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("channels", [1, 2, 5])
def test_detect_and_repair_isolated_known_impulses(channels, interpolation):
    audio = corrupted(channels=channels)
    options = ClickRepairOptions(interpolation=interpolation)
    detection = detect_clicks(audio, options)
    assert detection.count == 2 * channels
    assert detection.mask.dtype == np.bool_
    assert not detection.mask.flags.writeable
    assert detection.residual.dtype == np.float64
    assert not detection.residual.flags.writeable
    for channel in range(channels):
        assert tuple(np.flatnonzero(detection.mask[channel])) == (300, 1800)
    repaired = repair_clicks(audio, options)
    assert repaired.audio.frames == audio.frames
    assert repaired.audio.channels == audio.channels
    assert repaired.audio.sample_rate == audio.sample_rate
    np.testing.assert_array_equal(repaired.audio.samples, 0)
    assert repaired.report.corrected_frames == 2 * channels
    assert len(repaired.report.events) == 2 * channels
    assert repaired.report.input_peak == pytest.approx(.8)
    assert repaired.report.output_peak == 0
    report = repaired.report.as_dict()
    assert report["model_inference"] is False
    assert report["perceptual_quality_established"] is False
    assert report["automatic_transient_classification"] is False


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_manual_mask_repairs_only_selected_channel_samples(interpolation):
    rng = np.random.default_rng(991)
    data = rng.normal(0, .005, (2, 2000)).astype(np.float32)
    data[1, 400:402] = 1.
    audio = make_buffer(data, 16000)
    mask = np.zeros((2, 2000), dtype=bool)
    mask[1, 400:402] = True
    repaired = repair_clicks(
        audio, ClickRepairOptions(interpolation=interpolation), explicit_mask=mask,
    )
    np.testing.assert_array_equal(repaired.audio.samples[0], audio.samples[0])
    np.testing.assert_array_equal(repaired.audio.samples[1, :400],
                                  audio.samples[1, :400])
    np.testing.assert_array_equal(repaired.audio.samples[1, 402:],
                                  audio.samples[1, 402:])
    assert not np.array_equal(repaired.audio.samples[1, 400:402],
                              audio.samples[1, 400:402])
    assert repaired.report.corrected_frames == 2
    assert repaired.report.events[0].channel == 1
    assert repaired.report.events[0].start_frame == 400
    assert repaired.report.events[0].stop_frame == 402
    # The caller's explicit mask is neither modified nor retained mutably.
    mask[:] = False
    assert repaired.detection.count == 2


def test_channel_linkage_is_explicit_and_preserves_timing():
    frames = 2000
    samples = np.zeros((2, frames), dtype=np.float32)
    samples[0, 300] = .8
    samples[1, 400] = -.7
    audio = make_buffer(samples, 8000)
    independently = detect_clicks(audio)
    assert np.flatnonzero(independently.mask[0]).tolist() == [300]
    assert np.flatnonzero(independently.mask[1]).tolist() == [400]
    linked = detect_clicks(audio, ClickRepairOptions(link_channels=True))
    assert np.flatnonzero(linked.mask[0]).tolist() == [300, 400]
    assert np.flatnonzero(linked.mask[1]).tolist() == [300, 400]


def test_unmasked_samples_remain_bitwise_untouched():
    rng = np.random.default_rng(2026)
    data = rng.normal(0, .03, 4096).astype(np.float32)
    data[2000] = 1.
    audio = make_buffer(data, 24000)
    options = ClickRepairOptions(mad_multiplier=20., minimum_deviation=.4)
    out = repair_clicks(audio, options)
    protected = ~out.detection.mask[0]
    np.testing.assert_array_equal(out.audio.samples[0, protected],
                                  audio.samples[0, protected])
    assert out.report.corrected_frames <= 5


def test_linear_interpolation_exact_anchors_and_step():
    values = _linear(1., 5., 3)
    np.testing.assert_array_equal(values, [2., 3., 4.])


@pytest.mark.parametrize("samples,left,right", [
    ([-3., 0., 5., 6., 7., 10., 14.], 1, 5),
    ([3., 2., 1., 0., -1., -2., -3.], 1, 5),
    ([0., 2., 2., 2., 2., 2., 0.], 1, 5),
])
def test_cubic_interpolation_is_anchor_bounded(samples, left, right):
    data = np.array(samples, dtype=np.float64)
    values = _cubic(data, left, right)
    assert len(values) == right - left - 1
    assert np.isfinite(values).all()
    assert min(data[left], data[right]) - 1e-10 <= np.min(values)
    assert np.max(values) <= max(data[left], data[right]) + 1e-10


def test_no_correction_returns_identical_waveform():
    audio = quiet()
    result = repair_clicks(audio)
    assert result.detection.count == 0
    assert result.report.corrected_frames == 0
    assert result.report.events == ()
    np.testing.assert_array_equal(result.audio.samples, audio.samples)


def test_long_transients_are_not_automatically_interpolated():
    data = np.zeros(4000, dtype=np.float32)
    data[200:205] = 1
    audio = make_buffer(data, 16000)
    result = repair_clicks(audio)
    # A sustained five-sample transient exceeds max_run_frames=3.
    np.testing.assert_array_equal(result.audio.samples, audio.samples)


def test_dense_spikes_exceed_guard_and_do_not_return_output():
    data = np.zeros(4000, dtype=np.float32)
    data[100:3900:10] = .8
    with pytest.raises(AudioLimitError):
        detect_clicks(make_buffer(data, 16000))


@pytest.mark.parametrize("options", [
    dict(window_frames=4), dict(window_frames=2),
    dict(window_frames=32), dict(max_run_frames=5),
    dict(max_run_frames=0), dict(mad_multiplier=2),
    dict(mad_multiplier=float("nan")), dict(minimum_deviation=0),
    dict(max_fraction=.11), dict(max_fraction=-1),
    dict(interpolation="neural"), dict(link_channels=1),
])
def test_invalid_options_rejected(options):
    with pytest.raises(AudioContractError):
        ClickRepairOptions(**options)


@pytest.mark.parametrize("selected", [
    [(0, 0)], [(0, 999)], [(0, 500), (0, 501), (0, 502), (0, 503)],
])
def test_manual_mask_cannot_touch_edges_or_long_runs(selected):
    audio = quiet(1000)
    mask = np.zeros((1, 1000), dtype=bool)
    for c, frame in selected:
        mask[c, frame] = True
    with pytest.raises(AudioContractError):
        repair_clicks(audio, explicit_mask=mask)


def test_manual_mask_fraction_bound():
    audio = quiet(1000)
    mask = np.zeros((1, 1000), dtype=bool)
    mask[0, 10:900:10] = True
    with pytest.raises(AudioLimitError):
        repair_clicks(audio, explicit_mask=mask)


@pytest.mark.parametrize("bad", [
    None, np.zeros((3, 1000), dtype=bool),
    np.zeros((1, 1001), dtype=bool),
    np.zeros((1, 1000), dtype=np.float64),
])
def test_invalid_mask_geometry_and_dtype(bad):
    audio = quiet(1000)
    if bad is None:
        with pytest.raises(AudioContractError):
            ClickDetection(np.zeros((1, 1000), dtype=np.float32),
                           np.zeros((1, 1000), dtype=np.float64),
                           ClickRepairOptions())
        return
    with pytest.raises(AudioContractError):
        repair_clicks(audio, explicit_mask=bad)


def test_too_short_audio_is_rejected():
    audio = quiet(11)
    with pytest.raises(AudioContractError):
        detect_clicks(audio)


def test_detection_work_budget_rejected_without_large_allocations():
    audio = quiet(200000, 2)
    with pytest.raises(AudioLimitError):
        detect_clicks(audio, ClickRepairOptions(window_frames=31))


def test_options_hash_and_mask_copies_are_immutable():
    audio = corrupted()
    detected = detect_clicks(audio)
    with pytest.raises(ValueError):
        detected.mask[0, 300] = False
    with pytest.raises(ValueError):
        detected.residual[0, 300] = 0
    with pytest.raises(ValueError):
        detected.mask.setflags(write=True)
    with pytest.raises(ValueError):
        detected.residual.setflags(write=True)
