"""Clip-slicer timeline parity, reverse rates, non-destructive source and bounds."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.sample_slicer import (
    SampleSlice, SamplerTrigger, SamplerSettings,
    sample_slices_from_onsets, render_sampler,
)


def mono(frames=4000):
    values = np.sin(2 * np.pi * 160 * np.arange(frames) / 8000)
    return make_buffer(values.astype(np.float32), 8000)


@pytest.mark.parametrize("rate", [.5, 1., 1.5, 2., 4.])
def test_playback_rate_has_declared_sample_clock_duration(rate):
    source = mono()
    span = SampleSlice(100, 900)
    trigger = SamplerTrigger(0, 0, rate=rate, fade_frames=0)
    frames = int(np.ceil(span.frames / rate))
    plan = SamplerSettings(frames + 100, channels=1, gain=1.)
    result = render_sampler(source, (span,), (trigger,), plan)
    assert result.audio.frames == frames + 100
    assert result.report.events[0].stop_frame == frames
    assert result.report.events[0].playback_rate == rate
    np.testing.assert_array_equal(result.audio.samples[0, frames:], 0)


def test_unity_mono_sampler_is_exact_forward_slice():
    source = mono()
    span = SampleSlice(100, 300)
    plan = SamplerSettings(400, channels=1, gain=1.)
    result = render_sampler(
        source, (span,), (SamplerTrigger(10, 0, fade_frames=0),), plan,
    )
    np.testing.assert_array_equal(
        result.audio.samples[0, 10:210], source.samples[0, 100:300],
    )
    np.testing.assert_array_equal(result.audio.samples[0, :10], 0)
    np.testing.assert_array_equal(result.audio.samples[0, 210:], 0)


def test_unity_reverse_sampler_reverses_exact_half_open_slice():
    source = mono()
    span = SampleSlice(100, 300)
    plan = SamplerSettings(300, channels=1, gain=1.)
    result = render_sampler(
        source, (span,), (
            SamplerTrigger(0, 0, reverse=True, fade_frames=0),
        ), plan,
    )
    np.testing.assert_array_equal(
        result.audio.samples[0, :200], source.samples[0, 100:300][:, None].T[0, ::-1],
    )


def test_rate_two_uses_every_second_original_sample():
    source = mono()
    result = render_sampler(
        source, (SampleSlice(100, 300),),
        (SamplerTrigger(0, 0, rate=2, fade_frames=0),),
        SamplerSettings(200, channels=1, gain=1.),
    )
    np.testing.assert_array_equal(result.audio.samples[0, :100],
                                  source.samples[0, 100:300:2])


def test_mono_to_stereo_panning_endpoints():
    source = mono()
    span = (SampleSlice(100, 300),)
    p = SamplerSettings(400, channels=2, gain=.5)
    left = render_sampler(source, span, (
        SamplerTrigger(0, 0, pan=-1., fade_frames=0),
    ), p).audio.samples
    right = render_sampler(source, span, (
        SamplerTrigger(0, 0, pan=1., fade_frames=0),
    ), p).audio.samples
    assert np.max(np.abs(left[0])) > .1
    assert np.max(np.abs(left[1])) < 1e-10
    assert np.max(np.abs(right[1])) > .1
    assert np.max(np.abs(right[0])) < 1e-10


def test_source_custom_stereo_labels_preserved():
    source = make_buffer(
        np.ones((2, 500), dtype=np.float32) * .1, 8000,
        layout=AudioLayout(2, "discrete", ("drum-L", "drum-R")),
    )
    result = render_sampler(
        source, (SampleSlice(0, 100),),
        (SamplerTrigger(0, 0, fade_frames=0),),
        SamplerSettings(300, channels=2),
    )
    assert result.audio.spec.layout == source.spec.layout


def test_sample_regions_derived_from_onsets_are_half_open():
    source = mono(1000)
    regions = sample_slices_from_onsets(source, (100, 230, 400))
    assert [(x.start_frame, x.stop_frame) for x in regions] == [
        (0, 100), (100, 230), (230, 400), (400, 1000),
    ]
    assert [x.label for x in regions] == [
        "slice-1", "slice-2", "slice-3", "slice-4",
    ]
    without_lead = sample_slices_from_onsets(
        source, (100, 230, 400), include_lead=False,
    )
    assert without_lead[0].start_frame == 100
    filtered = sample_slices_from_onsets(
        source, (100, 130, 400), min_frames=100,
    )
    assert all(x.frames >= 100 for x in filtered)


def test_clip_and_rate_derived_duration_do_not_exceed_source():
    source = mono(900)
    result = render_sampler(
        source, (SampleSlice(100, 900),),
        (SamplerTrigger(0, 0, rate=.5, fade_frames=10),),
        SamplerSettings(2000, channels=1),
    )
    assert result.report.events[0].stop_frame == 1600
    assert np.isfinite(result.audio.samples).all()


def test_two_triggers_mix_exactly_when_no_fades():
    source = make_buffer(np.ones(500, dtype=np.float32) * .2, 8000)
    plan = SamplerSettings(500, channels=1, gain=.5)
    one = SamplerTrigger(0, 0, fade_frames=0)
    a = render_sampler(source, (SampleSlice(0, 100),), (one,), plan)
    b = render_sampler(source, (SampleSlice(0, 100),), (one, one), plan)
    np.testing.assert_allclose(b.audio.samples, 2 * a.audio.samples, atol=1e-7)
    assert b.report.peak_polyphony == 2


def test_empty_score_outputs_silence():
    source = mono()
    result = render_sampler(source, (SampleSlice(0, 100),), (),
                            SamplerSettings(500, channels=1))
    np.testing.assert_array_equal(result.audio.samples, 0)
    assert result.report.trigger_count == 0
    assert result.report.peak_polyphony == 0


def test_deterministic_repeat_and_source_immutable():
    source = mono()
    before = source.samples.copy()
    spans = (SampleSlice(0, 300), SampleSlice(300, 600))
    events = (
        SamplerTrigger(0, 0, rate=1.2),
        SamplerTrigger(300, 1, reverse=True, rate=.8),
    )
    plan = SamplerSettings(1200, channels=2)
    a = render_sampler(source, spans, events, plan)
    b = render_sampler(source, spans, events, plan)
    np.testing.assert_array_equal(a.audio.samples, b.audio.samples)
    assert a.report == b.report
    np.testing.assert_array_equal(source.samples, before)
    assert not a.audio.samples.flags.writeable
    assert a.report.as_dict()["pitch_preservation_claimed"] is False


@pytest.mark.parametrize("bad", [
    dict(start_frame=-1), dict(start_frame=10, stop_frame=10),
    dict(stop_frame=0), dict(label="bad\nlabel"), dict(label=123),
])
def test_invalid_slice_rejected(bad):
    fields = dict(start_frame=0, stop_frame=100)
    fields.update(bad)
    with pytest.raises(AudioContractError):
        SampleSlice(**fields)


@pytest.mark.parametrize("bad", [
    dict(start_frame=-1), dict(slice_index=-1),
    dict(velocity=-.1), dict(velocity=2),
    dict(rate=0), dict(rate=6), dict(rate=float("inf")),
    dict(reverse=1), dict(pan=2), dict(fade_frames=-1),
])
def test_invalid_trigger_rejected(bad):
    fields = dict(start_frame=0, slice_index=0)
    fields.update(bad)
    with pytest.raises(AudioContractError):
        SamplerTrigger(**fields)


@pytest.mark.parametrize("bad", [
    dict(frames=0), dict(channels=0),
    dict(channels=3), dict(gain=-1), dict(gain=2),
])
def test_invalid_output_plan_rejected(bad):
    fields = dict(frames=1000)
    fields.update(bad)
    with pytest.raises(AudioContractError):
        SamplerSettings(**fields)


def test_missing_slice_and_bad_clock_rejected():
    source = mono()
    with pytest.raises(AudioContractError):
        render_sampler(
            source, (SampleSlice(0, 100),),
            (SamplerTrigger(0, 1),),
            SamplerSettings(200, channels=1),
        )
    with pytest.raises(AudioContractError):
        render_sampler(
            source, (SampleSlice(0, 100),),
            (SamplerTrigger(190, 0),),
            SamplerSettings(200, channels=1),
        )


def test_active_voice_bound_is_preflighted():
    source = mono()
    simultaneous = tuple(SamplerTrigger(0, 0) for _ in range(65))
    with pytest.raises(AudioLimitError):
        render_sampler(source, (SampleSlice(0, 100),), simultaneous,
                       SamplerSettings(500, channels=1))


def test_invalid_onsets_and_duplicate_boundaries_rejected():
    source = mono(1000)
    for invalid in ((100, 100), (200, 100), (0, 100),
                    (100, 1000), (-1,), (True,), (100.5,)):
        with pytest.raises(AudioContractError):
            sample_slices_from_onsets(source, invalid)
    with pytest.raises(AudioContractError):
        sample_slices_from_onsets(source, [], include_lead=True)
