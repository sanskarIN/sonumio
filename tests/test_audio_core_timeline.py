"""Nondestructive multi-track rendering, source geometry and editing checks."""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.timeline import (
    Clip, Timeline, Track, render_timeline, timeline_report,
)
from sonumio.audio_core.segments import (
    chunks, concatenate, crossfade, detect_silence, loop,
    pad, reverse, split_at, trim, trim_edges_by_silence,
)
from sonumio.audio_core.errors import AudioContractError, AudioLimitError


@pytest.fixture
def one():
    a = np.zeros(2400, dtype=np.float32)
    a[240:2160] = .5
    return make_buffer(a, 24000)


@pytest.fixture
def two():
    return make_buffer(np.stack((
        np.linspace(-.25, .25, 2400, dtype=np.float32),
        np.linspace(.25, -.25, 2400, dtype=np.float32),
    )), 24000)


def test_clip_source_range_frames_shift_and_new_subclip(one):
    original = Clip("clip", one, 100, source_start=200,
                    source_stop=500, gain_db=-3)
    assert original.frames == 300
    assert original.stop_frame == 400
    assert original.shifted(10).start_frame == 110
    excerpt = original.subclip(20, 40)
    assert excerpt.start_frame == 120
    assert excerpt.source_start == 220
    assert excerpt.source_stop == 240
    assert excerpt.frames == 20
    with pytest.raises(AudioContractError):
        original.subclip(0, 500)


def test_mono_source_to_stereo_timeline_pan_is_equal_power(one):
    left = Clip("left", one, 0, pan=-1)
    right = Clip("right", one, 0, pan=1)
    timeline_left = Timeline(
        24000, AudioLayout.for_channels(2), 2400,
        (Track("track_a", (left,)),)
    )
    timeline_right = Timeline(
        24000, AudioLayout.for_channels(2), 2400,
        (Track("track_a", (right,)),)
    )
    l = render_timeline(timeline_left)
    r = render_timeline(timeline_right)
    assert l.channels == r.channels == 2
    np.testing.assert_allclose(l.samples[1], 0, atol=1e-7)
    np.testing.assert_allclose(r.samples[0], 0, atol=1e-7)
    assert l.samples[0, 500] == pytest.approx(.5)
    assert r.samples[1, 500] == pytest.approx(.5)


def test_overlapping_tracks_gain_normalization_and_input_immutability(one):
    a = Track("track_a", (Clip("a", one, 0),))
    b = Track("track_b", (Clip("b", one, 0),))
    canvas = Timeline(24000, AudioLayout.for_channels(1), 2400, (a, b))
    output = render_timeline(canvas, normalization="none")
    assert output.samples[0, 500] == pytest.approx(1)
    assert one.samples[0, 500] == pytest.approx(.5)
    report = timeline_report(canvas, rendered=output)
    assert report["track_count"] == 2
    assert report["clip_count"] == 2
    assert report["render_peak"] == pytest.approx(1)
    assert report["not_neural_inference"] is True


def test_mute_and_solo_precedence(two):
    a = Track("track_a", (Clip("a", two, 0),), solo=True)
    b = Track("track_b", (Clip("b", two, 0),))
    c = Track("track_c", (Clip("c", two, 0),), mute=True)
    canvas = Timeline(24000, AudioLayout.for_channels(2), 2400, (a, b, c))
    output = render_timeline(canvas)
    np.testing.assert_array_equal(output.samples, two.samples)
    none = Timeline(24000, AudioLayout.for_channels(2), 2400, (
        replace(a, mute=True), replace(b, mute=True),
    ))
    np.testing.assert_array_equal(render_timeline(none).samples,
                                  np.zeros((2, 2400), dtype=np.float32))


def test_timeline_tracks_are_immutable_and_ordered_by_ids(one):
    a = Track("a", (Clip("x", one, 0),))
    b = Track("b", (Clip("y", one, 0),))
    timeline = Timeline(24000, AudioLayout.for_channels(1), 2400, (a, b))
    reversed_tracks = timeline.reorder_tracks(["b", "a"])
    assert [x.track_id for x in reversed_tracks.tracks] == ["b", "a"]
    assert [x.track_id for x in timeline.tracks] == ["a", "b"]
    assert timeline.without_track("a").tracks == (b,)
    with pytest.raises(AudioContractError):
        timeline.reorder_tracks(["b", "b"])
    with pytest.raises(AudioContractError):
        timeline.with_track(a)


def test_invalid_clip_bounds_and_audio_clock_are_rejected(one):
    with pytest.raises(AudioContractError):
        Clip("?", one, 0)
    with pytest.raises(AudioContractError):
        Clip("a", one, -1)
    with pytest.raises(AudioContractError):
        Clip("a", one, 0, fade_in=2000, fade_out=1000)
    with pytest.raises(AudioContractError):
        Track("a", (Clip("same", one, 0), Clip("same", one, 0)))
    with pytest.raises(AudioContractError):
        Timeline(44100, AudioLayout.for_channels(1), 2400,
                 (Track("a", (Clip("x", one, 0),)),))
    with pytest.raises(AudioContractError):
        Timeline(24000, AudioLayout.for_channels(1), 2400,
                 (Track("a", (Clip("x", one, 10),)),))


def test_trim_reverse_and_padding_leave_original_untouched(one):
    small = trim(one, 200, 500)
    assert small.frames == 300
    assert small.samples[0, 100] == pytest.approx(.5)
    backwards = reverse(one)
    np.testing.assert_array_equal(backwards.samples[0], one.samples[0, ::-1])
    padded = pad(small, before=50, after=60, mode="zeros")
    assert padded.frames == 410
    assert np.all(padded.samples[:, :50] == 0)
    assert one.frames == 2400


def test_concatenation_boundary_and_sample_rate_protection(one, two):
    merged = concatenate((one, one))
    assert merged.frames == 4800
    assert merged.channels == 1
    with pytest.raises(AudioContractError):
        concatenate((one, two))
    other_clock = make_buffer(one.samples.copy(), 44100)
    with pytest.raises(AudioContractError):
        concatenate((one, other_clock))


def test_crossfade_concat_length_and_clip_edge_bounds(one):
    a = one.sliced(240, 1240)
    b = one.sliced(1240, 2240)
    for law in ("linear", "equal_power", "cosine"):
        result = crossfade(a, b, frames=100, law=law)
        assert result.frames == 1900
        assert np.isfinite(result.samples).all()
    with pytest.raises(AudioContractError):
        crossfade(a, b, frames=1001)


def test_loop_with_crossover_has_expected_length(one):
    small = one.sliced(240, 1240)
    repeated = loop(small, 4, blend_frames=100)
    assert repeated.frames == 4000 - 300
    assert repeated.channels == 1
    with pytest.raises(AudioContractError):
        loop(small, 0)
    with pytest.raises(AudioLimitError):
        loop(small, 100, max_seconds=1)


def test_splits_and_chunks_cover_original_frames(one):
    pieces = split_at(one, [100, 500, 1200])
    assert [x.frames for x in pieces] == [100, 400, 700, 1200]
    np.testing.assert_array_equal(concatenate(pieces).samples, one.samples)
    chunked = chunks(one, 512)
    assert [x.frames for x in chunked] == [512, 512, 512, 512, 352]
    with pytest.raises(AudioContractError):
        split_at(one, [500, 100])


def test_silence_detection_and_edge_trim_are_block_resolution_approximations(one):
    regions = detect_silence(one, threshold_dbfs=-45,
                              window_frames=64, minimum_frames=64)
    assert len(regions) == 2
    assert regions[0].start_frame == 0
    assert regions[-1].stop_frame == one.frames
    cropped = trim_edges_by_silence(one, threshold_dbfs=-45,
                                    window_frames=64)
    assert cropped.frames < one.frames
    assert cropped.frames > 1700
    with pytest.raises(AudioContractError):
        trim_edges_by_silence(make_buffer(np.zeros(1024, dtype=np.float32), 24000))
