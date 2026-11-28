"""Deterministic YIN-style F0 hypotheses for bounded monophonic references."""
import math

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.yin_pitch import (
    PitchSettings, PitchGeometry, PitchFrame,
    _difference, _cmndf, _select_lag, _refine_lag,
    track_pitch,
)


def tone(frequency, *, rate=16000, seconds=.75, channels=1, level=.4):
    t = np.arange(round(seconds * rate)) / rate
    base = (level * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    if channels == 1:
        return make_buffer(base, rate)
    return make_buffer(np.stack([base] + [(-1) ** i * base for i in range(1, channels)]),
                       rate)


@pytest.mark.parametrize("frequency", [90., 110., 220., 440., 660., 880., 1200.])
@pytest.mark.parametrize("rate", [8000, 16000, 24000])
def test_sine_pitch_converges_near_known_fundamental(frequency, rate):
    audio = tone(frequency, rate=rate, seconds=.6)
    settings = PitchSettings(
        frame_frames=1024, hop_frames=256,
        minimum_hz=60., maximum_hz=1400.,
    )
    result = track_pitch(audio, settings)
    voiced = [frame.frequency_hz for frame in result.frames if frame.voiced
              and .15 < frame.seconds < .45]
    assert voiced
    median = float(np.median(voiced))
    assert abs(median - frequency) < max(5, frequency * .015)
    assert result.summary.median_pitch_hz is not None
    assert result.summary.voiced_fraction > .4
    assert result.summary.median_confidence > .5
    assert result.summary.protocol == "sonumio-yin-monophonic-v1"


def test_silence_produces_zero_voiced_frames():
    audio = make_buffer(np.zeros(8000, dtype=np.float32), 8000)
    result = track_pitch(audio)
    assert result.summary.voiced_frames == 0
    assert result.summary.voiced_fraction == 0
    assert result.summary.median_pitch_hz is None
    assert result.summary.minimum_pitch_hz is None
    assert result.summary.maximum_pitch_hz is None
    assert all(v.frequency_hz is None and v.confidence == 0 for v in result.frames)


def test_opposite_polarity_channels_do_not_cancel_pitch():
    audio = tone(220, channels=2)
    result = track_pitch(audio)
    assert result.summary.voiced_frames > 0
    assert result.summary.median_pitch_hz == pytest.approx(220, abs=5)
    assert all(frame.selected_channel in (0, 1) for frame in result.frames)


def test_independent_frames_are_deterministic_and_source_unchanged():
    audio = tone(330, channels=2)
    before = audio.samples.copy()
    a = track_pitch(audio)
    b = track_pitch(audio)
    assert a.frames == b.frames
    assert a.summary == b.summary
    assert a.as_dict() == b.as_dict()
    np.testing.assert_array_equal(audio.samples, before)
    assert a.summary.as_dict()["model_inference"] is False
    assert a.summary.as_dict()["polyphonic_transcription_claimed"] is False


def test_normalized_difference_picks_known_period():
    rate = 8000
    period = 40
    data = np.sin(2 * np.pi * np.arange(1024 + 120) / period)
    curve = _cmndf(_difference(data, 1024, 120))
    lag = _select_lag(curve, 15, 100, .15, .3)
    assert lag == period
    assert curve[period] < 1e-6
    assert _refine_lag(curve, lag) == pytest.approx(period, abs=.1)


def test_manual_threshold_rejects_nonperiodic_difference():
    curve = np.ones(20, dtype=np.float64)
    assert _select_lag(curve, 2, 15, .15, .3) is None
    with pytest.raises(AudioContractError):
        _select_lag(curve, 3, 30, .15, .3)


def test_subsample_interpolation_is_optional():
    audio = tone(440)
    interpolate = track_pitch(audio, PitchSettings(interpolate_lag=True))
    coarse = track_pitch(audio, PitchSettings(interpolate_lag=False))
    candidates = [v.frequency_hz for v in interpolate.frames if v.voiced]
    rounded = [v.frequency_hz for v in coarse.frames if v.voiced]
    assert candidates and rounded
    assert np.std(candidates) <= max(3, .2 * np.std(rounded))
    assert interpolate.summary.sample_rate == coarse.summary.sample_rate


def test_pitch_geometry_work_and_memory_bounds():
    g = PitchGeometry(16000, 1, 8000, 1024, 256, 65., 1400.)
    assert g.output_frames == 32
    assert g.minimum_lag == 12
    assert g.maximum_lag == 246
    with pytest.raises(AudioLimitError):
        PitchGeometry(16000, 8, 16000 * 100, 4096, 64, 20., 1600.)


@pytest.mark.parametrize("bad", [
    dict(frame_frames=127), dict(frame_frames=4097),
    dict(frame_frames=True), dict(hop_frames=0),
    dict(hop_frames=512, frame_frames=256),
    dict(minimum_hz=10), dict(minimum_hz=1000, maximum_hz=500),
    dict(maximum_hz=9000), dict(minimum_hz=float("nan")),
    dict(yin_threshold=0), dict(yin_threshold=.99),
    dict(fallback_threshold=.01), dict(fallback_threshold=1),
    dict(silence_dbfs=-130), dict(silence_dbfs=0),
    dict(interpolate_lag=0),
])
def test_invalid_pitch_settings_rejected(bad):
    with pytest.raises(AudioContractError):
        PitchSettings(**bad)


def test_invalid_audio_and_geometry_rejected():
    with pytest.raises(AudioContractError):
        track_pitch(None)
    with pytest.raises(AudioContractError):
        track_pitch(tone(440), None)
    with pytest.raises(AudioContractError):
        PitchGeometry(16000, 0, 8000, 1024, 256, 65., 1400.)
    with pytest.raises(AudioContractError):
        PitchGeometry(16000, 1, 8000, 1024, 256, 500., 200.)


@pytest.mark.parametrize("bad", [
    np.zeros((2, 2)), np.zeros(2), np.array([0, 1, 2]),
    np.array([0., -1., 2.]), np.array([0., np.nan, 2.]),
])
def test_invalid_cmndf_input_rejected(bad):
    with pytest.raises(AudioContractError):
        _cmndf(bad)


def test_huge_difference_work_rejected_before_processing():
    audio = make_buffer(np.zeros(8000 * 119, dtype=np.float32), 8000)
    with pytest.raises(AudioLimitError):
        track_pitch(audio, PitchSettings(frame_frames=4096,
                                         hop_frames=32, minimum_hz=20))


def test_pitch_frame_report_requires_no_model_or_identity():
    audio = tone(100)
    result = track_pitch(audio)
    report = result.as_dict()
    assert report["model_inference"] is False
    assert all("voiced" in f for f in report["frames"])
    assert report["summary"]["sample_rate"] == 16000
    assert report["summary"]["source_frames"] == audio.frames
    assert not any("speaker" in key for key in report)
