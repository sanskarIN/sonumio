"""Rhythm analysis determinism, tempo hypotheses and bounded validation."""
import math

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.rhythm import (
    RhythmSettings, RhythmAnalysis, TempoCandidate, OnsetEvent,
    analyze_rhythm, novelty_envelope, pick_onsets, estimate_tempo,
)


def metronome(bpm=120, rate=8000, seconds=8, start=.1):
    data = np.zeros(rate * seconds, dtype=np.float32)
    interval = 60 / bpm
    beats = []
    for time in np.arange(start, seconds - .15, interval):
        i = round(time * rate)
        beats.append(i)
        # Windowed narrow impulse with nonzero rise/decay and finite energy.
        width = 32
        data[i:i + width] += (.7 * np.hanning(width)).astype(np.float32)
    return make_buffer(data, rate), beats


@pytest.mark.parametrize("bpm", [80, 100, 120, 160, 200])
def test_synthetic_click_track_has_expected_tempo_candidate(bpm):
    audio, beats = metronome(bpm=bpm)
    settings = RhythmSettings(n_fft=512, hop_frames=64, min_bpm=60,
                              max_bpm=220, max_candidates=8)
    result = analyze_rhythm(audio, settings)
    assert isinstance(result, RhythmAnalysis)
    assert len(result.onsets) >= len(beats) - 2
    assert len(result.tempos) > 0
    assert any(abs(candidate.bpm - bpm) < 5 for candidate in result.tempos)
    assert result.envelope.max() == pytest.approx(1)
    assert all(0 <= v < audio.frames for v in result.beat_frames)
    assert tuple(sorted(result.beat_frames)) == result.beat_frames
    assert not result.envelope.flags.writeable
    for beat in beats:
        assert any(abs(event.frame - beat) <= 256 for event in result.onsets)


def test_tempo_estimator_known_periodic_novelty():
    settings = RhythmSettings(hop_frames=100, min_bpm=60., max_bpm=200.)
    rate = 8000
    n = 1000
    envelope = np.zeros(n, dtype=np.float64)
    envelope[20:900:40] = 1
    onsets = tuple(
        OnsetEvent(i * 100, i * 100 / rate, 1)
        for i in np.flatnonzero(envelope)
    )
    candidates = estimate_tempo(envelope, rate, onsets, settings)
    assert candidates
    assert candidates[0].bpm == pytest.approx(120, abs=1e-9)
    assert candidates[0].lag_frames == 40
    assert candidates[0].score > .95
    assert candidates[0].interbeat_frames == pytest.approx(4000)


def test_silence_returns_no_unfounded_tempo():
    audio = make_buffer(np.zeros(8000, dtype=np.float32), 8000)
    result = analyze_rhythm(audio)
    assert len(result.onsets) == 0
    assert result.tempos == ()
    assert result.beat_frames == ()
    np.testing.assert_array_equal(result.envelope, 0)
    assert result.as_dict()["tempo_is_hypothesis"] is True


def test_sparse_short_audio_does_not_invent_bpm():
    audio = make_buffer(np.array([1.] + [0.] * 199, dtype=np.float32), 8000)
    result = analyze_rhythm(audio)
    assert result.tempos == ()
    assert result.beat_frames == ()


def test_multichannel_analysis_respects_source_immutability():
    source, _ = metronome()
    stereo = make_buffer(np.stack([source.samples[0], source.samples[0] * .5]), 8000)
    original = stereo.samples.copy()
    a = analyze_rhythm(stereo)
    b = analyze_rhythm(stereo)
    np.testing.assert_array_equal(a.envelope, b.envelope)
    assert a.onsets == b.onsets
    assert a.tempos == b.tempos
    assert a.beat_frames == b.beat_frames
    np.testing.assert_array_equal(stereo.samples, original)


def test_onset_picker_resolves_plateau_at_earliest_hop():
    settings = RhythmSettings(hop_frames=64, min_strength=.1, prominence=.1)
    e = np.zeros(100, dtype=np.float64)
    e[20:23] = 1
    e[50] = .9
    peaks = pick_onsets(e, 8000, 8000, settings)
    assert [x.frame for x in peaks] == [20 * 64, 50 * 64]


def test_onset_picker_keeps_strongest_in_refractory_period():
    settings = RhythmSettings(hop_frames=64, min_spacing_seconds=.08,
                              min_strength=.1, prominence=.1)
    e = np.zeros(100, dtype=np.float64)
    e[20], e[25], e[50] = .3, 1., .8
    peaks = pick_onsets(e, 8000, 8000, settings)
    assert [x.frame for x in peaks] == [25 * 64, 50 * 64]


def test_hypothesis_report_distinguishes_observation_from_inference():
    audio, beats = metronome()
    report = analyze_rhythm(audio).as_dict()
    assert report["protocol"] == "sonumio-rhythm-hypotheses-v1"
    assert report["model_inference"] is False
    assert report["meter_inferred"] is False
    assert report["perceptual_quality_established"] is False
    assert report["source_frames"] == audio.frames
    assert isinstance(report["onsets"], list)
    assert isinstance(report["tempos"], list)


@pytest.mark.parametrize("bad", [
    dict(n_fft=63), dict(n_fft=4097), dict(n_fft=True),
    dict(hop_frames=3), dict(hop_frames=500),
    dict(smoothing_frames=2), dict(smoothing_frames=16),
    dict(neighborhood_frames=0), dict(neighborhood_frames=4),
    dict(min_spacing_seconds=0), dict(min_spacing_seconds=3),
    dict(min_strength=-1), dict(min_strength=1.2),
    dict(prominence=-1), dict(prominence=float("nan")),
    dict(min_bpm=0), dict(max_bpm=350), dict(min_bpm=160, max_bpm=120),
    dict(max_candidates=0), dict(max_candidates=20),
])
def test_invalid_settings_rejected(bad):
    with pytest.raises(AudioContractError):
        RhythmSettings(**bad)


def test_onset_input_shape_and_finite_validation():
    settings = RhythmSettings()
    for bad in (
        np.ones((1, 100)), np.ones(100, dtype=np.int32),
        np.array([0., float("nan"), 1]), np.array([1.1] * 100),
        np.array([-1.] * 100), np.zeros(2, dtype=np.float64),
    ):
        with pytest.raises(AudioContractError):
            pick_onsets(bad, 8000, 20000, settings)


def test_untrusted_tempo_inputs_rejected():
    settings = RhythmSettings()
    e = np.ones(100, dtype=np.float64)
    with pytest.raises(AudioContractError):
        estimate_tempo(e, 8000, [OnsetEvent(1, .1, 1)], settings)
    with pytest.raises(AudioContractError):
        estimate_tempo(np.full(100, np.nan), 8000, (), settings)


def test_rhythm_analysis_immutable_defensive_copy():
    audio, _ = metronome()
    r = analyze_rhythm(audio)
    with pytest.raises(ValueError):
        r.envelope[0] = 1
    with pytest.raises(ValueError):
        r.envelope.setflags(write=True)


def test_analysis_frame_limit_without_fft_allocation():
    # This raises before constructing the large spectrogram.
    settings = RhythmSettings(n_fft=64, hop_frames=16)
    audio = make_buffer(np.zeros(800_000, dtype=np.float32), 8000)
    with pytest.raises(AudioLimitError):
        novelty_envelope(audio, settings)


def test_rhythm_result_rejects_bad_envelope_and_metadata():
    settings = RhythmSettings()
    for bad in (
        np.array([np.nan], dtype=np.float64),
        np.ones((1, 2), dtype=np.float64),
        np.zeros(5, dtype=np.float32),
        np.array([-1.], dtype=np.float64),
    ):
        with pytest.raises(AudioContractError):
            RhythmAnalysis(bad, 8000, 100, settings, (), (), ())
