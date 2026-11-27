"""Deterministic analytic and adversarial duration/pitch transform coverage."""
import math
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer, AudioLayout
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.time_scale import (
    TimeScaleSettings, stretch_audio, target_frames_for_ratio, _correlation_scores,
)
from sonumio.audio_core.pitch_shift import PitchShiftSettings, pitch_shift


def tone(frames=12000, rate=24000, frequency=440.):
    t = np.arange(frames, dtype=np.float64) / rate
    return (.35 * np.sin(2*np.pi*frequency*t)).astype(np.float32)


def dominant_frequency(samples, rate):
    n = len(samples)
    power = np.abs(np.fft.rfft(samples * np.hanning(n)))
    return np.argmax(power[1:]) * rate / n + rate / n


@pytest.mark.parametrize("method", ["wsola", "phase_vocoder"])
@pytest.mark.parametrize("ratio", [.5, .75, 1., 1.25, 1.5, 2.])
def test_exact_geometry_and_pitch(method, ratio):
    source = make_buffer(tone(), 24000)
    before = source.samples.tobytes()
    cfg = TimeScaleSettings(method=method, window_frames=1024, search_frames=192)
    target = target_frames_for_ratio(source.frames, ratio)
    result = stretch_audio(source, target, settings=cfg)
    assert result.audio.frames == target
    assert result.audio.sample_rate == source.sample_rate
    assert result.audio.spec.layout == source.spec.layout
    assert source.samples.tobytes() == before
    assert np.isfinite(result.audio.samples).all()
    middle = result.audio.samples[0, 1024:-1024]
    assert abs(dominant_frequency(middle, 24000) - 440) < 12
    assert .08 < np.sqrt(np.mean(middle**2)) < .5
    assert result.report.identity == (ratio == 1)
    if ratio == 1:
        np.testing.assert_array_equal(result.audio.samples, source.samples)
    assert result.report.as_dict()["model_inference"] is False


@pytest.mark.parametrize("method", ["wsola", "phase_vocoder"])
@pytest.mark.parametrize("ratio", [.5, 1.5, 2.])
def test_stereo_antiphase_and_layout(method, ratio):
    wave = tone(8001)
    layout = AudioLayout(2, "stereo", ("left custom", "right custom"))
    source = make_buffer(np.stack((wave, -wave)), 24000, layout=layout)
    settings = TimeScaleSettings(method=method, window_frames=512, search_frames=96)
    n = target_frames_for_ratio(source.frames, ratio)
    result = stretch_audio(source, n, settings=settings)
    assert result.audio.spec.layout == layout
    np.testing.assert_allclose(result.audio.samples[0], -result.audio.samples[1],
                               atol=1e-6, rtol=0)
    again = stretch_audio(source, n, settings=settings)
    np.testing.assert_array_equal(result.audio.samples, again.audio.samples)
    assert result.report == again.report


@pytest.mark.parametrize("method", ["wsola", "phase_vocoder"])
@pytest.mark.parametrize("frames", [1, 2, 17, 63, 64, 65, 127, 129])
def test_short_silence_exact_output(method, frames):
    cfg = TimeScaleSettings(method=method, window_frames=64, search_frames=16)
    source = make_buffer(np.zeros(frames, dtype=np.float32), 8000)
    for target in {frames, frames*2, (frames+1)//2}:
        output = stretch_audio(source, target, settings=cfg)
        assert output.audio.frames == target
        assert np.count_nonzero(output.audio.samples) == 0
        assert output.report.output_peak == 0


@pytest.mark.parametrize("method", ["wsola", "phase_vocoder"])
def test_explicit_peak_ceiling_only_attenuates(method):
    cfg = TimeScaleSettings(method=method, window_frames=256,
                            search_frames=32, peak_ceiling=.1)
    source = make_buffer(tone(5000), 24000)
    result = stretch_audio(source, 7500, settings=cfg)
    assert result.report.attenuation < 1
    assert result.report.output_peak <= .10000001
    silent = make_buffer(np.zeros(5000, dtype=np.float32), 24000)
    assert stretch_audio(silent, 7500, settings=cfg).report.attenuation == 1


@pytest.mark.parametrize("kwargs", [
    {"method": "invalid"}, {"window_frames": 65}, {"window_frames": True},
    {"search_frames": -1}, {"search_frames": True},
    {"window_frames": 64, "search_frames": 33},
    {"fft_hop": 0}, {"fft_hop": 257}, {"peak_ceiling": float("nan")},
    {"peak_ceiling": True}, {"peak_ceiling": 1.1},
])
def test_invalid_settings(kwargs):
    with pytest.raises(AudioContractError):
        TimeScaleSettings(**kwargs)


@pytest.mark.parametrize("target", [True, 1., 0, -1, 99, 401])
def test_invalid_targets(target):
    source = make_buffer(tone(200), 24000)
    with pytest.raises(AudioContractError):
        stretch_audio(source, target)


def test_preflight_work_bound(monkeypatch):
    import sonumio.audio_core.time_scale as module
    source = make_buffer(tone(1024), 24000)
    monkeypatch.setattr(module, "MAX_WORKING_BYTES", 1)
    with pytest.raises(AudioLimitError, match="working memory"):
        stretch_audio(source, 1536)


@pytest.mark.parametrize("method, constant", [
    ("wsola", "MAX_CORRELATION_PRODUCTS"),
    ("phase_vocoder", "MAX_FFT_POINTS"),
])
def test_algorithm_work_budget(monkeypatch, method, constant):
    import sonumio.audio_core.time_scale as module
    monkeypatch.setattr(module, constant, 1)
    source = make_buffer(tone(2048), 24000)
    with pytest.raises(AudioLimitError, match="work exceeds"):
        stretch_audio(source, 3072, settings=TimeScaleSettings(method=method))


def test_correlation_matches_centered_direct_reference():
    rng = np.random.default_rng(91)
    reference = rng.normal(size=(2, 23))
    candidates = rng.normal(size=(2, 42))
    scores = _correlation_scores(reference, candidates)
    ref = reference - reference.mean(axis=1, keepdims=True)
    expected = []
    for offset in range(20):
        candidate = candidates[:, offset:offset+23].copy()
        candidate -= candidate.mean(axis=1, keepdims=True)
        expected.append(np.sum(ref*candidate) /
                        np.sqrt(np.sum(ref*ref)*np.sum(candidate*candidate)))
    np.testing.assert_allclose(scores, expected, rtol=1e-12, atol=1e-12)
    assert _correlation_scores(np.zeros_like(reference), candidates) is None


@pytest.mark.parametrize("semitones", [-12, -7, 0, 5, 12])
def test_pitch_shift_exact_duration_and_frequency(semitones):
    source = make_buffer(tone(12000), 24000)
    cfg = PitchShiftSettings(semitones,
        time_scale=TimeScaleSettings(method="phase_vocoder", window_frames=1024))
    output = pitch_shift(source, cfg)
    expected = 440 * 2**(semitones/12)
    assert output.audio.frames == source.frames
    assert output.audio.sample_rate == source.sample_rate
    middle = output.audio.samples[0, 2048:-2048]
    assert abs(dominant_frequency(middle, 24000) - expected) < 14
    assert abs(output.report.effective_semitones - semitones) < .002
    assert abs(output.report.boundary_correction_frames) <= 1
    assert not output.report.as_dict()["formants_preserved"]
    if semitones == 0:
        np.testing.assert_array_equal(output.audio.samples, source.samples)


@pytest.mark.parametrize("frames", [5, 17, 333, 1001, 2049])
@pytest.mark.parametrize("semitones", [-12, -5, 7, 12])
def test_pitch_odd_frame_geometry(frames, semitones):
    source = make_buffer(np.zeros(frames, dtype=np.float32), 24000)
    cfg = PitchShiftSettings(semitones,
        time_scale=TimeScaleSettings(method="phase_vocoder", window_frames=64, search_frames=16))
    result = pitch_shift(source, cfg)
    assert result.audio.frames == frames
    assert not np.any(result.audio.samples)


def test_pitch_virtual_clock_guard():
    source = make_buffer(tone(800, rate=8000), 8000)
    with pytest.raises(AudioContractError, match="sample_rate"):
        pitch_shift(source, PitchShiftSettings(-12))


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), -13, 13])
def test_bad_pitch_values(value):
    with pytest.raises(AudioContractError):
        PitchShiftSettings(value)
