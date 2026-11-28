"""Pitch-class extraction and transparent triad hypothesis invariants."""
import math

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.stft import StftConfig, stft
from sonumio.audio_core.harmonic_chroma import (
    NAMES, ChromaSettings, ChordHypothesis, TonalAnalysis,
    chroma_from_spectrum, score_triad, chord_templates, analyze_chroma,
)


def tone(hz, frames=16000, rate=16000, gain=.2):
    n = np.arange(frames) / rate
    return gain * np.sin(2 * np.pi * hz * n)


def chord(notes, frames=16000, rate=16000):
    values = np.zeros(frames, dtype=np.float64)
    for midi in notes:
        hz = 440 * 2 ** ((midi - 69) / 12)
        values += tone(hz, frames, rate, .15)
    return make_buffer(values.astype(np.float32), rate)


@pytest.mark.parametrize("notes,symbol", [
    ((60, 64, 67), "C"),
    ((57, 60, 64), "Am"),
    ((55, 59, 62), "G"),
    ((62, 65, 69), "Dm"),
    ((65, 69, 72), "F"),
    ((64, 68, 71), "E"),
])
def test_synthetic_major_minor_triads_have_expected_chord(notes, symbol):
    audio = chord(notes)
    options = ChromaSettings(
        n_fft=4096, hop_frames=512, minimum_segment_frames=2,
        minimum_score_gap=.005,
    )
    result = analyze_chroma(audio, options)
    assert isinstance(result, TonalAnalysis)
    values = [s.symbol for s in result.segments]
    assert symbol in values
    center = result.chroma[:, result.chroma.shape[1] // 2]
    assert np.sum(center) == pytest.approx(1., abs=1e-8)
    assert result.as_dict()["chord_is_hypothesis"] is True


def test_opposite_phase_stereo_preserves_tonal_evidence():
    audio = chord((60, 64, 67))
    stereo = make_buffer(np.stack([audio.samples[0], -audio.samples[0]]), 16000)
    settings = ChromaSettings(n_fft=4096, minimum_score_gap=.005)
    mono = analyze_chroma(audio, settings)
    doubled = analyze_chroma(stereo, settings)
    np.testing.assert_allclose(mono.chroma, doubled.chroma, atol=1e-10)
    assert mono.segments == doubled.segments


def test_silence_is_not_classified_as_chord():
    audio = make_buffer(np.zeros(16000, dtype=np.float32), 16000)
    result = analyze_chroma(audio)
    assert result.segments == ()
    assert all(x is None for x in result.hypotheses)
    np.testing.assert_array_equal(result.chroma, 0)
    assert result.as_dict()["key_signature_inferred"] is False


def test_one_note_cannot_pass_full_triad_membership_rule():
    vector = np.zeros(12, dtype=np.float64)
    vector[0] = 1
    assert score_triad(vector) is None
    audio = make_buffer(tone(261.625565, gain=.2).astype(np.float32), 16000)
    result = analyze_chroma(audio)
    # This detector must not label single clean sine waves as full triads.
    assert not result.segments


def test_exact_major_minor_templates_and_pitch_class_names():
    templates = chord_templates()
    assert len(templates) == 24
    assert len(set((a, b) for a, b, _ in templates)) == 24
    assert NAMES[0] == "C"
    assert NAMES[9] == "A"
    for root, quality, mask in templates:
        assert np.sum(mask) == 3
        assert np.all((mask == 0) | (mask == 1))
        h = score_triad(mask.astype(np.float64))
        assert isinstance(h, ChordHypothesis)
        assert h.root == root
        assert h.quality == quality
        assert h.score == pytest.approx(1)
        assert h.symbol == root + ("m" if quality == "minor" else "")


def test_chroma_frame_normalization_and_readonly_storage():
    audio = chord((60, 64, 67), frames=12000)
    options = ChromaSettings(n_fft=2048, hop_frames=256)
    result = analyze_chroma(audio, options)
    assert result.chroma.shape[0] == 12
    assert result.chroma.shape[1] == len(result.hypotheses)
    assert np.isfinite(result.chroma).all()
    assert np.all(result.chroma >= 0)
    assert np.all(result.chroma <= 1)
    assert not result.chroma.flags.writeable
    with pytest.raises(ValueError):
        result.chroma.setflags(write=True)
    with pytest.raises(ValueError):
        result.chroma[0, 0] = 0


def test_report_is_json_compatible_and_no_model_claim():
    import json
    audio = chord((60, 64, 67))
    result = analyze_chroma(audio)
    value = result.as_dict()
    json.dumps(value, sort_keys=True, allow_nan=False)
    assert value["protocol"] == "sonumio-chroma-triad-hypotheses-v1"
    assert value["source_frames"] == audio.frames
    assert value["sample_rate"] == audio.sample_rate
    assert value["model_inference"] is False
    assert value["perceptual_quality_established"] is False
    assert len(value["overall_chroma"]) == 12
    assert len(value["chord_hypotheses"]) == result.chroma.shape[1]


def test_independent_call_results_and_no_source_mutation():
    audio = chord((57, 60, 64), frames=10000)
    original = audio.samples.copy()
    a = analyze_chroma(audio)
    b = analyze_chroma(audio)
    np.testing.assert_array_equal(a.chroma, b.chroma)
    assert a.segments == b.segments
    assert a.hypotheses == b.hypotheses
    np.testing.assert_array_equal(audio.samples, original)


def test_spectral_chroma_fails_if_frequency_cap_above_nyquist():
    audio = chord((60, 64, 67), frames=4000, rate=16000)
    spectrum = stft(audio, StftConfig(n_fft=512, hop_length=128))
    with pytest.raises(AudioContractError):
        chroma_from_spectrum(spectrum, ChromaSettings(highest_hz=9000.))


def test_triads_on_explicit_vectors_are_ambiguity_aware():
    ambiguous = np.ones(12, dtype=np.float64)
    assert score_triad(ambiguous) is None
    weak = np.zeros(12, dtype=np.float64)
    weak[0], weak[4], weak[7] = .95, .03, .02
    assert score_triad(weak) is None


@pytest.mark.parametrize("invalid", [
    dict(n_fft=255), dict(n_fft=8193), dict(n_fft=True),
    dict(hop_frames=1), dict(hop_frames=2000, n_fft=2048),
    dict(lowest_hz=0), dict(lowest_hz=3000, highest_hz=1000),
    dict(highest_hz=100000), dict(reference_a_hz=390),
    dict(reference_a_hz=float("nan")), dict(magnitude_power=3),
    dict(minimum_rms=-1), dict(minimum_rms=2),
    dict(minimum_triad_energy=.5), dict(minimum_chord_score=.1),
    dict(minimum_score_gap=.8), dict(minimum_segment_frames=0),
])
def test_invalid_options_rejected(invalid):
    with pytest.raises(AudioContractError):
        ChromaSettings(**invalid)


@pytest.mark.parametrize("vector", [
    np.zeros(11), np.zeros((12, 1)),
    np.array([1.] * 11 + [np.nan]), np.array([-1.] * 12),
    np.ones(12, dtype=np.int32),
])
def test_invalid_triad_scoring_vector(vector):
    with pytest.raises(AudioContractError):
        score_triad(vector)


def test_very_long_analysis_work_is_rejected_before_fft():
    audio = make_buffer(np.zeros(960000, dtype=np.float32), 8000)
    options = ChromaSettings(n_fft=256, hop_frames=32)
    with pytest.raises(AudioLimitError):
        analyze_chroma(audio, options)
