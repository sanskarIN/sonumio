"""Original procedural, non-neural signal synthesis and deterministic scoring."""
from __future__ import annotations

import math
import numpy as np
import pytest

from sonumio.audio_core.synthesis import (
    NOISE_COLORS, WAVEFORMS, SynthNote,
    deterministic_noise, midi_to_frequency, oscillator,
    render_synth_score, synth_score_report,
)
from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError


def test_middle_a_frequency_and_octave_intervals():
    assert midi_to_frequency(69) == pytest.approx(440)
    assert midi_to_frequency(81) == pytest.approx(880)
    assert midi_to_frequency(57) == pytest.approx(220)
    assert midi_to_frequency(69, concert_a=442) == pytest.approx(442)
    for bad in (True, -1, 128, 69.0):
        with pytest.raises(AudioContractError):
            midi_to_frequency(bad)


@pytest.mark.parametrize("waveform", WAVEFORMS)
def test_short_procedural_oscillators_are_finite_and_repeatable(waveform):
    a = oscillator(440, 24000, 2048, waveform=waveform)
    b = oscillator(440, 24000, 2048, waveform=waveform)
    assert a.shape == (2048,)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    assert np.max(np.abs(a)) <= 1 + 1e-6


def test_sine_oscillator_obeys_exact_zero_phase_and_expected_pitch():
    wave = oscillator(440, 24000, 24000, waveform="sine")
    assert wave[0] == pytest.approx(0)
    assert np.max(np.abs(wave)) > .99
    crossings = np.sum((wave[:-1] <= 0) & (wave[1:] > 0))
    assert 438 <= crossings <= 442


def test_frequency_automation_requires_frame_aligned_finite_float():
    values = np.linspace(200, 800, 2000, dtype=np.float32)
    tone = oscillator(values, 24000, len(values))
    assert tone.shape == values.shape
    with pytest.raises(AudioContractError):
        oscillator(values[:100], 24000, 2000)
    with pytest.raises(AudioContractError):
        oscillator(np.full(2000, np.nan, dtype=np.float32), 24000, 2000)
    with pytest.raises(AudioContractError):
        oscillator(24000, 24000, 2000)


@pytest.mark.parametrize("color", NOISE_COLORS)
def test_deterministic_seeded_noise_is_reproducible(color):
    a = deterministic_noise(1000, 2, seed=42, color=color)
    b = deterministic_noise(1000, 2, seed=42, color=color)
    assert a.shape == (2, 1000)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32
    assert np.max(np.abs(a)) <= 1.0 + 1e-6
    assert np.isfinite(a).all()


def test_independent_noise_seeds_change_bytes():
    a = deterministic_noise(1000, 1, seed=1)
    b = deterministic_noise(1000, 1, seed=2)
    assert not np.array_equal(a, b)


def test_invalid_noise_rng_input_and_waveforms_refused():
    for bad in (-1, True, 2**32):
        with pytest.raises(AudioContractError):
            deterministic_noise(100, 1, seed=bad)
    with pytest.raises(AudioContractError):
        deterministic_noise(100, 9, seed=1)
    with pytest.raises(AudioContractError):
        deterministic_noise(100, seed=1, color="script")
    with pytest.raises(AudioContractError):
        oscillator(440, 24000, 100, waveform="malformed")


def test_polyphonic_score_preserves_audio_clock_and_note_identity():
    notes = (
        SynthNote(60, 0, 10000, attack_frames=100,
                  decay_frames=200, release_frames=1000, pan=-1),
        SynthNote(64, 5000, 10000, attack_frames=100,
                  decay_frames=200, release_frames=1000, pan=1),
    )
    score = render_synth_score(notes, frames=20000, channels=2,
                               master_gain=.3)
    assert score.channels == 2
    assert score.frames == 20000
    assert score.sample_rate == 24000
    assert np.isfinite(score.samples).all()
    assert np.max(np.abs(score.samples)) > 0
    np.testing.assert_array_equal(
        render_synth_score(notes, frames=20000, channels=2,
                           master_gain=.3).samples,
        score.samples,
    )
    report = synth_score_report(notes, score)
    assert report["events"] == 2
    assert report["third_party_model_weights"] is False
    assert report["audio_files_saved"] is False
    assert report["generator_type"].endswith("not learned model")


def test_score_rejects_overlapping_polyphony_and_nyquist_pitch():
    events = tuple(SynthNote(60, 0, 1000, attack_frames=0,
                             decay_frames=0, release_frames=0)
                   for _ in range(3))
    with pytest.raises(AudioLimitError):
        render_synth_score(events, frames=2000, max_polyphony=2)
    with pytest.raises(AudioContractError):
        render_synth_score(events, sample_rate=8000, frames=500,
                           max_polyphony=4)


def test_score_validation_enforces_adsr_duration_and_bounded_coordinates():
    with pytest.raises(AudioContractError):
        SynthNote(60, 0, 100, attack_frames=50,
                  decay_frames=50, release_frames=50)
    with pytest.raises(AudioContractError):
        SynthNote(60, -1, 1000)
    with pytest.raises(AudioContractError):
        SynthNote(60, 0, 1000, pan=2)
    with pytest.raises(AudioContractError):
        render_synth_score((SynthNote(60, 10, 10000),),
                           frames=5000)


def test_reference_synthesis_is_not_original_sonumio_model():
    from sonumio.registry import list_models
    entries = list_models()
    assert all(not x["original_sonumio_model"] for x in entries)
