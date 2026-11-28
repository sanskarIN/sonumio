"""Damped delay-line string synthesis: deterministic sound and strict limits."""
import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.plucked_string import (
    StringVoice, PluckNote, PluckedStringPlan, render_string_voice,
    render_plucked_strings, loop_length,
)


def plan(*, frames=24000, channels=1, **kwargs):
    return PluckedStringPlan(
        sample_rate=24000, frames=frames, channels=channels,
        **kwargs,
    )


def note(*, midi=69, frames=12000, start=0, seed=1234, **kwargs):
    return PluckNote(start, frames, midi, seed=seed, **kwargs)


def test_original_string_render_is_deterministic():
    p = plan(frames=24000, channels=2)
    notes = (
        note(midi=60, start=0, frames=8000, pan=-.5),
        note(midi=64, start=4000, frames=7000, pan=.5),
        note(midi=67, start=6000, frames=6000),
    )
    a = render_plucked_strings(p, notes)
    b = render_plucked_strings(p, notes)
    np.testing.assert_array_equal(a.audio.samples, b.audio.samples)
    assert a.report == b.report
    assert a.report.maximum_polyphony == 3
    assert a.audio.frames == p.frames
    assert a.audio.channels == 2
    assert not a.audio.samples.flags.writeable
    assert a.report.as_dict()["model_inference"] is False
    assert a.report.as_dict()["pitch_is_exact"] is False


@pytest.mark.parametrize("pitch", [48, 57, 60, 69, 72, 81])
def test_string_loop_frequency_is_close_to_midi_reference(pitch):
    p = plan()
    n = note(midi=pitch)
    delay = loop_length(p, n)
    approx = p.sample_rate / (delay + .25)
    assert abs(approx / n.frequency_hz - 1) < .03
    assert delay >= 4


@pytest.mark.parametrize("brightness", [0., .25, .5, .75, 1.])
def test_feedback_loop_preserves_finite_energy(brightness):
    voice = StringVoice(brightness=brightness, damping=.99)
    signal, frequency, length = render_string_voice(note(), 24000, voice)
    assert signal.ndim == 1
    assert len(signal) == note().duration_frames + voice.release_frames
    assert np.isfinite(signal).all()
    assert np.max(np.abs(signal)) <= 1.0001
    assert signal[-1] == pytest.approx(0, abs=1e-12)
    assert frequency > 0
    assert length >= 4


def test_pluck_excitation_seed_changes_waveform():
    p = plan(frames=24000)
    a = render_plucked_strings(p, (note(seed=1),)).audio.samples
    b = render_plucked_strings(p, (note(seed=2),)).audio.samples
    assert not np.array_equal(a, b)
    c = render_plucked_strings(p, (note(seed=1),)).audio.samples
    np.testing.assert_array_equal(a, c)


def test_note_start_and_release_are_zero_outside_interval():
    p = plan(frames=20000)
    n = note(start=400, frames=2000)
    result = render_plucked_strings(p, (n,))
    signal = result.audio.samples[0]
    np.testing.assert_array_equal(signal[:400], 0)
    np.testing.assert_array_equal(signal[400 + 2000 + p.voice.release_frames:], 0)
    assert signal[400 + 2000 + p.voice.release_frames - 1] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("pan,dominant", [(-1., 0), (1., 1)])
def test_stereo_panning_respects_endpoints(pan, dominant):
    p = plan(channels=2)
    signal = render_plucked_strings(p, (note(pan=pan),)).audio.samples
    assert np.max(np.abs(signal[dominant])) > .001
    assert np.max(np.abs(signal[1 - dominant])) <= 1e-9


def test_zero_velocity_and_empty_score_are_silent():
    p = plan()
    empty = render_plucked_strings(p, ())
    np.testing.assert_array_equal(empty.audio.samples, 0)
    assert empty.report.maximum_polyphony == 0
    silent = render_plucked_strings(p, (note(velocity=0),))
    np.testing.assert_array_equal(silent.audio.samples, 0)
    assert silent.report.notes == 1


def test_wider_pick_and_brightness_change_spectrum():
    p = plan()
    n = note()
    a = render_plucked_strings(
        PluckedStringPlan(p.sample_rate, p.frames, 1, .2,
                          StringVoice(brightness=.2, pluck_position=.15)), (n,),
    ).audio.samples
    b = render_plucked_strings(
        PluckedStringPlan(p.sample_rate, p.frames, 1, .2,
                          StringVoice(brightness=.9, pluck_position=.35)), (n,),
    ).audio.samples
    assert not np.array_equal(a, b)


def test_attacks_and_releases_have_explicit_length():
    voice = StringVoice(attack_frames=20, release_frames=40)
    signal, _, _ = render_string_voice(note(frames=120), 24000, voice)
    assert len(signal) == 160
    assert signal[-1] == 0
    assert abs(signal[0]) <= 1.0
    assert np.max(np.abs(signal)) <= 1.0001


def test_polyphony_limit_and_voice_overlap():
    p = plan(frames=12000)
    notes = tuple(note(start=0, frames=3000, seed=i) for i in range(33))
    with pytest.raises(AudioLimitError):
        render_plucked_strings(p, notes)


def test_sequential_notes_do_not_count_as_overlapping():
    p = plan(frames=24000, voice=StringVoice(release_frames=0))
    notes = (
        note(start=0, frames=4000),
        note(start=4000, frames=4000),
        note(start=8000, frames=4000),
    )
    result = render_plucked_strings(p, notes)
    assert result.report.maximum_polyphony == 1


def test_note_end_cannot_exceed_output_clock():
    p = plan(frames=1000)
    with pytest.raises(AudioContractError):
        render_plucked_strings(p, (note(start=500, frames=500),))


def test_mono_panning_rejected():
    p = plan(channels=1)
    with pytest.raises(AudioContractError):
        render_plucked_strings(p, (note(pan=-.25),))


@pytest.mark.parametrize("bad", [
    dict(damping=.1), dict(damping=1.),
    dict(damping=float("nan")), dict(brightness=-1),
    dict(brightness=1.5), dict(pluck_position=.001),
    dict(pluck_position=.9), dict(attack_frames=-1),
    dict(release_frames=-1), dict(excitation_smoothing=0),
    dict(excitation_smoothing=16),
])
def test_invalid_voice_rejected(bad):
    with pytest.raises(AudioContractError):
        StringVoice(**bad)


@pytest.mark.parametrize("bad", [
    dict(start_frame=-1), dict(duration_frames=0),
    dict(midi_note=128), dict(midi_note=-1),
    dict(velocity=-1), dict(velocity=2),
    dict(pan=-1.1), dict(pan=1.1),
    dict(seed=-1), dict(seed=2**32), dict(seed=True),
])
def test_invalid_note_rejected(bad):
    args = dict(start_frame=0, duration_frames=1200, midi_note=69)
    args.update(bad)
    with pytest.raises(AudioContractError):
        PluckNote(**args)


@pytest.mark.parametrize("bad", [
    dict(channels=0), dict(channels=3),
    dict(frames=0), dict(gain=-1), dict(gain=1.2),
    dict(gain=float("inf")), dict(voice="bad"),
])
def test_invalid_synthesis_plan_rejected(bad):
    args = dict(sample_rate=24000, frames=24000)
    args.update(bad)
    with pytest.raises(AudioContractError):
        PluckedStringPlan(**args)


def test_high_note_is_beyond_reference_string_loop_limit():
    p = plan(frames=10000, channels=1)
    with pytest.raises(AudioContractError):
        render_plucked_strings(p, (note(midi=127, frames=1000),))


def test_data_report_contains_physical_model_limitations():
    result = render_plucked_strings(plan(), (note(),))
    report = result.report.as_dict()
    assert report["protocol"] == "sonumio-karplus-strong-reference-v1"
    assert report["model_inference"] is False
    assert report["physical_instrument_accuracy_claimed"] is False
    assert report["automatic_normalization"] is False
    assert report["notes"] == 1
    assert len(report["stats"]) == 1
