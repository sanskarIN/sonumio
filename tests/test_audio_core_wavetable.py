"""Original finite-Fourier wavetable synthesis: clock, spectra and limits."""
import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.wavetable import (
    VoiceEnvelope, WavetablePlan, WavetableNote,
    build_wavetable, permitted_harmonics,
    render_wavetable, _envelope, _polyphony,
)


def plan(waveform="saw", frames=8000, channels=1, rate=8000, **kwargs):
    return WavetablePlan(
        sample_rate=rate, frames=frames, channels=channels,
        waveform=waveform, **kwargs,
    )


def note(midi=69, frames=4000, start=0, **kwargs):
    return WavetableNote(start, frames, midi, **kwargs)


@pytest.mark.parametrize("waveform", ["sine", "saw", "square", "triangle"])
@pytest.mark.parametrize("harmonics", [1, 2, 11, 64, 128])
def test_tables_are_finite_periodic_and_immutable(waveform, harmonics):
    table = build_wavetable(waveform, 2048, harmonics)
    assert table.shape == (2048,)
    assert table.dtype == np.float64
    assert np.isfinite(table).all()
    assert np.max(np.abs(table)) == pytest.approx(1, abs=1e-12)
    assert abs(float(np.mean(table))) < 1e-12
    assert not table.flags.writeable
    with pytest.raises(ValueError):
        table.setflags(write=True)


def test_note_fundamental_matches_midi_a440():
    settings = plan("sine", 8000, envelope=VoiceEnvelope(
        attack_frames=0, decay_frames=0, sustain_level=1., release_frames=0,
    ))
    result = render_wavetable(settings, (note(69, 8000),))
    spectrum = np.abs(np.fft.rfft(result.audio.samples[0]))
    peak_index = int(np.argmax(spectrum))
    frequency = np.fft.rfftfreq(8000, 1 / 8000)[peak_index]
    assert frequency == pytest.approx(440, abs=1)
    assert result.report.stats[0].harmonics == 1
    assert result.report.notes == 1
    assert result.report.distinct_tables == 1
    assert not result.audio.samples.flags.writeable


@pytest.mark.parametrize("midi", [12, 36, 48, 60, 69, 81, 100])
def test_harmonic_count_is_below_sample_nyquist(midi):
    settings = plan("saw", rate=24000, frames=24000)
    selected = permitted_harmonics(settings, note(midi))
    assert selected >= 1
    assert selected <= settings.max_harmonics
    assert selected * note(midi).frequency_hz < settings.sample_rate / 2


def test_high_frequency_outside_nyquist_rejected():
    settings = plan("saw", rate=8000, frames=8000)
    with pytest.raises(AudioContractError):
        render_wavetable(settings, (note(127),))


@pytest.mark.parametrize("waveform", ["sine", "saw", "square", "triangle"])
def test_synthesizer_repeatability_for_three_notes(waveform):
    settings = plan(waveform, frames=8000, channels=2)
    notes = (
        note(60, 2800, 100, velocity=.6, pan=-.5),
        note(64, 2800, 300, velocity=.7, pan=0),
        note(67, 2800, 600, velocity=.8, pan=.5),
    )
    first = render_wavetable(settings, notes)
    second = render_wavetable(settings, notes)
    np.testing.assert_array_equal(first.audio.samples, second.audio.samples)
    assert first.report == second.report
    assert first.report.maximum_polyphony == 3
    assert first.audio.frames == 8000
    assert first.audio.channels == 2
    assert first.report.as_dict()["model_inference"] is False


def test_equal_power_panning_endpoints():
    settings = plan("sine", frames=8000, channels=2,
                    envelope=VoiceEnvelope(0, 0, 1., 0))
    left = render_wavetable(settings, (note(69, 1000, pan=-1.),)).audio.samples
    right = render_wavetable(settings, (note(69, 1000, pan=1.),)).audio.samples
    assert np.max(np.abs(left[0])) > .01
    assert np.max(np.abs(left[1])) < 1e-10
    assert np.max(np.abs(right[1])) > .01
    assert np.max(np.abs(right[0])) < 1e-10


def test_zero_gain_and_empty_notes_produce_silence():
    settings = plan("triangle", frames=1500)
    silent = render_wavetable(settings, ())
    assert silent.report.maximum_polyphony == 0
    assert silent.report.notes == 0
    assert silent.report.distinct_tables == 0
    np.testing.assert_array_equal(silent.audio.samples, 0)
    zero = render_wavetable(
        plan("triangle", frames=1500, master_gain=0.),
        (note(60, 500),),
    )
    np.testing.assert_array_equal(zero.audio.samples, 0)


def test_envelope_release_ends_at_zero_without_click():
    options = VoiceEnvelope(
        attack_frames=12, decay_frames=16,
        sustain_level=.5, release_frames=20,
    )
    values = _envelope(200 + options.release_frames, 200, options)
    assert values.shape == (220,)
    assert values[0] > 0
    assert values[11] == pytest.approx(1)
    assert values[27] == pytest.approx(.5)
    assert values[199] == pytest.approx(.5)
    assert values[-1] == pytest.approx(0)
    assert np.all((values >= 0) & (values <= 1))


def test_short_note_releases_from_actual_gate_level():
    options = VoiceEnvelope(100, 200, .3, 20)
    values = _envelope(30, 10, options)
    assert values[9] == pytest.approx(.1)
    assert values[10] < values[9]
    assert values[-1] == 0


def test_polyphony_overlap_and_touching_notes():
    notes = (note(60, 10, 0), note(61, 10, 10), note(62, 10, 20))
    assert _polyphony(notes, 0) == 1
    assert _polyphony(notes, 10) == 2


@pytest.mark.parametrize("bad", [
    dict(waveform="noise"), dict(channels=3),
    dict(table_size=1000), dict(table_size=32),
    dict(max_harmonics=400), dict(master_gain=1.1),
    dict(master_gain=float("nan")), dict(envelope=None),
])
def test_invalid_plan_rejected(bad):
    args = dict(sample_rate=8000, frames=8000)
    args.update(bad)
    with pytest.raises(AudioContractError):
        WavetablePlan(**args)


@pytest.mark.parametrize("bad", [
    dict(start_frame=-1), dict(gate_frames=0),
    dict(midi_note=-1), dict(midi_note=128),
    dict(velocity=-1), dict(velocity=2),
    dict(pan=-2), dict(pan=2),
    dict(initial_phase=float("nan")),
])
def test_invalid_note_rejected(bad):
    values = dict(start_frame=0, gate_frames=100, midi_note=69)
    values.update(bad)
    with pytest.raises(AudioContractError):
        WavetableNote(**values)


def test_note_release_must_fit_requested_frame_count():
    settings = plan(frames=1000)
    with pytest.raises(AudioContractError):
        render_wavetable(settings, (note(69, 600, 400),))


def test_more_than_max_polyphony_rejected():
    settings = plan(frames=4000, envelope=VoiceEnvelope(0, 0, 1., 0))
    notes = tuple(note(69, 1000) for _ in range(65))
    with pytest.raises(AudioLimitError):
        render_wavetable(settings, notes)


def test_mono_pan_requires_center():
    settings = plan(frames=4000, channels=1)
    with pytest.raises(AudioContractError):
        render_wavetable(settings, (note(69, 1000, pan=.5),))


def test_table_constructor_rejects_aliases_and_invalid_geometry():
    for args in (("neural", 2048, 4), ("saw", 1000, 4),
                 ("saw", 2048, 0), ("saw", 256, 256)):
        with pytest.raises(AudioContractError):
            build_wavetable(*args)


def test_synthesized_report_does_not_claim_clipping_or_alias_freedom():
    settings = plan(frames=4000)
    result = render_wavetable(settings, (note(69, 1000),))
    report = result.report.as_dict()
    assert report["harmonic_alias_free_guaranteed"] is False
    assert report["audio_clipped_or_normalized"] is False
    assert report["original_mathematical_waveforms"] is True
