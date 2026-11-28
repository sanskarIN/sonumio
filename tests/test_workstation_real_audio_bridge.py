"""Integration tests against Sonumio's REAL finite-Fourier audio synthesizer.

These run in the complete source repository. They skip only when this
additive workstation patch is tested without the existing audio_core package.
"""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("sonumio.audio_core.wavetable")
from sonumio.audio_core.errors import AudioLimitError, AudioContractError
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.wavetable_bridge import (
    WavetableBridgeSettings, build_wavetable_plan, render_score_wavetable,
)


def score_with_notes(*notes, channel=0, end=960, title="Original integration"):
    score = Score.empty(end=end, title=title)
    track = ScoreTrack(
        "melody", "Melody", channel, 0, 0, 0,
        tuple(sorted(notes, key=lambda note: (
            note.start, note.pitch, note.duration, note.id,
        ))),
    )
    return replace(score, tracks=(track,))


def settings(**changes):
    args = dict(
        sample_rate=8000, channels=1, waveform="sine",
        attack_frames=0, decay_frames=0, sustain_level=1.,
        release_frames=0, master_gain=.12, max_harmonics=8,
    )
    args.update(changes)
    return WavetableBridgeSettings(**args)


def test_real_sine_bridge_tracks_exact_440_hz_pitch():
    original = score_with_notes(ScoreNote("a4", 0, 960, 69), end=960)
    before = original.sha256
    result = render_score_wavetable(original, settings())
    pcm = result.audio.samples
    assert pcm.shape == (1, 8000)
    assert result.audio.sample_rate == 8000
    assert result.report.rendered_notes == 1
    assert result.report.skipped_percussion_notes == 0
    assert result.report.output_frames == 8000
    assert not pcm.flags.writeable
    assert float(np.max(np.abs(pcm))) > .02
    fft = np.abs(np.fft.rfft(pcm[0].astype(np.float64)))
    frequency = np.fft.rfftfreq(8000, 1 / 8000)[int(np.argmax(fft))]
    assert frequency == pytest.approx(440., abs=1.)
    assert original.sha256 == before


def test_real_stereo_pan_remains_explicit_equal_power():
    original = score_with_notes(ScoreNote("left", 0, 480, 60, pan=0),
                                ScoreNote("right", 480, 480, 64, pan=127))
    result = render_score_wavetable(original, settings(channels=2))
    samples = result.audio.samples
    assert samples.shape == (2, 8000)
    assert float(np.max(np.abs(samples[0, :4000]))) > .01
    assert float(np.max(np.abs(samples[1, :4000]))) < 1e-8
    assert float(np.max(np.abs(samples[1, 4000:]))) > .01
    assert float(np.max(np.abs(samples[0, 4000:]))) < 1e-8


def test_real_waveforms_are_distinct_and_replay_bit_exact():
    score = score_with_notes(ScoreNote("note", 0, 960, 60))
    sine = render_score_wavetable(score, settings(waveform="sine"))
    saw = render_score_wavetable(score, settings(waveform="saw"))
    repeated = render_score_wavetable(score, settings(waveform="saw"))
    np.testing.assert_array_equal(saw.audio.samples, repeated.audio.samples)
    assert not np.array_equal(sine.audio.samples, saw.audio.samples)
    assert sine.audio.spec == saw.audio.spec
    assert saw.report.as_dict()["trained_model_inference"] is False


def test_real_muted_score_is_silence_until_explicitly_included():
    score = score_with_notes(ScoreNote("hidden", 0, 960, 60))
    muted = replace(score, tracks=(replace(score.tracks[0], muted=True),))
    empty = render_score_wavetable(muted, settings())
    assert empty.report.rendered_notes == 0
    np.testing.assert_array_equal(empty.audio.samples, 0)
    audible = render_score_wavetable(muted, settings(include_muted=True))
    assert audible.report.rendered_notes == 1
    assert float(np.max(np.abs(audible.audio.samples))) > .01


def test_real_percussion_notes_are_not_accidentally_synthesized_as_pitched():
    score = score_with_notes(ScoreNote("kick", 0, 240, 36), channel=9)
    rendered = render_score_wavetable(score, settings())
    assert rendered.report.rendered_notes == 0
    assert rendered.report.skipped_percussion_notes == 1
    np.testing.assert_array_equal(rendered.audio.samples, 0)


def test_real_synthesizer_enforces_polyphony_not_just_bridge_note_count():
    notes = tuple(ScoreNote(f"n{i:02}", 0, 480, 60 + (i % 12))
                  for i in range(65))
    score = score_with_notes(*notes)
    with pytest.raises(AudioLimitError):
        render_score_wavetable(score, settings(maximum_notes=128))


def test_real_synthesizer_rejects_incompatible_harmonic_table_bounds():
    score = score_with_notes(ScoreNote("note", 0, 480, 60))
    # The transport is valid, but max_harmonics exceeds half the table.
    with pytest.raises(AudioContractError):
        build_wavetable_plan(score, settings(table_size=256, max_harmonics=128))


def test_real_release_tail_has_declared_frame_count():
    score = score_with_notes(ScoreNote("note", 0, 480, 60))
    result = render_score_wavetable(score, settings(release_frames=80))
    assert result.audio.frames == 8080
    assert result.report.output_frames == 8080
    assert result.report.output_peak == pytest.approx(
        float(np.max(np.abs(result.audio.samples))), abs=1e-8,
    )
