"""Original pattern generator reproducibility, meter, topology and safety."""
from dataclasses import replace
import json

import pytest

from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.patterns import (
    PatternCell, PatternVoice, PatternPlan, render_patterns, pulse_pattern,
)
from sonumio.midi_workstation.score import to_midi
from sonumio.midi_workstation.smf import encode_smf, parse_smf


def voice(id="pulse", channel=0, cells=None, **kw):
    if cells is None:
        cells = pulse_pattern((60, 64, 67), every_steps=4, total_steps=16)
    return PatternVoice(id, id.title(), channel, 0, cells, **kw)


def test_basic_repeat_builds_score_and_valid_midi_binary():
    plan = PatternPlan(bars=8)
    score, report = render_patterns(plan, (voice(),))
    assert score.end == 8 * 1920
    assert score.note_count == 32
    assert report.score_notes == 32
    assert report.source_cells == 4
    assert all(note.stop <= score.end for note in score.notes)
    midi = to_midi(score)
    parsed = parse_smf(encode_smf(midi))
    assert len(parsed.midi.tracks) >= 2
    assert len(parsed.midi.events) >= 64
    assert report.as_dict()["trained_model_used"] is False


def test_original_pattern_identity_stable_across_runs():
    plan = PatternPlan(bars=5, seed=23)
    cells = (
        PatternCell(0, 60, probability_percent=50),
        PatternCell(2, 64, probability_percent=60),
        PatternCell(4, 67, probability_percent=15),
    )
    a = render_patterns(plan, (voice(cells=cells),))
    b = render_patterns(plan, (voice(cells=cells),))
    assert a == b
    assert a[0].sha256 == b[0].sha256
    json.dumps(a[1].as_dict(), allow_nan=False)


def test_different_random_seeds_can_change_probability_outcomes():
    cells = tuple(PatternCell(i, 60 + i, probability_percent=50)
                  for i in range(12))
    results = set()
    for seed in range(12):
        score, _ = render_patterns(PatternPlan(seed=seed, bars=8),
                                   (voice(cells=cells),))
        results.add(score.sha256)
    assert len(results) > 1


def test_zero_and_full_probability_have_exact_outcomes():
    cells = (PatternCell(0, 60, probability_percent=0),
             PatternCell(4, 64, probability_percent=100))
    score, report = render_patterns(PatternPlan(bars=6), (voice(cells=cells),))
    assert score.note_count == 6
    assert report.excluded_by_probability == 6


def test_force_first_pass_excludes_only_explicit_zero_chance():
    cells = (PatternCell(0, 60, probability_percent=1),
             PatternCell(1, 61, probability_percent=0))
    score, _ = render_patterns(PatternPlan(force_first_pass=True, bars=2),
                               (voice(cells=cells),))
    assert score.note_count >= 1
    assert any(note.pitch == 60 for note in score.notes)
    assert all(note.pitch != 61 for note in score.notes)


@pytest.mark.parametrize("numerator,denominator,steps", [
    (3, 4, 12), (4, 4, 16), (6, 8, 12), (7, 8, 14),
    (2, 2, 16), (5, 16, 5),
])
def test_meter_subdivisions_use_integer_clock(numerator, denominator, steps):
    plan = PatternPlan(numerator=numerator, denominator=denominator,
                       bars=4, steps_per_quarter=4)
    assert plan.steps_per_bar == steps
    score, report = render_patterns(plan, (
        voice(cells=(PatternCell(0, 60),)),
    ))
    assert score.meter[0].numerator == numerator
    assert score.end == 4 * steps * plan.ticks_per_step
    assert report.bars == 4


def test_periodic_two_bar_pattern_repeats_correctly():
    one = voice(cells=(PatternCell(0, 60), PatternCell(20, 65)),
                repeats_every_bars=2)
    score, report = render_patterns(PatternPlan(bars=6), (one,))
    assert [note.start for note in score.notes] == [0, 2400, 3840, 6240, 7680, 10080]
    assert score.note_count == 6


def test_rotation_changes_pattern_position_each_pass():
    v = voice(cells=(PatternCell(0, 60),), rotate_steps_per_bar=3)
    score, _ = render_patterns(PatternPlan(bars=4), (v,))
    assert [n.start for n in score.notes] == [0, 1920+360, 3840+720, 5760+1080]


def test_microtiming_and_end_clipping_remain_in_score():
    v = voice(cells=(PatternCell(15, 60, duration_steps=6,
                                 microshift_ticks=30),))
    score, report = render_patterns(PatternPlan(bars=1), (v,))
    assert score.notes[0].stop == score.end
    assert report.clipped_notes == 1
    assert score.notes[0].start == 1830


def test_accent_scales_velocity_and_clamps():
    v = voice(cells=(PatternCell(0, 60, velocity=80, accent_percent=200),
                     PatternCell(4, 64, velocity=80, accent_percent=50)))
    score, report = render_patterns(PatternPlan(bars=1), (v,))
    assert [note.velocity for note in score.notes] == [127, 40]


def test_two_independent_tracks_have_valid_route_and_event_count():
    a = voice("lead", channel=0)
    b = voice("bass", channel=1, cells=(PatternCell(0, 36), PatternCell(8, 40)))
    score, report = render_patterns(PatternPlan(bars=4), (a, b))
    assert len(score.tracks) == 2
    assert score.note_count == 24
    assert report.tracks == 2
    assert {t.id for t in score.tracks} == {"lead", "bass"}


def test_muted_tracks_remain_stored_but_inactive():
    track = voice(muted=True)
    score, _ = render_patterns(PatternPlan(), (track,))
    assert score.note_count == 32
    assert not score.active_notes
    assert score.tracks[0].muted


@pytest.mark.parametrize("bad", [
    dict(ppqn=0), dict(ppqn=479, steps_per_quarter=4),
    dict(bpm=0), dict(bpm=400), dict(bars=65),
    dict(numerator=0), dict(denominator=3),
    dict(steps_per_quarter=0), dict(steps_per_quarter=33),
    dict(seed=-1), dict(seed=True), dict(force_first_pass=1),
])
def test_invalid_pattern_plan_rejected(bad):
    with pytest.raises(MidiWorkstationError):
        PatternPlan(**bad)


@pytest.mark.parametrize("bad", [
    dict(step=-1), dict(step=512), dict(pitch=128),
    dict(duration_steps=0), dict(velocity=0),
    dict(probability_percent=-1), dict(probability_percent=101),
    dict(accent_percent=0), dict(accent_percent=201),
    dict(microshift_ticks=2049), dict(pan=128),
    dict(label="invalid\nvalue"),
])
def test_invalid_pattern_cell_rejected(bad):
    fields = dict(step=0, pitch=60)
    fields.update(bad)
    with pytest.raises(MidiWorkstationError):
        PatternCell(**fields)


def test_duplicate_pattern_step_pitch_rejected():
    with pytest.raises(MidiWorkstationError):
        voice(cells=(PatternCell(0, 60), PatternCell(0, 60)))


def test_ordering_of_pattern_events_must_be_canonical():
    with pytest.raises(MidiWorkstationError):
        voice(cells=(PatternCell(4, 60), PatternCell(0, 60)))


def test_voice_repeat_span_must_fit_pattern_cells():
    with pytest.raises(MidiWorkstationError):
        render_patterns(PatternPlan(), (voice(cells=(PatternCell(16, 60),)),))


def test_incompatible_shared_channel_patches_are_rejected():
    a = voice("a", channel=0)
    b = PatternVoice(
        "b", "Second", 0, 1, (PatternCell(0, 64),),
    )
    with pytest.raises(MidiWorkstationError):
        render_patterns(PatternPlan(), (a, b))


def test_long_pattern_is_rejected_before_materializing_notes():
    # Conservative gate catches maximum theoretical notes before RNG.
    cells = tuple(PatternCell(i, 60 + i % 60) for i in range(256))
    voices = tuple(PatternVoice(f"v-{i}", f"Voice {i}", i % 16, 0,
                                cells, repeats_every_bars=16)
                   for i in range(48))
    with pytest.raises(MidiLimitError):
        render_patterns(PatternPlan(bars=64), voices)


def test_empty_cell_voice_is_valid_silent_score():
    score, report = render_patterns(PatternPlan(bars=2), (voice(cells=()),))
    assert score.note_count == 0
    assert report.score_notes == 0


def test_pulse_helper_makes_predictable_arpeggio():
    cells = pulse_pattern((60, 64, 67), every_steps=2, total_steps=12,
                          velocity=80, duration_steps=1)
    assert len(cells) == 6
    assert [c.pitch for c in cells] == [60, 64, 67, 60, 64, 67]
    assert [c.step for c in cells] == [0, 2, 4, 6, 8, 10]


def test_invalid_pulse_parameters():
    for kwargs in ({"pitches": ()}, {"pitches": (128,)},
                   {"pitches": (60,), "every_steps": 0},
                   {"pitches": (60,), "velocity": 0}):
        with pytest.raises(MidiWorkstationError):
            pulse_pattern(**kwargs)
