"""Finite tonal interval sets, chords, weighted hypotheses and voice-leading."""
import json
import pytest

from sonumio.midi_workstation.contracts import MidiWorkstationError
from sonumio.midi_workstation.harmony import (
    PITCH_CLASSES, SCALE_INTERVALS, CHORD_INTERVALS,
    Scale, Chord, ChordEvidence, VoicingSettings, VoicedChord,
    pitch_class_name, parse_pitch_class, candidate_voicings,
    estimate_chords, voice_leading_cost, arrange_voice_leading,
    transpose_to_scale,
)
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.tempo import TempoPoint, MeterPoint


@pytest.mark.parametrize("note,label", [
    (0, "C"), (1, "C#"), (60, "C"), (69, "A"), (127, "G"),
])
def test_pitch_class_names_are_equal_tempered(note, label):
    assert pitch_class_name(note) == label


@pytest.mark.parametrize("spelling,pitch", [
    ("C", 0), ("B#", 0), ("Db", 1), ("D#", 3),
    ("Eb", 3), ("E", 4), ("Fb", 4), ("E#", 5),
    ("Gb", 6), ("Ab", 8), ("Bb", 10), ("Cb", 11),
])
def test_pitch_class_aliases_are_explicit(spelling, pitch):
    assert parse_pitch_class(spelling) == pitch


def test_major_scale_contains_correct_pitch_classes():
    c = Scale(0, "major")
    assert c.classes == (0, 2, 4, 5, 7, 9, 11)
    assert c.contains(60)
    assert c.contains(64)
    assert not c.contains(61)
    assert c.degree(60) == 1
    assert c.degree(64) == 3
    assert c.degree(61) is None


@pytest.mark.parametrize("mode", list(SCALE_INTERVALS))
def test_every_named_mode_has_tonic_and_unique_classes(mode):
    scale = Scale(6, mode)
    assert scale.classes[0] == 6
    assert len(scale.classes) == len(set(scale.classes))
    assert all(0 <= p <= 11 for p in scale.classes)


def test_nearest_diatonic_pitch_policy_has_deterministic_ties():
    scale = Scale(0, "major")
    assert scale.closest(61) == 60
    assert scale.closest(61, prefer="up") == 62
    assert scale.closest(61, prefer="down") == 60
    assert scale.closest(63) == 62


def test_diatonic_register_limits_reject_unavailable_pitch():
    scale = Scale(0, "major")
    with pytest.raises(MidiWorkstationError):
        scale.closest(0, minimum=1, maximum=1)
    with pytest.raises(MidiWorkstationError):
        scale.closest(127, prefer="up", maximum=120)


@pytest.mark.parametrize("quality", list(CHORD_INTERVALS))
def test_every_chord_quality_has_valid_interval_classes(quality):
    chord = Chord(9, quality)
    assert chord.classes[0] == 9
    assert len(chord.classes) >= 2
    assert chord.contains(69)
    assert len(set(chord.classes)) == len(chord.classes)
    assert chord.symbol.startswith("A")


@pytest.mark.parametrize("quality,pitches", [
    ("major", (60, 64, 67)),
    ("minor", (60, 63, 67)),
    ("diminished", (60, 63, 66)),
    ("augmented", (60, 64, 68)),
    ("suspended2", (60, 62, 67)),
    ("major7", (60, 64, 67, 71)),
])
def test_root_chords_have_mathematically_defined_notes(quality, pitches):
    assert Chord(0, quality).pitches(4) == pitches


def test_triads_inversions_have_equivalent_pitch_classes():
    c = Chord(0, "major")
    first = Chord(0, "major", inversion=1)
    second = Chord(0, "major", inversion=2)
    assert c.pitches(4) == (60, 64, 67)
    assert first.pitches(4) == (64, 67, 72)
    assert second.pitches(4) == (67, 72, 76)
    assert set(c.classes) == set(first.classes) == set(second.classes)


def test_weighted_chord_hypothesis_recognizes_clean_triad():
    pitches = ((60, 1.), (64, 1.), (67, 1.))
    evidence = estimate_chords(pitches, candidates=10)
    assert evidence
    assert evidence[0].chord == Chord(0, "major")
    assert evidence[0].coverage == pytest.approx(1.)
    assert evidence[0].root_present


def test_single_note_does_not_claim_complete_major_triad():
    assert estimate_chords(((60, 1.),)) == ()
    assert estimate_chords(()) == ()


def test_dense_ambiguous_pitch_set_still_has_explicit_unconfidence():
    pitches = tuple((n, 1.) for n in range(60, 72))
    evidence = estimate_chords(pitches)
    assert evidence
    assert evidence[0].coverage < .4
    assert evidence[0].out_of_chord_weight > .6


def test_voice_leading_candidates_stay_in_declared_register():
    cfg = VoicingSettings(minimum_pitch=48, maximum_pitch=84, voices=3)
    values = candidate_voicings(Chord(0, "major"), cfg)
    assert values
    assert len(values) <= 4096
    assert all(48 <= v[0] <= v[-1] <= 84 for v in values)
    assert all(set(p % 12 for p in v) == set(Chord(0, "major").classes)
               for v in values)


def test_voice_leading_optimization_is_reproducible():
    chords = (Chord(0, "major"), Chord(5, "major"),
              Chord(7, "dominant7"), Chord(0, "major"))
    cfg = VoicingSettings(minimum_pitch=48, maximum_pitch=84,
                          voices=4, beam_width=80)
    first = arrange_voice_leading(chords, cfg)
    second = arrange_voice_leading(chords, cfg)
    assert first == second
    assert len(first) == 4
    assert all(len(step.pitches) == 4 for step in first)
    assert all(step.transition_cost >= -100 for step in first)


def test_repeated_chord_can_hold_common_tones():
    chords = (Chord(0, "major"), Chord(0, "major"), Chord(0, "major"))
    result = arrange_voice_leading(chords)
    assert result[0].pitches == result[1].pitches == result[2].pitches
    assert result[1].transition_cost <= 0


def test_direct_cost_is_symmetric_and_penalizes_large_leaps():
    settings = VoicingSettings()
    a = (60, 64, 67)
    b = (62, 65, 69)
    c = (72, 76, 79)
    assert voice_leading_cost(a, b, settings) == voice_leading_cost(b, a, settings)
    assert voice_leading_cost(a, c, settings) > voice_leading_cost(a, b, settings)


def source_score():
    return Score("Scale correction", 480, (
        ScoreTrack("keys", "Keys", 0, 0, 0, 0, (
            ScoreNote("one", 0, 120, 61),
            ScoreNote("two", 120, 120, 63),
            ScoreNote("three", 240, 120, 65),
        )),
        ScoreTrack("drum", "Percussion", 9, 0, 0, 0, (
            ScoreNote("kick", 0, 120, 36),
        )),
    ), (TempoPoint(0, 500000),), (MeterPoint(0, 4, 4),), 480)


def test_diatonic_pitch_correction_preserves_percussion_by_default():
    original = source_score()
    result, ledger = transpose_to_scale(original, Scale(0, "major"))
    assert [n.pitch for n in result.track_by_id("keys").notes] == [60, 62, 65]
    assert result.track_by_id("drum").notes == original.track_by_id("drum").notes
    assert original.notes[0].pitch == 61
    assert ledger["changed_pitches"] == 2
    assert ledger["source_mutated"] is False


def test_scale_correction_track_selection_only_modifies_requested_track():
    original = source_score()
    result, _ = transpose_to_scale(original, Scale(0), track_ids=("drum",))
    assert result.track_by_id("keys") == original.track_by_id("keys")


@pytest.mark.parametrize("bad", [
    lambda: Scale(12), lambda: Scale(0, "unknown"),
    lambda: Chord(-1), lambda: Chord(0, "nonexistent"),
    lambda: Chord(0, "major", 3),
    lambda: VoicingSettings(voices=0),
    lambda: VoicingSettings(minimum_pitch=90, maximum_pitch=80),
    lambda: VoicingSettings(beam_width=5000),
])
def test_invalid_harmony_models_fail_closed(bad):
    with pytest.raises(MidiWorkstationError):
        bad()


def test_no_candidate_register_has_clear_failure():
    with pytest.raises(MidiWorkstationError):
        arrange_voice_leading(
            (Chord(0, "major"),),
            VoicingSettings(minimum_pitch=60, maximum_pitch=61, voices=3),
        )
