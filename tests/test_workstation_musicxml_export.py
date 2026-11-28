"""Contract, timing, deterministic projection and resource-guard tests."""
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import random
import xml.etree.ElementTree as ET

import pytest

from sonumio.midi_workstation.contracts import MidiFormatError, MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.musicxml_common import (
    pitch_from_xml, pitch_to_xml, tempo_to_bpm, bpm_to_tempo,
)
from sonumio.midi_workstation.musicxml_export import (
    NoteFragment, allocate_voices, encode_musicxml, split_fragments, split_measures,
)
from sonumio.midi_workstation.musicxml_import import decode_musicxml
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack, note_sort_key
from sonumio.midi_workstation.tempo import MeterPoint, TempoPoint


def score(notes=(), *, end=1920, ppqn=480, meter=None, tempo=None,
          title="Original study", channels=(0,), track_names=None):
    if track_names is None:
        track_names = tuple(f"Part {i+1}" for i in range(len(channels)))
    tracks = tuple(ScoreTrack(f"layer-{i}", track_names[i], channel, 0, 0, 0,
                 tuple(sorted(notes[i] if isinstance(notes, tuple) and len(notes) == len(channels)
                     and len(channels) > 1 else notes, key=note_sort_key)))
                 for i, channel in enumerate(channels))
    return Score(title, ppqn, tracks, tuple(tempo or [TempoPoint(0, 500000)]),
                 tuple(meter or [MeterPoint(0, 4, 4)]), end)


def properties(original, restored):
    assert original.title == restored.title
    assert original.ppqn == restored.ppqn
    assert original.end == restored.end
    assert original.tempo == restored.tempo
    assert original.meter == restored.meter
    assert len(original.tracks) == len(restored.tracks)
    for source_track, target_track in zip(original.tracks, restored.tracks):
        assert source_track.id == target_track.id
        assert source_track.name == target_track.name
        assert source_track.channel == target_track.channel
        assert source_track.program == target_track.program
        assert [(n.id, n.start, n.duration, n.pitch) for n in source_track.notes] == [
            (n.id, n.start, n.duration, n.pitch) for n in target_track.notes]


def test_empty_score_canonical_roundtrip():
    original = score((), end=1)
    encoded, report = encode_musicxml(original)
    result, import_report = decode_musicxml(encoded)
    properties(original, result)
    assert report.notes == 0
    assert import_report.notes == 0
    assert encoded.startswith(b"<?xml ")
    assert b"DOCTYPE" not in encoded
    assert import_report.as_dict()["lossless_roundtrip_claimed"] is False


@pytest.mark.parametrize("start,duration,pitch", [
    (0, 1, 0), (0, 1920, 127), (1, 1919, 60), (1919, 1, 69),
    (400, 300, 36), (900, 1000, 72), (800, 1100, 127),
    (1, 40, 28), (100, 200, 45), (1050, 50, 111),
])
def test_single_pitched_note_preserves_timing_identity(start, duration, pitch):
    original = score((ScoreNote("test-note", start, duration, pitch),))
    encoded, report = encode_musicxml(original)
    result, receipt = decode_musicxml(encoded)
    properties(original, result)
    assert report.notes == 1
    assert receipt.notes == 1
    assert result.tracks[0].notes[0].velocity == 100
    assert result.tracks[0].notes[0].pan == 64


@pytest.mark.parametrize("pitch", list(range(128)))
def test_pitch_spelling_is_midi_exact(pitch):
    step, accidental, octave = pitch_to_xml(pitch)
    root = ET.Element("pitch")
    ET.SubElement(root, "step").text = step
    if accidental:
        ET.SubElement(root, "alter").text = str(accidental)
    ET.SubElement(root, "octave").text = str(octave)
    assert pitch_from_xml(root) == pitch


@pytest.mark.parametrize("microseconds", [
    1, 2, 3, 7, 10, 37, 100, 32767, 32768, 60000, 234567,
    444444, 499999, 500000, 500001, 750000, 999999, 1000000,
    1200000, 16777215,
])
def test_tempo_decimal_exactly_recovers_microseconds(microseconds):
    assert bpm_to_tempo(tempo_to_bpm(microseconds)) == microseconds


@pytest.mark.parametrize("meter", [(2, 2), (3, 4), (6, 8), (7, 8),
                                    (9, 16), (2, 1), (12, 32)])
def test_time_signatures_with_integer_ticks(meter):
    value = MeterPoint(0, *meter)
    span = int(value.measure_ticks(480))
    original = score((ScoreNote("n", 1, max(1, span-1), 60),), end=span,
                     meter=(value,))
    encoded, _ = encode_musicxml(original)
    restored, _ = decode_musicxml(encoded)
    properties(original, restored)


def test_midbar_meter_change_rejected_before_creating_output():
    value = score((), end=3840, meter=(MeterPoint(0, 4, 4),
                                         MeterPoint(100, 3, 4)))
    with pytest.raises(MidiFormatError, match="middle"):
        encode_musicxml(value)


def test_fractional_bar_ticks_rejected():
    value = score((), end=20, ppqn=1, meter=(MeterPoint(0, 1, 8),))
    with pytest.raises(MidiFormatError, match="whole metrical"):
        encode_musicxml(value)


def test_two_sections_meter_changes_preserve_bars():
    original = score((ScoreNote("a", 0, 300, 60), ScoreNote("b", 1900, 600, 62),
                      ScoreNote("c", 2500, 100, 64)), end=2640,
                     meter=(MeterPoint(0, 4, 4), MeterPoint(1920, 3, 4)))
    encoded, _ = encode_musicxml(original)
    restored, _ = decode_musicxml(encoded)
    properties(original, restored)
    assert len(split_measures(original)) == 2


def test_sustained_crossbar_note_creates_sound_and_notation_ties():
    original = score((ScoreNote("held", 500, 4000, 60),), end=4800)
    xml, _ = encode_musicxml(original)
    assert xml.count(b'<tie type="start"') == 2
    assert xml.count(b'<tie type="stop"') == 2
    assert xml.count(b'<tied type="start"') == 2
    restored, _ = decode_musicxml(xml)
    properties(original, restored)


def test_concurrent_notes_use_independent_voices_without_chord_guessing():
    notes = (ScoreNote("a", 0, 1920, 60), ScoreNote("b", 0, 1920, 64),
             ScoreNote("c", 0, 1920, 67), ScoreNote("d", 900, 500, 72))
    original = score(notes)
    xml, _ = encode_musicxml(original)
    assert xml.count(b"<backup>") == 3
    result, _ = decode_musicxml(xml)
    properties(original, result)


def test_three_tracks_have_global_conductor_tempo_but_distinct_routing():
    notes = ((ScoreNote("melody", 0, 960, 60),),
             (ScoreNote("bass", 60, 960, 36),),
             (ScoreNote("pad", 300, 1620, 55),))
    original = score(notes, channels=(0, 1, 2), end=3840,
                     tempo=(TempoPoint(0, 333333), TempoPoint(700, 500000),
                            TempoPoint(2200, 654321)))
    raw, r = encode_musicxml(original)
    restored, ledger = decode_musicxml(raw)
    properties(original, restored)
    assert r.parts == 3
    assert ledger.tempo_points == 3


def test_channel_10_percussion_fails_closed():
    original = score((ScoreNote("kick", 0, 150, 36),), channels=(9,))
    with pytest.raises(MidiWorkstationError, match="drum"):
        encode_musicxml(original)


def test_even_empty_drum_track_does_not_claim_pitched_mapping():
    original = score((), channels=(9,))
    with pytest.raises(MidiWorkstationError, match="drum"):
        encode_musicxml(original)


def test_notes_are_bounded_by_end():
    with pytest.raises(MidiWorkstationError):
        score((ScoreNote("late", 1900, 50, 60),), end=1920)


def test_duration_and_offsets_avoid_floats():
    original = score((ScoreNote("n1", 3, 773, 65),
                      ScoreNote("n2", 777, 1050, 67)),
                     tempo=(TempoPoint(0, 500000), TempoPoint(777, 537943)))
    raw, _ = encode_musicxml(original)
    reconstructed, _ = decode_musicxml(raw)
    properties(original, reconstructed)


@pytest.mark.parametrize("seed", list(range(60)))
def test_random_original_score_preserves_pitched_skeleton(seed):
    rng = random.Random(121213 + seed)
    end = 3840
    notes = []
    for n in range(rng.randint(0, 75)):
        start = rng.randrange(end)
        duration = rng.randint(1, end-start)
        notes.append(ScoreNote(f"note-{n:03}", start, duration, rng.randint(0, 127),
                               rng.randint(1,127),rng.randint(0,127), rng.randint(0,127)))
    original = score(tuple(sorted(notes, key=note_sort_key)), end=end,
        tempo=(TempoPoint(0, rng.randint(20000, 800000)),
               TempoPoint(2880, rng.randint(300000, 1700000))))
    encoded, report = encode_musicxml(original)
    reencoded, report2 = encode_musicxml(original)
    assert encoded == reencoded
    assert report.as_dict() == report2.as_dict()
    decoded, _ = decode_musicxml(encoded)
    properties(original, decoded)


def test_high_simultaneous_polyphony_fails_preflight():
    notes = tuple(ScoreNote(f"n-{i}", 0, 1920, i % 128) for i in range(65))
    original = score(tuple(sorted(notes,key=note_sort_key)))
    with pytest.raises(MidiLimitError, match="polyphony"):
        encode_musicxml(original)


def test_explicit_semantic_loss_manifest_present_on_export_and_import():
    original = score((ScoreNote("n", 0, 960, 60, 72, 55, 0),))
    xml, ledger = encode_musicxml(original)
    restored, imported = decode_musicxml(xml)
    assert ledger.losses
    assert imported.losses
    assert restored.notes[0].velocity == 100
    assert restored.notes[0].release_velocity == 0
    assert restored.notes[0].pan == 64
    assert restored.sha256 != original.sha256
    assert not ledger.as_dict()["lossless_roundtrip_claimed"]


def test_score_objects_unchanged_after_projection():
    original = score((ScoreNote("a", 0, 300, 60),))
    before = original.sha256
    encode_musicxml(original)
    assert before == original.sha256
