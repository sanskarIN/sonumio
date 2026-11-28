"""MusicXML hostile-input validation and supported external notation semantics."""
from dataclasses import replace
from hashlib import sha256
import random
import xml.etree.ElementTree as ET

import pytest

from sonumio.midi_workstation.contracts import (
    MidiFormatError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.musicxml_common import (
    bounded_integer, child, children, parse_musicxml, serialize_xml,
    MAX_MUSICXML_BYTES,
)
from sonumio.midi_workstation.musicxml_export import encode_musicxml
from sonumio.midi_workstation.musicxml_import import decode_musicxml
from sonumio.midi_workstation.score import Score, ScoreNote, note_sort_key
from sonumio.midi_workstation.tempo import MeterPoint, TempoPoint


def fixture():
    root = ET.fromstring(encode_musicxml(Score.empty())[0])
    measure_node = root.find("part").find("measure")
    for advance in list(measure_node.findall("forward")):
        measure_node.remove(advance)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def modify(fn):
    root = ET.fromstring(fixture())
    fn(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def measure(root):
    return root.find("part").find("measure")


def add_note(root, *, start=0, duration=480, pitch="C", octave=4,
             voice=1, chord=False, tie=(), note_id=None):
    parent = measure(root)
    if start:
        forward = ET.SubElement(parent, "forward")
        ET.SubElement(forward, "duration").text = str(start)
    fields = {} if note_id is None else {"id":note_id}
    note = ET.SubElement(parent, "note", fields)
    if chord:
        ET.SubElement(note, "chord")
    pitch_element = ET.SubElement(note, "pitch")
    ET.SubElement(pitch_element, "step").text = pitch
    ET.SubElement(pitch_element, "octave").text = str(octave)
    ET.SubElement(note, "duration").text = str(duration)
    ET.SubElement(note, "voice").text = str(voice)
    for kind in tie:
        ET.SubElement(note, "tie", type=kind)
    return note


def as_xml(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


@pytest.mark.parametrize("bad", [
    b"", b"<score-partwise", b"<>invalid</score-partwise>",
    b"<!DOCTYPE score-partwise [ <!ENTITY x 'y'> ]><score-partwise/>",
    b"<?xml version='1.0'?><!DOCTYPE score-partwise><score-partwise/>",
    b"\xff"*40, b"A"*128,
    b"<?xml version='1.0'?><!ENTITY x 'foo'><score-partwise/>",
])
def test_malformed_and_entity_payloads_are_rejected(bad):
    with pytest.raises((MidiFormatError, MidiLimitError)):
        decode_musicxml(bad)


def test_xml_size_budget_rejects_before_parse():
    with pytest.raises(MidiLimitError):
        decode_musicxml(b"x" * (MAX_MUSICXML_BYTES + 1))


def test_deep_nested_xml_rejected():
    raw = b"<score-partwise>" + b"<x>"*55 + b"</x>"*55 + b"</score-partwise>"
    with pytest.raises(MidiLimitError, match="nesting"):
        decode_musicxml(raw)


@pytest.mark.parametrize("tag", ["score-timewise", "score-document", "part", "html"])
def test_unsupported_musicxml_root_does_not_default_to_score(tag):
    raw = f"<{tag} version='4.0'>".encode() + b"x"*50 + f"</{tag}>".encode()
    with pytest.raises(MidiFormatError):
        decode_musicxml(raw)


def test_version_policy_rejects_unverified_versions():
    root = ET.fromstring(fixture())
    root.set("version", "6.0")
    with pytest.raises(MidiFormatError, match="version"):
        decode_musicxml(as_xml(root))


def test_empty_part_list_is_error():
    root = ET.fromstring(fixture())
    root.remove(root.find("part-list"))
    with pytest.raises(MidiFormatError):
        decode_musicxml(as_xml(root))


def test_duplicate_parts_are_rejected():
    root = ET.fromstring(fixture())
    item = root.find("part-list").find("score-part")
    root.find("part-list").append(ET.fromstring(ET.tostring(item)))
    with pytest.raises(MidiFormatError):
        decode_musicxml(as_xml(root))


def test_unrecognized_measure_feature_fails_closed():
    raw = modify(lambda root: ET.SubElement(measure(root), "barline"))
    with pytest.raises(MidiFormatError, match="barline"):
        decode_musicxml(raw)


@pytest.mark.parametrize("feature", ["unpitched", "grace", "time-modification",
                                      "play", "instrument", "lyric", "cue"])
def test_unsupported_note_semantics_fails_closed(feature):
    root = ET.fromstring(fixture())
    note = add_note(root)
    ET.SubElement(note, feature)
    with pytest.raises(MidiFormatError):
        decode_musicxml(as_xml(root))


@pytest.mark.parametrize("feature", ["direction-type", "words", "dynamics"])
def test_unsupported_direction_without_tempo_fails_closed(feature):
    root = ET.fromstring(fixture())
    direction = ET.SubElement(measure(root), "direction")
    ET.SubElement(direction, feature)
    with pytest.raises(MidiFormatError):
        decode_musicxml(as_xml(root))


def test_external_timed_note_is_reconstructed_without_fake_expression():
    root = ET.fromstring(fixture())
    add_note(root, start=120, duration=360, pitch="F", octave=5)
    score, ledger = decode_musicxml(as_xml(root))
    assert len(score.notes) == 1
    note = score.notes[0]
    assert (note.start, note.duration, note.pitch) == (120,360,77)
    assert note.velocity == 100
    assert ledger.notes == 1
    assert ledger.losses


def test_explicit_rest_moves_cursor_without_creating_note():
    root = ET.fromstring(fixture())
    note = ET.SubElement(measure(root), "note")
    ET.SubElement(note, "rest")
    ET.SubElement(note, "duration").text = "240"
    add_note(root, duration=240)
    score, _ = decode_musicxml(as_xml(root))
    assert score.notes[0].start == 240


def test_concurrent_chord_notes_share_start_time():
    root = ET.fromstring(fixture())
    add_note(root, duration=480, pitch="C", octave=4, voice=1)
    add_note(root, duration=480, pitch="E", octave=4, voice=1, chord=True)
    score, _ = decode_musicxml(as_xml(root))
    assert len(score.notes) == 2
    assert {(x.start,x.duration,x.pitch) for x in score.notes} == {
        (0,480,60),(0,480,64),
    }


def test_chord_without_previous_note_fails():
    root = ET.fromstring(fixture())
    add_note(root,chord=True)
    with pytest.raises(MidiFormatError, match="chord"):
        decode_musicxml(as_xml(root))


def test_chord_wrong_voice_fails():
    root = ET.fromstring(fixture())
    add_note(root, duration=240)
    add_note(root, duration=240, chord=True, voice=2)
    with pytest.raises(MidiFormatError, match="chord"):
        decode_musicxml(as_xml(root))


def test_backup_voice_independent_timing():
    root = ET.fromstring(fixture())
    add_note(root, duration=480, voice=1, pitch="C")
    backup = ET.SubElement(measure(root), "backup")
    ET.SubElement(backup, "duration").text = "480"
    add_note(root, duration=480, voice=2, pitch="G")
    score, _ = decode_musicxml(as_xml(root))
    assert {(n.start,n.pitch) for n in score.notes} == {(0,60),(0,67)}


def test_backup_before_zero_fails():
    root = ET.fromstring(fixture())
    backup = ET.SubElement(measure(root), "backup")
    ET.SubElement(backup, "duration").text = "1"
    with pytest.raises(MidiFormatError, match="boundary"):
        decode_musicxml(as_xml(root))


def test_forward_past_measure_fails():
    root = ET.fromstring(fixture())
    forward = ET.SubElement(measure(root), "forward")
    ET.SubElement(forward, "duration").text = "4000"
    with pytest.raises(MidiFormatError, match="cursor exceeds"):
        decode_musicxml(as_xml(root))


def test_negative_offset_rejected():
    root = ET.fromstring(fixture())
    direction = ET.SubElement(measure(root), "direction")
    ET.SubElement(direction, "offset").text = "-1"
    ET.SubElement(direction, "sound", tempo="130")
    with pytest.raises(MidiFormatError):
        decode_musicxml(as_xml(root))


def test_tempo_direction_at_zero_restores_constant_tempo():
    root = ET.fromstring(fixture())
    direction = ET.SubElement(measure(root), "direction")
    ET.SubElement(direction, "sound", tempo="120")
    score, _ = decode_musicxml(as_xml(root))
    assert score.tempo == (TempoPoint(0,500000),)


def test_pitch_flat_and_double_sharp_are_supported_semantically():
    root = ET.fromstring(fixture())
    note = add_note(root, pitch="D", octave=4)
    pitch = note.find("pitch")
    ET.SubElement(pitch, "alter").text = "-1"
    score, _ = decode_musicxml(as_xml(root))
    assert score.notes[0].pitch == 61


@pytest.mark.parametrize("alter", ["0.5", "nan", "abc", "3", "-5", "inf", "-0.25"])
def test_microtonal_pitch_fails_without_false_rounding(alter):
    root = ET.fromstring(fixture())
    note = add_note(root)
    ET.SubElement(note.find("pitch"), "alter").text = alter
    with pytest.raises(MidiFormatError):
        decode_musicxml(as_xml(root))


def test_unclosed_tie_fails():
    root = ET.fromstring(fixture())
    add_note(root, tie=("start",))
    with pytest.raises(MidiFormatError, match="unclosed"):
        decode_musicxml(as_xml(root))


def test_orphan_stop_tie_fails():
    root = ET.fromstring(fixture())
    add_note(root, tie=("stop",))
    with pytest.raises(MidiFormatError, match="Unmatched"):
        decode_musicxml(as_xml(root))


def test_duplicate_tie_types_fail():
    root = ET.fromstring(fixture())
    add_note(root, tie=("start", "start"))
    with pytest.raises(MidiFormatError):
        decode_musicxml(as_xml(root))


def test_note_unsupported_ornaments_fail_even_if_duration_well_formed():
    root = ET.fromstring(fixture())
    note = add_note(root)
    notations = ET.SubElement(note, "notations")
    ET.SubElement(notations, "ornaments")
    with pytest.raises(MidiFormatError, match="ornaments"):
        decode_musicxml(as_xml(root))


def test_empty_midi_instrument_fields_are_rejected():
    root = ET.fromstring(fixture())
    midi = root.find("part-list").find("score-part").find("midi-instrument")
    midi.remove(midi.find("midi-program"))
    with pytest.raises(MidiFormatError, match="both channel and program"):
        decode_musicxml(as_xml(root))


def test_measure_numbering_is_strict():
    root = ET.fromstring(fixture())
    measure(root).set("number", "0")
    with pytest.raises(MidiFormatError, match="consecutive"):
        decode_musicxml(as_xml(root))


def test_stray_processing_instruction_rejected():
    data = fixture().replace(b"<score-partwise", b"<?unsafe key='x' ?><score-partwise")
    with pytest.raises(MidiFormatError):
        decode_musicxml(data)


def test_duplicate_miscellaneous_fields_fail():
    root = ET.fromstring(fixture())
    misc = root.find("identification").find("miscellaneous")
    first = misc.find("miscellaneous-field")
    misc.append(ET.fromstring(ET.tostring(first)))
    with pytest.raises(MidiFormatError, match="Duplicate"):
        decode_musicxml(as_xml(root))


def test_output_is_independent_of_inline_xml_whitespace():
    original = fixture()
    score, _ = decode_musicxml(original)
    noisy = original.replace(b"><", b">  \n  <")
    score2, _ = decode_musicxml(noisy)
    assert score.sha256 == score2.sha256


def test_oversized_note_text_fails_resource_guard():
    root = ET.fromstring(fixture())
    title = root.find("movement-title")
    title.text = "A" * 16385
    with pytest.raises(MidiLimitError):
        decode_musicxml(as_xml(root))


def test_score_part_mismatch_rejected():
    root = ET.fromstring(fixture())
    root.find("part").set("id", "unknown")
    with pytest.raises(MidiFormatError):
        decode_musicxml(as_xml(root))


@pytest.mark.parametrize("value", ["-1", "0", "0.000001", "NaN", "Infinity", "a", "1e5000"])
def test_bad_musicxml_tempos_fail(value):
    root = ET.fromstring(fixture())
    direction = ET.SubElement(measure(root), "direction")
    ET.SubElement(direction, "sound", tempo=value)
    with pytest.raises((MidiFormatError, MidiWorkstationError)):
        decode_musicxml(as_xml(root))


def test_xml_namespace_default_is_supported():
    raw = fixture()
    with_namespace = raw.replace(b"<score-partwise version=", b"<score-partwise xmlns='http://www.musicxml.org/ns/musicxml' version=", 1)
    score, _ = decode_musicxml(with_namespace)
    assert score.note_count == 0


def test_import_does_not_change_source_bytes():
    raw = fixture()
    digest = sha256(raw).hexdigest()
    decode_musicxml(raw)
    assert sha256(raw).hexdigest() == digest
