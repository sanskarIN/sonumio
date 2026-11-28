"""Precise velocity, bank and metronome preservation through MusicXML.

The performance subset is not a full notation parser. These tests assert the
limited semantics we *do* claim to preserve and keep unsupported markups
fail-closed rather than silently discarding them.
"""
from dataclasses import replace
from decimal import Decimal
import xml.etree.ElementTree as ET

import pytest

from sonumio.midi_workstation.contracts import MidiFormatError
from sonumio.midi_workstation.musicxml_common import (
    velocity_to_dynamics, dynamics_to_velocity,
)
from sonumio.midi_workstation.musicxml_export import encode_musicxml
from sonumio.midi_workstation.musicxml_import import decode_musicxml
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.tempo import MeterPoint, TempoPoint


def score(*notes, msb=0, lsb=0, end=3840, tempo=500000):
    base = Score.empty(end=end)
    track = replace(base.tracks[0], bank_msb=msb, bank_lsb=lsb,
                    notes=tuple(sorted(notes, key=lambda n:(n.start,n.pitch,n.duration,n.id))))
    return replace(base, tracks=(track,), tempo=(TempoPoint(0,tempo),))


def modify(xml, callback):
    root = ET.fromstring(xml)
    callback(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


@pytest.mark.parametrize("onset", range(1, 128))
def test_all_onset_velocities_roundtrip(onset):
    source = score(ScoreNote("note", 0, 120, 67, onset))
    result, receipt = decode_musicxml(encode_musicxml(source)[0])
    assert result.notes[0].velocity == onset
    assert receipt.notes == 1
    assert dynamics_to_velocity(velocity_to_dynamics(onset), allow_zero=False) == onset


@pytest.mark.parametrize("release", range(0, 128))
def test_all_release_velocities_roundtrip(release):
    source = score(ScoreNote("note", 0, 120, 67, 83, release))
    result, _ = decode_musicxml(encode_musicxml(source)[0])
    assert result.notes[0].release_velocity == release
    assert dynamics_to_velocity(velocity_to_dynamics(release), allow_zero=True) == release


@pytest.mark.parametrize("msb,lsb", [
    (0,0), (0,1), (1,0), (1,1), (4,90), (63,127), (64,0),
    (90,64), (126,126), (127,0), (127,127),
])
def test_14bit_banks_remain_exact(msb,lsb):
    source = score(ScoreNote("note", 0, 120, 67), msb=msb, lsb=lsb)
    payload, _ = encode_musicxml(source)
    tree = ET.fromstring(payload)
    bank = tree.find("./part-list/score-part/midi-instrument/midi-bank")
    assert bank is not None and int(bank.text) == msb * 128 + lsb + 1
    restored, _ = decode_musicxml(payload)
    assert restored.tracks[0].bank_msb == msb
    assert restored.tracks[0].bank_lsb == lsb


@pytest.mark.parametrize("begin,duration", [
    (0,3840), (100,3000), (2000,1700), (1,3839), (1500,400),
])
def test_tied_note_keeps_expression_across_bar_boundaries(begin,duration):
    source = score(ScoreNote("held", begin, duration, 60, 3, 127))
    raw, _ = encode_musicxml(source)
    restored, _ = decode_musicxml(raw)
    assert restored.notes[0].velocity == 3
    assert restored.notes[0].release_velocity == 127
    assert restored.notes[0].duration == duration
    assert restored.notes[0].start == begin


@pytest.mark.parametrize("bad", [
    "", "-1", "NaN", "Infinity", "1e3", "1,2", "+1", "001.2e1",
    "999", "142", "0", "0.000", "100.\u03c0", "a" * 50,
    "1 " , " 1", "1.2.3", "1.00000000000000000000000000000",
])
def test_invalid_note_on_dynamics_rejected(bad):
    xml, _ = encode_musicxml(score(ScoreNote("note", 0, 120, 60)))
    damaged = modify(xml, lambda root:root.find("./part/measure/note").set("dynamics",bad))
    with pytest.raises(MidiFormatError):
        decode_musicxml(damaged)


@pytest.mark.parametrize("bad", [
    "-1", "nan", "inf", "142", "500", "3e4", "no", "1..1",
    "1.000000000000000000000000000000", "2 2",
])
def test_invalid_release_dynamics_rejected(bad):
    xml, _ = encode_musicxml(score(ScoreNote("note", 0, 120, 60)))
    damaged = modify(xml, lambda root:root.find("./part/measure/note").set("end-dynamics",bad))
    with pytest.raises(MidiFormatError):
        decode_musicxml(damaged)


@pytest.mark.parametrize("bank", ["0","16385","1e2","-1","", "none", "1.5"])
def test_invalid_midi_bank_rejected(bank):
    xml, _ = encode_musicxml(score(ScoreNote("note", 0, 120, 60)))
    damaged = modify(xml, lambda root:root.find("./part-list/score-part/midi-instrument/midi-bank").__setattr__("text",bank))
    with pytest.raises(MidiFormatError):
        decode_musicxml(damaged)


def test_missing_dynamics_uses_documented_safe_defaults():
    xml, _ = encode_musicxml(score(ScoreNote("note", 0, 120, 60, 23, 42)))
    def drop(root):
        root.find("./part/measure/note").attrib.pop("dynamics")
        root.find("./part/measure/note").attrib.pop("end-dynamics")
    imported, _ = decode_musicxml(modify(xml,drop))
    assert (imported.notes[0].velocity, imported.notes[0].release_velocity) == (100,0)


def test_missing_midi_bank_uses_general_midi_default():
    xml, _ = encode_musicxml(score(ScoreNote("note", 0, 120, 60), msb=5, lsb=1))
    def drop(root):
        midi = root.find("./part-list/score-part/midi-instrument")
        midi.remove(midi.find("midi-bank"))
    imported, _ = decode_musicxml(modify(xml,drop))
    assert (imported.tracks[0].bank_msb, imported.tracks[0].bank_lsb) == (0,0)


def test_inconsistent_tie_expression_is_not_silently_merged():
    source = score(ScoreNote("held", 100, 3500, 60, 42, 87))
    xml, _ = encode_musicxml(source)
    def change(root):
        second = root.findall("./part/measure")[1].find("note")
        second.set("dynamics", "100")
    with pytest.raises(MidiFormatError,match="expressive dynamics"):
        decode_musicxml(modify(xml,change))


def test_nonquarter_metronome_mark_raises():
    xml, _ = encode_musicxml(score(ScoreNote("note", 0, 120, 60)))
    def change(root):
        root.find("./part/measure/direction/direction-type/metronome/beat-unit").text = "eighth"
    with pytest.raises(MidiFormatError,match="quarter"):
        decode_musicxml(modify(xml,change))


def test_metronome_playback_disagreement_raises():
    xml, _ = encode_musicxml(score(ScoreNote("note", 0, 120, 60)))
    def change(root):
        root.find("./part/measure/direction/direction-type/metronome/per-minute").text="300"
    with pytest.raises(MidiFormatError, match="disagree"):
        decode_musicxml(modify(xml,change))


def test_direction_type_refuses_other_effects():
    xml, _ = encode_musicxml(score(ScoreNote("note", 0, 120, 60)))
    def change(root):
        parent=root.find("./part/measure/direction/direction-type")
        ET.SubElement(parent,"wedge",number="1",type="crescendo")
    with pytest.raises(MidiFormatError,match="direction-type"):
        decode_musicxml(modify(xml,change))


def test_musicxml_note_children_follow_schema_pitch_duration_tie_voice_notations():
    xml, _ = encode_musicxml(score(ScoreNote("note", 100, 3400, 60)))
    root=ET.fromstring(xml)
    first=root.find("./part/measure/note")
    assert [tag.tag for tag in first] == ["pitch", "duration", "tie", "voice", "notations"]
    assert root.find("./part/measure/direction/direction-type/metronome") is not None
    assert root.find("./part/measure/direction/sound").get("tempo")


def test_tie_expression_and_bank_are_idempotent_across_multiple_roundtrips():
    original=score(ScoreNote("note", 400, 3200, 60, 88, 15), msb=97, lsb=31)
    one,_=decode_musicxml(encode_musicxml(original)[0])
    two,_=decode_musicxml(encode_musicxml(one)[0])
    assert one==two
    assert (one.tracks[0].bank_msb,one.tracks[0].bank_lsb)==(97,31)
    assert one.notes[0].velocity==88 and one.notes[0].release_velocity==15
