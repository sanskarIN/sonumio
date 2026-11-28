"""Opt-in fixed-channel routing of program-changing original MIDI files."""
import json
from pathlib import Path

import pytest

from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision
from sonumio.midi_workstation.score import from_midi, to_midi
from sonumio.midi_workstation.smf import encode_smf, parse_smf
from sonumio.midi_workstation.cli import main


def event(tick, kind, channel, *data):
    return MidiEvent.channel(tick, kind, channel, *data)


def program_notes(channel=0, program_a=10, program_b=20):
    return MidiFile(0, TimeDivision(480), (
        MidiTrack.from_events((
            event(0, 0xC, channel, program_a),
            event(0, 0x9, channel, 60, 90),
            event(120, 0x8, channel, 60, 0),
            event(240, 0xC, channel, program_b),
            event(240, 0x9, channel, 64, 100),
            event(360, 0x8, channel, 64, 0),
            event(480, 0xC, channel, program_a),
            event(480, 0x9, channel, 67, 95),
            event(600, 0x8, channel, 67, 0),
        ), end=600),
    ))


def test_patch_changes_are_not_silently_rechanneled():
    midi = program_notes()
    with pytest.raises(MidiWorkstationError, match="explicitly enable"):
        from_midi(midi)


def test_opt_in_keeps_original_patch_on_source_channel_and_rechannels_other():
    original = program_notes()
    score, report = from_midi(original, allow_patch_changes=True)
    assert score.note_count == 3
    assert len(score.tracks) == 2
    assert {track.program for track in score.tracks} == {10, 20}
    assert {(track.channel, track.program) for track in score.tracks} == {(0, 10), (1, 20)}
    assert report["source_program_changes_supported"] is True
    assert report["source_program_changes_rechanneled"] is True
    assert report["program_channel_reassignments"] == [{
        "source_channel": 0, "output_channel": 1,
        "bank_msb": 0, "bank_lsb": 0, "program": 20,
    }]
    emitted = to_midi(score)
    parsed = parse_smf(encode_smf(emitted))
    assert len(parsed.midi.tracks) == 3
    assert {e.channel_number for e in parsed.midi.events if e.is_note_on} == {0, 1}
    assert len([e for e in parsed.midi.events if e.is_note_on]) == 3


def test_default_fixed_patch_import_retains_original_channels():
    original = MidiFile(0, TimeDivision(480), (
        MidiTrack.from_events((event(0, 0xC, 0, 10),
                               event(0, 0x9, 0, 60, 100),
                               event(120, 0x8, 0, 60, 0)), end=120),
    ))
    score, report = from_midi(original, allow_patch_changes=True)
    assert score.tracks[0].channel == 0
    assert report["program_channel_reassignments"] == []
    assert not report["source_program_changes_rechanneled"]


def test_unused_source_channels_only_are_allocated():
    midi = MidiFile(1, TimeDivision(480), (
        program_notes().tracks[0],
        MidiTrack.from_events((
            event(0, 0x9, 1, 55, 99),
            event(100, 0x8, 1, 55, 0),
        ), end=600),
    ))
    score, report = from_midi(midi, allow_patch_changes=True)
    assert {(x.channel, x.program) for x in score.tracks} == {(0, 10), (1, 0), (2, 20)}
    assert report["program_channel_reassignments"][0]["output_channel"] == 2


def test_patch_changes_never_reassign_drums_to_pitched_channel():
    with pytest.raises(MidiWorkstationError, match="Changing drum kits"):
        from_midi(program_notes(channel=9), allow_patch_changes=True)


def test_channel_exhaustion_fails_closed():
    tracks = [program_notes().tracks[0]]
    for channel in range(1, 16):
        if channel == 9:
            continue
        tracks.append(MidiTrack.from_events((
            event(0, 0x9, channel, 55, 100),
            event(100, 0x8, channel, 55, 0),
        ), end=600))
    midi = MidiFile(1, TimeDivision(480), tuple(tracks))
    with pytest.raises(MidiLimitError, match="Not enough unoccupied"):
        from_midi(midi, allow_patch_changes=True)


def test_rechanneling_does_not_mutate_binary_midi_source():
    midi = program_notes()
    original = encode_smf(midi)
    a, receipt_a = from_midi(midi, allow_patch_changes=True)
    b, receipt_b = from_midi(midi, allow_patch_changes=True)
    assert a.sha256 == b.sha256
    assert receipt_a == receipt_b
    assert encode_smf(midi) == original


def test_cli_opt_in_and_no_clobber(tmp_path, capsys):
    source = tmp_path / "program-change.mid"
    destination = tmp_path / "score.json"
    source.write_bytes(encode_smf(program_notes()))
    command = ["midi-import", str(source), "--output", str(destination)]
    assert main(command) == 2
    assert not destination.exists()
    capsys.readouterr()
    assert main(command + ["--allow-patch-rechannel"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["conversion"]["source_program_changes_rechanneled"] is True
    assert receipt["conversion"]["program_channel_reassignments"][0]["output_channel"] == 1
    assert main(command + ["--allow-patch-rechannel"]) == 2
    assert capsys.readouterr().out == ""


def test_cli_rejects_drum_rechannel_without_output(tmp_path, capsys):
    source = tmp_path / "drum.mid"
    target = tmp_path / "result.json"
    source.write_bytes(encode_smf(program_notes(channel=9)))
    assert main(["midi-import", str(source), "--output", str(target),
                 "--allow-patch-rechannel"]) == 2
    assert not target.exists()
    assert capsys.readouterr().out == ""
