"""Raw MIDI event transform command with strict local file-publication rules."""
import hashlib
import json
from pathlib import Path

import pytest

from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision
from sonumio.midi_workstation.smf import encode_smf, parse_smf


def source(tmp_path, *, channel=0, pitch=60):
    midi = MidiFile(0, TimeDivision(), (MidiTrack.from_events((
        MidiEvent.meta(0, 0x03, b"Local-original test"),
        MidiEvent.channel(0, 0xC, channel, 3),
        MidiEvent.channel(0, 0x9, channel, pitch, 40),
        MidiEvent.channel(100, 0x8, channel, pitch, 100),
        MidiEvent.sysex(100, 0xF0, b"\x7e\x7f\x09\x01"),
    ), end=120),))
    path = tmp_path / "input.mid"
    path.write_bytes(encode_smf(midi))
    return path, midi


def execute(arguments, capsys):
    status = main(arguments)
    result = capsys.readouterr()
    if status:
        assert result.out == ""
        assert "Sonumio MIDI workstation:" in result.err
        return status, None
    assert result.err == ""
    return status, json.loads(result.out)


def test_cli_transform_roundtrip_preserves_source_and_reports_digest(tmp_path, capsys):
    old, midi = source(tmp_path)
    output = tmp_path / "modified.mid"
    raw = old.read_bytes()
    exitcode, receipt = execute([
        "midi-transform", str(old), "--output", str(output),
        "--semitones", "7", "--velocity-numerator", "3",
        "--velocity-denominator", "2", "--velocity-offset", "-5",
    ], capsys)
    assert exitcode == 0
    assert old.read_bytes() == raw
    changed = parse_smf(output.read_bytes()).midi
    assert [(event.data[0], event.data[1]) for event in changed.events
            if event.kind == 9] == [(67, 55)]
    assert [event.data for event in changed.events if event.is_sysex] == [
        b"\x7e\x7f\x09\x01"]
    assert receipt["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["publication"]["output_sha256"] == receipt["transform"]["result_sha256"]
    assert receipt["transform"]["changed_pitch_events"] == 2
    assert receipt["transform"]["source_mutated"] is False


def test_cli_rejects_existing_target_without_overwrite(tmp_path, capsys):
    old, _ = source(tmp_path)
    output = tmp_path / "modified.mid"
    output.write_bytes(b"user data")
    status, _ = execute(["midi-transform", str(old), "--semitones", "3",
                         "--output", str(output)], capsys)
    assert status == 2
    assert output.read_bytes() == b"user data"


def test_cli_refuses_to_overwrite_input_even_with_overwrite(tmp_path, capsys):
    old, _ = source(tmp_path)
    before = old.read_bytes()
    status, _ = execute(["midi-transform", str(old), "--semitones", "3",
                         "--output", str(old), "--overwrite"], capsys)
    assert status == 2
    assert old.read_bytes() == before


def test_cli_can_overwrite_explicitly(tmp_path, capsys):
    old, _ = source(tmp_path)
    output = tmp_path / "modified.mid"
    output.write_bytes(b"test")
    status, receipt = execute(["midi-transform", str(old), "--semitones", "1",
                               "--output", str(output), "--overwrite"], capsys)
    assert status == 0
    assert parse_smf(output.read_bytes()).midi.tracks[0].note_events[0].data[0] == 61
    assert receipt["publication"]["overwrote"] is True


@pytest.mark.parametrize("arguments", [
    ["--semitones", "49"], ["--semitones", "-49"],
    ["--velocity-numerator", "0"], ["--velocity-denominator", "0"],
    ["--velocity-offset", "127"], ["--pitch-low", "110", "--pitch-high", "109"],
    ["--track", "999"], ["--channel", "16"],
    ["--track", "0", "--transpose-key-signatures"],
    ["--channel", "0", "--transpose-key-signatures"],
])
def test_cli_rejects_invalid_operations_without_publishing(tmp_path, capsys, arguments):
    old, _ = source(tmp_path)
    output = tmp_path / "modified.mid"
    status, _ = execute(["midi-transform", str(old), "--output", str(output),
                         *arguments], capsys)
    assert status == 2
    assert not output.exists()


def test_cli_rejects_wrong_input_sha256_before_publication(tmp_path, capsys):
    old, _ = source(tmp_path)
    output = tmp_path / "modified.mid"
    status, _ = execute(["midi-transform", str(old), "--output", str(output),
                         "--expected-sha256", "0" * 64], capsys)
    assert status == 2
    assert not output.exists()


def test_cli_rejects_wrong_output_sha256_before_publication(tmp_path, capsys):
    old, _ = source(tmp_path)
    output = tmp_path / "modified.mid"
    status, _ = execute(["midi-transform", str(old), "--output", str(output),
                         "--expected-output-sha256", "0" * 64], capsys)
    assert status == 2
    assert not output.exists()


def test_cli_identity_transform_reencodes_to_same_bytes(tmp_path, capsys):
    old, _ = source(tmp_path)
    output = tmp_path / "modified.mid"
    status, receipt = execute(["midi-transform", str(old), "--output", str(output)], capsys)
    assert status == 0
    assert old.read_bytes() == output.read_bytes()
    assert receipt["transform"]["changed_pitch_events"] == 0
    assert receipt["transform"]["changed_velocity_events"] == 0


def test_cli_include_percussion_is_explicit(tmp_path, capsys):
    old, _ = source(tmp_path, channel=9, pitch=36)
    ordinary = tmp_path / "ordinary.mid"
    changed = tmp_path / "changed.mid"
    assert execute(["midi-transform", str(old), "--semitones", "2",
                    "--output", str(ordinary)], capsys)[0] == 0
    assert execute(["midi-transform", str(old), "--semitones", "2",
                    "--include-percussion", "--output", str(changed)], capsys)[0] == 0
    assert parse_smf(ordinary.read_bytes()).midi.tracks[0].note_events[0].data[0] == 36
    assert parse_smf(changed.read_bytes()).midi.tracks[0].note_events[0].data[0] == 38


def test_cli_uses_canonical_selection_order(tmp_path, capsys):
    old, _ = source(tmp_path)
    output = tmp_path / "modified.mid"
    result = execute(["midi-transform", str(old), "--output", str(output),
                      "--track", "0", "--track", "0",
                      "--channel", "0", "--channel", "0",
                      "--semitones", "1"], capsys)
    assert result[0] == 0
    assert result[1]["transform"]["changed_pitch_events"] == 2


def test_cli_binary_symlink_target_not_followed(tmp_path, capsys):
    old, _ = source(tmp_path)
    real = tmp_path / "user.mid"
    real.write_bytes(b"existing")
    alias = tmp_path / "alias.mid"
    try:
        alias.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("Local filesystem disallows symlink creation")
    status, _ = execute(["midi-transform", str(old), "--output", str(alias),
                         "--overwrite"], capsys)
    assert status == 2
    assert real.read_bytes() == b"existing"
