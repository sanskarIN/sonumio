"""CLI gates, bounded JSON diagnostics and non-destructive MIDI auditing."""
import hashlib
import json

import pytest

from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision
from sonumio.midi_workstation.smf import encode_smf


def midi_file(tmp_path, events, *, name="input.mid"):
    midi = MidiFile(0, TimeDivision(480), (
        MidiTrack.from_events(events, end=480),
    ))
    source = tmp_path / name
    source.write_bytes(encode_smf(midi))
    return source


def call(args, capsys):
    status = main(args)
    seen = capsys.readouterr()
    if status:
        assert not seen.out
        assert "Sonumio MIDI workstation:" in seen.err
        return status, None
    assert not seen.err
    return status, json.loads(seen.out)


def pair():
    return (MidiEvent.channel(0, 0x9, 0, 60, 110),
            MidiEvent.channel(100, 0x8, 0, 60, 0))


def test_cli_inspects_clean_source_without_writing(tmp_path, capsys):
    source = midi_file(tmp_path, pair())
    baseline = source.read_bytes()
    status, report = call(["midi-lint", str(source)], capsys)
    assert status == 0
    assert report["lint"]["status"] == "clean"
    assert report["lint"]["matched_notes"] == 1
    assert report["source_sha256"] == hashlib.sha256(baseline).hexdigest()
    assert source.read_bytes() == baseline
    assert list(tmp_path.iterdir()) == [source]


def test_cli_error_report_is_returned_as_data_by_default(tmp_path, capsys):
    source = midi_file(tmp_path, (MidiEvent.channel(0, 0x8, 0, 60, 0),))
    status, report = call(["midi-lint", str(source)], capsys)
    assert status == 0
    assert report["lint"]["status"] == "errors"
    assert report["lint"]["issue_counts"]["unmatched_note_off"] == 1


def test_cli_lint_error_gate_has_nonzero_status_no_success_receipt(tmp_path, capsys):
    source = midi_file(tmp_path, (MidiEvent.channel(0, 0x8, 0, 60, 0),))
    status, report = call(["midi-lint", str(source), "--fail-on-errors"], capsys)
    assert status == 2 and report is None


def test_cli_warnings_gate_does_not_reject_information_only(tmp_path, capsys):
    source = midi_file(tmp_path, pair())
    assert call(["midi-lint", str(source), "--fail-on-warnings"], capsys)[0] == 0


def test_cli_warnings_gate_rejects_overlap(tmp_path, capsys):
    events = (
        MidiEvent.channel(0, 0x9, 0, 60, 80),
        MidiEvent.channel(10, 0x9, 0, 60, 90),
        MidiEvent.channel(100, 0x8, 0, 60, 0),
        MidiEvent.channel(120, 0x8, 0, 60, 0),
    )
    source = midi_file(tmp_path, events)
    status, report = call(["midi-lint", str(source), "--fail-on-warnings"], capsys)
    assert status == 2 and report is None
    status, report = call(["midi-lint", str(source), "--fail-on-errors"], capsys)
    assert status == 0
    assert report["lint"]["status"] == "warnings"


def test_cli_suppresses_info_and_reports_clean_no_issues(tmp_path, capsys):
    source = midi_file(tmp_path, pair())
    status, report = call(["midi-lint", str(source), "--no-information"], capsys)
    assert status == 0
    assert report["lint"]["total_findings"] == 0
    assert report["lint"]["findings"] == []


def test_cli_report_limit_keeps_totals_accurate(tmp_path, capsys):
    events = tuple(MidiEvent.channel(i, 0x8, 0, 60 + (i % 12), 0)
                   for i in range(90))
    source = midi_file(tmp_path, events)
    status, report = call(["midi-lint", str(source), "--max-findings", "2"], capsys)
    assert status == 0
    assert report["lint"]["total_findings"] == 91
    assert report["lint"]["findings_omitted"] == 89
    assert len(report["lint"]["findings"]) == 2


@pytest.mark.parametrize("limit", ["-1", "0", "2049"])
def test_cli_rejects_out_of_range_report_budget(tmp_path, capsys, limit):
    source = midi_file(tmp_path, pair())
    assert call(["midi-lint", str(source), "--max-findings", limit], capsys)[0] == 2


def test_cli_checksum_mismatch_stops_before_audit(tmp_path, capsys):
    source = midi_file(tmp_path, pair())
    status, report = call(["midi-lint", str(source),
                           "--expected-sha256", "0" * 64], capsys)
    assert status == 2 and report is None


def test_cli_many_repeated_runs_return_identical_json(tmp_path, capsys):
    source = midi_file(tmp_path, pair())
    a = call(["midi-lint", str(source)], capsys)
    b = call(["midi-lint", str(source)], capsys)
    assert a == b


def test_cli_rejects_symlinked_source(tmp_path, capsys):
    source = midi_file(tmp_path, pair())
    alias = tmp_path / "alias.mid"
    try:
        alias.symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable")
    status, _ = call(["midi-lint", str(alias)], capsys)
    assert status == 2


def test_subprocess_like_lint_usage_is_documented_by_parser(capsys):
    # This is not a full process spawn: argparse's help text is validated
    # without depending on external package installation or entrypoints.
    from sonumio.midi_workstation.cli import make_parser
    help_message = make_parser().format_help()
    assert "midi-lint" in help_message
    assert "midi-transform" in help_message
