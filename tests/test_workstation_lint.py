"""Bounded, non-mutating musical-semantic diagnostics for valid MIDI streams."""
import random

import pytest

from sonumio.midi_workstation.contracts import MidiWorkstationError
from sonumio.midi_workstation.lint import LintSettings, lint_midi
from sonumio.midi_workstation.model import MidiEvent, MidiFile, MidiTrack, TimeDivision
from sonumio.midi_workstation.smf import encode_smf, parse_smf


def on(tick, channel, pitch=60, velocity=100):
    return MidiEvent.channel(tick, 0x9, channel, pitch, velocity)


def off(tick, channel, pitch=60):
    return MidiEvent.channel(tick, 0x8, channel, pitch, 0)


def cc(tick, channel, controller, value):
    return MidiEvent.channel(tick, 0xB, channel, controller, value)


def program(tick, channel, value):
    return MidiEvent.channel(tick, 0xC, channel, value)


def example(*tracks, format=0, division=TimeDivision()):
    return MidiFile(format, division, tuple(
        MidiTrack.from_events(tuple(events),
                              end=max(480, max((e.tick for e in events), default=0)))
        for events in tracks
    ))


def codes(midi, **kw):
    return lint_midi(midi, LintSettings(**kw)).issue_counts


def test_empty_document_only_has_information_for_default_tempo():
    midi = MidiFile.empty()
    r = lint_midi(midi)
    assert r.status == "clean"
    assert r.issue_counts == {"implicit_initial_tempo": 1}
    assert r.source_events == 1
    assert r.note_ons == r.note_offs == 0


def test_information_can_be_suppressed_without_changing_notes():
    midi = MidiFile.empty()
    assert lint_midi(midi, LintSettings(include_information=False)).total_findings == 0
    assert lint_midi(midi, LintSettings(include_information=True)).total_findings == 1


def test_valid_note_pair_has_no_error_or_warning():
    midi = example((on(0, 0), off(120, 0)))
    r = lint_midi(midi)
    assert r.status == "clean"
    assert r.note_ons == 1 and r.note_offs == 1
    assert r.matched_notes == 1 and r.unclosed_notes == 0


def test_unmatched_key_off_is_error_not_parser_failure():
    midi = example((off(120, 0),))
    r = lint_midi(midi)
    assert r.status == "errors"
    assert r.issue_counts["unmatched_note_off"] == 1
    assert r.matched_notes == 0


def test_unreleased_note_is_error_at_origin_track():
    midi = example((on(10, 0),))
    r = lint_midi(midi)
    assert r.status == "errors"
    assert r.issue_counts["unclosed_note_on"] == 1
    item = next(x for x in r.findings if x.code == "unclosed_note_on")
    assert (item.tick, item.track, item.channel, item.pitch) == (10, 0, 0, 60)


def test_overlap_warning_with_two_matched_notes():
    midi = example((on(0, 0), on(10, 0), off(100, 0), off(150, 0)))
    r = lint_midi(midi)
    assert r.status == "warnings"
    assert r.issue_counts["overlapping_same_pitch"] == 1
    assert r.matched_notes == 2


def test_zero_duration_warning():
    midi = example((on(10, 0), off(10, 0)))
    r = lint_midi(midi)
    assert r.issue_counts["zero_length_note"] == 1
    assert r.status == "warnings"


def test_vel_zero_note_on_is_note_off():
    midi = example((on(0, 0), on(100, 0, 60, 0)))
    r = lint_midi(midi)
    assert r.matched_notes == 1
    assert r.note_ons == r.note_offs == 1


def test_program_change_inside_held_note_warned():
    midi = example((on(0, 0), program(50, 0, 32), off(100, 0)))
    r = lint_midi(midi)
    assert r.issue_counts["patch_change_during_held_notes"] == 1


def test_program_change_outside_held_note_is_safe():
    midi = example((on(0, 0), off(100, 0), program(200, 0, 32)))
    assert "patch_change_during_held_notes" not in codes(midi)


def test_channel_mode_closes_notes_and_reports_controller_ambiguity():
    midi = example((on(0, 0), on(10, 0, 64), cc(200, 0, 123, 0)))
    r = lint_midi(midi)
    assert r.issue_counts["channel_mode_closes_notes"] == 1
    assert r.unclosed_notes == 0


def test_reset_during_sustain_warns_once():
    midi = example((cc(0, 0, 64, 127), cc(120, 0, 121, 0)))
    assert codes(midi)["controllers_reset_while_sustaining"] == 1


def test_sustain_left_down_is_informational():
    midi = example((cc(0, 0, 64, 127),))
    r = lint_midi(midi)
    assert r.issue_counts["sustain_left_down"] == 1
    assert r.status == "clean"


def test_sustain_released_is_not_linted():
    midi = example((cc(0, 0, 64, 127), cc(100, 0, 64, 0)))
    assert "sustain_left_down" not in codes(midi)


def test_format_one_note_off_on_other_track_is_reported_but_matched():
    midi = example((on(0, 0),), (off(100, 0),), format=1)
    r = lint_midi(midi)
    assert r.matched_notes == 1
    assert r.issue_counts["cross_track_note_off"] == 1
    assert r.status == "clean"


def test_format_two_notes_do_not_share_channel_state():
    midi = example((on(0, 0),), (off(100, 0),), format=2)
    r = lint_midi(midi)
    assert r.issue_counts["unclosed_note_on"] == 1
    assert r.issue_counts["unmatched_note_off"] == 1
    assert "cross_track_note_off" not in r.issue_counts


def test_simultaneous_conflicting_tempos_are_errors():
    midi = example((MidiEvent.meta(0, 0x51, b"\x07\xa1\x20"),),
                   (MidiEvent.meta(0, 0x51, b"\x09\x27\xc0"),), format=1)
    assert codes(midi)["conflicting_conductor"] == 1


def test_format_two_independent_tempo_changes_not_conflict():
    midi = example((MidiEvent.meta(0, 0x51, b"\x07\xa1\x20"),),
                   (MidiEvent.meta(0, 0x51, b"\x09\x27\xc0"),), format=2)
    assert "conflicting_conductor" not in codes(midi)


def test_simultaneous_identical_tempo_is_information():
    midi = example((MidiEvent.meta(0, 0x51, b"\x07\xa1\x20"),),
                   (MidiEvent.meta(0, 0x51, b"\x07\xa1\x20"),), format=1)
    r = lint_midi(midi)
    assert r.issue_counts["duplicate_conductor"] == 1
    assert r.status == "clean"


def test_conflicting_meter_is_error():
    midi = example((MidiEvent.meta(0, 0x58, b"\x04\x02\x18\x08"),),
                   (MidiEvent.meta(0, 0x58, b"\x03\x02\x18\x08"),), format=1)
    assert codes(midi)["conflicting_conductor"] == 1


def test_smpte_timing_does_not_claim_implicit_ppq_tempo():
    midi = MidiFile(0, TimeDivision(None, -25, 80), (
        MidiTrack.from_events((), end=0),
    ))
    assert "implicit_initial_tempo" not in codes(midi)


def test_sysex_not_interpreted_and_counted():
    midi = example((MidiEvent.sysex(0, 0xF0, b"\x7e\x7f\x09\x01"),))
    r = lint_midi(midi)
    assert r.sysex_events == 1
    assert r.as_dict()["opaque_sysex_not_interpreted"] is True


def test_bounded_reporting_keeps_global_counts():
    midi = example(tuple(off(i, 0, i % 128) for i in range(100)))
    r = lint_midi(midi, LintSettings(maximum_findings=5))
    assert len(r.findings) == 5
    assert r.total_findings == 101
    assert r.findings_omitted == 96
    assert r.issue_counts["unmatched_note_off"] == 100
    assert r.status == "errors"


def test_tiny_budget_still_reports_actual_issue_counts():
    midi = example(tuple(on(i, 0, i) for i in range(100)))
    r = lint_midi(midi, LintSettings(maximum_findings=1, include_information=False))
    assert r.issue_counts["unclosed_note_on"] == 100
    assert r.total_findings == 100
    assert len(r.findings) == 1


@pytest.mark.parametrize("limit", [0, -1, 2049, True, 1.5, "8"])
def test_invalid_lint_budget_fails_closed(limit):
    with pytest.raises(MidiWorkstationError):
        LintSettings(maximum_findings=limit)


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_include_information_requires_exact_boolean(flag):
    with pytest.raises(MidiWorkstationError):
        LintSettings(include_information=flag)


def test_readonly_digest_and_source_unchanged_even_for_report_limit():
    midi = example((on(0, 0), on(20, 0), off(100, 0), off(110, 0),
                    MidiEvent.sysex(120, 0xF7, b"opaque")))
    before = encode_smf(midi)
    r = lint_midi(midi, LintSettings(maximum_findings=2))
    assert r.midi_sha256 == parse_smf(before).sha256
    assert encode_smf(midi) == before
    assert r.as_dict()["no_input_changes"] is True


def test_finding_order_deterministic_across_repeated_reads():
    midi = example((on(0, 0), on(0, 0), off(0, 0), cc(0, 0, 64, 127)))
    a = lint_midi(midi)
    b = lint_midi(midi)
    assert a.as_dict() == b.as_dict()
    assert a.total_findings >= 3


def test_randomized_valid_smf_diagnostic_roundtrip():
    rng = random.Random(16721)
    for _ in range(150):
        events = []
        tick = 0
        for i in range(rng.randint(3, 25)):
            pitch = rng.randint(45, 90)
            channel = i % 4
            events.extend((on(tick, channel, pitch), off(tick + 10, channel, pitch)))
            tick += 20
        midi = example(events)
        raw = encode_smf(midi)
        report = lint_midi(parse_smf(raw).midi)
        assert report.status == "clean"
        assert report.matched_notes == len(events) // 2
        assert report.midi_sha256 == parse_smf(raw).sha256
