"""Cross-feature regressions from original patterns to inspection and history.

These tests exercise a real multi-module path, not the optional audio-core
synthesizer. MIDI, JSON and symbolic calculations remain local and exact.
"""
from dataclasses import replace
from hashlib import sha256
import json

import pytest

from sonumio.midi_workstation.analysis import score_inspection
from sonumio.midi_workstation.compare import CompareSettings, compare_scores
from sonumio.midi_workstation.contracts import MidiIntegrityError, MidiWorkstationError
from sonumio.midi_workstation.harmony import Scale
from sonumio.midi_workstation.journal import RevisionJournal
from sonumio.midi_workstation.patterns import (
    PatternPlan, PatternCell, PatternVoice, render_patterns,
)
from sonumio.midi_workstation.pianoroll import (
    RollViewport, RollGesture, view_piano_roll, rectangular_selection,
    apply_roll_gesture,
)
from sonumio.midi_workstation.publish import publish_midi, publish_score
from sonumio.midi_workstation.quantize import QuantizeGrid, quantize_score
from sonumio.midi_workstation.regions import (
    TickRegion, crop_score, concatenate_scores,
)
from sonumio.midi_workstation.score import from_midi, to_midi
from sonumio.midi_workstation.score_io import encode_score, load_score
from sonumio.midi_workstation.smf import encode_smf, parse_smf, read_smf
from sonumio.midi_workstation.tempo import TempoMap
from sonumio.midi_workstation.transport import TransportSettings, build_schedule


def pattern():
    melody = PatternVoice("melody", "Original melody", 0, 0, (
        PatternCell(0, 60, duration_steps=2),
        PatternCell(4, 64, duration_steps=2),
        PatternCell(8, 67, duration_steps=2),
        PatternCell(12, 72, duration_steps=2),
    ))
    percussion = PatternVoice("kit", "Percussion", 9, 0, (
        PatternCell(0, 36), PatternCell(4, 38),
        PatternCell(8, 36), PatternCell(12, 42),
    ))
    return render_patterns(PatternPlan(bars=4, bpm=120, seed=17),
                           (melody, percussion))[0]


def test_original_pattern_roundtrip_through_real_midi_parser():
    score = pattern()
    midi = to_midi(score)
    encoded = encode_smf(midi)
    parsed = parse_smf(encoded)
    assert parsed.midi.format == 1
    assert parsed.midi.end == score.end
    assert len(parsed.midi.tracks) >= 3
    imported, diagnostics = from_midi(parsed.midi, title="Roundtrip")
    assert imported.note_count == score.note_count
    assert imported.ppqn == score.ppqn
    assert all(n.start >= 0 and n.stop <= imported.end for n in imported.notes)
    assert score.note_count == 32


def test_score_analysis_and_midi_transport_preserve_clock():
    score = pattern()
    info = score_inspection(score)
    schedule = build_schedule(score, TransportSettings(sample_rate=24000))
    assert schedule.source_sha256 == score.sha256
    assert schedule.output_frames == 4 * 4 * 12000
    assert len(schedule.notes) == score.note_count
    assert schedule.timeline_end_frame == schedule.output_frames
    assert info["protocol"].startswith("sonumio-")


def test_piano_roll_rectangle_and_quantization_cross_pipeline():
    score = pattern()
    view = view_piano_roll(score, RollViewport(0, 1920, 60, 72))
    assert view.visible_note_count == 4
    selected = rectangular_selection(view, first_tick=0, last_tick=960,
                                     lowest_pitch=60, highest_pitch=72)
    assert len(selected) == 2
    gesture = RollGesture(score.sha256, selected, shift_ticks=20)
    edited, receipt = apply_roll_gesture(score, gesture)
    assert receipt.changed_notes == 2
    quantized, report = quantize_score(edited, QuantizeGrid(step_ticks=120))
    assert quantized.sha256 != score.sha256
    assert report.original_sha256 == edited.sha256
    assert len(quantized.notes) == score.note_count


def test_original_score_is_unchanged_across_piano_roll_edit():
    source = pattern()
    before = encode_score(source)
    selection = (("melody", source.tracks[0].notes[0].id),)
    updated, report = apply_roll_gesture(
        source, RollGesture(source.sha256, selection, transpose_semitones=7),
    )
    assert encode_score(source) == before
    assert updated.tracks[0].notes[0].pitch == 67
    assert source.tracks[0].notes[0].pitch == 60
    assert report.as_dict()["no_remote_writes"] is True


def test_score_comparison_with_edits_detects_only_changed_pitch_class():
    source = pattern()
    selection = (("melody", source.tracks[0].notes[0].id),)
    updated, _ = apply_roll_gesture(source, RollGesture(
        source.sha256, selection, transpose_semitones=12,
    ))
    exact = compare_scores(source, updated)
    pitch_class = compare_scores(source, updated,
                                 CompareSettings(pitch_mode="class"))
    assert exact.missing_notes == 1
    assert exact.additional_notes == 1
    assert pitch_class.matched_notes == source.note_count
    assert pitch_class.exact_pitch_matches == source.note_count - 1


def test_crop_then_join_preserves_note_timing_and_tempo():
    source = pattern()
    part1, _ = crop_score(source, TickRegion(0, 3840))
    part2, _ = crop_score(source, TickRegion(3840, 7680))
    joined, report = concatenate_scores((part1, part2))
    assert joined.end == source.end
    assert joined.note_count == source.note_count
    assert joined.duration_microseconds == source.duration_microseconds
    assert report["source_unchanged"] is True


def test_render_save_reload_then_midi_export_pins_both_hashes(tmp_path):
    score = pattern()
    project = tmp_path / "composition.json"
    published = publish_score(project, score)
    assert published.output_sha256 == sha256(project.read_bytes()).hexdigest()
    reloaded = load_score(project, expected_sha256=published.output_sha256)
    assert reloaded.score.sha256 == score.sha256
    destination = tmp_path / "composition.mid"
    midi_pub = publish_midi(destination, to_midi(reloaded.score), input_path=project)
    snapshot = read_smf(destination, expected_sha256=midi_pub.output_sha256)
    assert snapshot.midi.end == score.end
    assert sha256(destination.read_bytes()).hexdigest() == midi_pub.output_sha256


def test_revision_history_recovery_after_pattern_edit():
    score = pattern()
    journal = RevisionJournal.begin(score, label="Original pattern")
    edited, _ = quantize_score(score, QuantizeGrid(step_ticks=120,
                                                    quantize_releases=True))
    journal = journal.commit(edited, "Quantize pattern",
                             expected_current_sha256=score.sha256)
    assert journal.can_undo
    recovered = RevisionJournal.from_bytes(journal.to_bytes())
    assert recovered.undo().current.sha256 == score.sha256
    assert recovered.current.sha256 == edited.sha256
    assert recovered.undo().redo() == journal


def test_revision_history_rejects_tampered_composed_note():
    source = pattern()
    journal = RevisionJournal.begin(source)
    payload = json.loads(journal.to_bytes())
    payload["entries"][0]["score"]["tracks"][0]["notes"][0]["pitch"] = 90
    with pytest.raises(MidiIntegrityError):
        RevisionJournal.from_bytes(json.dumps(payload).encode())


def test_original_pattern_requires_no_external_samples_or_model_weights():
    source = pattern()
    json_data = json.loads(encode_score(source))
    assert not any(k in json_data for k in ("source_wav", "url", "model", "weights"))
    assert all(track["notes"] for track in json_data["tracks"])
    assert sum(len(t["notes"]) for t in json_data["tracks"]) == 32


def test_repeated_identical_score_comparisons_are_stable():
    score = pattern()
    first = compare_scores(score, score)
    second = compare_scores(score, score)
    assert first == second
    assert first.matched_notes == 32
    assert first.missing_notes == 0
    assert first.additional_notes == 0
    assert first.as_dict()["quality_calibrated"] is False
