"""Test strict bridge against Sonumio's ACTUAL original arrangement model."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

pytest.importorskip("sonumio.arrangement")
from sonumio.arrangement import (
    ArrangementPlan, Event, arrangement_digest, compose_arrangement,
)
from sonumio.music_score_io import encode_score as encode_legacy_score, decode_score as decode_legacy_score
from sonumio.midi_workstation.arrangement_bridge import (
    ArrangementRoute, ArrangementBridgePolicy, bridge_to_arrangement,
)
from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack, note_sort_key
from sonumio.midi_workstation.score_io import encode_score as encode_workstation_score
from sonumio.midi_workstation.cli import main as midi_main
from sonumio.midi_workstation.tempo import MeterPoint, TempoPoint


def template(*, bpm=120, bars=4, swing=0., root=60, mode="major"):
    return compose_arrangement(ArrangementPlan(
        style="calm", bpm=bpm, bars=bars, swing=swing,
        root=root, mode=mode, seed=2026,
    ))


def note(identifier="one", *, start=0, duration=480, pitch=60,
         velocity=101, pan=64):
    return ScoreNote(identifier, start, duration, pitch, velocity,
                     release_velocity=32, pan=pan)


def score_of(*notes, plan=None, track_id="melody", channel=0, muted=False,
             ppqn=480, program=0, solo=False):
    plan = template() if plan is None else plan
    score = Score.empty(ppqn=ppqn, end=plan.plan.bars*4*ppqn, title="Original test")
    track = ScoreTrack(track_id, "Melody", channel, program, 0, 0,
                       tuple(sorted(notes, key=note_sort_key)), muted, solo)
    return replace(score, tracks=(track,),
                   tempo=(TempoPoint(0, (60_000_000+plan.plan.bpm//2)//plan.plan.bpm),))


def policy(*routes, include_muted=False, preserve_note_pan=True, maximum_events=4096):
    return ArrangementBridgePolicy(
        tuple(sorted(routes or (ArrangementRoute("melody", "lead"),),
                     key=lambda item:item.source_track_id)),
        include_muted=include_muted, preserve_note_pan=preserve_note_pan,
        maximum_events=maximum_events,
    )


def test_convert_note_preserves_template_harmony_and_sections():
    initial = template()
    source = score_of(note())
    initial_id, before = arrangement_digest(initial), source.sha256
    converted, report = bridge_to_arrangement(source, initial, policy())
    assert len(converted.events) == 1
    assert converted.events[0] == Event("lead", 60, 0, 8, 101/127, 0.)
    assert converted.plan == initial.plan
    assert converted.sections == initial.sections
    assert converted.harmony == initial.harmony
    assert arrangement_digest(initial) == initial_id
    assert source.sha256 == before
    assert report["template_arrangement_sha256"] == initial_id
    assert report["source_score_sha256"] == before
    assert report["output_arrangement_sha256"] == arrangement_digest(converted)
    assert report["exact_tick_grid"] is True
    assert report["exact_original_score_reconstructed"] is False
    assert "release_velocity" in report["discarded_expression"]


def test_canonical_legacy_score_can_be_encoded_and_redecoded():
    plan = template()
    source = score_of(note())
    converted, _ = bridge_to_arrangement(source, plan, policy())
    raw = encode_legacy_score(converted)
    assert decode_legacy_score(raw) == converted


def test_grid_mapping_allows_note_end_at_score_boundary():
    plan = template()
    end = plan.plan.bars*4*480
    score = score_of(note(start=end-480, duration=480))
    converted, _ = bridge_to_arrangement(score, plan, policy())
    assert converted.events[0].tick == plan.plan.ticks - 8
    assert converted.events[0].length == 8


def test_off_grid_start_rejected_without_quantizing():
    with pytest.raises(MidiWorkstationError, match="off arrangement eighth-beat grid"):
        bridge_to_arrangement(score_of(note(start=7)), template(), policy())


def test_off_grid_stop_rejected_without_quantizing():
    with pytest.raises(MidiWorkstationError, match="off arrangement eighth-beat grid"):
        bridge_to_arrangement(score_of(note(duration=481)), template(), policy())


def test_score_duration_must_exactly_match_template():
    plan = template()
    score = replace(score_of(note(), plan=plan), end=plan.plan.bars*4*480-480)
    with pytest.raises(MidiWorkstationError, match="identical exact musical duration"):
        bridge_to_arrangement(score, plan, policy())


def test_tempo_changes_are_refused_even_if_last_tempo_matches():
    score = replace(score_of(note()), tempo=(TempoPoint(0,500000),TempoPoint(240,500000)))
    with pytest.raises(MidiWorkstationError, match="tempo must exactly"):
        bridge_to_arrangement(score, template(), policy())


def test_non_44_meter_rejected():
    score = replace(score_of(note()), meter=(MeterPoint(0, 3, 4),))
    with pytest.raises(MidiWorkstationError, match="meter must be constant 4/4"):
        bridge_to_arrangement(score, template(), policy())


def test_swing_template_rejected_instead_of_silent_time_change():
    plan = template(swing=.3)
    score = score_of(note(), plan=plan)
    with pytest.raises(MidiWorkstationError, match="swing cannot be represented"):
        bridge_to_arrangement(score, plan, policy())


def test_out_of_scale_note_rejected_by_existing_arrangement_contract():
    with pytest.raises(MidiWorkstationError, match="legacy composition constraints"):
        bridge_to_arrangement(score_of(note(pitch=61)), template(), policy())


def test_arrangement_polyphony_limit_is_enforced():
    notes = tuple(note(f"n{i}", pitch=pitch) for i,pitch in enumerate((60,64,67,72,76)))
    with pytest.raises(MidiWorkstationError, match="legacy composition constraints"):
        bridge_to_arrangement(score_of(*notes), template(), policy())


def test_arrangement_duplicate_note_value_cannot_be_disguised_by_ids():
    a,b=note("a"),note("b")
    with pytest.raises(MidiWorkstationError, match="legacy composition constraints"):
        bridge_to_arrangement(score_of(a,b),template(),policy())


def test_pitch_24_to_96_bounds_enforced_before_arrangement_construction():
    with pytest.raises(MidiWorkstationError, match="24..96"):
        bridge_to_arrangement(score_of(note(pitch=23)),template(),policy())


def test_drum_channel_cannot_map_to_pitched_instrument():
    score=score_of(note(pitch=36),channel=9)
    with pytest.raises(MidiWorkstationError, match="Percussion MIDI channel"):
        bridge_to_arrangement(score,template(),policy())


def test_pitched_channel_cannot_map_to_legacy_drum():
    score=score_of(note())
    with pytest.raises(MidiWorkstationError, match="Pitched MIDI channel"):
        bridge_to_arrangement(score,template(),policy(ArrangementRoute("melody","kick")))


def test_drum_channel_can_map_to_snare_and_keeps_pitch():
    score=score_of(note(pitch=38),channel=9)
    converted, report=bridge_to_arrangement(score,template(),policy(ArrangementRoute("melody","snare")))
    assert converted.events[0].track=="snare"
    assert converted.events[0].pitch==38
    assert report["events_by_instrument"]["snare"]==1


def test_pan_left_center_right_and_override():
    notes=(note("a",start=0,pan=0),note("b",start=480,pan=64),
           note("c",start=960,pan=127))
    score=score_of(*notes)
    original,_=bridge_to_arrangement(score,template(),policy())
    assert [event.pan for event in original.events]==[-1.,0.,1.]
    centered,report=bridge_to_arrangement(score,template(),policy(preserve_note_pan=False))
    assert [event.pan for event in centered.events]==[0.,0.,0.]
    assert report["pan_source"]=="explicit centered override"


def test_velocity_from_midi_7bit_is_explicit():
    score=score_of(note(velocity=127))
    converted,report=bridge_to_arrangement(score,template(),policy())
    assert converted.events[0].velocity==1.
    assert "127" in report["velocity_source"]


def test_route_exactly_covers_included_note_tracks():
    source=score_of(note())
    with pytest.raises(MidiWorkstationError, match="unknown source track"):
        bridge_to_arrangement(source,template(),policy(ArrangementRoute("other","lead")))
    with pytest.raises(MidiWorkstationError, match="unique and sorted"):
        ArrangementBridgePolicy((ArrangementRoute("melody","lead"),
                                 ArrangementRoute("melody","keys")))


def test_muted_track_excluded_by_default_but_optional():
    plan=template()
    original=score_of(note(),plan=plan,track_id="active")
    muted=ScoreTrack("mute","Muted",1,0,0,0,(note("hidden",start=480),),True)
    original=replace(original,tracks=(original.tracks[0],muted))
    route=policy(ArrangementRoute("active","lead"))
    converted,report=bridge_to_arrangement(original,plan,route)
    assert len(converted.events)==1
    assert report["skipped_muted_notes"]==1
    included,report=bridge_to_arrangement(original,plan,policy(
        ArrangementRoute("active","lead"),ArrangementRoute("mute","keys"),include_muted=True))
    assert len(included.events)==2
    assert report["skipped_muted_notes"]==0


def test_note_budget_prevents_expensive_arrangement_validation():
    score=score_of(note(),note("later",start=480))
    with pytest.raises(MidiLimitError,match="event budget"):
        bridge_to_arrangement(score,template(),policy(maximum_events=1))


def test_transpose_must_be_explicit_and_result_in_scale():
    source=score_of(note(pitch=60))
    converted,_=bridge_to_arrangement(source,template(),policy(ArrangementRoute("melody","lead",4)))
    assert converted.events[0].pitch==64
    with pytest.raises(MidiWorkstationError,match="legacy composition constraints"):
        bridge_to_arrangement(source,template(),policy(ArrangementRoute("melody","lead",1)))
    with pytest.raises(MidiWorkstationError):
        ArrangementRoute("melody","lead",25)


def test_cli_atomic_publication_and_digest_snapshot(tmp_path,capsys):
    source=score_of(note())
    template_score=template()
    source_file=tmp_path/"source.json"
    template_file=tmp_path/"template.json"
    destination=tmp_path/"legacy.json"
    source_file.write_bytes(encode_workstation_score(source))
    template_file.write_bytes(encode_legacy_score(template_score))
    command=["arrangement-export",str(source_file),"--template",str(template_file),
             "--route","melody:lead","--output",str(destination)]
    assert midi_main(command)==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["bridge"]["events"]==1
    assert receipt["bridge"]["source_score_sha256"]==source.sha256
    output=decode_legacy_score(destination.read_bytes())
    assert output.plan==template_score.plan
    assert output.sections==template_score.sections
    assert midi_main(command)==2  # no clobber
    assert capsys.readouterr().out==""


def test_cli_refuses_template_as_output_even_with_overwrite(tmp_path,capsys):
    score=score_of(note())
    plan=template()
    source=tmp_path/"score.json"
    target=tmp_path/"template.json"
    source.write_bytes(encode_workstation_score(score))
    old=encode_legacy_score(plan)
    target.write_bytes(old)
    assert midi_main(["arrangement-export",str(source),"--template",str(target),
                      "--route","melody:lead","--output",str(target),
                      "--overwrite"])==2
    assert target.read_bytes()==old
    assert capsys.readouterr().out==""


def test_cli_wrong_template_sha_never_publishes(tmp_path,capsys):
    source=tmp_path/"score.json"
    template_path=tmp_path/"template.json"
    output=tmp_path/"out.json"
    source.write_bytes(encode_workstation_score(score_of(note())))
    template_path.write_bytes(encode_legacy_score(template()))
    assert midi_main(["arrangement-export",str(source),
                      "--template",str(template_path),"--route","melody:lead",
                      "--output",str(output),"--expected-template-sha256","f"*64])==2
    assert not output.exists()
    assert capsys.readouterr().out==""


def test_cli_output_digest_gate_fails_before_file_creation(tmp_path,capsys):
    source=tmp_path/"score.json"
    template_path=tmp_path/"template.json"
    output=tmp_path/"out.json"
    source.write_bytes(encode_workstation_score(score_of(note())))
    template_path.write_bytes(encode_legacy_score(template()))
    assert midi_main(["arrangement-export",str(source),
                      "--template",str(template_path),"--route","melody:lead",
                      "--output",str(output),"--expected-output-sha256","e"*64])==2
    assert not output.exists()
    assert capsys.readouterr().out==""


def test_cli_rejects_symlink_template(tmp_path,capsys):
    source=tmp_path/"score.json"
    original=tmp_path/"actual.json"
    alias=tmp_path/"template.json"
    target=tmp_path/"result.json"
    source.write_bytes(encode_workstation_score(score_of(note())))
    original.write_bytes(encode_legacy_score(template()))
    try:
        alias.symlink_to(original)
    except (OSError,NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert midi_main(["arrangement-export",str(source),"--template",str(alias),
                      "--route","melody:lead","--output",str(target)])==2
    assert not target.exists()
    assert capsys.readouterr().out==""
