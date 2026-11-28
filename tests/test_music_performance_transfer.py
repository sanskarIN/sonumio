"""Event mapping, conflict policy and combined sample geometry tests."""
from dataclasses import replace
import json

import pytest

from sonumio.arrangement import ArrangementPlan, Event, arrangement_digest, compose_arrangement,event_key
from sonumio.errors import GenerationError
from sonumio.music_performance import (
    Performance,Expression,compile_performance,encode_performance,read_performance,
)
from sonumio.music_performance_transfer import (
    TransferPolicy,transfer_performance,verify_transfer,merge_performances,
)
from sonumio.music_performance_transfer_cli import main
from sonumio.music_score_io import encode_score


@pytest.fixture
def score():
    plan=ArrangementPlan(bars=4,bpm=120,seed=17)
    events=(Event("lead",60,8,4,.5,0.),Event("lead",62,16,4,.5,0.),
            Event("lead",63,24,4,.5,0.))
    return replace(compose_arrangement(plan),events=events)


def overlay(score,*rows):
    return Performance(arrangement_digest(score),rows)


def test_shifted_canonical_index_keeps_event_identity(score):
    original=overlay(score,Expression(1,20,10,.8,.1))
    added=Event("keys",60,0,4,.5,0.)
    target=replace(score,events=tuple(sorted((*score.events,added),key=event_key)))
    result,report=transfer_performance(score,target,original)
    assert result.expressions==(Expression(2,20,10,.8,.1),)
    assert result.score_sha256==arrangement_digest(target)
    mapping,=report["mappings"]
    assert mapping["source_index"]==1 and mapping["target_index"]==2
    assert mapping["source_performed_frames"]==mapping["target_performed_frames"]
    assert report["result_validated"]
    assert verify_transfer(score,target,original,result,report)==report


def test_exact_rejects_velocity_change_but_geometry_preserves_relative_expression(score):
    original=overlay(score,Expression(0,0,0,.5,.1))
    target=replace(score,events=(replace(score.events[0],velocity=.8,pan=.2),*score.events[1:]))
    result,report=transfer_performance(score,target,original)
    assert result is None and report["conflicts"]
    result,report=transfer_performance(score,target,original,TransferPolicy(match="geometry"))
    assert result.expressions==original.expressions
    note=compile_performance(target,result)[0]
    assert note.velocity==pytest.approx(.4) and note.pan==pytest.approx(.3)
    row,=report["mappings"]
    assert row["base_velocity_changed"] and row["base_pan_changed"]


def test_removed_note_requires_explicit_drop(score):
    original=overlay(score,Expression(0,20),Expression(2,30))
    target=replace(score,events=score.events[1:])
    result,report=transfer_performance(score,target,original)
    assert result is None and report["dropped"]==0
    result,report=transfer_performance(score,target,original,TransferPolicy(missing="drop"))
    assert result.expressions==(Expression(1,30),)
    assert report["dropped"]==1 and report["transferred"]==1
    assert report["result_validated"]


def test_clock_change_requires_sample_offset_policy(score):
    original=overlay(score,Expression(1,20,30))
    target=replace(score,plan=replace(score.plan,bpm=100))
    result,report=transfer_performance(score,target,original)
    assert result is None and report["clock_changed"]
    result,report=transfer_performance(score,target,original,TransferPolicy(clock="sample-offset"))
    assert result.expressions==original.expressions
    row,=report["mappings"]
    assert row["source_base_frames"]!=row["target_base_frames"]
    assert row["target_performed_frames"][0]-row["target_base_frames"][0]==20


def test_new_event_conflicts_with_transferred_timing(score):
    # The original first note ends at tick 12; an adjacent new note
    # makes an existing release extension invalid on the target.
    original=overlay(score,Expression(0,end_delta=100))
    added=Event("lead",65,12,1,.5,0.)
    target=replace(score,events=tuple(sorted((*score.events,added),key=event_key)))
    result,report=transfer_performance(score,target,original)
    assert result is None
    assert report["conflicts"][0]["domain"]=="target_performance"
    assert "polyphony" in report["conflicts"][0]["reason"]


def test_stale_source_pin_rejected(score):
    with pytest.raises(GenerationError,match="SHA256"):
        transfer_performance(score,score,Performance("0"*64))


@pytest.mark.parametrize("changes",[
    {"match":"nearest"},{"missing":"ignore"},{"clock":"scale"},{"match":True},
])
def test_policy_strict(changes):
    with pytest.raises(GenerationError):
        TransferPolicy(**changes)


def test_deterministic_empty_transfer(score):
    original=overlay(score)
    result,report=transfer_performance(score,score,original)
    assert result==original and report["transferred"]==0
    assert transfer_performance(score,score,original)==(result,report)
    with pytest.raises(GenerationError,match="receipt"):
        verify_transfer(score,score,original,result,{**report,"authenticated":True})


def test_disjoint_expression_merge(score):
    base=overlay(score)
    left=overlay(score,Expression(0,10))
    right=overlay(score,Expression(2,0,0,.8))
    result,report=merge_performances(score,base,left,right)
    assert result.expressions==(left.expressions[0],right.expressions[0])
    assert report["combined_validated"]
    assert [r["origin"] for r in report["decisions"]]==["left","right"]


def test_removal_and_identical_edits_merge(score):
    base=overlay(score,Expression(0,10))
    left=overlay(score)
    right=overlay(score,Expression(0,10),Expression(2,0,0,.8))
    result,report=merge_performances(score,base,left,right)
    assert result.expressions==(right.expressions[1],)
    assert report["decisions"][0]["removed"]
    result,report=merge_performances(score,base,right,right)
    assert result==right


def test_conflicting_expression_edits_fail(score):
    base=overlay(score)
    left=overlay(score,Expression(0,10))
    right=overlay(score,Expression(0,0,0,.8))
    result,report=merge_performances(score,base,left,right)
    assert result is None and report["conflicts"][0]["index"]==0


def test_individually_valid_timing_edits_can_conflict_when_combined(score):
    # At 160 BPM, a one-tick gap is 1125 frames; moving either
    # endpoint by 700 frames is safe individually but not together.
    score=replace(score,plan=replace(score.plan,bpm=160),events=(
        Event("lead",60,4,4,.5,0.),Event("lead",62,9,4,.5,0.)))
    base=overlay(score)
    left=overlay(score,Expression(0,end_delta=700))
    right=overlay(score,Expression(1,onset_delta=-700))
    compile_performance(score,left)
    compile_performance(score,right)
    result,report=merge_performances(score,base,left,right)
    assert result is None
    assert report["conflicts"][0]["domain"]=="combined_performance"


def test_cli_pins_conflicts_and_publication(score,tmp_path,capsys):
    source=tmp_path/"source.json"
    target=tmp_path/"target.json"
    performance=tmp_path/"performance.json"
    output=tmp_path/"result.json"
    source.write_bytes(encode_score(score))
    target.write_bytes(encode_score(replace(score,events=score.events[1:])))
    original=overlay(score,Expression(0,10),Expression(2,10))
    performance.write_bytes(encode_performance(original))
    args=[str(source),str(target),str(performance)]
    assert main(["transfer",*args,"--output",str(output)])==3
    assert not output.exists()
    assert main(["preview-transfer",*args,"--missing","drop"])==0
    expected=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert main(["transfer",*args,"--missing","drop","--output",str(output),
                 "--expected-performance-sha256",original.sha256])==0
    assert read_performance(output).sha256==expected["result_performance_sha256"]
    assert main(["transfer",*args,"--missing","drop","--output",str(output)])==2
    assert main(["transfer",*args,"--missing","drop","--output",str(source),"--overwrite"])==2
