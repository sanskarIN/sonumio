"""Exhaustive small-lattice oracles and full-score preservation tests."""
from collections import Counter
from dataclasses import replace
import itertools
import json

import pytest

from sonumio.arrangement import (
    ArrangementPlan, Event, compose_arrangement, arrangement_digest, chord_classes, event_key,
)
from sonumio.errors import GenerationError
from sonumio.music import TICKS_PER_BAR, TICKS_PER_BEAT
from sonumio.music_reharmonize import (
    HarmonyPlan, ACCOMPANIMENT, ordered_voicing, melody_mismatch, optimize_harmony,
    reharmonize, encode_harmony_plan, decode_harmony_plan,
)
from sonumio.music_reharmonize_cli import main
from sonumio.music_score_io import encode_score, read_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4,bpm=120,seed=17))


@pytest.mark.parametrize("pitches,classes,shift", [
    (((60,2),(64,1),(67,3)), (0,4,7), 5),
    (((60,1),(62,2),(63,1)), (2,5,9), 7),
    (((24,1),(25,3)), (0,3,7), 6),
    (((95,2),(96,1)), (0,5,9), 10),
    (((60,1),(61,1)), (0,4,7), 0),
    ((), (0,4,7), 0),
])
def test_voicing_matches_exhaustive_oracle(pitches, classes, shift):
    candidates = [tuple(p for p in range(24,97) if p%12 in classes and abs(p-old)<=shift)
                  for old,_ in pitches]
    valid = []
    for path in itertools.product(*candidates):
        if any(a>=b for a,b in zip(path,path[1:])):
            continue
        cost = sum(abs(p-old)*count for p,(old,count) in zip(path,pitches,strict=True))
        valid.append((cost,path))
    mapping,cost = ordered_voicing(pitches,classes,shift)
    if not valid:
        assert mapping is None and cost is None
    else:
        expected_cost,path = min(valid)
        assert cost == expected_cost
        assert mapping == dict(zip((p for p,_ in pitches),path,strict=True))


@pytest.mark.parametrize("pitches,classes,shift", [
    (((60,0),), (0,4,7), 12),
    (((60,1),(60,2)), (0,4,7), 12),
    (((61,1),(60,1)), (0,4,7), 12),
    (((60,1),), (0,0,4), 12),
    (((60,1),), (0,4,12), 12),
    (((60,1),), (0,4,7), True),
])
def test_invalid_voicing_inputs(pitches,classes,shift):
    with pytest.raises(GenerationError):
        ordered_voicing(pitches,classes,shift)


def _transition(score,plan,a,b):
    common = len(set(chord_classes(score.plan,a)).intersection(chord_classes(score.plan,b)))
    return plan.movement_weight*(3-common+min((a-b)%7,(b-a)%7))


def test_harmony_search_matches_exhaustive_degree_paths(score):
    plan = HarmonyPlan(arrangement_digest(score),degrees=(0,2,4,5),
                       melody_weight=3,identity_weight=1,voice_weight=1,movement_weight=2)
    actual,cost,candidates,_ = optimize_harmony(score,plan)
    rows = [[degree for bar,degree in candidates if bar==i] for i in range(4)]
    oracle = []
    for path in itertools.product(*rows):
        total = sum(candidates[i,d][2] for i,d in enumerate(path))
        total += sum(_transition(score,plan,a,b) for a,b in zip(path,path[1:]))
        oracle.append((total,path))
    assert (cost,actual)==min(oracle)
    assert actual[-1]==0


def test_subregion_accounts_for_boundary_transition(score):
    plan = HarmonyPlan(arrangement_digest(score),first_bar=1,bars=2,degrees=(0,3,4))
    path,cost,candidates,_ = optimize_harmony(score,plan)
    valid = []
    for a,b in itertools.product((0,3,4),repeat=2):
        if (1,a) not in candidates or (2,b) not in candidates:
            continue
        c = candidates[1,a][2]+candidates[2,b][2]
        c += _transition(score,plan,score.harmony[0],a)
        c += _transition(score,plan,a,b)+_transition(score,plan,b,score.harmony[3])
        valid.append((c,(a,b)))
    assert (cost,path)==min(valid)


def test_melody_mismatch_counts_overlap_and_beat_emphasis(score):
    # B is not in C-minor tonic but is in some other diatonic modes;
    # use D (in C-minor scale, not in tonic chord).
    event = Event("lead",62,TICKS_PER_BAR-4,8,.5,0.)
    small = replace(score,events=(event,))
    assert melody_mismatch(small,0,0)==4
    assert melody_mismatch(small,1,0)==4
    strong = replace(small,events=(replace(event,tick=TICKS_PER_BEAT,length=3),))
    assert melody_mismatch(strong,0,0)==3+TICKS_PER_BEAT


def test_preserves_nonpitch_geometry_and_lead_drums(score):
    plan = HarmonyPlan(arrangement_digest(score),degrees=(0,),maximum_shift=12)
    result,report = reharmonize(score,plan)
    assert result.harmony==(0,0,0,0)
    assert result.plan==score.plan and result.sections==score.sections
    def geometry(e):
        return (e.track,e.tick,e.length,e.velocity,e.pan)
    assert Counter(map(geometry,result.events))==Counter(map(geometry,score.events))
    assert [e for e in result.events if e.track not in ACCOMPANIMENT]==[
        e for e in score.events if e.track not in ACCOMPANIMENT]
    for event in result.events:
        bar = event.tick//TICKS_PER_BAR
        if event.track in ACCOMPANIMENT and score.harmony[bar]!=0:
            assert event.pitch%12 in chord_classes(score.plan,0)
    assert report["result_score_sha256"]==arrangement_digest(result)
    assert report["changed_bars"]==sum(d!=0 for d in score.harmony)
    assert reharmonize(score,plan)==(result,report)


def test_unchanged_bar_never_rewrites_custom_accompaniment(score):
    event = Event("keys",62,0,4,.5,0.)
    small = replace(score,harmony=(0,0,0,0),events=(event,))
    result,report = reharmonize(small,HarmonyPlan(arrangement_digest(small),degrees=(0,)))
    assert result==small and report["changed_events"]==0


def test_crossing_accompaniment_forbids_changed_chord(score):
    event = Event("pad",60,TICKS_PER_BAR-4,8,.5,0.)
    small = replace(score,harmony=(0,0,0,0),events=(event,))
    plan = HarmonyPlan(arrangement_digest(small),first_bar=1,bars=1,degrees=(4,))
    with pytest.raises(GenerationError,match="No feasible"):
        reharmonize(small,plan)


def test_preserved_edges_and_dominant_cadence(score):
    plan = HarmonyPlan(arrangement_digest(score),preserve_edges=True,dominant_cadence=True)
    result,_ = reharmonize(score,plan)
    assert result.harmony[0]==score.harmony[0]
    assert result.harmony[-2:]==(4,0)


def test_incompatible_constraints_rejected(score):
    with pytest.raises(GenerationError,match="No feasible"):
        reharmonize(score,HarmonyPlan(arrangement_digest(score),degrees=(4,)))
    with pytest.raises(GenerationError,match="SHA256"):
        reharmonize(score,HarmonyPlan("0"*64))
    with pytest.raises(GenerationError,match="exceeds"):
        reharmonize(score,HarmonyPlan(arrangement_digest(score),first_bar=2,bars=3))


@pytest.mark.parametrize("changes",[
    {"degrees":(2,1)},{"degrees":(0,0)},{"degrees":(True,)},
    {"maximum_shift":25},{"bars":0},{"preserve_edges":1},
    {"melody_weight":0,"identity_weight":0,"voice_weight":0,"movement_weight":0},
])
def test_invalid_plan(score,changes):
    with pytest.raises(GenerationError):
        HarmonyPlan(arrangement_digest(score),**changes)


def test_document_roundtrip_and_duplicate_fields(score):
    plan=HarmonyPlan(arrangement_digest(score))
    assert decode_harmony_plan(encode_harmony_plan(plan))==plan
    with pytest.raises(GenerationError):
        decode_harmony_plan(b'{"format":1,"format":2}')


def test_cli_atomic_publish_and_inspect(score,tmp_path,capsys):
    source=tmp_path/"score.json"
    config=tmp_path/"harmony.json"
    output=tmp_path/"out.json"
    source.write_bytes(encode_score(score))
    plan=HarmonyPlan(arrangement_digest(score),degrees=(0,))
    config.write_bytes(encode_harmony_plan(plan))
    assert main(["inspect",str(source),str(config)])==0
    expected=json.loads(capsys.readouterr().out)
    args=["apply",str(source),str(config),"--output",str(output)]
    assert main(args+["--expected-plan-sha256","0"*64])==2
    assert not output.exists()
    assert main(args+["--expected-plan-sha256",plan.sha256])==0
    actual=json.loads(capsys.readouterr().out)
    assert actual["result_score_sha256"]==expected["result_score_sha256"]
    assert arrangement_digest(read_score(output))==actual["result_score_sha256"]
    assert main(args)==2
    assert main(["apply",str(source),str(config),"--output",str(source),"--overwrite"])==2
