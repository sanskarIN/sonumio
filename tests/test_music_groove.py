"""Exact groove optimization against exhaustive independent enumeration."""
from dataclasses import replace
from itertools import combinations
import hashlib
import json
import random

import pytest

from sonumio.arrangement import (
    ArrangementPlan, Event, PITCHED, compose_arrangement, arrangement_digest, event_key,
)
from sonumio.errors import GenerationError
from sonumio.music_groove_solver import (
    GrooveConstraints,GrooveObjective,solve_groove,validate_pattern,pattern_cost,SLOTS,
)
from sonumio.music_groove import (
    DrumRule,GroovePlan,encode_groove,decode_groove,generate_groove,MAX_BYTES,
)
from sonumio.music_groove_cli import main
from sonumio.music_score_io import encode_score,decode_score


def oracle(constraints,objective):
    candidates=[]
    for pattern in combinations(range(16),constraints.hits):
        if not set(constraints.mandatory)<=set(pattern) or set(constraints.forbidden)&set(pattern):
            continue
        gaps=[b-a for a,b in zip(pattern,pattern[1:])]
        if pattern and constraints.cyclic:
            gaps.append(pattern[0]+16-pattern[-1])
        if not all(constraints.minimum_gap<=gap<=constraints.maximum_gap for gap in gaps):
            continue
        cost=sum(objective.slot_costs[p] for p in pattern)
        cost+=sum(objective.gap_weight*abs(gap*constraints.hits-16) for gap in gaps)
        candidates.append((cost,pattern))
    return min(candidates) if candidates else None


@pytest.mark.parametrize("seed",range(16))
def test_solver_matches_exhaustive_independent_oracle(seed):
    rng=random.Random(seed)
    mandatory=tuple(sorted(rng.sample(range(16),seed%3)))
    forbidden=tuple(sorted(rng.sample([s for s in range(16) if s not in mandatory],seed%4)))
    hits=rng.randint(len(mandatory),min(8,16-len(forbidden)))
    low=rng.randint(1,3)
    high=rng.randint(low,16)
    constraints=GrooveConstraints(hits,low,high,mandatory,forbidden,bool(seed%2))
    objective=GrooveObjective(tuple(rng.randint(-20,20) for _ in range(16)),seed%5)
    expected=oracle(constraints,objective)
    if expected is None:
        with pytest.raises(GenerationError,match="No feasible"):
            solve_groove(constraints,objective)
    else:
        pattern,receipt=solve_groove(constraints,objective)
        assert (receipt["cost"],pattern)==expected
        assert receipt["optimal"] and receipt["feasible"]
        assert pattern_cost(pattern,constraints,objective)==expected[0]


def test_exact_even_pattern_and_lexicographic_ties():
    constraints=GrooveConstraints(4,4,4)
    pattern,receipt=solve_groove(constraints,GrooveObjective((0,)*16))
    assert pattern==(0,4,8,12)
    assert receipt["cost"]==0
    forced=GrooveConstraints(4,4,4,mandatory=(1,))
    assert solve_groove(forced,GrooveObjective((0,)*16))[0]==(1,5,9,13)


def test_single_hit_wraparound_and_empty_patterns():
    objective=GrooveObjective((0,)*16)
    assert solve_groove(GrooveConstraints(0),objective)[0]==()
    assert solve_groove(GrooveConstraints(1,16,16,mandatory=(15,)),objective)[0]==(15,)
    with pytest.raises(GenerationError,match="No feasible"):
        solve_groove(GrooveConstraints(1,1,15),objective)
    assert solve_groove(GrooveConstraints(1,1,15,cyclic=False),objective)[0]==(0,)


@pytest.mark.parametrize("factory",[
    lambda:GrooveConstraints(True),
    lambda:GrooveConstraints(17),
    lambda:GrooveConstraints(4,5,4),
    lambda:GrooveConstraints(4,mandatory=(3,3)),
    lambda:GrooveConstraints(4,mandatory=(4,),forbidden=(4,)),
    lambda:GrooveConstraints(0,mandatory=(4,)),
    lambda:GrooveConstraints(4,forbidden=(16,)),
    lambda:GrooveObjective((0,)*15),
    lambda:GrooveObjective((False,)*16),
])
def test_invalid_constraints(factory):
    with pytest.raises(GenerationError):
        factory()


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(style="rhythmic",bars=8,bpm=137,swing=.3,seed=17))


@pytest.fixture
def plan(score):
    return GroovePlan(arrangement_digest(score),0,score.plan.bars,19,(
        DrumRule("kick",GrooveConstraints(5,2,8,mandatory=(0,)),style="syncopated",length_ticks=4),
        DrumRule("snare",GrooveConstraints(2,8,8,mandatory=(4,12)),style="backbeat"),
        DrumRule("hat",GrooveConstraints(9,1,4),style="rolling",pan=.2),
    ),alternate_bars=True)


def test_groove_replaces_selected_drum_regions_only(score,plan):
    result,report=generate_groove(score,plan)
    assert result.plan==score.plan
    assert result.sections==score.sections
    assert result.harmony==score.harmony
    assert tuple(e for e in result.events if e.track in PITCHED)==tuple(e for e in score.events if e.track in PITCHED)
    assert report["result_score_sha256"]==arrangement_digest(result)
    assert report["source_score_sha256"]==arrangement_digest(score)
    assert report["groove_sha256"]==plan.sha256
    assert report["rows"]
    for row in report["rows"]:
        assert row["section"] not in ("intro","break")
        events=[e for e in result.events if e.track==row["track"] and e.tick//32==row["bar"]]
        assert len(events)==row["new_events"]
        assert tuple((e.tick%32)//2 for e in events)==tuple(row["selected_slots"])
        rule=next(r for r in plan.rules if r.track==row["track"])
        validate_pattern(tuple(row["selected_slots"]),rule.constraints)
        assert all(e.tick+e.length<=(row["bar"]+1)*32 for e in events)
    skipped={bar for s in score.sections if s.kind in ("intro","break")
             for bar in range(s.first_bar,s.first_bar+s.bars)}
    assert tuple(e for e in score.events if e.tick//32 in skipped)==tuple(
        e for e in result.events if e.tick//32 in skipped)
    assert generate_groove(score,plan)==(result,report)


def test_seed_reproducible_and_changes_variation_objective(score,plan):
    a,ar=generate_groove(score,plan)
    b,br=generate_groove(score,replace(plan,seed=20))
    assert ar["groove_sha256"]!=br["groove_sha256"]
    assert any(x["selected_slots"]!=y["selected_slots"] or x["cost"]!=y["cost"]
               for x,y in zip(ar["rows"],br["rows"]))


def test_mute_rule_and_unselected_tracks(score):
    plan=GroovePlan(arrangement_digest(score),1,2,0,(DrumRule("hat",GrooveConstraints(0)),),
                    preserve_intro=False,preserve_break=False)
    result,report=generate_groove(score,plan)
    assert not any(e.track=="hat" and 1<=e.tick//32<3 for e in result.events)
    assert tuple(e for e in score.events if e.track!="hat")==tuple(e for e in result.events if e.track!="hat")
    assert all(row["new_events"]==0 for row in report["rows"])


def test_boundary_cut_rejected_without_partial_update(score):
    events=[e for e in score.events if not(e.track=="kick" and e.tick<34 and e.tick+e.length>30)]
    events.append(Event("kick",36,30,4,.5,0.))
    changed=replace(score,events=tuple(sorted(events,key=event_key)))
    plan=GroovePlan(arrangement_digest(changed),1,1,0,(DrumRule("kick",GrooveConstraints(4)),),
                    preserve_intro=False,preserve_break=False)
    with pytest.raises(GenerationError,match="boundary"):
        generate_groove(changed,plan)
    assert arrangement_digest(changed)==plan.score_sha256


def test_roundtrip_unknown_fields_and_stale_score(score,plan):
    data=encode_groove(plan)
    assert decode_groove(data)==plan
    raw=json.loads(data)
    raw["rules"][0]["constraints"]["extra"]=1
    with pytest.raises(GenerationError):
        decode_groove(json.dumps(raw).encode())
    with pytest.raises(GenerationError,match="SHA256"):
        generate_groove(score,replace(plan,score_sha256="0"*64))


@pytest.mark.parametrize("data",[b"",b"{}",b"[]",b"\xff",b"x"*(MAX_BYTES+1),b'{"format":1,"format":2}'])
def test_malformed_documents(data):
    with pytest.raises(GenerationError):
        decode_groove(data)


def test_cli_pinned_plan_inspection_and_atomic_apply(score,tmp_path,capsys):
    source=tmp_path/"score.json"
    groove=tmp_path/"groove.json"
    output=tmp_path/"new.json"
    source.write_bytes(encode_score(score))
    assert main(["plan",str(source),"--rule","kick:4","--rule","hat:8",
                 "--alternate-bars","--style","syncopated"])==0
    groove.write_text(capsys.readouterr().out)
    plan=decode_groove(groove.read_bytes())
    assert main(["inspect",str(source),str(groove)])==0
    expected=json.loads(capsys.readouterr().out)
    args=["apply",str(source),str(groove),"--output",str(output)]
    assert main(args+["--expected-groove-sha256","0"*64])==2
    assert not output.exists()
    assert main(args+["--expected-groove-sha256",plan.sha256])==0
    actual=json.loads(capsys.readouterr().out)
    assert actual["result_score_sha256"]==expected["result_score_sha256"]
    assert actual["output_sha256"]==hashlib.sha256(output.read_bytes()).hexdigest()
    assert arrangement_digest(decode_score(output.read_bytes()))==actual["result_score_sha256"]
    assert main(args)==2
