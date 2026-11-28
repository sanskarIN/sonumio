"""Exact-path oracles and score invariants for melodic variation."""
from dataclasses import replace
import itertools
import hashlib
import json

import pytest

from sonumio.arrangement import (
    ArrangementPlan, Event, compose_arrangement, arrangement_digest, chord_classes, event_key,
)
from sonumio.errors import GenerationError
from sonumio.music import MODES, TICKS_PER_BAR, TICKS_PER_BEAT
from sonumio.music_melody_variation import (
    MelodyPlan, MODES_OF_CHANGE, optimize_pitch_path, vary_melody,
    encode_melody_plan, decode_melody_plan, MAX_DOCUMENT_BYTES,
)
from sonumio.music_melody_cli import main
from sonumio.music_score_io import encode_score, decode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=8,bpm=140,seed=12))


def config(score,**kw):
    return MelodyPlan(arrangement_digest(score),**kw)


def objective(path,original,targets,plan,before=None,after=None):
    if before is not None and abs(path[0]-before)>plan.max_leap:
        return None
    if after is not None and abs(path[-1]-after)>plan.max_leap:
        return None
    if any(abs(b-a)>plan.max_leap for a,b in zip(path,path[1:])):
        return None
    cost=sum(plan.contour_weight*abs(p-t)+plan.identity_weight*abs(p-o)
             for p,t,o in zip(path,targets,original,strict=True))
    cost+=sum(plan.movement_weight*abs(b-a)+plan.contour_weight*abs((b-a)-(y-x))
              for a,b,x,y in zip(path,path[1:],targets,targets[1:]))
    if before is not None:
        cost+=plan.movement_weight*abs(path[0]-before)
    if after is not None:
        cost+=plan.movement_weight*abs(path[-1]-after)
    return cost


@pytest.mark.parametrize("leap",[2,5,12])
@pytest.mark.parametrize("weights",[(1,0,0),(0,1,0),(0,0,1),(3,1,2)])
def test_dynamic_program_matches_independent_exhaustive_oracle(score,leap,weights):
    original=(60,62,65,64)
    targets=(60,58,55,64)
    candidates=((58,60,62),(58,60,62),(60,62,65),(60,62,64))
    plan=config(score,max_leap=leap,contour_weight=weights[0],
                identity_weight=weights[1],movement_weight=weights[2])
    feasible=[]
    for path in itertools.product(*candidates):
        value=objective(path,original,targets,plan,60,62)
        if value is not None:
            feasible.append((value,path))
    expected=min(feasible)
    actual,cost,trace=optimize_pitch_path(original,targets,candidates,plan,60,62)
    assert (cost,actual)==expected
    assert trace["reachable_states"]>0


def test_equal_cost_lexicographic_tie_break(score):
    plan=config(score,contour_weight=0,identity_weight=1,movement_weight=0)
    actual,cost,_=optimize_pitch_path((61,63),(61,63),((60,62),(62,64)),plan)
    assert actual==(60,62)
    assert cost==2


@pytest.mark.parametrize("before,after",[(24,None),(None,96),(None,None)])
def test_no_legal_path_rejected(score,before,after):
    plan=config(score,max_leap=1)
    with pytest.raises(GenerationError,match="legal"):
        optimize_pitch_path((60,72),(60,72),((60,),(72,)),plan,before,after)


@pytest.mark.parametrize("mode",MODES_OF_CHANGE)
def test_variation_preserves_every_non_pitch_property(score,mode):
    plan=config(score,mode=mode,bars=8,preserve_ends=False,max_leap=24,max_shift=24,
                minimum_pitch=36,maximum_pitch=96)
    result,report=vary_melody(score,plan)
    assert result.plan==score.plan
    assert result.sections==score.sections
    assert result.harmony==score.harmony
    assert len(result.events)==len(score.events)
    for original,changed in zip(score.events,result.events,strict=True):
        if original.track!="lead":
            assert original==changed
        else:
            assert replace(original,pitch=changed.pitch)==changed
            assert abs(original.pitch-changed.pitch)<=plan.max_shift
            assert changed.pitch%12 in {(score.plan.root+p)%12 for p in MODES[score.plan.mode]}
            if changed.tick%TICKS_PER_BEAT==0:
                assert changed.pitch%12 in chord_classes(score.plan,score.harmony[changed.tick//TICKS_PER_BAR])
    assert report["result_score_sha256"]==arrangement_digest(result)
    assert report["changed_notes"]==sum(a.pitch!=b.pitch for a,b in zip(score.events,result.events))
    assert vary_melody(score,plan)==(result,report)


def test_region_and_boundary_constraints(score):
    plan=config(score,first_bar=2,bars=2,max_leap=24,max_shift=24,
                preserve_ends=True,minimum_pitch=36,maximum_pitch=96)
    result,report=vary_melody(score,plan)
    selected=[e for e in score.events if e.track=="lead" and 64<=e.tick<128]
    new=[e for e in result.events if e.track=="lead" and 64<=e.tick<128]
    assert selected[0].pitch==new[0].pitch
    assert selected[-1].pitch==new[-1].pitch
    assert all(a==b for a,b in zip(score.events,result.events) if a.track!="lead" or not 64<=a.tick<128)
    assert report["notes"]==len(new)


def test_tonic_end_explicit(score):
    plan=config(score,bars=8,preserve_ends=False,tonic_end=True,max_leap=24,max_shift=24)
    result,_=vary_melody(score,plan)
    last=[e for e in result.events if e.track=="lead"][-1]
    assert last.pitch%12==score.plan.root%12


def test_exact_identity_when_shift_is_zero(score):
    plan=config(score,bars=8,max_shift=0,max_leap=24,minimum_pitch=24,maximum_pitch=96,
                strong_chord_tones=False)
    result,report=vary_melody(score,plan)
    assert result==score
    assert report["changed_notes"]==0


def test_cut_note_and_empty_region_rejected(score):
    event=Event("lead",60,31,2,.5,0.)
    source=replace(score,events=(event,))
    with pytest.raises(GenerationError,match="cuts"):
        vary_melody(source,config(source,first_bar=1,bars=1))
    with pytest.raises(GenerationError,match="requires"):
        vary_melody(source,config(source,first_bar=3,bars=1))


def test_wrong_digest_and_impossible_range(score):
    with pytest.raises(GenerationError,match="SHA256"):
        vary_melody(score,MelodyPlan("0"*64))
    with pytest.raises(GenerationError,match="legal pitch"):
        vary_melody(score,config(score,minimum_pitch=24,maximum_pitch=24,max_shift=0))
    with pytest.raises(GenerationError,match="exceeds"):
        vary_melody(score,config(score,first_bar=7,bars=2))


@pytest.mark.parametrize("kwargs",[
    {"track":"pad"},{"mode":"unknown"},{"bars":0},{"first_bar":True},
    {"minimum_pitch":23},{"maximum_pitch":97},{"max_leap":0},{"max_shift":37},
    {"preserve_ends":1},{"contour_weight":-1},{"identity_weight":17},
    {"contour_weight":0,"identity_weight":0,"movement_weight":0},
])
def test_invalid_configuration(score,kwargs):
    with pytest.raises(GenerationError):
        config(score,**kwargs)


@pytest.mark.parametrize("data",[b"",b"[]",b"{}",b"\xff",b"x"*(MAX_DOCUMENT_BYTES+1),
                                 b'{"format":1,"format":2}'])
def test_invalid_document(data):
    with pytest.raises(GenerationError):
        decode_melody_plan(data)


def test_canonical_round_trip(score):
    plan=config(score,bars=8,mode="arch")
    raw=encode_melody_plan(plan)
    assert decode_melody_plan(raw)==plan
    assert plan.sha256==hashlib.sha256(raw).hexdigest()
    obj=json.loads(raw)
    obj["external_path"]="melody.mid"
    with pytest.raises(GenerationError):
        decode_melody_plan(json.dumps(obj).encode())


def test_cli_atomic_result_and_pins(score,tmp_path,capsys):
    source=tmp_path/"original.json"
    recipe=tmp_path/"variation.json"
    target=tmp_path/"result.json"
    source.write_bytes(encode_score(score))
    plan=config(score,bars=8,max_leap=24)
    recipe.write_bytes(encode_melody_plan(plan))
    assert main(["inspect",str(source),str(recipe)])==0
    expected=json.loads(capsys.readouterr().out)
    args=["apply",str(source),str(recipe),"--output",str(target)]
    assert main(args+["--expected-variation-sha256","0"*64])==2
    assert not target.exists()
    assert main(args+["--expected-variation-sha256",plan.sha256])==0
    report=json.loads(capsys.readouterr().out)
    assert report["result_score_sha256"]==expected["result_score_sha256"]
    assert arrangement_digest(decode_score(target.read_bytes()))==report["result_score_sha256"]
    saved=target.read_bytes()
    assert main(args)==2
    assert target.read_bytes()==saved
    assert main(["apply",str(source),str(recipe),"--output",str(source),"--overwrite"])==2
