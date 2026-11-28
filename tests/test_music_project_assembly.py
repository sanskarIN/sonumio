"""Compare assembled automation against independent source sample intervals."""
from dataclasses import replace
import json

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan, TICKS_PER_BAR, compose_arrangement
from sonumio.errors import GenerationError
from sonumio.music_assembly import AssemblyPlan, Segment, SourcePin, encode_assembly
from sonumio.music_automation import AutomationPlan, MixLane, MixKey, compile_lanes
from sonumio.music_project import create_project, encode_project, read_project
from sonumio.music_project_assembly import (
    ProjectAssemblyPlan, ProjectPin, assemble_projects, crop_lane,
    encode_project_assembly, decode_project_assembly, _translated, _append_span,
)
from sonumio.music_project_assembly_cli import main


@pytest.fixture
def score():
    return compose_arrangement(ArrangementPlan(bars=4,bpm=120,swing=.2,seed=17))


def project(score, name="Source", keys=(), parameter="gain"):
    lanes = (MixLane("lead",parameter,keys),) if keys else ()
    return create_project(name,score,AutomationPlan(lanes))


def recipe(projects,segments):
    first = next(iter(projects.values())).score.plan
    bars = sum(segment.bars for segment in segments)
    assembly = AssemblyPlan(replace(first,bars=bars),tuple(SourcePin(k,v.score_sha256)
        for k,v in sorted(projects.items())),segments)
    return ProjectAssemblyPlan("Assembly",assembly,tuple(ProjectPin(k,v.sha256)
        for k,v in sorted(projects.items())))


def single(project):
    return recipe({"a":project},(Segment("a",0,4,"finale"),))


def samples(score,automation,identity,start,stop):
    from sonumio.music_automation import BLOCK_FRAMES
    lane=compile_lanes(score,automation).get(identity)
    if lane is None:
        return np.full(stop-start,1. if identity[1]=="gain" else 0.)
    return np.concatenate([lane.render(i,min(BLOCK_FRAMES,stop-i)) for i in range(start,stop,BLOCK_FRAMES)])


@pytest.mark.parametrize("law",["linear","smoothstep","step"])
def test_whole_span_sample_parity(score,law):
    source=project(score,keys=(MixKey(0,.2,law),MixKey(43,.8,law),MixKey(128,.5)))
    result,receipt=assemble_projects(single(source),{"a":source})
    total=score.plan.frame(score.plan.ticks)
    a=samples(score,source.automation,("lead","gain"),0,total)
    b=samples(result.score,result.automation,("lead","gain"),0,total)
    np.testing.assert_allclose(a,b,rtol=0,atol=2e-15)
    assert receipt["source_projects_unchanged"]
    assert result.revision==0 and result.parent_project_sha256 is None


def test_linear_partial_crop_preserves_samples_at_swung_clock(score):
    source=project(score,keys=(MixKey(0,.3,"linear"),MixKey(99,1.6)))
    lane=source.automation.lanes[0]
    cropped=crop_lane(score,lane,32,64,1.)
    assert cropped[0].tick==0 and cropped[-1].tick==32
    assert cropped[0].law=="linear"
    target=AutomationPlan((MixLane("lead","gain",cropped),))
    lo,hi=score.plan.frame(32),score.plan.frame(64)
    a=samples(score,source.automation,("lead","gain"),lo,hi)
    b=samples(score,target,("lead","gain"),0,hi-lo)
    np.testing.assert_allclose(a,b,rtol=0,atol=2e-15)


def test_partial_smoothstep_rejected(score):
    source=project(score,keys=(MixKey(0,.2,"smoothstep"),MixKey(96,1.2)))
    with pytest.raises(GenerationError,match="smoothstep"):
        crop_lane(score,source.automation.lanes[0],32,64,1.)


def test_constant_smoothstep_can_be_cropped(score):
    source=project(score,keys=(MixKey(0,.5,"smoothstep"),MixKey(96,.5)))
    assert crop_lane(score,source.automation.lanes[0],32,64,1.)==(
        MixKey(0,.5,"step"),MixKey(32,.5,"step"))


def test_step_jump_and_missing_lane_default(score):
    a=project(score,"A",(MixKey(0,.4,"step"),))
    b=project(score,"B")
    plan=recipe({"a":a,"b":b},(Segment("a",0,1,"intro",2),
        Segment("b",1,2,"theme"),Segment("a",3,4,"finale")))
    result,_=assemble_projects(plan,{"a":a,"b":b})
    lane=result.automation.lanes[0]
    assert lane.keys==(MixKey(0,.4,"step"),MixKey(64,1.,"step"),MixKey(96,.4,"step"))
    total=result.score.plan.frame(result.score.plan.ticks)
    actual=samples(result.score,result.automation,("lead","gain"),0,total)
    assert np.all(actual[:96000]==.4)
    assert np.all(actual[96000:144000]==1.)
    assert np.all(actual[144000:]==.4)


def test_continuous_seam_cannot_jump(score):
    a=project(score,"A",(MixKey(0,.4,"linear"),MixKey(32,.6)))
    b=project(score,"B")
    plan=recipe({"a":a,"b":b},(Segment("a",0,1,"intro"),
        Segment("b",1,3,"theme"),Segment("a",3,4,"finale")))
    with pytest.raises(GenerationError,match="seam"):
        assemble_projects(plan,{"a":a,"b":b})


def test_join_preserves_complete_smoothstep_when_values_match(score):
    a=project(score,"A",(MixKey(0,.4,"smoothstep"),MixKey(32,1.)))
    b=project(score,"B")
    plan=recipe({"a":a,"b":b},(Segment("a",0,1,"intro"),
        Segment("b",1,3,"theme"),Segment("b",3,4,"finale")))
    result,_=assemble_projects(plan,{"a":a,"b":b})
    a_audio=samples(score,a.automation,("lead","gain"),0,48000)
    b_audio=samples(result.score,result.automation,("lead","gain"),0,48000)
    np.testing.assert_allclose(a_audio,b_audio,rtol=0,atol=2e-15)


def test_nonintegral_clock_rejects_altered_ramp_duration(score):
    fractional=replace(score,plan=replace(score.plan,bpm=137))
    keys=(MixKey(0,.2,"linear"),MixKey(1,.8,"step"))
    pairs=[(a,b) for a in range(4) for b in range(4)
           if fractional.plan.frame(a*32+1)-fractional.plan.frame(a*32)!=
              fractional.plan.frame(b*32+1)-fractional.plan.frame(b*32)]
    # Find a pair across the bounded score, rather than assuming the
    # sample rounding differs at a particular musical tick.
    if not pairs:
        pairs=[(a,b) for a in range(127) for b in range(127)
               if fractional.plan.frame(a+1)-fractional.plan.frame(a)!=
                  fractional.plan.frame(b+1)-fractional.plan.frame(b)]
        a,b=pairs[0]
    else:
        a,b=(v*32 for v in pairs[0])
    with pytest.raises(GenerationError,match="duration"):
        _translated(fractional,fractional,keys,a,b)


def test_static_mixer_mismatch_and_digest_pins(score):
    a=project(score,"A")
    b=replace(project(score,"B"),automation=AutomationPlan((),replace(a.automation.mix,room=.5)))
    plan=recipe({"a":a,"b":b},(Segment("a",0,2,"intro"),Segment("b",2,4,"finale")))
    with pytest.raises(GenerationError,match="static mixer"):
        assemble_projects(plan,{"a":a,"b":b})
    with pytest.raises(GenerationError,match="digest"):
        assemble_projects(plan,{"a":a,"b":a})


def test_key_budget_rejected_after_repetition(score):
    keys=tuple(MixKey(tick,.25 if tick%2 else .75,"step") for tick in range(32))
    source=project(score,keys=keys)
    plan=recipe({"a":source},(Segment("a",0,1,"intro",4),Segment("a",3,4,"finale")))
    with pytest.raises(GenerationError,match="key budget"):
        assemble_projects(plan,{"a":source})


def test_pinned_recipe_roundtrip(score):
    source=project(score)
    plan=single(source)
    assert decode_project_assembly(encode_project_assembly(plan),expected_sha256=plan.sha256)==plan
    raw=json.loads(encode_project_assembly(plan))
    raw["title"]="Tampered"
    with pytest.raises(GenerationError,match="SHA256"):
        decode_project_assembly(json.dumps(raw).encode())


def test_cli_pins_source_projects_and_atomic_output(score,tmp_path,capsys):
    source=project(score,keys=(MixKey(0,.5),))
    plan=single(source)
    source_path=tmp_path/"source.sonumio"
    source_path.write_bytes(encode_project(source))
    score_recipe=tmp_path/"score-recipe.json"
    score_recipe.write_bytes(encode_assembly(plan.assembly))
    pinned=tmp_path/"project-recipe.json"
    binding=f"a={source_path}"
    assert main(["pin",str(score_recipe),"--source",binding,"--title",plan.title,"--output",str(pinned)])==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["recipe_sha256"]==plan.sha256
    out=tmp_path/"result.sonumio"
    args=["assemble",str(pinned),"--source",binding,"--output",str(out)]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert read_project(out).sha256==report["result_project_sha256"]
    assert main(args)==2
    assert main(["assemble",str(pinned),"--source",binding,"--output",str(source_path),"--overwrite"])==2
    assert source_path.read_bytes()==encode_project(source)
