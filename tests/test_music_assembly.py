import json
from dataclasses import asdict, replace

import pytest

from sonumio.arrangement import (
    ArrangementPlan, Event, Section, TICKS_PER_BAR, arrangement_digest,
    compose_arrangement, event_key,
)
from sonumio.errors import GenerationError
from sonumio.music_assembly import (
    FORMAT, AssemblyPlan, Segment, SourcePin, assemble, decode_assembly,
    decode_draft, describe_assembly, encode_assembly, extract_span, read_assembly,
)
from sonumio.music_assembly_cli import main
from sonumio.music_score_io import encode_score, read_score


@pytest.fixture
def scores():
    plan = ArrangementPlan(bars=4, bpm=120, seed=17, swing=.1)
    return {"a": compose_arrangement(plan),
            "b": compose_arrangement(replace(plan, seed=29))}


def pins(scores):
    return tuple(SourcePin(alias, arrangement_digest(score))
                 for alias, score in sorted(scores.items()))


def original_recipe(score):
    return AssemblyPlan(score.plan, pins({"a": score}), tuple(
        Segment("a", part.first_bar, part.first_bar+part.bars, part.kind)
        for part in score.sections))


def repeated_recipe(scores):
    return AssemblyPlan(replace(scores["a"].plan, bars=8, seed=91),
                        pins(scores), (
        Segment("a", 0, 1, "intro", 2),
        Segment("b", 1, 3, "theme", 2),
        Segment("a", 3, 4, "finale", 2),
    ))


def test_reassemble_original_exactly(scores):
    source = scores["a"]
    recipe = original_recipe(source)
    output, report = assemble(recipe, {"a": source})
    assert encode_score(output) == encode_score(source)
    assert report["result_score_sha256"] == arrangement_digest(source)
    assert report["clipped_endpoints"] == 0
    assert report["audio_rendered"] is False
    assert report["source_scores_unchanged"] is True
    assert describe_assembly(recipe, {"a": source}) == report


def test_repeat_spans_from_two_verified_scores(scores):
    recipe = repeated_recipe(scores)
    before = {alias: encode_score(score) for alias, score in scores.items()}
    output, report = assemble(recipe, scores)
    assert output.plan == recipe.plan
    assert output.sections == (Section("intro",0,2), Section("theme",2,4),
                               Section("finale",6,2))
    assert output.harmony == (scores["a"].harmony[:1]*2+
                              scores["b"].harmony[1:3]*2+
                              scores["a"].harmony[3:]*2)
    expected = []
    cursor = 0
    for segment in recipe.segments:
        span = extract_span(scores[segment.source], segment)
        width = segment.last_bar-segment.first_bar
        for index in range(segment.repeat):
            expected.extend(replace(event, tick=event.tick+
                            (cursor+index*width)*TICKS_PER_BAR)
                            for event in span.events)
        cursor += segment.bars
    assert output.events == tuple(sorted(expected, key=event_key))
    assert report["events"] == len(expected)
    assert sum(report["events_by_track"].values()) == len(expected)
    assert report["segments"][1]["output_first_bar"] == 2
    assert report["segments"][1]["output_last_bar"] == 6
    assert report["percussion_re_synthesized_on_render"] is True
    assert before == {alias: encode_score(score) for alias, score in scores.items()}


def test_adjacent_equal_sections_merge(scores):
    source = scores["a"]
    recipe = AssemblyPlan(source.plan, pins({"a": source}), (
        Segment("a",0,1,"theme"), Segment("a",1,2,"theme"),
        Segment("a",2,3,"finale"), Segment("a",3,4,"finale")))
    result, _ = assemble(recipe, {"a": source})
    assert result.sections == (Section("theme",0,2),Section("finale",2,2))


def test_boundary_clipping_is_opt_in(scores):
    source = replace(scores["a"], events=(
        Event("lead",60,28,8,.5,0.), Event("lead",60,92,8,.5,0.)))
    with pytest.raises(GenerationError, match="crosses"):
        extract_span(source, Segment("a",1,3,"theme"))
    span = extract_span(source, Segment("a",1,3,"theme",boundary="clip"))
    assert span.events == (Event("lead",60,0,4,.5,0.),
                           Event("lead",60,60,4,.5,0.))
    assert span.clipped_left == span.clipped_right == 1
    assert span.source_events == 2
    recipe = AssemblyPlan(source.plan, pins({"a": source}), (
        Segment("a",0,1,"intro",boundary="clip"),
        Segment("a",1,3,"theme",boundary="clip"),
        Segment("a",3,4,"finale",boundary="clip")))
    result, report = assemble(recipe, {"a": source})
    assert [(e.tick,e.length) for e in result.events] == [(28,4),(32,4),(92,4),(96,4)]
    assert report["clipped_endpoints"] == 4
    assert len(source.events) == 2


def test_exclusive_span_end(scores):
    source = replace(scores["a"], events=(
        Event("lead",60,24,8,.5,0.), Event("lead",60,32,8,.5,0.)))
    left = extract_span(source,Segment("a",0,1,"intro"))
    right = extract_span(source,Segment("a",1,2,"theme"))
    assert [(e.tick,e.length) for e in left.events] == [(24,8)]
    assert [(e.tick,e.length) for e in right.events] == [(0,8)]
    assert left.clipped == right.clipped == 0


def test_final_harmony_is_not_silently_modified(scores):
    source = scores["a"]
    recipe = AssemblyPlan(source.plan,pins({"a":source}),
                          (Segment("a",2,3,"finale",4),))
    with pytest.raises(GenerationError,match="harmony"):
        assemble(recipe,{"a":source})


@pytest.mark.parametrize("change", [
    {"root":61},{"mode":"major"},{"bpm":121},{"swing":.2},
])
def test_clock_mismatch_rejected(scores,change):
    recipe = repeated_recipe(scores)
    replacement = compose_arrangement(replace(scores["b"].plan,**change))
    changed = dict(scores,b=replacement)
    recipe = replace(recipe,sources=pins(changed))
    with pytest.raises(GenerationError,match="clock or tonality"):
        assemble(recipe,changed)


def test_seed_and_style_do_not_require_matching_clock(scores):
    recipe = repeated_recipe(scores)
    changed = dict(scores,b=compose_arrangement(replace(scores["b"].plan,style="calm")))
    result,report = assemble(replace(recipe,sources=pins(changed)),changed)
    assert result.plan.seed == 91
    assert report["source_score_sha256"]["b"] == arrangement_digest(changed["b"])


def test_missing_extra_and_stale_source(scores):
    recipe = repeated_recipe(scores)
    for wrong in ({"a":scores["a"]},dict(scores,c=scores["b"])):
        with pytest.raises(GenerationError,match="bindings"):
            assemble(recipe,wrong)
    with pytest.raises(GenerationError,match="digest mismatch"):
        assemble(recipe,dict(scores,b=scores["a"]))


def test_span_outside_source(scores):
    source=scores["a"]
    recipe=AssemblyPlan(source.plan,pins({"a":source}),(Segment("a",4,8,"finale"),))
    with pytest.raises(GenerationError,match="source bar count"):
        assemble(recipe,{"a":source})


@pytest.mark.parametrize("args", [
    ("a",-1,1,"intro"),("a",1,1,"theme"),("a",0,33,"theme"),
    ("A",0,1,"theme"),("a",False,1,"theme"),("a",0,1,"unknown"),
    ("a",0,1,"theme",0),("a",0,1,"theme",9),("a",0,1,"theme",1,"auto"),
])
def test_invalid_segments(args):
    with pytest.raises(GenerationError):
        Segment(*args)


def test_invalid_recipe_structure(scores):
    recipe=repeated_recipe(scores)
    for changes in (
        {"sources":tuple(reversed(recipe.sources))},
        {"sources":(recipe.sources[0],recipe.sources[0])},
        {"sources":recipe.sources[:1]},
        {"segments":recipe.segments[:2]},
        {"segments":(*recipe.segments[:2],replace(recipe.segments[-1],section="theme"))},
    ):
        with pytest.raises(GenerationError):
            replace(recipe,**changes)
    for digest in ("a"*63,"F"*64,None):
        with pytest.raises(GenerationError):
            SourcePin("a",digest)


def test_recipe_canonical_roundtrip(scores):
    recipe=repeated_recipe(scores)
    raw=encode_assembly(recipe)
    assert decode_assembly(raw,expected_sha256=recipe.sha256)==recipe
    assert encode_assembly(decode_assembly(raw))==raw
    draft=json.dumps({"format":FORMAT,"assembly":asdict(recipe)}).encode()
    assert decode_draft(draft)==recipe
    with pytest.raises(GenerationError):
        decode_assembly(draft)
    with pytest.raises(GenerationError):
        decode_draft(raw)


def test_recipe_digest_pins(scores):
    recipe=repeated_recipe(scores)
    raw=encode_assembly(recipe)
    with pytest.raises(GenerationError,match="digest mismatch"):
        decode_assembly(raw,expected_sha256="0"*64)
    changed=json.loads(raw)
    changed["assembly"]["plan"]["seed"]=123
    with pytest.raises(GenerationError,match="digest mismatch"):
        decode_assembly(json.dumps(changed).encode())


@pytest.mark.parametrize("raw", [
    b"",b"{}",b"not json",b"\xff",
    b'{"format":"x","format":"y","assembly":{}}',
    b'{"format":NaN,"assembly":{}}', b"x"*(64*1024+1),
])
def test_invalid_drafts(raw):
    with pytest.raises(GenerationError):
        decode_draft(raw)


def test_unknown_nested_fields_rejected(scores):
    raw=json.loads(encode_assembly(repeated_recipe(scores)))
    raw["assembly"]["segments"][0]["unknown"]=1
    with pytest.raises(GenerationError):
        decode_assembly(json.dumps(raw).encode())


def test_cli_pin_inspect_assemble(scores,tmp_path,capsys):
    recipe=repeated_recipe(scores)
    draft=tmp_path/"draft.json"
    pinned=tmp_path/"recipe.json"
    result=tmp_path/"result.json"
    draft.write_text(json.dumps({"format":FORMAT,"assembly":asdict(recipe)}))
    args=[]
    for alias,score in scores.items():
        path=tmp_path/(alias+".json")
        path.write_bytes(encode_score(score))
        args.extend(("--source",f"{alias}={path}"))
    assert main(["pin",str(draft),*args,"--output",str(pinned)])==0
    report=json.loads(capsys.readouterr().out)
    assert report["output_kind"]=="assembly_recipe"
    assert read_assembly(pinned)==recipe
    assert main(["inspect",str(pinned),*args,"--expected-recipe-sha256",recipe.sha256])==0
    inspection=json.loads(capsys.readouterr().out)
    assert inspection["result_score_sha256"]==report["result_score_sha256"]
    assert main(["assemble",str(pinned),*args,"--output",str(result)])==0
    expected,_=assemble(recipe,scores)
    assert encode_score(read_score(result))==encode_score(expected)
    original=result.read_bytes()
    assert main(["assemble",str(pinned),*args,"--output",str(result)])==2
    assert result.read_bytes()==original
    assert main(["assemble",str(pinned),*args,"--output",str(result),
                 "--overwrite","--expected-output-sha256","0"*64])==2
    assert result.read_bytes()==original


def test_cli_rejects_overwriting_source_or_recipe(scores,tmp_path):
    source=scores["a"]
    a=tmp_path/"a.json"
    a.write_bytes(encode_score(source))
    recipe=tmp_path/"recipe.json"
    recipe.write_bytes(encode_assembly(original_recipe(source)))
    for target in (a,recipe):
        before=target.read_bytes()
        assert main(["assemble",str(recipe),"--source",f"a={a}",
                     "--output",str(target),"--overwrite"])==2
        assert target.read_bytes()==before
    assert main(["inspect",str(recipe),"--source",f"a={a}",
                 "--source",f"a={a}"])==2
