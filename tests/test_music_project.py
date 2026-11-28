"""Self-contained project identity, revision links and atomic round trips."""
from dataclasses import replace
import hashlib
import json
import os

import pytest

from sonumio.arrangement import ArrangementPlan, compose_arrangement, arrangement_digest
from sonumio.errors import GenerationError
from sonumio.music_automation import AutomationPlan, MixKey, MixLane, encode_automation, generate_automated_music
from sonumio.music_edit import Retime, TrackArticulation
from sonumio.music_mix import MixPlan
from sonumio.music_project import (
    MusicProject, create_project, revise_project, verify_revision, describe_project,
    encode_project, decode_project, read_project, render_project, MAX_PROJECT_BYTES, MAX_REVISION,
)
from sonumio.music_project_cli import main
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4,bpm=120,seed=17))


@pytest.fixture(scope="module")
def automation():
    return AutomationPlan((MixLane("lead","gain",(MixKey(0,.3,"smoothstep"),MixKey(128,1.))),),
                          MixPlan(room=.3))


@pytest.fixture(scope="module")
def project(score,automation):
    return create_project("Original music",score,automation)


def test_canonical_project_roundtrip_and_nested_identity(project):
    data=encode_project(project)
    restored=decode_project(data,expected_sha256=project.sha256)
    assert restored==project
    assert encode_project(restored)==data
    meta=json.loads(data)["metadata"]
    assert meta["score_sha256"]==arrangement_digest(project.score)
    assert meta["score_bytes_sha256"]==hashlib.sha256(encode_score(project.score)).hexdigest()
    assert meta["automation_sha256"]==project.automation.sha256
    assert describe_project(restored)["project_sha256"]==project.sha256
    assert not describe_project(restored)["authenticated"]


def test_outer_formatting_is_not_part_of_semantic_identity(project):
    data=json.dumps(json.loads(encode_project(project)),indent=2).encode()
    assert decode_project(data).sha256==project.sha256
    assert encode_project(decode_project(data))==encode_project(project)


@pytest.mark.parametrize("title",[""," "," leading","trailing ","x"*81,"line\nbreak","\x00","e\u0301",True])
def test_invalid_title_contract(score,title):
    with pytest.raises(GenerationError,match="title"):
        create_project(title,score)


def test_unicode_nfc_title_roundtrips(score):
    project=create_project("संगीत · Étude",score)
    assert decode_project(encode_project(project)).title==project.title


@pytest.mark.parametrize("kwargs",[
    {"revision":-1},{"revision":True},{"revision":MAX_REVISION+1},
    {"revision":0,"parent_project_sha256":"a"*64},
    {"revision":1},{"revision":1,"parent_project_sha256":"ABC"},
])
def test_invalid_revision_geometry(project,kwargs):
    with pytest.raises(GenerationError):
        replace(project,**kwargs)


def test_default_project_embeds_empty_automation(score):
    project=create_project("Default",score)
    assert project.automation==AutomationPlan(())
    assert project.revision==0 and project.parent_project_sha256 is None


def test_incompatible_automation_clock_rejected(score):
    plan=AutomationPlan((MixLane("lead","gain",(MixKey(129,.5),)),))
    with pytest.raises(GenerationError,match="beyond"):
        create_project("Invalid",score,plan)


def test_revision_changes_are_content_linked_and_verified(project):
    initial=encode_project(project)
    second=revise_project(project,title="Second version")
    third=revise_project(second,score=Retime(bpm=128).apply(second.score))
    assert second.revision==1 and third.revision==2
    assert second.parent_project_sha256==project.sha256
    assert third.parent_project_sha256==second.sha256
    assert verify_revision(project,second)["changed"]==["title"]
    assert verify_revision(second,third)["changed"]==["score"]
    assert encode_project(project)==initial
    with pytest.raises(GenerationError,match="does not match"):
        verify_revision(project,third)


def test_revision_noop_and_exhaustion_rejected(project):
    with pytest.raises(GenerationError,match="no semantic change"):
        revise_project(project)
    exhausted=replace(project,revision=MAX_REVISION,parent_project_sha256="a"*64)
    with pytest.raises(GenerationError,match="exhausted"):
        revise_project(exhausted,title="Another")


def test_revision_revalidates_automation_against_changed_score(project):
    long=compose_arrangement(ArrangementPlan(bars=8,bpm=140,seed=17))
    auto=AutomationPlan((MixLane("lead","gain",(MixKey(200,.5),)),))
    current=create_project("Long",long,auto)
    with pytest.raises(GenerationError,match="beyond"):
        revise_project(current,score=project.score)
    shortened=revise_project(current,score=project.score,automation=project.automation)
    assert verify_revision(current,shortened)["changed"]==["score","automation"]


def test_parent_verification_is_not_automatic_authentication(project):
    child=revise_project(project,title="Revision")
    assert describe_project(child)["parent_verified"] is False
    receipt=verify_revision(project,child)
    assert receipt["parent_verified"] and receipt["authenticated"] is False
    forged=replace(child,parent_project_sha256="0"*64)
    with pytest.raises(GenerationError,match="link"):
        verify_revision(project,forged)


@pytest.mark.parametrize("mutation",[
    lambda x:x.update(project_sha256="0"*64),
    lambda x:x["metadata"].update(title="Modified"),
    lambda x:x["metadata"].update(score_sha256="0"*64),
    lambda x:x["metadata"].update(score_bytes_sha256="0"*64),
    lambda x:x["metadata"].update(automation_sha256="0"*64),
    lambda x:x["score"]["score"]["plan"].update(seed=123),
    lambda x:x["automation"]["mix"].update(room=.7),
    lambda x:x.update(external_path="/tmp/file"),
    lambda x:x["metadata"].update(extra=True),
])
def test_tampered_documents_are_rejected(project,mutation):
    obj=json.loads(encode_project(project))
    mutation(obj)
    with pytest.raises(GenerationError):
        decode_project(json.dumps(obj).encode())


def test_noncanonical_nested_automation_is_rejected(project):
    obj=json.loads(encode_project(project))
    # Numerically equivalent gain spellings must not alter a canonical
    # embedded document without detection.
    obj["automation"]["lanes"][0]["keys"][-1]["value"]=1
    with pytest.raises(GenerationError,match="not canonical"):
        decode_project(json.dumps(obj).encode())


@pytest.mark.parametrize("data",[
    b"",b"x"*(MAX_PROJECT_BYTES+1),b"\xff",b"[]",
    b'{"format":"sonumio-music-project-v1","format":"duplicate"}',
    b'{"format":"sonumio-music-project-v1","project_sha256":NaN}',
])
def test_invalid_project_bytes(data):
    with pytest.raises(GenerationError):
        decode_project(data)


def test_render_reuses_exact_embedded_score_and_mix(project):
    expected,prior=generate_automated_music(project.score,project.automation)
    actual,report=render_project(project)
    assert actual==expected
    assert report["project_sha256"]==project.sha256
    assert report["score_sha256"]==prior["score_sha256"]
    assert report["automation_sha256"]==prior["automation_sha256"]
    assert report["wav_sha256"]==hashlib.sha256(actual).hexdigest()


def test_cli_project_create_inspect_extract_revise_render(project,tmp_path,capsys):
    score=tmp_path/"score.json"
    automation=tmp_path/"mix.json"
    output=tmp_path/"music.sonumio"
    score.write_bytes(encode_score(project.score))
    automation.write_bytes(encode_automation(project.automation))
    args=["create","--title",project.title,"--score",str(score),"--automation",str(automation),
          "--expected-score-sha256",project.score_sha256,"--expected-automation-sha256",project.automation_sha256,
          "--output",str(output)]
    assert main(args)==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["project_sha256"]==project.sha256
    assert receipt["output_sha256"]==hashlib.sha256(output.read_bytes()).hexdigest()
    assert read_project(output)==project
    assert main(["inspect",str(output),"--expected-project-sha256",project.sha256])==0
    assert json.loads(capsys.readouterr().out)["project_sha256"]==project.sha256
    extracted_score=tmp_path/"extract.json"
    extracted_mix=tmp_path/"extract-mix.json"
    assert main(["extract-score",str(output),"--output",str(extracted_score)])==0
    assert main(["extract-automation",str(output),"--output",str(extracted_mix)])==0
    assert extracted_score.read_bytes()==score.read_bytes()
    assert extracted_mix.read_bytes()==automation.read_bytes()
    child=tmp_path/"child.sonumio"
    assert main(["revise",str(output),"--title","Second version","--output",str(child)])==0
    assert main(["verify-parent",str(output),str(child)])==0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["parent_verified"]
    wav=tmp_path/"render.wav"
    assert main(["render",str(output),"--output",str(wav)])==0
    assert wav.read_bytes()==render_project(project)[0]
    assert main(args)==2
    assert main(args+["--overwrite","--expected-output-sha256","0"*64])==2
    assert read_project(output).sha256==project.sha256


def test_cli_refuses_input_alias_and_stale_project_pin(project,tmp_path):
    path=tmp_path/"base.sonumio"
    path.write_bytes(encode_project(project))
    assert main(["revise",str(path),"--title","New title","--output",str(path),"--overwrite"])==2
    assert main(["inspect",str(path),"--expected-project-sha256","0"*64])==2
    assert read_project(path)==project
