"""Native project API round trips, streaming bounds and idempotent rendering."""
from dataclasses import asdict
from hashlib import sha256
import json
from uuid import uuid4

import pytest

from sonumio import api
from sonumio.arrangement import ArrangementPlan, compose_arrangement, arrangement_digest
from sonumio.music_automation import AutomationPlan, MixLane, MixKey, encode_automation
from sonumio.music_mix import MixPlan
from sonumio.music_project import (
    create_project, decode_project, encode_project, render_project, verify_revision,
)
from sonumio.music_score_io import encode_score, decode_score
from sonumio.music_project_workspace import MAX_MUTATION_BYTES
from test_music_studio_api import client, wait


@pytest.fixture
def project():
    score=compose_arrangement(ArrangementPlan(bars=4,bpm=120,seed=17,swing=.2))
    automation=AutomationPlan((MixLane("lead","gain",(MixKey(0,.3),MixKey(128,1.))),),
                              MixPlan(room=.25))
    return create_project("Original music",score,automation)


def raw_post(client,path,data,*,pin=None,headers=None):
    h={"Content-Type":"application/json",**(headers or {})}
    if pin is not None:
        h["X-Sonumio-Project-SHA256"]=pin
    return client.post(path,content=data,headers=h)


def inspect(client,project):
    response=raw_post(client,"/v1/music/project/inspect",encode_project(project),pin=project.sha256)
    assert response.status_code==200,response.text
    return response.json()


def test_native_project_inspection_and_component_receipts(client,project):
    result=inspect(client,project)
    assert result["project_sha256"]==project.sha256
    assert result["score_sha256"]==project.score_sha256
    assert result["automation_sha256"]==project.automation_sha256
    assert result["content_sha256"]==sha256(encode_project(project)).hexdigest()
    assert result["score_plan"]["note_events"]==[asdict(e) for e in project.score.events]
    assert result["automation_lane_count"]==1
    assert result["automation_keys"]==2
    assert result["preview_supported"] is True
    assert result["authenticated"] is False
    score=raw_post(client,"/v1/music/project/inspect-score",encode_score(project.score)).json()
    mix=raw_post(client,"/v1/music/project/inspect-automation",encode_automation(project.automation)).json()
    assert score["score_sha256"]==project.score_sha256
    assert mix=={"automation_sha256":project.automation_sha256,"lanes":1,"keys":2}
    assert not api.JOBS._jobs
    assert not (api.OUTPUT_DIR/".sonumio-api-audio").exists()


def test_create_preserves_score_number_spelling_and_exact_source_identity(client,project):
    request={"title":project.title,"score_document":encode_score(project.score).decode(),
             "expected_score_sha256":project.score_sha256,
             "automation_document":encode_automation(project.automation).decode(),
             "expected_automation_sha256":project.automation_sha256}
    response=client.post("/v1/music/project/create",json=request)
    assert response.status_code==200,response.text
    assert response.content==encode_project(project)
    assert response.headers["x-sonumio-project-sha256"]==project.sha256
    assert response.headers["x-sonumio-content-sha256"]==sha256(response.content).hexdigest()
    assert response.headers["cache-control"]=="no-store"
    assert decode_project(response.content)==project


def test_revise_notes_title_and_static_mix_preserves_existing_automation(client,project):
    before=encode_project(project)
    index=next(i for i,e in enumerate(project.score.events) if e.track=="lead")
    note=asdict(project.score.events[index])|{"velocity":.21}
    request={"project_document":before.decode(),"expected_project_sha256":project.sha256,
             "title":"Revision 1",
             "note_edits":{"base_score_sha256":project.score_sha256,
                           "replace":[{"index":index,"note":note}]},
             "mix":{"gains":[.2]*8,"room":.1}}
    response=client.post("/v1/music/project/revise",json=request)
    assert response.status_code==200,response.text
    child=decode_project(response.content,expected_sha256=response.headers["x-sonumio-project-sha256"])
    assert child.title=="Revision 1" and child.revision==1
    assert child.automation.lanes==project.automation.lanes
    assert child.automation.mix.room==.1
    assert child.score_sha256!=project.score_sha256
    assert verify_revision(project,child)["changed"]==["title","score","automation"]
    assert encode_project(project)==before
    assert inspect(client,child)["score_plan"]["note_events"][index]==note


def test_project_score_and_midi_exports_are_bound_to_original_file(client,project):
    data=encode_project(project)
    score=raw_post(client,"/v1/music/project/score",data,pin=project.sha256)
    midi=raw_post(client,"/v1/music/project/midi",data,pin=project.sha256)
    assert score.status_code==midi.status_code==200
    assert score.content==encode_score(project.score)
    assert decode_score(score.content)==project.score
    assert midi.content.startswith(b"MThd")
    for response in (score,midi):
        assert response.headers["x-sonumio-project-sha256"]==project.sha256
        assert response.headers["x-sonumio-score-sha256"]==project.score_sha256
        assert response.headers["x-sonumio-content-sha256"]==sha256(response.content).hexdigest()


def test_project_render_matches_offline_wav_and_idempotent_job(client,project):
    data=encode_project(project)
    identifier=str(uuid4())
    expected,_=render_project(project)
    result=raw_post(client,"/v1/jobs/music-project",data,pin=project.sha256,
                    headers={"X-Sonumio-Job-Id":identifier})
    assert result.status_code==202,result.text
    final=wait(client,identifier)
    assert final["status"]=="succeeded",final
    audio=final["result"]
    assert audio["project_sha256"]==project.sha256
    assert audio["score_sha256"]==project.score_sha256
    assert audio["automation_sha256"]==project.automation_sha256
    assert audio["wav_sha256"]==sha256(expected).hexdigest()
    assert client.get(audio["audio_url"]).content==expected
    repeat=raw_post(client,"/v1/jobs/music-project",data,pin=project.sha256,
                    headers={"X-Sonumio-Job-Id":identifier})
    assert repeat.status_code==202
    assert repeat.json()["result"]["id"]==audio["id"]
    different=create_project("Different title",project.score,project.automation)
    conflict=raw_post(client,"/v1/jobs/music-project",encode_project(different),pin=different.sha256,
                      headers={"X-Sonumio-Job-Id":identifier})
    assert conflict.status_code==409


@pytest.mark.parametrize("endpoint",["/v1/music/project/score","/v1/music/project/midi","/v1/jobs/music-project"])
def test_project_export_and_render_require_exact_pin(client,project,endpoint):
    data=encode_project(project)
    assert raw_post(client,endpoint,data).status_code==422
    assert raw_post(client,endpoint,data,pin="0"*64).status_code==422
    assert raw_post(client,endpoint,data,pin="not-a-hash").status_code==422
    assert not api.JOBS._jobs


def test_streaming_limits_and_media_type(client,project):
    path="/v1/music/project/inspect"
    data=encode_project(project)
    assert client.post(path,content=data,headers={"Content-Type":"text/plain"}).status_code==415
    assert raw_post(client,path,data,headers={"Content-Encoding":"gzip"}).status_code==415
    assert raw_post(client,path,b"").status_code==422
    assert raw_post(client,path,b"x"*(2*1024*1024+1)).status_code==413
    assert raw_post(client,path,data,headers={"Content-Length":"999999999"}).status_code==413
    assert raw_post(client,path,data,headers={"Content-Length":"1"}).status_code==400
    assert raw_post(client,"/v1/music/project/create",b"x"*(MAX_MUTATION_BYTES+1)).status_code==413
    assert not api.JOBS._jobs


def test_unbounded_chunked_request_is_rejected(client):
    def chunks():
        for _ in range(5):
            yield b"x"*(512*1024)
    response=client.post("/v1/music/project/inspect",content=chunks(),
                         headers={"Content-Type":"application/json"})
    assert response.status_code==413


@pytest.mark.parametrize("change",[
    {"expected_score_sha256":"0"*64},
    {"expected_automation_sha256":"0"*64},
    {"automation_document":None},
    {"expected_automation_sha256":None},
    {"title":"invalid\nname"},
    {"unknown_path":"/tmp/source"},
])
def test_invalid_creation_request(client,project,change):
    request={"title":project.title,"score_document":encode_score(project.score).decode(),
             "expected_score_sha256":project.score_sha256,
             "automation_document":encode_automation(project.automation).decode(),
             "expected_automation_sha256":project.automation_sha256}|change
    assert client.post("/v1/music/project/create",json=request).status_code==422
    assert not api.JOBS._jobs


@pytest.mark.parametrize("case",["noop","stale","invalid_note","both_sources","both_mixes","pin_without_source"])
def test_invalid_revision_is_transactional(client,project,case):
    request={"project_document":encode_project(project).decode(),
             "expected_project_sha256":project.sha256}
    if case=="stale":
        request["expected_project_sha256"]="0"*64
        request["title"]="Changed"
    elif case=="invalid_note":
        request["note_edits"]={"base_score_sha256":project.score_sha256,"remove":[4095]}
    elif case=="both_sources":
        request["score_document"]=encode_score(project.score).decode()
        request["expected_score_sha256"]=project.score_sha256
        request["note_edits"]={"base_score_sha256":project.score_sha256,"remove":[0]}
    elif case=="both_mixes":
        request["automation_document"]=encode_automation(project.automation).decode()
        request["expected_automation_sha256"]=project.automation_sha256
        request["mix"]={"room":.2}
    elif case=="pin_without_source":
        request["expected_score_sha256"]=project.score_sha256
    assert client.post("/v1/music/project/revise",json=request).status_code==422
    assert inspect(client,project)["revision"]==0
    assert not api.JOBS._jobs


def test_long_project_can_be_inspected_but_not_rendered(client):
    score=compose_arrangement(ArrangementPlan(bars=24,bpm=120,seed=17))
    project=create_project("Long arrangement",score)
    result=inspect(client,project)
    assert result["preview_supported"] is False and result["score_plan"] is None
    assert raw_post(client,"/v1/jobs/music-project",encode_project(project),pin=project.sha256).status_code==422
    assert raw_post(client,"/v1/music/project/midi",encode_project(project),pin=project.sha256).status_code==200
    assert not api.JOBS._jobs


def test_duplicate_json_fields_rejected_before_model_validation(client):
    data=b'{"title":"first","title":"second"}'
    result=raw_post(client,"/v1/music/project/create",data)
    assert result.status_code==422
    assert not api.JOBS._jobs
