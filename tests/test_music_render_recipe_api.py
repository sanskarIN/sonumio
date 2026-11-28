"""Bounded integrated recipe endpoints and shared queue verification."""
from hashlib import sha256
import json
from uuid import uuid4

import pytest

from sonumio import api
from sonumio.arrangement import ArrangementPlan, compose_arrangement
from sonumio.music_project import create_project, encode_project
from sonumio.music_performance import Performance, Expression, encode_performance
from sonumio.music_rack import RackPlan, TrackRack, encode_rack
from sonumio.music_rack_effects import Insert
from sonumio.music_render_recipe import RenderRecipe, encode_recipe, decode_recipe, render_recipe, MAX_RECIPE_BYTES
from sonumio.music_render_recipe_api import MAX_CREATE_BYTES
from test_music_studio_api import client, wait


@pytest.fixture
def project():
    return create_project("Integrated studio",compose_arrangement(ArrangementPlan(bars=4,bpm=160,seed=17)))


@pytest.fixture
def recipe(project):
    index=next(i for i,event in enumerate(project.score.events) if event.track=="lead")
    return RenderRecipe(project,Performance(project.score_sha256,(Expression(index,velocity_scale=.8),)),
                        RackPlan(project.automation,tracks=(TrackRack("lead",(Insert.create("gain",db=2.),)),)))


def post(client,path,data,pin=None,job=None,headers=None):
    h={"Content-Type":"application/json",**(headers or {})}
    if pin is not None:
        h["X-Sonumio-Recipe-SHA256"]=pin
    if job is not None:
        h["X-Sonumio-Job-Id"]=job
    return client.post(path,content=data,headers=h)


def test_create_and_inspect_are_canonical_without_queue(client,recipe):
    request={"project_document":encode_project(recipe.project).decode(),
             "expected_project_sha256":recipe.project.sha256,
             "performance_document":encode_performance(recipe.performance).decode(),
             "rack_document":encode_rack(recipe.rack).decode()}
    response=client.post("/v1/music/render-recipe/create",json=request)
    assert response.status_code==200,response.text
    assert response.content==encode_recipe(recipe)
    assert response.headers["x-sonumio-recipe-sha256"]==recipe.sha256
    assert response.headers["x-sonumio-content-sha256"]==sha256(response.content).hexdigest()
    inspected=post(client,"/v1/music/render-recipe/inspect",response.content,recipe.sha256)
    assert inspected.status_code==200
    report=inspected.json()
    assert report["recipe_sha256"]==recipe.sha256
    assert report["engine"]=="performance-rack"
    assert report["preview_supported"] and not report["rendered"]
    assert report["content_sha256"]==sha256(response.content).hexdigest()
    assert not api.JOBS._jobs


@pytest.mark.parametrize("engine",["project","performance","rack","performance-rack"])
def test_queued_preview_matches_offline_renderer(client,recipe,engine):
    recipe=RenderRecipe(recipe.project,recipe.performance if "performance" in engine else None,
                       recipe.rack if "rack" in engine else None)
    expected,receipt=render_recipe(recipe)
    data=encode_recipe(recipe)
    identifier=str(uuid4())
    accepted=post(client,"/v1/jobs/music-recipe",data,recipe.sha256,identifier)
    assert accepted.status_code==202,accepted.text
    final=wait(client,identifier)
    assert final["status"]=="succeeded",final
    report=final["result"]
    for key,value in recipe.identity.items():
        assert report[key]==value
    assert report["recipe_sha256"]==recipe.sha256
    assert report["receipt_sha256"]==receipt["receipt_sha256"]
    assert report["wav_sha256"]==sha256(expected).hexdigest()
    assert report["recipe_engine"]==engine
    assert client.get(report["audio_url"]).content==expected
    again=post(client,"/v1/jobs/music-recipe",data,recipe.sha256,identifier)
    assert again.status_code==202
    different=RenderRecipe(create_project("Other",recipe.project.score,recipe.project.automation))
    conflict=post(client,"/v1/jobs/music-recipe",encode_recipe(different),different.sha256,identifier)
    assert conflict.status_code==409


@pytest.mark.parametrize("pin",[None,"bad","0"*64])
def test_render_requires_matching_pin(client,recipe,pin):
    assert post(client,"/v1/jobs/music-recipe",encode_recipe(recipe),pin).status_code==422
    assert not api.JOBS._jobs


@pytest.mark.parametrize("change",[
    {"expected_project_sha256":"0"*64},
    {"project_document":"{}"},
    {"performance_document":"{}"},
    {"rack_document":"{}"},
    {"output_path":"/tmp/music.wav"},
])
def test_bad_create_is_rejected(client,project,change):
    body={"project_document":encode_project(project).decode(),"expected_project_sha256":project.sha256}|change
    assert client.post("/v1/music/render-recipe/create",json=body).status_code==422
    assert not api.JOBS._jobs


def test_stream_limits_and_media_type(client,recipe):
    path="/v1/music/render-recipe/inspect"
    data=encode_recipe(recipe)
    assert post(client,path,data,headers={"Content-Type":"text/plain"}).status_code==415
    assert post(client,path,data,headers={"Content-Encoding":"gzip"}).status_code==415
    assert post(client,path,b"x"*(MAX_RECIPE_BYTES+1)).status_code==413
    assert post(client,path,b"").status_code==422
    assert post(client,"/v1/music/render-recipe/create",b"x"*(MAX_CREATE_BYTES+1)).status_code==413
    assert post(client,path,data,headers={"Content-Length":"1"}).status_code==400
    assert not api.JOBS._jobs


def test_long_recipe_inspect_but_not_queue(client):
    score=compose_arrangement(ArrangementPlan(bars=24,bpm=120,seed=17))
    recipe=RenderRecipe(create_project("Long",score))
    data=encode_recipe(recipe)
    response=post(client,"/v1/music/render-recipe/inspect",data,recipe.sha256)
    assert response.status_code==200
    assert response.json()["preview_supported"] is False
    assert post(client,"/v1/jobs/music-recipe",data,recipe.sha256).status_code==422
    assert not api.JOBS._jobs
