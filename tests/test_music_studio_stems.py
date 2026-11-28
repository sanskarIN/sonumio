"""Independent archive parity and queue receipt tests for studio auditions."""
from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import pytest

from sonumio import api
from sonumio.arrangement import ArrangementPlan, TRACKS, compose_arrangement, arrangement_digest
from sonumio.music_automation import AutomationPlan, MixLane, MixKey, generate_automated_music
from sonumio.music_mix import MixPlan
from sonumio.music_stem_pack import create_stem_pack, select_stem
from sonumio.music_studio_stems import StemRecipe, render_stem_preview
from test_music_studio_api import client, body, wait, edited_request


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4,bpm=160,seed=42,swing=.4))


@pytest.mark.parametrize("track", TRACKS)
def test_dry_stem_matches_verified_archive_member(score, track):
    raw, _ = create_stem_pack(score)
    expected, _ = select_stem(raw, track)
    wav, report = render_stem_preview(StemRecipe(score, track, "dry"))
    assert wav == expected
    assert report["common_dry_attenuation"] is True
    assert report["static_mix_applied"] is False
    assert report["room_applied"] is False
    assert report["pcm_receipt"]["wav_sha256"] == sha256(wav).hexdigest()


def test_solo_matches_effective_automation_without_mutating_original(score):
    mix = MixPlan(mute=("lead",),solo=("keys",),room=.3)
    plan = AutomationPlan((MixLane("lead","gain",(MixKey(0,.5),MixKey(32,1.))),),mix)
    recipe = StemRecipe(score,"lead","solo",plan)
    wav, report = render_stem_preview(recipe)
    effective = replace(plan,mix=replace(mix,mute=(),solo=("lead",)))
    expected, _ = generate_automated_music(score,effective)
    assert wav == expected
    assert plan.mix.mute == ("lead",)
    assert report["automation_sha256"] == plan.sha256
    assert report["effective_automation_sha256"] == effective.sha256
    assert report["mute_solo_overridden"] is True


def request_for(client, track="lead", audition="dry"):
    request=body()
    result=client.post("/v1/music/plan",json=request).json()
    request.update({"track":track,"audition":audition,
                    "expected_score_sha256":result["score_sha256"]})
    return request


@pytest.mark.parametrize("mode",["dry","solo"])
def test_plan_queue_and_download_identity(client,mode):
    request=request_for(client,audition=mode)
    request["mix"]={"room":.2,"mute":["lead"]}
    planned=client.post("/v1/music/stem/plan",json=request)
    assert planned.status_code==200
    plan=planned.json()
    assert not api.JOBS._jobs
    identifier=str(uuid4())
    accepted=client.post("/v1/jobs/music-stem",json=request,headers={"X-Sonumio-Job-Id":identifier})
    assert accepted.status_code==202
    assert accepted.json()["kind"]=="music"
    final=wait(client,identifier)
    assert final["status"]=="succeeded",final
    result=final["result"]
    assert all(result[key]==value for key,value in plan.items())
    wav=client.get(result["audio_url"])
    assert wav.status_code==200
    assert len(wav.content)==44+4*plan["frames"]
    assert sha256(wav.content).hexdigest()==result["wav_sha256"]
    again=client.post("/v1/jobs/music-stem",json=request,headers={"X-Sonumio-Job-Id":identifier})
    assert again.status_code==202
    different=client.post("/v1/jobs/music-stem",json=request|{"track":"keys"},
                          headers={"X-Sonumio-Job-Id":identifier})
    assert different.status_code==409


@pytest.mark.parametrize("changes",[
    {"track":"voice"},{"audition":"wet"},{"expected_score_sha256":None},
    {"expected_score_sha256":"0"*64},{"output_path":"private.wav"},
    {"mix":{"gains":[1]*7}},{"mix":{"mute":["lead","lead"]}},
    {"automation":{"lanes":[],"expected_automation_sha256":"0"*64}},
])
def test_invalid_requests_fail_before_queue(client,changes):
    request=request_for(client)|changes
    for endpoint in ("/v1/music/stem/plan","/v1/jobs/music-stem"):
        response=client.post(endpoint,json=request)
        assert response.status_code==422,response.text
    assert not api.JOBS._jobs


def test_missing_score_pin_rejected(client):
    request=request_for(client)
    del request["expected_score_sha256"]
    assert client.post("/v1/jobs/music-stem",json=request).status_code==422


def test_audition_tracks_edited_score(client):
    original,_,_,request=edited_request(client)
    edited=client.post("/v1/music/plan",json=request).json()
    assert edited["score_sha256"]!=original["score_sha256"]
    request.update({"track":"lead","audition":"dry",
                    "expected_score_sha256":edited["score_sha256"]})
    result=client.post("/v1/music/stem/plan",json=request)
    assert result.status_code==200
    assert result.json()["score_sha256"]==edited["score_sha256"]
    request["expected_score_sha256"]=original["score_sha256"]
    assert client.post("/v1/jobs/music-stem",json=request).status_code==422
