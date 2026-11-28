"""Integration of local studio automation transport, queue and receipts."""
from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from sonumio import api
from sonumio.arrangement import ArrangementPlan, compose_arrangement, arrangement_digest
from sonumio.music_automation import (
    AutomationPlan, MixKey, MixLane, decode_automation, generate_automated_music,
)
from sonumio.music_studio import MusicRenderRequest, prepare_score, prepare_automation
from sonumio.music_mix import MixPlan
from sonumio.errors import GenerationError
from test_music_studio_api import client, body, wait


def request_for(client):
    request = body(swing=.2)
    base = client.post("/v1/music/plan", json=request).json()
    request["expected_score_sha256"] = base["score_sha256"]
    request["automation"] = {"lanes": [
        {"track": "keys", "parameter": "gain", "keys": [
            {"tick": 0, "value": .5, "law": "linear"},
            {"tick": 32, "value": 1.2, "law": "step"},
            {"tick": 64, "value": .3, "law": "smoothstep"},
        ]},
        {"track": "lead", "parameter": "balance", "keys": [
            {"tick": 0, "value": -1., "law": "linear"},
            {"tick": 128, "value": 1., "law": "linear"},
        ]},
    ]}
    return request


def pinned(client):
    request = request_for(client)
    response = client.post("/v1/music/automation/plan", json=request)
    assert response.status_code == 200, response.text
    receipt = response.json()
    request["automation"]["expected_automation_sha256"] = receipt["automation_sha256"]
    return request, receipt


def test_automation_plan_reports_exact_score_clock_without_rendering(client):
    request, receipt = pinned(client)
    score = compose_arrangement(ArrangementPlan(**request["composition"]))
    assert receipt["score_sha256"] == arrangement_digest(score)
    assert receipt["sample_rate"] == 24000
    assert receipt["frames"] == score.plan.frame(score.plan.ticks)
    assert receipt["keys"] == 5
    assert receipt["audio_rendered"] is False
    for lane, original in zip(receipt["lanes"], request["automation"]["lanes"], strict=True):
        assert lane["track"] == original["track"]
        assert lane["parameter"] == original["parameter"]
        for actual, source in zip(lane["keys"], original["keys"], strict=True):
            assert actual == source | {"frame": score.plan.frame(source["tick"])}
    assert not api.JOBS._jobs
    assert not (api.OUTPUT_DIR / ".sonumio-api-audio").exists()
    assert client.post("/v1/music/automation/plan", json=request).json() == receipt


def test_export_is_canonical_and_bound_to_score_and_automation(client):
    request, receipt = pinned(client)
    exported = client.post("/v1/music/automation/export", json=request)
    assert exported.status_code == 200
    assert exported.headers["x-sonumio-score-sha256"] == receipt["score_sha256"]
    assert exported.headers["x-sonumio-automation-sha256"] == receipt["automation_sha256"]
    assert exported.headers["x-sonumio-content-sha256"] == sha256(exported.content).hexdigest()
    assert exported.headers["cache-control"] == "no-store"
    plan = decode_automation(exported.content)
    assert plan.sha256 == receipt["automation_sha256"]
    assert len(plan.lanes) == 2
    assert not api.JOBS._jobs


def test_automated_preview_matches_offline_renderer(client):
    request, receipt = pinned(client)
    expected_score = compose_arrangement(ArrangementPlan(**request["composition"]))
    plan = decode_automation(client.post("/v1/music/automation/export", json=request).content)
    expected_wav, expected_report = generate_automated_music(expected_score, plan)
    accepted = client.post("/v1/jobs/music", json=request)
    assert accepted.status_code == 202, accepted.text
    result = wait(client, accepted.json()["id"])
    assert result["status"] == "succeeded", result
    audio = result["result"]
    assert audio["score_sha256"] == receipt["score_sha256"]
    assert audio["automation_sha256"] == receipt["automation_sha256"]
    assert audio["wav_sha256"] == expected_report["wav_sha256"]
    assert audio["automation_lanes"] == 2
    assert audio["learned_model"] is False
    assert client.get(audio["audio_url"]).content == expected_wav


def test_empty_automation_replays_static_mix(client):
    request = body()
    before = client.post("/v1/jobs/music", json=request).json()
    first = wait(client, before["id"])["result"]
    after = client.post("/v1/jobs/music", json=request | {"automation": {"lanes": []}}).json()
    second = wait(client, after["id"])["result"]
    assert first["score_sha256"] == second["score_sha256"]
    assert first["wav_sha256"] == second["wav_sha256"]
    assert second["automation_lanes"] == 0


def test_zero_gain_and_mix_pins(client):
    request = body() | {"mix": {"solo": ["lead"], "room": 0.},
                        "automation": {"lanes": [
                            {"track": "lead", "parameter": "gain",
                             "keys": [{"tick": 0, "value": 0.}]}
                        ]}}
    receipt = client.post("/v1/music/automation/plan", json=request).json()
    request["automation"]["expected_automation_sha256"] = receipt["automation_sha256"]
    accepted = client.post("/v1/jobs/music", json=request)
    final = wait(client, accepted.json()["id"])
    assert final["status"] == "succeeded"
    wav = client.get(final["result"]["audio_url"]).content
    assert not any(wav[44:])
    request["mix"]["room"] = .2
    assert client.post("/v1/jobs/music", json=request).status_code == 422


@pytest.mark.parametrize("case", [
    "stale", "duplicate_lane", "order_lane", "duplicate_tick", "order_tick",
    "negative_gain", "large_balance", "past_end", "bool_tick", "bool_value",
    "unknown_law", "unknown_track", "unknown_field", "empty_lane", "too_many_keys",
])
def test_invalid_automation_rejected_before_storage_or_queue(client, case):
    request = request_for(client)
    lanes = request["automation"]["lanes"]
    keys = lanes[0]["keys"]
    if case == "stale":
        request["automation"]["expected_automation_sha256"] = "0"*64
    elif case == "duplicate_lane":
        lanes.append(lanes[0])
    elif case == "order_lane":
        lanes.reverse()
    elif case == "duplicate_tick":
        keys[1]["tick"] = keys[0]["tick"]
    elif case == "order_tick":
        keys.reverse()
    elif case == "negative_gain":
        keys[0]["value"] = -.1
    elif case == "large_balance":
        lanes[1]["keys"][0]["value"] = 1.1
    elif case == "past_end":
        keys[-1]["tick"] = 129
    elif case == "bool_tick":
        keys[0]["tick"] = True
    elif case == "bool_value":
        keys[0]["value"] = True
    elif case == "unknown_law":
        keys[0]["law"] = "cubic"
    elif case == "unknown_track":
        lanes[0]["track"] = "voice"
    elif case == "unknown_field":
        keys[0]["path"] = "/tmp/audio"
    elif case == "empty_lane":
        lanes[0]["keys"] = []
    elif case == "too_many_keys":
        lanes[0]["keys"] = [{"tick": i, "value": 1.} for i in range(65)]
    for endpoint in ("/v1/music/automation/plan", "/v1/music/automation/export", "/v1/jobs/music"):
        response = client.post(endpoint, json=request)
        assert response.status_code == 422, (case, endpoint, response.text)
    assert not api.JOBS._jobs
    assert not (api.OUTPUT_DIR / ".sonumio-api-audio").exists()


def test_automation_does_not_change_score_identity(client):
    request, receipt = pinned(client)
    base = client.post("/v1/music/plan", json=body(swing=.2)).json()
    assert receipt["score_sha256"] == base["score_sha256"]
    raw = request_for(client)
    raw["automation"]["lanes"][0]["keys"][0]["value"] = .7
    changed = client.post("/v1/music/automation/plan", json=raw).json()
    assert changed["score_sha256"] == receipt["score_sha256"]
    assert changed["automation_sha256"] != receipt["automation_sha256"]


def test_note_edits_and_automation_are_bound_to_same_score(client):
    request = request_for(client)
    base = client.post("/v1/music/plan", json=body(swing=.2)).json()
    index = next(i for i,note in enumerate(base["note_events"]) if note["track"] == "lead")
    request.pop("expected_score_sha256")
    request["note_edits"] = {
        "base_score_sha256": base["score_sha256"],
        "replace": [{"index":index,"note":base["note_events"][index] | {"velocity": .21}}],
    }
    model = MusicRenderRequest.model_validate(request)
    score = prepare_score(model)
    plan = prepare_automation(model, score)
    assert arrangement_digest(score) != base["score_sha256"]
    request["expected_score_sha256"] = arrangement_digest(score)
    request["automation"]["expected_automation_sha256"] = plan.sha256
    response = client.post("/v1/music/automation/plan", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["score_sha256"] == arrangement_digest(score)
    request["expected_score_sha256"] = base["score_sha256"]
    assert client.post("/v1/music/automation/export", json=request).status_code == 422
