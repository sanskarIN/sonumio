"""End-to-end local original-music API contracts and shared queue behavior."""
from __future__ import annotations

from hashlib import sha256
import io
import json
import threading
import time
from uuid import uuid4
import wave

import pytest
from fastapi.testclient import TestClient

from sonumio import api, jobs, job_journal
from sonumio.arrangement import ArrangementPlan, TRACKS, arrangement_digest, compose_arrangement
from sonumio.music_score_io import decode_score
from sonumio.music_mix import DEFAULT_LEVELS
from sonumio.music_studio import ENGINE_ID, MAX_PREVIEW_FRAMES


@pytest.fixture
def client(tmp_path, monkeypatch):
    queue = jobs.LocalJobQueue(max_active=2, max_history=8, ttl_seconds=1800)
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "JOBS", queue)
    try:
        with TestClient(api.app) as client:
            yield client
    finally:
        queue.shutdown()


def composition(**overrides):
    return dict(style="cinematic", root=60, mode="minor", bpm=120,
                bars=4, seed=17, swing=.1, **{}) | overrides


def body(**overrides):
    return {"composition": composition(**overrides)}


def wait(client, identifier):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = client.get("/v1/jobs/" + identifier)
        assert response.status_code == 200
        result = response.json()
        if result["status"] in {"succeeded", "failed", "cancelled"}:
            return result
        time.sleep(.01)
    pytest.fail("Music job did not complete")


def test_music_plan_is_original_deterministic_and_partitioned(client):
    request = body()
    a = client.post("/v1/music/plan", json=request)
    b = client.post("/v1/music/plan", json=request)
    assert a.status_code == b.status_code == 200
    plan = a.json()
    assert plan == b.json()
    expected = compose_arrangement(ArrangementPlan(**request["composition"]))
    assert plan["score_sha256"] == arrangement_digest(expected)
    assert plan["engine_id"] == ENGINE_ID
    assert plan["learned_model"] is False
    assert plan["external_recordings_used"] is False
    assert plan["frames"] == expected.plan.frame(expected.plan.ticks)
    assert plan["duration_seconds"] == plan["frames"] / 24000
    assert sum(section["bars"] for section in plan["sections"]) == 4
    assert plan["sections"][-1]["kind"] == "finale"
    assert set(plan["tracks"]) == set(TRACKS)
    assert sum(track["events"] for track in plan["tracks"].values()) == plan["events"]
    assert not api.JOBS._jobs


@pytest.mark.parametrize("endpoint", ["/v1/music/plan", "/v1/music/score",
                                      "/v1/music/midi", "/v1/jobs/music"])
@pytest.mark.parametrize("override", [
    {"bpm": True}, {"bpm": 120.5}, {"bpm": "120"}, {"bpm": 59}, {"bpm": 161},
    {"style": "unknown"}, {"root": 72}, {"mode": "dorian"}, {"bars": 3},
    {"bars": 17}, {"seed": -1}, {"seed": 2**32}, {"swing": .61},
    {"bars": 16, "bpm": 60}, {"output_path": "/private/path"},
])
def test_invalid_composition_rejected_before_queue_or_storage(client, endpoint, override):
    response = client.post(endpoint, json=body(**override))
    assert response.status_code == 422
    assert not api.JOBS._jobs
    assert not (api.OUTPUT_DIR / ".sonumio-api-audio").exists()


@pytest.mark.parametrize("endpoint", ["/v1/music/plan", "/v1/music/score",
                                      "/v1/music/midi", "/v1/jobs/music"])
def test_wrong_digest_is_rejected_without_render(client, endpoint):
    request = body() | {"expected_score_sha256": "0" * 64}
    response = client.post(endpoint, json=request)
    assert response.status_code == 422
    assert "digest" in response.json()["detail"].lower()
    assert not api.JOBS._jobs


@pytest.mark.parametrize("mix", [
    {"gains": [1] * 7}, {"gains": [1] * 9}, {"gains": [True] * 8},
    {"gains": [-1] + [1] * 7}, {"gains": [3] + [1] * 7},
    {"mute": ["keys", "keys"]}, {"solo": ["pad", "pad"]},
    {"mute": ["keys"], "solo": ["keys"]}, {"mute": ["no-such-track"]},
    {"room": -1}, {"room": 1.1}, {"peak_ceiling": .01}, {"peak_ceiling": 1},
    {"path": "private.wav"},
])
def test_invalid_mix_rejected_before_render(client, monkeypatch, mix):
    def impossible(*args):
        pytest.fail("Invalid mix reached renderer")
    monkeypatch.setattr(api, "render_preview", impossible)
    response = client.post("/v1/jobs/music", json=body() | {"mix": mix})
    assert response.status_code == 422
    assert not api.JOBS._jobs


def test_exact_duration_boundary(client):
    accepted = client.post("/v1/music/plan", json=body(bars=15, bpm=120))
    assert accepted.status_code == 200
    assert accepted.json()["frames"] == MAX_PREVIEW_FRAMES
    rejected = client.post("/v1/music/plan", json=body(bars=15, bpm=119))
    assert rejected.status_code == 422


def test_exports_are_pinned_to_score_and_body_checksums(client):
    plan = client.post("/v1/music/plan", json=body()).json()
    request = body() | {"expected_score_sha256": plan["score_sha256"]}
    score = client.post("/v1/music/score", json=request)
    midi = client.post("/v1/music/midi", json=request)
    for response in [score, midi]:
        assert response.status_code == 200
        assert response.headers["x-sonumio-score-sha256"] == plan["score_sha256"]
        assert response.headers["x-sonumio-content-sha256"] == sha256(response.content).hexdigest()
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-disposition"].startswith("attachment;")
    parsed = decode_score(score.content, expected_sha256=plan["score_sha256"])
    assert parsed.plan.seed == 17
    assert midi.content.startswith(b"MThd")
    assert midi.headers["content-type"] == "audio/midi"
    assert score.headers["content-type"] == "application/json"
    assert not api.JOBS._jobs


def test_music_job_receipt_matches_wav_and_http_range(client):
    plan = client.post("/v1/music/plan", json=body()).json()
    request = body() | {"expected_score_sha256": plan["score_sha256"]}
    identifier = str(uuid4())
    accepted = client.post("/v1/jobs/music", json=request,
                           headers={"X-Sonumio-Job-Id": identifier})
    assert accepted.status_code == 202
    assert accepted.json()["id"] == identifier
    assert accepted.json()["kind"] == "music"
    final = wait(client, identifier)
    assert final["status"] == "succeeded", final
    result = final["result"]
    assert result["score_sha256"] == plan["score_sha256"]
    assert result["learned_model"] is False
    assert result["engine_id"] == ENGINE_ID
    assert result["frames"] == plan["frames"]
    assert result["channels"] == 2
    wav = client.get(result["audio_url"])
    assert wav.status_code == 200
    assert sha256(wav.content).hexdigest() == result["wav_sha256"]
    with wave.open(io.BytesIO(wav.content), "rb") as reader:
        assert reader.getnchannels() == 2
        assert reader.getframerate() == 24000
        assert reader.getsampwidth() == 2
        assert reader.getnframes() == result["frames"]
    part = client.get(result["audio_url"], headers={"Range": "bytes=0-43"})
    assert part.status_code == 206
    assert part.content == wav.content[:44]
    repeated = client.post("/v1/jobs/music", json=request,
                           headers={"X-Sonumio-Job-Id": identifier})
    assert repeated.status_code == 202
    assert repeated.json()["result"]["id"] == result["id"]
    conflict = client.post("/v1/jobs/music", json=body(seed=18),
                           headers={"X-Sonumio-Job-Id": identifier})
    assert conflict.status_code == 409


def test_mix_controls_change_audio_but_not_score(client):
    baseline = client.post("/v1/jobs/music", json=body()).json()
    first = wait(client, baseline["id"])["result"]
    muted = client.post("/v1/jobs/music", json=body() | {
        "mix": {"gains": list(DEFAULT_LEVELS), "mute": list(TRACKS)}
    }).json()
    second = wait(client, muted["id"])["result"]
    assert first["score_sha256"] == second["score_sha256"]
    assert first["wav_sha256"] != second["wav_sha256"]
    raw = client.get(second["audio_url"]).content
    with wave.open(io.BytesIO(raw), "rb") as reader:
        assert not any(reader.readframes(reader.getnframes()))


def test_music_shares_capacity_and_cancellation_with_sound(client, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original = api.render_preview

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(api, "render_preview", blocked)
    music = client.post("/v1/jobs/music", json=body()).json()
    try:
        assert entered.wait(3)
        sound = client.post("/v1/jobs/generations", json={"prompt": "rain", "duration_seconds": .5})
        assert sound.status_code == 202
        assert sound.json()["status"] == "queued"
        assert client.post("/v1/jobs/music", json=body()).status_code == 429
        assert client.delete("/v1/jobs/" + music["id"]).json()["status"] == "cancelling"
    finally:
        release.set()
    assert wait(client, music["id"])["status"] == "cancelled"
    assert wait(client, sound.json()["id"])["status"] == "succeeded"
    assert len(list((api.OUTPUT_DIR / ".sonumio-api-audio").glob("*.wav"))) == 1


def test_music_journal_recovers_only_audio_reference(tmp_path):
    path = tmp_path / "jobs.json"
    identifier = str(uuid4())
    reference = {"id": identifier, "audio_url": f"/v1/generations/{identifier}/audio"}
    q = jobs.LocalJobQueue(journal_path=path)
    try:
        item = q.submit("music", lambda: reference | {"score_sha256": "a" * 64}, lambda _: None)
        deadline = time.monotonic() + 5
        while q.get(item["id"])["status"] != "succeeded":
            assert time.monotonic() < deadline
            time.sleep(.01)
        records = job_journal.LocalJobJournal(path).load()
        assert records[0]["kind"] == "music"
        assert records[0]["result"] == reference
        assert "score_sha256" not in json.dumps(records)
    finally:
        q.shutdown()
    restored = jobs.LocalJobQueue(journal_path=path)
    try:
        assert restored.get(item["id"])["result"] == reference
    finally:
        restored.shutdown()
