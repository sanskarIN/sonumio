"""Recover a lost submission without dispatching a second local render."""

from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from sonumio import api, jobs
from test_jobs import poll, wait_for


HEADER = "X-Sonumio-Job-Id"
GENERATIONS = "/v1/jobs/generations"
SCENES = "/v1/jobs/scenes"
SOUND = {"prompt": "rain", "duration_seconds": 0.5}
SCENE = {"duration_seconds": 0.5, "layers": [{"prompt": "rain", "duration_seconds": 0.5}]}


@pytest.fixture
def local_api(tmp_path, monkeypatch):
    queue = jobs.LocalJobQueue(max_active=1, max_history=4)
    monkeypatch.setattr(api, "JOBS", queue)
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    try:
        with TestClient(api.app) as client:
            yield client
    finally:
        queue.shutdown()


def test_concurrent_same_id_submissions_share_one_capacity_slot_and_wav(local_api, monkeypatch):
    identifier = str(uuid4())
    headers = {HEADER: identifier}
    enter_render = threading.Event()
    release_render = threading.Event()
    calls = []
    render = api.generate_audio

    def blocked_render(**kwargs):
        calls.append(kwargs)
        enter_render.set()
        assert release_render.wait(5)
        return render(**kwargs)

    monkeypatch.setattr(api, "generate_audio", blocked_render)
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(
                lambda _: local_api.post(GENERATIONS, json=SOUND, headers=headers),
                range(6),
            ))
        assert enter_render.wait(2)
        assert all(response.status_code == 202 for response in responses)
        assert all(response.json()["id"] == identifier for response in responses)
        assert local_api.post(GENERATIONS, json=SOUND).status_code == 429
        assert local_api.get(f"/v1/jobs/{identifier}").json()["status"] == "running"
        assert local_api.post(GENERATIONS, json={**SOUND, "seed": 99}, headers=headers).status_code == 409
        assert local_api.post(SCENES, json=SCENE, headers=headers).status_code == 409
    finally:
        release_render.set()

    finished = poll(local_api, identifier)
    assert finished["status"] == "succeeded"
    # JSON member order and explicitly supplied defaults describe the same job.
    retry = local_api.post(GENERATIONS, headers=headers, json={
        "seed": 0, "model_id": "procedural-sfx-v0", "duration_seconds": 0.5, "prompt": "rain",
    })
    assert retry.status_code == 202
    assert retry.json() == finished
    assert local_api.get(finished["result"]["audio_url"]).content.startswith(b"RIFF")
    assert len(calls) == 1


def test_scene_id_stays_bound_to_the_validated_layer_plan(local_api):
    identifier = str(uuid4())
    headers = {HEADER: identifier}
    admitted = local_api.post(SCENES, json=SCENE, headers=headers)
    assert admitted.status_code == 202
    assert admitted.json()["id"] == identifier
    finished = poll(local_api, identifier)
    assert finished["result"]["layer_count"] == 1
    assert local_api.post(SCENES, json=SCENE, headers=headers).json() == finished

    changed = {**SCENE, "layers": [{**SCENE["layers"][0], "pan": 0.5}]}
    assert local_api.post(SCENES, json=changed, headers=headers).status_code == 409


@pytest.mark.parametrize("values", [
    [""], ["not-a-uuid"], ["00000000-0000-0000-0000-000000000000"],
    ["A2FBAF4B-D381-4FD4-A663-51320A105EB0"], [str(uuid4()), str(uuid4())],
])
def test_rejects_invalid_or_repeated_job_id_headers(local_api, values):
    response = local_api.post(GENERATIONS, json=SOUND, headers=[(HEADER, value) for value in values])
    assert response.status_code == 422
    assert not api.JOBS._jobs


def test_capacity_rejection_reserves_no_identifier(local_api, monkeypatch):
    identifier = str(uuid4())
    headers = {HEADER: identifier}
    entered = threading.Event()
    release = threading.Event()
    render = api.generate_audio

    def block_once(**kwargs):
        entered.set()
        assert release.wait(5)
        return render(**kwargs)

    monkeypatch.setattr(api, "generate_audio", block_once)
    try:
        first = local_api.post(GENERATIONS, json=SOUND).json()
        assert entered.wait(2)
        assert local_api.post(GENERATIONS, json=SOUND, headers=headers).status_code == 429
        assert local_api.get(f"/v1/jobs/{identifier}").status_code == 404
    finally:
        release.set()
    assert poll(local_api, first["id"])["status"] == "succeeded"
    assert local_api.post(GENERATIONS, json=SOUND, headers=headers).json()["id"] == identifier


def test_cancelled_job_replays_until_its_original_retention_expires(monkeypatch):
    queue = jobs.LocalJobQueue(max_active=2, max_history=4, ttl_seconds=10)
    entered = threading.Event()
    release = threading.Event()
    fingerprint = bytes(32)
    identifier = str(uuid4())
    clock = jobs.monotonic
    try:
        def block():
            entered.set()
            assert release.wait(5)
            return {}

        queue.submit("scene", block, lambda _: None)
        assert entered.wait(2)
        task = queue.submit("generation", lambda: pytest.fail("cancelled render ran"), lambda _: None,
                            job_id=identifier, request_fingerprint=fingerprint)
        assert task["status"] == "queued"
        cancelled = queue.cancel(identifier)
        monkeypatch.setattr(jobs, "monotonic", lambda: clock() + 5)
        assert queue.submit("generation", dict, lambda _: None,
                            job_id=identifier, request_fingerprint=fingerprint) == cancelled
        monkeypatch.setattr(jobs, "monotonic", lambda: clock() + 11)
        with pytest.raises(jobs.JobNotFoundError):
            queue.get(identifier)
        new_task = queue.submit("generation", dict, lambda _: None,
                                job_id=identifier, request_fingerprint=b"a" * 32)
        assert new_task["status"] == "queued"
        release.set()
        assert wait_for(queue, identifier, {"succeeded"})["result"] == {}
    finally:
        release.set()
        queue.shutdown()


def test_journal_recovers_known_id_without_persisting_submission_fingerprint(tmp_path, monkeypatch):
    journal = tmp_path / "jobs.json"
    identifier = str(uuid4())
    headers = {HEADER: identifier}
    queue = jobs.LocalJobQueue(journal_path=journal)
    monkeypatch.setattr(api, "JOBS", queue)
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    try:
        with TestClient(api.app) as client:
            assert client.post(GENERATIONS, json=SOUND, headers=headers).status_code == 202
            completed = poll(client, identifier)["result"]
    finally:
        queue.shutdown()

    snapshot = journal.read_text(encoding="utf-8")
    assert "fingerprint" not in snapshot and "rain" not in snapshot
    recovered = jobs.LocalJobQueue(journal_path=journal)
    monkeypatch.setattr(api, "JOBS", recovered)
    try:
        with TestClient(api.app) as client:
            replay = client.get(f"/v1/jobs/{identifier}").json()
            assert replay["status"] == "succeeded"
            assert replay["result"] == {key: completed[key] for key in ("id", "audio_url")}
            assert client.post(GENERATIONS, json=SOUND, headers=headers).status_code == 409
            assert client.get(replay["result"]["audio_url"]).content.startswith(b"RIFF")
    finally:
        recovered.shutdown()
