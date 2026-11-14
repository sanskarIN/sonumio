"""Concurrency/retention guarantees of the in-process local development queue."""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from sonumio import api, jobs


def wait_for(queue, identifier, states, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = queue.get(identifier)
        if state["status"] in states:
            return state
        time.sleep(0.01)
    pytest.fail("Job never entered requested state: " + ",".join(states))


def test_bounded_single_worker_queue_and_running_cancellation():
    queue = jobs.LocalJobQueue(max_active=2, max_history=4, ttl_seconds=10)
    started = threading.Event()
    release = threading.Event()
    cleaned = []
    calls = []

    def work():
        calls.append('render')
        started.set()
        assert release.wait(timeout=4)
        return {"id": "generated"}

    try:
        first = queue.submit('generation', work, lambda value: cleaned.append(value))
        assert started.wait(timeout=2)
        second = queue.submit('scene', lambda: calls.append('unexpected'), lambda _: None)
        assert second['status'] == 'queued'
        with pytest.raises(jobs.QueueFullError):
            queue.submit('generation', lambda: None, lambda _: None)
        cancelled = queue.cancel(second['id'])
        assert cancelled['status'] == 'cancelled'
        again = queue.cancel(second['id'])
        assert again['status'] == 'cancelled'
        ongoing = queue.cancel(first['id'])
        assert ongoing['status'] == 'cancelling'
        release.set()
        assert wait_for(queue, first['id'], {'cancelled'})['result'] is None
        assert cleaned == [{"id": "generated"}]
        assert calls == ['render']
    finally:
        release.set()
        queue.shutdown()


def test_job_completion_and_expiry_cleans_files(monkeypatch):
    queue = jobs.LocalJobQueue(max_active=1, max_history=2, ttl_seconds=1)
    removed = []
    try:
        task = queue.submit('generation', lambda: {'id': 'file-1'}, lambda result: removed.append(result['id']))
        completed = wait_for(queue, task['id'], {'succeeded'})
        assert completed['result'] == {'id': 'file-1'}
        assert completed['finished_at'] is not None
        with pytest.raises(jobs.JobConflictError):
            queue.cancel(task['id'])
        actual = jobs.monotonic
        monkeypatch.setattr(jobs, 'monotonic', lambda: actual() + 5)
        with pytest.raises(jobs.JobNotFoundError):
            queue.get(task['id'])
        assert removed == ['file-1']
    finally:
        queue.shutdown()


def test_errors_are_sanitized_and_worker_survives():
    queue = jobs.LocalJobQueue(max_active=2)
    try:
        def unsafe():
            raise RuntimeError('/private/user/path should not be exposed')
        bad = queue.submit('generation', unsafe, lambda _: None)
        assert wait_for(queue, bad['id'], {'failed'})['error'] == 'Generation failed; see local server logs'
        good = queue.submit('generation', lambda: {'ok': True}, lambda _: None)
        assert wait_for(queue, good['id'], {'succeeded'})['result'] == {'ok': True}
        with pytest.raises(jobs.JobNotFoundError):
            queue.get('../abc')
    finally:
        queue.shutdown()


@pytest.fixture
def client(tmp_path, monkeypatch):
    queue = jobs.LocalJobQueue(max_active=4, max_history=8, ttl_seconds=1800)
    monkeypatch.setattr(api, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(api, 'JOBS', queue)
    with TestClient(api.app) as client:
        yield client
    queue.shutdown()


def poll(client, identifier, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get('/v1/jobs/' + identifier)
        assert r.status_code == 200
        if r.json()['status'] in {'succeeded', 'failed', 'cancelled'}:
            return r.json()
        time.sleep(0.02)
    pytest.fail('Job did not finish')


def test_api_generation_job_flow(client):
    created = client.post('/v1/jobs/generations', json={'prompt': 'rain and wind', 'duration_seconds': 0.5, 'seed': 12})
    assert created.status_code == 202
    item = created.json()
    assert item['status_url'] == '/v1/jobs/' + item['id']
    finished = poll(client, item['id'])
    assert finished['status'] == 'succeeded'
    assert finished['result']['sample_rate'] == 24000
    assert 'not a trained AI' in finished['result']['disclaimer']
    wav = client.get(finished['result']['audio_url'])
    assert wav.status_code == 200 and wav.content.startswith(b'RIFF')
    assert client.delete('/v1/jobs/' + item['id']).status_code == 409


def test_api_scene_job_flow(client):
    scene = {'duration_seconds': 2, 'layers': [
        {'prompt': 'thunder', 'duration_seconds': 1, 'start_seconds': 0.5, 'seed': 4}]}
    created = client.post('/v1/jobs/scenes', json=scene)
    assert created.status_code == 202
    finished = poll(client, created.json()['id'])
    assert finished['status'] == 'succeeded'
    assert finished['result']['layer_count'] == 1
    assert client.get(finished['result']['audio_url']).status_code == 200


@pytest.mark.parametrize('endpoint,payload', [
    ('generations', {'prompt': 'copy her voice', 'duration_seconds': 1}),
    ('generations', {'prompt': 'rain', 'model_id': 'meta-musicgen-small'}),
    ('generations', {'prompt': 'unrecognized sound effect'}),
    ('scenes', {'duration_seconds': 2, 'layers': [{'prompt': 'rain', 'start_seconds': 1.9, 'duration_seconds': 1}]}),
    ('scenes', {'duration_seconds': 2, 'layers': [{'prompt': 'imitate the voice', 'duration_seconds': 1}]}),
])
def test_invalid_job_is_rejected_before_enqueue(client, endpoint, payload):
    response = client.post('/v1/jobs/' + endpoint, json=payload)
    assert response.status_code == 422
    assert not api.JOBS._jobs


def test_api_missing_job_id(client):
    for method in (client.get, client.delete):
        assert method('/v1/jobs/not-a-real-id').status_code == 404


def test_api_queue_capacity_and_queued_cancellation(client, monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    original = api.generate_audio

    def slow_generate(**kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return original(**kwargs)

    monkeypatch.setattr(api, 'generate_audio', slow_generate)
    try:
        ids = []
        first = client.post('/v1/jobs/generations', json={'prompt': 'rain', 'duration_seconds': 0.5})
        assert first.status_code == 202
        ids.append(first.json()['id'])
        assert entered.wait(timeout=2)
        for _ in range(3):
            queued = client.post('/v1/jobs/generations', json={'prompt': 'wind', 'duration_seconds': 0.5})
            assert queued.status_code == 202
            ids.append(queued.json()['id'])
        assert client.post('/v1/jobs/generations', json={'prompt': 'rain'}).status_code == 429
        assert client.delete('/v1/jobs/' + ids[2]).json()['status'] == 'cancelled'
        assert api.JOBS.get(ids[2])['status'] == 'cancelled'
    finally:
        release.set()
    assert poll(client, ids[0])['status'] == 'succeeded'
    assert poll(client, ids[1])['status'] == 'succeeded'
    assert poll(client, ids[3])['status'] == 'succeeded'


def test_api_expiry_removes_generated_audio(client, monkeypatch):
    created = client.post('/v1/jobs/generations', json={'prompt': 'ocean', 'duration_seconds': 0.5})
    assert created.status_code == 202
    output = poll(client, created.json()['id'])
    assert output['status'] == 'succeeded'
    audio_url = output['result']['audio_url']
    assert client.get(audio_url).status_code == 200
    api.JOBS.ttl_seconds = 1
    real_clock = jobs.monotonic
    monkeypatch.setattr(jobs, 'monotonic', lambda: real_clock() + 3)
    assert client.get('/v1/jobs/' + created.json()['id']).status_code == 404
    assert client.get(audio_url).status_code == 404
