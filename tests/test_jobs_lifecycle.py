"""Failure-injection tests for Sonumio's bounded in-process job lifecycle.

These tests use local events and exact injected failures; no model training,
network services, production queues, or external audio datasets are required.
"""
from __future__ import annotations

import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from sonumio import api, jobs


def _eventually(queue: jobs.LocalJobQueue, identifier: str, allowed: set[str]):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = queue.get(identifier)
        if status["status"] in allowed:
            return status
        time.sleep(0.005)
    pytest.fail("Local job did not enter " + repr(allowed))


def test_uuid_collisions_are_retried_without_overwriting_running_job(monkeypatch):
    queue = jobs.LocalJobQueue(max_active=2)
    started = threading.Event()
    release = threading.Event()
    first_uuid, second_uuid = uuid.UUID(int=1), uuid.UUID(int=2)
    used = iter([first_uuid, first_uuid, first_uuid, second_uuid])
    monkeypatch.setattr(jobs, "uuid4", lambda: next(used))

    def blocking():
        started.set()
        assert release.wait(4)
        return {"id": "old"}

    try:
        first = queue.submit("generation", blocking, lambda _: None)
        assert started.wait(2)
        second = queue.submit("scene", lambda: {"id": "new"}, lambda _: None)
        assert first["id"] == str(first_uuid)
        assert second["id"] == str(second_uuid)
        assert queue.get(first["id"])["status"] == "running"
        assert queue.get(second["id"])["status"] == "queued"
        release.set()
        assert _eventually(queue, first["id"], {"succeeded"})["result"]["id"] == "old"
        assert _eventually(queue, second["id"], {"succeeded"})["result"]["id"] == "new"
    finally:
        release.set()
        queue.shutdown()


def test_collision_exhaustion_preserves_completed_history(monkeypatch):
    queue = jobs.LocalJobQueue(max_active=1, max_history=1)
    removed = []
    try:
        first = queue.submit(
            "generation", lambda: {"id": "kept"},
            lambda result: removed.append(result["id"]),
        )
        assert _eventually(queue, first["id"], {"succeeded"})
        monkeypatch.setattr(jobs, "uuid4", lambda: uuid.UUID(first["id"]))
        with pytest.raises(jobs.QueueUnavailableError, match="unique local job ID"):
            queue.submit("generation", lambda: {"id": "never"}, lambda _: None)
        assert queue.get(first["id"])["result"] == {"id": "kept"}
        assert removed == []
        assert not queue._pending
    finally:
        queue.shutdown()
    assert removed == ["kept"]


def test_worker_start_error_rolls_back_unrunnable_job(monkeypatch):
    queue = jobs.LocalJobQueue(max_active=1)
    class BrokenThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("failed to allocate worker")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(jobs.threading, "Thread", BrokenThread)
            with pytest.raises(jobs.QueueUnavailableError, match="worker unavailable"):
                queue.submit("generation", lambda: {"id": "orphan"}, lambda _: None)
        assert queue._thread is None
        assert not queue._jobs
        assert not queue._pending
        valid = queue.submit("generation", lambda: {"id": "recovered"}, lambda _: None)
        assert _eventually(queue, valid["id"], {"succeeded"})["result"] == {
            "id": "recovered"
        }
    finally:
        queue.shutdown()


def test_worker_constructor_error_also_rolls_back(monkeypatch):
    queue = jobs.LocalJobQueue(max_active=1)

    def cannot_construct(*args, **kwargs):
        raise OSError("no thread resources")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(jobs.threading, "Thread", cannot_construct)
            with pytest.raises(jobs.QueueUnavailableError, match="worker unavailable"):
                queue.submit("generation", lambda: {"id": "orphan"}, lambda _: None)
        assert not queue._jobs
        assert not queue._pending
        good = queue.submit("scene", lambda: {"ok": True}, lambda _: None)
        assert _eventually(queue, good["id"], {"succeeded"})["result"]["ok"]
    finally:
        queue.shutdown()


def test_nonblocking_shutdown_cancels_pending_immediately_and_running_later():
    queue = jobs.LocalJobQueue(max_active=3, max_history=5)
    started = threading.Event()
    release = threading.Event()
    rendered = []
    cleaned = []

    def blocked():
        started.set()
        assert release.wait(4)
        rendered.append("active")
        return {"id": "generated"}

    try:
        active = queue.submit(
            "generation", blocked, lambda result: cleaned.append(result["id"])
        )
        assert started.wait(2)
        pending = [
            queue.submit(
                "scene", lambda: rendered.append("unexpected") or {"id": "unused"},
                lambda result: cleaned.append(result["id"]),
            )
            for _ in range(2)
        ]
        queue.shutdown(wait=False)
        assert queue.get(active["id"])["status"] == "cancelling"
        assert not queue._pending
        for task in pending:
            result = queue.get(task["id"])
            assert result["status"] == "cancelled"
            assert result["finished_at"] is not None
        with pytest.raises(jobs.QueueUnavailableError, match="shutting down"):
            queue.submit("generation", lambda: {}, lambda _: None)
        release.set()
        assert _eventually(queue, active["id"], {"cancelled"})["result"] is None
        assert rendered == ["active"]
        assert cleaned == ["generated"]
    finally:
        release.set()
        queue.shutdown()


def test_renderer_can_request_shutdown_without_deadlocking_its_worker():
    queue = jobs.LocalJobQueue()
    try:
        def stop_from_worker():
            queue.shutdown(wait=True)
            return {"id": "discard-this"}

        job = queue.submit("generation", stop_from_worker, lambda _: None)
        assert _eventually(queue, job["id"], {"cancelled"})["result"] is None
    finally:
        queue.shutdown()


@pytest.mark.parametrize("kind", [SystemExit, KeyboardInterrupt])
def test_worker_survives_renderer_process_control_exceptions(kind):
    queue = jobs.LocalJobQueue(max_active=2)
    try:
        def interrupted():
            raise kind("private error details")

        failed = queue.submit("generation", interrupted, lambda _: None)
        outcome = _eventually(queue, failed["id"], {"failed"})
        assert "private error details" not in outcome["error"]
        assert outcome["error"] == "Generation failed; see local server logs"
        recovered = queue.submit(
            "generation", lambda: {"id": "valid"}, lambda _: None
        )
        assert _eventually(queue, recovered["id"], {"succeeded"})["result"] == {
            "id": "valid"
        }
    finally:
        queue.shutdown()


@pytest.mark.parametrize("unexpected", [None, 5, "audio", [1, 2]])
def test_invalid_renderer_response_never_reports_success(unexpected):
    queue = jobs.LocalJobQueue()
    try:
        task = queue.submit("generation", lambda: unexpected, lambda _: None)
        outcome = _eventually(queue, task["id"], {"failed"})
        assert outcome["result"] is None
        assert outcome["error"] == "Generation failed; invalid renderer result"
    finally:
        queue.shutdown()


@pytest.mark.parametrize("config", [
    {"max_active": True},
    {"max_active": 1.0},
    {"max_history": True},
    {"max_history": 2.0},
    {"ttl_seconds": True},
    {"ttl_seconds": float("nan")},
    {"ttl_seconds": float("inf")},
])
def test_constructor_rejects_invalid_queue_limits(config):
    with pytest.raises(ValueError):
        jobs.LocalJobQueue(**config)


def test_non_string_lookup_returns_not_found():
    queue = jobs.LocalJobQueue()
    try:
        with pytest.raises(jobs.JobNotFoundError):
            queue.get(None)
        with pytest.raises(jobs.JobNotFoundError):
            queue.cancel(12)
    finally:
        queue.shutdown()


def test_api_returns_503_for_shutting_down_queue(tmp_path, monkeypatch):
    queue = jobs.LocalJobQueue()
    queue.shutdown()
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "JOBS", queue)
    with TestClient(api.app) as client:
        response = client.post(
            "/v1/jobs/generations",
            json={"prompt": "rain", "duration_seconds": 0.5},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "Local generation queue unavailable"
