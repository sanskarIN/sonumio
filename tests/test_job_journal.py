"""Restart/recovery boundaries for the opt-in, local-only job-state ledger.

No renderer callback is replayed. The journal contains only finite bounded
metadata and a validated audio URL, never prompts or arbitrary result fields.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from sonumio import api, jobs, job_journal


def _wait(queue, identifier, statuses):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = queue.get(identifier)
        if result["status"] in statuses:
            return result
        time.sleep(0.005)
    pytest.fail("Job did not reach expected status")


def _reference():
    identifier = str(uuid4())
    return {
        "id": identifier,
        "audio_url": f"/v1/generations/{identifier}/audio",
        "private_prompt": "never persist this",
        "metrics": {"private": "not recorded"},
        "seed": 77,
    }


def test_successful_restart_recovers_only_audio_reference(tmp_path):
    journal = tmp_path / "jobs.json"
    produced = _reference()
    q1 = jobs.LocalJobQueue(journal_path=journal)
    try:
        task = q1.submit("generation", lambda: produced, lambda _: None)
        before = _wait(q1, task["id"], {"succeeded"})
        assert before["result"]["seed"] == 77
        assert before["result"]["private_prompt"] == "never persist this"
        raw = journal.read_text(encoding="utf-8")
        assert produced["id"] in raw
        assert produced["audio_url"] in raw
        assert "never persist this" not in raw
        assert "private_prompt" not in raw
        assert "metrics" not in raw
        assert "seed" not in raw
    finally:
        q1.shutdown()
    q2 = jobs.LocalJobQueue(journal_path=journal)
    try:
        result = q2.get(task["id"])
        assert result["status"] == "succeeded"
        assert result["result"] == {
            "id": produced["id"], "audio_url": produced["audio_url"]
        }
        assert not q2._pending
        assert q2._thread is None
    finally:
        q2.shutdown()


@pytest.mark.parametrize("initial", ["queued", "running", "cancelling"])
def test_crash_recovery_marks_unfinished_job_failed_without_replay(tmp_path, initial):
    journal_path = tmp_path / "jobs.json"
    identifier = str(uuid4())
    item = {
        "id": identifier, "kind": "generation", "status": initial,
        "created_at": "2026-11-22T00:00:00+00:00",
        "started_at": "2026-11-22T00:00:01+00:00" if initial != "queued" else None,
        "finished_at": None, "finished_epoch": None,
        "result": None, "error": None,
    }
    job_journal.LocalJobJournal(journal_path).save([item])
    queue = jobs.LocalJobQueue(journal_path=journal_path)
    try:
        recovered = queue.get(identifier)
        assert recovered["status"] == "failed"
        assert recovered["result"] is None
        assert recovered["error"] == "Generation interrupted by local server restart"
        assert recovered["finished_at"] is not None
        assert not queue._pending
        assert queue._thread is None
        on_disk = job_journal.LocalJobJournal(journal_path).load()
        assert on_disk[0]["status"] == "failed"
        assert on_disk[0]["finished_epoch"] is not None
        assert on_disk[0]["result"] is None
    finally:
        queue.shutdown()


def test_recovered_success_expiry_removes_owned_audio(tmp_path, monkeypatch):
    identifier = str(uuid4())
    reference = {"id": identifier, "audio_url": f"/v1/generations/{identifier}/audio"}
    journal_path = tmp_path / "jobs.json"
    removed = []
    q1 = jobs.LocalJobQueue(journal_path=journal_path, ttl_seconds=10)
    try:
        job = q1.submit("scene", lambda: reference, lambda _: None)
        assert _wait(q1, job["id"], {"succeeded"})
    finally:
        q1.shutdown()
    q2 = jobs.LocalJobQueue(
        journal_path=journal_path, ttl_seconds=10,
        recovered_cleanup=lambda result: removed.append(result["id"]),
    )
    try:
        assert q2.get(job["id"])["status"] == "succeeded"
        real_clock = jobs.monotonic
        with monkeypatch.context() as patch:
            patch.setattr(jobs, "monotonic", lambda: real_clock() + 100)
            with pytest.raises(jobs.JobNotFoundError):
                q2.get(job["id"])
        assert removed == [identifier]
        assert job_journal.LocalJobJournal(journal_path).load() == []
    finally:
        q2.shutdown()


def test_expired_entries_are_not_restored(tmp_path):
    journal_path = tmp_path / "jobs.json"
    marker = _reference()
    record = {
        "id": str(uuid4()), "kind": "scene", "status": "succeeded",
        "created_at": "2026-01-01T00:00:00+00:00",
        "started_at": None, "finished_at": "2026-01-01T00:01:00+00:00",
        "finished_epoch": 1, "result": {
            "id": marker["id"], "audio_url": marker["audio_url"]
        }, "error": None,
    }
    job_journal.LocalJobJournal(journal_path).save([record])
    cleaned = []
    queue = jobs.LocalJobQueue(
        journal_path=journal_path, ttl_seconds=60,
        recovered_cleanup=lambda result: cleaned.append(result["id"]),
    )
    try:
        with pytest.raises(jobs.JobNotFoundError):
            queue.get(record["id"])
        assert cleaned == [marker["id"]]
        assert job_journal.LocalJobJournal(journal_path).load() == []
    finally:
        queue.shutdown()


@pytest.mark.parametrize("raw", [
    b"",
    b"not json",
    b'{"version":999,"jobs":[]}',
    b'{"version":1,"jobs":{}}',
    b'{"version":1,"jobs":[{"id":"abc"}]}',
])
def test_corrupt_or_unsupported_ledger_fails_closed(tmp_path, raw):
    path = tmp_path / "jobs.json"
    path.write_bytes(raw)
    with pytest.raises(job_journal.JobJournalError):
        jobs.LocalJobQueue(journal_path=path)
    assert path.read_bytes() == raw


def test_symlink_journal_does_not_follow_external_file(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text('{"version":1,"jobs":[]}')
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink unavailable")
    with pytest.raises(job_journal.JobJournalError, match="Unsafe"):
        jobs.LocalJobQueue(journal_path=alias)
    assert outside.read_text() == '{"version":1,"jobs":[]}'


def test_parent_symlink_is_rejected(tmp_path):
    actual = tmp_path / "owned"
    actual.mkdir()
    alias = tmp_path / "redirected"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlink unavailable")
    with pytest.raises(job_journal.JobJournalError, match="directory"):
        jobs.LocalJobQueue(journal_path=alias / "jobs.json")


def test_too_large_journal_is_not_loaded(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_bytes(b" " * (job_journal.MAX_BYTES + 1))
    with pytest.raises(job_journal.JobJournalError, match="oversized"):
        jobs.LocalJobQueue(journal_path=path)


def test_atomic_replace_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "jobs.json"
    journal = job_journal.LocalJobJournal(path)
    journal.save([])
    expected = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated directory failure")

    monkeypatch.setattr(job_journal.os, "replace", fail)
    with pytest.raises(job_journal.JobJournalError, match="commit"):
        journal.save([])
    assert path.read_bytes() == expected
    assert not list(tmp_path.glob(".sonumio-jobs-*.tmp"))


def test_admission_rejects_broken_journal_without_reserving_job(tmp_path, monkeypatch):
    journal_path = tmp_path / "jobs.json"
    q = jobs.LocalJobQueue(journal_path=journal_path)
    try:
        def fail(_rows):
            raise job_journal.JobJournalError("disk full")

        monkeypatch.setattr(q._journal, "save", fail)
        with pytest.raises(jobs.QueueUnavailableError, match="journal unavailable"):
            q.submit("generation", lambda: {"id": "never"}, lambda _: None)
        assert q._jobs == {}
        assert not q._pending
        assert q._thread is None
    finally:
        q.shutdown()


def test_unique_ids_survive_restart_and_collisions(tmp_path, monkeypatch):
    path = tmp_path / "jobs.json"
    result = _reference()
    q1 = jobs.LocalJobQueue(journal_path=path)
    try:
        old = q1.submit("generation", lambda: result, lambda _: None)
        assert _wait(q1, old["id"], {"succeeded"})
    finally:
        q1.shutdown()
    q2 = jobs.LocalJobQueue(journal_path=path)
    try:
        fresh = str(uuid4())
        old_uuid = old["id"]
        sequence = iter([__import__("uuid").UUID(old_uuid), __import__("uuid").UUID(fresh)])
        monkeypatch.setattr(jobs, "uuid4", lambda: next(sequence))
        new = q2.submit("generation", lambda: result, lambda _: None)
        assert new["id"] == fresh
        assert _wait(q2, new["id"], {"succeeded"})
        assert q2.get(old_uuid)["status"] == "succeeded"
    finally:
        q2.shutdown()


def test_interrupted_audio_job_status_visible_through_api(tmp_path, monkeypatch):
    journal_path = tmp_path / "jobs.json"
    identifier = str(uuid4())
    job_journal.LocalJobJournal(journal_path).save([{
        "id": identifier, "kind": "generation", "status": "running",
        "created_at": "2026-11-22T00:00:00+00:00",
        "started_at": "2026-11-22T00:00:01+00:00",
        "finished_at": None, "finished_epoch": None,
        "result": None, "error": None,
    }])
    q = jobs.LocalJobQueue(journal_path=journal_path)
    monkeypatch.setattr(api, "JOBS", q)
    try:
        with TestClient(api.app) as client:
            response = client.get(f"/v1/jobs/{identifier}")
        assert response.status_code == 200
        assert response.json()["status"] == "failed"
        assert response.json()["error"] == "Generation interrupted by local server restart"
    finally:
        q.shutdown()


def test_new_renderer_results_do_not_leak_secrets_in_ledger(tmp_path):
    path = tmp_path / "jobs.json"
    queue = jobs.LocalJobQueue(journal_path=path)
    try:
        task = queue.submit("generation", lambda: {
            "id": "not-a-uuid",
            "audio_url": "file:///private/secrets",
            "private_key": "extremely-secret-token",
        }, lambda _: None)
        assert _wait(queue, task["id"], {"succeeded"})
        payload = path.read_text()
        assert "extremely-secret-token" not in payload
        assert "file:///private/secrets" not in payload
        assert "private_key" not in payload
    finally:
        queue.shutdown()



def test_api_opt_in_expired_recovery_works_during_initial_import(tmp_path):
    """Recovery cleanup can execute before later API helpers are declared."""
    output = tmp_path / "outputs"
    output.mkdir()
    journal_path = output / ".sonumio-api-jobs.json"
    item = _reference()
    finished = {
        "id": str(uuid4()), "kind": "generation", "status": "succeeded",
        "created_at": "2026-01-01T00:00:00+00:00",
        "started_at": "2026-01-01T00:00:01+00:00",
        "finished_at": "2026-01-01T00:00:02+00:00",
        "finished_epoch": 1.0,
        "error": None,
        "result": {"id": item["id"], "audio_url": item["audio_url"]},
    }
    job_journal.LocalJobJournal(journal_path).save([finished])
    environment = {
        **os.environ,
        "SONUMIO_OUTPUT_DIR": str(output),
        "SONUMIO_LOCAL_JOB_JOURNAL": "1",
    }
    child = subprocess.run(
        [sys.executable, "-c",
         "from sonumio import api; assert api.JOBS._journal is not None"],
        env=environment, capture_output=True, text=True, timeout=25,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert job_journal.LocalJobJournal(journal_path).load() == []


def test_api_opt_in_refuses_corrupted_restart_journal(tmp_path):
    output = tmp_path / "outputs"
    output.mkdir()
    path = output / ".sonumio-api-jobs.json"
    path.write_bytes(b"not valid JSON")
    child = subprocess.run(
        [sys.executable, "-c", "from sonumio import api"],
        env={**os.environ, "SONUMIO_OUTPUT_DIR": str(output),
             "SONUMIO_LOCAL_JOB_JOURNAL": "1"},
        capture_output=True, text=True, timeout=25, check=False,
    )
    assert child.returncode != 0
    assert "JobJournalError" in child.stderr
    assert path.read_bytes() == b"not valid JSON"



def test_failed_new_admission_preserves_full_history_and_existing_audio(
    tmp_path, monkeypatch,
):
    """A failed snapshot must not evict the only known successful WAV."""
    path = tmp_path / "jobs.json"
    original_audio = _reference()
    removed = []
    q = jobs.LocalJobQueue(
        max_active=1, max_history=1, journal_path=path,
    )
    try:
        old = q.submit(
            "generation", lambda: original_audio,
            lambda result: removed.append(result["id"]),
        )
        assert _wait(q, old["id"], {"succeeded"})
        previous = path.read_bytes()

        with monkeypatch.context() as patch:
            patch.setattr(
                q._journal, "save",
                lambda _: (_ for _ in ()).throw(job_journal.JobJournalError("full")),
            )
            with pytest.raises(jobs.QueueUnavailableError, match="journal unavailable"):
                q.submit("generation", lambda: _reference(), lambda _: None)

        assert q.get(old["id"])["status"] == "succeeded"
        assert q.get(old["id"])["result"] == original_audio
        assert removed == []
        assert path.read_bytes() == previous
        assert not q._pending
        assert len(q._jobs) == 1
    finally:
        q.shutdown()



def test_actual_api_audio_survives_graceful_journaled_restart(
    tmp_path, monkeypatch,
):
    """New queue instance recovers the same playable, local-only WAV link."""
    journal_path = tmp_path / ".sonumio-api-jobs.json"
    store = api._store_for(tmp_path)

    def clean_audio(result):
        store.delete(result.get("id"))

    first_queue = jobs.LocalJobQueue(
        journal_path=journal_path, recovered_cleanup=clean_audio,
    )
    monkeypatch.setattr(api, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(api, "JOBS", first_queue)
    with TestClient(api.app) as client:
        response = client.post(
            "/v1/jobs/generations",
            json={"prompt": "rain", "duration_seconds": 0.5, "seed": 17},
        )
        assert response.status_code == 202
        identifier = response.json()["id"]
        finished = _wait(first_queue, identifier, {"succeeded"})
        audio_url = finished["result"]["audio_url"]
        original_wav = client.get(audio_url)
        assert original_wav.status_code == 200
        assert original_wav.content.startswith(b"RIFF")
    first_queue.shutdown()

    restarted = jobs.LocalJobQueue(
        journal_path=journal_path, recovered_cleanup=clean_audio,
    )
    monkeypatch.setattr(api, "JOBS", restarted)
    try:
        with TestClient(api.app) as client:
            result = client.get(f"/v1/jobs/{identifier}")
            assert result.status_code == 200
            payload = result.json()
            assert payload["status"] == "succeeded"
            assert payload["result"] == {
                "id": finished["result"]["id"],
                "audio_url": audio_url,
            }
            assert client.get(audio_url).content == original_wav.content
            assert not restarted._pending
            assert restarted._thread is None
    finally:
        restarted.shutdown()
