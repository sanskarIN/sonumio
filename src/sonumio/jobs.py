"""Small, bounded, in-process generation queue for LOCAL DEVELOPMENT ONLY.

One worker prevents simultaneous heavy renderers from exhausting memory. It is
not durable across restarts; cancel of an active renderer is non-preemptive.
No model weights or network access are involved in scheduling.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from math import isfinite
from pathlib import Path
import re
import threading
from time import monotonic, time as wall_time
from typing import Callable
from uuid import uuid4

from .errors import GenerationError
from .job_journal import (
    JobJournalError, LocalJobJournal, MAX_RECORDS,
    recoverable_audio_result,
)

LOG = logging.getLogger(__name__)
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
ACTIVE = frozenset({"queued", "running", "cancelling"})
ID_PATTERN = re.compile(r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$")


class QueueFullError(RuntimeError):
    """The local concurrency / queue limit was reached."""


class QueueUnavailableError(RuntimeError):
    """A local worker cannot accept a job or is shutting down."""


class JobNotFoundError(KeyError):
    """The job has expired or does not exist."""


class JobConflictError(RuntimeError):
    """An already completed job cannot be cancelled."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class _Job:
    id: str
    kind: str
    produce: Callable[[], dict]
    cleanup: Callable[[dict], None]
    status: str = "queued"
    created_at: str = field(default_factory=_utc)
    started_at: str | None = None
    finished_at: str | None = None
    finished_clock: float | None = None
    finished_epoch: float | None = None
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False

    def public(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result if self.status == "succeeded" else None,
            "error": self.error if self.status == "failed" else None,
        }


class LocalJobQueue:
    """Single-worker bounded RAM queue with terminal-record and audio expiry.

    No guarantees of isolation, fault tolerance, durable persistence, or
    cancellation of an already running model. Not suitable for public servers.
    """

    def __init__(
        self, *, max_active: int = 4, max_history: int = 32,
        ttl_seconds: float = 1800,
        journal_path: str | Path | None = None,
        recovered_cleanup: Callable[[dict], None] | None = None,
    ):
        if type(max_active) is not int or not 1 <= max_active <= 128:
            raise ValueError("max_active must be in [1, 128]")
        if type(max_history) is not int or max_history < max_active:
            raise ValueError("max_history must be >= max_active")
        if (type(ttl_seconds) not in (int, float)
                or not 1 <= ttl_seconds <= 86400 or not isfinite(ttl_seconds)):
            raise ValueError("ttl_seconds must be between 1 and 86400")
        if journal_path is not None and max_history > MAX_RECORDS:
            raise ValueError("journaled job history exceeds bounded record limit")
        if recovered_cleanup is not None and not callable(recovered_cleanup):
            raise TypeError("recovered_cleanup must be callable")
        self.max_active = max_active
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds
        self._cv = threading.Condition()
        self._pending: deque[str] = deque()
        self._jobs: dict[str, _Job] = {}
        self._thread: threading.Thread | None = None
        self._closing = False
        self._journal = (LocalJobJournal(journal_path)
                         if journal_path is not None else None)
        self._recovered_cleanup = recovered_cleanup or (lambda result: None)
        if self._journal is not None:
            self._recover()

    def _journal_rows(self) -> list[dict]:
        rows = []
        for job in self._jobs.values():
            rows.append({
                "id": job.id,
                "kind": job.kind,
                "status": job.status,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "finished_epoch": job.finished_epoch,
                "result": (recoverable_audio_result(job.result)
                           if job.status == "succeeded" else None),
                "error": (
                    job.error if job.error in (
                        "Generation failed; see local server logs",
                        "Generation failed; invalid renderer result",
                        "Generation interrupted by local server restart",
                    ) else "Generation failed; see local server logs"
                ) if job.status == "failed" else None,
            })
        return rows

    def _save_journal(self, *, strict: bool = False) -> None:
        if self._journal is None:
            return
        try:
            self._journal.save(self._journal_rows())
        except JobJournalError as exc:
            if strict:
                raise QueueUnavailableError("Local job journal unavailable") from exc
            LOG.exception("Unable to update local job metadata journal")

    def _recover(self) -> None:
        """Recover terminal metadata; interrupted work must NEVER execute."""
        assert self._journal is not None
        now_epoch, now_clock = wall_time(), monotonic()
        for row in self._journal.load():
            status = row["status"]
            finished_epoch = row["finished_epoch"]
            finished_at = row["finished_at"]
            error = row.get("error")
            result = row.get("result")
            if status in ACTIVE:
                status = "failed"
                finished_epoch = now_epoch
                finished_at = _utc()
                error = "Generation interrupted by local server restart"
                result = None
            # Wall-clock ages survive restarts; monotonic() timestamps do not.
            age = max(0.0, now_epoch - finished_epoch)
            if age >= self.ttl_seconds:
                if result is not None:
                    try:
                        self._recovered_cleanup(result)
                    except Exception:
                        LOG.exception("Could not expire recovered local job audio")
                continue
            self._jobs[row["id"]] = _Job(
                id=row["id"], kind=row["kind"],
                produce=lambda: {}, cleanup=self._recovered_cleanup,
                status=status, created_at=row["created_at"],
                started_at=row["started_at"], finished_at=finished_at,
                finished_clock=now_clock - age,
                finished_epoch=finished_epoch, result=result, error=error,
            )
        while len(self._jobs) > self.max_history:
            key = next(iter(self._jobs))
            self._cleanup(self._jobs.pop(key))
        # Commit the interrupted-to-failed transition before new admissions.
        self._save_journal(strict=True)

    def _cleanup(self, job: _Job) -> None:
        if job.result is not None:
            try:
                job.cleanup(job.result)
            except Exception:
                LOG.exception("Could not remove expired local job audio")

    def _prune(self) -> None:
        """Invoked with the condition lock; never removes active jobs."""
        now = monotonic()
        old = [job_id for job_id, job in self._jobs.items()
               if job.finished_clock is not None and now - job.finished_clock >= self.ttl_seconds]
        changed = False
        for job_id in old:
            self._cleanup(self._jobs.pop(job_id))
            changed = True
        while len(self._jobs) > self.max_history:
            oldest_finished = next((job_id for job_id, job in self._jobs.items()
                                    if job.status in TERMINAL), None)
            if oldest_finished is None:
                break
            self._cleanup(self._jobs.pop(oldest_finished))
            changed = True
        if changed:
            self._save_journal()

    def submit(self, kind: str, produce: Callable[[], dict], cleanup: Callable[[dict], None]) -> dict:
        if kind not in {"generation", "scene"}:
            raise ValueError("unsupported job kind")
        with self._cv:
            if self._closing:
                raise QueueUnavailableError("Local generation queue is shutting down")
            self._prune()
            if sum(job.status in ACTIVE for job in self._jobs.values()) >= self.max_active:
                raise QueueFullError("Local generation queue is full; retry after another job finishes")
            # Reserve an unused ID before modifying terminal history. A UUID
            # collision must never overwrite a still-visible or queued job.
            job_id = None
            for _ in range(8):
                candidate = str(uuid4())
                if candidate not in self._jobs:
                    job_id = candidate
                    break
            if job_id is None:
                raise QueueUnavailableError("Could not allocate a unique local job ID")

            # Reserve one record while preserving the freshest terminal history.
            if len(self._jobs) >= self.max_history:
                oldest_finished = next((key for key, value in self._jobs.items()
                                        if value.status in TERMINAL), None)
                if oldest_finished is None:
                    raise QueueFullError("Job history is full; retry later")
                self._cleanup(self._jobs.pop(oldest_finished))
            task = _Job(id=job_id, kind=kind, produce=produce, cleanup=cleanup)
            self._jobs[job_id] = task
            self._pending.append(job_id)
            try:
                # The caller must never receive a job ID that cannot even be
                # recorded as queued on disk (when opt-in journaling is on).
                self._save_journal(strict=True)
            except QueueUnavailableError:
                self._pending.remove(job_id)
                del self._jobs[job_id]
                raise
            if self._thread is None or not self._thread.is_alive():
                # A worker-start error must not leave a queued record that
                # can never run. Roll back both in-memory reservations before
                # reporting unavailability to the local HTTP endpoint.
                try:
                    worker = threading.Thread(
                        target=self._work, name="sonumio-local-jobs", daemon=True
                    )
                    worker.start()
                except Exception as exc:
                    self._pending.remove(job_id)
                    del self._jobs[job_id]
                    self._thread = None
                    self._save_journal()
                    raise QueueUnavailableError(
                        "Local generation worker unavailable"
                    ) from exc
                self._thread = worker
            self._cv.notify_all()
            return task.public()

    def _find(self, job_id: str) -> _Job:
        if not isinstance(job_id, str) or not ID_PATTERN.fullmatch(job_id):
            raise JobNotFoundError(job_id)
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def get(self, job_id: str) -> dict:
        with self._cv:
            self._prune()
            return self._find(job_id).public()

    def cancel(self, job_id: str) -> dict:
        with self._cv:
            self._prune()
            task = self._find(job_id)
            if task.status in {"succeeded", "failed"}:
                raise JobConflictError("A finished job cannot be cancelled")
            if task.status == "cancelled":
                return task.public()
            task.cancel_requested = True
            if task.status == "queued":
                task.status = "cancelled"
                task.finished_at = _utc()
                task.finished_clock = monotonic()
                task.finished_epoch = wall_time()
                # Remove the pending ID now: cancellation must not leave an
                # unbounded deque of tombstones while a renderer is blocked.
                self._pending.remove(job_id)
            else:
                task.status = "cancelling"
            self._save_journal()
            self._cv.notify_all()
            return task.public()

    def _work(self) -> None:
        while True:
            with self._cv:
                self._prune()
                while not self._pending and not self._closing:
                    self._cv.wait(timeout=1.0)
                    self._prune()
                if self._closing:
                    return
                job_id = self._pending.popleft()
                task = self._jobs.get(job_id)
                if task is None or task.status != "queued":
                    continue
                task.status = "running"
                task.started_at = _utc()
                self._save_journal()
                self._cv.notify_all()
            result = None
            message = None
            try:
                result = task.produce()
                if not isinstance(result, dict):
                    message = "Generation failed; invalid renderer result"
                    result = None
            except GenerationError as exc:
                message = str(exc)
            except Exception:
                LOG.exception("Local audio generation job failed")
                message = "Generation failed; see local server logs"
            except (SystemExit, KeyboardInterrupt):
                # Renderers sometimes invoke sys.exit() or surface an
                # interrupt inside the daemon thread. Never strand a running
                # record or silently kill the queue's sole worker.
                LOG.exception("Local audio generation worker interrupted")
                message = "Generation failed; see local server logs"
            with self._cv:
                if task.cancel_requested or self._closing:
                    if result is not None:
                        task.result = result
                        self._cleanup(task)
                        task.result = None
                    task.status = "cancelled"
                elif message is not None:
                    task.status = "failed"
                    task.error = message
                else:
                    task.status = "succeeded"
                    task.result = result
                task.finished_at = _utc()
                task.finished_clock = monotonic()
                task.finished_epoch = wall_time()
                self._save_journal()
                self._cv.notify_all()

    def shutdown(self, *, wait: bool = True) -> None:
        with self._cv:
            self._closing = True
            now = monotonic()
            for task in self._jobs.values():
                if task.status == "queued":
                    # Queued renders cannot execute after shutdown. Make them
                    # terminal *immediately* and clear their pending IDs, even
                    # when the active renderer is still blocked.
                    task.cancel_requested = True
                    task.status = "cancelled"
                    task.finished_at = _utc()
                    task.finished_clock = now
                    task.finished_epoch = wall_time()
                elif task.status in ("running", "cancelling"):
                    task.cancel_requested = True
                    task.status = "cancelling"
            self._pending.clear()
            self._save_journal()
            self._cv.notify_all()
        worker = self._thread
        # Calling shutdown from a renderer must not attempt to join itself.
        if wait and worker is not None and worker is not threading.current_thread():
            worker.join(timeout=10)
        with self._cv:
            if worker is None or not worker.is_alive():
                if self._journal is None:
                    # Preserve existing in-memory-only shutdown semantics.
                    for task in self._jobs.values():
                        self._cleanup(task)
                    self._jobs.clear()
                # In journal mode, retain terminal metadata and published
                # WAVs until TTL; the new process will perform expiry.
                self._pending.clear()
                self._save_journal()
