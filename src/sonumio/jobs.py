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
import re
import threading
from time import monotonic
from typing import Callable
from uuid import uuid4

from .errors import GenerationError

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

    def __init__(self, *, max_active: int = 4, max_history: int = 32, ttl_seconds: float = 1800):
        if type(max_active) is not int or not 1 <= max_active <= 128:
            raise ValueError("max_active must be in [1, 128]")
        if type(max_history) is not int or max_history < max_active:
            raise ValueError("max_history must be >= max_active")
        if (type(ttl_seconds) not in (int, float)
                or not 1 <= ttl_seconds <= 86400 or not isfinite(ttl_seconds)):
            raise ValueError("ttl_seconds must be between 1 and 86400")
        self.max_active = max_active
        self.max_history = max_history
        self.ttl_seconds = ttl_seconds
        self._cv = threading.Condition()
        self._pending: deque[str] = deque()
        self._jobs: dict[str, _Job] = {}
        self._thread: threading.Thread | None = None
        self._closing = False

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
        for job_id in old:
            self._cleanup(self._jobs.pop(job_id))
        while len(self._jobs) > self.max_history:
            oldest_finished = next((job_id for job_id, job in self._jobs.items()
                                    if job.status in TERMINAL), None)
            if oldest_finished is None:
                break
            self._cleanup(self._jobs.pop(oldest_finished))

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
                # Remove the pending ID now: cancellation must not leave an
                # unbounded deque of tombstones while a renderer is blocked.
                self._pending.remove(job_id)
            else:
                task.status = "cancelling"
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
                elif task.status in ("running", "cancelling"):
                    task.cancel_requested = True
                    task.status = "cancelling"
            self._pending.clear()
            self._cv.notify_all()
        worker = self._thread
        # Calling shutdown from a renderer must not attempt to join itself.
        if wait and worker is not None and worker is not threading.current_thread():
            worker.join(timeout=10)
        with self._cv:
            if worker is None or not worker.is_alive():
                for task in self._jobs.values():
                    self._cleanup(task)
                self._jobs.clear()
                self._pending.clear()
