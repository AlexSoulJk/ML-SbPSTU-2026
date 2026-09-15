from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BackgroundJob:
    job_id: str
    kind: str
    status: str = "queued"
    stage: str = "queued"
    total: int = 0
    done: int = 0
    message: str | None = None
    cancel_requested: bool = False
    created_at: str = field(default_factory=utc_iso)
    updated_at: str = field(default_factory=utc_iso)

    def snapshot(self) -> dict[str, object]:
        return asdict(self)


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = Lock()

    def create(
        self,
        kind: str,
        *,
        total: int = 0,
        message: str | None = None,
        stage: str = "queued",
    ) -> BackgroundJob:
        with self._lock:
            job = BackgroundJob(
                job_id=uuid.uuid4().hex,
                kind=kind,
                stage=stage,
                total=total,
                message=message,
            )
            self._jobs[job.job_id] = job
            return job

    def get(self, job_id: str) -> BackgroundJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        done: int | None = None,
        total: int | None = None,
        message: str | None = None,
    ) -> BackgroundJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
            if stage is not None:
                job.stage = stage
            if done is not None:
                job.done = done
            if total is not None:
                job.total = total
            if message is not None:
                job.message = message
            job.updated_at = utc_iso()
            return job

    def request_cancel(self, job_id: str) -> BackgroundJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {"queued", "running"}:
                job.cancel_requested = True
                job.status = "cancel_requested"
                job.updated_at = utc_iso()
            return job


jobs = InMemoryJobStore()
