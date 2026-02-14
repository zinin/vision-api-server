import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class Job:
    job_id: str
    status: JobStatus
    progress: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    completed_at: datetime | None = None
    error: str | None = None
    input_path: Path | None = None
    output_path: Path | None = None
    params: dict = field(default_factory=dict)
    stats: dict | None = None


class JobManager:
    """In-memory job manager with async queue and TTL cleanup."""

    def __init__(self, jobs_dir: str, ttl_seconds: int = 3600, max_queued: int = 10):
        self.jobs_dir = Path(jobs_dir)
        self.ttl_seconds = ttl_seconds
        self.max_queued = max_queued
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    def check_queue_capacity(self) -> None:
        """Raise RuntimeError if queue is full. Call before expensive upload."""
        queued_count = sum(
            1 for j in self._jobs.values() if j.status == JobStatus.QUEUED
        )
        if queued_count >= self.max_queued:
            raise RuntimeError(
                f"Too many queued jobs ({queued_count}/{self.max_queued})"
            )

    def create_job(self, params: dict) -> Job:
        self.check_queue_capacity()

        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            input_path=job_dir / "input.mp4",
            params=params,
        )
        self._jobs[job_id] = job
        self._queue.put_nowait(job_id)
        logger.info(f"Job created: {job_id}")
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update_progress(self, job_id: str, progress: int) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.progress = progress

    def mark_processing(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            if job.status != JobStatus.QUEUED:
                logger.warning(f"Job {job_id} unexpected state for processing: {job.status}")
            job.status = JobStatus.PROCESSING
            logger.info(f"Job processing: {job_id}")

    def mark_completed(
        self, job_id: str, output_path: Path, stats: dict
    ) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.completed_at = datetime.now(tz=timezone.utc)
            job.output_path = output_path
            job.stats = stats
            logger.info(f"Job completed: {job_id}")

    def mark_failed(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(tz=timezone.utc)
            job.error = error
            logger.error(f"Job failed: {job_id}: {error}")

    async def get_next_job_id(self) -> str:
        return await self._queue.get()

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = []
        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if job.completed_at:
                    elapsed = now - job.completed_at.timestamp()
                    if elapsed > self.ttl_seconds:
                        expired.append(job_id)

        for job_id in expired:
            self._jobs.pop(job_id)
            job_dir = self.jobs_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            logger.info(f"Cleaned up expired job: {job_id}")

        return len(expired)

    async def _cleanup_loop(self, interval: int = 60) -> None:
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=interval
                )
                break
            except asyncio.TimeoutError:
                evicted = self.cleanup_expired()
                if evicted:
                    logger.info(f"Job cleanup: removed {evicted} expired job(s)")

    def startup_sweep(self) -> int:
        """Delete all job directories and orphan tmp files on startup."""
        if not self.jobs_dir.exists():
            return 0
        count = 0
        for entry in self.jobs_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
                count += 1
            elif entry.is_file() and entry.suffix == ".tmp":
                entry.unlink(missing_ok=True)
                count += 1
        if count:
            logger.info(f"Startup sweep: removed {count} orphan item(s)")
        return count

    def start_cleanup_task(self, interval: int = 60) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._shutdown_event.clear()
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(interval)
            )

    async def shutdown(self) -> None:
        logger.info("Shutting down JobManager...")
        self._shutdown_event.set()
        if self._cleanup_task and not self._cleanup_task.done():
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._cleanup_task.cancel()
        logger.info("JobManager shutdown complete")
