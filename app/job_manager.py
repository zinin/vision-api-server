import asyncio
import logging
import shutil
import threading
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
    CANCELLED = "cancelled"


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
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobManager:
    """In-memory job manager with async queue and TTL cleanup.

    Thread-safety invariant:
        All status-mutating methods (``mark_processing``, ``mark_completed``,
        ``mark_failed``, ``mark_cancelled``, ``request_cancel``) must be
        called from the asyncio event-loop thread only. ``_jobs`` and
        ``_queue`` are not protected by a lock — the single-event-loop
        assumption is what keeps them consistent.

        ``cancel_event`` on each ``Job`` is the only sanctioned cross-thread
        channel: set from the event-loop thread, read from the executor
        thread inside ``VideoAnnotator.annotate``. ``threading.Event`` is
        safe for this handoff.

        ``update_progress`` is the one exception to the event-loop-only
        rule — the annotator progress callback invokes it from the executor
        thread. It writes a single int field (``job.progress``), which is
        atomic under the GIL. Do not extend this method to touch ``status``
        or any multi-field state without adding a lock.
    """

    def __init__(self, jobs_dir: str, ttl_seconds: int = 3600, max_queued: int = 10):
        self.jobs_dir = Path(jobs_dir)
        self.ttl_seconds = ttl_seconds
        self.max_queued = max_queued
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        logger.info(
            f"JobManager initialized: jobs_dir={jobs_dir}, ttl={ttl_seconds}s, max_queued={max_queued}"
        )

    def check_queue_capacity(self) -> None:
        """Raise RuntimeError if queue is full. Call before expensive upload."""
        queued_count = self._queue.qsize()
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
            logger.debug(f"Job {job_id}: progress {progress}%")

    def mark_processing(self, job_id: str) -> bool:
        """CAS: flip QUEUED → PROCESSING atomically.

        Returns True if the transition happened, False if the job is no longer
        QUEUED (for example, /cancel arrived between the worker's guard check
        and this call). Worker must treat False as a signal to skip.
        """
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatus.QUEUED:
            return False
        job.status = JobStatus.PROCESSING
        logger.info(f"Job processing: {job_id}")
        return True

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
            logger.info(f"Job completed: {job_id}, output={output_path}, stats={stats}")

    def mark_failed(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(tz=timezone.utc)
            job.error = error
            logger.error(f"Job failed: {job_id}: {error}")

    def mark_cancelled(self, job_id: str) -> None:
        """Called by the worker after JobCancelledError propagates out of annotate().

        No-op if the job is unknown or already in a terminal status."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            logger.warning(
                f"Refusing to mark {job.status.value} job {job_id} as cancelled"
            )
            return
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(tz=timezone.utc)
        logger.info(f"Job cancelled: {job_id}")

    def request_cancel(self, job_id: str) -> Job:
        """Idempotent cancellation request.

        Raises:
            KeyError: if job_id is unknown.
            ValueError: if job is in a non-cancellable terminal status
                (COMPLETED or FAILED).
        """
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            raise ValueError(
                f"Cannot cancel job in terminal status '{job.status.value}'"
            )

        prev_status = job.status
        job.cancel_event.set()

        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.now(tz=timezone.utc)
            # Delete the uploaded input file so it does not linger until TTL.
            # The worker's finally block never runs for queued-skipped jobs.
            if job.input_path is not None and job.input_path.exists():
                try:
                    job.input_path.unlink()
                except OSError as e:
                    logger.warning(
                        f"Failed to delete input file for cancelled job {job_id}: {e}"
                    )

        if prev_status == JobStatus.CANCELLED:
            logger.debug(f"Job {job_id}: idempotent cancel no-op")
        else:
            logger.info(f"Job {job_id}: cancel requested (was {prev_status.value})")
        return job

    async def get_next_job_id(self) -> str:
        return await self._queue.get()

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = []
        for job_id, job in self._jobs.items():
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at:
                    elapsed = now - job.completed_at.timestamp()
                    if elapsed > self.ttl_seconds:
                        expired.append(job_id)

        for job_id in expired:
            self._jobs.pop(job_id)
            job_dir = self.jobs_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
                logger.debug(f"Removed directory: {job_dir}")
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
        # Safety: refuse to sweep well-known system directories
        resolved = self.jobs_dir.resolve()
        _dangerous = {Path(p) for p in ("/", "/tmp", "/var", "/home", "/root", "/etc", "/usr")}
        if resolved in _dangerous:
            logger.warning(f"Refusing to sweep dangerous path: {resolved}")
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
