import asyncio
import pytest
from pathlib import Path

from job_manager import JobManager, JobStatus


@pytest.fixture
def tmp_jobs_dir(tmp_path):
    return str(tmp_path / "jobs")


@pytest.fixture
def manager(tmp_jobs_dir):
    return JobManager(jobs_dir=tmp_jobs_dir, ttl_seconds=10, max_queued=3)


def test_create_job(manager):
    job = manager.create_job(params={"conf": 0.5})
    assert job.status == JobStatus.QUEUED
    assert job.progress == 0
    assert job.input_path is not None
    assert job.input_path.parent.exists()


def test_create_job_returns_unique_ids(manager):
    j1 = manager.create_job(params={})
    j2 = manager.create_job(params={})
    assert j1.job_id != j2.job_id


def test_get_job(manager):
    job = manager.create_job(params={"conf": 0.5})
    found = manager.get_job(job.job_id)
    assert found is not None
    assert found.job_id == job.job_id


def test_get_job_not_found(manager):
    assert manager.get_job("nonexistent") is None


def test_check_queue_capacity(manager):
    for _ in range(3):
        manager.create_job(params={})
    with pytest.raises(RuntimeError, match="Too many queued jobs"):
        manager.check_queue_capacity()


def test_queue_overflow(manager):
    for _ in range(3):
        manager.create_job(params={})
    with pytest.raises(RuntimeError, match="Too many queued jobs"):
        manager.create_job(params={})


def test_job_lifecycle(manager, tmp_jobs_dir):
    job = manager.create_job(params={})
    job_id = job.job_id
    output = Path(tmp_jobs_dir) / job_id / "output.mp4"

    manager.mark_processing(job_id)
    assert manager.get_job(job_id).status == JobStatus.PROCESSING

    manager.update_progress(job_id, 50)
    assert manager.get_job(job_id).progress == 50

    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch()
    manager.mark_completed(job_id, output_path=output, stats={"total_frames": 100})
    completed = manager.get_job(job_id)
    assert completed.status == JobStatus.COMPLETED
    assert completed.progress == 100
    assert completed.completed_at is not None
    assert completed.stats == {"total_frames": 100}


def test_mark_failed(manager):
    job = manager.create_job(params={})
    manager.mark_failed(job.job_id, error="test error")
    failed = manager.get_job(job.job_id)
    assert failed.status == JobStatus.FAILED
    assert failed.error == "test error"


@pytest.mark.asyncio
async def test_get_next_job_id(manager):
    job = manager.create_job(params={})
    next_id = await asyncio.wait_for(manager.get_next_job_id(), timeout=1.0)
    assert next_id == job.job_id


def test_cleanup_expired(manager, tmp_jobs_dir):
    job = manager.create_job(params={})
    job_id = job.job_id
    output = Path(tmp_jobs_dir) / job_id / "output.mp4"
    output.touch()
    manager.mark_completed(job_id, output_path=output, stats={})

    # Not expired yet
    assert manager.cleanup_expired() == 0

    # Fake expiry by backdating completed_at
    from datetime import datetime, timedelta, timezone
    manager.get_job(job_id).completed_at = datetime.now(tz=timezone.utc) - timedelta(seconds=20)

    assert manager.cleanup_expired() == 1
    assert manager.get_job(job_id) is None
    assert not (Path(tmp_jobs_dir) / job_id).exists()


def test_startup_sweep(tmp_jobs_dir):
    jobs_dir = Path(tmp_jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    # Create orphan directories
    (jobs_dir / "orphan1").mkdir()
    (jobs_dir / "orphan2").mkdir()
    (jobs_dir / "orphan2" / "output.mp4").touch()
    # Create orphan tmp files
    (jobs_dir / "upload_abc123.tmp").touch()

    mgr = JobManager(jobs_dir=tmp_jobs_dir, ttl_seconds=10, max_queued=3)
    removed = mgr.startup_sweep()
    assert removed == 3
    assert not (jobs_dir / "orphan1").exists()
    assert not (jobs_dir / "orphan2").exists()
    assert not (jobs_dir / "upload_abc123.tmp").exists()


# --- Async lifecycle tests ---

@pytest.mark.asyncio
async def test_start_cleanup_task_creates_task(manager):
    manager.start_cleanup_task(interval=60)
    assert manager._cleanup_task is not None
    assert not manager._cleanup_task.done()
    await manager.shutdown()


@pytest.mark.asyncio
async def test_start_cleanup_task_idempotent(manager):
    manager.start_cleanup_task(interval=60)
    first_task = manager._cleanup_task
    manager.start_cleanup_task(interval=60)
    assert manager._cleanup_task is first_task
    await manager.shutdown()


@pytest.mark.asyncio
async def test_shutdown_stops_cleanup_task(manager):
    manager.start_cleanup_task(interval=60)
    task = manager._cleanup_task
    await manager.shutdown()
    assert task.done()


@pytest.mark.asyncio
async def test_shutdown_without_cleanup_task(manager):
    # Should not raise
    await manager.shutdown()


@pytest.mark.asyncio
async def test_cleanup_expired_removes_failed_jobs(manager, tmp_jobs_dir):
    from datetime import datetime, timedelta, timezone

    job = manager.create_job(params={})
    job_id = job.job_id
    manager.mark_failed(job_id, error="some error")

    # Not expired yet
    assert manager.cleanup_expired() == 0

    # Backdate completed_at to make it expired
    manager.get_job(job_id).completed_at = datetime.now(tz=timezone.utc) - timedelta(seconds=20)

    assert manager.cleanup_expired() == 1
    assert manager.get_job(job_id) is None
