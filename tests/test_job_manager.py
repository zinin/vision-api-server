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


def test_queue_capacity_counts_cancelled_queued_jobs(manager):
    """Cancelled jobs still in the asyncio.Queue must count against capacity."""
    # Fill to capacity, cancel them all.
    for _ in range(3):
        j = manager.create_job(params={})
        manager.request_cancel(j.job_id)
    # All 3 jobs are CANCELLED but still in _queue (worker hasn't drained).
    # Capacity must reject a new create.
    with pytest.raises(RuntimeError, match="Too many queued jobs"):
        manager.create_job(params={})


def test_queue_capacity_recovers_after_worker_drains_cancelled(manager):
    """After the worker drains cancelled-queued job ids, capacity must recover."""
    # Fill to capacity with cancelled-while-queued jobs.
    for _ in range(3):
        j = manager.create_job(params={})
        manager.request_cancel(j.job_id)
    with pytest.raises(RuntimeError, match="Too many queued jobs"):
        manager.create_job(params={})

    # Simulate worker pulling + skipping all 3 cancelled ids.
    for _ in range(3):
        manager._queue.get_nowait()

    # Capacity recovers — a fresh create_job must succeed.
    manager.check_queue_capacity()
    new_job = manager.create_job(params={})
    assert new_job.status == JobStatus.QUEUED


def test_job_lifecycle(manager, tmp_jobs_dir):
    job = manager.create_job(params={})
    job_id = job.job_id
    output = Path(tmp_jobs_dir) / job_id / "output.mp4"

    assert manager.mark_processing(job_id) is True
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


def test_startup_sweep_refuses_dangerous_path():
    """Verify startup_sweep refuses to sweep dangerous system directories."""
    for dangerous in ("/tmp", "/", "/var", "/home"):
        mgr = JobManager(jobs_dir=dangerous, ttl_seconds=10, max_queued=3)
        assert mgr.startup_sweep() == 0


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


import threading


def test_cancelled_is_a_status():
    from job_manager import JobStatus
    assert JobStatus.CANCELLED.value == "cancelled"


def test_job_has_cancel_event(manager):
    job = manager.create_job(params={})
    assert isinstance(job.cancel_event, threading.Event)
    assert not job.cancel_event.is_set()


def test_each_job_gets_its_own_cancel_event(manager):
    j1 = manager.create_job(params={})
    j2 = manager.create_job(params={})
    assert j1.cancel_event is not j2.cancel_event


def test_request_cancel_queued_marks_cancelled_and_deletes_input(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    # request_cancel on QUEUED must delete input file so it does not leak
    # (worker's finally block never runs for skipped-queued jobs).
    job.input_path.parent.mkdir(parents=True, exist_ok=True)
    job.input_path.write_bytes(b"fake video")
    assert job.input_path.exists()

    result = manager.request_cancel(job.job_id)

    assert result.status == JobStatus.CANCELLED
    assert result.cancel_event.is_set()
    assert result.completed_at is not None
    assert not job.input_path.exists()


def test_request_cancel_processing_sets_event_only(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    assert manager.mark_processing(job.job_id) is True
    result = manager.request_cancel(job.job_id)
    # Event set, but worker has not yet observed it.
    assert result.cancel_event.is_set()
    assert result.status == JobStatus.PROCESSING
    assert result.completed_at is None


def test_request_cancel_idempotent_on_cancelled(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    manager.request_cancel(job.job_id)
    result = manager.request_cancel(job.job_id)
    assert result.status == JobStatus.CANCELLED
    assert result.cancel_event.is_set()


def test_request_cancel_completed_raises_value_error(manager, tmp_jobs_dir):
    job = manager.create_job(params={})
    output = Path(tmp_jobs_dir) / job.job_id / "output.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch()
    manager.mark_completed(job.job_id, output_path=output, stats={})
    with pytest.raises(ValueError, match="terminal status"):
        manager.request_cancel(job.job_id)


def test_request_cancel_failed_raises_value_error(manager):
    job = manager.create_job(params={})
    manager.mark_failed(job.job_id, error="boom")
    with pytest.raises(ValueError, match="terminal status"):
        manager.request_cancel(job.job_id)


def test_request_cancel_unknown_raises_key_error(manager):
    with pytest.raises(KeyError):
        manager.request_cancel("nonexistent")


def test_mark_processing_cas_queued_returns_true(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    assert manager.mark_processing(job.job_id) is True
    assert manager.get_job(job.job_id).status == JobStatus.PROCESSING


def test_mark_processing_cas_cancelled_returns_false(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    manager.request_cancel(job.job_id)  # flips QUEUED → CANCELLED
    assert manager.mark_processing(job.job_id) is False
    # Must NOT overwrite CANCELLED with PROCESSING.
    assert manager.get_job(job.job_id).status == JobStatus.CANCELLED


def test_mark_processing_cas_completed_returns_false(manager, tmp_path):
    job = manager.create_job(params={})
    manager.mark_processing(job.job_id)
    manager.mark_completed(job.job_id, output_path=tmp_path / "out.mp4", stats={})
    assert manager.mark_processing(job.job_id) is False
    assert manager.get_job(job.job_id).status == JobStatus.COMPLETED


def test_mark_processing_cas_failed_returns_false(manager):
    job = manager.create_job(params={})
    manager.mark_processing(job.job_id)
    manager.mark_failed(job.job_id, error="boom")
    assert manager.mark_processing(job.job_id) is False
    assert manager.get_job(job.job_id).status == JobStatus.FAILED


def test_mark_cancelled_sets_status_and_completed_at(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    manager.mark_processing(job.job_id)
    manager.mark_cancelled(job.job_id)
    result = manager.get_job(job.job_id)
    assert result.status == JobStatus.CANCELLED
    assert result.completed_at is not None


def test_mark_cancelled_refuses_to_overwrite_completed(manager, tmp_jobs_dir):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    output = Path(tmp_jobs_dir) / job.job_id / "output.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch()
    manager.mark_completed(job.job_id, output_path=output, stats={})
    manager.mark_cancelled(job.job_id)  # must be no-op
    assert manager.get_job(job.job_id).status == JobStatus.COMPLETED


def test_mark_cancelled_refuses_to_overwrite_failed(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    manager.mark_failed(job.job_id, error="boom")
    manager.mark_cancelled(job.job_id)
    assert manager.get_job(job.job_id).status == JobStatus.FAILED


def test_cleanup_expired_includes_cancelled(manager, tmp_jobs_dir):
    from datetime import datetime, timedelta, timezone
    job = manager.create_job(params={})
    manager.mark_processing(job.job_id)
    manager.mark_cancelled(job.job_id)

    # Not yet expired
    assert manager.cleanup_expired() == 0

    # Backdate completed_at past TTL
    manager.get_job(job.job_id).completed_at = (
        datetime.now(tz=timezone.utc) - timedelta(seconds=20)
    )
    assert manager.cleanup_expired() == 1
    assert manager.get_job(job.job_id) is None
    assert not (Path(tmp_jobs_dir) / job.job_id).exists()
