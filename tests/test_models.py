from models import JobCreatedResponse, JobStatusResponse, JobStats


def test_job_created_response():
    resp = JobCreatedResponse(
        job_id="abc123",
        status="queued",
        message="Video annotation job created",
    )
    data = resp.model_dump()
    assert data["job_id"] == "abc123"
    assert data["status"] == "queued"


def test_job_status_response_processing():
    resp = JobStatusResponse(
        job_id="abc123",
        status="processing",
        progress=45,
        created_at="2026-02-14T12:00:00",
    )
    data = resp.model_dump()
    assert data["progress"] == 45
    assert data["download_url"] is None
    assert data["stats"] is None


def test_job_status_response_completed():
    stats = JobStats(
        total_frames=900,
        detected_frames=180,
        tracked_frames=720,
        total_detections=1250,
        processing_time_ms=45000,
    )
    resp = JobStatusResponse(
        job_id="abc123",
        status="completed",
        progress=100,
        created_at="2026-02-14T12:00:00",
        download_url="/jobs/abc123/download",
        stats=stats,
    )
    data = resp.model_dump()
    assert data["stats"]["total_frames"] == 900
    assert data["download_url"] == "/jobs/abc123/download"
