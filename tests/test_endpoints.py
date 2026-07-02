import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import Settings, get_settings
from dependencies import get_job_manager, get_model_manager
from job_manager import JobManager
from main import app


@asynccontextmanager
async def _noop_lifespan(app: FastAPI):
    yield


@pytest.fixture
def test_settings(tmp_path):
    return Settings(yolo_models="{}", video_jobs_dir=str(tmp_path))


@pytest.fixture
def job_manager_for_tests(tmp_path):
    return JobManager(jobs_dir=str(tmp_path), ttl_seconds=3600, max_queued=10)


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    entry = MagicMock()
    entry.model = MagicMock()
    entry.model.names = {0: "person", 1: "car"}
    entry.visualizer = MagicMock()
    entry.model_name = "yolo26s.pt"
    mm.get_model = AsyncMock(return_value=entry)
    return mm


@pytest.fixture
def client(test_settings, job_manager_for_tests, mock_model_manager):
    app.router.lifespan_context = _noop_lifespan
    app.dependency_overrides[get_job_manager] = lambda: job_manager_for_tests
    app.dependency_overrides[get_model_manager] = lambda: mock_model_manager
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _make_video_file(content: bytes = b"fake video data", filename: str = "test.mp4"):
    return ("file", (filename, io.BytesIO(content), "video/mp4"))


# --- POST /detect/video/visualize ---

class TestAnnotateVideo:
    def test_submit_success(self, client, job_manager_for_tests):
        resp = client.post("/detect/video/visualize", files=[_make_video_file()])
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_invalid_format(self, client):
        resp = client.post(
            "/detect/video/visualize",
            files=[("file", ("video.txt", io.BytesIO(b"data"), "text/plain"))],
        )
        assert resp.status_code == 400
        assert "Invalid video format" in resp.json()["detail"]

    def test_queue_full(self, client, job_manager_for_tests):
        job_manager_for_tests.max_queued = 1
        job_manager_for_tests.create_job(params={})
        resp = client.post("/detect/video/visualize", files=[_make_video_file()])
        assert resp.status_code == 429

    def test_too_large(self, client, test_settings):
        with patch("main.MAX_VIDEO_SIZE", 10):
            resp = client.post(
                "/detect/video/visualize",
                files=[_make_video_file(content=b"x" * 100)],
            )
        assert resp.status_code == 413
        # Verify temp file is cleaned up
        tmp_files = list(Path(test_settings.video_jobs_dir).glob("*.tmp"))
        assert tmp_files == [], f"Temp files not cleaned up: {tmp_files}"

    def test_invalid_model(self, client, mock_model_manager):
        mock_model_manager.get_model = AsyncMock(side_effect=RuntimeError("not found"))
        resp = client.post(
            "/detect/video/visualize?model=bad.pt",
            files=[_make_video_file()],
        )
        assert resp.status_code == 400

    def test_classes_parsed(self, client, job_manager_for_tests):
        resp = client.post(
            "/detect/video/visualize?classes=person,car",
            files=[_make_video_file()],
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        job = job_manager_for_tests.get_job(job_id)
        assert job.params["classes"] == ["person", "car"]


# --- GET /jobs/{job_id} ---

class TestJobStatus:
    def test_queued(self, client, job_manager_for_tests):
        job = job_manager_for_tests.create_job(params={})
        resp = client.get(f"/jobs/{job.job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert data["progress"] == 0

    def test_completed(self, client, job_manager_for_tests, tmp_path):
        job = job_manager_for_tests.create_job(params={})
        output = tmp_path / job.job_id / "output.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        job_manager_for_tests.mark_completed(
            job.job_id,
            output_path=output,
            stats={
                "total_frames": 100,
                "detected_frames": 20,
                "tracked_frames": 80,
                "total_detections": 50,
                "processing_time_ms": 5000,
            },
        )
        resp = client.get(f"/jobs/{job.job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["download_url"] == f"/jobs/{job.job_id}/download"
        assert data["stats"]["total_frames"] == 100

    def test_failed(self, client, job_manager_for_tests):
        job = job_manager_for_tests.create_job(params={})
        job_manager_for_tests.mark_failed(job.job_id, error="boom")
        resp = client.get(f"/jobs/{job.job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["error"] == "boom"

    def test_not_found(self, client):
        resp = client.get("/jobs/nonexistent")
        assert resp.status_code == 404


# --- GET /jobs/{job_id}/download ---

class TestJobDownload:
    def test_success(self, client, job_manager_for_tests, tmp_path):
        job = job_manager_for_tests.create_job(params={})
        output = tmp_path / job.job_id / "output.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake video content")
        job_manager_for_tests.mark_completed(
            job.job_id,
            output_path=output,
            stats={
                "total_frames": 10,
                "detected_frames": 2,
                "tracked_frames": 8,
                "total_detections": 5,
                "processing_time_ms": 1000,
            },
        )
        resp = client.get(f"/jobs/{job.job_id}/download")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert resp.content == b"fake video content"

    def test_not_ready(self, client, job_manager_for_tests):
        job = job_manager_for_tests.create_job(params={})
        resp = client.get(f"/jobs/{job.job_id}/download")
        assert resp.status_code == 400
        assert "not ready" in resp.json()["detail"].lower()

    def test_not_found(self, client):
        resp = client.get("/jobs/nonexistent/download")
        assert resp.status_code == 404

    def test_missing_file(self, client, job_manager_for_tests, tmp_path):
        job = job_manager_for_tests.create_job(params={})
        missing_output = tmp_path / "does_not_exist.mp4"
        job_manager_for_tests.mark_completed(
            job.job_id,
            output_path=missing_output,
            stats={
                "total_frames": 10,
                "detected_frames": 2,
                "tracked_frames": 8,
                "total_detections": 5,
                "processing_time_ms": 1000,
            },
        )
        resp = client.get(f"/jobs/{job.job_id}/download")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestCancelJob:
    def test_cancel_queued(self, client, job_manager_for_tests):
        from job_manager import JobStatus
        job = job_manager_for_tests.create_job(params={})
        resp = client.post(f"/jobs/{job.job_id}/cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job.job_id
        assert body["status"] == "cancelled"
        assert job_manager_for_tests.get_job(job.job_id).status == JobStatus.CANCELLED

    def test_cancel_idempotent(self, client, job_manager_for_tests):
        job = job_manager_for_tests.create_job(params={})
        client.post(f"/jobs/{job.job_id}/cancel")
        resp = client.post(f"/jobs/{job.job_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_processing_returns_200_with_processing_status(
        self, client, job_manager_for_tests
    ):
        from job_manager import JobStatus
        job = job_manager_for_tests.create_job(params={})
        assert job_manager_for_tests.mark_processing(job.job_id) is True

        resp = client.post(f"/jobs/{job.job_id}/cancel")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processing"
        # Event was set; worker will flip to CANCELLED later.
        assert job.cancel_event.is_set()
        # Status NOT flipped by the endpoint — still PROCESSING.
        assert job_manager_for_tests.get_job(job.job_id).status == JobStatus.PROCESSING

    def test_cancel_processing_idempotent(self, client, job_manager_for_tests):
        """Repeated /cancel on PROCESSING job is idempotent: 200 + status unchanged."""
        from job_manager import JobStatus
        job = job_manager_for_tests.create_job(params={})
        assert job_manager_for_tests.mark_processing(job.job_id) is True

        # First cancel: sets event, status stays PROCESSING
        resp1 = client.post(f"/jobs/{job.job_id}/cancel")
        assert resp1.status_code == 200
        assert resp1.json()["status"] == "processing"
        assert job.cancel_event.is_set()

        # Second cancel: still 200, still processing, event already set
        resp2 = client.post(f"/jobs/{job.job_id}/cancel")
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "processing"
        assert job.cancel_event.is_set()
        # Status stable — worker hasn't run in this test.
        assert job_manager_for_tests.get_job(job.job_id).status == JobStatus.PROCESSING

    def test_cancel_unknown_returns_404(self, client):
        resp = client.post("/jobs/does-not-exist/cancel")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Job not found"

    def test_cancel_completed_returns_409(self, client, job_manager_for_tests, tmp_path):
        from pathlib import Path
        job = job_manager_for_tests.create_job(params={})
        output = Path(job_manager_for_tests.jobs_dir) / job.job_id / "output.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        job_manager_for_tests.mark_completed(job.job_id, output_path=output, stats={})
        resp = client.post(f"/jobs/{job.job_id}/cancel")
        assert resp.status_code == 409
        assert "terminal status" in resp.json()["detail"]

    def test_cancel_failed_returns_409(self, client, job_manager_for_tests):
        job = job_manager_for_tests.create_job(params={})
        job_manager_for_tests.mark_failed(job.job_id, error="nope")
        resp = client.post(f"/jobs/{job.job_id}/cancel")
        assert resp.status_code == 409


# --- GET /health fd observability ---

MAIN_LOGGER = "main"


class TestHealthFdStats:
    @pytest.fixture(autouse=True)
    def _reset_fd_warning_state(self):
        import main
        main._last_fd_warning_ts = None
        yield
        main._last_fd_warning_ts = None

    def test_health_reports_fd_stats(self, client):
        import main
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        if main.resource is not None:
            assert data["fd_soft_limit"] > 0
        else:  # Windows: resource module absent -> sentinel 0
            assert data["fd_soft_limit"] == 0
        if sys.platform.startswith("linux"):
            assert data["open_fds"] > 0
            assert data["fd_deleted"] >= 0

    def test_health_passes_deleted_count_through(self, client):
        with patch("main._fd_stats", return_value=(150, 42, 1000)):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["fd_deleted"] == 42

    def test_health_warns_when_fd_usage_high(self, client, caplog):
        with patch("main._fd_stats", return_value=(900, 0, 1000)):
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["open_fds"] == 900
        assert resp.json()["fd_soft_limit"] == 1000
        assert "fd usage" in caplog.text.lower()

    def test_health_warning_rate_limited(self, client, caplog):
        with patch("main._fd_stats", return_value=(900, 0, 1000)):
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                client.get("/health")          # first hit over the threshold logs
                assert "fd usage" in caplog.text.lower()
                caplog.clear()
                client.get("/health")          # second hit within the hour must NOT
        assert "fd usage" not in caplog.text.lower()

    def test_health_warning_fires_again_after_expiry(self, client, caplog):
        import main
        with patch("main._fd_stats", return_value=(900, 0, 1000)):
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                client.get("/health")               # arms the rate limiter
                caplog.clear()
                main._last_fd_warning_ts -= 3601.0  # pretend the hour has passed
                client.get("/health")
        assert "fd usage" in caplog.text.lower()

    def test_health_no_warning_at_normal_usage(self, client, caplog):
        with patch("main._fd_stats", return_value=(100, 0, 1000)):
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["open_fds"] == 100
        assert "fd usage" not in caplog.text.lower()

    def test_health_handles_missing_procfs(self, client, caplog):
        with patch("main._fd_stats", return_value=(None, None, 1000)):
            with caplog.at_level(logging.WARNING, logger=MAIN_LOGGER):
                resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["open_fds"] is None
        assert resp.json()["fd_deleted"] is None
        assert "fd usage" not in caplog.text.lower()

    def test_health_survives_emfile_in_ffmpeg_check(self, client):
        with patch("main._fd_stats", return_value=(100, 0, 1000)):
            with patch("main.VideoFrameExtractor", side_effect=OSError(24, "Too many open files")):
                resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["video_processing"] is False
        assert data["open_fds"] == 100
        assert data["fd_soft_limit"] == 1000


class TestFdStatsHelper:
    """Direct unit tests for the _fd_stats() degradation branches."""

    def test_no_resource_module(self, monkeypatch):
        import main
        monkeypatch.setattr(main, "resource", None)
        assert main._fd_stats() == (None, None, 0)

    @pytest.mark.skipif(sys.platform == "win32", reason="requires the resource module")
    def test_procfs_unavailable(self):
        import main
        soft_limit = main.resource.getrlimit(main.resource.RLIMIT_NOFILE)[0]
        with patch("main.os.listdir", side_effect=OSError(24, "Too many open files")):
            assert main._fd_stats() == (None, None, soft_limit)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="requires /proc")
    def test_counts_deleted_fd(self, tmp_path):
        import main
        victim = tmp_path / "leaked.tmp"
        victim.write_text("x")
        handle = victim.open("r")
        try:
            victim.unlink()
            open_fds, fd_deleted, _ = main._fd_stats()
        finally:
            handle.close()
        assert open_fds is not None and open_fds > 0
        assert fd_deleted >= 1
