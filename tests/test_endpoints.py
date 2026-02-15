import io
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
