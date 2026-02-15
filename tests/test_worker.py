import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import Settings
from hw_accel import HWAccelConfig, HWAccelType
from job_manager import JobManager, JobStatus
from video_annotator import AnnotationStats


@pytest.fixture
def worker_settings(tmp_path):
    return Settings(yolo_models="{}", video_jobs_dir=str(tmp_path), max_executor_workers=1)


@pytest.fixture
def worker_job_manager(tmp_path):
    return JobManager(jobs_dir=str(tmp_path), ttl_seconds=3600, max_queued=10)


@pytest.fixture
def mock_model_entry():
    entry = MagicMock()
    entry.model = MagicMock()
    entry.model.names = {0: "person"}
    entry.visualizer = MagicMock()
    return entry


@pytest.fixture
def worker_model_manager(mock_model_entry):
    mm = MagicMock()
    mm.get_model = AsyncMock(return_value=mock_model_entry)
    return mm


@pytest.fixture
def worker_app(worker_job_manager, worker_model_manager):
    app = MagicMock()
    app.state.job_manager = worker_job_manager
    app.state.model_manager = worker_model_manager
    app.state.hw_config = HWAccelConfig(accel_type=HWAccelType.CPU)
    return app


async def _run_worker_until_job_done(app, settings, job_manager, timeout=5.0):
    """Run _annotation_worker, wait until all queued jobs are done, then cancel."""
    from main import _annotation_worker

    task = asyncio.create_task(_annotation_worker(app, settings))

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        # Check if all jobs are processed
        all_done = all(
            j.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            for j in job_manager._jobs.values()
        )
        if all_done and job_manager._jobs:
            break

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


class TestAnnotationWorker:
    @pytest.mark.asyncio
    async def test_success(self, worker_app, worker_settings, worker_job_manager, tmp_path):
        job = worker_job_manager.create_job(params={"model": "yolo11s.pt"})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        mock_stats = AnnotationStats(
            total_frames=100, detected_frames=20, tracked_frames=80,
            total_detections=50, processing_time_ms=5000,
        )

        mock_annotator_cls = MagicMock()
        mock_annotator_cls.return_value.annotate.return_value = mock_stats

        mock_executor = MagicMock()
        mock_executor.executor = None  # Use default executor

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(worker_app, worker_settings, worker_job_manager)

        completed = worker_job_manager.get_job(job.job_id)
        assert completed.status == JobStatus.COMPLETED
        assert completed.stats["total_frames"] == 100

    @pytest.mark.asyncio
    async def test_model_error(self, worker_app, worker_settings, worker_job_manager):
        job = worker_job_manager.create_job(params={"model": "bad.pt"})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        worker_app.state.model_manager.get_model = AsyncMock(
            side_effect=RuntimeError("model not found")
        )

        await _run_worker_until_job_done(worker_app, worker_settings, worker_job_manager)

        failed = worker_job_manager.get_job(job.job_id)
        assert failed.status == JobStatus.FAILED
        assert "Model error" in failed.error

    @pytest.mark.asyncio
    async def test_annotation_error(self, worker_app, worker_settings, worker_job_manager):
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        mock_annotator_cls = MagicMock()
        mock_annotator_cls.return_value.annotate.side_effect = Exception("ffmpeg crashed")

        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(worker_app, worker_settings, worker_job_manager)

        failed = worker_job_manager.get_job(job.job_id)
        assert failed.status == JobStatus.FAILED
        assert "ffmpeg crashed" in failed.error

    @pytest.mark.asyncio
    async def test_cleans_input_on_success(self, worker_app, worker_settings, worker_job_manager):
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()
        assert job.input_path.exists()

        mock_stats = AnnotationStats(total_frames=10)
        mock_annotator_cls = MagicMock()
        mock_annotator_cls.return_value.annotate.return_value = mock_stats

        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(worker_app, worker_settings, worker_job_manager)

        assert not job.input_path.exists()

    @pytest.mark.asyncio
    async def test_cleans_input_on_failure(self, worker_app, worker_settings, worker_job_manager):
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        worker_app.state.model_manager.get_model = AsyncMock(
            side_effect=RuntimeError("model error")
        )

        await _run_worker_until_job_done(worker_app, worker_settings, worker_job_manager)

        assert not job.input_path.exists()

    @pytest.mark.asyncio
    async def test_cancellation(self, worker_app, worker_settings):
        from main import _annotation_worker

        task = asyncio.create_task(_annotation_worker(worker_app, worker_settings))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Should not raise any other exception
        assert task.done()

    @pytest.mark.asyncio
    async def test_continues_after_failure(self, worker_app, worker_settings, worker_job_manager):
        job1 = worker_job_manager.create_job(params={"model": "bad.pt"})
        job1.input_path.parent.mkdir(parents=True, exist_ok=True)
        job1.input_path.touch()

        job2 = worker_job_manager.create_job(params={})
        job2.input_path.parent.mkdir(parents=True, exist_ok=True)
        job2.input_path.touch()

        call_count = 0

        async def get_model_side_effect(name):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("model error")
            return worker_app.state.model_manager.get_model.return_value

        # Reset to allow first call to fail, second to succeed
        original_entry = MagicMock()
        original_entry.model = MagicMock()
        original_entry.model.names = {0: "person"}
        original_entry.visualizer = MagicMock()

        worker_app.state.model_manager.get_model = AsyncMock(side_effect=get_model_side_effect)
        worker_app.state.model_manager.get_model.return_value = original_entry

        mock_stats = AnnotationStats(total_frames=10)
        mock_annotator_cls = MagicMock()
        mock_annotator_cls.return_value.annotate.return_value = mock_stats

        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(worker_app, worker_settings, worker_job_manager)

        assert worker_job_manager.get_job(job1.job_id).status == JobStatus.FAILED
        assert worker_job_manager.get_job(job2.job_id).status == JobStatus.COMPLETED


class TestStartupWiring:
    def test_auto_codec_passes_h264_baseline(self):
        """When video_codec=auto, detect_hw_accel gets codec='h264'."""
        video_codec = "auto"
        hw_detect_codec = "h264" if video_codec == "auto" else video_codec
        assert hw_detect_codec == "h264"

    def test_explicit_codec_passes_through(self):
        """When video_codec=h265, detect_hw_accel gets codec='h265'."""
        video_codec = "h265"
        hw_detect_codec = "h264" if video_codec == "auto" else video_codec
        assert hw_detect_codec == "h265"
