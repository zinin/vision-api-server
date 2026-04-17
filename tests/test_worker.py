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
    terminal = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        all_done = all(j.status in terminal for j in job_manager._jobs.values())
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
        job = worker_job_manager.create_job(params={"model": "yolo26s.pt"})
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


class TestAnnotationWorkerCancellation:
    @pytest.mark.asyncio
    async def test_skip_queued_but_cancelled(
        self, worker_app, worker_settings, worker_job_manager, tmp_path
    ):
        """Job cancelled while queued is skipped — annotator never called,
        queue is drained, input file removed (by request_cancel)."""
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.write_bytes(b"fake")

        # Cancel before worker picks it up. request_cancel also deletes input.
        worker_job_manager.request_cancel(job.job_id)
        assert not job.input_path.exists()

        mock_annotator_cls = MagicMock()
        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(
                worker_app, worker_settings, worker_job_manager
            )

        assert worker_job_manager.get_job(job.job_id).status == JobStatus.CANCELLED
        mock_annotator_cls.assert_not_called()
        # Worker must have pulled the id out of the queue.
        assert worker_job_manager._queue.empty()

    @pytest.mark.asyncio
    async def test_cancel_during_processing_marks_cancelled(
        self, worker_app, worker_settings, worker_job_manager, tmp_path
    ):
        """JobCancelledError from annotator → status CANCELLED,
        partial output deleted, and cancel_event was really passed in."""
        from video_annotator import JobCancelledError

        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        # Simulate a partial output file written before cancellation.
        output_path = job.input_path.parent / "output.mp4"
        output_path.write_bytes(b"partial")

        mock_annotator_cls = MagicMock()
        mock_annotator_cls.return_value.annotate.side_effect = JobCancelledError()

        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(
                worker_app, worker_settings, worker_job_manager
            )

        final = worker_job_manager.get_job(job.job_id)
        assert final.status == JobStatus.CANCELLED
        assert final.error is None
        assert not output_path.exists()
        # Assert the worker passed the job's cancel_event into annotate().
        call_kwargs = mock_annotator_cls.return_value.annotate.call_args.kwargs
        assert call_kwargs["cancel_event"] is job.cancel_event

    @pytest.mark.asyncio
    async def test_cancel_during_model_load(
        self, worker_app, worker_settings, worker_job_manager, tmp_path
    ):
        """If cancel arrives while get_model() is in flight, worker must
        observe the event right after and never construct the annotator."""
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        real_get_model = worker_app.state.model_manager.get_model

        async def slow_get_model(name=None):
            # Simulate /cancel arriving during model load.
            worker_job_manager.request_cancel(job.job_id)
            return await real_get_model(name)

        worker_app.state.model_manager.get_model = AsyncMock(side_effect=slow_get_model)

        mock_annotator_cls = MagicMock()
        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(
                worker_app, worker_settings, worker_job_manager
            )

        assert worker_job_manager.get_job(job.job_id).status == JobStatus.CANCELLED
        mock_annotator_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_precedence_over_model_load_failure(
        self, worker_app, worker_settings, worker_job_manager, tmp_path
    ):
        """If cancel is set and get_model() also fails, status must be
        CANCELLED (cancel wins over pre-annotate failure)."""
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        async def cancel_then_fail(name=None):
            worker_job_manager.request_cancel(job.job_id)
            raise RuntimeError("model not found")

        worker_app.state.model_manager.get_model = AsyncMock(side_effect=cancel_then_fail)

        mock_annotator_cls = MagicMock()
        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(
                worker_app, worker_settings, worker_job_manager
            )

        final = worker_job_manager.get_job(job.job_id)
        assert final.status == JobStatus.CANCELLED
        assert final.error is None
        mock_annotator_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_failure_marks_failed(
        self, worker_app, worker_settings, worker_job_manager, tmp_path
    ):
        """Setup-time exception (e.g. VideoAnnotator construction) must
        terminalize the job as FAILED, not leave it stuck in PROCESSING."""
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        # VideoAnnotator constructor raises.
        mock_annotator_cls = MagicMock(side_effect=RuntimeError("boom"))
        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(
                worker_app, worker_settings, worker_job_manager
            )

        final = worker_job_manager.get_job(job.job_id)
        assert final.status == JobStatus.FAILED
        assert "Setup error" in final.error

    @pytest.mark.asyncio
    async def test_cancel_precedence_over_setup_failure(
        self, worker_app, worker_settings, worker_job_manager, tmp_path
    ):
        """If cancel is set and setup raises, status must be CANCELLED."""
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        def annotator_side_effect(*args, **kwargs):
            worker_job_manager.request_cancel(job.job_id)
            raise RuntimeError("boom during setup")

        mock_annotator_cls = MagicMock(side_effect=annotator_side_effect)
        mock_executor = MagicMock()
        mock_executor.executor = None

        with (
            patch("main.VideoAnnotator", mock_annotator_cls),
            patch("main.get_executor", return_value=mock_executor),
        ):
            await _run_worker_until_job_done(
                worker_app, worker_settings, worker_job_manager
            )

        final = worker_job_manager.get_job(job.job_id)
        assert final.status == JobStatus.CANCELLED
        assert final.error is None
