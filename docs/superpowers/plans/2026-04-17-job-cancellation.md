# Job Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /jobs/{job_id}/cancel` to abort a queued or processing video annotation job; introduce a new terminal status `CANCELLED` with cooperative cancellation checked on every frame.

**Architecture:** A `threading.Event` on each `Job` is set by the event-loop thread in `JobManager.request_cancel` and polled by the executor thread at the top of both decode loops in `VideoAnnotator.annotate`. A new `JobCancelledError` is raised on the worker, which then calls `mark_cancelled` and deletes any partial output. Queue-time cancellation skips the job in the worker dispatch loop.

**Tech Stack:** Python 3.12, FastAPI, asyncio, `threading.Event`, pytest. Existing modules only — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-17-job-cancellation-design.md`

---

## File Structure

| File | Role |
|------|------|
| `app/job_manager.py` | Add `CANCELLED` enum value, `cancel_event` field on `Job`, `request_cancel`, `mark_cancelled`. |
| `app/video_annotator.py` | Add `JobCancelledError`; accept optional `cancel_event` in `annotate`, `_pass1_collect`, `_pass2_render`; check event at top of each per-frame loop. |
| `app/main.py` | Add `POST /jobs/{job_id}/cancel` endpoint. Worker: skip queued-but-cancelled jobs, pass `cancel_event` into annotator, catch `JobCancelledError`, delete partial output. |
| `tests/test_job_manager.py` | Tests for `request_cancel`, `mark_cancelled`, `CANCELLED` in TTL cleanup. |
| `tests/test_video_annotator.py` | Tests that `annotate` respects `cancel_event` in pass 1 and pass 2; no-op when `None`. |
| `tests/test_endpoints.py` | HTTP-level tests for the new endpoint (200/404/409). |
| `tests/test_worker.py` | Tests for worker's skip-queued-cancelled path and `JobCancelledError` handling. |
| `.claude/rules/api.md` | Document new endpoint. |
| `CLAUDE.md` | Add endpoint row to table. |

Tests import app modules flatly (`from job_manager import ...`) via `tests/conftest.py` `sys.path` injection. Follow the same convention for new imports.

---

## Task 1: Add `CANCELLED` status and `cancel_event` field

**Files:**
- Modify: `app/job_manager.py`
- Test: `tests/test_job_manager.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_job_manager.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_job_manager.py::test_cancelled_is_a_status tests/test_job_manager.py::test_job_has_cancel_event tests/test_job_manager.py::test_each_job_gets_its_own_cancel_event -v`

Expected: FAIL (`AttributeError: CANCELLED` or `AttributeError: cancel_event`).

- [ ] **Step 3: Implement**

In `app/job_manager.py`:

Add import near the top:

```python
import threading
```

Add to `JobStatus`:

```python
class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

Add field to `Job` dataclass (keep existing fields in place, add after `stats`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_job_manager.py -v`

Expected: all previous tests + 3 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/job_manager.py tests/test_job_manager.py
git commit -m "feat(jobs): add CANCELLED status and per-job cancel_event"
```

---

## Task 2: Implement `JobManager.request_cancel`

**Files:**
- Modify: `app/job_manager.py`
- Test: `tests/test_job_manager.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_job_manager.py`:

```python
def test_request_cancel_queued_marks_cancelled(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    result = manager.request_cancel(job.job_id)
    assert result.status == JobStatus.CANCELLED
    assert result.cancel_event.is_set()
    assert result.completed_at is not None


def test_request_cancel_processing_sets_event_only(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    manager.mark_processing(job.job_id)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_job_manager.py -k request_cancel -v`

Expected: FAIL (`AttributeError: 'JobManager' object has no attribute 'request_cancel'`).

- [ ] **Step 3: Implement `request_cancel`**

In `app/job_manager.py`, add this method inside `JobManager` (for example, after `mark_failed`):

```python
def request_cancel(self, job_id: str) -> Job:
    """Idempotent cancellation request.

    Raises:
        KeyError: if job_id is unknown.
        ValueError: if job is in a non-cancellable terminal status (COMPLETED or FAILED).
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

    logger.info(f"Job {job_id}: cancel requested (was {prev_status.value})")
    return job
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_job_manager.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/job_manager.py tests/test_job_manager.py
git commit -m "feat(jobs): add JobManager.request_cancel"
```

---

## Task 3: Implement `JobManager.mark_cancelled` and TTL inclusion

**Files:**
- Modify: `app/job_manager.py`
- Test: `tests/test_job_manager.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_job_manager.py`:

```python
def test_mark_cancelled_sets_status_and_completed_at(manager):
    from job_manager import JobStatus
    job = manager.create_job(params={})
    manager.mark_processing(job.job_id)
    manager.mark_cancelled(job.job_id)
    result = manager.get_job(job.job_id)
    assert result.status == JobStatus.CANCELLED
    assert result.completed_at is not None


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_job_manager.py -k "mark_cancelled or cleanup_expired_includes_cancelled" -v`

Expected: FAIL (no `mark_cancelled`; `CANCELLED` not handled in cleanup).

- [ ] **Step 3: Implement `mark_cancelled` and extend cleanup**

In `app/job_manager.py`, add inside `JobManager`:

```python
def mark_cancelled(self, job_id: str) -> None:
    """Called by the worker after JobCancelledError propagates out of annotate()."""
    job = self._jobs.get(job_id)
    if job is None:
        return
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.now(tz=timezone.utc)
    logger.info(f"Job cancelled: {job_id}")
```

Update `cleanup_expired` — change the status check:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_job_manager.py -v`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/job_manager.py tests/test_job_manager.py
git commit -m "feat(jobs): add mark_cancelled and include CANCELLED in TTL cleanup"
```

---

## Task 4: Add `JobCancelledError` and cancellation checks in `VideoAnnotator.annotate`

**Files:**
- Modify: `app/video_annotator.py`
- Test: `tests/test_video_annotator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_video_annotator.py`:

```python
import threading


class TestAnnotateCancellation:
    def _make_frames(self, num_frames: int, width: int = 640, height: int = 480):
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(num_frames)]

    def _make_decoder_mock(self, frames: list[np.ndarray]):
        mock_decoder = MagicMock()
        mock_decoder.read_frame.side_effect = list(frames) + [None]
        mock_decoder.__enter__ = MagicMock(return_value=mock_decoder)
        mock_decoder.__exit__ = MagicMock(return_value=False)
        return mock_decoder

    def _ffprobe_result(self, num_frames: int) -> MagicMock:
        stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": str(num_frames),
        }
        r = MagicMock()
        r.returncode = 0
        r.stdout = json.dumps({"streams": [stream]})
        return r

    def test_cancel_during_pass1_raises(self, mock_model, mock_visualizer, hw_config, tmp_path):
        from video_annotator import JobCancelledError

        num_frames = 20
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        decoder2 = self._make_decoder_mock(frames)
        mock_decoder_cls = MagicMock(side_effect=[decoder1, decoder2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        cancel_event = threading.Event()
        call_count = {"n": 0}

        def fake_predict(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                cancel_event.set()
            return [_make_yolo_result([(10, 20, 100, 200, 0, 0.9)])]

        mock_model.predict.side_effect = fake_predict

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # Pass 2 must never start (second decoder never used).
        assert decoder2.read_frame.call_count == 0
        # FFmpegDecoder __exit__ was invoked for pass 1.
        assert decoder1.__exit__.called

    def test_cancel_during_pass2_raises(self, mock_model, mock_visualizer, hw_config, tmp_path):
        from video_annotator import JobCancelledError

        num_frames = 10
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        decoder2 = self._make_decoder_mock(frames)
        mock_decoder_cls = MagicMock(side_effect=[decoder1, decoder2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        cancel_event = threading.Event()

        # Set event after pass 1 by hooking into write_frame (pass 2 only).
        write_calls = {"n": 0}

        def fake_write(frame):
            write_calls["n"] += 1
            if write_calls["n"] >= 2:
                cancel_event.set()

        mock_encoder.write_frame.side_effect = fake_write

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            with pytest.raises(JobCancelledError):
                annotator.annotate(
                    input_path, tmp_path / "out.mp4",
                    AnnotationParams(detect_every=1),
                    cancel_event=cancel_event,
                )

        # Fewer than all frames written in pass 2.
        assert mock_encoder.write_frame.call_count < num_frames

    def test_cancel_event_none_runs_to_completion(self, mock_model, mock_visualizer, hw_config, tmp_path):
        num_frames = 3
        frames = self._make_frames(num_frames)
        decoder1 = self._make_decoder_mock(frames)
        decoder2 = self._make_decoder_mock(frames)
        mock_decoder_cls = MagicMock(side_effect=[decoder1, decoder2])

        mock_encoder = MagicMock()
        mock_encoder.__enter__ = MagicMock(return_value=mock_encoder)
        mock_encoder.__exit__ = MagicMock(return_value=False)
        mock_encoder_cls = MagicMock(return_value=mock_encoder)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0),
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=self._ffprobe_result(num_frames)),
        ):
            # cancel_event omitted — behaviour unchanged.
            stats = annotator.annotate(
                input_path, tmp_path / "out.mp4",
                AnnotationParams(detect_every=1),
            )

        assert stats.total_frames == num_frames
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_video_annotator.py::TestAnnotateCancellation -v`

Expected: FAIL (`JobCancelledError` does not exist; `annotate` does not accept `cancel_event`).

- [ ] **Step 3: Implement**

In `app/video_annotator.py`:

Add near the top (after existing imports), add `threading`:

```python
import threading
```

Add the exception class (next to other module-level code, e.g. just above `VideoAnnotator`):

```python
class JobCancelledError(Exception):
    """Raised inside annotate() when cancel_event is set."""
```

Change `annotate` signature and pass `cancel_event` through to helpers:

```python
def annotate(
    self,
    input_path: Path,
    output_path: Path,
    params: AnnotationParams,
    progress_callback: Callable[[int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> AnnotationStats:
```

Inside `annotate`, propagate `cancel_event` to both passes:

```python
raw_detections, actual_frames = self._pass1_collect(
    input_path, metadata, params, yolo_conf,
    progress_callback, stats,
    cancel_event=cancel_event,
)
```

```python
self._pass2_render(
    input_path, output_path, metadata, params, stabilized,
    effective_codec, effective_crf, effective_bitrate,
    font_scale, actual_frames, progress_callback,
    cancel_event=cancel_event,
)
```

Update `_pass1_collect` signature:

```python
def _pass1_collect(
    self,
    input_path: Path,
    metadata: VideoMetadata,
    params: AnnotationParams,
    yolo_conf: float,
    progress_callback: Callable[[int], None] | None,
    stats: AnnotationStats,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[int, list[RawDetection]], int]:
```

Add the check at the top of the per-frame `while True:` loop, **before** `decoder.read_frame()`:

```python
with FFmpegDecoder(input_path, metadata.width, metadata.height, self.hw_config) as decoder:
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelledError()
        frame = decoder.read_frame()
        if frame is None:
            break
        ...
```

Update `_pass2_render` signature similarly:

```python
def _pass2_render(
    self,
    input_path: Path,
    output_path: Path,
    metadata: VideoMetadata,
    params: AnnotationParams,
    stabilized: dict[int, StabilizedFrame],
    effective_codec: str,
    effective_crf: int | None,
    effective_bitrate: int | None,
    font_scale: float,
    total_frames: int,
    progress_callback: Callable[[int], None] | None,
    cancel_event: threading.Event | None = None,
) -> None:
```

Add the check at the top of the pass-2 per-frame `while True:` loop, before `decoder.read_frame()`:

```python
while True:
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelledError()
    frame = decoder.read_frame()
    if frame is None:
        break
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_video_annotator.py -v`

Expected: all existing + 3 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add app/video_annotator.py tests/test_video_annotator.py
git commit -m "feat(annotator): cooperative cancel via threading.Event in both passes"
```

---

## Task 5: Worker — skip queued-but-cancelled, handle `JobCancelledError`

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_worker.py`:

```python
class TestAnnotationWorkerCancellation:
    @pytest.mark.asyncio
    async def test_skip_queued_but_cancelled(
        self, worker_app, worker_settings, worker_job_manager, tmp_path
    ):
        """Job cancelled while queued is skipped — annotator is never called."""
        job = worker_job_manager.create_job(params={})
        job.input_path.parent.mkdir(parents=True, exist_ok=True)
        job.input_path.touch()

        # Cancel before worker picks it up.
        worker_job_manager.request_cancel(job.job_id)

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
    async def test_cancel_during_processing_marks_cancelled(
        self, worker_app, worker_settings, worker_job_manager, tmp_path
    ):
        """JobCancelledError from annotator -> status CANCELLED, partial output deleted."""
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
```

Also adjust `_run_worker_until_job_done` to recognise `CANCELLED` as done. Update the helper (replace the existing definition near the top of the file):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_worker.py::TestAnnotationWorkerCancellation -v`

Expected: FAIL — `skip_queued_but_cancelled` hangs or fails because worker currently enters annotation path; `cancel_during_processing_marks_cancelled` fails because the generic `except Exception` branch maps the error to FAILED.

- [ ] **Step 3: Implement worker changes**

In `app/main.py`, update `_annotation_worker`:

(a) After fetching the job, skip it if its queued status was flipped to `CANCELLED`. Replace the existing block immediately after `job = job_manager.get_job(job_id)`:

```python
job_id = await job_manager.get_next_job_id()
job = job_manager.get_job(job_id)
if job is None:
    continue
if job.status != JobStatus.QUEUED:
    logger.info(f"Job {job_id} cancelled while queued, skipping")
    continue

job_manager.mark_processing(job_id)
```

(b) Import `JobCancelledError` at the top of `main.py` along with other `video_annotator` imports:

```python
from video_annotator import (
    AnnotationParams,
    JobCancelledError,
    VideoAnnotator,
)
```

(If the existing import style differs, match it — just ensure `JobCancelledError` is imported.)

(c) Pass `cancel_event=job.cancel_event` to `annotator.annotate(...)`:

```python
stats = await loop.run_in_executor(
    executor,
    lambda: annotator.annotate(
        input_path=job.input_path,
        output_path=output_path,
        params=params,
        progress_callback=progress_cb,
        cancel_event=job.cancel_event,
    ),
)
```

(d) Catch `JobCancelledError` separately, **before** the generic `Exception` branch. Replace the existing `try/except` block around the `run_in_executor` call:

```python
try:
    stats = await loop.run_in_executor(
        executor,
        lambda: annotator.annotate(
            input_path=job.input_path,
            output_path=output_path,
            params=params,
            progress_callback=progress_cb,
            cancel_event=job.cancel_event,
        ),
    )

    logger.info(
        f"Job {job_id} annotation finished: {stats.total_frames} frames, "
        f"{stats.processing_time_ms}ms"
    )
    job_manager.mark_completed(
        job_id,
        output_path=output_path,
        stats={
            "total_frames": stats.total_frames,
            "detected_frames": stats.detected_frames,
            "tracked_frames": stats.tracked_frames,
            "total_detections": stats.total_detections,
            "processing_time_ms": stats.processing_time_ms,
        },
    )

except JobCancelledError:
    logger.info(f"Job {job_id} cancelled during processing")
    job_manager.mark_cancelled(job_id)
    try:
        if output_path.exists():
            output_path.unlink()
    except OSError as e:
        logger.warning(f"Failed to remove partial output for {job_id}: {e}")

except Exception as e:
    logger.error(
        f"Annotation failed for job {job_id}: {e}",
        exc_info=True,
    )
    job_manager.mark_failed(job_id, str(e))
```

The existing `finally` block that deletes `input.mp4` stays untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_worker.py -v`

Expected: all tests pass (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_worker.py
git commit -m "feat(worker): handle job cancellation (queued skip + JobCancelledError)"
```

---

## Task 6: Add `POST /jobs/{job_id}/cancel` endpoint

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_endpoints.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_endpoints.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_endpoints.py::TestCancelJob -v`

Expected: FAIL (endpoint returns 404 / 405 because it does not exist yet).

- [ ] **Step 3: Implement the endpoint**

In `app/main.py`, add a new handler immediately after the `GET /jobs/{job_id}/download` endpoint:

```python
@app.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse, tags=["Jobs"])
async def cancel_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
):
    """Cancel a queued or processing video annotation job.

    Idempotent for already-cancelled jobs. Returns 409 when the job is in a
    terminal non-cancellable status (completed/failed). A client cancelling
    a PROCESSING job will see `status: "processing"` in the response; the
    worker flips the job to `cancelled` shortly after, observable via
    GET /jobs/{job_id}.
    """
    try:
        job = job_manager.request_cancel(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    stats = None
    if job.stats:
        stats = JobStats(**job.stats)

    download_url = None
    if job.status == JobStatus.COMPLETED:
        download_url = f"/jobs/{job.job_id}/download"

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        created_at=job.created_at.isoformat(),
        completed_at=(
            job.completed_at.isoformat() if job.completed_at else None
        ),
        download_url=download_url,
        error=job.error,
        stats=stats,
    )
```

No other files change — `JobStatus`, `JobStatusResponse`, and `JobStats` are already imported in `main.py` (used by `GET /jobs/{job_id}`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_endpoints.py -v`

Expected: all tests pass (existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_endpoints.py
git commit -m "feat(api): add POST /jobs/{job_id}/cancel"
```

---

## Task 7: Documentation

**Files:**
- Modify: `.claude/rules/api.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `.claude/rules/api.md`**

Add a new section after the existing Job-related documentation (or after `## Info Endpoints` if Job endpoints live there). Insert:

```markdown
### POST /jobs/{job_id}/cancel

Cancel a queued or processing video annotation job.

Cooperative cancellation: the running annotation stops at the next frame boundary (typically <1 second). A PROCESSING job will briefly show `status: "processing"` in the response; poll `GET /jobs/{job_id}` to observe the transition to `cancelled`.

**Response codes:**

| Case | Code |
|------|------|
| QUEUED or PROCESSING | 200 + `JobStatusResponse` |
| Already CANCELLED (idempotent) | 200 + `JobStatusResponse` |
| COMPLETED or FAILED | 409 Conflict |
| Unknown / TTL-expired | 404 Not Found |

**Example:**

```bash
curl -X POST http://localhost:3001/jobs/abc123def456/cancel
```

Response:
```json
{
  "job_id": "abc123def456",
  "status": "cancelled",
  "progress": 42,
  "created_at": "2026-04-17T12:00:00+00:00",
  "completed_at": "2026-04-17T12:01:30+00:00",
  "download_url": null,
  "error": null,
  "stats": null
}
```
```

- [ ] **Step 2: Update `CLAUDE.md`**

In the Endpoints table, add a new row after the `/jobs/{job_id}/download` row:

```markdown
| `/jobs/{job_id}/cancel` | POST | Cancel queued or running job |
```

- [ ] **Step 3: Commit**

```bash
git add .claude/rules/api.md CLAUDE.md
git commit -m "docs: document POST /jobs/{job_id}/cancel"
```

---

## Final Verification

- [ ] **Run full test suite**

Run: `python -m pytest tests/ -v`

Expected: all tests pass.

- [ ] **Smoke test manually (optional, requires running server)**

Terminal A:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Terminal B:
```bash
# Submit a job
JOB=$(curl -s -X POST http://localhost:8000/detect/video/visualize \
  -F "file=@sample.mp4" | python -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

# Cancel it
curl -X POST http://localhost:8000/jobs/$JOB/cancel

# Check status
curl http://localhost:8000/jobs/$JOB
```

Expected: `status` transitions to `"cancelled"` within ~1 second.
