# Video Annotation (VAS-2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add async video annotation endpoint that takes input video and returns video with YOLO-detected objects highlighted with bounding boxes, using CSRT tracker between detection frames.

**Architecture:** POST endpoint accepts video, creates async job (returns 202 + job_id). Background worker decodes video frame-by-frame, runs YOLO every Nth frame, uses OpenCV CSRT tracker for intermediate frames, draws bboxes, encodes output video, merges original audio via FFmpeg. Client polls job status and downloads result.

**Tech Stack:** FastAPI, Ultralytics YOLO, OpenCV (CSRT tracker from opencv-contrib), FFmpeg, asyncio (Queue + background worker)

**Design doc:** `docs/plans/2026-02-14-video-annotation-design.md`

---

### Task 1: Update Requirements and Set Up Test Infrastructure

**Files:**
- Modify: `requirements.txt:5`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Update opencv dependency**

In `requirements.txt`, replace line 5:
```
opencv-python-headless>=4.12.0.88,<5.0.0
```
with:
```
opencv-contrib-python-headless>=4.12.0.88,<5.0.0
```

This is a drop-in replacement that adds the `cv2.legacy` module with CSRT tracker. Also add `httpx` for FastAPI TestClient:
```
httpx>=0.28.0,<1.0.0
```

**Step 2: Create test directory**

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

# Add app directory to path so tests can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
```

**Step 3: Verify pytest works**

Run: `cd /opt/github/zinin/vision-api-server && python -m pytest tests/ -v --co`
Expected: "no tests ran" (collected 0 items)

**Step 4: Commit**

```bash
git add requirements.txt tests/
git commit -m "feat: add opencv-contrib for CSRT tracker, set up test infrastructure"
```

---

### Task 2: Add Configuration Settings

**Files:**
- Modify: `app/config.py:22` (after `max_executor_workers`)
- Create: `tests/test_config.py`

**Step 1: Write failing test**

Create `tests/test_config.py`:
```python
from config import Settings


def test_video_job_settings_defaults():
    s = Settings(yolo_models="{}")
    assert s.video_job_ttl == 3600
    assert s.video_jobs_dir == "/tmp/vision_jobs"
    assert s.max_queued_jobs == 10
    assert s.default_detect_every == 5
```

**Step 2: Run test to verify it fails**

Run: `cd /opt/github/zinin/vision-api-server && python -m pytest tests/test_config.py::test_video_job_settings_defaults -v`
Expected: FAIL with `AttributeError` — fields don't exist yet

**Step 3: Add settings to config.py**

In `app/config.py`, after line 22 (`max_executor_workers: int = 4`), add:
```python
    # Video annotation job settings
    video_job_ttl: int = 3600  # 1 hour TTL for completed jobs
    video_jobs_dir: str = "/tmp/vision_jobs"
    max_queued_jobs: int = 10
    default_detect_every: int = 5
```

**Step 4: Run test to verify it passes**

Run: `cd /opt/github/zinin/vision-api-server && python -m pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add video annotation job settings to config"
```

---

### Task 3: Add Pydantic Response Models

**Files:**
- Modify: `app/models.py:160` (append at end)
- Create: `tests/test_models.py`

**Step 1: Write failing test**

Create `tests/test_models.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd /opt/github/zinin/vision-api-server && python -m pytest tests/test_models.py -v`
Expected: FAIL with `ImportError` — classes don't exist yet

**Step 3: Add models to models.py**

Append to `app/models.py` after line 160:
```python


class JobCreatedResponse(BaseModel):
    """Response when a video annotation job is created."""
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status")
    message: str = Field(description="Human-readable message")


class JobStats(BaseModel):
    """Statistics for a completed annotation job."""
    total_frames: int = Field(description="Total frames in video")
    detected_frames: int = Field(description="Frames with YOLO detection")
    tracked_frames: int = Field(description="Frames with tracker only")
    total_detections: int = Field(description="Total object detections")
    processing_time_ms: int = Field(description="Total processing time in ms")


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status: queued, processing, completed, failed")
    progress: int = Field(ge=0, le=100, description="Progress percentage")
    created_at: str = Field(description="Job creation timestamp ISO format")
    completed_at: str | None = Field(default=None, description="Completion timestamp")
    download_url: str | None = Field(default=None, description="URL to download result")
    error: str | None = Field(default=None, description="Error message if failed")
    stats: JobStats | None = Field(default=None, description="Job statistics")
```

**Step 4: Run tests to verify they pass**

Run: `cd /opt/github/zinin/vision-api-server && python -m pytest tests/test_models.py -v`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add Pydantic response models for video annotation jobs"
```

---

### Task 4: Create JobManager

**Files:**
- Create: `app/job_manager.py`
- Create: `tests/test_job_manager.py`

**Step 1: Write failing tests**

Create `tests/test_job_manager.py`:
```python
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
    import time

    job = manager.create_job(params={})
    job_id = job.job_id
    output = Path(tmp_jobs_dir) / job_id / "output.mp4"
    output.touch()
    manager.mark_completed(job_id, output_path=output, stats={})

    # Not expired yet
    assert manager.cleanup_expired() == 0

    # Fake expiry by backdating completed_at
    from datetime import datetime, timedelta
    manager.get_job(job_id).completed_at = datetime.now() - timedelta(seconds=20)

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

    mgr = JobManager(jobs_dir=tmp_jobs_dir, ttl_seconds=10, max_queued=3)
    removed = mgr.startup_sweep()
    assert removed == 2
    assert not (jobs_dir / "orphan1").exists()
    assert not (jobs_dir / "orphan2").exists()
```

**Step 2: Run tests to verify they fail**

Run: `cd /opt/github/zinin/vision-api-server && python -m pytest tests/test_job_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'job_manager'`

**Step 3: Implement JobManager**

Create `app/job_manager.py`:
```python
import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    progress: int = 0
    created_at: datetime = field(default_factory=datetime.now)
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

    def create_job(self, params: dict) -> Job:
        queued_count = sum(
            1 for j in self._jobs.values() if j.status == JobStatus.QUEUED
        )
        if queued_count >= self.max_queued:
            raise RuntimeError(
                f"Too many queued jobs ({queued_count}/{self.max_queued})"
            )

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
            job.status = JobStatus.PROCESSING
            logger.info(f"Job processing: {job_id}")

    def mark_completed(
        self, job_id: str, output_path: Path, stats: dict
    ) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.completed_at = datetime.now()
            job.output_path = output_path
            job.stats = stats
            logger.info(f"Job completed: {job_id}")

    def mark_failed(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
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
        """Delete all job directories on startup (orphan cleanup after restart)."""
        if not self.jobs_dir.exists():
            return 0
        count = 0
        for entry in self.jobs_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
                count += 1
        if count:
            logger.info(f"Startup sweep: removed {count} orphan job dir(s)")
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /opt/github/zinin/vision-api-server && python -m pytest tests/test_job_manager.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add app/job_manager.py tests/test_job_manager.py
git commit -m "feat: add JobManager with async queue and TTL cleanup"
```

---

### Task 5: Create VideoAnnotator

**Files:**
- Create: `app/video_annotator.py`
- Modify: `app/visualization.py` (rename private methods to public)

No unit tests for this task — VideoAnnotator requires YOLO model, video files, and FFmpeg. It will be tested via integration in Task 7. TODO: add unit tests with mocked cv2/subprocess later.

**Step 0: Make visualization methods public**

In `app/visualization.py`, rename:
- `_draw_detection` → `draw_detection` (line 157)
- `_calculate_adaptive_font_scale` → `calculate_adaptive_font_scale` (line 86)

Update all internal callers in `draw_yolo_results` accordingly.

**Step 1: Create video_annotator.py**

Create `app/video_annotator.py`:
```python
import logging
import subprocess
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from visualization import DetectionVisualizer, DetectionBox

logger = logging.getLogger(__name__)


def _create_csrt_tracker() -> cv2.Tracker:
    """Create CSRT tracker, handling different OpenCV versions."""
    if hasattr(cv2, "TrackerCSRT"):
        return cv2.TrackerCSRT.create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT"):
        return cv2.legacy.TrackerCSRT.create()
    raise RuntimeError(
        "CSRT tracker not available. Install opencv-contrib-python-headless."
    )


@dataclass
class AnnotationParams:
    conf: float = 0.5
    imgsz: int = 640
    max_det: int = 100
    detect_every: int = 5
    classes: list[str] | None = None
    line_width: int = 2
    show_labels: bool = True
    show_conf: bool = True


@dataclass
class AnnotationStats:
    total_frames: int = 0
    detected_frames: int = 0
    tracked_frames: int = 0
    total_detections: int = 0
    processing_time_ms: int = 0


class VideoAnnotator:
    """Annotate video with YOLO detections and CSRT tracking."""

    def __init__(
        self,
        model: Any,
        visualizer: DetectionVisualizer,
        class_names: dict[int, str],
    ):
        self.model = model
        self.visualizer = visualizer
        self.class_names = class_names

    def annotate(
        self,
        input_path: Path,
        output_path: Path,
        params: AnnotationParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> AnnotationStats:
        """
        Annotate video with bounding boxes.

        This is a blocking method — run in a thread pool from async code.

        Args:
            input_path: Path to input video
            output_path: Path for final output (with audio)
            params: Annotation parameters
            progress_callback: Called with progress 0-99 during processing
        """
        # Get metadata via ffprobe (more reliable than cv2.CAP_PROP for VFR video)
        fps, width, height, total_frames = self._get_video_metadata(input_path)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        try:

            video_only_path = output_path.parent / "video_only.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(video_only_path), fourcc, fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("Cannot create video writer")

            stats = AnnotationStats(total_frames=total_frames)
            trackers: list[tuple[cv2.Tracker, DetectionBox]] = []
            current_detections: list[DetectionBox] = []
            frame_num = 0
            start_time = time.perf_counter()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % params.detect_every == 0:
                    results = self.model.predict(
                        source=frame,
                        conf=params.conf,
                        imgsz=params.imgsz,
                        max_det=params.max_det,
                        verbose=False,
                    )
                    current_detections = self._extract_detections(
                        results, params.classes
                    )
                    trackers = self._init_trackers(frame, current_detections)
                    stats.detected_frames += 1
                    stats.total_detections += len(current_detections)
                else:
                    current_detections = self._update_trackers(
                        frame, trackers
                    )
                    stats.tracked_frames += 1
                    stats.total_detections += len(current_detections)

                self._draw_detections(frame, current_detections, params)
                writer.write(frame)
                frame_num += 1

                if progress_callback and frame_num % 10 == 0:
                    progress = int(
                        (frame_num / max(total_frames, 1)) * 100
                    )
                    progress_callback(min(progress, 99))

            writer.release()
            stats.total_frames = frame_num
            stats.processing_time_ms = int(
                (time.perf_counter() - start_time) * 1000
            )

            self._merge_audio(input_path, video_only_path, output_path)

            if video_only_path.exists():
                video_only_path.unlink()

            return stats

        finally:
            cap.release()

    @staticmethod
    def _get_video_metadata(
        video_path: Path,
    ) -> tuple[float, int, int, int]:
        """Get video metadata via ffprobe. Returns (fps, width, height, total_frames)."""
        import json as _json

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "v:0",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                data = _json.loads(result.stdout)
                stream = data["streams"][0]
                # Parse fps from r_frame_rate (e.g. "30/1")
                r_rate = stream.get("r_frame_rate", "30/1")
                num, den = map(int, r_rate.split("/"))
                fps = num / den if den else 30.0
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))
                total_frames = int(stream.get("nb_frames", 0))
                if total_frames == 0:
                    # Estimate from duration
                    duration = float(stream.get("duration", 0))
                    total_frames = int(duration * fps)
                return fps, width, height, total_frames
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"ffprobe failed: {e}, falling back to cv2")

        # Fallback to cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return fps, width, height, total_frames

    def _extract_detections(
        self, results: list, class_filter: list[str] | None
    ) -> list[DetectionBox]:
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                class_id = int(cls[i])
                class_name = self.class_names.get(
                    class_id, f"class_{class_id}"
                )
                if class_filter and class_name not in class_filter:
                    continue
                detections.append(
                    DetectionBox(
                        x1=int(xyxy[i][0]),
                        y1=int(xyxy[i][1]),
                        x2=int(xyxy[i][2]),
                        y2=int(xyxy[i][3]),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(conf[i]),
                    )
                )
        return detections

    def _init_trackers(
        self, frame: np.ndarray, detections: list[DetectionBox]
    ) -> list[tuple[cv2.Tracker, DetectionBox]]:
        trackers = []
        for det in detections:
            tracker = _create_csrt_tracker()
            w = det.x2 - det.x1
            h = det.y2 - det.y1
            if w <= 0 or h <= 0:
                continue
            tracker.init(frame, (det.x1, det.y1, w, h))
            trackers.append((tracker, det))
        return trackers

    def _update_trackers(
        self,
        frame: np.ndarray,
        trackers: list[tuple[cv2.Tracker, DetectionBox]],
    ) -> list[DetectionBox]:
        updated = []
        for tracker, orig_det in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                updated.append(
                    DetectionBox(
                        x1=x,
                        y1=y,
                        x2=x + w,
                        y2=y + h,
                        class_id=orig_det.class_id,
                        class_name=orig_det.class_name,
                        confidence=orig_det.confidence,
                    )
                )
        return updated

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[DetectionBox],
        params: AnnotationParams,
    ) -> None:
        for det in detections:
            self.visualizer.draw_detection(
                image=frame,
                det=det,
                line_width=params.line_width,
                show_labels=params.show_labels,
                show_conf=params.show_conf,
                font_scale=self.visualizer.calculate_adaptive_font_scale(
                    frame.shape[0]
                ),
                text_thickness=1,
            )

    def _merge_audio(
        self, original: Path, video_only: Path, output: Path
    ) -> None:
        """Merge annotated video with audio from original using FFmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_only),
            "-i", str(original),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            str(output),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                logger.warning(
                    f"FFmpeg audio merge failed (rc={result.returncode}): "
                    f"{result.stderr[:500]}"
                )
                # Fallback: use video without audio
                shutil.copy2(str(video_only), str(output))
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"FFmpeg audio merge failed ({type(e).__name__}), using video only")
            shutil.copy2(str(video_only), str(output))
```

**Step 2: Verify it imports correctly**

Run: `cd /opt/github/zinin/vision-api-server/app && python -c "from video_annotator import VideoAnnotator, AnnotationParams, AnnotationStats; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add app/video_annotator.py
git commit -m "feat: add VideoAnnotator with YOLO + CSRT tracker pipeline"
```

---

### Task 6: Add API Endpoints and Worker Loop

**Files:**
- Modify: `app/main.py` (imports at line 1-30, lifespan at 61-120, new endpoints after line 673)
- Modify: `app/dependencies.py` (add `get_job_manager`)

**Step 1: Add `get_job_manager` dependency**

In `app/dependencies.py`, add after line 43:
```python


async def get_job_manager(request: Request) -> "JobManager":
    """Get JobManager from app state."""
    manager = getattr(request.app.state, "job_manager", None)
    if manager is None:
        raise RuntimeError("Job manager not initialized")
    return manager
```

Also add import at top of `app/dependencies.py` inside `TYPE_CHECKING` block — add `from job_manager import JobManager` after line 9 (`from model_manager import ModelManager`):
```python
    from job_manager import JobManager
```

**Step 2: Add imports to main.py**

In `app/main.py`, add to the imports section.

After line 12 (`from contextlib import asynccontextmanager`), add:
```python
import asyncio
import uuid
from pathlib import Path
```

Add `FileResponse` to the FastAPI imports:
```python
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
```

After line 16 (`from dependencies import get_model_manager`), change to:
```python
from dependencies import get_model_manager, get_job_manager
```

After line 28 (`from visualization import encode_image_to_bytes`), add:
```python
from job_manager import JobManager, JobStatus
from video_annotator import VideoAnnotator, AnnotationParams, AnnotationStats
```

Add to the models import (line 20-27), add `JobCreatedResponse, JobStatusResponse, JobStats`:
```python
from models import (
    DetectionResponse,
    VideoDetectionResponse,
    FrameDetection,
    Detection,
    BoundingBox,
    FrameExtractionResponse,
    ExtractedFrameData,
    JobCreatedResponse,
    JobStatusResponse,
    JobStats,
)
```

**Step 3: Add JobManager init and worker to lifespan**

In the lifespan function (`app/main.py`), after line 92 (`app.state.model_manager = model_manager`), add:
```python

        # Initialize JobManager
        job_manager = JobManager(
            jobs_dir=settings.video_jobs_dir,
            ttl_seconds=settings.video_job_ttl,
            max_queued=settings.max_queued_jobs,
        )
        job_manager.startup_sweep()
        job_manager.start_cleanup_task(interval=60)
        app.state.job_manager = job_manager

        # Start annotation worker
        app.state.worker_task = asyncio.create_task(
            _annotation_worker(app, settings)
        )
```

In the cleanup section (after line 115 `await app.state.model_manager.shutdown()`), add:
```python

        if hasattr(app.state, "job_manager"):
            await app.state.job_manager.shutdown()
```

Also cancel the worker task. Change the yield section. After `yield` (line 107), before the cleanup `logger.info`, add:
```python

    # Cancel worker
    if hasattr(app.state, "worker_task"):
        app.state.worker_task.cancel()
        try:
            await app.state.worker_task
        except asyncio.CancelledError:
            pass
```

**Step 4: Add worker function**

Before the `app = FastAPI(...)` line (line 122), add the worker function:
```python


async def _annotation_worker(app: FastAPI, settings: Settings) -> None:
    """Background worker that processes video annotation jobs."""
    logger.info("Annotation worker started")
    job_manager: JobManager = app.state.job_manager
    model_manager: ModelManager = app.state.model_manager

    while True:
        try:
            job_id = await job_manager.get_next_job_id()
            job = job_manager.get_job(job_id)
            if job is None:
                continue

            job_manager.mark_processing(job_id)

            # Get model
            model_name = job.params.get("model")
            try:
                model_entry = await model_manager.get_model(model_name)
            except (RuntimeError, ValueError) as e:
                job_manager.mark_failed(job_id, f"Model error: {e}")
                continue

            annotator = VideoAnnotator(
                model=model_entry.model,
                visualizer=model_entry.visualizer,
                class_names=model_entry.model.names,
            )

            params = AnnotationParams(
                conf=job.params.get("conf", 0.5),
                imgsz=job.params.get("imgsz", 640),
                max_det=job.params.get("max_det", 100),
                detect_every=job.params.get(
                    "detect_every", settings.default_detect_every
                ),
                classes=job.params.get("classes"),
                line_width=job.params.get("line_width", 2),
                show_labels=job.params.get("show_labels", True),
                show_conf=job.params.get("show_conf", True),
            )

            output_path = job.input_path.parent / "output.mp4"

            def progress_cb(progress: int) -> None:
                job_manager.update_progress(job_id, progress)

            # Run in executor (blocking I/O + YOLO inference)
            loop = asyncio.get_running_loop()
            try:
                stats = await loop.run_in_executor(
                    None,
                    lambda: annotator.annotate(
                        input_path=job.input_path,
                        output_path=output_path,
                        params=params,
                        progress_callback=progress_cb,
                    ),
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

                # Clean up input file
                if job.input_path.exists():
                    job.input_path.unlink()

            except Exception as e:
                logger.error(
                    f"Annotation failed for job {job_id}: {e}",
                    exc_info=True,
                )
                job_manager.mark_failed(job_id, str(e))

        except asyncio.CancelledError:
            logger.info("Annotation worker cancelled")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            await asyncio.sleep(1)
```

**Step 5: Add the three endpoints**

Append before `@app.get("/models", ...)` (line 660), add:

```python

# --- Video Annotation Endpoints ---

DetectEveryQuery = Annotated[
    int | None,
    Query(ge=1, le=300, description="Run YOLO detection every N frames (default from config)"),
]
ClassesQuery = Annotated[
    str | None,
    Query(description="Comma-separated class filter (e.g. person,car)"),
]


@app.post(
    "/detect/video/visualize",
    response_model=JobCreatedResponse,
    status_code=202,
    tags=["Video Annotation"],
)
async def annotate_video(
    file: UploadFile = File(..., description="Video file for annotation"),
    conf: ConfidenceQuery = 0.5,
    imgsz: ImageSizeQuery = 640,
    max_det: MaxDetQuery = 100,
    detect_every: DetectEveryQuery = None,
    classes: ClassesQuery = None,
    line_width: Annotated[
        int, Query(ge=1, le=10, description="Bounding box line width")
    ] = 2,
    show_labels: Annotated[bool, Query(description="Show class labels")] = True,
    show_conf: Annotated[
        bool, Query(description="Show confidence scores")
    ] = True,
    model: ModelQuery = None,
    job_manager: JobManager = Depends(get_job_manager),
    settings: Settings = Depends(get_settings),
):
    """
    Submit video for annotation with bounding boxes.

    Returns a job ID. Poll GET /jobs/{job_id} for status.
    Download result via GET /jobs/{job_id}/download when completed.
    """
    # Validate extension
    if file.filename:
        ext = (
            "." + file.filename.rsplit(".", 1)[-1].lower()
            if "." in file.filename
            else ""
        )
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format. Allowed: "
                f"{', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
            )

    # Resolve detect_every default from settings
    if detect_every is None:
        detect_every = settings.default_detect_every

    # Parse classes filter
    classes_list = None
    if classes:
        classes_list = [c.strip() for c in classes.split(",") if c.strip()]

    # Stream upload to temp file, then validate size
    import tempfile
    import shutil as _shutil
    tmp_dir = Path(settings.video_jobs_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"upload_{uuid.uuid4().hex[:8]}.tmp"
    try:
        total_size = 0
        with open(tmp_file, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_VIDEO_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video too large. Maximum: "
                        f"{MAX_VIDEO_SIZE // (1024 * 1024)} MB",
                    )
                f.write(chunk)
    except HTTPException:
        tmp_file.unlink(missing_ok=True)
        raise

    # Create job only after successful upload
    try:
        job = job_manager.create_job(
            params={
                "conf": conf,
                "imgsz": imgsz,
                "max_det": max_det,
                "detect_every": detect_every,
                "classes": classes_list,
                "line_width": line_width,
                "show_labels": show_labels,
                "show_conf": show_conf,
                "model": model,
            }
        )
    except RuntimeError as e:
        tmp_file.unlink(missing_ok=True)
        raise HTTPException(status_code=429, detail=str(e))

    # Move uploaded file to job directory
    _shutil.move(str(tmp_file), str(job.input_path))

    logger.info(
        f"Video annotation job created: {job.job_id}, "
        f"size={total_size // 1024}KB, detect_every={detect_every}"
    )

    return JobCreatedResponse(
        job_id=job.job_id,
        status=job.status.value,
        message="Video annotation job created",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
):
    """Get status of a video annotation job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    stats = None
    if job.stats:
        stats = JobStats(**job.stats)

    download_url = None
    if job.status == JobStatus.COMPLETED:
        download_url = f"/jobs/{job_id}/download"

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


@app.get("/jobs/{job_id}/download", tags=["Jobs"])
async def download_job_result(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
):
    """Download annotated video result."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not ready. Status: {job.status.value}",
        )

    if job.output_path is None or not job.output_path.exists():
        raise HTTPException(
            status_code=404, detail="Output file not found"
        )

    return FileResponse(
        path=str(job.output_path),
        media_type="video/mp4",
        filename=f"annotated_{job_id}.mp4",
    )
```

**Step 6: Verify the app starts**

Run: `cd /opt/github/zinin/vision-api-server/app && python -c "from main import app; print('Endpoints:', [r.path for r in app.routes])"`

Expected: output should include `/detect/video/visualize`, `/jobs/{job_id}`, `/jobs/{job_id}/download`

**Step 7: Commit**

```bash
git add app/main.py app/dependencies.py
git commit -m "feat: add video annotation endpoints and background worker"
```

---

### Task 7: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update endpoints table and configuration**

In `CLAUDE.md`, add to the Endpoints table:
```
| `/detect/video/visualize` | POST | Submit video for annotation (async, returns job_id) |
| `/jobs/{job_id}` | GET | Job status and progress |
| `/jobs/{job_id}/download` | GET | Download annotated video |
```

Add to Configuration section:
```
VIDEO_JOB_TTL=3600              # Completed job TTL seconds
VIDEO_JOBS_DIR=/tmp/vision_jobs  # Job files directory
MAX_CONCURRENT_JOBS=1            # Parallel video processing
MAX_QUEUED_JOBS=10               # Queue limit
DEFAULT_DETECT_EVERY=5           # YOLO every N frames
```

Add to features list in root endpoint description: `"video_annotation"`.

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with video annotation endpoints"
```

---

### Task 8: Manual Integration Test

No automated test — requires running YOLO model and FFmpeg.

**Step 1: Start the server**

```bash
cd /opt/github/zinin/vision-api-server
YOLO_MODELS='{"yolo11n.pt":"cpu"}' uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Step 2: Submit a video**

```bash
curl -X POST "http://localhost:8000/detect/video/visualize?detect_every=10&conf=0.3" \
  -F "file=@test_video.mp4"
```

Expected: 202 response with `job_id`

**Step 3: Poll status**

```bash
curl http://localhost:8000/jobs/{job_id}
```

Expected: status transitions from `queued` → `processing` (with progress) → `completed`

**Step 4: Download result**

```bash
curl http://localhost:8000/jobs/{job_id}/download -o annotated.mp4
```

Expected: MP4 file with bounding boxes drawn on detected objects

**Step 5: Verify the output**

- Open `annotated.mp4` — should have bounding boxes
- Check audio is preserved
- Check video duration matches original
