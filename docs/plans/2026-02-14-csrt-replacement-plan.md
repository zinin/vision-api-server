# Replace CSRT Tracker with YOLO + Hold Mode — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove CPU-bound CSRT tracker from VideoAnnotator, replace with GPU YOLO detection + hold mode for ~15x speedup.

**Architecture:** On detection frames (`frame_num % detect_every == 0`), run `model.predict()`. On intermediate frames, reuse last detections (hold). No tracker state, no CPU bottleneck.

**Tech Stack:** Python, OpenCV (VideoCapture/VideoWriter only), Ultralytics YOLO (`model.predict()`), pytest

**Design doc:** `docs/plans/2026-02-14-csrt-replacement-design.md`

---

### Task 1: Remove CSRT code from VideoAnnotator

**Files:**
- Modify: `app/video_annotator.py`

**Step 1: Remove `_create_csrt_tracker()` function**

Delete lines 24-32 (the entire function):
```python
# DELETE THIS:
def _create_csrt_tracker() -> cv2.Tracker:
    """Create CSRT tracker, handling different OpenCV versions."""
    if hasattr(cv2, "TrackerCSRT"):
        return cv2.TrackerCSRT.create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT"):
        return cv2.legacy.TrackerCSRT.create()
    raise RuntimeError(
        "CSRT tracker not available. Install opencv-contrib-python-headless."
    )
```

**Step 2: Remove `_init_trackers()` method**

Delete lines 290-302:
```python
# DELETE THIS:
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
```

**Step 3: Remove `_update_trackers()` method**

Delete lines 304-327:
```python
# DELETE THIS:
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
                if w <= 0 or h <= 0:
                    continue
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
```

**Step 4: Remove `numpy` import if unused**

Check: `np` is used in `_extract_detections` parameter type hint (`np.ndarray` in `_draw_detections` param) — **keep `import numpy as np`**.

**Step 5: Rewrite frame loop and remove `trackers` variable**

Replace lines 121-166 (from `stats = AnnotationStats...` through end of loop) with:

```python
            stats = AnnotationStats(total_frames=total_frames)
            current_detections: list[DetectionBox] = []
            font_scale = self.visualizer.calculate_adaptive_font_scale(height)
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
                    stats.detected_frames += 1
                    stats.total_detections += len(current_detections)
                    logger.debug(
                        f"Frame {frame_num}: YOLO detected "
                        f"{len(current_detections)} objects"
                    )
                else:
                    stats.tracked_frames += 1
                    stats.total_detections += len(current_detections)

                self._draw_detections(
                    frame, current_detections, params, font_scale
                )
                writer.write(frame)
                frame_num += 1

                if (
                    progress_callback
                    and total_frames > 0
                    and frame_num % 10 == 0
                ):
                    progress = int((frame_num / total_frames) * 100)
                    progress_callback(min(progress, 99))
```

Key differences from old code:
- Removed `trackers` variable (line 122)
- Removed `self._init_trackers(frame, current_detections)` call (line 144)
- Replaced `self._update_trackers(frame, trackers)` with simple hold (reuse `current_detections`)
- Removed "lost trackers" logging from else branch

**Step 6: Update class docstring**

```python
# OLD:
class VideoAnnotator:
    """Annotate video with YOLO detections and CSRT tracking."""

# NEW:
class VideoAnnotator:
    """Annotate video with YOLO detections and hold mode."""
```

**Step 7: Run tests**

Run: `python -m pytest tests/test_video_annotator.py -v`
Expected: Some tests FAIL (TestCreateCsrtTracker, TestInitTrackers, TestUpdateTrackers — they import/reference removed code). TestAnnotatePipeline.test_full_pipeline may also fail due to tracker mock.

**Step 8: Commit (WIP)**

```bash
git add app/video_annotator.py
git commit -m "refactor: remove CSRT tracker, implement hold mode (VAS-2)"
```

---

### Task 2: Remove CSRT tests

**Files:**
- Modify: `tests/test_video_annotator.py`

**Step 1: Remove `_create_csrt_tracker` from import**

```python
# OLD (line 10-14):
from video_annotator import (
    VideoAnnotator,
    AnnotationParams,
    _create_csrt_tracker,
)

# NEW:
from video_annotator import (
    VideoAnnotator,
    AnnotationParams,
)
```

**Step 2: Remove `TestCreateCsrtTracker` class**

Delete lines 76-99 (entire class with 3 tests: `test_modern_api`, `test_legacy_api`, `test_unavailable`).

**Step 3: Remove `TestInitTrackers` class**

Delete lines 235-251 (entire class with 2 tests: `test_success`, `test_skip_zero_size`).

**Step 4: Remove `TestUpdateTrackers` class**

Delete lines 256-281 (entire class with 3 tests: `test_success`, `test_lost_tracker`, `test_zero_size_dropped`).

**Step 5: Update `test_full_pipeline`**

Remove tracker mock and its patch:

```python
# OLD (lines 387-388):
        mock_tracker = MagicMock()
        mock_tracker.update.return_value = (True, (10, 20, 90, 180))

# DELETE these 2 lines entirely.

# OLD (lines 401-406):
        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.cv2.VideoWriter", return_value=mock_writer),
            patch("video_annotator.cv2.VideoWriter_fourcc", return_value=0),
            patch("video_annotator.subprocess.run", side_effect=subprocess_side_effect),
            patch("video_annotator._create_csrt_tracker", return_value=mock_tracker),
        ):

# NEW:
        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.cv2.VideoWriter", return_value=mock_writer),
            patch("video_annotator.cv2.VideoWriter_fourcc", return_value=0),
            patch("video_annotator.subprocess.run", side_effect=subprocess_side_effect),
        ):
```

Stats assertions stay the same (detected_frames=2, tracked_frames=4).

**Step 6: Run tests**

Run: `python -m pytest tests/test_video_annotator.py -v`
Expected: All remaining tests PASS. 8 tests removed (3 + 2 + 3), pipeline test updated.

**Step 7: Commit**

```bash
git add tests/test_video_annotator.py
git commit -m "test: remove CSRT tracker tests, update pipeline test (VAS-2)"
```

---

### Task 3: Add hold mode tests

**Files:**
- Modify: `tests/test_video_annotator.py`

**Step 1: Write test for detect_every=1 (all frames detected)**

Add to `TestAnnotatePipeline` class:

```python
    def test_detect_every_1(self, mock_model, mock_visualizer, tmp_path):
        """When detect_every=1, every frame gets YOLO detection, no hold frames."""
        num_frames = 4
        mock_cap, mock_writer = self._setup_pipeline_mocks(num_frames)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        ffprobe_stream = {
            "r_frame_rate": "30/1",
            "width": 640,
            "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        merge_result = MagicMock()
        merge_result.returncode = 0

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names)

        def subprocess_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            return merge_result

        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.cv2.VideoWriter", return_value=mock_writer),
            patch("video_annotator.cv2.VideoWriter_fourcc", return_value=0),
            patch("video_annotator.subprocess.run", side_effect=subprocess_side_effect),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=1)
            )

        assert stats.total_frames == num_frames
        assert stats.detected_frames == num_frames  # Every frame detected
        assert stats.tracked_frames == 0  # No hold frames
        assert mock_model.predict.call_count == num_frames
```

**Step 2: Write test for hold behavior (detections reused on intermediate frames)**

Add to `TestAnnotatePipeline` class:

```python
    def test_hold_reuses_detections(self, mock_model, mock_visualizer, tmp_path):
        """Intermediate frames reuse last YOLO detections (hold mode)."""
        num_frames = 3
        mock_cap, mock_writer = self._setup_pipeline_mocks(num_frames)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        ffprobe_stream = {
            "r_frame_rate": "30/1",
            "width": 640,
            "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        merge_result = MagicMock()
        merge_result.returncode = 0

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        annotator = VideoAnnotator(mock_model, mock_visualizer, mock_model.names)

        def subprocess_side_effect(cmd, **kwargs):
            if cmd[0] == "ffprobe":
                return ffprobe_result
            return merge_result

        with (
            patch("video_annotator.cv2.VideoCapture", return_value=mock_cap),
            patch("video_annotator.cv2.VideoWriter", return_value=mock_writer),
            patch("video_annotator.cv2.VideoWriter_fourcc", return_value=0),
            patch("video_annotator.subprocess.run", side_effect=subprocess_side_effect),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=3)
            )

        assert stats.total_frames == num_frames
        # Frame 0: YOLO detection
        assert stats.detected_frames == 1
        assert mock_model.predict.call_count == 1
        # Frames 1, 2: hold (reuse detections from frame 0)
        assert stats.tracked_frames == 2
        # All 3 frames have 1 detection each = 3 total
        assert stats.total_detections == 3
        # Visualizer called for all 3 frames (1 detection per frame)
        assert mock_visualizer.draw_detection.call_count == 3
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_video_annotator.py -v`
Expected: All tests PASS.

**Step 4: Commit**

```bash
git add tests/test_video_annotator.py
git commit -m "test: add hold mode tests for video annotation (VAS-2)"
```

---

### Task 4: Final verification

**Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

**Step 2: Verify no CSRT references remain**

Run: `grep -ri "csrt\|_init_trackers\|_update_trackers\|_create_csrt" app/`
Expected: No output (no matches).

**Step 3: Check test count**

Old: 77 tests. Removed 8 (3 + 2 + 3), added 2. Expected: **71 tests**.

**Step 4: Squash or keep commits as-is, push**

Commits from this plan:
1. `refactor: remove CSRT tracker, implement hold mode (VAS-2)`
2. `test: remove CSRT tracker tests, update pipeline test (VAS-2)`
3. `test: add hold mode tests for video annotation (VAS-2)`
