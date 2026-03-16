# Detection Stabilizer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-pass video annotation pipeline with detection stabilization to eliminate bbox flickering and class instability.

**Architecture:** New `DetectionStabilizer` module performs IoU-based track building, weighted class voting, and bidirectional gap filling. `VideoAnnotator` is refactored from a single-pass to a two-pass pipeline: pass 1 collects raw detections and caches frames to disk, pass 2 renders stabilized results.

**Tech Stack:** Python, numpy, dataclasses. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-16-detection-stabilizer-design.md`

---

## Chunk 1: DetectionStabilizer Core

### Task 1: IoU utility and data structures

**Files:**
- Create: `app/detection_stabilizer.py`
- Test: `tests/test_detection_stabilizer.py`

- [ ] **Step 1: Write IoU and data structure tests**

```python
# tests/test_detection_stabilizer.py
import pytest

from detection_stabilizer import (
    RawDetection,
    StabilizerConfig,
    compute_iou,
)


class TestComputeIoU:
    def test_identical_boxes(self):
        assert compute_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_partial_overlap(self):
        # 50x50 overlap, union = 100*100 + 100*100 - 50*50 = 17500
        iou = compute_iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert iou == pytest.approx(2500 / 17500, abs=0.001)

    def test_one_inside_other(self):
        # Inner 50x50 inside 100x100, union = 10000
        iou = compute_iou((0, 0, 100, 100), (25, 25, 75, 75))
        assert iou == pytest.approx(2500 / 10000, abs=0.001)

    def test_zero_area_box(self):
        assert compute_iou((50, 50, 50, 50), (0, 0, 100, 100)) == 0.0

    def test_touching_edges(self):
        assert compute_iou((0, 0, 50, 50), (50, 0, 100, 50)) == 0.0


class TestStabilizerConfig:
    def test_defaults(self):
        config = StabilizerConfig()
        assert config.conf_factor == 0.4
        assert config.iou_threshold == 0.3
        assert config.min_vote_conf == 0.3
        assert config.grace_center_sec == 2.0
        assert config.grace_edge_sec == 0.5
        assert config.center_zone == 0.6
        assert config.max_cache_gb == 50.0


class TestRawDetection:
    def test_bbox_tuple(self):
        det = RawDetection(
            frame_num=0, x1=10, y1=20, x2=110, y2=120,
            class_id=0, class_name="person", confidence=0.9,
        )
        assert det.bbox == (10, 20, 110, 120)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detection_stabilizer.py -v`
Expected: FAIL — `detection_stabilizer` module does not exist.

- [ ] **Step 3: Implement data structures and IoU**

```python
# app/detection_stabilizer.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from visualization import DetectionBox

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StabilizerConfig:
    """Stabilizer configuration (from env variables via Settings)."""
    conf_factor: float = 0.4
    iou_threshold: float = 0.3
    min_vote_conf: float = 0.3
    grace_center_sec: float = 2.0
    grace_edge_sec: float = 0.5
    center_zone: float = 0.6
    max_cache_gb: float = 50.0


@dataclass(slots=True)
class RawDetection:
    """Single detection from one frame."""
    frame_num: int
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class Track:
    """A tracked object across multiple frames."""
    track_id: int
    detections: dict[int, RawDetection] = field(default_factory=dict)
    stable_class_id: int = 0
    stable_class_name: str = ""
    first_frame: int = 0
    last_frame: int = 0


@dataclass(slots=True)
class StabilizedFrame:
    """Ready-to-render detections for one frame."""
    detections: list[DetectionBox]


def compute_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    """Compute Intersection over Union of two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    if union == 0:
        return 0.0

    return intersection / union
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_detection_stabilizer.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add app/detection_stabilizer.py tests/test_detection_stabilizer.py
git commit -m "feat: add detection stabilizer data structures and IoU utility"
```

---

### Task 2: Track building (greedy IoU matching)

**Files:**
- Modify: `app/detection_stabilizer.py`
- Modify: `tests/test_detection_stabilizer.py`

- [ ] **Step 1: Write track building tests**

Append to `tests/test_detection_stabilizer.py`:

```python
from detection_stabilizer import DetectionStabilizer


def _make_raw(frame_num, x1, y1, x2, y2, class_id=0, class_name="person", conf=0.9):
    return RawDetection(
        frame_num=frame_num, x1=x1, y1=y1, x2=x2, y2=y2,
        class_id=class_id, class_name=class_name, confidence=conf,
    )


class TestBuildTracks:
    def _config(self, **overrides):
        return StabilizerConfig(**overrides)

    def test_single_object_across_frames(self):
        """One object detected on frames 0, 5, 10 — should form one track."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [_make_raw(0, 100, 100, 200, 200)],
            5: [_make_raw(5, 105, 105, 205, 205)],
            10: [_make_raw(10, 110, 110, 210, 210)],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5)
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 3

    def test_two_separate_objects(self):
        """Two non-overlapping objects — should form two tracks."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [
                _make_raw(0, 0, 0, 50, 50),
                _make_raw(0, 500, 500, 600, 600),
            ],
            5: [
                _make_raw(5, 5, 5, 55, 55),
                _make_raw(5, 505, 505, 605, 605),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5)
        assert len(tracks) == 2

    def test_low_conf_does_not_create_track(self):
        """Detection below conf_threshold should not create a new track."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [_make_raw(0, 100, 100, 200, 200, conf=0.3)],
        }
        tracks, unmatched = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5)
        assert len(tracks) == 0
        assert len(unmatched[0]) == 1

    def test_low_conf_extends_existing_track(self):
        """Low-conf detection matching an existing track should extend it."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [_make_raw(0, 100, 100, 200, 200, conf=0.8)],
            5: [_make_raw(5, 105, 105, 205, 205, conf=0.3)],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5)
        assert len(tracks) == 1
        assert len(tracks[0].detections) == 2

    def test_no_match_creates_new_track(self):
        """When IoU is too low, a high-conf detection starts a new track."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [_make_raw(0, 0, 0, 50, 50, conf=0.9)],
            5: [_make_raw(5, 500, 500, 600, 600, conf=0.9)],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5)
        assert len(tracks) == 2

    def test_greedy_1to1_assignment(self):
        """Each detection matches at most one track, and each track at most one detection."""
        stabilizer = DetectionStabilizer(self._config())
        raw = {
            0: [
                _make_raw(0, 100, 100, 200, 200, conf=0.9),
                _make_raw(0, 300, 300, 400, 400, conf=0.9),
            ],
            5: [
                _make_raw(5, 105, 105, 205, 205, conf=0.9),
                _make_raw(5, 305, 305, 405, 405, conf=0.9),
            ],
        }
        tracks, _ = stabilizer._build_tracks(raw, conf_threshold=0.5, detect_every=5)
        assert len(tracks) == 2
        for t in tracks:
            assert len(t.detections) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detection_stabilizer.py::TestBuildTracks -v`
Expected: FAIL — `DetectionStabilizer` and `_build_tracks` not defined.

- [ ] **Step 3: Implement _build_tracks**

Add to `app/detection_stabilizer.py`:

```python
class DetectionStabilizer:
    """Post-processor that stabilizes YOLO detections across video frames."""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self._next_track_id = 0

    def _new_track_id(self) -> int:
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    def _build_tracks(
        self,
        raw_detections: dict[int, list[RawDetection]],
        conf_threshold: float,
        detect_every: int,
    ) -> tuple[list[Track], dict[int, list[RawDetection]]]:
        """Build tracks from raw detections using greedy IoU matching.

        Returns:
            (tracks, unmatched_weak): tracks list and dict of frame_num → unmatched
            weak detections for backward extension.
        """
        tracks: list[Track] = []
        unmatched_weak: dict[int, list[RawDetection]] = {}

        for frame_num in sorted(raw_detections.keys()):
            detections = raw_detections[frame_num]
            if not detections:
                continue

            # All existing tracks are candidates — the staleness window is intentionally
            # wide because YOLO may miss an object for many consecutive detection frames.
            # Gap filling (Step 3) handles the temporal boundaries; here we just match by IoU.
            active_tracks = tracks

            # Compute IoU for all (track, detection) pairs
            pairs: list[tuple[float, int, int]] = []  # (iou, track_idx, det_idx)
            for ti, track in enumerate(active_tracks):
                # Use the most recent detection bbox from this track
                latest_frame = max(t for t in track.detections.keys() if t < frame_num) \
                    if any(t < frame_num for t in track.detections.keys()) else None
                if latest_frame is None:
                    continue
                track_bbox = track.detections[latest_frame].bbox
                for di, det in enumerate(detections):
                    iou = compute_iou(track_bbox, det.bbox)
                    if iou >= self.config.iou_threshold:
                        pairs.append((iou, ti, di))

            # Greedy 1:1 assignment (highest IoU first)
            pairs.sort(key=lambda x: x[0], reverse=True)
            assigned_tracks: set[int] = set()
            assigned_dets: set[int] = set()

            for iou_val, ti, di in pairs:
                if ti in assigned_tracks or di in assigned_dets:
                    continue
                active_tracks[ti].detections[frame_num] = detections[di]
                assigned_tracks.add(ti)
                assigned_dets.add(di)

            # Handle unassigned detections
            for di, det in enumerate(detections):
                if di in assigned_dets:
                    continue
                if det.confidence >= conf_threshold:
                    # New track
                    track = Track(
                        track_id=self._new_track_id(),
                        detections={frame_num: det},
                    )
                    tracks.append(track)
                else:
                    # Weak unmatched — save for backward extension
                    unmatched_weak.setdefault(frame_num, []).append(det)

        return tracks, unmatched_weak
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_detection_stabilizer.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add app/detection_stabilizer.py tests/test_detection_stabilizer.py
git commit -m "feat: add track building with greedy IoU matching"
```

---

### Task 3: Class voting

**Files:**
- Modify: `app/detection_stabilizer.py`
- Modify: `tests/test_detection_stabilizer.py`

- [ ] **Step 1: Write class voting tests**

Append to `tests/test_detection_stabilizer.py`:

```python
class TestClassVoting:
    def _config(self, **overrides):
        return StabilizerConfig(**overrides)

    def test_single_class(self):
        """All detections same class — voting returns that class."""
        stabilizer = DetectionStabilizer(self._config())
        track = Track(track_id=0, detections={
            0: _make_raw(0, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.8),
            5: _make_raw(5, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.7),
        })
        stabilizer._vote_class(track)
        assert track.stable_class_id == 1
        assert track.stable_class_name == "car"

    def test_weighted_voting(self):
        """Class with higher total weighted score wins."""
        stabilizer = DetectionStabilizer(self._config(min_vote_conf=0.3))
        track = Track(track_id=0, detections={
            0: _make_raw(0, 100, 100, 200, 200, class_id=0, class_name="person", conf=0.4),
            5: _make_raw(5, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.8),
            10: _make_raw(10, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.7),
        })
        # person: 0.4, car: 0.8+0.7=1.5 → car wins
        stabilizer._vote_class(track)
        assert track.stable_class_id == 1
        assert track.stable_class_name == "car"

    def test_low_conf_excluded_from_voting(self):
        """Detections below min_vote_conf don't participate in voting."""
        stabilizer = DetectionStabilizer(self._config(min_vote_conf=0.5))
        track = Track(track_id=0, detections={
            0: _make_raw(0, 100, 100, 200, 200, class_id=0, class_name="person", conf=0.6),
            5: _make_raw(5, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.3),
            10: _make_raw(10, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.4),
        })
        # Only person (0.6) passes min_vote_conf=0.5
        stabilizer._vote_class(track)
        assert track.stable_class_id == 0
        assert track.stable_class_name == "person"

    def test_empty_track_after_filtering(self):
        """If all detections are below min_vote_conf, use the highest confidence one."""
        stabilizer = DetectionStabilizer(self._config(min_vote_conf=0.9))
        track = Track(track_id=0, detections={
            0: _make_raw(0, 100, 100, 200, 200, class_id=0, class_name="person", conf=0.3),
            5: _make_raw(5, 100, 100, 200, 200, class_id=1, class_name="car", conf=0.5),
        })
        # No detection passes 0.9 — fallback to highest conf
        stabilizer._vote_class(track)
        assert track.stable_class_id == 1
        assert track.stable_class_name == "car"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detection_stabilizer.py::TestClassVoting -v`
Expected: FAIL — `_vote_class` not defined.

- [ ] **Step 3: Implement _vote_class**

Add to `DetectionStabilizer` class in `app/detection_stabilizer.py`:

```python
    def _vote_class(self, track: Track) -> None:
        """Determine stable class for a track via weighted voting."""
        scores: dict[int, float] = {}
        names: dict[int, str] = {}

        for det in track.detections.values():
            if det.confidence >= self.config.min_vote_conf:
                scores[det.class_id] = scores.get(det.class_id, 0.0) + det.confidence
                names[det.class_id] = det.class_name

        if not scores:
            # Fallback: use highest-confidence detection
            best = max(track.detections.values(), key=lambda d: d.confidence)
            track.stable_class_id = best.class_id
            track.stable_class_name = best.class_name
            return

        winner_id = max(scores, key=lambda k: scores[k])
        track.stable_class_id = winner_id
        track.stable_class_name = names[winner_id]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_detection_stabilizer.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add app/detection_stabilizer.py tests/test_detection_stabilizer.py
git commit -m "feat: add weighted class voting for track stabilization"
```

---

### Task 4: Bidirectional gap filling and grace period

**Files:**
- Modify: `app/detection_stabilizer.py`
- Modify: `tests/test_detection_stabilizer.py`

- [ ] **Step 1: Write gap filling and grace period tests**

Append to `tests/test_detection_stabilizer.py`:

```python
class TestGapFilling:
    def _config(self, **overrides):
        defaults = dict(grace_center_sec=2.0, grace_edge_sec=0.5, center_zone=0.6)
        defaults.update(overrides)
        return StabilizerConfig(**defaults)

    def test_forward_grace_period_center(self):
        """Object in center gets longer grace period."""
        config = self._config(grace_center_sec=1.0, grace_edge_sec=0.5)
        stabilizer = DetectionStabilizer(config)
        # Object at center of 1000x1000 frame, fps=10
        track = Track(track_id=0, detections={
            50: _make_raw(50, 400, 400, 600, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=200, detect_every=5, unmatched_weak={})
        # grace = 1.0 * 10 = 10 frames
        assert track.last_frame == 60
        # backward grace = 10 frames
        assert track.first_frame == 40

    def test_forward_grace_period_edge(self):
        """Object near edge gets shorter grace period."""
        config = self._config(grace_center_sec=2.0, grace_edge_sec=0.5)
        stabilizer = DetectionStabilizer(config)
        # Object at right edge of 1000x1000 frame (center_zone=0.6 → edge starts at x=800)
        track = Track(track_id=0, detections={
            50: _make_raw(50, 900, 400, 1000, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=200, detect_every=5, unmatched_weak={})
        # grace = 0.5 * 10 = 5 frames
        assert track.last_frame == 55
        assert track.first_frame == 45

    def test_grace_period_capped_at_total_frames(self):
        """last_frame cannot exceed total_frames - 1."""
        config = self._config(grace_center_sec=10.0)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            95: _make_raw(95, 400, 400, 600, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=100, detect_every=5, unmatched_weak={})
        assert track.last_frame == 99

    def test_grace_period_capped_at_zero(self):
        """first_frame cannot go below 0."""
        config = self._config(grace_center_sec=10.0)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            3: _make_raw(3, 400, 400, 600, 600, conf=0.9),
        })
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=200, detect_every=5, unmatched_weak={})
        assert track.first_frame == 0

    def test_backward_extension_with_unmatched_weak(self):
        """Weak unmatched detections before first detection extend the track backward."""
        config = self._config(grace_center_sec=0.0, grace_edge_sec=0.0)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            15: _make_raw(15, 100, 100, 200, 200, conf=0.9),
        })
        unmatched = {
            5: [_make_raw(5, 95, 95, 195, 195, conf=0.3)],
            10: [_make_raw(10, 98, 98, 198, 198, conf=0.25)],
        }
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=100, detect_every=5, unmatched_weak=unmatched)
        # Should adopt both weak detections
        assert 5 in track.detections
        assert 10 in track.detections
        assert track.first_frame == 5
        assert track.last_frame == 15

    def test_backward_extension_stops_on_no_match(self):
        """Backward extension stops when IoU drops below threshold."""
        config = self._config(grace_center_sec=0.0, grace_edge_sec=0.0)
        stabilizer = DetectionStabilizer(config)
        track = Track(track_id=0, detections={
            15: _make_raw(15, 100, 100, 200, 200, conf=0.9),
        })
        unmatched = {
            5: [_make_raw(5, 800, 800, 900, 900, conf=0.3)],   # far away, no IoU
            10: [_make_raw(10, 98, 98, 198, 198, conf=0.25)],   # close, good IoU
        }
        stabilizer._vote_class(track)
        stabilizer._fill_gaps(track, frame_width=1000, frame_height=1000, fps=10.0,
                              total_frames=100, detect_every=5, unmatched_weak=unmatched)
        assert 10 in track.detections
        assert 5 not in track.detections

    def test_interpolation_between_detections(self):
        """Bbox coordinates are linearly interpolated between real detections."""
        config = self._config(grace_center_sec=0.0, grace_edge_sec=0.0)
        stabilizer = DetectionStabilizer(config)
        raw = {
            0: [_make_raw(0, 100, 100, 200, 200, conf=0.9)],
            10: [_make_raw(10, 200, 200, 300, 300, conf=0.9)],
        }
        result = stabilizer.stabilize(
            raw, frame_width=1000, frame_height=1000,
            fps=10.0, total_frames=11, detect_every=10, conf_threshold=0.5,
        )
        # Frame 5 should be midpoint
        mid = result[5].detections[0]
        assert mid.x1 == 150
        assert mid.y1 == 150
        assert mid.x2 == 250
        assert mid.y2 == 250
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detection_stabilizer.py::TestGapFilling -v`
Expected: FAIL — `_fill_gaps` and `stabilize` not defined.

- [ ] **Step 3: Implement _fill_gaps, _is_in_center, _interpolate_bbox, and stabilize**

Add to `DetectionStabilizer` class in `app/detection_stabilizer.py`:

```python
    def _is_in_center(self, det: RawDetection, frame_width: int, frame_height: int) -> bool:
        """Check if bbox center is within the central zone of the frame."""
        cx = (det.x1 + det.x2) / 2
        cy = (det.y1 + det.y2) / 2
        margin_x = frame_width * (1 - self.config.center_zone) / 2
        margin_y = frame_height * (1 - self.config.center_zone) / 2
        return (margin_x <= cx <= frame_width - margin_x and
                margin_y <= cy <= frame_height - margin_y)

    def _grace_frames(self, det: RawDetection, frame_width: int, frame_height: int,
                      fps: float) -> int:
        """Compute grace period in frames based on bbox position."""
        if self._is_in_center(det, frame_width, frame_height):
            return int(self.config.grace_center_sec * fps)
        return int(self.config.grace_edge_sec * fps)

    def _fill_gaps(
        self,
        track: Track,
        frame_width: int,
        frame_height: int,
        fps: float,
        total_frames: int,
        detect_every: int,
        unmatched_weak: dict[int, list[RawDetection]],
    ) -> None:
        """Bidirectional gap filling: backward extension + grace periods."""
        if not track.detections:
            return

        sorted_frames = sorted(track.detections.keys())
        first_det_frame = sorted_frames[0]
        last_det_frame = sorted_frames[-1]

        # Backward extension: adopt unmatched weak detections
        earliest_det = track.detections[first_det_frame]
        current_bbox = earliest_det.bbox
        check_frames = sorted(
            [f for f in unmatched_weak if f < first_det_frame],
            reverse=True,
        )
        for frame_num in check_frames:
            best_match = None
            best_iou = 0.0
            for det in unmatched_weak[frame_num]:
                iou = compute_iou(current_bbox, det.bbox)
                if iou >= self.config.iou_threshold and iou > best_iou:
                    best_match = det
                    best_iou = iou
            if best_match is None:
                break
            track.detections[frame_num] = best_match
            current_bbox = best_match.bbox

        # Recalculate after backward extension
        sorted_frames = sorted(track.detections.keys())
        first_det_frame = sorted_frames[0]
        last_det_frame = sorted_frames[-1]

        # Grace periods
        first_det = track.detections[first_det_frame]
        last_det = track.detections[last_det_frame]

        backward_grace = self._grace_frames(first_det, frame_width, frame_height, fps)
        forward_grace = self._grace_frames(last_det, frame_width, frame_height, fps)

        track.first_frame = max(0, first_det_frame - backward_grace)
        track.last_frame = min(total_frames - 1, last_det_frame + forward_grace)

    @staticmethod
    def _interpolate_bbox(
        det_a: RawDetection,
        det_b: RawDetection,
        frame_num: int,
    ) -> tuple[int, int, int, int]:
        """Linearly interpolate bbox between two detections."""
        if det_a.frame_num == det_b.frame_num:
            return det_a.bbox

        t = (frame_num - det_a.frame_num) / (det_b.frame_num - det_a.frame_num)
        x1 = round(det_a.x1 + t * (det_b.x1 - det_a.x1))
        y1 = round(det_a.y1 + t * (det_b.y1 - det_a.y1))
        x2 = round(det_a.x2 + t * (det_b.x2 - det_a.x2))
        y2 = round(det_a.y2 + t * (det_b.y2 - det_a.y2))
        return (x1, y1, x2, y2)

    @staticmethod
    def _get_bbox_and_conf_for_frame(
        track: Track, frame_num: int, sorted_frames: list[int]
    ) -> tuple[tuple[int, int, int, int], float]:
        """Get interpolated bbox and nearest confidence for a frame.

        Args:
            sorted_frames: pre-sorted list of detection frame numbers for this track.
        """
        # Exact match
        if frame_num in track.detections:
            det = track.detections[frame_num]
            return det.bbox, det.confidence

        # Before first detection — hold first bbox
        if frame_num < sorted_frames[0]:
            det = track.detections[sorted_frames[0]]
            return det.bbox, det.confidence

        # After last detection — hold last bbox
        if frame_num > sorted_frames[-1]:
            det = track.detections[sorted_frames[-1]]
            return det.bbox, det.confidence

        # Between two detections — interpolate (binary search for efficiency)
        import bisect
        idx = bisect.bisect_right(sorted_frames, frame_num)
        prev_frame = sorted_frames[idx - 1]
        next_frame = sorted_frames[idx]

        bbox = DetectionStabilizer._interpolate_bbox(
            track.detections[prev_frame],
            track.detections[next_frame],
            frame_num,
        )
        # Confidence from nearest detection
        if frame_num - prev_frame <= next_frame - frame_num:
            conf = track.detections[prev_frame].confidence
        else:
            conf = track.detections[next_frame].confidence

        return bbox, conf

    def stabilize(
        self,
        raw_detections: dict[int, list[RawDetection]],
        frame_width: int,
        frame_height: int,
        fps: float,
        total_frames: int,
        detect_every: int,
        conf_threshold: float,
    ) -> dict[int, StabilizedFrame]:
        """Main entry point: raw detections in, stabilized frames out.

        Returns a sparse dict: only frames with at least one active track are
        included. The caller should treat missing keys as "no detections on this frame".
        """
        tracks, unmatched_weak = self._build_tracks(
            raw_detections, conf_threshold, detect_every
        )

        for track in tracks:
            self._vote_class(track)
            self._fill_gaps(
                track, frame_width, frame_height, fps,
                total_frames, detect_every, unmatched_weak,
            )

        logger.info(
            f"Stabilization: {len(tracks)} tracks from "
            f"{sum(len(d) for d in raw_detections.values())} raw detections"
        )

        # Precompute sorted frame lists per track for efficient interpolation
        track_sorted_frames = {
            id(track): sorted(track.detections.keys()) for track in tracks
        }

        # Generate per-frame output (sparse — only frames with detections)
        result: dict[int, StabilizedFrame] = {}
        for frame_num in range(total_frames):
            boxes: list[DetectionBox] = []
            for track in tracks:
                if track.first_frame <= frame_num <= track.last_frame:
                    bbox, conf = self._get_bbox_and_conf_for_frame(
                        track, frame_num, track_sorted_frames[id(track)]
                    )
                    boxes.append(DetectionBox(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                        class_id=track.stable_class_id,
                        class_name=track.stable_class_name,
                        confidence=conf,
                    ))
            if boxes:
                result[frame_num] = StabilizedFrame(detections=boxes)

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_detection_stabilizer.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add app/detection_stabilizer.py tests/test_detection_stabilizer.py
git commit -m "feat: add bidirectional gap filling, interpolation, and stabilize() entry point"
```

---

## Chunk 2: Config, VideoAnnotator Refactor, Integration

### Task 5: Add stabilizer settings to config

**Files:**
- Modify: `app/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write config tests**

Append to `tests/test_config.py`:

```python
class TestStabilizerSettings:
    def test_defaults(self):
        s = Settings(yolo_models='{}')
        assert s.stabilizer_conf_factor == 0.4
        assert s.stabilizer_iou_threshold == 0.3
        assert s.stabilizer_min_vote_conf == 0.3
        assert s.stabilizer_grace_center == 2.0
        assert s.stabilizer_grace_edge == 0.5
        assert s.stabilizer_center_zone == 0.6
        assert s.stabilizer_max_cache_gb == 50.0

    def test_custom_values(self):
        s = Settings(
            yolo_models='{}',
            stabilizer_conf_factor=0.3,
            stabilizer_iou_threshold=0.5,
            stabilizer_grace_center=3.0,
        )
        assert s.stabilizer_conf_factor == 0.3
        assert s.stabilizer_iou_threshold == 0.5
        assert s.stabilizer_grace_center == 3.0

    def test_conf_factor_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_conf_factor=0.0)

    def test_conf_factor_must_be_lte_1(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_conf_factor=1.5)

    def test_iou_threshold_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_iou_threshold=0.0)

    def test_max_cache_gb_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_max_cache_gb=0.0)

    def test_grace_non_negative(self):
        # 0 is valid (disables grace)
        s = Settings(yolo_models='{}', stabilizer_grace_center=0.0, stabilizer_grace_edge=0.0)
        assert s.stabilizer_grace_center == 0.0

    def test_grace_negative_rejected(self):
        with pytest.raises(ValidationError):
            Settings(yolo_models='{}', stabilizer_grace_center=-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py::TestStabilizerSettings -v`
Expected: FAIL — fields not defined in `Settings`.

- [ ] **Step 3: Add stabilizer fields to Settings**

Add to `app/config.py` inside the `Settings` class, after the `vaapi_device` field:

```python
    # Detection stabilizer settings
    stabilizer_conf_factor: float = Field(default=0.4, gt=0, le=1)
    stabilizer_iou_threshold: float = Field(default=0.3, gt=0, le=1)
    stabilizer_min_vote_conf: float = Field(default=0.3, ge=0, le=1)
    stabilizer_grace_center: float = Field(default=2.0, ge=0)
    stabilizer_grace_edge: float = Field(default=0.5, ge=0)
    stabilizer_center_zone: float = Field(default=0.6, gt=0, le=1)
    stabilizer_max_cache_gb: float = Field(default=50.0, gt=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add stabilizer configuration fields to Settings"
```

---

### Task 6: Update JobStats description in models.py

**Files:**
- Modify: `app/models.py`

- [ ] **Step 1: Update tracked_frames description**

In `app/models.py`, change line 173:

```python
    # Old:
    tracked_frames: int = Field(description="Frames with held detections (reused from last YOLO run)")
    # New:
    tracked_frames: int = Field(description="Frames with stabilized detections (interpolated or grace-extended, not direct YOLO)")
```

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest tests/test_models.py -v`
Expected: All PASS (description change doesn't break anything).

- [ ] **Step 3: Commit**

```bash
git add app/models.py
git commit -m "fix: update tracked_frames description for stabilizer semantics"
```

---

### Task 7: Refactor VideoAnnotator to two-pass pipeline

**Files:**
- Modify: `app/video_annotator.py`
- Modify: `tests/test_video_annotator.py`

This is the largest task. The `annotate()` method is replaced with a two-pass pipeline.

- [ ] **Step 1: Update existing tests for new pipeline**

Update `tests/test_video_annotator.py`:

Add import at top:
```python
from detection_stabilizer import StabilizerConfig
```

Update the `annotator` fixture:
```python
@pytest.fixture
def annotator(mock_model, mock_visualizer, hw_config):
    return VideoAnnotator(
        mock_model, mock_visualizer, mock_model.names, hw_config,
        stabilizer_config=StabilizerConfig(),
    )
```

Update every `VideoAnnotator(...)` call in `TestAnnotatePipeline` and `TestAutoCodecResolve` to include `stabilizer_config=StabilizerConfig()`.

**Remove `TestExtractDetections` class entirely** — it tests `_extract_detections` which is replaced by `_extract_raw_detections`. Add new tests for `_extract_raw_detections`:

```python
class TestExtractRawDetections:
    def test_single_detection(self, annotator):
        result = _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        dets = annotator._extract_raw_detections([result], frame_num=5, class_filter=None)
        assert len(dets) == 1
        assert dets[0].frame_num == 5
        assert dets[0].class_name == "person"
        assert dets[0].confidence == pytest.approx(0.9)
        assert dets[0].bbox == (10, 20, 100, 200)

    def test_class_filter(self, annotator):
        result = _make_yolo_result([
            (10, 20, 100, 200, 0, 0.9),
            (50, 60, 150, 250, 1, 0.8),
        ])
        dets = annotator._extract_raw_detections([result], frame_num=0, class_filter=["person"])
        assert len(dets) == 1
        assert dets[0].class_name == "person"

    def test_empty_boxes(self, annotator):
        result = _make_yolo_result([])
        dets = annotator._extract_raw_detections([result], frame_num=0, class_filter=None)
        assert dets == []
```

**Update `TestAnnotatePipeline` tests** for the two-pass pipeline. The key change: the stabilizer will create tracks from detections and may extend them via grace periods, so `tracked_frames` and `total_detections` values change. Use `StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)` in tests to disable grace periods for predictable behavior.

Example updated `test_full_pipeline`:
```python
    def test_full_pipeline(self, mock_model, mock_visualizer, hw_config, tmp_path):
        num_frames = 6
        detect_every = 3
        frames = self._make_frames(num_frames)
        mock_decoder_cls, mock_encoder_cls, mock_encoder = self._setup_ffmpeg_mocks(frames)

        mock_model.predict.return_value = [
            _make_yolo_result([(10, 20, 100, 200, 0, 0.9)])
        ]

        ffprobe_stream = {
            "r_frame_rate": "30/1", "width": 640, "height": 480,
            "nb_frames": str(num_frames),
        }
        ffprobe_result = MagicMock()
        ffprobe_result.returncode = 0
        ffprobe_result.stdout = json.dumps({"streams": [ffprobe_stream]})

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        # Disable grace periods for predictable test behavior
        config = StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)
        annotator = VideoAnnotator(
            mock_model, mock_visualizer, mock_model.names, hw_config,
            stabilizer_config=config,
        )

        with (
            patch("video_annotator.FFmpegDecoder", mock_decoder_cls),
            patch("video_annotator.FFmpegEncoder", mock_encoder_cls),
            patch("video_annotator.subprocess.run", return_value=ffprobe_result),
        ):
            stats = annotator.annotate(
                input_path, output_path, AnnotationParams(detect_every=detect_every)
            )

        assert stats.total_frames == num_frames
        # Frames 0, 3 are detection frames
        assert stats.detected_frames == 2
        # YOLO returns same bbox on both frames → one track → stabilized on all 6 frames
        # Non-detection frames with stabilized output: 1, 2, 4, 5
        assert stats.tracked_frames == 4
        assert mock_encoder.write_frame.call_count == num_frames
```

Apply the same pattern (zero grace period, `StabilizerConfig`) to all tests in `TestAnnotatePipeline` and `TestAutoCodecResolve`.

For `test_hold_clears_on_empty_detection`: with the stabilizer, frame 0 detects an object and frame 3 detects nothing. The stabilizer creates a track starting at frame 0 with one detection. With zero grace, the track only covers frame 0. So detections appear on fewer frames than before.

- [ ] **Step 2: Run tests to verify they fail (constructor changed)**

Run: `python -m pytest tests/test_video_annotator.py -v`
Expected: FAIL — `stabilizer_config` not accepted yet, `_extract_detections` removed.

- [ ] **Step 3: Refactor VideoAnnotator**

Replace `app/video_annotator.py` content. Key changes:

1. Import `StabilizerConfig`, `RawDetection`, `DetectionStabilizer`, `StabilizedFrame` from `detection_stabilizer`.
2. Add `stabilizer_config` parameter to `__init__`.
3. Replace `annotate()` single loop with:
   - `_pass1_collect()` — decode, cache frames to `frames.raw`, run YOLO, return raw detections and actual frame count.
   - Call `DetectionStabilizer.stabilize()`.
   - `_pass2_render()` — read cached frames, draw stabilized detections, encode.
   - Delete `frames.raw`.
4. Cache size check before pass 1.
5. Progress: 0-80% in pass 1, 80-99% in pass 2.

The full implementation (replace the `annotate` method and add helper methods):

```python
    def __init__(
        self,
        model: Any,
        visualizer: DetectionVisualizer,
        class_names: dict[int, str],
        hw_config: HWAccelConfig,
        codec: str = "h264",
        crf: int = 18,
        stabilizer_config: StabilizerConfig | None = None,
    ):
        self.model = model
        self.visualizer = visualizer
        self.class_names = class_names
        self.hw_config = hw_config
        self.codec = codec
        self.crf = crf
        self.stabilizer_config = stabilizer_config or StabilizerConfig()

    def annotate(
        self,
        input_path: Path,
        output_path: Path,
        params: AnnotationParams,
        progress_callback: Callable[[int], None] | None = None,
    ) -> AnnotationStats:
        metadata = self._get_video_metadata(input_path)

        # Resolve codec (unchanged logic)
        effective_codec, effective_crf, effective_bitrate = self._resolve_codec(metadata)

        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "ckpt_path", "unknown")
        model_device = getattr(self.model, "device", "unknown")
        logger.info(
            f"Starting annotation: {input_path.name}, "
            f"{metadata.width}x{metadata.height} @ {metadata.fps:.1f}fps, ~{metadata.total_frames} frames, "
            f"model={model_name}, device={model_device}, "
            f"detect_every={params.detect_every}, conf={params.conf}"
        )

        # Cache size check
        cache_path = output_path.parent / "frames.raw"
        frame_size = metadata.width * metadata.height * 3
        if metadata.total_frames > 0:
            estimated_gb = (metadata.total_frames * frame_size) / (1024 ** 3)
            if estimated_gb > self.stabilizer_config.max_cache_gb:
                raise RuntimeError(
                    f"Estimated frame cache {estimated_gb:.1f} GB exceeds limit "
                    f"{self.stabilizer_config.max_cache_gb} GB"
                )

        stats = AnnotationStats(total_frames=metadata.total_frames)
        start_time = time.perf_counter()

        # Pass 1: collect detections + cache frames
        yolo_conf = params.conf * self.stabilizer_config.conf_factor
        raw_detections, actual_frames = self._pass1_collect(
            input_path, metadata, params, yolo_conf, cache_path, frame_size,
            progress_callback, stats,
        )
        stats.total_frames = actual_frames

        # Stabilize
        stabilizer = DetectionStabilizer(self.stabilizer_config)
        stabilized = stabilizer.stabilize(
            raw_detections,
            frame_width=metadata.width,
            frame_height=metadata.height,
            fps=metadata.fps,
            total_frames=actual_frames,
            detect_every=params.detect_every,
            conf_threshold=params.conf,
        )

        # Count tracked frames: frames with stabilized output that aren't detection frames
        for frame_num in range(actual_frames):
            is_detection_frame = (frame_num % params.detect_every == 0)
            if frame_num in stabilized and not is_detection_frame:
                stats.tracked_frames += 1

        # Pass 2: render
        font_scale = self.visualizer.calculate_adaptive_font_scale(metadata.height)
        try:
            self._pass2_render(
                input_path, output_path, metadata, params, stabilized,
                effective_codec, effective_crf, effective_bitrate,
                font_scale, actual_frames, progress_callback,
            )
        finally:
            if cache_path.exists():
                cache_path.unlink()

        stats.processing_time_ms = int((time.perf_counter() - start_time) * 1000)
        fps_actual = actual_frames / max(stats.processing_time_ms / 1000, 0.001)
        logger.info(
            f"Frame processing complete: {actual_frames} frames in {stats.processing_time_ms}ms "
            f"({fps_actual:.1f} fps), detected={stats.detected_frames}, "
            f"tracked={stats.tracked_frames}, total_detections={stats.total_detections}"
        )
        return stats

    def _resolve_codec(self, metadata: VideoMetadata):
        """Resolve effective codec, crf, and bitrate. Returns (codec, crf, bitrate)."""
        if self.codec == "auto":
            resolved_codec = _CODEC_NAME_MAP.get(metadata.codec_name, None)
            if resolved_codec is not None and metadata.bit_rate is not None:
                result = (resolved_codec, None, metadata.bit_rate)
            elif resolved_codec is not None:
                result = (resolved_codec, 18, None)
            else:
                result = ("h264", 18, None)
            logger.info(
                f"Auto codec: source={metadata.codec_name}, resolved={result[0]}, "
                f"bitrate={result[2]}, crf={result[1]}"
            )
            return result
        return self.codec, self.crf, None

    def _pass1_collect(
        self,
        input_path: Path,
        metadata: VideoMetadata,
        params: AnnotationParams,
        yolo_conf: float,
        cache_path: Path,
        frame_size: int,
        progress_callback: Callable[[int], None] | None,
        stats: AnnotationStats,
    ) -> tuple[dict[int, list[RawDetection]], int]:
        """Pass 1: decode frames, cache to disk, run YOLO, collect raw detections."""
        raw_detections: dict[int, list[RawDetection]] = {}
        frame_num = 0

        with FFmpegDecoder(input_path, metadata.width, metadata.height, self.hw_config) as decoder, \
             open(cache_path, "wb") as cache_file:

            while True:
                frame = decoder.read_frame()
                if frame is None:
                    break

                try:
                    cache_file.write(frame.tobytes())
                except OSError as e:
                    raise RuntimeError(f"Failed to write frame cache (disk full?): {e}") from e

                if frame_num % params.detect_every == 0:
                    results = self.model.predict(
                        source=frame,
                        conf=yolo_conf,
                        imgsz=params.imgsz,
                        max_det=params.max_det,
                        verbose=False,
                    )
                    dets = self._extract_raw_detections(results, frame_num, params.classes)
                    if dets:
                        raw_detections[frame_num] = dets
                    stats.detected_frames += 1
                    stats.total_detections += len(dets)

                frame_num += 1

                if progress_callback and metadata.total_frames > 0 and frame_num % 10 == 0:
                    progress = int((frame_num / metadata.total_frames) * 80)
                    progress_callback(min(progress, 80))

        return raw_detections, frame_num

    def _extract_raw_detections(
        self, results: list, frame_num: int, class_filter: list[str] | None
    ) -> list[RawDetection]:
        """Extract RawDetection list from YOLO results with optional class filter."""
        detections = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            conf = result.boxes.conf.cpu().numpy()
            for i in range(len(cls)):
                class_id = int(cls[i])
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                if class_filter and class_name not in class_filter:
                    continue
                detections.append(RawDetection(
                    frame_num=frame_num,
                    x1=int(xyxy[i][0]), y1=int(xyxy[i][1]),
                    x2=int(xyxy[i][2]), y2=int(xyxy[i][3]),
                    class_id=class_id, class_name=class_name,
                    confidence=float(conf[i]),
                ))
        return detections

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
    ) -> None:
        """Pass 2: read cached frames, draw stabilized detections, encode."""
        cache_path = output_path.parent / "frames.raw"
        frame_size = metadata.width * metadata.height * 3

        with open(cache_path, "rb") as cache_file, \
             FFmpegEncoder(input_path, output_path, metadata.width, metadata.height,
                           metadata.fps, self.hw_config, effective_codec,
                           crf=effective_crf, bitrate=effective_bitrate) as encoder:

            for frame_num in range(total_frames):
                raw_bytes = cache_file.read(frame_size)
                if len(raw_bytes) < frame_size:
                    break
                frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                    (metadata.height, metadata.width, 3)
                ).copy()

                if frame_num in stabilized:
                    self._draw_detections(frame, stabilized[frame_num].detections,
                                          params, font_scale)

                encoder.write_frame(frame)

                if progress_callback and total_frames > 0 and frame_num % 10 == 0:
                    progress = 80 + int((frame_num / total_frames) * 19)
                    progress_callback(min(progress, 99))
```

Remove old `_extract_detections` method (replaced by `_extract_raw_detections`). Keep `_draw_detections` and `_get_video_metadata` unchanged.

- [ ] **Step 4: Run all tests and fix remaining failures**

Run: `python -m pytest tests/test_video_annotator.py -v`

The two-pass pipeline uses real file I/O for frame caching (via `tmp_path`), so no mock needed for cache files. The `_setup_ffmpeg_mocks` only provides frames for pass 1 (the decoder). Pass 2 reads from the cache file on disk.

Key: the stabilizer changes stats semantics. With `StabilizerConfig(grace_center_sec=0.0, grace_edge_sec=0.0)`, tracks only span frames where YOLO actually found the object. Adjust assertions accordingly. If a test still fails, debug by checking what the stabilizer produces for the test's detection pattern.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add app/video_annotator.py tests/test_video_annotator.py
git commit -m "feat: refactor VideoAnnotator to two-pass pipeline with detection stabilizer"
```

---

### Task 8: Wire StabilizerConfig in main.py

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Add StabilizerConfig import and construction**

In `app/main.py`, add import:
```python
from detection_stabilizer import StabilizerConfig
```

In `_annotation_worker`, update `VideoAnnotator` construction (around line 204):
```python
                stabilizer_config = StabilizerConfig(
                    conf_factor=settings.stabilizer_conf_factor,
                    iou_threshold=settings.stabilizer_iou_threshold,
                    min_vote_conf=settings.stabilizer_min_vote_conf,
                    grace_center_sec=settings.stabilizer_grace_center,
                    grace_edge_sec=settings.stabilizer_grace_edge,
                    center_zone=settings.stabilizer_center_zone,
                    max_cache_gb=settings.stabilizer_max_cache_gb,
                )
                annotator = VideoAnnotator(
                    model=model_entry.model,
                    visualizer=model_entry.visualizer,
                    class_names=model_entry.model.names,
                    hw_config=app.state.hw_config,
                    codec=settings.video_codec,
                    crf=settings.video_crf,
                    stabilizer_config=stabilizer_config,
                )
```

- [ ] **Step 2: Run worker tests**

Run: `python -m pytest tests/test_worker.py -v`
Expected: PASS (or adjust mocks if needed).

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: wire StabilizerConfig from settings into VideoAnnotator"
```

---

### Task 9: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add stabilizer env vars to Configuration section**

In `CLAUDE.md`, add to the Configuration block after `VIDEO_HW_ACCEL`:

```
STABILIZER_CONF_FACTOR=0.4      # YOLO conf multiplier for stabilizer (0-1]
STABILIZER_IOU_THRESHOLD=0.3    # IoU threshold for track matching (0-1]
STABILIZER_MIN_VOTE_CONF=0.3    # Min conf for class voting [0-1]
STABILIZER_GRACE_CENTER=2.0     # Grace period seconds, object in center
STABILIZER_GRACE_EDGE=0.5       # Grace period seconds, object at edge
STABILIZER_CENTER_ZONE=0.6      # Frame fraction considered "center" (0-1]
STABILIZER_MAX_CACHE_GB=50      # Max frame cache disk usage in GB
```

Add `app/detection_stabilizer.py` to the Architecture table:
```
| `app/detection_stabilizer.py` | Detection track stabilizer, IoU matching, class voting |
```

Update the Key Patterns section — add:
```
**Detection Stabilizer**: Two-pass pipeline. Pass 1 caches frames + collects YOLO detections with lowered conf. DetectionStabilizer links detections into tracks via IoU, votes on stable class, fills gaps bidirectionally with position-aware grace periods. Pass 2 renders stabilized boxes.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add detection stabilizer to CLAUDE.md"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 2: Verify no import errors**

Run: `cd app && python -c "from detection_stabilizer import DetectionStabilizer, StabilizerConfig; print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Verify config loads**

Run: `cd app && python -c "from config import Settings; s = Settings(yolo_models='{}'); print(f'conf_factor={s.stabilizer_conf_factor}, grace={s.stabilizer_grace_center}s')"`
Expected: `conf_factor=0.4, grace=2.0s`
