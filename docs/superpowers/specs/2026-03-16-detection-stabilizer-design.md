# Detection Stabilizer: Video Annotation Post-Processing

## Problem

The `/detect/video/visualize` endpoint produces flickering, unreliable bounding boxes:

1. **Detection gaps**: YOLO fails to detect an object on most frames even when it is clearly visible (e.g., a moose in the center of the frame). Bounding boxes appear and disappear erratically.
2. **Class instability**: The same object is classified differently across frames (moose → dog → cow → horse → bear). Even large, prominent animals get misclassified frame-to-frame.
3. **No persistence**: Current hold mode simply reuses the last YOLO result. When YOLO returns nothing, boxes vanish immediately.

These issues occur even with the largest model (yolo26x) at high resolution.

## Solution

A two-pass video annotation pipeline with a `DetectionStabilizer` post-processor that links detections into tracks, stabilizes class labels via weighted voting, and fills gaps bidirectionally.

## Architecture

### Two-Pass Pipeline

The current single-pass `VideoAnnotator.annotate()` is replaced by three phases:

```
Pass 1 (collection)         Stabilization                Pass 2 (render)
───────────────────    ─────────────────────────    ──────────────────────
FFmpeg decode      →   DetectionStabilizer          Read cached frames
  ↓                      - build tracks (IoU)         ↓
Cache frame to tmp       - vote class (weighted)     Draw stabilized boxes
  ↓                      - fill gaps (bidir)           ↓
YOLO predict             - apply grace period        FFmpeg encode
  (lowered conf)           (position-aware)
  ↓
Store raw detections
```

### Frame Cache

Raw BGR24 frames are written to a single binary file `{job_dir}/frames.raw` during pass 1. Frames have fixed size (`width * height * 3` bytes), so any frame can be read by offset: `frame_num * frame_size`. The cache file is deleted after pass 2 completes. Cleanup on failure is handled by the existing `JobManager` job directory cleanup.

### Confidence Threshold Strategy

YOLO is called with `conf * STABILIZER_CONF_FACTOR` (default factor: 0.4, so user conf=0.5 becomes 0.2 for YOLO). This captures weak detections that would otherwise be lost. The original user `conf` is used as the threshold for creating *new* tracks — weak detections can only extend existing tracks, not start new ones.

## DetectionStabilizer Module

New file: `app/detection_stabilizer.py`

### Data Structures

```python
@dataclass
class StabilizerConfig:
    """Configuration from environment variables."""
    conf_factor: float          # STABILIZER_CONF_FACTOR (default 0.4)
    iou_threshold: float        # STABILIZER_IOU_THRESHOLD (default 0.3)
    min_vote_conf: float        # STABILIZER_MIN_VOTE_CONF (default 0.3)
    grace_center_sec: float     # STABILIZER_GRACE_CENTER (default 2.0)
    grace_edge_sec: float       # STABILIZER_GRACE_EDGE (default 0.5)
    center_zone: float          # STABILIZER_CENTER_ZONE (default 0.6)

@dataclass
class RawDetection:
    """Single detection from one frame."""
    frame_num: int
    x1: int; y1: int; x2: int; y2: int
    class_id: int
    class_name: str
    confidence: float

@dataclass
class Track:
    """A tracked object across multiple frames."""
    track_id: int
    detections: dict[int, RawDetection]  # frame_num → detection
    stable_class_id: int                 # result of weighted voting
    stable_class_name: str
    first_frame: int                     # after bidirectional fill
    last_frame: int

@dataclass
class StabilizedFrame:
    """Ready-to-render detections for one frame."""
    detections: list[DetectionBox]
```

### Public Interface

```python
class DetectionStabilizer:
    def __init__(self, config: StabilizerConfig): ...

    def stabilize(
        self,
        raw_detections: dict[int, list[RawDetection]],
        frame_width: int,
        frame_height: int,
        fps: float,
        total_frames: int,
        detect_every: int,
        conf_threshold: float,   # user's original conf
    ) -> dict[int, StabilizedFrame]:
        """Pure function: raw detections in, stabilized frames out."""
```

### Algorithm

#### Step 1: Build Tracks (IoU Matching)

Process detection frames in order. For each detection on frame N, find the best-matching track from frame N-detect_every by IoU:

- IoU >= `iou_threshold` → add detection to existing track
- IoU < threshold for all tracks AND `confidence >= conf_threshold` → create new track
- IoU >= threshold AND `confidence < conf_threshold` → add to existing track (lowered bar for known objects)

Matching is greedy: sort all (track, detection) pairs by IoU descending, assign 1:1 (Hungarian algorithm is an option for later, but greedy is sufficient for typical sparse detections).

#### Step 2: Class Voting

For each track:
- Filter detections with `confidence >= min_vote_conf`
- Compute weighted score per class: `score[class_id] += confidence`
- Winner = class with highest total score → `stable_class_id`, `stable_class_name`

#### Step 3: Bidirectional Gap Filling

For each track, determine the active frame range:

**Backward extension**: From the first detection in the track, look backward through earlier detection frames for weak detections (below `conf_threshold` but above YOLO's lowered threshold) with IoU overlap. Extend `first_frame` to include these.

**Forward extension (grace period)**: From the last detection, extend `last_frame` by a grace period that depends on position:
- Compute the center of the last known bbox
- If center is within the central `center_zone` (default 60%) of the frame → grace = `grace_center_sec * fps` frames
- If center is outside this zone (near edge) → grace = `grace_edge_sec * fps` frames
- Cap `last_frame` at `total_frames - 1`

**Backward grace period**: Same logic applies to `first_frame` — extend backward from first detection using position-aware grace period, capped at frame 0.

**Bbox interpolation**: For frames between two real detections within a track, linearly interpolate bbox coordinates. For grace period frames (beyond the last/first real detection), hold the last/first known bbox.

#### Step 4: Generate StabilizedFrames

For each frame 0..total_frames-1:
- Collect all tracks active on this frame
- For each track: produce a `DetectionBox` with `stable_class_name`, interpolated coordinates, and the confidence from the nearest real detection
- Return as `StabilizedFrame`

## Changes to Existing Files

### `app/config.py`

Add to Pydantic Settings:

```python
stabilizer_conf_factor: float = 0.4
stabilizer_iou_threshold: float = 0.3
stabilizer_min_vote_conf: float = 0.3
stabilizer_grace_center: float = 2.0
stabilizer_grace_edge: float = 0.5
stabilizer_center_zone: float = 0.6
```

Env variables: `STABILIZER_CONF_FACTOR`, `STABILIZER_IOU_THRESHOLD`, `STABILIZER_MIN_VOTE_CONF`, `STABILIZER_GRACE_CENTER`, `STABILIZER_GRACE_EDGE`, `STABILIZER_CENTER_ZONE`.

### `app/video_annotator.py`

`VideoAnnotator.__init__()`: accept `StabilizerConfig`.

`VideoAnnotator.annotate()`: replace single loop with:

1. **Pass 1**: decode frames, write to `frames.raw`, run YOLO with `conf * config.conf_factor`, collect `dict[int, list[RawDetection]]`
2. **Stabilize**: call `DetectionStabilizer.stabilize()`
3. **Pass 2**: open `frames.raw` for reading, iterate frames, draw from `StabilizedFrame`, encode
4. **Cleanup**: delete `frames.raw`

`AnnotationStats`: update `tracked_frames` semantics — now means "frames with interpolated/held detections" rather than "hold mode frames".

### `app/main.py`

No API changes. `StabilizerConfig` is constructed from `settings` when creating `VideoAnnotator`. The stabilization is always enabled.

## New Files

| File | Purpose |
|------|---------|
| `app/detection_stabilizer.py` | `StabilizerConfig`, `RawDetection`, `Track`, `StabilizedFrame`, `DetectionStabilizer` |
| `tests/test_detection_stabilizer.py` | Unit tests for stabilizer (no YOLO/FFmpeg dependencies) |

## Test Plan

Unit tests for `DetectionStabilizer` with synthetic data:

1. **Two objects, one disappears for several frames** — verify grace period fills the gap, object reappears
2. **One object with varying classes** — verify weighted voting picks the correct stable class
3. **Object near frame edge disappears** — verify shorter grace period vs center
4. **Bidirectional fill** — object first detected at frame 50, but weak detections exist at frames 30-45 — verify backward extension
5. **Low-confidence detections don't create new tracks** — only extend existing ones
6. **Non-overlapping objects** — verify they get separate tracks
7. **IoU matching correctness** — known bbox pairs with expected IoU values
8. **Linear interpolation** — verify bbox coordinates between two real detections
9. **Grace period capping** — verify `last_frame` doesn't exceed `total_frames`

## Design Decisions

- **Two-pass over streaming**: Required for bidirectional stabilization. Frame decode cost is small relative to YOLO inference.
- **Raw file cache over in-memory**: Video frames are large (1920×1080×3 ≈ 6MB per frame). A 1-minute 30fps video = 1800 frames ≈ 10GB. Disk cache is the only viable option.
- **Greedy IoU over Hungarian**: Simpler, fast enough for typical detection density (1-10 objects per frame). IoU matching strategy is isolated and replaceable.
- **Always enabled**: The old behavior (raw YOLO output) is nearly unusable for video annotation. No reason to keep it as an option.
- **Stabilizer as separate module**: Clean separation of concerns. Testable without YOLO or FFmpeg. Can be improved independently.
