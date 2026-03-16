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

**Disk space limits**: A 1920x1080 video at 30fps produces ~6 MB/frame ≈ 10 GB/minute of raw cache. The existing 500 MB upload limit caps input video length in practice, but long low-bitrate videos can still produce large caches. Before starting pass 1, estimate the cache size as `total_frames * width * height * 3` and refuse the job if it exceeds `STABILIZER_MAX_CACHE_GB` (default 50 GB, env variable). During pass 1, if a disk write fails (ENOSPC), abort the job with a clear error. The `VIDEO_JOBS_DIR` should be on a volume with sufficient space.

### Confidence Threshold Strategy

YOLO is called with `conf * STABILIZER_CONF_FACTOR` (default factor: 0.4, so user conf=0.5 becomes 0.2 for YOLO). This captures weak detections that would otherwise be lost. The original user `conf` is used as the threshold for creating *new* tracks — weak detections can only extend existing tracks, not start new ones.

### Class Filtering

If the user specifies `classes` (class name filter), filtering is applied *after* YOLO inference but *before* detections are fed to the stabilizer. This ensures that lowered confidence does not introduce unwanted classes into the stabilization pipeline. The stabilizer only sees detections that pass the class filter.

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

Matching is greedy, per detection frame: sort all (track, detection) pairs by IoU descending, assign 1:1 — each detection goes to at most one track, each track gets at most one detection per frame. Unmatched detections with `confidence < conf_threshold` are stored in a separate `unmatched_weak` list for use in backward extension (Step 3). Hungarian algorithm is an option for later, but greedy is sufficient for typical sparse detections.

#### Step 2: Class Voting

For each track:
- Filter detections with `confidence >= min_vote_conf`
- Compute weighted score per class: `score[class_id] += confidence`
- Winner = class with highest total score → `stable_class_id`, `stable_class_name`

#### Step 3: Bidirectional Gap Filling

For each track, determine the active frame range:

**Backward extension**: From the first detection in the track, look backward through the `unmatched_weak` detections (collected in Step 1) on earlier detection frames. For each earlier frame, check if any unmatched weak detection has IoU >= `iou_threshold` with the track's earliest known bbox. If so, adopt it into the track and extend `first_frame`. Continue backward until no match is found.

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
- For each track: produce a `DetectionBox` with `stable_class_id`, `stable_class_name` (from voting), interpolated coordinates, and the confidence from the temporally nearest real detection (ensures correct color mapping in visualizer and meaningful confidence display)
- Return as `StabilizedFrame`

## Changes to Existing Files

### `app/config.py`

Add to Pydantic Settings:

```python
stabilizer_conf_factor: float = 0.4       # (0, 1]
stabilizer_iou_threshold: float = 0.3     # (0, 1]
stabilizer_min_vote_conf: float = 0.3     # [0, 1]
stabilizer_grace_center: float = 2.0      # >= 0, seconds
stabilizer_grace_edge: float = 0.5        # >= 0, seconds
stabilizer_center_zone: float = 0.6       # (0, 1]
stabilizer_max_cache_gb: float = 50.0     # > 0, max frame cache size in GB
```

Env variables: `STABILIZER_CONF_FACTOR`, `STABILIZER_IOU_THRESHOLD`, `STABILIZER_MIN_VOTE_CONF`, `STABILIZER_GRACE_CENTER`, `STABILIZER_GRACE_EDGE`, `STABILIZER_CENTER_ZONE`, `STABILIZER_MAX_CACHE_GB`. All fields should have Pydantic validators enforcing the bounds listed above.

### `app/video_annotator.py`

`VideoAnnotator.__init__()`: accept `StabilizerConfig`.

`VideoAnnotator.annotate()`: replace single loop with:

1. **Pass 1**: decode frames, write to `frames.raw`, run YOLO with `conf * config.conf_factor`, apply class filter, collect `dict[int, list[RawDetection]]`. Record actual frame count.
2. **Stabilize**: call `DetectionStabilizer.stabilize()` with actual frame count from pass 1 (not ffprobe estimate).
3. **Pass 2**: open `frames.raw` for reading, iterate frames, draw from `StabilizedFrame`, encode.
4. **Cleanup**: delete `frames.raw`.

**Progress callback**: Pass 1 (YOLO inference, the slow part) reports 0-80%. Stabilization is near-instant, no progress update. Pass 2 (render) reports 80-99%.

`AnnotationStats`: update `tracked_frames` to mean "frames where at least one track is active via interpolation or grace period (not a direct YOLO detection)". Update the description in `models.py` accordingly.

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

## Out of Scope (v1)

- **Track merging**: Two separate tracks that are actually the same object (e.g., after occlusion) are not merged. They remain as independent tracks.
- **Non-linear interpolation**: Bbox interpolation is linear. Spline or EMA smoothing may be added later if linear interpolation produces jarring jumps.
- **Ultralytics built-in tracker**: `model.track()` (BoT-SORT/ByteTrack) is not used in v1 but could replace IoU matching in a future version.
