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
FFmpeg decode      →   DetectionStabilizer          FFmpeg decode (2nd)
  ↓                      - build tracks (IoU)         ↓
YOLO predict             - vote class (weighted)     Draw stabilized boxes
  (lowered conf)         - fill gaps (bidir)           ↓
  ↓                      - apply grace period        FFmpeg encode
Store raw detections       (position-aware)
  (metadata only)
```

### Two-Pass Decode (No Frame Cache)

Instead of caching raw BGR24 frames to disk (which would require ~10 GB/minute for 1080p@30fps), the pipeline decodes the source video twice via FFmpeg:

- **Pass 1**: `FFmpegDecoder` decodes frames → YOLO inference → collect `RawDetection` metadata only (no frames saved to disk).
- **Stabilization**: Runs on metadata only (lightweight).
- **Pass 2**: A second `FFmpegDecoder` instance decodes the same source video from the beginning → draw stabilized boxes → `FFmpegEncoder`.

FFmpeg decode is fast relative to YOLO inference, so the double-decode overhead is negligible. This eliminates all disk space concerns, `STABILIZER_MAX_CACHE_GB`, and cache cleanup logic.

**Note**: This requires `workers=1` (already enforced by the existing architecture). Each job directory is isolated, so no path conflicts.

### Confidence Threshold Strategy

YOLO is called with `conf * STABILIZER_CONF_FACTOR` (default factor: 0.4, so user conf=0.5 becomes 0.2 for YOLO). This captures weak detections that would otherwise be lost. The original user `conf` is used as the threshold for creating *new* tracks — weak detections can only extend existing tracks, not start new ones.

### Class Filtering

If the user specifies `classes` (class name filter), filtering is applied *after* stabilization, not before. All classes are passed to the stabilizer so that the weighted voting mechanism can work with full data (e.g., an object classified as "truck" on one frame and "car" on another — both contribute to voting). After `stabilize()` returns, tracks whose `stable_class_name` is not in the user's `classes` list are removed before rendering.

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
    max_staleness_sec: float    # STABILIZER_MAX_STALENESS (default 5.0)

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

Process detection frames in order. For each detection on frame N, find the best-matching active track by IoU:

- A track is **active** if its latest detection is within `max_staleness_sec * fps` frames of frame N. Stale tracks are excluded from matching to prevent "ghost tracks" spanning minutes of video.
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

**Backward extension**: From the first detection in the track, step backward strictly by `detect_every` frames: `first_det_frame - detect_every`, `first_det_frame - 2*detect_every`, etc. On each step, check if `unmatched_weak` contains a detection on that frame with IoU >= `iou_threshold` against the track's earliest known bbox. If found, adopt it and continue stepping back. If the target frame is missing from `unmatched_weak` or no detection has sufficient IoU, **stop immediately** (do not skip gaps). A deep copy of `unmatched_weak` is used to prevent nondeterministic behavior when multiple tracks compete for the same weak detections.

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
- For each track: produce a `DetectionBox` with `class_id=track.stable_class_id`, `class_name=track.stable_class_name` (from voting), interpolated coordinates (rounded to `int`), and `confidence` = maximum confidence among detections of the winning class in this track (consistent with the displayed class, not borrowed from a different class)
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
stabilizer_max_staleness: float = 5.0     # > 0, seconds
```

Env variables: `STABILIZER_CONF_FACTOR`, `STABILIZER_IOU_THRESHOLD`, `STABILIZER_MIN_VOTE_CONF`, `STABILIZER_GRACE_CENTER`, `STABILIZER_GRACE_EDGE`, `STABILIZER_CENTER_ZONE`, `STABILIZER_MAX_STALENESS`. All fields should have Pydantic validators enforcing the bounds listed above.

### `app/video_annotator.py`

`VideoAnnotator.__init__()`: add `stabilizer_config: StabilizerConfig` as an additional parameter alongside existing `model`, `visualizer`, `class_names`, `hw_config`, `codec`, `crf`.

`VideoAnnotator.annotate()`: replace single loop with:

1. **Pass 1**: `FFmpegDecoder` decodes frames → run YOLO with `conf * config.conf_factor` → collect `dict[int, list[RawDetection]]` (all classes, no class filter). Record actual frame count. No frames are saved to disk.
2. **Stabilize**: call `DetectionStabilizer.stabilize()` with actual frame count from pass 1 (not ffprobe estimate). Then filter tracks by `classes` if specified.
3. **Pass 2**: a second `FFmpegDecoder` decodes the same source video → draw stabilized boxes from `StabilizedFrame` → `FFmpegEncoder`.

**Progress callback**: Pass 1 (YOLO inference, the slow part) reports 0-80%. Stabilization is near-instant, no progress update. Pass 2 (render) reports 80-99%.

`AnnotationStats`:
- `tracked_frames` = frames where at least one stabilized track is active via interpolation or grace period (not a direct YOLO detection frame). Counted as: `frame_num in stabilized and frame_num % detect_every != 0`.
- `total_detections` = sum of stabilized `DetectionBox` across all rendered frames (reflects what the user actually sees, not raw YOLO output).

Update the description in `models.py` accordingly.

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

- **Two-pass decode over frame caching**: Decoding the source video twice via FFmpeg is fast (relative to YOLO) and eliminates all disk space concerns. Raw BGR24 frames at 1080p would require ~10 GB/minute on disk — infeasible.
- **Greedy IoU over Hungarian**: Simpler, fast enough for typical detection density (≤10 objects per frame). IoU matching strategy is isolated and replaceable.
- **Always enabled**: The old behavior (raw YOLO output) is nearly unusable for video annotation. No reason to keep it as an option.
- **Stabilizer as separate module**: Clean separation of concerns. Testable without YOLO or FFmpeg. Can be improved independently.
- **Class filtering after stabilization**: Allows the voting mechanism to see all YOLO classes, improving class stability. Filtering by `stable_class_name` after voting is more effective than filtering raw detections before.
- **Max staleness for tracks**: Prevents "ghost tracks" where objects in the same location minutes apart get merged into one track with interpolation spanning the entire gap.
- **workers=1 requirement**: The annotation pipeline requires single-worker mode (already enforced). Frame decode state and job directories are not isolated for concurrent processing.

## Out of Scope (v1)

- **Track merging**: Two separate tracks that are actually the same object (e.g., after occlusion) are not merged. They remain as independent tracks. This may cause visual "rebirth" artifacts for long occlusions (> grace period).
- **Non-linear interpolation**: Bbox interpolation is linear. Spline or EMA smoothing may be added later if linear interpolation produces jarring jumps.
- **Ultralytics built-in tracker**: `model.track()` (BoT-SORT/ByteTrack) is not used in v1 but could replace IoU matching in a future version.
- **Batch YOLO inference**: Frames are processed one at a time. Batch inference could provide 1.5-2x speedup but requires frame buffering.
