# Video Annotation Design (VAS-2)

Video endpoint that takes input video and returns the same video with detected objects highlighted with bounding boxes.

## Approach

YOLO detection every Nth frame + OpenCV CSRT tracker for intermediate frames. Async job API.

## API

### POST /detect/video/visualize

Creates annotation job. Returns 202 with `job_id`.

Parameters:
- `file` (file, required) — video file
- `conf` (float, 0.5) — confidence threshold, 0.0-1.0
- `imgsz` (int, 640) — YOLO inference size, 32-2016
- `max_det` (int, 100) — max detections per frame, 1-1000
- `model` (str, null) — YOLO model name
- `detect_every` (int, 5) — run YOLO every N frames, 1-300
- `classes` (str, null) — comma-separated class filter (e.g. `person,car`)
- `line_width` (int, 2) — bbox line width, 1-10
- `show_labels` (bool, true) — show class names
- `show_conf` (bool, true) — show confidence scores

Response (202):
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Video annotation job created"
}
```

### GET /jobs/{job_id}

Job status and result info.

Response (processing):
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": 45,
  "created_at": "2026-02-14T12:00:00"
}
```

Response (completed):
```json
{
  "job_id": "uuid",
  "status": "completed",
  "progress": 100,
  "download_url": "/jobs/{job_id}/download",
  "stats": {
    "total_frames": 900,
    "detected_frames": 180,
    "tracked_frames": 720,
    "total_detections": 1250,
    "processing_time_ms": 45000
  }
}
```

### GET /jobs/{job_id}/download

Returns MP4 file as FileResponse.

## Processing Pipeline

1. Early reject if queue is full (before expensive upload)
2. Stream uploaded video to `/tmp/vision_jobs/{job_id}/input.mp4` (async chunked write via aiofiles, not full read into RAM)
3. Validate file size after write, create job only after successful save
4. Get metadata (fps, resolution, duration) via ffprobe with validation (width>0, height>0, fps>0; cv2 fallback)
5. Open video with `cv2.VideoCapture` (streaming decoder, ~50 MB RAM)
6. Create `cv2.VideoWriter` for output video (writes to disk frame-by-frame, released in finally)
7. Frame loop:
   - If `frame_num % detect_every == 0`: run YOLO, reinitialize CSRT trackers
   - Else: update CSRT trackers to get box positions
   - Draw bboxes using `DetectionVisualizer.draw_detection()` (public method)
   - Write frame to VideoWriter
8. Close VideoWriter
9. FFmpeg: merge annotated video stream with audio from original (best effort — no audio is not an error)
10. Update job status to completed
11. Clean up intermediate files (input.mp4, video_only.mp4) — separate try/except to avoid overwriting completed status

Tracker: `cv2.TrackerCSRT` — best accuracy/speed balance for short tracking intervals.

Reinit strategy: on every YOLO frame, trackers are fully recreated from fresh detections. Prevents drift accumulation.

## Job Management

In-memory `JobManager` with:
- `dict[str, Job]` state storage (lost on restart, acceptable — by design)
- FIFO queue with `asyncio.Queue`
- Single worker (one video at a time, GPU bottleneck)
- Background cleanup task (every 60s, removes expired jobs)
- Startup sweep: delete all directories and orphan .tmp files in `VIDEO_JOBS_DIR` on startup (orphan cleanup after restart)
- Queue capacity pre-check (`check_queue_capacity()`) — early reject before upload

Job states: `queued` → `processing` → `completed` | `failed`

Limits:
- `MAX_QUEUED_JOBS=10` — 429 when exceeded
- `MAX_VIDEO_SIZE=500MB` — existing limit

**Deployment constraint:** Requires `workers=1` (single process). In-memory state is not shared across processes. Multi-worker deployment (gunicorn, multiple pods) is not supported.

**Audio:** Best effort. If source video has no audio or FFmpeg merge fails, output will be video-only without error.

## Storage

```
/tmp/vision_jobs/{job_id}/
├── input.mp4          # uploaded video (deleted after processing)
├── video_only.mp4     # VideoWriter output, no audio (deleted after merge)
└── output.mp4         # final result with audio (deleted by TTL)
```

Disk: ~2-3x input size temporarily, ~1x after cleanup.
RAM: ~50-100 MB per job regardless of video length.

## Configuration

New env variables in `config.py`:
- `VIDEO_JOB_TTL=3600` — completed job TTL (seconds)
- `VIDEO_JOBS_DIR=/tmp/vision_jobs` — job files directory
- `MAX_QUEUED_JOBS=10` — queue limit
- `DEFAULT_DETECT_EVERY=5` — default YOLO interval

## New Files

| File | Purpose |
|------|---------|
| `app/job_manager.py` | JobManager, Job dataclass, queue, TTL cleanup |
| `app/video_annotator.py` | Pipeline: decode → YOLO + CSRT tracker → encode → FFmpeg audio merge |

## Changes to Existing Files

| File | Changes |
|------|---------|
| `app/main.py` | +3 endpoints, JobManager init in lifespan |
| `app/config.py` | +4 env variables |
| `app/visualization.py` | Rename `_draw_detection` → `draw_detection`, `_calculate_adaptive_font_scale` → `calculate_adaptive_font_scale` (make public) |
| `app/models.py` | +3 Pydantic response models |

## Reused Without Changes

- `app/visualization.py` — `DetectionVisualizer` (methods renamed to public, API unchanged)
- `app/inference_utils.py` — `run_inference()` used as-is
- `app/video_utils.py` — not used (different pipeline: per-frame, not keyframe)
