---
paths: "app/**/*.py"
---

# API Reference

## Detection Endpoints

### POST /detect

Image object detection returning JSON.

**Parameters:**
| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `file` | file | required | — | Image file |
| `conf` | float | 0.5 | 0.0-1.0 | Confidence threshold |
| `imgsz` | int | 640 | 32-2016 | Inference image size |
| `max_det` | int | 100 | 1-1000 | Max detections |
| `model` | string | null | — | Model name (e.g. yolo26s.pt) |

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95,
      "bbox": {"x1": 100, "y1": 50, "x2": 300, "y2": 400}
    }
  ],
  "count": 1,
  "processing_time_ms": 45,
  "image_size": [1920, 1080],
  "model": "yolo26s.pt"
}
```

### POST /detect/video

Video analysis on motion-selected frames.

**Parameters:**
| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `file` | file | required | — | Video file |
| `conf` | float | 0.5 | 0.0-1.0 | Confidence threshold |
| `imgsz` | int | 640 | 32-2016 | Inference image size |
| `max_det` | int | 100 | 1-1000 | Max detections per frame |
| `max_gap` | float | 4.0 | 0.5-30.0 | Grid step, seconds |
| `motion_threshold` | float | 0.001 | 0.0001-0.1 | Motion threshold, fraction of the frame |
| `min_interval` | float | 1.0 | 0.1-30.0 | Min seconds between any two selected frames |
| `max_frames` | int | 6 | 1-200 | Cap on selected frames |
| `model` | string | null | — | Model name |

**Frame selection** (`app/frame_selection.py`):
1. Frame 0 is always taken (`reason: first`).
2. A grid frame every `max_gap` seconds, or `min_interval` if that is larger (`reason: grid`), counted from the previous grid frame — a motion peak never shifts the grid. The grid is thinned uniformly to `max_frames − 1` frames, keeping the first and, when at least two remain, the last, so one slot is left for step 3 (with `max_frames=2` the slot replaces the last grid frame; `max_frames=1` returns frame 0 only). The held-back frame returns to the grid when no peak can use it — nothing above `motion_threshold`, or every candidate closer than `min_interval` to a selected frame — and a storm segment keeps the full grid of `max_frames`.
3. Motion peaks fill the remaining budget: frames are ranked by `blob`, the area of the largest changed region between neighbouring frames (gray, 640 px wide) as a fraction of the frame, taken while above `motion_threshold` and at least `min_interval` from every selected frame (`reason: motion`). If the median `blob` over the segment exceeds 0.02 (rain, snow in IR), peaks are skipped and only the grid remains.

`frame_number` is the frame index in the source video (0-based), `timestamp` its presentation time, `video_duration` comes from ffprobe. Unknown query parameters (e.g. the removed `scene_threshold`) are ignored.

**Upgrading from 2.x.** Clients should send `max_frames=6` (frigate-analyzer: `DETECT_MAX_FRAMES=6`) and drop `scene_threshold`; with the old `max_frames=50` a 16-second segment with motion returns up to 16 frames where 2.x returned 2, roughly 8× the inferences on `/detect/video` until the client is updated. `/detect/video` now hands YOLO frames in BGR (2.x passed RGB by mistake), so detections on the same recordings differ from 2.x.

### POST /detect/visualize

Returns annotated image with bounding boxes.

**Additional Parameters:**
| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `line_width` | int | 2 | 1-10 | Bbox line width |
| `show_labels` | bool | true | — | Show class labels |
| `show_conf` | bool | true | — | Show confidence |
| `quality` | int | 90 | 1-100 | JPEG quality |

**Response:** JPEG image with headers:
- `X-Processing-Time-Ms`
- `X-Detections-Count`

### POST /extract/frames

Extract motion-selected key frames without detection.

**Parameters:** `file`, `max_gap`, `motion_threshold`, `min_interval`, `max_frames` as in `/detect/video`, plus `quality` (int, 85, 1-100, JPEG quality).

**Response:**
```json
{
  "success": true,
  "video_duration": 16.0,
  "video_resolution": [2880, 1620],
  "frames_extracted": 5,
  "frames": [
    {"frame_number": 0,   "timestamp": 0.0,  "reason": "first",  "image_base64": "...", "width": 2880, "height": 1620},
    {"frame_number": 31,  "timestamp": 2.48, "reason": "motion", "image_base64": "...", "width": 2880, "height": 1620},
    {"frame_number": 50,  "timestamp": 4.0,  "reason": "grid",   "image_base64": "...", "width": 2880, "height": 1620},
    {"frame_number": 100, "timestamp": 8.0,  "reason": "grid",   "image_base64": "...", "width": 2880, "height": 1620},
    {"frame_number": 150, "timestamp": 12.0, "reason": "grid",   "image_base64": "...", "width": 2880, "height": 1620}
  ],
  "processing_time_ms": 1700
}
```

The list is never empty on success: frame 0 is always included. Cost: about 1.7 s for a 16-second 2880×1620 segment (two ffmpeg decodes: a 640 px gray scan and a fetch of the selected frames).

## Job Endpoints

### POST /jobs/{job_id}/cancel

Cancel a queued or processing video annotation job.

Cooperative cancellation. The immediate response to `/cancel` on a PROCESSING job returns `status: "processing"`; the worker flips it to `cancelled` after observing the event. Poll `GET /jobs/{job_id}` to see the terminal status.

**Latency has two parts:**

- **Checkpoint latency** — time from `/cancel` to the worker raising `JobCancelledError` inside `annotate`. Bounded by one frame's work (one YOLO inference in pass 1, or one decode+encode frame in pass 2). Typically sub-second on GPU; a few seconds on CPU / large models.
- **Terminal-transition latency** — time from `JobCancelledError` until `status` flips to `cancelled`. The exception must propagate through the FFmpeg context managers, which wait for the subprocesses to exit (decoder up to ~10 s, encoder up to ~300 s — encoders normally need to flush buffers and rewrite the MP4 `moov` atom). GPU path is typically 1–2 s; CPU / 4K encode may take tens of seconds to a few minutes in the worst case. Hard-killing FFmpeg is a non-goal.

**Completion race.** If `/cancel` arrives while the annotator is finalising the last frame, the job may have already transitioned to `completed` by the time the event would have been observed. In that case the follow-up `GET /jobs/{job_id}` returns `completed`, not `cancelled`. Clients must treat `completed` after a `/cancel` call as "work finished before cancellation took effect" and use `/jobs/{job_id}/download` as normal.

**Response codes:**

| Case | Code |
|------|------|
| QUEUED or PROCESSING | 200 + `JobStatusResponse` |
| Already CANCELLED (idempotent) | 200 + `JobStatusResponse` |
| COMPLETED or FAILED | 409 Conflict |
| Unknown / TTL-expired | 404 Not Found |

**409 Conflict body:**

```json
{"detail": "Cannot cancel job in terminal status 'completed'"}
```

The `detail` string names the current terminal status (`completed` or `failed`).

**Example:**

```bash
curl -X POST http://localhost:3001/jobs/abc123def456/cancel
```

Response:
```json
{
  "job_id": "abc123def456",
  "status": "processing",
  "progress": 42,
  "created_at": "2026-04-17T12:00:00+00:00",
  "completed_at": null,
  "download_url": null,
  "error": null,
  "stats": null
}
```

Poll `GET /jobs/{job_id}` — after the worker observes the cancel, `status` becomes `"cancelled"` and `completed_at` is populated. For a job cancelled while still `QUEUED`, the immediate response already shows `status: "cancelled"`.

## Info Endpoints

### GET /models

List loaded models with status.

**Response:**
```json
{
  "preloaded": [{"name": "yolo26s.pt", "device": "cuda:0"}],
  "cached": [{"name": "yolo26m.pt", "device": "cuda:0", "expires_in_seconds": 800}],
  "default_device": "cuda:0",
  "ttl_seconds": 900
}
```

### GET /health

Health check.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2,
  "preloaded_count": 1,
  "cached_count": 1,
  "default_device": "cuda:0",
  "video_processing": true,
  "open_fds": 123,
  "fd_deleted": 0,
  "fd_soft_limit": 65536
}
```

`open_fds` counts `/proc/self/fd` entries; `fd_deleted` counts those pointing at deleted files —
the exact signature of the ROCm/MIOpen leak (`fd_deleted` growing = compile-path leak;
`open_fds` growing while `fd_deleted` stays ≈ 0 = load-path leak). Both are `null` where `/proc`
is unavailable (e.g. non-Linux dev); `fd_soft_limit` is `0` where the `resource` module is absent.
Values are a point-in-time snapshot and reflect the process's `RLIMIT_NOFILE` (in Docker —
whatever `ulimits` grants; on a bare host — the shell default). A WARNING is logged (at most once
per hour) when `open_fds` reaches 80% of `fd_soft_limit` — early signal of an FD leak.

### GET /

Service info.

## Testing Examples

```bash
# Health check
curl http://localhost:3001/health

# Image detection
curl -X POST "http://localhost:3001/detect?conf=0.6" \
  -F "file=@image.jpg"

# With specific model
curl -X POST "http://localhost:3001/detect?model=yolo26m.pt" \
  -F "file=@image.jpg"

# Video detection
curl -X POST "http://localhost:3001/detect/video?max_frames=20" \
  -F "file=@video.mp4"

# Visualize
curl -X POST "http://localhost:3001/detect/visualize" \
  -F "file=@image.jpg" -o annotated.jpg

# List models
curl http://localhost:3001/models
```

## Error Responses

All errors return JSON:
```json
{
  "detail": "Error message",
  "type": "ExceptionType"
}
```

**Status Codes:**
- `400` — Invalid input (format, size, missing model)
- `413` — File too large
- `422` — Unreadable video, no video stream, or query parameter out of range
- `500` — Internal error (model load, inference failure)

## Models

Available YOLO26 models (ordered by speed/accuracy):
- `yolo26n.pt` — Nano, fastest
- `yolo26s.pt` — Small, good balance
- `yolo26m.pt` — Medium
- `yolo26l.pt` — Large
- `yolo26x.pt` — Extra large, most accurate
