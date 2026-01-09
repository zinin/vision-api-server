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
| `model` | string | null | — | Model name (e.g. yolo11s.pt) |

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
  "model": "yolo11s.pt"
}
```

### POST /detect/video

Video analysis with smart frame extraction.

**Parameters:**
| Name | Type | Default | Range | Description |
|------|------|---------|-------|-------------|
| `file` | file | required | — | Video file |
| `conf` | float | 0.5 | 0.0-1.0 | Confidence threshold |
| `imgsz` | int | 640 | 32-2016 | Inference image size |
| `max_det` | int | 100 | 1-1000 | Max detections per frame |
| `scene_threshold` | float | 0.05 | 0.01-0.5 | Scene change sensitivity |
| `min_interval` | float | 1.0 | 0.1-30.0 | Min seconds between frames |
| `max_frames` | int | 50 | 1-200 | Max frames to extract |
| `model` | string | null | — | Model name |

**Frame Extraction Algorithm:**
1. Always extracts first frame
2. Extracts on scene changes (respecting min_interval)
3. Extracts middle frame if only first was selected

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

Extract key frames without detection.

**Parameters:** Same as `/detect/video` except detection params.

**Response:**
```json
{
  "video_duration": 30.5,
  "video_resolution": [1920, 1080],
  "frames_extracted": 10,
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "image_base64": "...",
      "width": 1920,
      "height": 1080
    }
  ],
  "processing_time_ms": 500
}
```

## Info Endpoints

### GET /models

List loaded models with status.

**Response:**
```json
{
  "preloaded": [{"name": "yolo11s.pt", "device": "cuda:0"}],
  "cached": [{"name": "yolo11m.pt", "device": "cuda:0", "expires_in_seconds": 800}],
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
  "video_processing": true
}
```

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
curl -X POST "http://localhost:3001/detect?model=yolo11m.pt" \
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
- `500` — Internal error (model load, inference failure)

## Models

Available YOLO11 models (ordered by speed/accuracy):
- `yolo11n.pt` — Nano, fastest
- `yolo11s.pt` — Small, good balance
- `yolo11m.pt` — Medium
- `yolo11l.pt` — Large
- `yolo11x.pt` — Extra large, most accurate
