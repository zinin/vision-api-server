# Vision API Server

YOLO-based object detection API with FastAPI. Supports NVIDIA CUDA, AMD ROCm, and CPU backends.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Docker:
```bash
cd docker && ./docker-up-nvidia.sh   # NVIDIA GPU
cd docker && ./docker-up-amd.sh      # AMD GPU
cd docker && ./docker-up-cpu.sh      # CPU only
```

## Architecture

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, endpoints, lifespan |
| `app/model_manager.py` | YOLO model lifecycle, two-tier caching |
| `app/video_utils.py` | FFmpeg frame extraction, scene detection |
| `app/inference_utils.py` | Async inference with ThreadPoolExecutor |
| `app/image_utils.py` | Image validation, decoding |
| `app/visualization.py` | DetectionVisualizer, bbox rendering |
| `app/config.py` | Pydantic settings, env vars |
| `app/models.py` | Request/response Pydantic models |
| `app/dependencies.py` | FastAPI dependency injection |
| `app/hw_accel.py` | Hardware acceleration detection (NVIDIA/AMD/CPU) |
| `app/ffmpeg_pipe.py` | FFmpeg pipe-based video decoder/encoder |
| `app/job_manager.py` | Video annotation job lifecycle, async queue, TTL cleanup |
| `app/video_annotator.py` | YOLO detection + hold mode video annotation pipeline |

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/detect` | POST | Image detection (JSON) |
| `/detect/video` | POST | Video detection with smart frames |
| `/detect/visualize` | POST | Image with drawn bboxes |
| `/extract/frames` | POST | Extract frames as base64 |
| `/detect/video/visualize` | POST | Submit video for annotation (async, returns job_id) |
| `/jobs/{job_id}` | GET | Job status and progress |
| `/jobs/{job_id}/download` | GET | Download annotated video |
| `/models` | GET | List loaded/cached models |
| `/health` | GET | Health check |

## Configuration

`.env` file or environment variables:
```
YOLO_MODELS='{"yolo26s.pt":"cuda:0"}'  # JSON: model->device
YOLO_DEVICE=cuda                        # Default device for dynamic loads
YOLO_MODEL_TTL=900                      # Cache TTL seconds (min 60)
MAX_FILE_SIZE=10485760                  # Max image size (default 10MB)
MAX_EXECUTOR_WORKERS=4                  # ThreadPool workers
INFERENCE_TIMEOUT=30.0                  # Timeout seconds
LOG_LEVEL=INFO                          # Logging level (DEBUG, INFO, WARNING, ERROR)
VIDEO_JOB_TTL=3600                      # Completed job TTL seconds
VIDEO_JOBS_DIR=/tmp/vision_jobs         # Job files directory
MAX_QUEUED_JOBS=10                      # Queue limit
DEFAULT_DETECT_EVERY=5                  # YOLO every N frames
VIDEO_CODEC=auto                        # auto (match source) | h264 | h265 | av1
                                        # To force previous behavior: VIDEO_CODEC=h264
VIDEO_CRF=18                            # Quality: 0=lossless, 18=near-lossless, 23=default
VIDEO_HW_ACCEL=auto                     # auto | nvidia | amd | cpu
VAAPI_DEVICE=/dev/dri/renderD128        # VAAPI render device path
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

Tests cover config, Pydantic models, JobManager, and VideoAnnotator (mocked YOLO/FFmpeg).

## Key Patterns

**Async Inference**: YOLO runs in ThreadPoolExecutor via `run_in_executor()`.

**Two-Tier Caching**: Preloaded (never evicted) + cached (TTL-based eviction).

**Smart Frames**: FFmpeg scene detection with fallback to interval-based extraction.

**Video Annotation**: Async job API — YOLO every Nth frame + hold mode (reuse detections) for intermediate frames. Single worker, in-memory job state (requires `workers=1`).

## Limits

- Images: 10 MB default, 100 MB max configurable
- Videos: 500 MB
- Formats: jpg, jpeg, png, webp, bmp | mp4, avi, mov, mkv, webm, wmv, flv

## Port Mapping

Container: 8000 → Host: 3001

## Modular Docs

See `.claude/rules/` for detailed documentation:
- `api.md` — Endpoints, parameters, examples
- `docker.md` — Docker deployment, scripts

## Jira

Project tracker: https://jira.zinin.ru/ (project key: **VAS**)
Available via MCP `mcp-atlassian` — use `jira_search`, `jira_get_issue`, `jira_create_issue` etc. with `project_key: "FV"`.
When creating issues, assign them to user **azinin**.
