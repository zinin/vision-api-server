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

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/detect` | POST | Image detection (JSON) |
| `/detect/video` | POST | Video detection with smart frames |
| `/detect/visualize` | POST | Image with drawn bboxes |
| `/extract/frames` | POST | Extract frames as base64 |
| `/models` | GET | List loaded/cached models |
| `/health` | GET | Health check |

## Configuration

`.env` file or environment variables:
```
YOLO_MODELS='{"yolo11s.pt":"cuda:0"}'  # JSON: model->device
YOLO_DEVICE=cuda                        # Default device for dynamic loads
YOLO_MODEL_TTL=900                      # Cache TTL seconds (min 60)
MAX_FILE_SIZE=10485760                  # Max image size (default 10MB)
MAX_EXECUTOR_WORKERS=4                  # ThreadPool workers
INFERENCE_TIMEOUT=30.0                  # Timeout seconds
DEBUG=false                             # Detailed errors
```

## Key Patterns

**Async Inference**: YOLO runs in ThreadPoolExecutor via `run_in_executor()`.

**Two-Tier Caching**: Preloaded (never evicted) + cached (TTL-based eviction).

**Smart Frames**: FFmpeg scene detection with fallback to interval-based extraction.

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
