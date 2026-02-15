---
paths: "docker/**/*"
---

# Docker Deployment

## Directory Structure

```
docker/
├── .env                      # Shared environment variables
├── nvidia/Dockerfile         # NVIDIA CUDA image
├── amd/Dockerfile            # AMD ROCm image
├── cpu/Dockerfile            # CPU-only image
├── docker-compose-nvidia.yml
├── docker-compose-amd.yml
├── docker-compose-cpu.yml
├── docker-up-*.sh            # Start scripts
├── docker-up-detach-*.sh     # Start detached
└── docker-down-*.sh          # Stop scripts
```

## Quick Start

```bash
cd docker

# NVIDIA GPU
./docker-up-nvidia.sh         # Foreground
./docker-up-detach-nvidia.sh  # Detached

# AMD GPU
./docker-up-amd.sh
./docker-up-detach-amd.sh

# CPU only
./docker-up-cpu.sh
./docker-up-detach-cpu.sh

# Stop
./docker-down-nvidia.sh
./docker-down-amd.sh
./docker-down-cpu.sh
```

## Environment Configuration

Edit `docker/.env`:

```bash
COMPOSE_PROJECT_NAME=detect-server

# Models to preload at startup (JSON format)
YOLO_MODELS='{"yolo26s.pt":"cuda:0"}'

# Default device for on-demand loaded models
YOLO_DEVICE=cuda:0

# TTL for cached models (seconds)
YOLO_MODEL_TTL=900
```

**Device options:**
- `cpu` — CPU only
- `cuda` / `cuda:0` / `cuda:1` — NVIDIA GPU
- `mps` — Apple Silicon (not for Docker)

## Port Mapping

| Container | Host |
|-----------|------|
| 8000 | 3001 |

Access API at `http://localhost:3001`

## Base Images

| Variant | Base Image |
|---------|------------|
| NVIDIA | `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` |
| AMD | `rocm/pytorch:latest` |
| CPU | `python:3.12-slim` |

## Dockerfile Overview

All Dockerfiles follow the same pattern:

1. Install system deps (Python, FFmpeg, OpenCV libs)
2. Create venv at `/app/venv`
3. Install PyTorch with appropriate backend
4. Install `requirements.txt`
5. Copy `app/*.py` to `/app`
6. Expose port 8000
7. Run uvicorn

## Health Check

All compose files include:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## GPU Requirements

### NVIDIA

- NVIDIA Driver 530+
- Docker with `nvidia-container-toolkit`
- CUDA 13.0 compatible GPU

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          capabilities: [gpu]
```

### AMD

- ROCm 5.0+ installed
- AMD GPU with ROCm support (RX 6000+, MI100+)

```yaml
devices:
  - /dev/kfd
  - /dev/dri
group_add:
  - video
```

## Building Images

```bash
cd docker

# Build specific variant
docker compose -f docker-compose-nvidia.yml build
docker compose -f docker-compose-amd.yml build
docker compose -f docker-compose-cpu.yml build
```

## Logs

```bash
# Follow logs
docker compose -f docker-compose-nvidia.yml logs -f

# Last 100 lines
docker compose -f docker-compose-nvidia.yml logs --tail=100
```

## Model Persistence

Models are downloaded to container at runtime. For persistence, add volume:

```yaml
volumes:
  - ./models:/models
```

And set `YOLO_MODELS` to use `/models/` path prefix.

## Troubleshooting

**CUDA out of memory:**
- Use smaller model (yolo26n.pt, yolo26s.pt)
- Reduce `imgsz` parameter
- Check other GPU processes

**FFmpeg not found:**
- Ensure `ffmpeg` is in Dockerfile apt-get install

**Slow startup:**
- First run downloads models (~25MB for yolo26s.pt)
- Use volume mount for model persistence
