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
7. Run `supervisor.py`, which runs uvicorn as a child and restarts the container when `/health` hangs (see Health Check)

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

The healthcheck only *reports* status; plain Docker never acts on `unhealthy`. Inside the container `supervisor.py` polls the same endpoint with the same thresholds and, after 3 consecutive failures (or no healthy answer within 600 s of start), SIGKILLs uvicorn's process group and exits with code 3, so `restart: unless-stopped` recreates the container. All compose files set `init: true` (tini as PID 1). Tune with `WATCHDOG_*` (see `deploy/.env.example`); `WATCHDOG_ENABLED=false` disables the watchdog. On the host, `docker inspect -f '{{.RestartCount}}' <container>` counts watchdog restarts.

### Verifying the watchdog

`init: true` needs `docker-init` on the host. Check it **before** the first `--force-recreate`:
without it the old container is removed and the new one fails to start.

```bash
docker run --rm --init alpine true     # must exit 0
```

Fire drill on a running container — freeze uvicorn and watch the watchdog react:

```bash
docker top <container> -o pid,cmd      # two lines contain "uvicorn main:app": the supervisor's
                                       # own command line and the real uvicorn (the child)
sudo kill -STOP <pid of the real uvicorn>
# ~2 minutes later (3 probes x (30 s interval + 10 s timeout)):
docker logs --since 5m <container> 2>&1 | grep supervisor:
#   restarting the container: reason=health_failed ...
docker inspect -f '{{.RestartCount}} {{.State.Health.Status}}' <container>
#   1 healthy
```

`kill -CONT <pid>` undoes a mistaken SIGSTOP. Stopping the supervisor process instead of the child
does nothing visible: the probe keeps failing only if uvicorn itself is frozen.

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

**Container restarts every few minutes:**
- The watchdog is firing: `docker logs <container> 2>&1 | grep supervisor:` shows `restarting the container: reason=...`
- `reason=startup_timeout` — model preload took longer than `WATCHDOG_STARTUP_TIMEOUT` (600 s); raise it or warm the MIOpen cache volume
- `reason=health_failed` shortly after start — the GPU is probably hung: check `rocm-smi` / `nvidia-smi` and `dmesg`; reboot the host if the GPU never comes back
- `child ... is still alive ... after SIGKILL` and the container never comes back — the child is
  stuck in an uninterruptible kernel state (D state, almost always the GPU driver). SIGKILL cannot
  touch it and the exiting PID namespace waits for it, so the container stays in "exiting"; only a
  host reboot recovers it
- Emergency: `WATCHDOG_ENABLED=false` in `.env` and `docker compose up -d`
