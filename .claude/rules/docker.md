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

# Process watchdog: false runs uvicorn without supervisor.py. The only WATCHDOG_* variable
# the dev compose files forward; the deploy files forward more (see deploy/.env.example)
WATCHDOG_ENABLED=true
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
| AMD | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` (pinned, see `docker/amd/Dockerfile`) |
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

The healthcheck only *reports* status; plain Docker never acts on `unhealthy`. Inside the container `supervisor.py` polls the same endpoint with the same thresholds and, after 3 consecutive failures (or no healthy answer within 600 s of start), SIGKILLs uvicorn's process group and exits with code 3, so `restart: unless-stopped` recreates the container. All compose files set `init: true` (tini as PID 1). `WATCHDOG_ENABLED=false` in `.env` disables the watchdog with every compose file; the other `WATCHDOG_*` knobs are forwarded by the deploy files only (see `deploy/.env.example`). On the host, `docker inspect -f '{{.RestartCount}}' <container>` counts watchdog restarts.

### Verifying the watchdog

`init: true` needs `docker-init` on the host. Check it **before** the first `--force-recreate`:
without it the old container is removed and the new one fails to start.

```bash
docker run --rm --init alpine true     # must exit 0
```

Fire drill on a running container — freeze uvicorn and watch the watchdog react. Run it only once
the container has been up for more than `WATCHDOG_MIN_UPTIME` (600 s): a hang detected earlier
counts as flapping and the restart waits `WATCHDOG_FLAP_COOLDOWN` (900 s) first, so the drill
would take 17 minutes instead of 2.

```bash
docker top <container> -o pid,ppid,cmd # three lines contain "uvicorn main:app": docker-init (PID 1,
                                       # whose command line also carries the whole child command),
                                       # the supervisor ("python3 supervisor.py uvicorn main:app ..."),
                                       # and the real uvicorn ("/opt/venv/bin/python3
                                       # /opt/venv/bin/uvicorn main:app ...", whose PPID is the
                                       # supervisor). STOP the real uvicorn.
sudo kill -STOP <pid of the real uvicorn>
# ~2 minutes later (3 probes x (30 s interval + 10 s timeout)):
docker logs --since 5m <container> 2>&1 | grep supervisor:
#   restarting the container: reason=health_failed ...
docker inspect -f '{{.RestartCount}} {{.State.Health.Status}}' <container>
#   1 healthy
```

`kill -CONT <pid>` undoes a mistaken SIGSTOP. Stopping docker-init or the supervisor instead of the
child does nothing visible: the probe keeps failing only if uvicorn itself is frozen.

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

**One CPU core pegged at 100% while the service is healthy:**
- Not a hang -- the container answers `/health` and serves requests normally. A native thread of the
  ROCm runtime is busy-spinning; `py-spy dump` will not show it, because it is not a Python thread
- Confirm with `perf record -F 199 -g -t <tid> -- sleep 5` -- `rocr::core::Runtime::AsyncEventsLoop`
  in `libhsa-runtime64.so` means the base image drifted off the pinned ROCm tag. Check with
  `python3 -c "import torch; torch.zeros(1).cuda(); import time; time.sleep(60)"` in the image and watch
  it with `docker stats --no-stream`: an idle process must stay near 0%. The sleep is required -- without
  it Python exits immediately and there is no idle process whose CPU usage can be measured
- `/proc/<tid>/syscall` reading `running` plus a frozen `voluntary_ctxt_switches` distinguishes a
  busy-spin from ordinary blocking work

**Container restarts every few minutes:**
- The watchdog is firing: `docker logs <container> 2>&1 | grep supervisor:` shows `restarting the container: reason=...`
- `reason=startup_timeout` — model preload took longer than `WATCHDOG_STARTUP_TIMEOUT` (600 s); raise it or warm the MIOpen cache volume
- `reason=health_failed` shortly after start — the GPU is probably hung: check `rocm-smi` / `nvidia-smi` and `dmesg`; reboot the host if the GPU never comes back
- `child ... is still alive ... after SIGKILL` and the container never comes back — the child is
  stuck in an uninterruptible kernel state (D state, almost always the GPU driver). SIGKILL cannot
  touch it and the exiting PID namespace waits for it, so the container stays in "exiting"; only a
  host reboot recovers it
- Emergency: `WATCHDOG_ENABLED=false` in `.env` and `docker compose up -d`
