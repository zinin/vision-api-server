# Pinned deliberately. rocm/pytorch:latest moved on to ROCm 7.14 / torch 2.13, whose runtime
# spawns an AsyncEventsLoop thread that busy-spins a whole CPU core forever once a GPU context
# exists -- an idle, perfectly healthy service burns 100% of a core. Reproduce without the app.
# The container has to run detached and keep sleeping: an attached or exited one leaves nothing
# for docker stats to measure.
#   docker run -d --rm --name rocm-spin-check --device /dev/kfd --device /dev/dri \
#     --group-add video <image> \
#     python3 -c "import torch; torch.zeros(1).cuda(); import time; time.sleep(120)"
#   sleep 45                                   # let torch and ROCm finish initialising
#   docker stats --no-stream rocm-spin-check   # ~0% on this tag, ~100% on ROCm 7.14
# No HSA_* knob switches it off (INTERRUPT, MWAITX, ASYNCEVENTS_THREAD_PRIORITY,
# GPU_MAX_HW_QUEUES, SDMA were all tested). Run that check before moving this tag.
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    python3-pip \
    ffmpeg gcc g++ \
    libva-dev mesa-va-drivers vainfo \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

# Install deps except ultralytics/opencv (they need --no-deps to preserve ROCm PyTorch)
RUN pip install --no-cache-dir $(grep -vE '^(ultralytics|opencv|#|$)' requirements.txt)
# Install with pinned versions from requirements.txt but without deps to preserve ROCm PyTorch
RUN pip install --no-cache-dir --no-deps $(grep '^ultralytics' requirements.txt) $(grep '^opencv' requirements.txt)

COPY app/ .

RUN mkdir -p /models

EXPOSE 8000

# supervisor.py runs uvicorn as a child and restarts the container when /health hangs
# (see app/supervisor.py). WATCHDOG_ENABLED=false makes it exec uvicorn directly.
CMD ["python3", "supervisor.py", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
