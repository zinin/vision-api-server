FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

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
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

# Install deps except ultralytics/opencv (they need --no-deps to preserve ROCm PyTorch)
RUN pip install --no-cache-dir $(grep -vE '^(ultralytics|opencv)' requirements.txt)
# Install with pinned versions from requirements.txt but without deps to preserve ROCm PyTorch
RUN pip install --no-cache-dir --no-deps $(grep '^ultralytics' requirements.txt) $(grep '^opencv' requirements.txt)

COPY app/ .

RUN mkdir -p /models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
