FROM rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- Системные зависимости ---
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    python3-pip \
    ffmpeg gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# --- Обновление pip ---
RUN python3 -m pip install --upgrade pip

# --- Рабочая директория ---
WORKDIR /app

RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir uvicorn[standard]
RUN pip install --no-cache-dir python-multipart
RUN pip install --no-cache-dir pydantic
RUN pip install --no-cache-dir pydantic-settings
RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir pytest
RUN pip install --no-cache-dir pytest-asyncio

RUN pip install --no-cache-dir --no-deps ultralytics
RUN pip install --no-cache-dir --no-deps opencv-python-headless

# --- Копируем приложение ---
COPY app/*.py .

# --- Папка для моделей ---
RUN mkdir -p /models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
