FROM python:3.14-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    ffmpeg gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p /models

EXPOSE 8000

# supervisor.py runs uvicorn as a child and restarts the container when /health hangs
# (see app/supervisor.py). WATCHDOG_ENABLED=false makes it exec uvicorn directly.
CMD ["python3", "supervisor.py", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
