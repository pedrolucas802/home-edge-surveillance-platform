FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PREFER_BINARY=1 \
    FFMPEG_BIN=ffmpeg \
    DASHBOARD_DIR=/app/data/dashboard \
    TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.docker.txt ./
RUN pip install --upgrade pip \
    && pip install --index-url "${TORCH_CPU_INDEX_URL}" torch torchvision \
    && pip install -r requirements.docker.txt

COPY . .

RUN mkdir -p /app/data/dashboard /app/data/logs && chmod +x /app/scripts/run_dashboard_stack.sh

EXPOSE 8501

CMD ["bash", "scripts/run_dashboard_stack.sh"]
