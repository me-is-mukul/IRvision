# IRVision model server (FastAPI), CPU build.
#   docker build -t irvision-api .
#   docker run -p 7860:7860 irvision-api          -> http://localhost:7860/api/health
# Port 7860 is the Hugging Face Spaces default; set PORT to change it.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/app/.cache/huggingface \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

# OpenCV (pulled in by ultralytics) needs these system libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-api.txt

# code, configuration and demo images
COPY pyproject.toml ./
COPY irvision ./irvision
COPY config ./config
COPY demo ./demo
RUN pip install --no-deps -e .

# trained models, fitted baselines, evaluation results, pre-trained detector
COPY outputs/models/unet_l1_ssim_9city/best.pt outputs/models/unet_l1_ssim_9city/best.pt
COPY outputs/models/segmenter/best.pt outputs/models/segmenter/best.pt
COPY outputs/models/baseline_lut.npy outputs/models/baseline_mean_color.npy outputs/models/
COPY outputs/results/ outputs/results/
COPY outputs/pretrained/yolov8n-obb.pt outputs/pretrained/yolov8n-obb.pt

# bake the EDSR weights into the image so the server needs no downloads at runtime
RUN python -c "from irvision.models.super_resolution import SuperResolver; SuperResolver.load()" \
    && chmod -R a+rX /app

# runtime: use only the baked-in weights (no Hugging Face calls); writable settings dir for YOLO
ENV HF_HUB_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp

EXPOSE 7860
CMD ["sh", "-c", "uvicorn irvision.api.server:app --host 0.0.0.0 --port ${PORT}"]
