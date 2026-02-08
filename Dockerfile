# syntax=docker/dockerfile:1.6

########################
# Base (shared, CPU)
########################
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    ARTIFACTS_DIR=/app/artifacts \
    REPORTS_DIR=/app/reports

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


########################
# Data-only target (CPU)
########################
FROM base AS data

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy code needed for data jobs + shared pipeline code
COPY jobs/ /app/jobs/
COPY galaxy_pipeline/ /app/galaxy_pipeline/

CMD ["python", "jobs/download_cutouts.py"]


########################
# ML target (GPU-enabled)
########################
# Use TF GPU base so CUDA/cuDNN user-space libs are present.
# Pin to the TF version you're using (you have 2.20.0).
FROM tensorflow/tensorflow:2.20.0-gpu AS ml

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    ARTIFACTS_DIR=/app/artifacts \
    REPORTS_DIR=/app/reports \
    TF_FORCE_GPU_ALLOW_GROWTH=true

WORKDIR /app

# Minimal extras (often not needed; keep only if you have wheels that need compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY requirements-ml.txt /app/requirements-ml.txt

# IMPORTANT:
# - requirements-ml.txt should NOT include "tensorflow"
# - We install shared deps + ML deps on top of the TF GPU base
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt -r /app/requirements-ml.txt

# Copy code needed for inference
COPY jobs/ /app/jobs/
COPY galaxy_pipeline/ /app/galaxy_pipeline/

CMD ["python", "-c", "print('ML GPU image ready')"]
