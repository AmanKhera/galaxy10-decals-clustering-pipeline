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

COPY jobs/ /app/jobs/
COPY galaxy_pipeline/ /app/galaxy_pipeline/

CMD ["python", "jobs/download_cutouts.py"]


########################
# ML target (GPU via pip CUDA deps)
########################
FROM base AS ml

# Optional build tools (keep if any deps need compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY requirements-ml.txt /app/requirements-ml.txt

# Install your project deps, then install TF with CUDA runtime libs bundled
# IMPORTANT: requirements-ml.txt must NOT contain "tensorflow"
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt -r /app/requirements-ml.txt && \
    pip install "tensorflow[and-cuda]==2.20.0"

# Prevent TF from grabbing all VRAM up front
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

COPY jobs/ /app/jobs/
COPY galaxy_pipeline/ /app/galaxy_pipeline/

CMD ["python", "-c", "print('ML image ready')"]
