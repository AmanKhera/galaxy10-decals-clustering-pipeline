# syntax=docker/dockerfile:1.6

########################
# Base (shared)
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
# Data-only target
########################
FROM base AS data

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy code needed for data jobs + shared pipeline code
COPY jobs/ /app/jobs/
COPY galaxy_pipeline/ /app/galaxy_pipeline/

# Default (compose will override)
CMD ["python", "jobs/download_cutouts.py"]


########################
# ML target (heavy)
########################
FROM base AS ml

# (Optional) build tools for compiled Python packages (safe to keep)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY requirements-ml.txt /app/requirements-ml.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt -r /app/requirements-ml.txt

# Copy code needed for inference
COPY jobs/ /app/jobs/
COPY galaxy_pipeline/ /app/galaxy_pipeline/

# Default (compose will override). Keep something that exists.
CMD ["python", "-c", "print('ML image ready')"]
