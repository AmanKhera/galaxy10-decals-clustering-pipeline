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

# System deps (keep minimal). Add only what you need.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

########################
# Data-only target
########################
FROM base AS data

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy code
COPY jobs/ /app/jobs/
# If you have shared helpers:
# COPY src/ /app/src/

# Default command runs ingestion stage (you can override)
CMD ["python", "jobs/download_cutouts.py"]

########################
# ML target (heavy)
########################
FROM base AS ml

COPY requirements.txt /app/requirements.txt
COPY requirements-ml.txt /app/requirements-ml.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt -r /app/requirements-ml.txt

# Copy code
COPY jobs/ /app/jobs/
# COPY src/ /app/src/

# Single pipeline entrypoint (full run)
CMD ["python", "jobs/run_pipeline.py"]
