# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Build arguments let you pick CPU or CUDA wheels for DGL/torch
# Suggested combos:
#   CPU (tested):    DGL_PACKAGE=dgl==1.1.3       TORCH_VERSION=2.0.1
#   CUDA 12.4:       DGL_WHEEL_SRC=https://data.dgl.ai/wheels/cu124/repo.html DGL_PACKAGE=dgl-cu124==2.4.0 TORCH_VERSION=2.4.0
ARG DGL_WHEEL_SRC=
ARG DGL_PACKAGE=dgl==1.1.3
ARG TORCH_VERSION=2.0.1
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    DGL_WHEEL_SRC=${DGL_WHEEL_SRC} \
    DGL_PACKAGE=${DGL_PACKAGE} \
    TORCH_VERSION=${TORCH_VERSION} \
    DGL_DISABLE_GRAPHBOLT=1

WORKDIR /app

# System deps for scientific/python tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    # Remove any stale torch pins in requirements.txt before install; then install with desired wheel source.
    sed -e '/^torch==/d' -e '/^dgl==/d' -e '/^torchdata/d' requirements.txt > /tmp/reqs.txt && \
    pip install torch==${TORCH_VERSION} && \
    if [ -n "${DGL_WHEEL_SRC}" ]; then \
        pip install --find-links ${DGL_WHEEL_SRC} ${DGL_PACKAGE}; \
    else \
        pip install ${DGL_PACKAGE}; \
    fi && \
    if [ -n "${DGL_WHEEL_SRC}" ]; then \
        pip install --find-links ${DGL_WHEEL_SRC} -r /tmp/reqs.txt; \
    else \
        pip install -r /tmp/reqs.txt; \
    fi

# Copy project
COPY . .

# Example default command (override as needed)
CMD ["python", "train.py", "--model", "HAN", "--dataset", "abortion"]
