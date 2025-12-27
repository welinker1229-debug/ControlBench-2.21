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
    # Strip out torch/dgl/torchdata to manage manually.
    # ALSO strip out simcse to prevent "scipy<1.6" dependency crash.
    sed -e '/^torch/d' \
        -e '/^dgl/d' \
        -e '/^torchdata/d' \
        -e '/^simcse/d' requirements.txt > /tmp/reqs.txt && \
    \
    #############################################
    # 1. Install torch first
    #############################################
    pip install torch==${TORCH_VERSION} && \
    \
    #############################################
    # 2. Install DGL (with or without custom wheel source)
    #############################################
    if [ -n "${DGL_WHEEL_SRC}" ]; then \
        pip install --find-links ${DGL_WHEEL_SRC} ${DGL_PACKAGE}; \
    else \
        pip install ${DGL_PACKAGE}; \
    fi && \
    \
    #############################################
    # 3. Install PyTorch Geometric stack from prebuilt wheels
    #    (for torch==2.4.0 + cu124)
    #############################################
    pip install --find-links https://data.pyg.org/whl/torch-2.4.0+cu124.html \
        torch-scatter==2.1.2 \
        torch-sparse==0.6.18 \
        torch-cluster==1.6.3 \
        torch-spline-conv==1.2.2 \
        torch-geometric==2.7.0 && \
    \
    #############################################
    # 4. Install the rest of the requirements
    #############################################
    pip install -r /tmp/reqs.txt && \
    \
    #############################################
    # 5. Install SimCSE and H2GB, WITHOUT pulling deps
    #    (SimCSE requires this to bypass scipy version check)
    #############################################
    pip install --no-deps simcse==0.4 && \
    pip install --no-deps H2GB


# Copy project
COPY . .

# Example default command (override as needed)
CMD ["python", "train.py", "--model", "HAN", "--dataset", "abortion"]