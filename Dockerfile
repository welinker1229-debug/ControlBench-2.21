# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Arguments for version control
ARG TORCH_VERSION=2.4.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG DGL_PACKAGE=dgl==2.4.0
ARG DGL_WHEEL_SRC=https://data.dgl.ai/wheels/torch-2.4/repo.html
ARG PYG_WHEEL_SRC=https://data.pyg.org/whl/torch-2.4.0+cpu.html

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    DGL_DISABLE_GRAPHBOLT=1

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# 1. Install Base PyTorch
RUN pip install --upgrade pip && \
    pip install torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL}

# 2. Install DGL (Separated for caching/debugging)
RUN if [ -n "${DGL_WHEEL_SRC}" ]; then \
        pip install --find-links ${DGL_WHEEL_SRC} ${DGL_PACKAGE}; \
    else \
        pip install ${DGL_PACKAGE}; \
    fi

# 3. Install PyTorch Geometric Dependencies
RUN pip install --find-links ${PYG_WHEEL_SRC} \
        torch-scatter \
        torch-sparse \
        torch-cluster \
        torch-spline-conv \
        torch-geometric

# 4. Install Remaining Requirements
RUN sed -e '/^torch/d' \
        -e '/^dgl/d' \
        -e '/^torchdata/d' \
        -e '/^simcse/d' requirements.txt > /tmp/reqs.txt && \
    pip install -r /tmp/reqs.txt

# 5. Install Standalone Libraries
RUN pip install --no-deps simcse==0.4 && \
    pip install --no-deps H2GB

COPY . .

CMD ["python", "train.py", "--model", "HAN", "--dataset", "abortion"]