# =============================================================================
# HyperTensor — Dockerfile
#
# Produces a reproducible research image with:
#   - All C build tools and LAPACKE
#   - Python 3.12 + core research dependencies
#   - libhypercore.so compiled and installed
#   - A non-root user `researcher` ready to run reproducibility scripts
#
# Build:
#   docker build -t hypertensor .
#
# Run Path A (CPU mathematical verification, no GPU needed):
#   docker run --rm hypertensor python scripts/faithfulness_rigorous.py
#
# Interactive shell:
#   docker run --rm -it hypertensor bash
#
# Core reproducibility scripts are the intended default entry point.
# =============================================================================

FROM ubuntu:24.04 AS base

# Avoid interactive prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build toolchain
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    # LAPACK / BLAS (OpenBLAS ships LAPACKE)
    libopenblas-dev \
    liblapacke-dev \
    # Python
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    # Utilities
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

# =============================================================================
# Stage: build — compile libhypercore.so
# =============================================================================
FROM base AS build

WORKDIR /src

# Copy only the files needed for the C build (keeps this layer cache-friendly)
COPY CMakeLists.txt .
COPY hypercore/hypercore.c hypercore/hypercore.h hypercore/
COPY lib/ lib/

RUN cmake -B /build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DHT_BUILD_RUNTIME=OFF \
    -DHT_BUILD_TESTS=OFF \
        -GNinja \
    && cmake --build /build --parallel \
    && cmake --install /build

# =============================================================================
# Stage: python-deps — install Python packages into a venv
# =============================================================================
FROM base AS python-deps

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Core research dependencies (no GPU torch; override with build-arg if needed)
ARG TORCH_INDEX="https://download.pytorch.org/whl/cpu"
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    scipy>=1.11 \
    mpmath>=1.3 \
    sympy>=1.12 \
    safetensors>=0.4

# Install torch CPU — override TORCH_INDEX for CUDA wheels:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
RUN pip install --no-cache-dir torch --index-url "${TORCH_INDEX}"

# HuggingFace stack (needed for Path B / model analysis)
RUN pip install --no-cache-dir \
    transformers>=4.40 \
    huggingface-hub>=0.22 \
    accelerate>=0.29 \
    bitsandbytes>=0.43 || true   # bitsandbytes may fail on pure-CPU; that's OK

# =============================================================================
# Stage: final — lean runtime image
# =============================================================================
FROM base AS final

# Pull in compiled shared libraries
COPY --from=build /usr/local/lib/libhypercore* /usr/local/lib/
COPY --from=build /usr/local/include/hypertensor/ /usr/local/include/hypertensor/

RUN ldconfig

# Pull in Python venv
COPY --from=python-deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

# Copy project source
WORKDIR /workspace
COPY . /workspace

# Tell the ctypes wrapper where to find libhypercore.so
ENV HT_LIB_DIR=/usr/local/lib

# Create non-root user
RUN useradd -m -s /bin/bash researcher \
 && chown -R researcher:researcher /workspace
USER researcher

# Smoke-test that the library loads and Python imports work
RUN python -c "import sys; sys.path.insert(0, '.'); \
        from hypercore import GeodesicMetric; \
        print('hypercore import OK')"

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '.'); import hypercore; print('healthy')"

# Default: run the CPU mathematical verification suite (Path A)
CMD ["python", "scripts/faithfulness_rigorous.py"]
