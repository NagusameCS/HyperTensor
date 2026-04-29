#!/usr/bin/env bash
# scripts/ec2_paperA_cachefit/build_ubuntu_cuda.sh
# Build geodessical (CUDA) on Ubuntu 22.04 + AWS Deep Learning Base AMI.
#
# Pre-reqs (DLAMI provides):
#   - NVIDIA driver + /usr/local/cuda (12.x or 13.x)
#   - build-essential
#
# We install:
#   - zig 0.14.0 -> /opt/zig
#   - libopenblas-dev (apt)
#
# Output:
#   $SRC_DIR/cuda_kernels.so
#   $SRC_DIR/geodessical
#
# Usage:
#   SRC_DIR=/opt/hypertensor SM_ARCH=sm_89 bash build_ubuntu_cuda.sh
set -euo pipefail

SRC_DIR="${SRC_DIR:-/opt/hypertensor}"
SM_ARCH="${SM_ARCH:-sm_89}"   # L40S=sm_89, A10G=sm_86, A100=sm_80, H100=sm_90

cd "$SRC_DIR"

echo "[build] using SRC_DIR=$SRC_DIR  SM_ARCH=$SM_ARCH"

# 1) zig 0.14.0
if [ ! -x /opt/zig/zig ]; then
  echo "[build] installing zig 0.14.0..."
  curl -fsSL -o /tmp/zig.tar.xz https://ziglang.org/download/0.14.0/zig-linux-x86_64-0.14.0.tar.xz
  sudo mkdir -p /opt/zig
  sudo tar xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1
fi

# 2) openblas
if ! dpkg -s libopenblas-dev >/dev/null 2>&1; then
  echo "[build] apt installing libopenblas-dev..."
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libopenblas-dev pkg-config
fi

# 3) locate CUDA
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  # try common alternatives
  for p in /usr/local/cuda-13.1 /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda-12.1; do
    if [ -x "$p/bin/nvcc" ]; then CUDA_HOME="$p"; break; fi
  done
fi
echo "[build] CUDA_HOME=$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version | head -n4

export PATH="$CUDA_HOME/bin:$PATH"

# 4) cuda kernels (separate .so loaded by backend_cuda.c)
echo "[build] compiling cuda_kernels.so ($SM_ARCH)..."
"$CUDA_HOME/bin/nvcc" -O2 -shared -Xcompiler -fPIC -arch="$SM_ARCH" \
    -lineinfo \
    runtime/nn/cuda_kernels.cu \
    -I. -Iruntime/nn \
    -o cuda_kernels.so

# 5) host binary with CUDA backend enabled
echo "[build] compiling geodessical (CUDA-enabled)..."
/opt/zig/zig cc -target x86_64-linux-gnu -O2 \
    -msse2 -mavx2 -mfma \
    -DGEODESSICAL_HOSTED=1 -DENABLE_CUDA -D_GNU_SOURCE \
    -Ihost/shims -I. -Ihost \
    -isystem /usr/include/x86_64-linux-gnu \
    -Wno-everything \
    host/hal.c host/main.c host/api_server.c host/gd_daemon.c host/mcp_server.c \
    runtime/nn/llm.c runtime/nn/gguf.c runtime/nn/backend.c runtime/nn/backend_cuda.c \
    runtime/nn/model_meta.c runtime/nn/tensor_bridge.c runtime/nn/mod_package.c \
    runtime/nn/token_comm.c runtime/nn/hf_download.c runtime/nn/flash_attn.c \
    runtime/nn/axiom_linalg.c runtime/nn/axiom_geo.c runtime/nn/axiom_beta.c \
    runtime/nn/axiom_exploit.c runtime/nn/axiom_gauge.c runtime/nn/axiom_vis.c \
    runtime/nn/online_basis.c runtime/nn/geo_research.c runtime/nn/mcr_compress.c \
    runtime/nn/thermal_rank.c runtime/nn/qspec_basis.c runtime/nn/jit_kernel.c \
    runtime/jit/x86_jit.c runtime/jit/llm_jit.c \
    -o geodessical.new \
    -L/usr/lib/x86_64-linux-gnu -lopenblas -lm -lpthread -ldl

# 6) swap in
[ -f geodessical ] && cp geodessical "geodessical.bak.$(date +%s)" || true
mv geodessical.new geodessical
chmod +x geodessical

echo "[build] OK -> $(ls -la geodessical cuda_kernels.so)"
./geodessical --help 2>&1 | grep -E "axex-weight-pca|axex-attn-only|axex-skip-o" | head -n5 || true
