#!/usr/bin/env bash
# scripts/campaign/build_remote_arch_cuda.sh
# Build geodessical for Linux/CUDA on the Arch Linux remote (ssh.opencs.dev).
# Run AFTER source has been rsync'd or tarball'd into /root/HyperTensor.
# Tested against:
#   - Arch Linux 6.19.9, RTX 3050 6GB, driver 595.58.03
#   - CUDA 13.1.1 (/opt/cuda)
#   - openblas 0.3.32-2 (/usr/lib/libopenblas.so)
#   - zig 0.14.0 (/opt/zig/zig)
#
# Produces:
#   /root/HyperTensor/cuda_kernels.so   (~1.5 MB, sm_86)
#   /root/HyperTensor/geodessical       (CUDA-enabled host binary)
set -euo pipefail

cd /root/HyperTensor

# 1) zig
if [ ! -x /opt/zig/zig ]; then
  echo "[build] installing zig 0.14.0..."
  curl -fsSL -o /tmp/zig.tar.xz https://ziglang.org/download/0.14.0/zig-linux-x86_64-0.14.0.tar.xz
  mkdir -p /opt/zig && tar xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1
fi

# 2) openblas
if [ ! -f /usr/include/cblas.h ]; then
  pacman -Sy --noconfirm openblas
fi

export PATH=/opt/cuda/bin:$PATH

# 3) cuda kernels (separate .so loaded at runtime by backend_cuda.c)
echo "[build] compiling cuda_kernels.so (sm_86 = RTX 30xx)..."
nvcc -O2 -shared -Xcompiler -fPIC -arch=sm_86 \
    runtime/nn/cuda_kernels.cu \
    -I. -Iruntime/nn \
    -o cuda_kernels.so

# 4) host binary with CUDA backend enabled
echo "[build] compiling geodessical (CUDA-enabled)..."
/opt/zig/zig cc -target x86_64-linux-gnu -O2 \
    -msse2 -mavx2 -mfma \
    -DGEODESSICAL_HOSTED=1 -DENABLE_CUDA \
    -Ihost/shims -I. -Ihost -I/usr/include \
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
    -L/usr/lib -lopenblas -lm -lpthread -ldl

# 5) swap in
[ -f geodessical ] && cp geodessical "geodessical.bak.$(date +%s)"
mv geodessical.new geodessical
chmod +x geodessical

echo "[build] OK"
ls -lh /root/HyperTensor/geodessical /root/HyperTensor/cuda_kernels.so
echo "[build] verifying weight-pca flag..."
./geodessical --help 2>&1 | grep -q "axex-weight-pca" && echo "  axex-weight-pca: present" || echo "  WARNING: axex-weight-pca not in help (flag still works in argv parser)"
