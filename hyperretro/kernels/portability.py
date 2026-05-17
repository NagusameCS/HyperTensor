#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................:::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............:::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............:::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""HyperRetro hardware coverage — build and portability guide.

Supported backends and their status:

  CUDA (NV)      : source at csrc/cuda/gemv_dual_q8_0.cu  — build script ready
  CPU AVX2 (x86) : source at csrc/cpu/hyperretro_cpu_avx2.c — build script below
  HIP (AMD)      : source at csrc/hip/gemv_dual_q8_0.hip  — algorithmic port
  Metal (Apple)  : source at csrc/metal/gemv_dual_q8_0.metal — algorithmic port

Build commands and portability notes below.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"


# ---------------------------------------------------------------------------
# CPU AVX2 build
# ---------------------------------------------------------------------------

def build_cpu_avx2(target_dir: str | Path | None = None) -> bool:
    """Compile the AVX2 CPU kernel into a shared library.

    Requires gcc or clang with AVX2+FMA support (any x86_64 CPU from
    Haswell (2013) onwards).

    Returns True on success.
    """
    src = CSRC / "cpu" / "hyperretro_cpu_avx2.c"
    if not src.exists():
        print(f"CPU AVX2 source not found: {src}")
        return False

    target = Path(target_dir) if target_dir else CSRC / "cpu"
    target.mkdir(parents=True, exist_ok=True)

    if sys.platform == "win32":
        out = target / "hyperretro_cpu_avx2.dll"
        cc = shutil.which("gcc") or shutil.which("clang")
        if cc is None:
            print("No gcc/clang found on Windows. Install MinGW or use WSL.")
            return False
    elif sys.platform == "darwin":
        out = target / "hyperretro_cpu_avx2.dylib"
        cc = "clang"
    else:
        out = target / "hyperretro_cpu_avx2.so"
        cc = shutil.which("gcc") or "clang"

    cmd = [
        cc, "-O3", "-mavx2", "-mfma", "-shared", "-fPIC",
        "-o", str(out), str(src),
    ]

    print(f"Building CPU AVX2 kernel: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build FAILED:\n{result.stderr}")
        return False

    size_kb = out.stat().st_size / 1024
    print(f"  Built: {out} ({size_kb:.0f} KB)")
    return True


# ---------------------------------------------------------------------------
# HIP (AMD) build — algorithmic port of CUDA kernel
# ---------------------------------------------------------------------------

HIP_KERNEL_SOURCE = '''
/* HyperRetro HIP backend — fused dual Q8_0 GEMV (AMD GPUs).

   Algorithmic port of csrc/cuda/gemv_dual_q8_0.cu.
   HIP is source-compatible with CUDA for kernel code; the same algorithm
   runs on AMD GPUs (MI200, MI300, RX 7000, etc.).

   Build:
       hipcc -O3 -shared -fPIC -o gemv_dual_q8_0.hip.so gemv_dual_q8_0.hip

   The kernel uses:
     - Shared memory for input vector (loaded once → 2 dot products)
     - Block size 32 (one warp on AMD, two on NV)
     - Q8_0 dequant: int8 codes × float32 scale per 32-element block
*/

#include <hip/hip_runtime.h>
#include <cstdint>

static constexpr int BLOCK_SIZE = 32;

__global__ void kernel_gemv_dual_q8_0(
    const float *__restrict__ x,
    const float *__restrict__ scale_a,
    const int8_t *__restrict__ codes_a,
    const float *__restrict__ scale_b,
    const int8_t *__restrict__ codes_b,
    float *__restrict__ out_a,
    float *__restrict__ out_b,
    int rows,
    int n_blocks
) {
    __shared__ float x_shared[BLOCK_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (tid < BLOCK_SIZE) {
        x_shared[tid] = x[tid];
    }
    __syncthreads();

    if (row >= rows) return;

    float sum_a = 0.0f, sum_b = 0.0f;
    int row_offset = row * n_blocks * BLOCK_SIZE;

    for (int b = 0; b < n_blocks; ++b) {
        int block_base = b * BLOCK_SIZE;
        float s_a = scale_a[row * n_blocks + b];
        float s_b = scale_b[row * n_blocks + b];
        float dot_a = 0.0f, dot_b = 0.0f;

        for (int i = tid; i < BLOCK_SIZE; i += blockDim.x) {
            float xi = x_shared[i];
            dot_a += static_cast<float>(codes_a[row_offset + block_base + i]) * xi;
            dot_b += static_cast<float>(codes_b[row_offset + block_base + i]) * xi;
        }

        // Warp-level reduction (AMD: wavefront)
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_a += __shfl_down(dot_a, offset);
            dot_b += __shfl_down(dot_b, offset);
        }

        sum_a += dot_a * s_a;
        sum_b += dot_b * s_b;
    }

    if (tid == 0) {
        out_a[row] = sum_a;
        out_b[row] = sum_b;
    }
}
'''


# ---------------------------------------------------------------------------
# Metal (Apple) build — algorithmic port for M-series GPUs
# ---------------------------------------------------------------------------

METAL_KERNEL_SOURCE = '''
/* HyperRetro Metal backend — fused dual Q8_0 GEMV (Apple Silicon).

   Algorithmic port of the CUDA/HIP kernel for Apple M-series GPUs.
   Uses Metal Shading Language (MSL).

   The kernel shares the input vector load between two Q8_0 dot products.

   Build:
       xcrun -sdk macosx metal -c gemv_dual_q8_0.metal -o gemv_dual_q8_0.air
       xcrun -sdk macosx metallib gemv_dual_q8_0.air -o gemv_dual_q8_0.metallib

   Q8_0 layout (same as CUDA/HIP):
     - 32-element blocks
     - Per block: float32 scale, int8 codes[32]
     - Scales packed as float array, codes as int8 array
*/

#include <metal_stdlib>
using namespace metal;

constant int BLOCK_SIZE = 32;

kernel void gemv_dual_q8_0(
    device const float *x           [[buffer(0)]],
    device const float *scales_a    [[buffer(1)]],
    device const char  *codes_a     [[buffer(2)]],
    device const float *scales_b    [[buffer(3)]],
    device const char  *codes_b     [[buffer(4)]],
    device float       *out_a       [[buffer(5)]],
    device float       *out_b       [[buffer(6)]],
    constant int  &rows             [[buffer(7)]],
    constant int  &n_blocks         [[buffer(8)]],
    uint           tid              [[thread_position_in_threadgroup]],
    uint           row              [[threadgroup_position_in_grid]]
) {
    threadgroup float x_shared[BLOCK_SIZE];

    if (tid < BLOCK_SIZE) {
        x_shared[tid] = x[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= (uint)rows) return;

    float sum_a = 0.0f, sum_b = 0.0f;
    int row_offset = row * n_blocks * BLOCK_SIZE;

    for (int b = 0; b < n_blocks; ++b) {
        int block_base = b * BLOCK_SIZE;
        float s_a = scales_a[row * n_blocks + b];
        float s_b = scales_b[row * n_blocks + b];
        float dot_a = 0.0f, dot_b = 0.0f;

        for (int i = tid; i < BLOCK_SIZE; i += BLOCK_SIZE) {
            float xi = x_shared[i];
            dot_a += float(codes_a[row_offset + block_base + i]) * xi;
            dot_b += float(codes_b[row_offset + block_base + i]) * xi;
        }

        // Simdgroup reduction
        dot_a = simd_sum(dot_a);
        dot_b = simd_sum(dot_b);

        sum_a += dot_a * s_a;
        sum_b += dot_b * s_b;
    }

    if (tid == 0) {
        out_a[row] = sum_a;
        out_b[row] = sum_b;
    }
}
'''


# ---------------------------------------------------------------------------
# Portability summary
# ---------------------------------------------------------------------------

def print_portability_matrix():
    """Print the hardware coverage matrix."""
    print("""
HyperRetro Hardware Coverage Matrix
═══════════════════════════════════════
 Backend        │ Source              │ Status
────────────────┼─────────────────────┼──────────
 CUDA (NV)      │ csrc/cuda/*.cu      │  Source + build script
 CPU AVX2 (x86) │ csrc/cpu/*.c        │  Source + build script
 HIP (AMD)      │ csrc/hip/*.hip      │  Algorithmic port
 Metal (Apple)  │ csrc/metal/*.metal  │  Algorithmic port
 Torch GPU      │ kernels/gpu.py      │  Working (no nvcc)
 Torch CPU      │ kernels/__init__.py │  Fallback
 NumPy          │ kernels/__init__.py │  Always available
────────────────┼─────────────────────┼──────────

All backends share the identical Q8_0 layout and algorithm
(fused dual GEMV with shared input load). Porting between
backends is mechanical — same 32-element block structure,
same dequant formula.
""")


if __name__ == "__main__":
    print_portability_matrix()
