/*
 * TensorOS CUDA Kernels — GPU Implementations
 *
 * Real CUDA kernel implementations for LLM inference.
 * Compile with: nvcc -shared -o cuda_kernels.dll cuda_kernels.cu
 *               -DCUDA_KERNELS_EXPORTS --compiler-options /MD -lcublas
 *         Linux: nvcc -shared -o cuda_kernels.so  cuda_kernels.cu
 *               -DCUDA_KERNELS_EXPORTS -fPIC -lcublas
 *
 * Key optimizations:
 *   - Q4_0 dequant fused into GEMV (one kernel, no intermediate buffer)
 *   - Warp-level reductions for RMSNorm/Softmax (shfl_down_sync)
 *   - Coalesced memory access patterns
 *   - Shared memory for attention score computation
 *   - cuBLAS batched GEMV for compressed FFN gate+up fusion
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "cuda_kernels.h"

/* IEEE754 negative infinity for CUDA device code */
#define NEG_INF __int_as_float(0xFF800000)

/* ═══════════════════════════════════════════════════════════════════════
 * Helpers
 * ════════════════════════════════════════════════════════════════════════ */

/* FP16 → FP32 conversion (device) */
__device__ __forceinline__ float fp16_to_f32(uint16_t h) {
    return __half2float(__ushort_as_half(h));
}

/* Q4_0 block structure: fp16 scale + 16 bytes (32 nibbles) = 18 bytes */
#pragma pack(push, 1)
struct q4_0_block {
    uint16_t d;      /* FP16 scale */
    uint8_t  qs[16]; /* 32 x 4-bit quants packed into 16 bytes */
};
#pragma pack(pop)

/* Q4_1 block structure: fp16 scale + fp16 min + 16 bytes (32 nibbles) = 20 bytes */
#pragma pack(push, 1)
struct q4_1_block {
    uint16_t d;      /* FP16 scale */
    uint16_t m;      /* FP16 minimum */
    uint8_t  qs[16]; /* 32 x 4-bit quants packed into 16 bytes */
};
#pragma pack(pop)

/* Q8_0 block: fp16 scale + 32 int8 = 34 bytes */
#pragma pack(push, 1)
struct q8_0_block {
    uint16_t d;
    int8_t   qs[32];
};
#pragma pack(pop)

/* Q6_K super-block: 256 elements, 210 bytes */
#pragma pack(push, 1)
struct q6_k_block {
    uint8_t ql[128];    /* lower 4 bits of quants */
    uint8_t qh[64];     /* upper 2 bits of quants */
    int8_t  scales[16]; /* 16 sub-block scales */
    uint16_t d;         /* FP16 super-block scale */
};
#pragma pack(pop)

/* Warp-level sum reduction */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

/* Warp-level max reduction */
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

/* Block-level sum using shared memory */
__device__ float block_reduce_sum(float val, float *shared) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);
    return val;
}

/* Block-level max using shared memory */
__device__ float block_reduce_max(float val, float *shared) {
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int n_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < n_warps) ? shared[threadIdx.x] : NEG_INF;
    if (warp_id == 0) val = warp_reduce_max(val);
    return val;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GEMV Kernel — Q4_0 quantized matrix @ float vector
 *
 * Shared memory tiling: input vector loaded once per block into shared
 * memory, shared by all 8 warps. Eliminates redundant L2 traffic.
 *
 * Warp-per-row: 256 threads = 8 warps, each warp handles one row.
 * Uses dp4a (INT8 dot product) with on-the-fly Q8 input quantization
 * for maximum throughput on Ada Lovelace/Ampere GPUs.
 * ════════════════════════════════════════════════════════════════════════ */

#define GEMV_Q4_TILE 4096  /* input elements per shared-memory tile */

__global__ void kernel_gemv_q4_0(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    extern __shared__ char smem_raw[];
    int nb = in_dim / 32;
    int8_t *q8       = (int8_t *)smem_raw;                        /* [in_dim] */
    float  *q8_sc    = (float *)(smem_raw + in_dim);              /* [nb] */
    float  *q8_sums  = q8_sc + nb;                                /* [nb] */

    int n_warps = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * n_warps + warp_id;

    /* Step 1: Cooperatively quantize input x to Q8 in shared memory */
    for (int b = warp_id; b < nb; b += n_warps) {
        float val = x[b * 32 + lane];
        float amax = fabsf(val);
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
        float inv_sc = (amax > 1e-10f) ? 127.0f / amax : 0.0f;
        int qi = __float2int_rn(val * inv_sc);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        q8[b * 32 + lane] = (int8_t)qi;
        int isum = qi;
        for (int off = 16; off > 0; off >>= 1)
            isum += __shfl_xor_sync(0xFFFFFFFF, isum, off);
        if (lane == 0) {
            float sc = amax / 127.0f;
            q8_sc[b]   = sc;
            q8_sums[b] = sc * (float)isum;
        }
    }
    __syncthreads();

    if (row >= out_dim) return;

    /* Step 2: dp4a GEMV for this row */
    const struct q4_0_block *blocks =
        (const struct q4_0_block *)W + (int64_t)row * nb;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float d = fp16_to_f32(blocks[b].d);
        uint32_t raw[4];
        memcpy(raw, blocks[b].qs, 16);
        uint32_t qlo[4], qhi[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            qlo[k] = raw[k] & 0x0F0F0F0Fu;
            qhi[k] = (raw[k] >> 4) & 0x0F0F0F0Fu;
        }
        const int32_t *q8_lo = (const int32_t *)(q8 + b * 32);
        const int32_t *q8_hi = (const int32_t *)(q8 + b * 32 + 16);
        int isum_all = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            isum_all = __dp4a((int)qlo[k], q8_lo[k], isum_all);
            isum_all = __dp4a((int)qhi[k], q8_hi[k], isum_all);
        }
        sum += d * (q8_sc[b] * (float)isum_all - 8.0f * q8_sums[b]);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

/* Quantize input vector x (nb blocks of 32) to Q8 once per GEMV call. */
__global__ void kernel_quantize_x_q8(
    const float *x,
    int8_t      *q8,
    float       *q8_sc,
    float       *q8_sums,
    int          nb)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    if (warp >= nb) return;

    float val = x[warp * 32 + lane];
    float amax = fabsf(val);
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));

    float inv_sc = (amax > 1e-10f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(val * inv_sc);
    if (qi > 127) qi = 127;
    if (qi < -127) qi = -127;
    q8[warp * 32 + lane] = (int8_t)qi;

    int isum = qi;
    for (int off = 16; off > 0; off >>= 1)
        isum += __shfl_xor_sync(0xFFFFFFFF, isum, off);
    if (lane == 0) {
        float sc = amax / 127.0f;
        q8_sc[warp]   = sc;
        q8_sums[warp] = sc * (float)isum;
    }
}

/* Q4_0 GEMV using pre-quantized x (q8/q8_sc/q8_sums) in global memory. */
__global__ void kernel_gemv_q4_0_prequant(
    float        *out,
    const void   *W,
    const int8_t *q8,
    const float  *q8_sc,
    const float  *q8_sums,
    int           out_dim,
    int           in_dim)
{
    int nb = in_dim / 32;
    int n_warps = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * n_warps + warp_id;
    if (row >= out_dim) return;

    const struct q4_0_block *blocks =
        (const struct q4_0_block *)W + (int64_t)row * nb;

    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float d = fp16_to_f32(blocks[b].d);
        uint32_t raw[4];
        memcpy(raw, blocks[b].qs, 16);
        uint32_t qlo[4], qhi[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            qlo[k] = raw[k] & 0x0F0F0F0Fu;
            qhi[k] = (raw[k] >> 4) & 0x0F0F0F0Fu;
        }

        const int32_t *q8_lo = (const int32_t *)(q8 + b * 32);
        const int32_t *q8_hi = (const int32_t *)(q8 + b * 32 + 16);
        int isum_all = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            isum_all = __dp4a((int)qlo[k], q8_lo[k], isum_all);
            isum_all = __dp4a((int)qhi[k], q8_hi[k], isum_all);
        }
        sum += d * (q8_sc[b] * (float)isum_all - 8.0f * q8_sums[b]);
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

/* GEMV for Q4_1: each block = {fp16 scale, fp16 min, uint8[16]} = 20 bytes
 * Dequantized value: scale * q + min, where q = nibble value (0..15) */
__global__ void kernel_gemv_q4_1(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int n_blocks = in_dim / 32;
    const struct q4_1_block *blocks =
        (const struct q4_1_block *)W + (int64_t)row * n_blocks;

    float sum = 0.0f;

    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float scale = fp16_to_f32(blocks[b].d);
        float min   = fp16_to_f32(blocks[b].m);
        const float4 *xv = (const float4 *)(x + b * 32);

        float4 xlo0 = xv[0], xlo1 = xv[1], xlo2 = xv[2], xlo3 = xv[3];
        float4 xhi0 = xv[4], xhi1 = xv[5], xhi2 = xv[6], xhi3 = xv[7];

        /* Accumulate: scale * sum(q * x) + min * sum(x) */
        float scaled = 0.0f;
        float xsum = 0.0f;
        uint8_t p;
        p = blocks[b].qs[ 0]; scaled += (float)(p&0xF)*xlo0.x + (float)(p>>4)*xhi0.x;
        p = blocks[b].qs[ 1]; scaled += (float)(p&0xF)*xlo0.y + (float)(p>>4)*xhi0.y;
        p = blocks[b].qs[ 2]; scaled += (float)(p&0xF)*xlo0.z + (float)(p>>4)*xhi0.z;
        p = blocks[b].qs[ 3]; scaled += (float)(p&0xF)*xlo0.w + (float)(p>>4)*xhi0.w;
        p = blocks[b].qs[ 4]; scaled += (float)(p&0xF)*xlo1.x + (float)(p>>4)*xhi1.x;
        p = blocks[b].qs[ 5]; scaled += (float)(p&0xF)*xlo1.y + (float)(p>>4)*xhi1.y;
        p = blocks[b].qs[ 6]; scaled += (float)(p&0xF)*xlo1.z + (float)(p>>4)*xhi1.z;
        p = blocks[b].qs[ 7]; scaled += (float)(p&0xF)*xlo1.w + (float)(p>>4)*xhi1.w;
        p = blocks[b].qs[ 8]; scaled += (float)(p&0xF)*xlo2.x + (float)(p>>4)*xhi2.x;
        p = blocks[b].qs[ 9]; scaled += (float)(p&0xF)*xlo2.y + (float)(p>>4)*xhi2.y;
        p = blocks[b].qs[10]; scaled += (float)(p&0xF)*xlo2.z + (float)(p>>4)*xhi2.z;
        p = blocks[b].qs[11]; scaled += (float)(p&0xF)*xlo2.w + (float)(p>>4)*xhi2.w;
        p = blocks[b].qs[12]; scaled += (float)(p&0xF)*xlo3.x + (float)(p>>4)*xhi3.x;
        p = blocks[b].qs[13]; scaled += (float)(p&0xF)*xlo3.y + (float)(p>>4)*xhi3.y;
        p = blocks[b].qs[14]; scaled += (float)(p&0xF)*xlo3.z + (float)(p>>4)*xhi3.z;
        p = blocks[b].qs[15]; scaled += (float)(p&0xF)*xlo3.w + (float)(p>>4)*xhi3.w;

        xsum = xlo0.x+xlo0.y+xlo0.z+xlo0.w + xlo1.x+xlo1.y+xlo1.z+xlo1.w
             + xlo2.x+xlo2.y+xlo2.z+xlo2.w + xlo3.x+xlo3.y+xlo3.z+xlo3.w
             + xhi0.x+xhi0.y+xhi0.z+xhi0.w + xhi1.x+xhi1.y+xhi1.z+xhi1.w
             + xhi2.x+xhi2.y+xhi2.z+xhi2.w + xhi3.x+xhi3.y+xhi3.z+xhi3.w;

        sum += scale * scaled + min * xsum;
    }

    __shared__ float smem[32];
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0) out[row] = sum;
}

/* GEMV for Q8_0: each block = {fp16 scale, int8[32]} */
__global__ void kernel_gemv_q8_0(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int n_blocks = in_dim / 32;
    const struct q8_0_block *blocks =
        (const struct q8_0_block *)W + (int64_t)row * n_blocks;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float scale = fp16_to_f32(blocks[b].d);
        const float *xp = x + b * 32;
        float local = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; j++)
            local += (float)blocks[b].qs[j] * xp[j];
        sum += scale * local;
    }

    __shared__ float smem[32];
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0) out[row] = sum;
}

/* Q8_0 GEMV element-parallel: each of 32 warp threads processes one element
 * position across ALL quant-blocks, then warp-reduces.  For in_dim=512 the
 * original kernel leaves threads 16-31 idle (n_blocks=16 < blockDim.x=32);
 * this variant keeps all 32 threads active regardless of n_blocks. */
__global__ void kernel_gemv_q8_0_elem(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;

    int n_blocks = in_dim / 32;
    const struct q8_0_block *blks =
        (const struct q8_0_block *)W + (int64_t)row * n_blocks;

    int j = threadIdx.x;   /* element index within each quant-block [0..31] */
    float sum = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        float scale = fp16_to_f32(blks[b].d);
        sum += scale * (float)blks[b].qs[j] * x[b * 32 + j];
    }
    sum = warp_reduce_sum(sum);
    if (j == 0) out[row] = sum;
}

/* Q8_0 GEMV batched: 8 rows per block (one warp per row).
 * This reduces block-launch count 8x versus row-per-block kernels and
 * improves scheduler occupancy on high out_dim decode/prefill paths.
 */
__global__ void kernel_gemv_q8_0_batched(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * 8 + warp_id;
    if (row >= out_dim) return;

    int n_blocks = in_dim / 32;
    const struct q8_0_block *blks =
        (const struct q8_0_block *)W + (int64_t)row * n_blocks;

    float sum = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        float scale = fp16_to_f32(blks[b].d);
        sum += scale * (float)blks[b].qs[lane] * x[b * 32 + lane];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

/* GEMV for F32: simple dense matrix-vector */
__global__ void kernel_gemv_f32(
    float       *out,
    const float *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int row = blockIdx.x;
    if (row >= out_dim) return;

    const float *w_row = W + (int64_t)row * in_dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        sum += w_row[i] * x[i];

    __shared__ float smem[32];
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0) out[row] = sum;
}

/* F32 GEMV batched: 8 rows per block (1 warp per row), mirroring kernel_gemv_f16.
 * Reduces block count by 8× (e.g. Pt projection 512→64 blocks) cutting scheduler
 * overhead and improving SM occupancy. */
__global__ void kernel_gemv_f32_batched(
    float       *out,
    const float *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * 8 + warp_id;
    if (row >= out_dim) return;

    const float *w_row = W + (int64_t)row * in_dim;
    float sum = 0.0f;
    for (int i = lane; i < in_dim; i += 32)
        sum += w_row[i] * x[i];

    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

/* GEMV for F16: fp16 weights, f32 activation */
__global__ void kernel_gemv_f16(
    float       *out,
    const __half *W,
    const float  *x,
    int           out_dim,
    int           in_dim)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * 8 + warp_id;
    if (row >= out_dim) return;

    const __half *w_row = W + (int64_t)row * in_dim;
    float sum = 0.0f;
    for (int i = lane; i < in_dim; i += 32)
        sum += __half2float(w_row[i]) * x[i];

    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * RMSNorm Kernel
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_rmsnorm(
    float       *out,
    const float *x,
    const float *w,
    int          dim,
    float        eps)
{
    __shared__ float smem[32];
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum_sq += x[i] * x[i];

    sum_sq = block_reduce_sum(sum_sq, smem);

    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out[i] = x[i] * inv_rms * w[i];
}

/* ═══════════════════════════════════════════════════════════════════════
 * LayerNorm Kernel
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_layernorm(
    float       *out,
    const float *x,
    const float *w,
    const float *bias,
    int          dim,
    float        eps)
{
    __shared__ float smem[32];

    /* Compute mean */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum += x[i];
    sum = block_reduce_sum(sum, smem);

    __shared__ float mean_val;
    if (threadIdx.x == 0) mean_val = sum / (float)dim;
    __syncthreads();

    /* Compute variance */
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = x[i] - mean_val;
        var_sum += d * d;
    }
    var_sum = block_reduce_sum(var_sum, smem);

    __shared__ float inv_std;
    if (threadIdx.x == 0) inv_std = rsqrtf(var_sum / (float)dim + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float norm = (x[i] - mean_val) * inv_std;
        out[i] = norm * w[i] + (bias ? bias[i] : 0.0f);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Softmax Kernel
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_softmax(float *x, int n) {
    __shared__ float smem[32];

    /* Pass 1: find max */
    float max_val = NEG_INF;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        max_val = fmaxf(max_val, x[i]);
    max_val = block_reduce_max(max_val, smem);

    __shared__ float max_shared;
    if (threadIdx.x == 0) max_shared = max_val;
    __syncthreads();

    /* Pass 2: exp and sum */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] = expf(x[i] - max_shared);
        sum += x[i];
    }
    sum = block_reduce_sum(sum, smem);

    __shared__ float inv_sum;
    if (threadIdx.x == 0) inv_sum = 1.0f / sum;
    __syncthreads();

    /* Pass 3: normalize */
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        x[i] *= inv_sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Elementwise Kernels
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_silu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ void kernel_gelu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

/* Fused GeGLU: out[i] = GELU(gate[i]) * up[i] — saves one kernel launch + memory round-trip */
__global__ void kernel_fused_geglu(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        float g = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        gate[i] = g * up[i];
    }
}

/* Fused SwiGLU: out[i] = SiLU(gate[i]) * up[i] */
__global__ void kernel_fused_swiglu(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

__global__ void kernel_mul(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void kernel_add(float *out, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void kernel_scale(float *out, const float *x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] * s;
}

__global__ void kernel_softcap(float *x, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ISWA Precompute Kernels — GPU-resident for Gemma4
 * ════════════════════════════════════════════════════════════════════════ */

/* Batched RMSNorm: normalize n_slices independent vectors of slice_dim elements.
 * One block per slice. Norm weights are shared across all slices. */
__global__ void kernel_batched_rmsnorm(
    float       *data,      /* [n_slices * slice_dim], in/out */
    const float *w,         /* [slice_dim], shared norm weights */
    int          n_slices,
    int          slice_dim,
    float        eps)
{
    int slice = blockIdx.x;
    if (slice >= n_slices) return;

    float *sl = data + slice * slice_dim;
    __shared__ float smem[32];

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < slice_dim; i += blockDim.x)
        sum_sq += sl[i] * sl[i];
    sum_sq = block_reduce_sum(sum_sq, smem);

    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)slice_dim + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < slice_dim; i += blockDim.x)
        sl[i] = sl[i] * inv_rms * w[i];
}

/* Fused ISWA combine: out[i] = (tok_embd[i] + proj[i]) * scale */
__global__ void kernel_iswa_combine(
    float       *out,
    const float *tok_embd,
    const float *proj,
    float        scale,
    int          n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (tok_embd[i] + proj[i]) * scale;
}

/* ═══════════════════════════════════════════════════════════════════════
 * RoPE Kernel
 *
 * Applies rotary position encoding to Q and K vectors.
 * Each pair (x[2i], x[2i+1]) is rotated by angle pos * freq_i.
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_rope(
    float       *q,
    float       *k,
    int          head_dim,
    int          n_heads,
    int          n_kv_heads,
    const int   *d_pos,
    float        base,
    const float *freqs)  /* precomputed freq table [head_dim/2], or NULL */
{
    int pos = *d_pos;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_hd = head_dim / 2;
    int total_pairs = n_heads * half_hd;

    if (i < total_pairs) {
        int head = i / half_hd;
        int pair = i % half_hd;

        float freq;
        if (freqs)
            freq = freqs[pair];
        else
            freq = 1.0f / powf(base, (float)(2 * pair) / (float)head_dim);

        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        /* Rotate Q */
        int qi = head * head_dim + 2 * pair;
        float q0 = q[qi], q1 = q[qi + 1];
        q[qi]     = q0 * cos_a - q1 * sin_a;
        q[qi + 1] = q0 * sin_a + q1 * cos_a;

        /* Rotate K (only for KV heads) */
        if (head < n_kv_heads) {
            int ki = head * head_dim + 2 * pair;
            float k0 = k[ki], k1 = k[ki + 1];
            k[ki]     = k0 * cos_a - k1 * sin_a;
            k[ki + 1] = k0 * sin_a + k1 * cos_a;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Dequantize Kernel
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_dequantize_q4_0(
    float      *out,
    const void *data,
    int         n_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_blocks = n_elements / 32;

    if (i < n_blocks) {
        const struct q4_0_block *b = (const struct q4_0_block *)data + i;
        float scale = fp16_to_f32(b->d);
        float *o = out + i * 32;

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t packed = b->qs[j];
            o[j]      = scale * ((float)(packed & 0x0F) - 8.0f);
            o[j + 16] = scale * ((float)(packed >> 4)   - 8.0f);
        }
    }
}

__global__ void kernel_dequantize_q8_0(
    float      *out,
    const void *data,
    int         n_elements)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_blocks = n_elements / 32;

    if (i < n_blocks) {
        const struct q8_0_block *b = (const struct q8_0_block *)data + i;
        float scale = fp16_to_f32(b->d);
        float *o = out + i * 32;

        #pragma unroll
        for (int j = 0; j < 32; j++)
            o[j] = scale * (float)b->qs[j];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Attention Kernel
 *
 * Single-query attention with GQA support.
 * For each head: score = Q_h @ K_cache[kv_h], softmax, V_cache[kv_h] @ score
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_attention(
    float       *out,        /* [n_heads * head_dim] */
    const float *Q,          /* [n_heads * head_dim] */
    const float *K_cache,    /* [n_kv_heads * max_seq * head_dim] */
    const float *V_cache,    /* [n_kv_heads * max_seq * head_dim] */
    int          n_heads,
    int          n_kv_heads,
    int          head_dim,
    const int   *d_seq_len,
    int          max_seq,    /* stride for KV cache (positions allocated) */
    float        scale,
    float        softcap,
    int          smem_seq)   /* shared mem was sized for this seq_len */
{
    int seq_len = *d_seq_len;
    if (seq_len > smem_seq) seq_len = smem_seq;  /* safety clamp */
    /* One block per attention head */
    int head = blockIdx.x;
    if (head >= n_heads) return;

    int kv_head = head * n_kv_heads / n_heads; /* GQA head mapping */

    extern __shared__ float shared[];
    float *scores = shared;                       /* [seq_len] */
    float *smem = shared + smem_seq;              /* [32] for reductions */

    /* Phase 1: Compute attention scores = Q_h @ K_cache[kv_h, :seq_len, :] */
    const float *q_h = Q + head * head_dim;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        const float *k_t = K_cache + ((int64_t)kv_head * max_seq + t) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q_h[d] * k_t[d];
        dot *= scale;

        /* Apply softcap if enabled */
        if (softcap > 0.0f)
            dot = softcap * tanhf(dot / softcap);

        scores[t] = dot;
    }
    __syncthreads();

    /* Phase 2: Softmax over scores */
    /* Find max */
    float max_val = NEG_INF;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x)
        max_val = fmaxf(max_val, scores[t]);
    max_val = block_reduce_max(max_val, smem);

    __shared__ float max_shared_val;
    if (threadIdx.x == 0) max_shared_val = max_val;
    __syncthreads();

    /* Exp and sum */
    float sum = 0.0f;
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        scores[t] = expf(scores[t] - max_shared_val);
        sum += scores[t];
    }
    sum = block_reduce_sum(sum, smem);

    __shared__ float inv_sum_shared;
    if (threadIdx.x == 0) inv_sum_shared = 1.0f / sum;
    __syncthreads();

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x)
        scores[t] *= inv_sum_shared;
    __syncthreads();

    /* Phase 3: Weighted sum of V: out_h = scores @ V_cache[kv_h] */
    float *out_h = out + head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const float *v_t = V_cache + ((int64_t)kv_head * max_seq + t) * head_dim;
            val += scores[t] * v_t[d];
        }
        out_h[d] = val;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * KV Cache Update
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_kv_update(
    float       *K_cache,    /* [n_kv_heads * max_seq * head_dim] */
    float       *V_cache,
    const float *K_new,      /* [n_kv_heads * head_dim] */
    const float *V_new,
    int          n_kv_heads,
    int          head_dim,
    const int   *d_pos,
    int          max_seq)
{
    int pos = *d_pos;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv_heads * head_dim;
    if (i >= total) return;

    int kv_head = i / head_dim;
    int d = i % head_dim;

    int64_t cache_idx = ((int64_t)kv_head * max_seq + pos) * head_dim + d;
    K_cache[cache_idx] = K_new[i];
    V_cache[cache_idx] = V_new[i];
}

/* ═══════════════════════════════════════════════════════════════════════
 * Embedding Lookup
 * ════════════════════════════════════════════════════════════════════════ */

/* For Q4_0 embedding: dequantize row token_id from table → out */
__global__ void kernel_embed_q4_0(
    float      *out,
    const void *table,
    int         token_id,
    int         dim)
{
    int n_blocks = dim / 32;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    const struct q4_0_block *blocks =
        (const struct q4_0_block *)table + (int64_t)token_id * n_blocks + b;

    float scale = fp16_to_f32(blocks->d);
    float *o = out + b * 32;

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint8_t packed = blocks->qs[j];
        o[j]      = scale * ((float)(packed & 0x0F) - 8.0f);
        o[j + 16] = scale * ((float)(packed >> 4)   - 8.0f);
    }
}

__global__ void kernel_embed_f32(
    float       *out,
    const float *table,
    int          token_id,
    int          dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = table[(int64_t)token_id * dim + i];
}

/* Embedding lookup for Q6_K: dequantize one row of Q6_K-quantized table */
__global__ void kernel_embed_q6_k(
    float      *out,
    const void *table,
    int         token_id,
    int         dim)
{
    int n_sb = dim / 256;
    int sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_sb) return;

    const uint8_t *row_base = (const uint8_t *)table + (int64_t)token_id * n_sb * 210;
    const struct q6_k_block *b = (const struct q6_k_block *)(row_base + sb * 210);
    float d_val = fp16_to_f32(b->d);
    float *o = out + sb * 256;

    for (int half = 0; half < 2; half++) {
        const uint8_t *ql_h = b->ql + half * 64;
        const uint8_t *qh_h = b->qh + half * 32;
        const int8_t *sc_h = b->scales + half * 8;
        for (int l = 0; l < 32; l++) {
            int q0 = (int)(ql_h[l] & 0xF)       | (int)(((qh_h[l] >> 0) & 3) << 4);
            int q1 = (int)(ql_h[l + 32] & 0xF)   | (int)(((qh_h[l] >> 2) & 3) << 4);
            int q2 = (int)(ql_h[l] >> 4)          | (int)(((qh_h[l] >> 4) & 3) << 4);
            int q3 = (int)(ql_h[l + 32] >> 4)     | (int)(((qh_h[l] >> 6) & 3) << 4);
            int si = l / 16;
            o[half*128 + l]      = d_val * (float)sc_h[0 + si] * (float)(q0 - 32);
            o[half*128 + l + 32] = d_val * (float)sc_h[2 + si] * (float)(q1 - 32);
            o[half*128 + l + 64] = d_val * (float)sc_h[4 + si] * (float)(q2 - 32);
            o[half*128 + l + 96] = d_val * (float)sc_h[6 + si] * (float)(q3 - 32);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Dot product Kernel (returns scalar via device-to-host copy)
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_dot(float *result, const float *a, const float *b, int n) {
    __shared__ float smem[32];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += a[i] * b[i];
    sum = block_reduce_sum(sum, smem);
    if (threadIdx.x == 0) *result = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GEMM Kernel — F32 (simple tiled)
 * ════════════════════════════════════════════════════════════════════════ */

#define TILE_SIZE 16

__global__ void kernel_gemm(
    float       *C,
    const float *A,
    const float *B,
    int          M,
    int          N,
    int          K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ?
            A[(int64_t)row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ?
            B[(int64_t)b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[(int64_t)row * N + col] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * C API Wrappers (extern "C")
 * ════════════════════════════════════════════════════════════════════════ */

/* Type IDs from GGML */
#define TYPE_F32     0
#define TYPE_F16     1
#define TYPE_Q4_0    2
#define TYPE_Q4_1    3
#define TYPE_Q8_0    8
#define TYPE_Q4_K    12
#define TYPE_Q6_K    14
#define TYPE_IQ2_XS  17
#define TYPE_BF16    30

/* ═══════════════════════════════════════════════════════════════════════
 * IQ2_XS GEMV — native 2-bit imatrix quantization, 256 elements/74 bytes
 *
 * Block layout: fp16 d (2B) + uint16 qs[32] (64B) + uint8 scales[8] (8B)
 * Each qs[i]: bits[8:0]=grid index into kgrid[512], bits[15:9]=7 sign bits
 * kgrid[gi] encodes 8 pairs of 2-bit values → {1,3,5,7} via iq2xs_vals
 * Dequant: val = d * (0.5 + nibble_scale) * 0.25 * codebook_val * sign
 * ════════════════════════════════════════════════════════════════════════ */

#pragma pack(push, 1)
struct iq2_xs_block {
    uint16_t d;        /* FP16 super-block scale */
    uint16_t qs[32];   /* 32 codeword entries */
    uint8_t  scales[8];/* group scales: low nibble = first 16, high nibble = last 16 */
};
#pragma pack(pop)

__constant__ uint16_t iq2xs_kgrid[512] = {
       0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
      73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
     260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
     352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
     640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
    1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
    1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
    2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
    2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
    4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
    4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
    5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
    5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
    8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
    8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
   10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
   16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
   16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
   16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
   17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
   18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
   20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
   21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
   22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
   24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
   32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
   33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
   33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
   35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
   37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
   40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
   42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
};

__constant__ uint8_t iq2xs_vals[4] = {1, 3, 5, 7};

/* IQ2_XS GEMV: warp-per-row, 8 rows per block (256 threads = 8 warps).
 * Each lane handles ceil(n_sb / 32) super-blocks; dequants on-the-fly from constant cache. */
__global__ void kernel_gemv_iq2_xs(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * 8 + warp_id;
    if (row >= out_dim) return;

    int nb = in_dim / 256;  /* super-blocks per row */
    const struct iq2_xs_block *blocks =
        (const struct iq2_xs_block *)W + (int64_t)row * nb;

    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float d = fp16_to_f32(blocks[b].d);
        const float *xb = x + (int64_t)b * 256;
        int xoff = 0;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            float dl0 = d * (0.5f + (float)(blocks[b].scales[ib32] & 0xF)) * 0.25f;
            float dl1 = d * (0.5f + (float)(blocks[b].scales[ib32] >> 4))  * 0.25f;
            for (int l = 0; l < 4; l++) {
                float dl = (l < 2) ? dl0 : dl1;
                uint16_t qv = blocks[b].qs[ib32 * 4 + l];
                uint16_t gi = qv & 511u;
                uint8_t  s7 = (uint8_t)(qv >> 9);
                /* Extend 7 sign bits to 8 using even parity */
                uint8_t  s8 = s7 | (uint8_t)((__popc((unsigned)s7) & 1) << 7);
                uint16_t gv16 = iq2xs_kgrid[gi];
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    float gk = (float)iq2xs_vals[(gv16 >> (2 * k)) & 3u];
                    if (s8 & (uint8_t)(1u << k)) gk = -gk;
                    sum += dl * gk * xb[xoff + k];
                }
                xoff += 8;
            }
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GEMV for BF16 weights — BF16 is upper 16 bits of IEEE 754 float32.
 * conversion: uint32(bf16) << 16 → float32 bits.
 * Warp-per-row, 8 rows per block, vectorized uint loads.
 * ════════════════════════════════════════════════════════════════════════ */

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float result;
    asm volatile("mov.b32 %0, %1;" : "=f"(result) : "r"(bits));
    return result;
}

__global__ void kernel_gemv_bf16(
    float          *out,
    const uint16_t *W,
    const float    *x,
    int             out_dim,
    int             in_dim)
{
    extern __shared__ float x_tile_bf16[];

    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * 8 + warp_id;

    /* Cooperatively load input into shared memory */
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        x_tile_bf16[i] = x[i];
    __syncthreads();

    if (row >= out_dim) return;

    const uint16_t *row_w = W + (int64_t)row * in_dim;
    float sum = 0.0f;

    /* Vectorized: load 4 BF16 values at a time via uint64_t */
    int i4 = lane * 4;
    for (; i4 + 3 < in_dim; i4 += 128) {
        uint64_t packed = *(const uint64_t *)(row_w + i4);
        uint16_t w0 = (uint16_t)(packed);
        uint16_t w1 = (uint16_t)(packed >> 16);
        uint16_t w2 = (uint16_t)(packed >> 32);
        uint16_t w3 = (uint16_t)(packed >> 48);
        sum += bf16_to_f32(w0) * x_tile_bf16[i4];
        sum += bf16_to_f32(w1) * x_tile_bf16[i4 + 1];
        sum += bf16_to_f32(w2) * x_tile_bf16[i4 + 2];
        sum += bf16_to_f32(w3) * x_tile_bf16[i4 + 3];
    }
    /* Scalar tail: process elements in the partial 4-element chunk at i4.
     * The vectorized loop exits when i4+3 >= in_dim, so if i4 < in_dim there
     * are 1-3 elements left that weren't covered by the 4-wide load. */
    if (i4 < in_dim) {
        int end4 = i4 + 4 < in_dim ? i4 + 4 : in_dim;
        for (int i = i4; i < end4; i++)
            sum += bf16_to_f32(row_w[i]) * x_tile_bf16[i];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

/* Q6_K struct moved to top of file */

/* GEMV for Q6_K: each super-block = 256 elements, 210 bytes
 * Intra-superblock parallelism: all 32 lanes process elements WITHIN
 * each super-block (lanes iterate over super-blocks sequentially).
 * This gives 100% lane utilization even when n_sb < 32 (e.g. lmhead
 * with n_sb=6 was only 19% utilized with lane-per-superblock). */
__global__ void kernel_gemv_q4_k(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    /* One warp (32 threads) per output row, 8 rows per block.
     * Each lane processes lane-indexed elements within each 32-element
     * sub-block window across the 256-element super-block. */
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * 8 + warp_id;
    if (row >= out_dim) return;

    int n_sb = in_dim / 256;
    /* Q4_K block: 2(d) + 2(dmin) + 12(scales) + 128(qs) = 144 bytes */
    const uint8_t *row_base = (const uint8_t *)W + (int64_t)row * n_sb * 144;

    float sum = 0.0f;

    for (int sb = 0; sb < n_sb; sb++) {
        const uint8_t *b = row_base + sb * 144;
        uint16_t dh, dmh;
        /* Aligned read via memcpy to avoid misaligned access */
        dh  = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
        dmh = (uint16_t)b[2] | ((uint16_t)b[3] << 8);
        float d    = fp16_to_f32(dh);
        float dmin = fp16_to_f32(dmh);
        const uint8_t *scales = b + 4;   /* 12 bytes */
        const uint8_t *qs     = b + 16;  /* 128 bytes */
        const float   *xp     = x + sb * 256;

        int is = 0;
        for (int chunk = 0; chunk < 256; chunk += 64) {
            /* Extract scale/min for sub-blocks is+0 and is+1 */
            uint8_t sc0, m0, sc1, m1;
            if (is < 4) {
                sc0 = scales[is] & 63;      m0 = scales[is + 4] & 63;
            } else {
                sc0 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                m0  = (scales[is + 4] >> 4)  | ((scales[is + 0] >> 6) << 4);
            }
            int is1 = is + 1;
            if (is1 < 4) {
                sc1 = scales[is1] & 63;     m1 = scales[is1 + 4] & 63;
            } else {
                sc1 = (scales[is1 + 4] & 0xF) | ((scales[is1 - 4] >> 6) << 4);
                m1  = (scales[is1 + 4] >> 4)  | ((scales[is1 + 0] >> 6) << 4);
            }
            float d1 = d * (float)sc0;  float m1f = dmin * (float)m0;
            float d2 = d * (float)sc1;  float m2f = dmin * (float)m1;

            /* Nibble pair for lane l: qs[chunk/2 + l] covers positions chunk+l and chunk+32+l */
            uint8_t byte = qs[chunk / 2 + lane];
            float q_lo = (float)(byte & 0xF);
            float q_hi = (float)(byte >> 4);
            sum += (d1 * q_lo - m1f) * xp[chunk + lane];
            sum += (d2 * q_hi - m2f) * xp[chunk + 32 + lane];
            is += 2;
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

__global__ void kernel_gemv_q6_k(
    float       *out,
    const void  *W,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * 8 + warp_id;
    if (row >= out_dim) return;

    int n_sb = in_dim / 256;  /* super-blocks per row */
    const uint8_t *row_base = (const uint8_t *)W + (int64_t)row * n_sb * 210;

    float sum = 0.0f;

    for (int sb = 0; sb < n_sb; sb++) {
        const struct q6_k_block *b = (const struct q6_k_block *)(row_base + sb * 210);
        float d_val = fp16_to_f32(b->d);
        const float *xp = x + sb * 256;

        /* Each lane handles l=lane within each half of the super-block.
         * Produces 4 output values per half × 2 halves = 8 values per lane per sb. */
        for (int half = 0; half < 2; half++) {
            const uint8_t *ql_h = b->ql + half * 64;
            const uint8_t *qh_h = b->qh + half * 32;
            const int8_t *sc_h = b->scales + half * 8;

            int l = lane;
            int q0 = (int)(ql_h[l] & 0xF)       | (int)(((qh_h[l] >> 0) & 3) << 4);
            int q1 = (int)(ql_h[l + 32] & 0xF)   | (int)(((qh_h[l] >> 2) & 3) << 4);
            int q2 = (int)(ql_h[l] >> 4)          | (int)(((qh_h[l] >> 4) & 3) << 4);
            int q3 = (int)(ql_h[l + 32] >> 4)     | (int)(((qh_h[l] >> 6) & 3) << 4);

            int si = l / 16;
            float s0 = d_val * (float)sc_h[0 + si];
            float s1 = d_val * (float)sc_h[2 + si];
            float s2 = d_val * (float)sc_h[4 + si];
            float s3 = d_val * (float)sc_h[6 + si];

            sum += s0 * (float)(q0 - 32) * xp[half*128 + l];
            sum += s1 * (float)(q1 - 32) * xp[half*128 + l + 32];
            sum += s2 * (float)(q2 - 32) * xp[half*128 + l + 64];
            sum += s3 * (float)(q3 - 32) * xp[half*128 + l + 96];
        }
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) out[row] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Fused Per-Head Q/K RMSNorm + RoPE Kernel
 *
 * Eliminates per-layer CPU round-trip:
 *   GPU Q/K projection → GPU norm+RoPE → GPU attention
 *   (was: GPU proj → download → CPU norm+RoPE → upload → GPU attn)
 *
 * One block per head. Does:
 *   1. Per-head RMSNorm with learned weights
 *   2. RoPE rotation of each pair
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_fused_qk_norm_rope(
    float       *Q,              /* [n_heads * head_dim], in/out */
    float       *K,              /* [n_kv_heads * head_dim], in/out */
    const float *q_norm_w,       /* [n_heads * head_dim] or NULL */
    const float *k_norm_w,       /* [n_kv_heads * head_dim] or NULL */
    int          n_heads,
    int          n_kv_heads,
    int          head_dim,
    const int   *d_pos,
    float        rope_base,
    const float *rope_freqs,     /* precomputed [head_dim/2] or NULL */
    float        eps,
    int          rope_dim)       /* how many dims to rotate (0 = head_dim) */
{
    int head = blockIdx.x;
    int total_heads = n_heads + n_kv_heads;
    if (head >= total_heads) return;

    int is_k = (head >= n_heads);
    int actual_head = is_k ? (head - n_heads) : head;
    float *vec = is_k ? (K + actual_head * head_dim) : (Q + actual_head * head_dim);
    const float *norm_w = is_k ? k_norm_w : q_norm_w;
    int rdim = (rope_dim > 0) ? rope_dim : head_dim;

    __shared__ float smem[32];

    /* Step 1: RMSNorm if weights provided.
     * Norm weights are [head_dim] shared across all heads — index by i only. */
    if (norm_w) {
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
            sum_sq += vec[i] * vec[i];
        sum_sq = block_reduce_sum(sum_sq, smem);

        __shared__ float inv_rms;
        if (threadIdx.x == 0)
            inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
        __syncthreads();

        for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
            vec[i] = vec[i] * inv_rms * norm_w[i];
        __syncthreads();
    }

    /* Step 2: RoPE — rotate pairs (2i, 2i+1) */
    int pos = *d_pos;
    int half_rd = rdim / 2;
    for (int p = threadIdx.x; p < half_rd; p += blockDim.x) {
        float freq;
        if (rope_freqs)
            freq = rope_freqs[p];
        else
            freq = 1.0f / powf(rope_base, (float)(2 * p) / (float)head_dim);

        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float v0 = vec[2 * p];
        float v1 = vec[2 * p + 1];
        vec[2 * p]     = v0 * cos_a - v1 * sin_a;
        vec[2 * p + 1] = v0 * sin_a + v1 * cos_a;
    }
}

/* Fused per-head V magnitude normalization (no learned weights, just unit norm) */
__global__ void kernel_v_norm(
    float *V,            /* [n_kv_heads * head_dim], in/out */
    int    n_kv_heads,
    int    head_dim,
    float  eps)
{
    int kv = blockIdx.x;
    if (kv >= n_kv_heads) return;

    float *vh = V + kv * head_dim;
    __shared__ float smem[32];

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        sum_sq += vh[i] * vh[i];
    sum_sq = block_reduce_sum(sum_sq, smem);

    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        vh[i] *= inv_rms;
}



/* ═══════════════════════════════════════════════════════════════════════

 * Batched prefill kernels — process n tokens in parallel

 *

 * These replace the per-token loop in llm_forward_prefill_batch_gpu:

 *   kernel_batch_fused_qk_norm_rope  — RMSNorm + RoPE for all n tokens

 *   kernel_batch_v_norm              — V normalisation for all n tokens

 *   kernel_batch_kv_update           — write K,V cache for all n tokens

 *   kernel_prefill_attn_batched      — causal attention for all n tokens

 * ════════════════════════════════════════════════════════════════════════ */



/* Grid: (n_heads + n_kv_heads, n_tokens)

 * blockIdx.x = head index (0..n_heads-1 for Q, n_heads.. for K)

 * blockIdx.y = token index 0..n-1, position = start_pos + blockIdx.y  */

__global__ void kernel_batch_fused_qk_norm_rope(

    float       *Q,          /* [n, n_heads * head_dim] */

    float       *K,          /* [n, n_kv_heads * head_dim] or NULL */

    const float *q_norm_w,   /* [head_dim] or NULL */

    const float *k_norm_w,   /* [head_dim] or NULL */

    int          n_heads,

    int          n_kv_heads,

    int          head_dim,

    int          start_pos,

    float        rope_base,

    const float *rope_freqs, /* [head_dim/2] or NULL */

    float        eps,

    int          rope_dim)

{

    int head = blockIdx.x;

    int i    = blockIdx.y; /* token index */

    int total_heads = n_heads + n_kv_heads;

    if (head >= total_heads) return;



    int is_k = (head >= n_heads);

    int actual_head = is_k ? (head - n_heads) : head;



    float *vec;

    if (is_k) {

        if (!K) return;

        vec = K + (int64_t)i * (n_kv_heads * head_dim) + actual_head * head_dim;

    } else {

        vec = Q + (int64_t)i * (n_heads * head_dim) + actual_head * head_dim;

    }



    const float *norm_w = is_k ? k_norm_w : q_norm_w;

    int rdim = (rope_dim > 0) ? rope_dim : head_dim;

    int pos  = start_pos + i;



    __shared__ float smem[32];



    /* Step 1: RMSNorm if weights provided */

    if (norm_w) {

        float sum_sq = 0.0f;

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)

            sum_sq += vec[d] * vec[d];

        sum_sq = block_reduce_sum(sum_sq, smem);



        __shared__ float inv_rms;

        if (threadIdx.x == 0)

            inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);

        __syncthreads();



        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)

            vec[d] = vec[d] * inv_rms * norm_w[d];

        __syncthreads();

    }



    /* Step 2: RoPE — rotate pairs */

    int half_rd = rdim / 2;

    for (int p = threadIdx.x; p < half_rd; p += blockDim.x) {

        float freq = rope_freqs

            ? rope_freqs[p]

            : 1.0f / powf(rope_base, (float)(2 * p) / (float)head_dim);

        float angle = (float)pos * freq;

        float cos_a = cosf(angle);

        float sin_a = sinf(angle);

        float v0 = vec[2 * p];

        float v1 = vec[2 * p + 1];

        vec[2 * p]     = v0 * cos_a - v1 * sin_a;

        vec[2 * p + 1] = v0 * sin_a + v1 * cos_a;

    }

}



/* Grid: (n_kv_heads, n_tokens)  — per-head V normalisation */

__global__ void kernel_batch_v_norm(

    float *V,        /* [n, n_kv_heads * head_dim] */

    int    n_kv_heads,

    int    head_dim,

    float  eps)

{

    int kv = blockIdx.x;

    int i  = blockIdx.y;

    if (kv >= n_kv_heads) return;



    float *vh = V + (int64_t)i * (n_kv_heads * head_dim) + kv * head_dim;

    __shared__ float smem[32];



    float sum_sq = 0.0f;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)

        sum_sq += vh[d] * vh[d];

    sum_sq = block_reduce_sum(sum_sq, smem);



    __shared__ float inv_rms;

    if (threadIdx.x == 0)

        inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);

    __syncthreads();



    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)

        vh[d] *= inv_rms;

}



/* Grid: (ceil(n_kv_heads*head_dim / 256), n_tokens)

 * Writes K_new[i] and V_new[i] into cache at position start_pos + i  */

__global__ void kernel_batch_kv_update(

    float       *K_cache,  /* [n_kv_heads * max_seq * head_dim] */

    float       *V_cache,

    const float *K_new,    /* [n_tokens, n_kv_heads * head_dim] */

    const float *V_new,

    int          n_kv_heads,

    int          head_dim,

    int          start_pos,

    int          max_seq)

{

    int i   = blockIdx.y; /* token index */

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = n_kv_heads * head_dim;

    if (idx >= total) return;



    int kv  = idx / head_dim;

    int d   = idx % head_dim;

    int pos = start_pos + i;



    int64_t cache_off = ((int64_t)kv * max_seq + pos) * head_dim + d;

    int64_t new_off   = (int64_t)i * total + (int64_t)kv * head_dim + d;



    K_cache[cache_off] = K_new[new_off];

    V_cache[cache_off] = V_new[new_off];

}



/* Grid: (n_heads, n_tokens)

 * blockIdx.x = head, blockIdx.y = token index.

 * Shared: scores[smem_seq] + smem[32]

 * Token qi at position start_pos + qi attends causally to KV positions

 * 0 .. start_pos + qi (all prior tokens already in the KV cache).           */

__global__ void kernel_prefill_attn_batched(

    float       *O,         /* [n, n_heads * head_dim] */

    const float *Q,         /* [n, n_heads * head_dim] */

    const float *K_cache,   /* [n_kv_heads * max_seq * head_dim] */

    const float *V_cache,

    int          n_heads,

    int          n_kv_heads,

    int          head_dim,

    int          n,         /* number of new tokens */

    int          start_pos,

    int          max_seq,

    float        scale,

    float        softcap,

    int          smem_seq)

{

    int head = blockIdx.x;

    int i    = blockIdx.y; /* token index 0..n-1 */

    if (head >= n_heads || i >= n) return;



    int kv_head = head * n_kv_heads / n_heads; /* GQA mapping */

    int seq_len = start_pos + i + 1;           /* causal: attend 0..seq_pos */

    if (seq_len > smem_seq) seq_len = smem_seq;



    extern __shared__ float shared[];

    float *scores = shared;            /* [smem_seq] */

    float *smem   = shared + smem_seq; /* [32] reduction scratch */



    const float *q_h = Q + (int64_t)i * (n_heads * head_dim) + head * head_dim;

    float       *o_h = O + (int64_t)i * (n_heads * head_dim) + head * head_dim;



    /* Phase 1: QK^T */

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {

        const float *k_t = K_cache + ((int64_t)kv_head * max_seq + t) * head_dim;

        float dot = 0.0f;

        for (int d = 0; d < head_dim; d++)

            dot += q_h[d] * k_t[d];

        dot *= scale;

        if (softcap > 0.0f)

            dot = softcap * tanhf(dot / softcap);

        scores[t] = dot;

    }

    __syncthreads();



    /* Phase 2: online softmax */

    float max_val = NEG_INF;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x)

        max_val = fmaxf(max_val, scores[t]);

    max_val = block_reduce_max(max_val, smem);



    __shared__ float max_shared;

    if (threadIdx.x == 0) max_shared = max_val;

    __syncthreads();



    float sum_val = 0.0f;

    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {

        scores[t] = expf(scores[t] - max_shared);

        sum_val += scores[t];

    }

    sum_val = block_reduce_sum(sum_val, smem);



    __shared__ float inv_sum;

    if (threadIdx.x == 0) inv_sum = 1.0f / sum_val;

    __syncthreads();



    for (int t = threadIdx.x; t < seq_len; t += blockDim.x)

        scores[t] *= inv_sum;

    __syncthreads();



    /* Phase 3: weighted sum of V */

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {

        float val = 0.0f;

        for (int t = 0; t < seq_len; t++) {

            const float *v_t = V_cache + ((int64_t)kv_head * max_seq + t) * head_dim;

            val += scores[t] * v_t[d];

        }

        o_h[d] = val;

    }

}

/* ═══════════════════════════════════════════════════════════════════════
 * Fused kernels: add_rmsnorm, rmsnorm_add, gelu_mul, dual/triple GEMV
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_add_rmsnorm(
    float       *norm_out,
    float       *x_inout,
    const float *residual,
    const float *norm_w,
    int          dim,
    float        eps)
{
    __shared__ float smem[32];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_inout[i] + residual[i];
        x_inout[i] = v;
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq, smem);
    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        norm_out[i] = x_inout[i] * inv_rms * norm_w[i];
}

__global__ void kernel_rmsnorm_add(
    float       *x_inout,
    const float *data,
    const float *norm_w,
    int          dim,
    float        eps)
{
    __shared__ float smem[32];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = data[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq, smem);
    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)dim + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        x_inout[i] += data[i] * inv_rms * norm_w[i];
}

__global__ void kernel_gelu_mul(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        a[i] = gelu * b[i];
    }
}

__global__ void kernel_gemv_dual_q4_0(
    float       *out_a,
    float       *out_b,
    const void  *W_a,
    const void  *W_b,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    extern __shared__ char smem_raw[];
    int nb = in_dim / 32;
    int8_t *q8       = (int8_t *)smem_raw;
    float  *q8_sc    = (float *)(smem_raw + in_dim);
    float  *q8_sums  = q8_sc + nb;

    int n_warps = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * n_warps + warp_id;

    for (int b = warp_id; b < nb; b += n_warps) {
        float val = x[b * 32 + lane];
        float amax = fabsf(val);
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
        float inv_sc = (amax > 1e-10f) ? 127.0f / amax : 0.0f;
        int qi = __float2int_rn(val * inv_sc);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        q8[b * 32 + lane] = (int8_t)qi;
        int isum = qi;
        for (int off = 16; off > 0; off >>= 1)
            isum += __shfl_xor_sync(0xFFFFFFFF, isum, off);
        if (lane == 0) {
            float sc = amax / 127.0f;
            q8_sc[b]   = sc;
            q8_sums[b] = sc * (float)isum;
        }
    }
    __syncthreads();

    if (row >= out_dim) return;

    {
        const struct q4_0_block *blocks =
            (const struct q4_0_block *)W_a + (int64_t)row * nb;
        float sum = 0.0f;
        for (int b = lane; b < nb; b += 32) {
            float d = fp16_to_f32(blocks[b].d);
            uint32_t raw[4];
            memcpy(raw, blocks[b].qs, 16);
            uint32_t qlo[4], qhi[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                qlo[k] = raw[k] & 0x0F0F0F0Fu;
                qhi[k] = (raw[k] >> 4) & 0x0F0F0F0Fu;
            }
            const int32_t *q8_lo = (const int32_t *)(q8 + b * 32);
            const int32_t *q8_hi = (const int32_t *)(q8 + b * 32 + 16);
            int isum_all = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                isum_all = __dp4a((int)qlo[k], q8_lo[k], isum_all);
                isum_all = __dp4a((int)qhi[k], q8_hi[k], isum_all);
            }
            sum += d * (q8_sc[b] * (float)isum_all - 8.0f * q8_sums[b]);
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) out_a[row] = sum;
    }

    {
        const struct q4_0_block *blocks =
            (const struct q4_0_block *)W_b + (int64_t)row * nb;
        float sum = 0.0f;
        for (int b = lane; b < nb; b += 32) {
            float d = fp16_to_f32(blocks[b].d);
            uint32_t raw[4];
            memcpy(raw, blocks[b].qs, 16);
            uint32_t qlo[4], qhi[4];
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                qlo[k] = raw[k] & 0x0F0F0F0Fu;
                qhi[k] = (raw[k] >> 4) & 0x0F0F0F0Fu;
            }
            const int32_t *q8_lo = (const int32_t *)(q8 + b * 32);
            const int32_t *q8_hi = (const int32_t *)(q8 + b * 32 + 16);
            int isum_all = 0;
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                isum_all = __dp4a((int)qlo[k], q8_lo[k], isum_all);
                isum_all = __dp4a((int)qhi[k], q8_hi[k], isum_all);
            }
            sum += d * (q8_sc[b] * (float)isum_all - 8.0f * q8_sums[b]);
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) out_b[row] = sum;
    }
}

/* Fused dual Q8_0 GEMV: computes two output vectors with shared x load/launch.
 * Useful for GP K/V projected decode where both matrices read the same x_sub. */
__global__ void kernel_gemv_dual_q8_0(
    float       *out_a,
    float       *out_b,
    const void  *W_a,
    const void  *W_b,
    const float *x,
    int          out_dim,
    int          in_dim)
{
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * 8 + warp_id;
    if (row >= out_dim) return;

    int n_blocks = in_dim / 32;

    const struct q8_0_block *ba =
        (const struct q8_0_block *)W_a + (int64_t)row * n_blocks;
    const struct q8_0_block *bb =
        (const struct q8_0_block *)W_b + (int64_t)row * n_blocks;

    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const float xv = x[b * 32 + lane];
        sum_a += fp16_to_f32(ba[b].d) * (float)ba[b].qs[lane] * xv;
        sum_b += fp16_to_f32(bb[b].d) * (float)bb[b].qs[lane] * xv;
    }

    sum_a = warp_reduce_sum(sum_a);
    sum_b = warp_reduce_sum(sum_b);
    if (lane == 0) {
        out_a[row] = sum_a;
        out_b[row] = sum_b;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Fused RMSNorm + Triple Q4_0 GEMV (GPU)
 *
 * Eliminates the separate ck_rmsnorm kernel launch that writes d_xn before
 * the triple GEMV. Instead, one kernel:
 *   Phase 0 — all threads cooperatively compute sum_sq(x) → inv_rms
 *   Phase 1 — quantize x * inv_rms * norm_w to Q8 (cooperative, shared mem)
 *   Phase 2 — each warp computes its assigned output row via dp4a
 *
 * Saves one d_xn write (dim floats) and its re-read per projection.
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_fused_rmsnorm_triple_q4_0(
    float       *out_q,
    float       *out_k,
    float       *out_v,
    const void  *W_q,
    const void  *W_k,
    const void  *W_v,
    const float *x,
    const float *norm_w,
    float        eps,
    int          q_dim,
    int          k_dim,
    int          v_dim,
    int          in_dim)
{
    extern __shared__ char smem_raw[];
    int nb = in_dim / 32;
    int8_t *q8      = (int8_t *)smem_raw;
    float  *q8_sc   = (float *)(smem_raw + in_dim);
    float  *q8_sums = q8_sc + nb;
    float  *s_inv   = q8_sums + nb;  /* 1-element: shared inv_rms */

    int n_warps = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * n_warps + warp_id;
    int total_rows = q_dim + k_dim + v_dim;

    /* Phase 0: cooperative sum_sq → inv_rms */
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x)
        local_sq += x[i] * x[i];
    for (int off = 16; off > 0; off >>= 1)
        local_sq += __shfl_xor_sync(0xFFFFFFFF, local_sq, off);
    __shared__ float warp_sq[8];
    if (lane == 0) warp_sq[warp_id] = local_sq;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_sq[w];
        s_inv[0] = rsqrtf(total / (float)in_dim + eps);
    }
    __syncthreads();
    const float inv_rms = s_inv[0];

    /* Phase 1: quantize xn = x * inv_rms * norm_w → Q8 (shared memory) */
    for (int b = warp_id; b < nb; b += n_warps) {
        float val = x[b * 32 + lane] * inv_rms * norm_w[b * 32 + lane];
        float amax = fabsf(val);
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
        float inv_sc = (amax > 1e-10f) ? 127.0f / amax : 0.0f;
        int qi = __float2int_rn(val * inv_sc);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        q8[b * 32 + lane] = (int8_t)qi;
        int isum = qi;
        for (int off = 16; off > 0; off >>= 1)
            isum += __shfl_xor_sync(0xFFFFFFFF, isum, off);
        if (lane == 0) {
            float sc = amax / 127.0f;
            q8_sc[b]   = sc;
            q8_sums[b] = sc * (float)isum;
        }
    }
    __syncthreads();

    if (row >= total_rows) return;

    /* Phase 2: dp4a GEMV for this row (Q, K, or V) */
    float *out;
    const void *W;
    int local_row;
    if (row < q_dim) {
        out = out_q; W = W_q; local_row = row;
    } else if (row < q_dim + k_dim) {
        out = out_k; W = W_k; local_row = row - q_dim;
    } else {
        out = out_v; W = W_v; local_row = row - q_dim - k_dim;
    }

    const struct q4_0_block *blocks =
        (const struct q4_0_block *)W + (int64_t)local_row * nb;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float d = fp16_to_f32(blocks[b].d);
        uint32_t raw[4];
        memcpy(raw, blocks[b].qs, 16);
        uint32_t qlo[4], qhi[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            qlo[k] = raw[k] & 0x0F0F0F0Fu;
            qhi[k] = (raw[k] >> 4) & 0x0F0F0F0Fu;
        }
        const int32_t *q8_lo = (const int32_t *)(q8 + b * 32);
        const int32_t *q8_hi = (const int32_t *)(q8 + b * 32 + 16);
        int isum_all = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            isum_all = __dp4a((int)qlo[k], q8_lo[k], isum_all);
            isum_all = __dp4a((int)qhi[k], q8_hi[k], isum_all);
        }
        sum += d * (q8_sc[b] * (float)isum_all - 8.0f * q8_sums[b]);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[local_row] = sum;
}

__global__ void kernel_gemv_triple_q4_0(
    float       *out_q,
    float       *out_k,
    float       *out_v,
    const void  *W_q,
    const void  *W_k,
    const void  *W_v,
    const float *x,
    int          q_dim,
    int          k_dim,
    int          v_dim,
    int          in_dim)
{
    extern __shared__ char smem_raw[];
    int nb = in_dim / 32;
    int8_t *q8       = (int8_t *)smem_raw;
    float  *q8_sc    = (float *)(smem_raw + in_dim);
    float  *q8_sums  = q8_sc + nb;

    int n_warps = blockDim.x >> 5;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int row     = blockIdx.x * n_warps + warp_id;
    int total_rows = q_dim + k_dim + v_dim;

    for (int b = warp_id; b < nb; b += n_warps) {
        float val = x[b * 32 + lane];
        float amax = fabsf(val);
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
        float inv_sc = (amax > 1e-10f) ? 127.0f / amax : 0.0f;
        int qi = __float2int_rn(val * inv_sc);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        q8[b * 32 + lane] = (int8_t)qi;
        int isum = qi;
        for (int off = 16; off > 0; off >>= 1)
            isum += __shfl_xor_sync(0xFFFFFFFF, isum, off);
        if (lane == 0) {
            float sc = amax / 127.0f;
            q8_sc[b]   = sc;
            q8_sums[b] = sc * (float)isum;
        }
    }
    __syncthreads();

    if (row >= total_rows) return;

    float *out;
    const void *W;
    int local_row;
    if (row < q_dim) {
        out = out_q; W = W_q; local_row = row;
    } else if (row < q_dim + k_dim) {
        out = out_k; W = W_k; local_row = row - q_dim;
    } else {
        out = out_v; W = W_v; local_row = row - q_dim - k_dim;
    }

    const struct q4_0_block *blocks =
        (const struct q4_0_block *)W + (int64_t)local_row * nb;
    float sum = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float d = fp16_to_f32(blocks[b].d);
        uint32_t raw[4];
        memcpy(raw, blocks[b].qs, 16);
        uint32_t qlo[4], qhi[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            qlo[k] = raw[k] & 0x0F0F0F0Fu;
            qhi[k] = (raw[k] >> 4) & 0x0F0F0F0Fu;
        }
        const int32_t *q8_lo = (const int32_t *)(q8 + b * 32);
        const int32_t *q8_hi = (const int32_t *)(q8 + b * 32 + 16);
        int isum_all = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            isum_all = __dp4a((int)qlo[k], q8_lo[k], isum_all);
            isum_all = __dp4a((int)qhi[k], q8_hi[k], isum_all);
        }
        sum += d * (q8_sc[b] * (float)isum_all - 8.0f * q8_sums[b]);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) out[local_row] = sum;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CUDA Stream Management — Overlap compute with memory transfers
 * ════════════════════════════════════════════════════════════════════════ */

static cudaStream_t stream_compute = 0;
static cudaStream_t stream_transfer = 0;
static cublasHandle_t g_cublas = NULL;

/* ═══════════════════════════════════════════════════════════════════════
 * CUDA Graph support — capture/replay to eliminate launch overhead
 * ════════════════════════════════════════════════════════════════════════ */
static cudaGraphExec_t g_graph_exec = NULL;
static int  g_capturing_graph = 0;

/* Scratch buffers for pre-quantized Q8 input in Q4_0 GEMV path. */
static int8_t *d_q8_x_scratch = NULL;
static float  *d_q8_sc_scratch = NULL;
static float  *d_q8_sums_scratch = NULL;
static int     g_q8_scratch_in_dim = 0;

/* ═══════════════════════════════════════════════════════════════════════
 * Batch Prefill Scratch — shared Q8 scratch for all batch-GEMV calls
 * Layout: [max_batch × max_in_dim] for q8; [max_batch × max_nb] for scales/sums
 * ════════════════════════════════════════════════════════════════════════ */
static int8_t *d_bq8  = NULL;
static float  *d_bsc  = NULL;
static float  *d_bsu  = NULL;
static int     g_bq8_batch = 0;
static int     g_bq8_dim   = 0;

static int ensure_batch_q8(int batch, int in_dim) {
    int nb = in_dim / 32;
    if (nb <= 0 || batch <= 0) return -1;
    if (d_bq8 && batch <= g_bq8_batch && in_dim <= g_bq8_dim) return 0;
    if (d_bq8) { cudaFree(d_bq8); d_bq8 = NULL; }
    if (d_bsc) { cudaFree(d_bsc); d_bsc = NULL; }
    if (d_bsu) { cudaFree(d_bsu); d_bsu = NULL; }
    int alloc_b = batch > g_bq8_batch ? batch : g_bq8_batch + 1;
    int alloc_d = in_dim > g_bq8_dim   ? in_dim : g_bq8_dim + 32;
    int alloc_nb = alloc_d / 32;
    if (cudaMalloc(&d_bq8, (int64_t)alloc_b * alloc_d) != cudaSuccess) {
        g_bq8_batch = 0; g_bq8_dim = 0; return -1;
    }
    if (cudaMalloc(&d_bsc, (int64_t)alloc_b * alloc_nb * sizeof(float)) != cudaSuccess) {
        cudaFree(d_bq8); d_bq8 = NULL; g_bq8_batch = 0; g_bq8_dim = 0; return -1;
    }
    if (cudaMalloc(&d_bsu, (int64_t)alloc_b * alloc_nb * sizeof(float)) != cudaSuccess) {
        cudaFree(d_bq8); d_bq8 = NULL;
        cudaFree(d_bsc); d_bsc = NULL;
        g_bq8_batch = 0; g_bq8_dim = 0; return -1;
    }
    g_bq8_batch = alloc_b;
    g_bq8_dim   = alloc_d;
    return 0;
}

/* ─── Batch Q8 quantization: one block per batch item, 256 threads ─── */
__global__ void kernel_batch_quant_q8(
    const float *X,        /* [batch × in_dim] */
    int8_t      *q8,       /* [batch × in_dim] */
    float       *sc,       /* [batch × nb]     */
    float       *su,       /* [batch × nb]     */
    int          in_dim)
{
    int b    = blockIdx.x;
    int nb   = in_dim / 32;
    const float *x = X  + (int64_t)b * in_dim;
    int8_t      *q = q8 + (int64_t)b * in_dim;
    float       *s = sc + (int64_t)b * nb;
    float       *u = su + (int64_t)b * nb;
    int n_warps = blockDim.x >> 5;
    int wid  = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    for (int bl = wid; bl < nb; bl += n_warps) {
        float val = x[bl * 32 + lane];
        float amax = fabsf(val);
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
        float inv = (amax > 1e-10f) ? (127.0f / amax) : 0.0f;
        int qi = __float2int_rn(val * inv);
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        q[bl * 32 + lane] = (int8_t)qi;
        int isum = qi;
        for (int off = 16; off > 0; off >>= 1)
            isum += __shfl_xor_sync(0xFFFFFFFF, isum, off);
        if (lane == 0) { s[bl] = amax / 127.0f; u[bl] = s[bl] * (float)isum; }
    }
}

/* ─── Batched Q4_0 GEMV using pre-quantized scratch ─── */
/* Grid: (ceil(out_dim/8), batch); Block: 256 threads = 8 warps              */
/* C[b, row] = dequant(W[row]) · X_q8[b]                                     */
#define BGEMV_WARPS 8
__global__ void kernel_batch_gemv_q4_prequant(
    float        *C,
    const void   *W,
    const int8_t *q8,   /* [batch × in_dim] */
    const float  *sc,   /* [batch × nb]     */
    const float  *su,   /* [batch × nb]     */
    int           out_dim,
    int           in_dim,
    int           batch)
{
    int nb   = in_dim / 32;
    int b    = blockIdx.y;
    int row  = blockIdx.x * BGEMV_WARPS + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (b >= batch || row >= out_dim) return;
    const int8_t *q = q8 + (int64_t)b * in_dim;
    const float  *s = sc + (int64_t)b * nb;
    const float  *u = su + (int64_t)b * nb;
    const struct q4_0_block *blks =
        (const struct q4_0_block *)W + (int64_t)row * nb;
    float sum = 0.0f;
    for (int bl = lane; bl < nb; bl += 32) {
        float d = fp16_to_f32(blks[bl].d);
        uint32_t raw[4];
        memcpy(raw, blks[bl].qs, 16);
        uint32_t qlo[4], qhi[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            qlo[k] = raw[k] & 0x0F0F0F0Fu;
            qhi[k] = (raw[k] >> 4) & 0x0F0F0F0Fu;
        }
        const int32_t *ql = (const int32_t *)(q + bl * 32);
        const int32_t *qh = (const int32_t *)(q + bl * 32 + 16);
        int isum = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            isum = __dp4a((int)qlo[k], ql[k], isum);
            isum = __dp4a((int)qhi[k], qh[k], isum);
        }
        sum += d * (s[bl] * (float)isum - 8.0f * u[bl]);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) C[(int64_t)b * out_dim + row] = sum;
}

/* ─── Out-of-place batched RMSNorm ─── */
__global__ void kernel_batched_rmsnorm_out(
    float       *out,      /* [n_slices × slice_dim] */
    const float *in,
    const float *w,
    int          n_slices,
    int          slice_dim,
    float        eps)
{
    int s = blockIdx.x;
    if (s >= n_slices) return;
    const float *xi = in  + (int64_t)s * slice_dim;
    float       *xo = out + (int64_t)s * slice_dim;
    __shared__ float smem[32];
    __shared__ float inv_rms;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < slice_dim; i += blockDim.x)
        ss += xi[i] * xi[i];
    ss = block_reduce_sum(ss, smem);
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(ss / (float)slice_dim + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < slice_dim; i += blockDim.x)
        xo[i] = xi[i] * inv_rms * w[i];
}

static int ensure_q8_scratch(int in_dim) {
    int nb = in_dim / 32;
    if (in_dim <= 0 || nb <= 0) return -1;
    if (d_q8_x_scratch && d_q8_sc_scratch && d_q8_sums_scratch &&
        g_q8_scratch_in_dim >= in_dim) {
        return 0;
    }

    if (d_q8_x_scratch)   { cudaFree(d_q8_x_scratch);   d_q8_x_scratch = NULL; }
    if (d_q8_sc_scratch)  { cudaFree(d_q8_sc_scratch);  d_q8_sc_scratch = NULL; }
    if (d_q8_sums_scratch){ cudaFree(d_q8_sums_scratch);d_q8_sums_scratch = NULL; }

    if (cudaMalloc(&d_q8_x_scratch, (size_t)in_dim * sizeof(int8_t)) != cudaSuccess)
        return -1;
    if (cudaMalloc(&d_q8_sc_scratch, (size_t)nb * sizeof(float)) != cudaSuccess)
        return -1;
    if (cudaMalloc(&d_q8_sums_scratch, (size_t)nb * sizeof(float)) != cudaSuccess)
        return -1;

    g_q8_scratch_in_dim = in_dim;
    return 0;
}

/* Device-side position vars for capture-once graphs.
 * Kernels read pos/seq_len from device memory so the graph topology
 * stays constant — only the pointed-to values change between replays. */
static int   *d_pos_var     = NULL;
static int   *d_seq_len_var = NULL;
static int   *d_argmax_result = NULL;
static int   *h_argmax_result = NULL;
static float *d_dot_result  = NULL;  /* pre-allocated: avoids per-call cudaMalloc in ck_dot */

extern "C" {

/* ─── Device management ─── */

CUDA_API int ck_init(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) return -1;
    cudaSetDevice(0);

    /* Create streams for compute/transfer overlap */
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_transfer);

    /* Initialize cuBLAS handle bound to the compute stream */
    if (cublasCreate(&g_cublas) == CUBLAS_STATUS_SUCCESS)
        cublasSetStream(g_cublas, stream_compute);

    /* Allocate device-side position vars for CUDA graph capture */
    if (cudaMalloc(&d_pos_var, sizeof(int)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_seq_len_var, sizeof(int)) != cudaSuccess) {
        cudaFree(d_pos_var); d_pos_var = NULL; return -1;
    }
    int zero = 0;
    cudaMemcpy(d_pos_var, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq_len_var, &zero, sizeof(int), cudaMemcpyHostToDevice);

    /* Pre-allocate dot product result buffer (avoids per-call cudaMalloc). */
    if (cudaMalloc(&d_dot_result, sizeof(float)) != cudaSuccess) {
        cudaFree(d_pos_var); d_pos_var = NULL;
        cudaFree(d_seq_len_var); d_seq_len_var = NULL; return -1;
    }

    /* Pinned host buffer for low-latency argmax result readback. */
    if (cudaMallocHost(&h_argmax_result, sizeof(int)) != cudaSuccess) {
        cudaFree(d_pos_var); d_pos_var = NULL;
        cudaFree(d_seq_len_var); d_seq_len_var = NULL;
        cudaFree(d_dot_result); d_dot_result = NULL; return -1;
    }

    return 0;
}

CUDA_API void ck_shutdown(void) {
    if (g_graph_exec) { cudaGraphExecDestroy(g_graph_exec); g_graph_exec = NULL; }
    if (d_q8_x_scratch)    { cudaFree(d_q8_x_scratch);    d_q8_x_scratch = NULL; }
    if (d_q8_sc_scratch)   { cudaFree(d_q8_sc_scratch);   d_q8_sc_scratch = NULL; }
    if (d_q8_sums_scratch) { cudaFree(d_q8_sums_scratch); d_q8_sums_scratch = NULL; }
    g_q8_scratch_in_dim = 0;
    if (d_bq8) { cudaFree(d_bq8); d_bq8 = NULL; }
    if (d_bsc) { cudaFree(d_bsc); d_bsc = NULL; }
    if (d_bsu) { cudaFree(d_bsu); d_bsu = NULL; }
    g_bq8_batch = 0; g_bq8_dim = 0;
    if (d_dot_result)    { cudaFree(d_dot_result);    d_dot_result    = NULL; }
    if (h_argmax_result) { cudaFreeHost(h_argmax_result); h_argmax_result = NULL; }
    if (d_pos_var)       { cudaFree(d_pos_var);       d_pos_var       = NULL; }
    if (d_seq_len_var)   { cudaFree(d_seq_len_var);   d_seq_len_var   = NULL; }
    if (d_argmax_result) { cudaFree(d_argmax_result); d_argmax_result = NULL; }
    if (stream_compute)  { cudaStreamDestroy(stream_compute);  stream_compute  = 0; }
    if (stream_transfer) { cudaStreamDestroy(stream_transfer); stream_transfer = 0; }
    if (g_cublas)        { cublasDestroy(g_cublas);            g_cublas        = NULL; }
    /* Do NOT call cudaDeviceReset() — it destroys the entire CUDA context for
     * the process, breaking any other library or static destructor that still
     * holds GPU resources.  Explicit free of every allocation above is sufficient. */
}

CUDA_API int ck_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

CUDA_API uint64_t ck_free_memory(int dev) {
    size_t free_mem = 0, total = 0;
    cudaSetDevice(dev);
    cudaMemGetInfo(&free_mem, &total);
    return (uint64_t)free_mem;
}

/* ─── Memory ops ─── */

CUDA_API void *ck_alloc(uint64_t size) {
    void *ptr = NULL;
    if (cudaMalloc(&ptr, size) != cudaSuccess) return NULL;
    return ptr;
}

CUDA_API void ck_free(void *ptr) {
    if (ptr) cudaFree(ptr);
}

CUDA_API int ck_upload(void *dst, const void *src, uint64_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : -1;
}

CUDA_API int ck_download(void *dst, const void *src, uint64_t size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -1;
}

CUDA_API void ck_sync(void) {
    if (g_capturing_graph) return;  /* Cannot sync during graph capture */
    cudaStreamSynchronize(stream_compute);
    cudaStreamSynchronize(stream_transfer);
}

/* ─── Compute wrappers ─── */

CUDA_API void ck_gemv(float *out, const void *W, const float *x,
                      int out_dim, int in_dim, int type_id) {
    /* Adaptive thread count: match work per block to avoid idle threads.
     * For quantized types, work items = number of quant blocks per row.
     * Round up to nearest warp (32) for efficient execution. */
    int work_items;
    switch (type_id) {
        case TYPE_Q4_0: work_items = in_dim / 32; break;
        case TYPE_Q4_1: work_items = in_dim / 32; break;
        case TYPE_Q8_0: work_items = in_dim / 32; break;
        case TYPE_Q6_K: work_items = in_dim / 256; break;
        case TYPE_F16:  work_items = in_dim; break;
        default:        work_items = in_dim; break;
    }
    int threads = ((work_items + 31) / 32) * 32;  /* round up to warp */
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;

    switch (type_id) {
        case TYPE_Q4_0: {
            int nb = in_dim / 32;
            if (ensure_q8_scratch(in_dim) == 0) {
                int q_threads = 256;
                int warps_per_block = q_threads >> 5;
                int q_blocks = (nb + warps_per_block - 1) / warps_per_block;
                kernel_quantize_x_q8<<<q_blocks, q_threads, 0, stream_compute>>>(
                    x, d_q8_x_scratch, d_q8_sc_scratch, d_q8_sums_scratch, nb);
                /* Use 512 threads (16 rows/block) for large out_dim to halve block count */
                int gemv_t = (out_dim > 32768) ? 512 : 256;
                int gemv_rpb = gemv_t >> 5;
                kernel_gemv_q4_0_prequant<<<(out_dim + gemv_rpb - 1) / gemv_rpb, gemv_t, 0, stream_compute>>>(
                    out, W, d_q8_x_scratch, d_q8_sc_scratch, d_q8_sums_scratch,
                    out_dim, in_dim);
            } else {
                int q4_smem = in_dim + nb * 8;
                kernel_gemv_q4_0<<<(out_dim + 7) / 8, 256, q4_smem, stream_compute>>>(out, W, x, out_dim, in_dim);
            }
            break;
        }
        case TYPE_Q4_1:
            kernel_gemv_q4_1<<<out_dim, threads, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            break;
        case TYPE_Q8_0:
            /* Prefer warp-element kernels when in_dim is block-aligned (32).
             * For large in_dim/out_dim, use 8-row batching to cut launch overhead. */
            if ((in_dim & 31) == 0) {
                if (in_dim >= 1024 && out_dim >= 256)
                    kernel_gemv_q8_0_batched<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
                else
                    kernel_gemv_q8_0_elem<<<out_dim, 32, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            } else {
                kernel_gemv_q8_0<<<out_dim, threads, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            }
            break;
        case TYPE_F32:
            /* Batched: 8 rows per block (mirrors F16), reduces block count 8× */
            kernel_gemv_f32_batched<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, (const float *)W, x, out_dim, in_dim);
            break;
        case TYPE_F16:
            kernel_gemv_f16<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, (const __half *)W, x, out_dim, in_dim);
            break;
        case TYPE_Q4_K:
            kernel_gemv_q4_k<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            break;
        case TYPE_Q6_K:
            kernel_gemv_q6_k<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            break;
        case TYPE_BF16: {
            int smem = in_dim * sizeof(float);
            kernel_gemv_bf16<<<(out_dim + 7) / 8, 256, smem, stream_compute>>>(out, (const uint16_t *)W, x, out_dim, in_dim);
            break;
        }
        case TYPE_IQ2_XS:
            kernel_gemv_iq2_xs<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            break;
        default:
            break;
    }
}

CUDA_API void ck_gemm(float *C, const float *A, const float *B,
                      int M, int N, int K) {
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);
    kernel_gemm<<<blocks, threads, 0, stream_compute>>>(C, A, B, M, N, K);
}

CUDA_API void ck_rmsnorm(float *out, const float *x, const float *w,
                         int dim, float eps) {
    int threads = (dim < 1024) ? dim : 1024;
    kernel_rmsnorm<<<1, threads, 0, stream_compute>>>(out, x, w, dim, eps);
}

CUDA_API void ck_layernorm(float *out, const float *x, const float *w,
                           const float *bias, int dim, float eps) {
    int threads = (dim < 1024) ? dim : 1024;
    kernel_layernorm<<<1, threads, 0, stream_compute>>>(out, x, w, bias, dim, eps);
}

CUDA_API void ck_rope(float *q, float *k, int head_dim, int n_heads,
                      int n_kv_heads, int pos, float base, const float *freqs) {
    /* Update device pos var (skip during graph capture — already set) */
    if (!g_capturing_graph)
        cudaMemcpyAsync(d_pos_var, &pos, sizeof(int), cudaMemcpyHostToDevice, stream_compute);
    int total = n_heads * (head_dim / 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_rope<<<blocks, threads, 0, stream_compute>>>(q, k, head_dim, n_heads, n_kv_heads,
                                     d_pos_var, base, freqs);
}

CUDA_API void ck_softmax(float *x, int n) {
    int threads = (n < 1024) ? n : 1024;
    kernel_softmax<<<1, threads, 0, stream_compute>>>(x, n);
}

CUDA_API void ck_silu(float *x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_silu<<<blocks, threads, 0, stream_compute>>>(x, n);
}

CUDA_API void ck_gelu(float *x, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_gelu<<<blocks, threads, 0, stream_compute>>>(x, n);
}

CUDA_API void ck_fused_geglu(float *gate, const float *up, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_fused_geglu<<<blocks, threads, 0, stream_compute>>>(gate, up, n);
}

CUDA_API void ck_fused_swiglu(float *gate, const float *up, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_fused_swiglu<<<blocks, threads, 0, stream_compute>>>(gate, up, n);
}

CUDA_API void ck_batched_rmsnorm(float *data, const float *w,
                                  int n_slices, int slice_dim, float eps) {
    int threads = (slice_dim < 256) ? ((slice_dim + 31) / 32) * 32 : 256;
    if (threads < 32) threads = 32;
    kernel_batched_rmsnorm<<<n_slices, threads, 0, stream_compute>>>(data, w, n_slices, slice_dim, eps);
}

CUDA_API void ck_iswa_combine(float *out, const float *tok_embd,
                               const float *proj, float scale, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_iswa_combine<<<blocks, threads, 0, stream_compute>>>(out, tok_embd, proj, scale, n);
}

CUDA_API void ck_mul(float *out, const float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_mul<<<blocks, threads, 0, stream_compute>>>(out, a, b, n);
}

CUDA_API void ck_add(float *out, const float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_add<<<blocks, threads, 0, stream_compute>>>(out, a, b, n);
}

CUDA_API void ck_scale(float *out, const float *x, float s, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_scale<<<blocks, threads, 0, stream_compute>>>(out, x, s, n);
}

CUDA_API float ck_dot(const float *a, const float *b, int n) {
    /* Use pre-allocated device result buffer — avoids cudaMalloc overhead per call. */
    if (!d_dot_result)
        cudaMalloc(&d_dot_result, sizeof(float));  /* fallback if init was skipped */
    int threads = (n < 1024) ? n : 1024;
    kernel_dot<<<1, threads, 0, stream_compute>>>(d_dot_result, a, b, n);
    /* Async copy on stream_compute ensures ordering after the kernel. */
    float result = 0.0f;
    cudaMemcpyAsync(&result, d_dot_result, sizeof(float), cudaMemcpyDeviceToHost, stream_compute);
    cudaStreamSynchronize(stream_compute);
    return result;
}

CUDA_API void ck_dequantize(float *out, const void *data, int n_elements,
                            int type_id) {
    int n_blocks = n_elements / 32;
    int threads = 256;
    int grid = (n_blocks + threads - 1) / threads;
    switch (type_id) {
        case TYPE_Q4_0:
            kernel_dequantize_q4_0<<<grid, threads, 0, stream_compute>>>(out, data, n_elements);
            break;
        case TYPE_Q8_0:
            kernel_dequantize_q8_0<<<grid, threads, 0, stream_compute>>>(out, data, n_elements);
            break;
        case TYPE_F32:
            cudaMemcpyAsync(out, data, n_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream_compute);
            break;
        default:
            break;
    }
}

CUDA_API void ck_attention(float *out, const float *Q, const float *K,
                           const float *V, int n_heads, int n_kv_heads,
                           int head_dim, int seq_len, int max_seq,
                           float scale, float softcap) {
    /* Update device seq_len var (skip during graph capture — already set) */
    if (!g_capturing_graph)
        cudaMemcpyAsync(d_seq_len_var, &seq_len, sizeof(int), cudaMemcpyHostToDevice, stream_compute);
    int threads = 256;
    /* Shared memory: scores[max_seq] + smem[32]; use max_seq for graph compat */
    int smem_seq = max_seq;
    int shared_bytes = (smem_seq + 32) * sizeof(float);
    kernel_attention<<<n_heads, threads, shared_bytes, stream_compute>>>(
        out, Q, K, V, n_heads, n_kv_heads, head_dim, d_seq_len_var, max_seq,
        scale, softcap, smem_seq);
}

CUDA_API void ck_kv_update(float *K_cache, float *V_cache,
                           const float *K_new, const float *V_new,
                           int n_kv_heads, int head_dim, int pos,
                           int max_seq, int layer) {
    (void)layer;
    /* Update device pos var (skip during graph capture — already set) */
    if (!g_capturing_graph)
        cudaMemcpyAsync(d_pos_var, &pos, sizeof(int), cudaMemcpyHostToDevice, stream_compute);
    int total = n_kv_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_kv_update<<<blocks, threads, 0, stream_compute>>>(K_cache, V_cache, K_new, V_new,
                                          n_kv_heads, head_dim, d_pos_var, max_seq);
}

CUDA_API void ck_embed_lookup(float *out, const void *table, int token_id,
                              int dim, int type_id) {
    int threads = 256;
    switch (type_id) {
        case TYPE_Q4_0: {
            int n_blocks = dim / 32;
            int grid = (n_blocks + threads - 1) / threads;
            kernel_embed_q4_0<<<grid, threads, 0, stream_compute>>>(out, table, token_id, dim);
            break;
        }
        case TYPE_F32: {
            int grid = (dim + threads - 1) / threads;
            kernel_embed_f32<<<grid, threads, 0, stream_compute>>>(out, (const float *)table, token_id, dim);
            break;
        }
        case TYPE_Q6_K: {
            int n_sb = dim / 256;
            int grid = (n_sb + threads - 1) / threads;
            kernel_embed_q6_k<<<grid, threads, 0, stream_compute>>>(out, table, token_id, dim);
            break;
        }
        default:
            break;
    }
}

CUDA_API void ck_softcap(float *x, int n, float cap) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_softcap<<<blocks, threads, 0, stream_compute>>>(x, n, cap);
}

/* ─── Async memory ops (non-blocking via stream_transfer) ─── */

CUDA_API int ck_upload_async(void *dst, const void *src, uint64_t size) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream_transfer)
           == cudaSuccess ? 0 : -1;
}

CUDA_API int ck_download_async(void *dst, const void *src, uint64_t size) {
    return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream_transfer)
           == cudaSuccess ? 0 : -1;
}

CUDA_API void ck_stream_sync_transfer(void) {
    cudaStreamSynchronize(stream_transfer);
}

CUDA_API void ck_stream_sync_compute(void) {
    cudaStreamSynchronize(stream_compute);
}

/* ─── L2 Persistent Cache: pin Pt F16 in GPU L2 across all decode steps ─── *
 * Call once after uploading all Pt F16 matrices to GPU.  The RTX 4070 has a  *
 * 36 MB L2 cache; all 30 per-layer Pt F16 matrices (17.7 MB) fit easily.     *
 * After the first token, every Pt access hits L2 (~1800 GB/s) instead of    *
 * HBM (~340 GB/s), saving ~35 MB of HBM bandwidth per decode token.         */
CUDA_API void ck_l2_persist(const void *ptr, size_t bytes) {
    if (!ptr || bytes == 0 || !stream_compute) return;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return;
    /* NOTE: cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize) is intentionally
     * omitted.  Setting it would permanently reduce the L2 cache available for
     * all streaming kernels (baseline, softmax, attention, etc.) for the entire
     * process lifetime, even when the persisting window is no longer needed.
     * On RTX 4070 that shrinks usable L2 from 36 MB → 14 MB and measurably
     * hurts every path that doesn't benefit from persistence.
     *
     * Without the device-level reservation the access-policy window below is
     * best-effort only (CUDA may evict freely).  For CUDA-graph decode this is
     * already a no-op because graph nodes capture stream state at record time
     * and ignore access-policy changes made after cudaStreamBeginCapture.
     * The window is still applied for non-graph paths (prefill, first token). */
    size_t clamp = (bytes < prop.accessPolicyMaxWindowSize)
                   ? bytes : prop.accessPolicyMaxWindowSize;
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.base_ptr  = (void *)ptr;
    attr.accessPolicyWindow.num_bytes = clamp;
    attr.accessPolicyWindow.hitRatio  = 1.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream_compute, cudaStreamAttributeAccessPolicyWindow, &attr);
    printf("[CUDA] L2 persist hint: %.2f MB (device max window %.2f MB, no global reservation)\n",
           (double)clamp / (1 << 20),
           (double)prop.accessPolicyMaxWindowSize / (1 << 20));
}

/* ─── Fused QKV norm + RoPE ─── */

CUDA_API void ck_fused_qk_norm_rope(
    float *Q, float *K,
    const float *q_norm_w, const float *k_norm_w,
    int n_heads, int n_kv_heads, int head_dim,
    int pos, float rope_base, const float *rope_freqs,
    float eps, int rope_dim)
{
    /* Update device pos var (skip during graph capture — already set) */
    if (!g_capturing_graph)
        cudaMemcpyAsync(d_pos_var, &pos, sizeof(int), cudaMemcpyHostToDevice, stream_compute);
    /* One block per head (Q heads + K heads) */
    int total_heads = n_heads + n_kv_heads;
    int threads = (head_dim < 256) ? ((head_dim + 31) / 32) * 32 : 256;
    if (threads < 32) threads = 32;
    kernel_fused_qk_norm_rope<<<total_heads, threads, 0, stream_compute>>>(
        Q, K, q_norm_w, k_norm_w,
        n_heads, n_kv_heads, head_dim,
        d_pos_var, rope_base, rope_freqs, eps, rope_dim);
}

CUDA_API void ck_v_norm(float *V, int n_kv_heads, int head_dim, float eps) {
    int threads = (head_dim < 256) ? ((head_dim + 31) / 32) * 32 : 256;
    if (threads < 32) threads = 32;
    kernel_v_norm<<<n_kv_heads, threads, 0, stream_compute>>>(V, n_kv_heads, head_dim, eps);
}

/* ─── GEMV on compute stream (non-blocking launches) ─── */

CUDA_API void ck_gemv_async(float *out, const void *W, const float *x,
                            int out_dim, int in_dim, int type_id) {
    int work_items;
    switch (type_id) {
        case TYPE_Q4_0: work_items = in_dim / 32; break;
        case TYPE_Q4_1: work_items = in_dim / 32; break;
        case TYPE_Q8_0: work_items = in_dim / 32; break;
        case TYPE_Q4_K: work_items = in_dim / 256; break;
        case TYPE_Q6_K: work_items = in_dim / 256; break;
        case TYPE_F16:  work_items = in_dim; break;
        default:        work_items = in_dim; break;
    }
    int threads = ((work_items + 31) / 32) * 32;
    if (threads < 32) threads = 32;
    if (threads > 256) threads = 256;

    switch (type_id) {
        case TYPE_Q4_0: {
            int nb = in_dim / 32;
            if (ensure_q8_scratch(in_dim) == 0) {
                int q_threads = 256;
                int warps_per_block = q_threads >> 5;
                int q_blocks = (nb + warps_per_block - 1) / warps_per_block;
                kernel_quantize_x_q8<<<q_blocks, q_threads, 0, stream_compute>>>(
                    x, d_q8_x_scratch, d_q8_sc_scratch, d_q8_sums_scratch, nb);
                /* Use 512 threads (16 rows/block) for large out_dim to halve block count */
                int gemv_t = (out_dim > 32768) ? 512 : 256;
                int gemv_rpb = gemv_t >> 5;
                kernel_gemv_q4_0_prequant<<<(out_dim + gemv_rpb - 1) / gemv_rpb, gemv_t, 0, stream_compute>>>(
                    out, W, d_q8_x_scratch, d_q8_sc_scratch, d_q8_sums_scratch,
                    out_dim, in_dim);
            } else {
                int q4_smem = in_dim + nb * 8;
                kernel_gemv_q4_0<<<(out_dim + 7) / 8, 256, q4_smem, stream_compute>>>(out, W, x, out_dim, in_dim);
            }
            break;
        }
        case TYPE_Q4_1:
            kernel_gemv_q4_1<<<out_dim, threads, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            break;
        case TYPE_Q8_0:
            if ((in_dim & 31) == 0) {
                if (in_dim >= 1024 && out_dim >= 256)
                    kernel_gemv_q8_0_batched<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
                else
                    kernel_gemv_q8_0_elem<<<out_dim, 32, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            } else {
                kernel_gemv_q8_0<<<out_dim, threads, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            }
            break;
        case TYPE_F32:
            kernel_gemv_f32_batched<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, (const float *)W, x, out_dim, in_dim);
            break;
        case TYPE_F16:
            kernel_gemv_f16<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, (const __half *)W, x, out_dim, in_dim);
            break;
        case TYPE_Q4_K:
            kernel_gemv_q4_k<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            break;
        case TYPE_Q6_K:
            kernel_gemv_q6_k<<<(out_dim + 7) / 8, 256, 0, stream_compute>>>(out, W, x, out_dim, in_dim);
            break;
        case TYPE_BF16: {
            int smem = in_dim * sizeof(float);
            kernel_gemv_bf16<<<(out_dim + 7) / 8, 256, smem, stream_compute>>>(out, (const uint16_t *)W, x, out_dim, in_dim);
            break;
        }
        default: break;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Q4_0 → FP16 Dequantization Kernel
 *
 * Converts quantized Q4_0 weights to FP16 for more efficient GEMV access.
 * Each thread dequantizes one Q4_0 block (32 elements).
 * ════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_dequant_q4_0_to_f16(
    __half      *out,
    const void  *q4_data,
    int          n_blocks)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n_blocks) return;

    const struct q4_0_block *blk = (const struct q4_0_block *)q4_data + bid;
    float scale = fp16_to_f32(blk->d);
    __half *dst = out + bid * 32;

    for (int j = 0; j < 16; j++) {
        uint8_t qs = blk->qs[j];
        dst[j]      = __float2half(scale * ((int)(qs & 0xF) - 8));
        dst[j + 16] = __float2half(scale * ((int)(qs >> 4) - 8));
    }
}

CUDA_API void ck_dequant_q4_0_to_f16(void *out, const void *q4_data,
                                      int n_rows, int in_dim) {
    int n_blocks = n_rows * (in_dim / 32);
    int threads = 256;
    int blocks = (n_blocks + threads - 1) / threads;
    kernel_dequant_q4_0_to_f16<<<blocks, threads, 0, stream_compute>>>(
        (__half *)out, q4_data, n_blocks);
}

/* ─── Fused kernel wrappers ─── */

CUDA_API void ck_add_rmsnorm(float *norm_out, float *x_inout,
                              const float *residual, const float *norm_w,
                              int dim, float eps) {
    int threads = (dim < 256) ? ((dim + 31) / 32) * 32 : 256;
    if (threads < 32) threads = 32;
    kernel_add_rmsnorm<<<1, threads, 0, stream_compute>>>(norm_out, x_inout, residual, norm_w, dim, eps);
}

CUDA_API void ck_rmsnorm_add(float *x_inout, const float *data,
                              const float *norm_w, int dim, float eps) {
    int threads = (dim < 256) ? ((dim + 31) / 32) * 32 : 256;
    if (threads < 32) threads = 32;
    kernel_rmsnorm_add<<<1, threads, 0, stream_compute>>>(x_inout, data, norm_w, dim, eps);
}

CUDA_API void ck_gelu_mul(float *a, const float *b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_gelu_mul<<<blocks, threads, 0, stream_compute>>>(a, b, n);
}

CUDA_API void ck_gemv_dual_q4_0(
    float *out_a, float *out_b,
    const void *W_a, const void *W_b,
    const float *x,
    int out_dim, int in_dim)
{
    int nb = in_dim / 32;
    int smem = in_dim + nb * 8;
    int grid = (out_dim + 7) / 8;
    kernel_gemv_dual_q4_0<<<grid, 256, smem, stream_compute>>>(
        out_a, out_b, W_a, W_b, x, out_dim, in_dim);
}

CUDA_API void ck_gemv_dual_q8_0(
    float *out_a, float *out_b,
    const void *W_a, const void *W_b,
    const float *x,
    int out_dim, int in_dim)
{
    int grid = (out_dim + 7) / 8;
    kernel_gemv_dual_q8_0<<<grid, 256, 0, stream_compute>>>(
        out_a, out_b, W_a, W_b, x, out_dim, in_dim);
}

CUDA_API void ck_gemv_triple_q4_0(
    float *out_q, float *out_k, float *out_v,
    const void *W_q, const void *W_k, const void *W_v,
    const float *x,
    int q_dim, int k_dim, int v_dim,
    int in_dim)
{
    int nb = in_dim / 32;
    int smem = in_dim + nb * 8;
    int total_rows = q_dim + k_dim + v_dim;
    int grid = (total_rows + 7) / 8;
    kernel_gemv_triple_q4_0<<<grid, 256, smem, stream_compute>>>(
        out_q, out_k, out_v, W_q, W_k, W_v, x,
        q_dim, k_dim, v_dim, in_dim);
}

/* Fused RMSNorm + triple Q4_0 GEMV.
 * smem: q8[in_dim] + q8_sc[nb*4] + q8_sums[nb*4] + inv_rms[4] + warp_sq[32] */
CUDA_API void ck_fused_rmsnorm_triple_q4_0(
    float *out_q, float *out_k, float *out_v,
    const void *W_q, const void *W_k, const void *W_v,
    const float *x, const float *norm_w, float eps,
    int q_dim, int k_dim, int v_dim, int in_dim)
{
    int nb = in_dim / 32;
    /* smem: q8(in_dim) + q8_sc(nb*4) + q8_sums(nb*4) + s_inv(4) */
    int smem = in_dim + nb * 8 + 4;
    int total_rows = q_dim + k_dim + v_dim;
    int grid = (total_rows + 7) / 8;
    kernel_fused_rmsnorm_triple_q4_0<<<grid, 256, smem, stream_compute>>>(
        out_q, out_k, out_v, W_q, W_k, W_v, x, norm_w, eps,
        q_dim, k_dim, v_dim, in_dim);
}

/* ═══════════════════════════════════════════════════════════════════════
 * cuBLAS Batched GEMV — fused gate+up compressed-weight decode
 *
 * Computes:  y[i] = A[i] * x[i]  for i in [0, batch_count)
 *   A[i] : [M × K] row-major float matrix  (pointer from d_Aarray[i])
 *   x[i] : [K]     float vector             (pointer from d_xarray[i])
 *   y[i] : [M]     float vector (output)    (pointer from d_yarray[i])
 *
 * cuBLAS is column-major, so row-major A[M×K] is treated as col-major
 * A^T[K×M], and CUBLAS_OP_T transposes it back → effectively A*x.
 *
 * Pointer arrays (d_Aarray, d_xarray, d_yarray) must live in DEVICE memory
 * so they are compatible with CUDA graph capture.
 * ════════════════════════════════════════════════════════════════════════ */

CUDA_API void ck_sgemm_batched_f32(
    int M, int K,
    const float * const *d_Aarray,   /* [batch] device ptrs to [M×K] matrices */
    const float * const *d_xarray,   /* [batch] device ptrs to [K] vectors    */
    float * const       *d_yarray,   /* [batch] device ptrs to [M] outputs    */
    int batch_count)
{
    if (!g_cublas || batch_count <= 0 || M <= 0 || K <= 0) return;
    if (!d_Aarray || !d_xarray || !d_yarray) return;
    const float alpha = 1.0f, beta = 0.0f;
    /* Row-major A[M×K] = col-major (A^T)[K×M].
     * cublasSgemvBatched with CUBLAS_OP_T on (K×M) col-major → A*x. */
    cublasStatus_t st = cublasSgemvBatched(g_cublas, CUBLAS_OP_T,
        K, M,                              /* (rows, cols) of the col-major view */
        &alpha,
        d_Aarray, K,                       /* leading dim of col-major A^T = K  */
        d_xarray, 1,
        &beta,
        d_yarray, 1,
        batch_count);
    /* On failure the output vectors are undefined — callers should validate
     * results if status checking is enabled at a higher level. */
    (void)st;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GPU-side Argmax — returns token ID without downloading all logits
 * ════════════════════════════════════════════════════════════════════════ */

__global__ void kernel_argmax(int *result, const float *data, int n) {
    __shared__ float s_val[32];
    __shared__ int   s_idx[32];

    float best = NEG_INF;
    int best_i = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = data[i];
        if (v > best) { best = v; best_i = i; }
    }
    /* Warp reduction */
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xFFFFFFFF, best, off);
        int   oi = __shfl_down_sync(0xFFFFFFFF, best_i, off);
        if (ov > best) { best = ov; best_i = oi; }
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) { s_val[warp] = best; s_idx[warp] = best_i; }
    __syncthreads();

    int nw = (blockDim.x + 31) >> 5;
    if (threadIdx.x < (unsigned)nw) { best = s_val[threadIdx.x]; best_i = s_idx[threadIdx.x]; }
    else { best = NEG_INF; best_i = 0; }
    if (warp == 0) {
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xFFFFFFFF, best, off);
            int   oi = __shfl_down_sync(0xFFFFFFFF, best_i, off);
            if (ov > best) { best = ov; best_i = oi; }
        }
    }
    if (threadIdx.x == 0) *result = best_i;
}

CUDA_API int ck_argmax(const float *data, int n) {
    if (!d_argmax_result) cudaMalloc(&d_argmax_result, sizeof(int));
    if (!h_argmax_result) cudaMallocHost(&h_argmax_result, sizeof(int));
    kernel_argmax<<<1, 1024, 0, stream_compute>>>(d_argmax_result, data, n);
    cudaMemcpyAsync(h_argmax_result, d_argmax_result, sizeof(int),
                    cudaMemcpyDeviceToHost, stream_compute);
    cudaStreamSynchronize(stream_compute);
    return *h_argmax_result;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CUDA Graph Capture / Replay / Launch
 * ════════════════════════════════════════════════════════════════════════ */

CUDA_API void ck_graph_destroy(void) {
    if (g_graph_exec) { cudaGraphExecDestroy(g_graph_exec); g_graph_exec = NULL; }
}

/* Update device-side position variables (call BEFORE graph_launch) */
CUDA_API void ck_set_decode_pos(int pos, int seq_len) {
    cudaMemcpyAsync(d_pos_var,     &pos,     sizeof(int), cudaMemcpyHostToDevice, stream_compute);
    cudaMemcpyAsync(d_seq_len_var, &seq_len, sizeof(int), cudaMemcpyHostToDevice, stream_compute);
}

/* Drain any pending stream error before attempting graph capture.
 * cudaStreamBeginCapture fails with cudaErrorMisalignedAddress (716) if a
 * previous kernel left an error in the stream.  Synchronise + consume the
 * error with cudaGetLastError() to reset the stream to a clean state. */
CUDA_API void ck_stream_drain_error(void) {
    cudaStreamSynchronize(stream_compute);
    cudaGetLastError(); /* discard — stream is now clean */
}

CUDA_API int ck_graph_begin_capture(void) {
    g_capturing_graph = 1;
    /* Drain any pending stream error so begin_capture succeeds */
    cudaStreamSynchronize(stream_compute);
    cudaGetLastError();
    /* Capture on stream_compute (non-NULL stream required for graph capture) */
    cudaError_t err = cudaStreamBeginCapture(stream_compute, cudaStreamCaptureModeRelaxed);
    if (err != cudaSuccess) {
        g_capturing_graph = 0;
        printf("[CUDA Graph] begin_capture error: %s (%d)\n",
               cudaGetErrorString(err), (int)err);
        return -1;
    }
    return 0;
}

CUDA_API int ck_graph_end_capture(void) {
    g_capturing_graph = 0;
    cudaGraph_t new_graph = NULL;
    cudaError_t err = cudaStreamEndCapture(stream_compute, &new_graph);
    if (err != cudaSuccess || !new_graph) {
        printf("[CUDA Graph] end_capture error: %s (%d)\n",
               cudaGetErrorString(err), (int)err);
        return -1;
    }

    if (g_graph_exec) {
        cudaGraphExecUpdateResultInfo updateInfo;
        err = cudaGraphExecUpdate(g_graph_exec, new_graph, &updateInfo);
        if (err != cudaSuccess || updateInfo.result != cudaGraphExecUpdateSuccess) {
            cudaGraphExecDestroy(g_graph_exec);
            g_graph_exec = NULL;
            err = cudaGraphInstantiate(&g_graph_exec, new_graph, 0);
        }
    } else {
        err = cudaGraphInstantiate(&g_graph_exec, new_graph, 0);
    }

    cudaGraphDestroy(new_graph);
    return (err == cudaSuccess && g_graph_exec) ? 0 : -1;
}

CUDA_API int ck_graph_launch(void) {
    if (!g_graph_exec) return -1;
    cudaError_t err = cudaGraphLaunch(g_graph_exec, stream_compute);
    return (err == cudaSuccess) ? 0 : -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Batch Prefill API — batched Q8 quant + batched Q4_0 GEMV + rmsnorm_out
 * ════════════════════════════════════════════════════════════════════════ */

CUDA_API void ck_prefill_batch_presized(int max_batch, int max_dim) {
    ensure_batch_q8(max_batch, max_dim);
}

CUDA_API void ck_prefill_batch_quant(const float *X, int batch, int in_dim) {
    if (ensure_batch_q8(batch, in_dim) != 0) return;
    /* grid=(batch), block=256 — 8 warps, each warp handles 32 elements/quant-block */
    kernel_batch_quant_q8<<<batch, 256, 0, stream_compute>>>(
        X, d_bq8, d_bsc, d_bsu, in_dim);
}

CUDA_API void ck_prefill_batch_gemv_q4(
    float      *C,
    const void *W,
    int         out_dim,
    int         in_dim,
    int         batch)
{
    if (!d_bq8) return;
    dim3 grid((out_dim + BGEMV_WARPS - 1) / BGEMV_WARPS, batch);
    kernel_batch_gemv_q4_prequant<<<grid, 256, 0, stream_compute>>>(
        C, W, d_bq8, d_bsc, d_bsu, out_dim, in_dim, batch);
}

CUDA_API void ck_batched_rmsnorm_out(
    float       *out,
    const float *in,
    const float *w,
    int          n_slices,
    int          slice_dim,
    float        eps)
{
    int threads = slice_dim < 256 ? ((slice_dim + 31) & ~31) : 256;
    if (threads < 32) threads = 32;
    kernel_batched_rmsnorm_out<<<n_slices, threads, 0, stream_compute>>>(
        out, in, w, n_slices, slice_dim, eps);
}



/* ─── Batched prefill attention API ─── */



CUDA_API void ck_batch_fused_qk_norm_rope(

    float *Q, float *K,

    const float *q_norm_w, const float *k_norm_w,

    int n_heads, int n_kv_heads, int head_dim,

    int n, int start_pos, float rope_base, const float *rope_freqs,

    float eps, int rope_dim)

{

    /* K may be NULL for layers that reuse another layer's KV cache */

    int kv_heads_for_grid = (K != NULL) ? n_kv_heads : 0;

    int total_heads = n_heads + kv_heads_for_grid;

    dim3 grid(total_heads, n);

    int threads = (head_dim < 256) ? ((head_dim + 31) / 32) * 32 : 256;

    if (threads < 32) threads = 32;

    kernel_batch_fused_qk_norm_rope<<<grid, threads, 0, stream_compute>>>(

        Q, K, q_norm_w, k_norm_w,

        n_heads, kv_heads_for_grid, head_dim,

        start_pos, rope_base, rope_freqs, eps, rope_dim);

}



CUDA_API void ck_batch_v_norm(float *V, int n_kv_heads, int head_dim, int n, float eps) {

    dim3 grid(n_kv_heads, n);

    int threads = (head_dim < 256) ? ((head_dim + 31) / 32) * 32 : 256;

    if (threads < 32) threads = 32;

    kernel_batch_v_norm<<<grid, threads, 0, stream_compute>>>(V, n_kv_heads, head_dim, eps);

}



CUDA_API void ck_batch_kv_update(

    float *K_cache, float *V_cache,

    const float *K_new, const float *V_new,

    int n_kv_heads, int head_dim, int n, int start_pos, int max_seq)

{

    int total = n_kv_heads * head_dim;

    int threads = 256;

    int blocks_x = (total + threads - 1) / threads;

    dim3 grid(blocks_x, n);

    kernel_batch_kv_update<<<grid, threads, 0, stream_compute>>>(

        K_cache, V_cache, K_new, V_new, n_kv_heads, head_dim, start_pos, max_seq);

}



CUDA_API void ck_prefill_attn_batched(

    float *O, const float *Q, const float *K_cache, const float *V_cache,

    int n_heads, int n_kv_heads, int head_dim,

    int n, int start_pos, int max_seq, float scale, float softcap)

{

    dim3 grid(n_heads, n);

    int threads = 256;

    int smem_seq = max_seq;

    int shared_bytes = (smem_seq + 32) * (int)sizeof(float);

    kernel_prefill_attn_batched<<<grid, threads, shared_bytes, stream_compute>>>(

        O, Q, K_cache, V_cache,

        n_heads, n_kv_heads, head_dim,

        n, start_pos, max_seq, scale, softcap, smem_seq);

}

} /* extern "C" */
