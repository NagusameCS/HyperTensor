/*
 * TensorOS CUDA Kernels — Shared Header
 *
 * Declares the C interface for CUDA kernel launch wrappers.
 * Used by both cuda_kernels.cu (implementation) and backend_cuda.c (consumer).
 *
 * All functions use extern "C" linkage and are exported from cuda_kernels.dll.
 */

#ifndef TENSOROS_CUDA_KERNELS_H
#define TENSOROS_CUDA_KERNELS_H

#include <stdint.h>

#ifdef _WIN32
  #ifdef CUDA_KERNELS_EXPORTS
    #define CUDA_API __declspec(dllexport)
  #else
    #define CUDA_API __declspec(dllimport)
  #endif
#else
  #define CUDA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Device init/shutdown ─── */
CUDA_API int  ck_init(void);
CUDA_API void ck_shutdown(void);
CUDA_API int  ck_device_count(void);
CUDA_API uint64_t ck_free_memory(int dev);

/* ─── Memory ops ─── */
CUDA_API void *ck_alloc(uint64_t size);
CUDA_API void  ck_free(void *ptr);
CUDA_API int   ck_upload(void *dst, const void *src, uint64_t size);
CUDA_API int   ck_download(void *dst, const void *src, uint64_t size);
CUDA_API void  ck_sync(void);

/* ─── Compute kernels ─── */

/* GEMV: out[od] = W[od, id] @ x[id], W is quantized (type_id: 2=Q4_0, 8=Q8_0, etc) */
CUDA_API void ck_gemv(float *out, const void *W, const float *x,
                      int out_dim, int in_dim, int type_id);

/* GEMM: C[M,N] = A[M,K] @ B[K,N], all F32 */
CUDA_API void ck_gemm(float *C, const float *A, const float *B,
                      int M, int N, int K);

/* RMSNorm: out[d] = x[d] / sqrt(mean(x^2) + eps) * w[d] */
CUDA_API void ck_rmsnorm(float *out, const float *x, const float *w,
                         int dim, float eps);

/* LayerNorm: out[d] = (x[d] - mean) / sqrt(var + eps) * w[d] + b[d] */
CUDA_API void ck_layernorm(float *out, const float *x, const float *w,
                           const float *bias, int dim, float eps);

/* RoPE: apply rotary position encoding to q and k */
CUDA_API void ck_rope(float *q, float *k, int head_dim, int n_heads,
                      int n_kv_heads, int pos, float base, const float *freqs);

/* Softmax: in-place softmax over n elements */
CUDA_API void ck_softmax(float *x, int n);

/* SiLU: x = x * sigmoid(x) */
CUDA_API void ck_silu(float *x, int n);

/* GELU: x = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
CUDA_API void ck_gelu(float *x, int n);

/* Fused GeGLU: gate[i] = GELU(gate[i]) * up[i] — one kernel instead of two */
CUDA_API void ck_fused_geglu(float *gate, const float *up, int n);

/* Fused SwiGLU: gate[i] = SiLU(gate[i]) * up[i] */
CUDA_API void ck_fused_swiglu(float *gate, const float *up, int n);

/* Batched RMSNorm: normalize n_slices independent vectors, shared weights */
CUDA_API void ck_batched_rmsnorm(float *data, const float *w,
                                  int n_slices, int slice_dim, float eps);

/* ISWA combine: out[i] = (tok_embd[i] + proj[i]) * scale */
CUDA_API void ck_iswa_combine(float *out, const float *tok_embd,
                               const float *proj, float scale, int n);

/* Element-wise multiply: out = a * b */
CUDA_API void ck_mul(float *out, const float *a, const float *b, int n);

/* Element-wise add: out = a + b */
CUDA_API void ck_add(float *out, const float *a, const float *b, int n);

/* Scale: out = x * s */
CUDA_API void ck_scale(float *out, const float *x, float s, int n);

/* Dot product: returns sum(a[i] * b[i]) */
CUDA_API float ck_dot(const float *a, const float *b, int n);

/* Dequantize: quantized data → float, type_id as in GGML */
CUDA_API void ck_dequantize(float *out, const void *data, int n_elements,
                            int type_id);

/* Multi-head attention:
 * out[n_heads * head_dim] = Attention(Q, K_cache, V_cache) */
CUDA_API void ck_attention(float *out, const float *Q, const float *K,
                           const float *V, int n_heads, int n_kv_heads,
                           int head_dim, int seq_len, int max_seq,
                           float scale, float softcap);

/* KV cache update: insert new K,V vectors at position pos */
CUDA_API void ck_kv_update(float *K_cache, float *V_cache,
                           const float *K_new, const float *V_new,
                           int n_kv_heads, int head_dim, int pos,
                           int max_seq, int layer);

/* Embedding lookup: out[dim] = table[token_id * dim] (with dequant) */
CUDA_API void ck_embed_lookup(float *out, const void *table, int token_id,
                              int dim, int type_id);

/* Softcap: x = cap * tanh(x / cap) */
CUDA_API void ck_softcap(float *x, int n, float cap);

/* GPU-side argmax: returns index of max element (avoids downloading all logits) */
CUDA_API int ck_argmax(const float *data, int n);

/* ─── Async memory ops (non-blocking) ─── */
CUDA_API int  ck_upload_async(void *dst, const void *src, uint64_t size);
CUDA_API int  ck_download_async(void *dst, const void *src, uint64_t size);
CUDA_API void ck_stream_sync_transfer(void);
CUDA_API void ck_stream_sync_compute(void);

/* ─── Fused QKV normalization + RoPE ─── */
CUDA_API void ck_fused_qk_norm_rope(
    float *Q, float *K,
    const float *q_norm_w, const float *k_norm_w,
    int n_heads, int n_kv_heads, int head_dim,
    int pos, float rope_base, const float *rope_freqs,
    float eps, int rope_dim);

/* Per-head V magnitude normalization */
CUDA_API void ck_v_norm(float *V, int n_kv_heads, int head_dim, float eps);

/* GEMV on compute stream (non-blocking) */
CUDA_API void ck_gemv_async(float *out, const void *W, const float *x,
                            int out_dim, int in_dim, int type_id);

/* Q4_0 → FP16 dequantization on GPU */
CUDA_API void ck_dequant_q4_0_to_f16(void *out, const void *q4_data,
                                      int n_rows, int in_dim);

/* ─── Fused compute kernels ─── */

/* Add residual + RMSNorm: x_inout += residual, norm_out = rmsnorm(x_inout) */
CUDA_API void ck_add_rmsnorm(float *norm_out, float *x_inout,
                              const float *residual, const float *norm_w,
                              int dim, float eps);

/* RMSNorm + Add: x_inout += rmsnorm(data) */
CUDA_API void ck_rmsnorm_add(float *x_inout, const float *data,
                              const float *norm_w, int dim, float eps);

/* GELU-multiply: a[i] = GELU(a[i]) * b[i] */
CUDA_API void ck_gelu_mul(float *a, const float *b, int n);

/* Fused dual Q4_0 GEMV: gate+up in one kernel (shared Q8 quantization) */
CUDA_API void ck_gemv_dual_q4_0(
    float *out_a, float *out_b,
    const void *W_a, const void *W_b,
    const float *x,
    int out_dim, int in_dim);

/* Fused dual Q8_0 GEMV: two outputs share the same input x and launch. */
CUDA_API void ck_gemv_dual_q8_0(
  float *out_a, float *out_b,
  const void *W_a, const void *W_b,
  const float *x,
  int out_dim, int in_dim);

/* Fused triple Q4_0 GEMV: Q+K+V in one kernel (shared Q8 quantization) */
CUDA_API void ck_gemv_triple_q4_0(
    float *out_q, float *out_k, float *out_v,
    const void *W_q, const void *W_k, const void *W_v,
    const float *x,
    int q_dim, int k_dim, int v_dim,
    int in_dim);

/* Fused RMSNorm + triple Q4_0 GEMV: normalizes x on-the-fly and computes Q+K+V
 * in one kernel launch (no d_xn write).  Returns void; caller checks availability
 * via the dynamic-load path in backend_cuda.c. */
CUDA_API void ck_fused_rmsnorm_triple_q4_0(
    float *out_q, float *out_k, float *out_v,
    const void *W_q, const void *W_k, const void *W_v,
    const float *x, const float *norm_w, float eps,
    int q_dim, int k_dim, int v_dim,
    int in_dim);

/* ─── cuBLAS Batched GEMV (row-major F32) — for compressed weight gate+up fusion ─── */
/*   y[i] = A[i] * x[i],  A[i] is [M×K] row-major, x[i] is [K], y[i] is [M].       */
/*   All pointer arrays (d_Aarray, d_xarray, d_yarray) must be in device memory.      */
CUDA_API void ck_sgemm_batched_f32(
    int M, int K,
    const float * const *d_Aarray,
    const float * const *d_xarray,
    float * const       *d_yarray,
    int batch_count);

/* ─── CUDA Graph support ─── */
CUDA_API void ck_graph_destroy(void);
CUDA_API void ck_set_decode_pos(int pos, int seq_len);
CUDA_API int  ck_graph_begin_capture(void);
CUDA_API int  ck_graph_end_capture(void);
CUDA_API int  ck_graph_launch(void);

/* ─── Batch Prefill API ─── */
CUDA_API void ck_prefill_batch_presized(int max_batch, int max_dim);
CUDA_API void ck_prefill_batch_quant(const float *X, int batch, int in_dim);
CUDA_API void ck_prefill_batch_gemv_q4(float *C, const void *W,
    int out_dim, int in_dim, int batch);
CUDA_API void ck_batched_rmsnorm_out(float *out, const float *in,
    const float *w, int n_slices, int slice_dim, float eps);

#ifdef __cplusplus
}
#endif

#endif /* TENSOROS_CUDA_KERNELS_H */
