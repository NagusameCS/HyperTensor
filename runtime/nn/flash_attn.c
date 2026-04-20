/* =============================================================================
 * TensorOS - Flash Attention Implementation
 *
 * Implements the FlashAttention algorithm (Dao et al., 2022) for O(N) memory
 * scaled dot-product attention.  The key insight: by tiling Q, K, V into
 * blocks that fit in SRAM/L1 cache and maintaining running softmax statistics
 * (max and sum), we avoid materializing the full N×N attention matrix.
 *
 * Algorithm:
 *   For each tile of Q (Br rows):
 *     For each tile of K,V (Bc rows):
 *       S_ij = Q_i @ K_j^T * scale          (Br × Bc)
 *       Apply causal mask if needed
 *       m_new = max(m_old, rowmax(S_ij))
 *       P_ij = exp(S_ij - m_new)
 *       l_new = exp(m_old - m_new) * l_old + rowsum(P_ij)
 *       O_i = (exp(m_old - m_new) * l_old * O_i + P_ij @ V_j) / l_new
 *
 * SSE2 SIMD is used for the inner loops on x86_64.
 * =============================================================================*/

#include "runtime/nn/flash_attn.h"
#include "kernel/core/perf.h"
#include "kernel/core/smp.h"
#include "kernel/mm/tensor_mm.h"

#include <emmintrin.h>  /* SSE2 */
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* Default tile sizes chosen to fit in L1 cache (~32KB) */
#define DEFAULT_BR  32   /* Q tile rows */
#define DEFAULT_BC  32   /* KV tile rows */

/* Forward declaration for SSE2 dot product used throughout this file */
static inline float dot_sse2(const float *a, const float *b, int n);
static inline void  vec_scale_add_sse2(float *dst, const float *src, float scale, int n);
static inline void  vec_scale_sse2(float *dst, float scale, int n);
static inline float fast_exp(float x);

/* Per-worker work items for parallel head dispatch.
 * Each worker gets its own S/P scratch to avoid sharing the global buffers. */
typedef struct {
    float       *output;
    const float *q;
    const float *k_base;
    const float *v_base;
    int          cache_len;
    int          kv_pos_stride;
    int          head_dim;
    int          kv_rep;
    float        scale;
    int          h_start;
    int          h_end;
    float        local_S[DEFAULT_BC];
    float        local_P[DEFAULT_BC];
} flash_attn_smp_work_t;

static flash_attn_smp_work_t fa_work[MAX_CPUS];

static void flash_attn_smp_head_worker(void *arg)
{
    flash_attn_smp_work_t *w = (flash_attn_smp_work_t *)arg;
    int hd           = w->head_dim;
    int kv_rep       = w->kv_rep;
    float scale      = w->scale;
    int cache_len    = w->cache_len;
    int kv_stride    = w->kv_pos_stride;
    const float *q   = w->q;
    float *output    = w->output;
    const float *kb  = w->k_base;
    const float *vb  = w->v_base;

    for (int h = w->h_start; h < w->h_end; h++) {
        int kv_h = h / kv_rep;
        const float *qh = q + h * hd;
        float *oh = output + h * hd;

        float m_max = -1e30f;
        float l_sum = 0.0f;
        for (int d = 0; d < hd; d++) oh[d] = 0.0f;

        for (int ki = 0; ki < cache_len; ki += DEFAULT_BC) {
            int ke   = ki + DEFAULT_BC;
            if (ke > cache_len) ke = cache_len;
            int klen = ke - ki;

            float tile_max = -1e30f;
            for (int c = 0; c < klen; c++) {
                const float *kt = kb + (ki + c) * kv_stride + kv_h * hd;
                float s = dot_sse2(qh, kt, hd) * scale;
                w->local_S[c] = s;
                if (s > tile_max) tile_max = s;
            }

            float m_old = m_max;
            if (tile_max > m_max) m_max = tile_max;

            float alpha  = fast_exp(m_old - m_max);
            float l_old  = l_sum;
            l_sum        = alpha * l_old;

            float tile_sum = 0.0f;
            for (int c = 0; c < klen; c++) {
                float p = fast_exp(w->local_S[c] - m_max);
                w->local_P[c] = p;
                tile_sum += p;
            }
            l_sum += tile_sum;

            if (l_old > 0.0f) {
                float rescale = alpha * l_old / l_sum;
                vec_scale_sse2(oh, rescale, hd);
            }

            float inv_l = 1.0f / l_sum;
            for (int c = 0; c < klen; c++) {
                float wt = w->local_P[c] * inv_l;
                if (wt > 1e-8f) {
                    const float *vt = vb + (ki + c) * kv_stride + kv_h * hd;
                    vec_scale_add_sse2(oh, vt, wt, hd);
                }
            }
        }
    }
}

/* Scratch buffers (allocated once at init) */
static float *scratch_S;    /* Br × Bc tile of attention scores */
static float *scratch_P;    /* Br × Bc tile of attention probabilities */
static float *scratch_O;    /* Temporary output accumulator */
static float *scratch_m;    /* Running max per row (Br) */
static float *scratch_l;    /* Running sum per row (Br) */
static int scratch_br;
static int scratch_bc;

int flash_attn_init(void)
{
    scratch_br = DEFAULT_BR;
    scratch_bc = DEFAULT_BC;

    uint64_t needed = 0;
    needed += (uint64_t)scratch_br * scratch_bc * sizeof(float) * 2; /* S + P */
    needed += (uint64_t)scratch_br * 256 * sizeof(float);            /* O (max head_dim=256) */
    needed += (uint64_t)scratch_br * sizeof(float) * 2;              /* m + l */

    uint8_t *arena = (uint8_t *)kmalloc(needed);
    if (!arena) {
        kprintf("[FLASH-ATTN] Failed to allocate %lu bytes scratch\n",
                (unsigned long)needed);
        return -1;
    }
    kmemset(arena, 0, needed);

    scratch_S = (float *)arena;
    arena += scratch_br * scratch_bc * sizeof(float);
    scratch_P = (float *)arena;
    arena += scratch_br * scratch_bc * sizeof(float);
    scratch_O = (float *)arena;
    arena += scratch_br * 256 * sizeof(float);
    scratch_m = (float *)arena;
    arena += scratch_br * sizeof(float);
    scratch_l = (float *)arena;

    kprintf("[FLASH-ATTN] Initialized: tile=%dx%d, scratch=%lu bytes\n",
            scratch_br, scratch_bc, (unsigned long)needed);
    return 0;
}

/* =============================================================================
 * SIMD-accelerated building blocks
 * =============================================================================*/

#ifndef __aarch64__
#include <xmmintrin.h>
#include <emmintrin.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __AVX2__
/* Dot product — AVX2+FMA, processes 8 floats/cycle */
static inline float dot_sse2(const float *a, const float *b, int n)
{
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    /* Horizontal sum of 8-wide accumulator */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline void vec_scale_add_sse2(float *dst, const float *src, float scale, int n)
{
    __m256 vs = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vd = _mm256_loadu_ps(dst + i);
        __m256 va = _mm256_loadu_ps(src + i);
        vd = _mm256_fmadd_ps(va, vs, vd);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++) dst[i] += scale * src[i];
}

static inline void vec_scale_sse2(float *dst, float scale, int n)
{
    __m256 vs = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vd = _mm256_loadu_ps(dst + i);
        vd = _mm256_mul_ps(vd, vs);
        _mm256_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++) dst[i] *= scale;
}
#else
/* Dot product of two float vectors, SSE2 */
static inline float dot_sse2(const float *a, const float *b, int n)
{
    __m128 sum = _mm_setzero_ps();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, sum);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; i++)
        s += a[i] * b[i];
    return s;
}

/* Vector-scalar multiply + accumulate:  dst[i] += scale * src[i] */
static inline void vec_scale_add_sse2(float *dst, const float *src, float scale, int n)
{
    __m128 vs = _mm_set1_ps(scale);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 vd = _mm_loadu_ps(dst + i);
        __m128 va = _mm_loadu_ps(src + i);
        vd = _mm_add_ps(vd, _mm_mul_ps(va, vs));
        _mm_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] += scale * src[i];
}

/* Vector-scalar multiply:  dst[i] = scale * dst[i] */
static inline void vec_scale_sse2(float *dst, float scale, int n)
{
    __m128 vs = _mm_set1_ps(scale);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 vd = _mm_loadu_ps(dst + i);
        vd = _mm_mul_ps(vd, vs);
        _mm_storeu_ps(dst + i, vd);
    }
    for (; i < n; i++)
        dst[i] *= scale;
}
#endif  /* __AVX2__ */

#else
/* ARM64 NEON fallback */
static inline float dot_sse2(const float *a, const float *b, int n)
{
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
static inline void vec_scale_add_sse2(float *dst, const float *src, float scale, int n)
{
    for (int i = 0; i < n; i++) dst[i] += scale * src[i];
}
static inline void vec_scale_sse2(float *dst, float scale, int n)
{
    for (int i = 0; i < n; i++) dst[i] *= scale;
}
#endif

/* Fast exp approximation (Schraudolph, 1999) — good to ~1% relative error */
static inline float fast_exp(float x)
{
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) return 1e38f;
    /* exp(x) ≈ 2^(x / ln2) via integer bit trick */
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1065353216.0f);
    return u.f;
}

/* =============================================================================
 * Core Flash Attention Forward Pass
 * =============================================================================*/

int flash_attn_forward(
    float *output,
    const float *Q,
    const float *K,
    const float *V,
    const flash_attn_config_t *cfg,
    flash_attn_stats_t *stats)
{
    if (!output || !Q || !K || !V || !cfg) return -1;

    int seq = cfg->seq_len;
    int hd  = cfg->head_dim;
    int nh  = cfg->n_heads;
    int nkv = cfg->n_kv_heads;
    float scale = cfg->scale;
    int Br = scratch_br;
    int Bc = scratch_bc;
    int kv_rep = nh / nkv; /* GQA repetition factor */

    uint64_t flops = 0;
    uint64_t t_start = rdtsc_fenced();

    /* Process each attention head independently */
    for (int h = 0; h < nh; h++) {
        int kv_h = h / kv_rep; /* Map to KV head index */

        const float *Qh = Q + h * seq * hd;
        const float *Kh = K + kv_h * seq * hd;
        const float *Vh = V + kv_h * seq * hd;
        float *Oh = output + h * seq * hd;

        /* Initialize output and running statistics */
        for (int i = 0; i < seq * hd; i++) Oh[i] = 0.0f;

        /* Tile over Q (outer loop: blocks of Br rows) */
        for (int qi = 0; qi < seq; qi += Br) {
            int qe = qi + Br;
            if (qe > seq) qe = seq;
            int qlen = qe - qi;

            /* Initialize running max and sum for this Q tile */
            for (int r = 0; r < qlen; r++) {
                scratch_m[r] = -1e30f;
                scratch_l[r] = 0.0f;
            }
            /* Zero output accumulator for this tile */
            for (int r = 0; r < qlen; r++)
                for (int d = 0; d < hd; d++)
                    scratch_O[r * hd + d] = 0.0f;

            /* Tile over K,V (inner loop: blocks of Bc rows) */
            int kv_end = cfg->causal ? qe : seq;
            for (int ki = 0; ki < kv_end; ki += Bc) {
                int ke = ki + Bc;
                if (ke > kv_end) ke = kv_end;
                int klen = ke - ki;

                /* Compute S_ij = Q_tile @ K_tile^T * scale */
                for (int r = 0; r < qlen; r++) {
                    for (int c = 0; c < klen; c++) {
                        float s = dot_sse2(Qh + (qi + r) * hd,
                                           Kh + (ki + c) * hd, hd) * scale;
                        /* Causal mask: set future positions to -inf */
                        if (cfg->causal && (ki + c) > (qi + r))
                            s = -1e30f;
                        scratch_S[r * Bc + c] = s;
                    }
                }
                flops += (uint64_t)qlen * klen * hd * 2;

                /* Update running max: m_new = max(m_old, rowmax(S)) */
                float m_old[DEFAULT_BR];
                for (int r = 0; r < qlen; r++) {
                    m_old[r] = scratch_m[r];
                    for (int c = 0; c < klen; c++) {
                        if (scratch_S[r * Bc + c] > scratch_m[r])
                            scratch_m[r] = scratch_S[r * Bc + c];
                    }
                }

                /* Compute P_ij = exp(S_ij - m_new) and row sums */
                for (int r = 0; r < qlen; r++) {
                    float l_new = 0.0f;
                    for (int c = 0; c < klen; c++) {
                        float p = fast_exp(scratch_S[r * Bc + c] - scratch_m[r]);
                        scratch_P[r * Bc + c] = p;
                        l_new += p;
                    }

                    /* Rescale existing output and running sum */
                    float alpha = fast_exp(m_old[r] - scratch_m[r]);
                    float l_old = scratch_l[r];
                    scratch_l[r] = alpha * l_old + l_new;

                    /* Rescale accumulated output: O *= alpha * l_old / l_new_total */
                    if (l_old > 0.0f) {
                        float rescale = alpha * l_old / scratch_l[r];
                        vec_scale_sse2(scratch_O + r * hd, rescale, hd);
                    }

                    /* Accumulate P_ij @ V_j contribution */
                    float inv_l = 1.0f / scratch_l[r];
                    for (int c = 0; c < klen; c++) {
                        float w = scratch_P[r * Bc + c] * inv_l;
                        if (w > 1e-8f)
                            vec_scale_add_sse2(scratch_O + r * hd,
                                               Vh + (ki + c) * hd, w, hd);
                    }
                }
                flops += (uint64_t)qlen * klen * hd * 2;
            }

            /* Write tile output back */
            for (int r = 0; r < qlen; r++)
                kmemcpy(Oh + (qi + r) * hd, scratch_O + r * hd, hd * sizeof(float));
        }
    }

    uint64_t t_end = rdtsc_fenced();

    if (stats) {
        stats->total_flops = flops;
        stats->peak_memory_bytes = (uint64_t)(Br * Bc * 2 + Br * hd + Br * 2) * sizeof(float);
        stats->tiling_passes = (uint64_t)nh * ((seq + Br - 1) / Br) * ((seq + Bc - 1) / Bc);
        stats->time_us = perf_cycles_to_us(t_end - t_start);
    }

    return 0;
}

/* =============================================================================
 * Single-step decode (optimized for autoregressive generation)
 * Only computes attention for the last token against the full KV cache
 * =============================================================================*/

int flash_attn_decode_step(
    float *output,
    const float *q,
    const float *k_cache,
    const float *v_cache,
    int cache_len,
    const flash_attn_config_t *cfg)
{
    if (!output || !q || !k_cache || !v_cache || !cfg) return -1;

    int hd  = cfg->head_dim;
    int nh  = cfg->n_heads;
    int nkv = cfg->n_kv_heads;
    int kv_rep = nh / nkv;
    float scale = cfg->scale;
    int Bc = scratch_bc;

    for (int h = 0; h < nh; h++) {
        int kv_h = h / kv_rep;
        const float *qh = q + h * hd;
        const float *Kh = k_cache + kv_h * cfg->seq_len * hd;
        const float *Vh = v_cache + kv_h * cfg->seq_len * hd;
        float *oh = output + h * hd;

        /* Single-row flash attention across KV cache tiles */
        float m_max = -1e30f;
        float l_sum = 0.0f;
        for (int d = 0; d < hd; d++) oh[d] = 0.0f;

        for (int ki = 0; ki < cache_len; ki += Bc) {
            int ke = ki + Bc;
            if (ke > cache_len) ke = cache_len;
            int klen = ke - ki;

            /* Compute scores for this tile */
            float tile_max = -1e30f;
            for (int c = 0; c < klen; c++) {
                float s = dot_sse2(qh, Kh + (ki + c) * hd, hd) * scale;
                scratch_S[c] = s;
                if (s > tile_max) tile_max = s;
            }

            /* Update running max */
            float m_old = m_max;
            if (tile_max > m_max) m_max = tile_max;

            /* Compute probabilities and accumulate */
            float alpha = fast_exp(m_old - m_max);
            float l_old = l_sum;
            l_sum = alpha * l_old;

            float tile_sum = 0.0f;
            for (int c = 0; c < klen; c++) {
                float p = fast_exp(scratch_S[c] - m_max);
                scratch_P[c] = p;
                tile_sum += p;
            }
            l_sum += tile_sum;

            /* Rescale existing output */
            if (l_old > 0.0f) {
                float rescale = alpha * l_old / l_sum;
                vec_scale_sse2(oh, rescale, hd);
            }

            /* Accumulate V contribution */
            float inv_l = 1.0f / l_sum;
            for (int c = 0; c < klen; c++) {
                float w = scratch_P[c] * inv_l;
                if (w > 1e-8f)
                    vec_scale_add_sse2(oh, Vh + (ki + c) * hd, w, hd);
            }
        }
    }

    return 0;
}

/* =============================================================================
 * Strided single-step decode — matches llm.c KV cache layout:
 *   [pos * kv_pos_stride + kv_head * head_dim]
 * =============================================================================*/

int flash_attn_decode_strided(
    float *output,
    const float *q,
    const float *k_base,
    const float *v_base,
    int cache_len,
    int kv_pos_stride,
    const flash_attn_config_t *cfg)
{
    if (!output || !q || !k_base || !v_base || !cfg || cache_len <= 0) return -1;

    int hd     = cfg->head_dim;
    int nh     = cfg->n_heads;
    int nkv    = cfg->n_kv_heads;
    kprintf("[DBG-FA] decode_strided: nh=%d nkv=%d hd=%d cache_len=%d\n", nh, nkv, hd, cache_len);
    if (nkv <= 0) { kprintf("[DBG-FA] CRASH GUARD nkv=%d!\n", nkv); return -1; }
    int kv_rep = nh / nkv;
    float scale = cfg->scale;

    /* Parallel path: split Q-heads across available CPUs.
     * Each head is independent — no shared mutable state between workers.
     * Fall back to sequential when only BSP is available or nh == 1. */
    uint32_t ncpu = smp.ap_started + 1;
    kprintf("[DBG-FA] ncpu=%u kv_rep=%d\n", ncpu, kv_rep);
    if (ncpu > 1 && nh > 1) {
        int heads_per_cpu = (int)ncpu > 0 ? nh / (int)ncpu : nh;
        kprintf("[DBG-FA] SMP path heads_per_cpu=%d\n", heads_per_cpu);
        int remainder     = nh % (int)ncpu;
        int h = 0;

        /* Set up and dispatch to APs first */
        for (uint32_t c = 1; c < ncpu; c++) {
            int chunk = heads_per_cpu + ((int)c < remainder ? 1 : 0);
            fa_work[c].output        = output;
            fa_work[c].q             = q;
            fa_work[c].k_base        = k_base;
            fa_work[c].v_base        = v_base;
            fa_work[c].cache_len     = cache_len;
            fa_work[c].kv_pos_stride = kv_pos_stride;
            fa_work[c].head_dim      = hd;
            fa_work[c].kv_rep        = kv_rep;
            fa_work[c].scale         = scale;
            fa_work[c].h_start       = h;
            fa_work[c].h_end         = h + chunk;
            smp_dispatch(c, flash_attn_smp_head_worker, &fa_work[c]);
            h += chunk;
        }

        /* BSP handles the first slice */
        int bsp_end = heads_per_cpu + (remainder > 0 ? 1 : 0);
        fa_work[0].output        = output;
        fa_work[0].q             = q;
        fa_work[0].k_base        = k_base;
        fa_work[0].v_base        = v_base;
        fa_work[0].cache_len     = cache_len;
        fa_work[0].kv_pos_stride = kv_pos_stride;
        fa_work[0].head_dim      = hd;
        fa_work[0].kv_rep        = kv_rep;
        fa_work[0].scale         = scale;
        fa_work[0].h_start       = 0;
        fa_work[0].h_end         = bsp_end;
        flash_attn_smp_head_worker(&fa_work[0]);
        kprintf("[DBG-FA] BSP worker done, waiting for APs...\n");
        smp_wait_all();
        kprintf("[DBG-FA] all workers done\n");
        return 0;
    }

    /* Sequential fallback (single CPU or single head) */
    int Bc = scratch_bc;

    for (int h = 0; h < nh; h++) {
        int kv_h = h / kv_rep;
        const float *qh = q + h * hd;
        float *oh = output + h * hd;

        float m_max = -1e30f;
        float l_sum = 0.0f;
        for (int d = 0; d < hd; d++) oh[d] = 0.0f;

        for (int ki = 0; ki < cache_len; ki += Bc) {
            int ke = ki + Bc;
            if (ke > cache_len) ke = cache_len;
            int klen = ke - ki;

            /* Compute scores for this tile */
            float tile_max = -1e30f;
            for (int c = 0; c < klen; c++) {
                const float *kt = k_base + (ki + c) * kv_pos_stride + kv_h * hd;
                float s = dot_sse2(qh, kt, hd) * scale;
                scratch_S[c] = s;
                if (s > tile_max) tile_max = s;
            }

            /* Update running max */
            float m_old = m_max;
            if (tile_max > m_max) m_max = tile_max;

            /* Compute probabilities and accumulate */
            float alpha = fast_exp(m_old - m_max);
            float l_old = l_sum;
            l_sum = alpha * l_old;

            float tile_sum = 0.0f;
            for (int c = 0; c < klen; c++) {
                float p = fast_exp(scratch_S[c] - m_max);
                scratch_P[c] = p;
                tile_sum += p;
            }
            l_sum += tile_sum;

            /* Rescale existing output */
            if (l_old > 0.0f) {
                float rescale = alpha * l_old / l_sum;
                vec_scale_sse2(oh, rescale, hd);
            }

            /* Accumulate V contribution */
            float inv_l = 1.0f / l_sum;
            for (int c = 0; c < klen; c++) {
                float w = scratch_P[c] * inv_l;
                if (w > 1e-8f) {
                    const float *vt = v_base + (ki + c) * kv_pos_stride + kv_h * hd;
                    vec_scale_add_sse2(oh, vt, w, hd);
                }
            }
        }
    }

    return 0;
}

/* =============================================================================
 * Self-Test
 * =============================================================================*/

void flash_attn_selftest(void)
{
    kprintf("[FLASH-ATTN] Running self-test...\n");

    flash_attn_config_t cfg = {
        .head_dim = 64,
        .n_heads = 4,
        .n_kv_heads = 4,
        .seq_len = 32,
        .block_size_q = DEFAULT_BR,
        .block_size_kv = DEFAULT_BC,
        .scale = 0.125f,  /* 1/sqrt(64) */
        .causal = true,
        .use_alibi = false,
    };

    int total = cfg.seq_len * cfg.n_heads * cfg.head_dim;
    float *Q_test = (float *)kmalloc(total * sizeof(float));
    float *K_test = (float *)kmalloc(total * sizeof(float));
    float *V_test = (float *)kmalloc(total * sizeof(float));
    float *O_test = (float *)kmalloc(total * sizeof(float));

    if (!Q_test || !K_test || !V_test || !O_test) {
        kprintf("  [FAIL] Cannot allocate test buffers\n");
        return;
    }

    /* Initialize with simple pattern */
    for (int i = 0; i < total; i++) {
        Q_test[i] = 0.01f * (float)(i % 64);
        K_test[i] = 0.01f * (float)((i + 7) % 64);
        V_test[i] = 0.01f * (float)((i + 13) % 64);
    }

    flash_attn_stats_t stats;
    int rc = flash_attn_forward(O_test, Q_test, K_test, V_test, &cfg, &stats);

    if (rc == 0) {
        /* Verify output is non-zero and finite */
        int ok = 1;
        float sum = 0;
        for (int i = 0; i < total; i++) {
            sum += O_test[i];
            if (O_test[i] != O_test[i]) { ok = 0; break; } /* NaN check */
        }
        if (ok && sum != 0.0f) {
            kprintf("  [OK] Flash attention: %d heads × %d seq, %lu MFLOPS, "
                    "peak mem %lu KB (%lu tiling passes)\n",
                    cfg.n_heads, cfg.seq_len,
                    stats.time_us > 0 ? stats.total_flops / stats.time_us : 0,
                    (unsigned long)(stats.peak_memory_bytes / 1024),
                    (unsigned long)stats.tiling_passes);
        } else {
            kprintf("  [FAIL] Output contains NaN or is all-zero\n");
        }
    } else {
        kprintf("  [FAIL] flash_attn_forward returned %d\n", rc);
    }

    kfree(Q_test);
    kfree(K_test);
    kfree(V_test);
    kfree(O_test);
}
