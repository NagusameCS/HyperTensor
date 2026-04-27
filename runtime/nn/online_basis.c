/*
 * online_basis.c - Online PCA basis update triggered by speculative-decode rejections
 *
 * See online_basis.h for design documentation.
 */

#include "runtime/nn/online_basis.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef GEODESSICAL_HOSTED
#  include "host/hal.h"
#  define ONB_ALLOC(n)    malloc(n)
#  define ONB_FREE(p)     free(p)
#else
#  include "kernel/core/kernel.h"
#  include "kernel/mm/tensor_mm.h"
#  define ONB_ALLOC(n)    tensor_alloc(n)
#  define ONB_FREE(p)     tensor_free(p)
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#endif

/* --------------------------------------------------------------------------
 * Internal: Gram-Schmidt deflated Oja update for k components.
 *
 * For each principal direction w_i (row i of ls->W):
 *   score_i = dot(x, w_i)
 *   w_i += eta * score_i * x
 *   deflate x from w_i (prevents all components collapsing to PC1):
 *     x_deflated = x - sum_{j<i}(dot(x,w_j)*w_j)  -- done once outside loop
 *   normalise w_i
 * --------------------------------------------------------------------------
 */
static void oja_update_k(onb_layer_state_t *ls, const float *x)
{
    const int dim = ls->dim;
    const int k   = ls->k;
    const double eta = ls->eta0 / sqrt((double)(ls->t + 1));

    /* Working copy of x for deflation */
    float *xd = (float *)ONB_ALLOC((size_t)dim * sizeof(float));
    if (!xd) return;
    memcpy(xd, x, (size_t)dim * sizeof(float));

    for (int i = 0; i < k; i++) {
        float *wi = ls->W + (size_t)i * dim;

        /* score = dot(xd, wi) */
        float score = 0.0f;
#ifdef __AVX2__
        int j = 0;
        __m256 acc = _mm256_setzero_ps();
        for (; j + 8 <= dim; j += 8) {
            __m256 a = _mm256_loadu_ps(xd + j);
            __m256 b = _mm256_loadu_ps(wi + j);
            acc = _mm256_fmadd_ps(a, b, acc);
        }
        /* horizontal sum */
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        score = _mm_cvtss_f32(lo);
        for (; j < dim; j++) score += xd[j] * wi[j];
#else
        for (int j = 0; j < dim; j++) score += xd[j] * wi[j];
#endif

        /* w_i += eta * score * xd */
        float step = (float)(eta * score);
        float norm_sq = 0.0f;
        for (int j = 0; j < dim; j++) {
            wi[j] += step * xd[j];
            norm_sq += wi[j] * wi[j];
        }

        /* Normalise */
        if (norm_sq > 1e-12f) {
            float inv_norm = 1.0f / sqrtf(norm_sq);
            for (int j = 0; j < dim; j++) wi[j] *= inv_norm;
        }

        /* Deflate xd: xd -= dot(xd, wi_new) * wi_new */
        float proj = 0.0f;
        for (int j = 0; j < dim; j++) proj += xd[j] * wi[j];
        for (int j = 0; j < dim; j++) xd[j] -= proj * wi[j];
    }

    ls->t++;
    ONB_FREE(xd);
}

/* --------------------------------------------------------------------------
 * Lifecycle
 * --------------------------------------------------------------------------
 */

int onb_init(onb_ctx_t *ctx,
             int n_layers,
             const int *dims,
             const int *ks,
             float * const *existing_bases,
             double eta0)
{
    if (!ctx || !dims || !ks || n_layers <= 0) return -1;
    if (n_layers > ONB_MAX_LAYERS) n_layers = ONB_MAX_LAYERS;

    memset(ctx, 0, sizeof(*ctx));
    ctx->n_layers = n_layers;
    ctx->eta0     = (eta0 > 0.0) ? eta0 : 0.01;
    ctx->updates_per_rejection         = 1;
    ctx->min_rejections_before_update  = 4;

    for (int l = 0; l < n_layers; l++) {
        onb_layer_state_t *ls = &ctx->layers[l];
        int dim = dims[l];
        int k   = ks[l];
        if (dim <= 0 || dim > ONB_MAX_DIM) dim = 1;
        if (k   <= 0 || k   > ONB_MAX_K)   k   = 1;
        if (k > dim) k = dim;

        ls->dim    = dim;
        ls->k      = k;
        ls->eta0   = ctx->eta0;
        ls->t      = 0;
        ls->version = 0;
        ls->active  = 0;

        ls->W = (float *)ONB_ALLOC((size_t)k * dim * sizeof(float));
        if (!ls->W) {
            /* Back out any already-allocated layers */
            for (int j = 0; j < l; j++) ONB_FREE(ctx->layers[j].W);
            return -2;
        }

        if (existing_bases && existing_bases[l]) {
            memcpy(ls->W, existing_bases[l], (size_t)k * dim * sizeof(float));
        } else {
            /* Identity start: w_i = e_i (or random if dim > k) */
            memset(ls->W, 0, (size_t)k * dim * sizeof(float));
            for (int i = 0; i < k && i < dim; i++)
                ls->W[(size_t)i * dim + i] = 1.0f;
        }
    }
    return 0;
}

void onb_destroy(onb_ctx_t *ctx)
{
    if (!ctx) return;
    for (int l = 0; l < ctx->n_layers; l++)
        if (ctx->layers[l].W) { ONB_FREE(ctx->layers[l].W); ctx->layers[l].W = 0; }
    memset(ctx, 0, sizeof(*ctx));
}

void onb_reset_stats(onb_ctx_t *ctx)
{
    if (!ctx) return;
    ctx->total_rejections = 0;
    ctx->total_updates    = 0;
    ctx->queue_len        = 0;
    for (int l = 0; l < ctx->n_layers; l++)
        ctx->layers[l].t = 0;
}

/* --------------------------------------------------------------------------
 * Recording rejections
 * --------------------------------------------------------------------------
 */

int onb_record_rejection(onb_ctx_t *ctx,
                         int layer,
                         const float *target_hidden,
                         const float *draft_hidden)
{
    if (!ctx || !target_hidden || !draft_hidden) return -1;
    if (layer < 0 || layer >= ctx->n_layers) return -1;

    const int dim = ctx->layers[layer].dim;

    /* Compute residual and enqueue */
    if (ctx->queue_len >= ONB_QUEUE_CAP) {
        /* Queue full - drop oldest entry to make room */
        memmove(&ctx->queue[0], &ctx->queue[1],
                (size_t)(ONB_QUEUE_CAP - 1) * sizeof(onb_pending_t));
        ctx->queue_len = ONB_QUEUE_CAP - 1;
    }

    onb_pending_t *entry = &ctx->queue[ctx->queue_len++];
    entry->layer = layer;
    int copy_dim = dim < ONB_MAX_DIM ? dim : ONB_MAX_DIM;
    for (int i = 0; i < copy_dim; i++)
        entry->residual[i] = target_hidden[i] - draft_hidden[i];

    ctx->total_rejections++;
    return 0;
}

int onb_record_residual(onb_ctx_t *ctx, int layer, const float *residual)
{
    if (!ctx || !residual) return -1;
    if (layer < 0 || layer >= ctx->n_layers) return -1;

    if (ctx->queue_len >= ONB_QUEUE_CAP) {
        memmove(&ctx->queue[0], &ctx->queue[1],
                (size_t)(ONB_QUEUE_CAP - 1) * sizeof(onb_pending_t));
        ctx->queue_len = ONB_QUEUE_CAP - 1;
    }

    onb_pending_t *entry = &ctx->queue[ctx->queue_len++];
    entry->layer = layer;
    int copy_dim = ctx->layers[layer].dim < ONB_MAX_DIM
                 ? ctx->layers[layer].dim : ONB_MAX_DIM;
    memcpy(entry->residual, residual, (size_t)copy_dim * sizeof(float));
    ctx->total_rejections++;
    return 0;
}

/* --------------------------------------------------------------------------
 * Applying pending updates
 * --------------------------------------------------------------------------
 */

int onb_apply_pending(onb_ctx_t *ctx)
{
    if (!ctx || ctx->queue_len == 0) return 0;

    /* Gate: require minimum rejections to avoid updating on noise */
    if (ctx->total_rejections < (uint64_t)ctx->min_rejections_before_update)
        return 0;

    int updates = 0;
    for (int qi = 0; qi < ctx->queue_len; qi++) {
        const onb_pending_t *entry = &ctx->queue[qi];
        int l = entry->layer;
        if (l < 0 || l >= ctx->n_layers) continue;

        onb_layer_state_t *ls = &ctx->layers[l];

        /* L2-normalise the residual before presenting to Oja */
        float norm_sq = 0.0f;
        for (int i = 0; i < ls->dim; i++)
            norm_sq += entry->residual[i] * entry->residual[i];
        if (norm_sq < 1e-12f) continue; /* zero residual, skip */

        float inv_norm = 1.0f / sqrtf(norm_sq);
        float *x_norm = (float *)ONB_ALLOC((size_t)ls->dim * sizeof(float));
        if (!x_norm) continue;
        for (int i = 0; i < ls->dim; i++)
            x_norm[i] = entry->residual[i] * inv_norm;

        for (int rep = 0; rep < ctx->updates_per_rejection; rep++)
            oja_update_k(ls, x_norm);

        ONB_FREE(x_norm);

        ls->active = 1;
        ls->version++;
        ctx->basis_version[l] = ls->version;
        ctx->total_updates++;
        updates++;
    }

    ctx->queue_len = 0;
    return updates;
}

void onb_oja_update(onb_layer_state_t *ls, const float *x)
{
    oja_update_k(ls, x);
    ls->version++;
}

/* --------------------------------------------------------------------------
 * GP weight re-projection
 * --------------------------------------------------------------------------
 */

void onb_reproject_weight(const float *W_orig, float *W_proj,
                          const float *basis,
                          int m, int dim, int k)
{
    /*
     * W_proj[m x k] = W_orig[m x dim] @ basis^T[dim x k]
     * i.e. W_proj[r][c] = dot(W_orig[r], basis[c])
     *
     * basis is [k x dim] row-major, so basis[c][d] = basis[c*dim + d].
     */
    for (int r = 0; r < m; r++) {
        const float *row = W_orig + (size_t)r * dim;
        float *out       = W_proj + (size_t)r * k;
        for (int c = 0; c < k; c++) {
            const float *bc = basis + (size_t)c * dim;
            float s = 0.0f;
#ifdef __AVX2__
            int d = 0;
            __m256 acc = _mm256_setzero_ps();
            for (; d + 8 <= dim; d += 8) {
                __m256 a = _mm256_loadu_ps(row + d);
                __m256 b = _mm256_loadu_ps(bc  + d);
                acc = _mm256_fmadd_ps(a, b, acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            s = _mm_cvtss_f32(lo);
            for (; d < dim; d++) s += row[d] * bc[d];
#else
            for (int d = 0; d < dim; d++) s += row[d] * bc[d];
#endif
            out[c] = s;
        }
    }
}

/* --------------------------------------------------------------------------
 * Statistics
 * --------------------------------------------------------------------------
 */

void onb_get_stats(const onb_ctx_t *ctx, onb_stats_t *stats)
{
    if (!ctx || !stats) return;
    memset(stats, 0, sizeof(*stats));
    stats->total_rejections = ctx->total_rejections;
    stats->total_updates    = ctx->total_updates;
    for (int l = 0; l < ctx->n_layers; l++)
        if (ctx->layers[l].active) stats->layers_updated++;
}

void onb_print_stats(const onb_ctx_t *ctx)
{
    if (!ctx) return;
    onb_stats_t s;
    onb_get_stats(ctx, &s);
    printf("[ONB] rejections=%llu updates=%llu layers_updated=%d\n",
           (unsigned long long)s.total_rejections,
           (unsigned long long)s.total_updates,
           s.layers_updated);
}
