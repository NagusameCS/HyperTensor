/*
 * Gauge-Optimal Residual Stream Compression — Implementation
 *
 * See axiom_gauge.h for theory and API documentation.
 *
 * Implementation summary
 * ──────────────────────
 * Newton iteration for optimal diagonal gauge.
 *
 * Objective: minimise total ORIGINAL reconstruction error across all
 * compressible matrices.  For a READ matrix W [m×d] with gauge g:
 *   error(W,g) = ‖(W_gauged - trunc_r(W_gauged)) · diag(g)‖²_F
 *              = Σ_i g_i² · col_tail_gauged_i
 * For a WRITE matrix W [d×n]:
 *   error(W,g) = ‖diag(1/g) · (W_gauged - trunc_r(W_gauged))‖²_F
 *              = Σ_i (1/g_i²) · row_tail_gauged_i
 *
 * Total cost: F(g) = Σ_i [ g_i² · A_gauged_i + (1/g_i²) · B_gauged_i ]
 *   where A_gauged_i = Σ_{reads}  col_tail_gauged_i
 *         B_gauged_i = Σ_{writes} row_tail_gauged_i
 *
 * Analytic minimum (treating SVDs fixed):
 *   g*_i = (B_gauged_i / A_gauged_i)^{1/4}
 *
 * Algorithm (Newton fixed-point iteration):
 *   1. For every read matrix W:  apply diag(1/g), SVD, A_i += col_tail.
 *   2. For every write matrix W: apply diag(g),   SVD, B_i += row_tail.
 *   3. g ← (B/A)^{1/4}  (direct Newton step — exact minimum given SVDs).
 *   4. Normalise: g ← g / geomean(g)
 *   Repeat until convergence (typically 3–5 iterations).
 */

#include "runtime/nn/axiom_gauge.h"
#include "runtime/nn/axiom_exploit.h"
#include "runtime/nn/axiom_linalg.h"
#include "runtime/nn/llm.h"
#include "runtime/nn/gguf.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ── CBLAS (same guard as axiom_exploit.c) ─────────────────────────────── */
#if defined(__linux__) || defined(__APPLE__)
#  include <cblas.h>
#  define GAUGE_HAVE_CBLAS 1
#elif defined(_WIN32) || defined(_WIN64)
#  if __has_include("openblas/include/cblas.h")
#    include "openblas/include/cblas.h"
#    define GAUGE_HAVE_CBLAS 1
#  else
#    define GAUGE_HAVE_CBLAS 0
#  endif
#else
#  define GAUGE_HAVE_CBLAS 0
#endif

/* ── Forward declarations ──────────────────────────────────────────────── */
/* In kernel / backend — lightweight heap used throughout runtime. */
extern void *tensor_alloc(uint64_t bytes);
extern void  tensor_free(void *ptr);

/* ── Internal helpers ──────────────────────────────────────────────────── */

/* Matrix-vector: y[m] = A[m×n] * x[n]  (row-major, naive fallback) */
static void mat_vec(const float *A, const float *x, float *y, int m, int n)
{
#if GAUGE_HAVE_CBLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n,
                1.0f, A, n, x, 1, 0.0f, y, 1);
#else
    for (int i = 0; i < m; i++) {
        float s = 0.0f;
        for (int j = 0; j < n; j++) s += A[i * n + j] * x[j];
        y[i] = s;
    }
#endif
}

/* ─────────────────────────────────────────────────────────────────────────
 * Accumulate tail energies for one matrix into accum[dim].
 *
 * For READ matrices  (is_write=0): W [m×dim], gauge-scaled as W·diag(1/g).
 *   accum[i] += col_tail_gauged[i]  (→ A accumulator)
 * For WRITE matrices (is_write=1): W [dim×n], gauge-scaled as diag(g)·W.
 *   accum[i] += row_tail_gauged[i]  (→ B accumulator)
 *
 * These are the raw gauged tail energies used by the Newton formula:
 *   g*_i = (B[i] / A[i])^{1/4}
 *
 * Returns 0 on success, -1 on OOM.
 * ─────────────────────────────────────────────────────────────────────────*/
static int gauge_accum(const float *W_orig, int rows, int cols,
                       int is_write,        /* 0=read, 1=write */
                       const float *g,      /* current diagonal gauge [dim] */
                       int dim,             /* residual stream dimension */
                       int rank,            /* SVD rank budget */
                       float *accum,        /* tail energy accumulator [dim] */
                       float *W_scratch)    /* caller-provided work buffer [rows×cols] */
{
    /* Apply gauge in-place on the scratch buffer. */
    memcpy(W_scratch, W_orig, (uint64_t)rows * cols * sizeof(float));

    if (!is_write) {
        /* READ: scale column i by 1/g[i].  dim == cols. */
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                W_scratch[i * cols + j] /= g[j];
    } else {
        /* WRITE: scale row i by g[i].  dim == rows. */
        for (int i = 0; i < rows; i++) {
            float gi = g[i];
            for (int j = 0; j < cols; j++)
                W_scratch[i * cols + j] *= gi;
        }
    }

    int eff_rank = rank;
    if (eff_rank > rows) eff_rank = rows;
    if (eff_rank > cols) eff_rank = cols;
    if (eff_rank < 1)    eff_rank = 1;

    float *U_buf  = (float *)tensor_alloc((uint64_t)rows * eff_rank * sizeof(float));
    float *S_buf  = (float *)tensor_alloc((uint64_t)eff_rank * sizeof(float));
    float *Vt_buf = (float *)tensor_alloc((uint64_t)eff_rank * cols * sizeof(float));
    if (!U_buf || !S_buf || !Vt_buf) {
        if (U_buf)  tensor_free(U_buf);
        if (S_buf)  tensor_free(S_buf);
        if (Vt_buf) tensor_free(Vt_buf);
        return -1;
    }

    axex_svd_pub(W_scratch, rows, cols, eff_rank, U_buf, S_buf, Vt_buf);

    if (!is_write) {
        /*
         * READ: accumulate col_tail_gauged[i] into A accumulator.
         *   A[i] += ‖X[:,i]‖² - Σ_k S_k²·Vt[k,i]²
         * where X = W_scratch (= W_orig · diag(1/g)).
         */
        for (int i = 0; i < dim; i++) {
            float col_sq = 0.0f;
            for (int j = 0; j < rows; j++) {
                float v = W_scratch[j * cols + i];
                col_sq += v * v;
            }
            float cap_sq = 0.0f;
            for (int k = 0; k < eff_rank; k++) {
                float vki = Vt_buf[k * cols + i];
                cap_sq += S_buf[k] * S_buf[k] * vki * vki;
            }
            accum[i] += (col_sq - cap_sq);
        }
    } else {
        /*
         * WRITE: accumulate row_tail_gauged[i] into B accumulator.
         *   B[i] += ‖X[i,:]‖² - Σ_k S_k²·U[i,k]²
         * where X = W_scratch (= diag(g) · W_orig).
         */
        for (int i = 0; i < dim; i++) {
            float row_sq = 0.0f;
            for (int j = 0; j < cols; j++) {
                float v = W_scratch[i * cols + j];
                row_sq += v * v;
            }
            float cap_sq = 0.0f;
            for (int k = 0; k < eff_rank; k++) {
                float uik = U_buf[i * eff_rank + k];
                cap_sq += S_buf[k] * S_buf[k] * uik * uik;
            }
            accum[i] += (row_sq - cap_sq);
        }
    }

    tensor_free(U_buf);
    tensor_free(S_buf);
    tensor_free(Vt_buf);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Compute total tail energy for logging / convergence reporting.
 * ─────────────────────────────────────────────────────────────────────────*/
static double tail_energy_one(const float *W, int m, int n, int rank,
                              const float *g, int dim, int is_write,
                              float *W_scratch)
{
    memcpy(W_scratch, W, (uint64_t)m * n * sizeof(float));
    if (!is_write) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                W_scratch[i * n + j] /= g[j];
    } else {
        for (int i = 0; i < m; i++) {
            float gi = g[i];
            for (int j = 0; j < n; j++)
                W_scratch[i * n + j] *= gi;
        }
    }
    int r = rank;
    if (r > m) r = m; if (r > n) r = n; if (r < 1) r = 1;
    float *U  = (float *)tensor_alloc((uint64_t)m * r * sizeof(float));
    float *S  = (float *)tensor_alloc((uint64_t)r * sizeof(float));
    float *Vt = (float *)tensor_alloc((uint64_t)r * n * sizeof(float));
    if (!U || !S || !Vt) { if(U) tensor_free(U); if(S) tensor_free(S); if(Vt) tensor_free(Vt); return 0.0; }

    axex_svd_pub(W_scratch, m, n, r, U, S, Vt);

    double total = 0.0, captured = 0.0;
    for (uint64_t i = 0; i < (uint64_t)m * n; i++) total += (double)W_scratch[i] * W_scratch[i];
    for (int k = 0; k < r; k++) captured += (double)S[k] * S[k];
    tensor_free(U); tensor_free(S); tensor_free(Vt);
    return (total > captured) ? total - captured : 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API — axex_gauge_optimize
 * ═══════════════════════════════════════════════════════════════════════════ */

axex_gauge_t *axex_gauge_optimize(const llm_model_t *model,
                                   int rank, int n_iter)
{
    if (!model || rank < 1) return NULL;

    int dim  = model->dim;
    int nl   = model->n_layers;
    if (dim <= 0 || nl <= 0) return NULL;

    /* For large models, cap iterations to keep wall time reasonable.
     * dim > 1024 or ff > 2048 → single-step Newton (n_iter = 1). */
    int max_ff = model->ff_dim;
    for (int l = 0; l < nl; l++) {
        int lff = model->layers[l].ff_dim_layer;
        if (lff > max_ff) max_ff = lff;
    }
    if ((dim > 1024 || max_ff > 2048) && n_iter > 3) {
        printf("[AXEX-GAUGE] large model (dim=%d ff=%d): clamping n_iter %d → 3\n",
               dim, max_ff, n_iter);
        n_iter = 3;
    }
    if (n_iter < 1) n_iter = 1;

    /* Allocate output gauge and working arrays. */
    axex_gauge_t *gauge = (axex_gauge_t *)tensor_alloc(sizeof(axex_gauge_t));
    if (!gauge) return NULL;
    memset(gauge, 0, sizeof(*gauge));
    gauge->dim = dim;
    gauge->g   = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    if (!gauge->g) { axex_gauge_free(gauge); return NULL; }

    /* Initialise g = ones. */
    for (int i = 0; i < dim; i++) gauge->g[i] = 1.0f;

    /* Newton accumulator arrays: A (read tails) and B (write tails). */
    float *A_accum = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *B_accum = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    if (!A_accum || !B_accum) {
        tensor_free(A_accum); tensor_free(B_accum);
        axex_gauge_free(gauge);
        return NULL;
    }

    /* Pre-allocate the largest dequant scratch buffer. */
    uint64_t max_elems = 0;
    for (int l = 0; l < nl; l++) {
        int lff = (model->layers[l].ff_dim_layer > 0)
                  ? model->layers[l].ff_dim_layer : model->ff_dim;
        int hd  = (model->layers[l].head_dim_layer > 0)
                  ? model->layers[l].head_dim_layer : model->head_dim;
        int lq  = model->n_heads * hd;
        /* Largest matrix per layer: FFN gate/up = lff × dim */
        uint64_t sz = (uint64_t)(lff > lq ? lff : lq) * dim;
        if (sz > max_elems) max_elems = sz;
    }
    float *W_scratch = (max_elems > 0)
                       ? (float *)tensor_alloc(max_elems * sizeof(float)) : NULL;
    if (!W_scratch && max_elems > 0) {
        tensor_free(A_accum); tensor_free(B_accum);
        axex_gauge_free(gauge);
        return NULL;
    }

    double tail_first = -1.0;

    for (int iter = 0; iter < n_iter; iter++) {
        memset(A_accum, 0, (uint64_t)dim * sizeof(float));
        memset(B_accum, 0, (uint64_t)dim * sizeof(float));

        int skipped = 0;
        for (int l = 0; l < nl; l++) {
            const llm_layer_t *ly = &model->layers[l];
            if (!ly->q_weight) continue;

            int lff = (ly->ff_dim_layer > 0) ? ly->ff_dim_layer : model->ff_dim;
            int hd  = (ly->head_dim_layer > 0) ? ly->head_dim_layer : model->head_dim;
            int lq  = model->n_heads * hd;

            /* ── READ matrices ─────────────────────────────────────────── */
            /* Q: [lq × dim] */
            if (ly->q_weight && lq > 0 && dim > 0) {
                uint64_t need = (uint64_t)lq * dim;
                float *Wf = (need <= max_elems) ? W_scratch
                            : (float *)tensor_alloc(need * sizeof(float));
                if (Wf) {
                    int ok = 1;
                    for (int r = 0; r < lq && ok; r++) {
                        size_t rb = ggml_tensor_size(ly->q_type, (uint64_t)dim);
                        if (ax_dequant_row_f32(
                                (const char *)ly->q_weight + (uint64_t)r * rb,
                                Wf + (uint64_t)r * dim, dim, ly->q_type) != 0) ok = 0;
                    }
                    if (ok)
                        gauge_accum(Wf, lq, dim, 0, gauge->g, dim, rank, A_accum, Wf);
                    if (Wf != W_scratch) tensor_free(Wf);
                }
            }

            /* K: [kv_dim × dim] */
            if (ly->k_weight) {
                int kv_hd = hd; /* head_dim same for K */
                int n_kv  = (model->n_kv_heads > 0) ? model->n_kv_heads : model->n_heads;
                int kv_dim = n_kv * kv_hd;
                if (kv_dim > 0 && dim > 0) {
                    uint64_t need = (uint64_t)kv_dim * dim;
                    float *Wf = (need <= max_elems) ? W_scratch
                                : (float *)tensor_alloc(need * sizeof(float));
                    if (Wf) {
                        int ok = 1;
                        for (int r = 0; r < kv_dim && ok; r++) {
                            size_t rb = ggml_tensor_size(ly->k_type, (uint64_t)dim);
                            if (ax_dequant_row_f32(
                                    (const char *)ly->k_weight + (uint64_t)r * rb,
                                    Wf + (uint64_t)r * dim, dim, ly->k_type) != 0) ok = 0;
                        }
                        if (ok)
                            gauge_accum(Wf, kv_dim, dim, 0, gauge->g, dim, rank, A_accum, Wf);
                        if (Wf != W_scratch) tensor_free(Wf);
                    }
                }
            }

            /* FFN gate: [lff × dim] */
            if (ly->ffn_gate && lff > 0 && dim > 0) {
                uint64_t need = (uint64_t)lff * dim;
                float *Wf = (need <= max_elems) ? W_scratch
                            : (float *)tensor_alloc(need * sizeof(float));
                if (Wf) {
                    int ok = 1;
                    for (int r = 0; r < lff && ok; r++) {
                        size_t rb = ggml_tensor_size(ly->gate_type, (uint64_t)dim);
                        if (ax_dequant_row_f32(
                                (const char *)ly->ffn_gate + (uint64_t)r * rb,
                                Wf + (uint64_t)r * dim, dim, ly->gate_type) != 0) ok = 0;
                    }
                    if (ok)
                        gauge_accum(Wf, lff, dim, 0, gauge->g, dim, rank, A_accum, Wf);
                    if (Wf != W_scratch) tensor_free(Wf);
                }
            }

            /* FFN up: [lff × dim] */
            if (ly->ffn_up && lff > 0 && dim > 0) {
                uint64_t need = (uint64_t)lff * dim;
                float *Wf = (need <= max_elems) ? W_scratch
                            : (float *)tensor_alloc(need * sizeof(float));
                if (Wf) {
                    int ok = 1;
                    for (int r = 0; r < lff && ok; r++) {
                        size_t rb = ggml_tensor_size(ly->up_type, (uint64_t)dim);
                        if (ax_dequant_row_f32(
                                (const char *)ly->ffn_up + (uint64_t)r * rb,
                                Wf + (uint64_t)r * dim, dim, ly->up_type) != 0) ok = 0;
                    }
                    if (ok)
                        gauge_accum(Wf, lff, dim, 0, gauge->g, dim, rank, A_accum, Wf);
                    if (Wf != W_scratch) tensor_free(Wf);
                }
            }

            /* ── WRITE matrices ────────────────────────────────────────── */
            /* O: [dim × lq] */
            if (ly->o_weight && lq > 0 && dim > 0) {
                uint64_t need = (uint64_t)dim * lq;
                float *Wf = (need <= max_elems) ? W_scratch
                            : (float *)tensor_alloc(need * sizeof(float));
                if (Wf) {
                    int ok = 1;
                    for (int r = 0; r < dim && ok; r++) {
                        size_t rb = ggml_tensor_size(ly->o_type, (uint64_t)lq);
                        if (ax_dequant_row_f32(
                                (const char *)ly->o_weight + (uint64_t)r * rb,
                                Wf + (uint64_t)r * lq, lq, ly->o_type) != 0) ok = 0;
                    }
                    if (ok)
                        gauge_accum(Wf, dim, lq, 1, gauge->g, dim, rank, B_accum, Wf);
                    if (Wf != W_scratch) tensor_free(Wf);
                }
            }

            /* FFN down: [dim × lff] */
            if (ly->ffn_down && lff > 0 && dim > 0) {
                uint64_t need = (uint64_t)dim * lff;
                float *Wf = (need <= max_elems) ? W_scratch
                            : (float *)tensor_alloc(need * sizeof(float));
                if (Wf) {
                    int ok = 1;
                    for (int r = 0; r < dim && ok; r++) {
                        size_t rb = ggml_tensor_size(ly->down_type, (uint64_t)lff);
                        if (ax_dequant_row_f32(
                                (const char *)ly->ffn_down + (uint64_t)r * rb,
                                Wf + (uint64_t)r * lff, lff, ly->down_type) != 0) ok = 0;
                    }
                    if (ok)
                        gauge_accum(Wf, dim, lff, 1, gauge->g, dim, rank, B_accum, Wf);
                    if (Wf != W_scratch) tensor_free(Wf);
                }
            }

            (void)skipped;
        } /* end layer loop */

        /* ── Newton step on the log-space objective ─────────────────── */
        /*
         * Objective: F_i(g_i) = g_i² · A_gauged_i  +  B_gauged_i / g_i²
         *
         * Newton step in log space:
         *   delta_i = -(g²·A - B/g²) / (2·(g²·A + B/g²))
         *   |delta_i| ≤ 0.5  (bounded Newton)
         *
         * We project onto the normalisation constraint (Σ log g = 0) by
         * subtracting the mean delta, then clamp the PROJECTED step to
         * ±0.35 to prevent divergence caused by mean-subtraction amplifying
         * individual large steps.
         */
        const float eps_tail = 1e-30f;

        /* Pass 1: compute raw Newton deltas and their mean. */
        float mean_delta = 0.0f;
        {
            for (int i = 0; i < dim; i++) {
                float gi   = gauge->g[i];
                float gi2  = gi * gi;
                float a    = A_accum[i];
                float b    = B_accum[i];
                float den  = 2.0f * (gi2 * a + b / gi2);
                float d    = (den < eps_tail) ? 0.0f : -(gi2 * a - b / gi2) / den;
                /* Clamp raw Newton step to [-0.5, 0.5]. */
                if (d < -0.5f) d = -0.5f;
                if (d >  0.5f) d =  0.5f;
                mean_delta += d;
            }
            mean_delta /= (float)dim;
        }

        /* Pass 2: apply projected step and update g. */
        double sum_log = 0.0;
        for (int i = 0; i < dim; i++) {
            float gi   = gauge->g[i];
            float gi2  = gi * gi;
            float a    = A_accum[i];
            float b    = B_accum[i];
            float den  = 2.0f * (gi2 * a + b / gi2);
            float d    = (den < eps_tail) ? 0.0f : -(gi2 * a - b / gi2) / den;
            if (d < -0.5f) d = -0.5f;
            if (d >  0.5f) d =  0.5f;
            /* Projected step (zero-mean) → clamp to ±0.35 per iter. */
            float proj = d - mean_delta;
            if (proj < -0.35f) proj = -0.35f;
            if (proj >  0.35f) proj =  0.35f;
            float g_new = gi * expf(proj);
            gauge->g[i] = g_new;
            sum_log += logf(g_new > 1e-20f ? g_new : 1e-20f);
        }

        /* ── Normalise (geometric mean = 1) ─────────────────────────── */
        float mean_log = (float)(sum_log / dim);
        float geomean_scale = expf(-mean_log);
        for (int i = 0; i < dim; i++) gauge->g[i] *= geomean_scale;

        /* ── Progress report ─────────────────────────────────────────── */
        {
            float g_min = gauge->g[0], g_max = gauge->g[0];
            for (int i = 1; i < dim; i++) {
                if (gauge->g[i] < g_min) g_min = gauge->g[i];
                if (gauge->g[i] > g_max) g_max = gauge->g[i];
            }
            /* Newton gradient residual (should → 0 on convergence) */
            double grad_rms = 0.0;
            for (int i = 0; i < dim; i++) {
                float gi = gauge->g[i];
                float gi2 = gi * gi;
                float a = A_accum[i], b = B_accum[i];
                float grad_i = 2.0f * gi2 * a - 2.0f * b / gi2;
                grad_rms += (double)grad_i * grad_i;
            }
            grad_rms = sqrt(grad_rms / dim);
            printf("[AXEX-GAUGE] iter %d/%d  |Newton_grad|_rms=%.3e  "
                   "g_range=[%.3f, %.3f]\n",
                   iter + 1, n_iter, grad_rms, g_min, g_max);
            if (iter == 0) tail_first = grad_rms;
        }
    } /* end iteration loop */

    /* ── Summary ─────────────────────────────────────────────────────── */
    gauge->tail_before = tail_first > 0.0 ? tail_first : 0.0;
    gauge->tail_after  = 0.0; /* could compute exactly, but expensive */
    gauge->gain_pct    = 0.0f;

    printf("[AXEX-GAUGE] Optimal diagonal gauge found (dim=%d, rank=%d, "
           "iters=%d)\n", dim, rank, n_iter);
    {
        float g_min = gauge->g[0], g_max = gauge->g[0];
        double sum_log = 0.0;
        for (int i = 0; i < dim; i++) {
            if (gauge->g[i] < g_min) g_min = gauge->g[i];
            if (gauge->g[i] > g_max) g_max = gauge->g[i];
            sum_log += logf(gauge->g[i]);
        }
        printf("[AXEX-GAUGE] g_range=[%.4f, %.4f]  geomean=%.6f\n",
               g_min, g_max, (float)exp(sum_log / dim));
    }

    if (W_scratch) tensor_free(W_scratch);
    tensor_free(A_accum);
    tensor_free(B_accum);

    return gauge;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Apply / Bake helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

void axex_gauge_apply_read(float *W, int m, int dim, const float *g)
{
    if (!W || !g) return;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < dim; j++)
            W[i * dim + j] /= g[j];
}

void axex_gauge_apply_write(float *W, int dim, int n, const float *g)
{
    if (!W || !g) return;
    for (int i = 0; i < dim; i++) {
        float gi = g[i];
        for (int j = 0; j < n; j++)
            W[i * n + j] *= gi;
    }
}

void axex_gauge_bake_vt(float *Vt, int rank, int dim, const float *g)
{
    if (!Vt || !g) return;
    /* Vt[k × dim]: scale column j by g[j]. */
    for (int k = 0; k < rank; k++)
        for (int j = 0; j < dim; j++)
            Vt[k * dim + j] *= g[j];
}

void axex_gauge_bake_u(float *U, int dim, int rank, const float *g)
{
    if (!U || !g) return;
    /* U[dim × rank]: scale row i by 1/g[i]. */
    for (int i = 0; i < dim; i++) {
        float inv_gi = 1.0f / g[i];
        for (int k = 0; k < rank; k++)
            U[i * rank + k] *= inv_gi;
    }
}

void axex_gauge_free(axex_gauge_t *gauge)
{
    if (!gauge) return;
    if (gauge->g) tensor_free(gauge->g);
    tensor_free(gauge);
}
