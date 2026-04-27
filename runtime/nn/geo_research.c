/*
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
 * ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
 * ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
 * ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
 * ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
 * ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
 * :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
 * :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
 * ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
 * :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
 * ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
 * ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
 * :::::::::................................:@@@@@@@@@@%:...............................::::::
 * ::::::::..................................*@@@@@@@@@-................................::::::::
 * ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
 * :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
 * :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
 * :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
 * :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
 * :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
 * :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
 * :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
 * :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
 * :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
 * ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
 * ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
 * :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
 * ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
 * :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
 * :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
 * ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
 * ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
 * :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
 * ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
 * ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
 * :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */

/*
 * geo_research.c - Real Ricci signal, basis transfer, phase-conditional
 *                  compression, differentiable plan selection.
 *
 * See geo_research.h for design documentation.
 */

#include "runtime/nn/geo_research.h"
#include "runtime/nn/axiom_geo.h"
#include "runtime/nn/axiom_linalg.h"
#include "runtime/nn/axiom_exploit.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#ifdef GEODESSICAL_HOSTED
#  include "host/hal.h"
#  define GR_ALLOC(n)   malloc(n)
#  define GR_FREE(p)    free(p)
#else
#  include "kernel/core/kernel.h"
#  include "kernel/mm/tensor_mm.h"
#  define GR_ALLOC(n)   tensor_alloc(n)
#  define GR_FREE(p)    tensor_free(p)
#endif

/* =========================================================================
 * Utility
 * ========================================================================= */

static float pearson_r_d(const double *x, const double *y, int n)
{
    if (n < 2) return 0.0f;
    double mx = 0.0, my = 0.0;
    for (int i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
    mx /= n; my /= n;
    double cov = 0.0, vx = 0.0, vy = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - mx, dy = y[i] - my;
        cov += dx * dy; vx += dx * dx; vy += dy * dy;
    }
    double denom = sqrt(vx * vy);
    return (denom < 1e-20) ? 0.0f : (float)(cov / denom);
}

static void softmax_inplace(float *v, int n)
{
    float mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { v[i] = expf(v[i] - mx); sum += v[i]; }
    if (sum > 0.0f) for (int i = 0; i < n; i++) v[i] /= sum;
}

/* =========================================================================
 * Feature 1: True sectional curvature compression signal
 *
 * For each layer:
 *   1. Build a Riemannian metric field from the sample covariance inverses.
 *   2. Compute global Christoffel symbols via axgeo_compute_christoffel_global.
 *   3. Contract to Ricci / scalar curvature via axgeo_compute_curvature.
 *   4. Record K_mean = R / (d*(d-1)) as the sectional curvature proxy.
 * ========================================================================= */

/* Local covariance from k nearest neighbours */
static void local_covariance(const double *pts, int n, int d, int i,
                              double *cov_out, int k_nn)
{
    double *dists = (double *)GR_ALLOC((size_t)n * sizeof(double));
    int    *order = (int    *)GR_ALLOC((size_t)n * sizeof(int));
    if (!dists || !order) { GR_FREE(dists); GR_FREE(order); return; }

    for (int j = 0; j < n; j++) {
        order[j] = j;
        double d2 = 0.0;
        for (int c = 0; c < d; c++) {
            double diff = pts[(size_t)i*d+c] - pts[(size_t)j*d+c];
            d2 += diff * diff;
        }
        dists[j] = d2;
    }
    for (int a = 0; a < k_nn && a < n; a++) {
        int min_j = a;
        for (int b = a+1; b < n; b++)
            if (dists[order[b]] < dists[order[min_j]]) min_j = b;
        int tmp = order[a]; order[a] = order[min_j]; order[min_j] = tmp;
    }

    memset(cov_out, 0, (size_t)d * d * sizeof(double));
    double mean[AXGEO_MAX_DIM];
    memset(mean, 0, (size_t)d * sizeof(double));
    int k = (k_nn < n) ? k_nn : n;
    for (int a = 0; a < k; a++) {
        int j = order[a];
        for (int c = 0; c < d; c++) mean[c] += pts[(size_t)j*d+c];
    }
    for (int c = 0; c < d; c++) mean[c] /= k;
    for (int a = 0; a < k; a++) {
        int j = order[a];
        for (int r = 0; r < d; r++) {
            double dr = pts[(size_t)j*d+r] - mean[r];
            for (int c = 0; c < d; c++)
                cov_out[(size_t)r*d+c] += dr * (pts[(size_t)j*d+c] - mean[c]);
        }
    }
    double inv_k = 1.0 / k;
    for (int r = 0; r < d*d; r++) cov_out[r] *= inv_k;

    GR_FREE(dists);
    GR_FREE(order);
}

/* Ridge-regularised matrix inversion (Gauss-Jordan in place).
 * mat is destroyed; result goes to inv.  Returns 0 on success, -1 if singular. */
static int ridge_invert(double *mat, double *inv, int d, double eps)
{
    for (int i = 0; i < d; i++) {
        mat[(size_t)i*d+i] += eps;
        for (int j = 0; j < d; j++)
            inv[(size_t)i*d+j] = (i == j) ? 1.0 : 0.0;
    }
    for (int col = 0; col < d; col++) {
        int piv = col;
        for (int r = col+1; r < d; r++)
            if (fabs(mat[(size_t)r*d+col]) > fabs(mat[(size_t)piv*d+col])) piv = r;
        for (int c = 0; c < d; c++) {
            double tmp;
            tmp = mat[(size_t)col*d+c]; mat[(size_t)col*d+c] = mat[(size_t)piv*d+c]; mat[(size_t)piv*d+c] = tmp;
            tmp = inv[(size_t)col*d+c]; inv[(size_t)col*d+c] = inv[(size_t)piv*d+c]; inv[(size_t)piv*d+c] = tmp;
        }
        double diag = mat[(size_t)col*d+col];
        if (fabs(diag) < 1e-30) return -1;
        for (int c = 0; c < d; c++) {
            mat[(size_t)col*d+c] /= diag;
            inv[(size_t)col*d+c] /= diag;
        }
        for (int r = 0; r < d; r++) {
            if (r == col) continue;
            double f = mat[(size_t)r*d+col];
            for (int c = 0; c < d; c++) {
                mat[(size_t)r*d+c] -= f * mat[(size_t)col*d+c];
                inv[(size_t)r*d+c] -= f * inv[(size_t)col*d+c];
            }
        }
    }
    return 0;
}

int grcurv_compute(const grcurv_layer_input_t *inputs,
                   int n_layers,
                   grcurv_per_layer_t *out)
{
    if (!inputs || n_layers <= 0 || !out) return -1;
    if (n_layers > GRCURV_MAX_LAYERS) n_layers = GRCURV_MAX_LAYERS;

    memset(out, 0, sizeof(*out));
    out->n_layers = n_layers;

    for (int l = 0; l < n_layers; l++) {
        const grcurv_layer_input_t *inp = &inputs[l];
        out->proxy[l] = inp->proxy_ricci;

        if (!inp->samples || inp->n_samples < 4 || inp->dim < 2) continue;

        int n      = inp->n_samples;
        int full_d = inp->dim;
        int d      = full_d < AXGEO_MAX_DIM ? full_d : AXGEO_MAX_DIM;

        double *pts = (double *)GR_ALLOC((size_t)n * d * sizeof(double));
        if (!pts) continue;
        for (int i = 0; i < n; i++)
            for (int c = 0; c < d; c++)
                pts[(size_t)i*d+c] = (double)inp->samples[(size_t)i*full_d+c];

        axgeo_metric_field_t mf = axgeo_metric_field_create(n, d);
        if (!mf.points || !mf.metrics) { GR_FREE(pts); continue; }

        double *cov  = (double *)GR_ALLOC((size_t)d * d * sizeof(double));
        double *inv  = (double *)GR_ALLOC((size_t)d * d * sizeof(double));
        double *work = (double *)GR_ALLOC((size_t)d * d * sizeof(double));
        if (!cov || !inv || !work) {
            GR_FREE(cov); GR_FREE(inv); GR_FREE(work);
            GR_FREE(pts); axgeo_metric_field_destroy(&mf);
            continue;
        }

        int k_nn = n < 8 ? n : 8;
        for (int i = 0; i < n; i++) {
            memcpy(axgeo_point_at(&mf, i), pts + (size_t)i*d, (size_t)d*sizeof(double));
            local_covariance(pts, n, d, i, cov, k_nn);
            memcpy(work, cov, (size_t)d*d*sizeof(double));
            double *g = axgeo_metric_at(&mf, i);
            if (ridge_invert(work, inv, d, 0.01) == 0)
                memcpy(g, inv, (size_t)d*d*sizeof(double));
            else {
                memset(g, 0, (size_t)d*d*sizeof(double));
                for (int j = 0; j < d; j++) g[(size_t)j*d+j] = 1.0;
            }
        }
        GR_FREE(cov); GR_FREE(inv); GR_FREE(work);

        axgeo_christoffel_t ch = axgeo_christoffel_create(1, d);
        if (!ch.gamma) { GR_FREE(pts); axgeo_metric_field_destroy(&mf); continue; }

        if (axgeo_compute_christoffel_global(&mf, &ch) == 0) {
            axgeo_curvature_t curv = axgeo_curvature_create(1, d);
            if (curv.ricci && axgeo_compute_curvature(&ch, &mf, &curv) == 0) {
                double R = curv.scalar_curv[0];
                out->scalar[l]     = R;
                out->sectional[l]  = (d >= 2) ? R / ((double)d * (d - 1)) : 0.0;
                out->ricci_diag[l] = curv.ricci[0]; /* R_{00} */
            }
            axgeo_curvature_destroy(&curv);
        }

        axgeo_christoffel_destroy(&ch);
        axgeo_metric_field_destroy(&mf);
        GR_FREE(pts);
    }

    out->correlation = pearson_r_d(out->sectional, out->proxy, n_layers);
    return 0;
}

void grcurv_to_rank_budget(const grcurv_per_layer_t *curv,
                           int min_rank, int max_rank,
                           int total_rank_budget,
                           int *ranks_out)
{
    if (!curv || !ranks_out) return;
    int n = curv->n_layers;
    if (n <= 0) return;

    const float alpha = 0.5f;
    double K_max = 0.0;
    for (int l = 0; l < n; l++) {
        double k = fabs(curv->sectional[l]);
        if (k > K_max) K_max = k;
    }

    float *raw = (float *)GR_ALLOC((size_t)n * sizeof(float));
    if (!raw) {
        int base = total_rank_budget / n;
        for (int l = 0; l < n; l++) ranks_out[l] = base;
        return;
    }

    int base_rank = total_rank_budget / n;
    float total = 0.0f;
    for (int l = 0; l < n; l++) {
        float w = (K_max > 1e-20) ? (float)(fabs(curv->sectional[l]) / K_max) : 0.0f;
        float r = (float)base_rank * (1.0f + alpha * w);
        if (r < (float)min_rank) r = (float)min_rank;
        if (r > (float)max_rank) r = (float)max_rank;
        raw[l] = r;
        total += r;
    }

    float scale = (total > 0.0f) ? (float)total_rank_budget / total : 1.0f;
    int used = 0;
    for (int l = 0; l < n - 1; l++) {
        int r = (int)roundf(raw[l] * scale);
        if (r < min_rank) r = min_rank;
        if (r > max_rank) r = max_rank;
        ranks_out[l] = r;
        used += r;
    }
    int last = total_rank_budget - used;
    if (last < min_rank) last = min_rank;
    if (last > max_rank) last = max_rank;
    ranks_out[n - 1] = last;
    GR_FREE(raw);
}

void grcurv_print_comparison(const grcurv_per_layer_t *curv)
{
    if (!curv) return;
    printf("[GRCURV] Real vs Proxy curvature (%d layers)\n", curv->n_layers);
    printf("[GRCURV] Pearson corr(sectional, proxy) = %.4f\n", curv->correlation);
    printf("[GRCURV] %4s  %12s  %12s  %12s  %12s\n",
           "Lyr", "K_mean", "R_scalar", "Ric_diag0", "Proxy");
    for (int l = 0; l < curv->n_layers; l++) {
        printf("[GRCURV] %4d  %+12.5e  %+12.5e  %+12.5e  %+12.5e\n",
               l, curv->sectional[l], curv->scalar[l],
               curv->ricci_diag[l], curv->proxy[l]);
    }
}


/* =========================================================================
 * Feature 2: Cross-model basis transfer
 * ========================================================================= */

static void align_basis_vector(const float *src, int dim_s, float *dst, int dim_l)
{
    int copy_n = dim_s < dim_l ? dim_s : dim_l;
    memcpy(dst, src, (size_t)copy_n * sizeof(float));
    if (dim_l > dim_s)
        memset(dst + dim_s, 0, (size_t)(dim_l - dim_s) * sizeof(float));
    float norm_sq = 0.0f;
    for (int i = 0; i < dim_l; i++) norm_sq += dst[i] * dst[i];
    if (norm_sq > 1e-12f) {
        float inv = 1.0f / sqrtf(norm_sq);
        for (int i = 0; i < dim_l; i++) dst[i] *= inv;
    }
}

static double explained_variance_fraction(const float *samples, int n, int dim,
                                           const float *basis, int k)
{
    if (!samples || !basis || n < 1 || k < 1) return 0.0;
    double total = 0.0, proj = 0.0;
    for (int i = 0; i < n; i++) {
        const float *s = samples + (size_t)i * dim;
        double s2 = 0.0;
        for (int d = 0; d < dim; d++) s2 += (double)s[d] * s[d];
        total += s2;
        for (int c = 0; c < k; c++) {
            const float *bc = basis + (size_t)c * dim;
            double p = 0.0;
            for (int d = 0; d < dim; d++) p += (double)s[d] * bc[d];
            proj += p * p;
        }
    }
    if (total < 1e-20) return 0.0;
    double frac = proj / total;
    return frac > 1.0 ? 1.0 : frac;
}

int xfer_basis_small_to_large(const axpca_t *small_pcas,
                               int small_n_layers,
                               int large_n_layers,
                               const int *large_layer_dims,
                               float * const *large_layer_samples,
                               int n_validation_samples,
                               double acceptance_variance_threshold,
                               xfer_result_t *result)
{
    if (!small_pcas || small_n_layers <= 0 || !large_layer_dims || !result)
        return -1;

    memset(result, 0, sizeof(*result));
    result->acceptance_threshold = acceptance_variance_threshold;

    int n_layers = large_n_layers;
    if (n_layers > XFER_MAX_LAYERS) n_layers = XFER_MAX_LAYERS;
    result->n_layers = n_layers;

    for (int l = 0; l < n_layers; l++) {
        int sl = (small_n_layers > 0) ? (l * small_n_layers / n_layers) : l;
        if (sl >= small_n_layers) sl = small_n_layers - 1;

        const axpca_t *src = &small_pcas[sl];
        if (!src->components.data || src->n_components <= 0 || src->dim <= 0) {
            result->n_rejected++;
            continue;
        }

        int dim_s = src->dim;
        int k_s   = src->n_components;
        int dim_l = large_layer_dims[l] > 0 ? large_layer_dims[l] : dim_s;
        int k_l   = k_s;

        float *basis_out = (float *)GR_ALLOC((size_t)k_l * dim_l * sizeof(float));
        if (!basis_out) { result->n_rejected++; continue; }

        /* axpca_t.components is [n_components x dim] double row-major */
        const double *src_W = src->components.data;
        for (int c = 0; c < k_l; c++) {
            /* Convert double -> float for source row, then align */
            float *tmp = (float *)GR_ALLOC((size_t)dim_s * sizeof(float));
            if (!tmp) { GR_FREE(basis_out); basis_out = 0; break; }
            const double *sv = src_W + (size_t)c * dim_s;
            for (int i = 0; i < dim_s; i++) tmp[i] = (float)sv[i];
            align_basis_vector(tmp, dim_s, basis_out + (size_t)c * dim_l, dim_l);
            GR_FREE(tmp);
        }
        if (!basis_out) { result->n_rejected++; continue; }

        double ev = 0.0;
        if (large_layer_samples && large_layer_samples[l] && n_validation_samples > 0)
            ev = explained_variance_fraction(large_layer_samples[l],
                                             n_validation_samples, dim_l,
                                             basis_out, k_l);

        result->bases[l]              = basis_out;
        result->ks[l]                 = k_l;
        result->dims[l]               = dim_l;
        result->variance_explained[l] = ev;
        result->accepted[l]           = (ev >= acceptance_variance_threshold) ? 1 : 0;
        if (result->accepted[l]) result->n_accepted++;
        else                     result->n_rejected++;
    }

    double sum_ev = 0.0; int count = 0;
    for (int l = 0; l < n_layers; l++) {
        if (result->accepted[l]) { sum_ev += result->variance_explained[l]; count++; }
    }
    result->mean_variance_explained = (count > 0) ? sum_ev / count : 0.0;
    return 0;
}

void xfer_result_destroy(xfer_result_t *result)
{
    if (!result) return;
    for (int l = 0; l < result->n_layers; l++) {
        if (result->bases[l]) { GR_FREE(result->bases[l]); result->bases[l] = 0; }
    }
    memset(result, 0, sizeof(*result));
}

void xfer_result_print(const xfer_result_t *result)
{
    if (!result) return;
    printf("[XFER] Cross-model basis transfer: %d layers accepted=%d rejected=%d threshold=%.2f\n",
           result->n_layers, result->n_accepted, result->n_rejected,
           result->acceptance_threshold);
    printf("[XFER] Mean variance explained (accepted): %.4f\n",
           result->mean_variance_explained);
    printf("[XFER] %4s  %7s  %8s  %5s  %5s\n", "Lyr", "Accept", "VarExpl", "Rank", "Dim");
    for (int l = 0; l < result->n_layers; l++) {
        printf("[XFER] %4d  %7s  %8.4f  %5d  %5d\n",
               l, result->accepted[l] ? "YES" : "NO",
               result->variance_explained[l], result->ks[l], result->dims[l]);
    }
}


/* =========================================================================
 * Feature 3: Phase-conditional compression plans
 * ========================================================================= */

int phaseplan_build(phased_plan_t *pp,
                    const axex_offload_plan_t *offload_plan,
                    int n_layers,
                    float retrieval_quality,
                    float reasoning_quality,
                    float generation_quality)
{
    if (!pp || !offload_plan || n_layers <= 0) return -1;
    if (n_layers > PHASEPLAN_MAX_LAYERS) n_layers = PHASEPLAN_MAX_LAYERS;

    memset(pp, 0, sizeof(*pp));
    pp->retrieval_entropy_min  = 1.5f;
    pp->generation_entropy_max = 0.8f;
    pp->min_tokens_per_phase   = 8;
    pp->attn_entropy_ema       = 1.5f;
    pp->current_phase          = PHASE_RETRIEVAL;

    pp->retrieval.n_layers   = n_layers;
    pp->reasoning.n_layers   = n_layers;
    pp->generation.n_layers  = n_layers;
    pp->retrieval.quality_floor  = retrieval_quality;
    pp->reasoning.quality_floor  = reasoning_quality;
    pp->generation.quality_floor = generation_quality;
    pp->retrieval.name  = "retrieval";
    pp->reasoning.name  = "reasoning";
    pp->generation.name = "generation";

    for (int l = 0; l < n_layers; l++) {
        int ol  = l < offload_plan->n_layers ? l : offload_plan->n_layers - 1;
        float pri = offload_plan->entries[ol].gpu_priority; /* 0..1 */
        int base  = (int)roundf(32.0f + 224.0f * pri);     /* [32, 256] */

        /* quality in (0,1).  rank ~ base / (1 - quality + eps) */
        int r_ret = (int)roundf((float)base / (1.05f - retrieval_quality));
        int r_rea = (int)roundf((float)base / (1.05f - reasoning_quality));
        int r_gen = (int)roundf((float)base / (1.05f - generation_quality));

        pp->retrieval.ranks[l]  = r_ret  < 8 ? 8 : (r_ret  > 512 ? 512 : r_ret);
        pp->reasoning.ranks[l]  = r_rea  < 8 ? 8 : (r_rea  > 512 ? 512 : r_rea);
        pp->generation.ranks[l] = r_gen  < 8 ? 8 : (r_gen  > 512 ? 512 : r_gen);
    }
    return 0;
}

gen_phase_t phaseplan_update(phased_plan_t *pp, float attn_entropy, float kv_merge_rate)
{
    (void)kv_merge_rate;
    if (!pp) return PHASE_UNKNOWN;

    const float alpha = 0.1f;
    pp->attn_entropy_ema = (1.0f - alpha) * pp->attn_entropy_ema + alpha * attn_entropy;
    pp->tokens_in_phase++;

    if (pp->tokens_in_phase < pp->min_tokens_per_phase) return pp->current_phase;

    gen_phase_t new_phase;
    if      (pp->attn_entropy_ema >= pp->retrieval_entropy_min) new_phase = PHASE_RETRIEVAL;
    else if (pp->attn_entropy_ema <= pp->generation_entropy_max) new_phase = PHASE_GENERATION;
    else                                                          new_phase = PHASE_REASONING;

    if (new_phase != pp->current_phase) {
        pp->current_phase   = new_phase;
        pp->tokens_in_phase = 0;
    }
    return pp->current_phase;
}

const phase_plan_t *phaseplan_active(const phased_plan_t *pp)
{
    if (!pp) return 0;
    switch (pp->current_phase) {
    case PHASE_RETRIEVAL:  return &pp->retrieval;
    case PHASE_REASONING:  return &pp->reasoning;
    case PHASE_GENERATION: return &pp->generation;
    default:               return &pp->retrieval;
    }
}

void phaseplan_print(const phased_plan_t *pp)
{
    if (!pp) return;
    const char *names[] = {"unknown", "retrieval", "reasoning", "generation"};
    printf("[PHASE] Current: %s  entropy_ema=%.3f  tokens_in_phase=%d\n",
           names[pp->current_phase], (double)pp->attn_entropy_ema, pp->tokens_in_phase);
    printf("[PHASE] Sample ranks (layer 0..9): RET");
    for (int l = 0; l < 10 && l < pp->retrieval.n_layers; l++)
        printf(" %d", pp->retrieval.ranks[l]);
    printf(" | REA");
    for (int l = 0; l < 10 && l < pp->reasoning.n_layers; l++)
        printf(" %d", pp->reasoning.ranks[l]);
    printf(" | GEN");
    for (int l = 0; l < 10 && l < pp->generation.n_layers; l++)
        printf(" %d", pp->generation.ranks[l]);
    printf("\n");
}


/* =========================================================================
 * Feature 4: Differentiable compression plan selection
 * ========================================================================= */

static void diffplan_forward(diffplan_t *dp)
{
    for (int l = 0; l < dp->n_layers; l++) {
        float probs[DIFFPLAN_N_LEVELS];
        for (int r = 0; r < dp->n_levels; r++) probs[r] = dp->theta[l][r];
        softmax_inplace(probs, dp->n_levels);
        for (int r = 0; r < dp->n_levels; r++) dp->p[l][r] = probs[r];

        float expected = 0.0f;
        for (int r = 0; r < dp->n_levels; r++)
            expected += probs[r] * (float)DIFFPLAN_RANK_LEVELS[r];
        dp->rank_soft[l] = expected;

        int best = 0; float best_p = probs[0];
        for (int r = 1; r < dp->n_levels; r++) {
            if (probs[r] > best_p) { best_p = probs[r]; best = r; }
        }
        dp->rank_hard[l] = DIFFPLAN_RANK_LEVELS[best];
    }
}

static float diffplan_loss(const diffplan_t *dp)
{
    float loss = 0.0f;
    for (int l = 0; l < dp->n_layers; l++) {
        float err = dp->frob_err[l] +
                    dp->sv_slope[l] * (dp->rank_soft[l] - (float)dp->rank_hard[l]);
        if (err < 0.0f) err = 0.0f;
        loss += err + dp->lambda * dp->rank_soft[l];
    }
    return loss;
}

static void diffplan_backward(diffplan_t *dp,
                              float grads[DIFFPLAN_MAX_LAYERS][DIFFPLAN_N_LEVELS])
{
    for (int l = 0; l < dp->n_layers; l++) {
        float dL_drank = dp->sv_slope[l] + dp->lambda;
        for (int r = 0; r < dp->n_levels; r++) {
            grads[l][r] = dL_drank * dp->p[l][r] *
                          ((float)DIFFPLAN_RANK_LEVELS[r] - dp->rank_soft[l]);
        }
    }
}

int diffplan_init(diffplan_t *dp,
                  const diffplan_layer_data_t *layer_data,
                  int n_layers,
                  float lambda,
                  float quality_floor)
{
    if (!dp || n_layers <= 0) return -1;
    if (n_layers > DIFFPLAN_MAX_LAYERS) n_layers = DIFFPLAN_MAX_LAYERS;

    memset(dp, 0, sizeof(*dp));
    dp->n_layers        = n_layers;
    dp->n_levels        = DIFFPLAN_N_LEVELS;
    dp->lr              = 0.01f;
    dp->momentum_coef   = 0.9f;
    dp->lambda          = lambda;
    dp->quality_floor   = quality_floor;
    dp->max_iter        = 200;
    dp->convergence_tol = 1e-4f;

    for (int l = 0; l < n_layers; l++) {
        for (int r = 0; r < dp->n_levels; r++) {
            dp->theta[l][r] = dp->momentum[l][r] = 0.0f;
        }
        if (layer_data) {
            dp->frob_err[l] = layer_data[l].frob_err;
            dp->sv_slope[l] = layer_data[l].sv_slope;
        } else {
            dp->frob_err[l] = 0.05f;
            dp->sv_slope[l] = -0.001f;
        }
    }

    diffplan_forward(dp);
    return 0;
}

float diffplan_step(diffplan_t *dp)
{
    diffplan_forward(dp);
    float loss = diffplan_loss(dp);

    float grads[DIFFPLAN_MAX_LAYERS][DIFFPLAN_N_LEVELS];
    diffplan_backward(dp, grads);

    for (int l = 0; l < dp->n_layers; l++) {
        for (int r = 0; r < dp->n_levels; r++) {
            dp->momentum[l][r] = dp->momentum_coef * dp->momentum[l][r]
                                + (1.0f - dp->momentum_coef) * grads[l][r];
            dp->theta[l][r] -= dp->lr * dp->momentum[l][r];
        }
    }

    if (dp->n_iter < 256) dp->loss_history[dp->n_iter] = loss;
    dp->n_iter++;
    return loss;
}

float diffplan_optimise(diffplan_t *dp, int n_iter)
{
    if (!dp) return 0.0f;
    float prev_loss = FLT_MAX, loss = 0.0f;
    for (int i = 0; i < n_iter; i++) {
        loss = diffplan_step(dp);
        if (fabsf(loss - prev_loss) < dp->convergence_tol) break;
        prev_loss = loss;
    }
    return loss;
}

void diffplan_get_ranks(const diffplan_t *dp, int *ranks_out)
{
    if (!dp || !ranks_out) return;
    for (int l = 0; l < dp->n_layers; l++) ranks_out[l] = dp->rank_hard[l];
}

void diffplan_print(const diffplan_t *dp)
{
    if (!dp) return;
    printf("[DIFFPLAN] %d layers  %d iters  lambda=%.4f\n",
           dp->n_layers, dp->n_iter, (double)dp->lambda);
    float ts = 0.0f; int th = 0;
    for (int l = 0; l < dp->n_layers; l++) { ts += dp->rank_soft[l]; th += dp->rank_hard[l]; }
    printf("[DIFFPLAN] Total rank soft=%.1f hard=%d\n", (double)ts, th);
    printf("[DIFFPLAN] Loss curve (last 8): ");
    int start = dp->n_iter > 8 ? dp->n_iter - 8 : 0;
    int cap   = dp->n_iter < 256 ? dp->n_iter : 256;
    for (int i = start; i < cap; i++) printf("%.5f ", (double)dp->loss_history[i]);
    printf("\n");
    printf("[DIFFPLAN] Hard ranks (0..19): ");
    for (int l = 0; l < dp->n_layers && l < 20; l++) printf("%d ", dp->rank_hard[l]);
    printf("\n");
}