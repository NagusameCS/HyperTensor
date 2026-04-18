/*
 * Geodessical — Axiomatic Differential Geometry Engine
 *
 * Riemannian geometry: metric tensors, Christoffel symbols, curvature
 * tensors, and geodesic integration (RK4) for neural manifold analysis.
 */

#include "runtime/nn/axiom_geo.h"
#include "runtime/nn/axiom_linalg.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef GEODESSICAL_HOSTED
#include "host/hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Metric Field
 * ═══════════════════════════════════════════════════════════════════════════ */

axgeo_metric_field_t axgeo_metric_field_create(int n_points, int dim)
{
    axgeo_metric_field_t mf;
    mf.n_points = n_points;
    mf.dim = dim;
    mf.points = (double *)tensor_alloc((uint64_t)n_points * dim * sizeof(double));
    mf.metrics = (double *)tensor_alloc((uint64_t)n_points * dim * dim * sizeof(double));
    if (mf.points) memset(mf.points, 0, (uint64_t)n_points * dim * sizeof(double));
    if (mf.metrics) memset(mf.metrics, 0,
                           (uint64_t)n_points * dim * dim * sizeof(double));
    return mf;
}

void axgeo_metric_field_destroy(axgeo_metric_field_t *mf)
{
    if (!mf) return;
    if (mf->points) { tensor_free(mf->points); mf->points = NULL; }
    if (mf->metrics) { tensor_free(mf->metrics); mf->metrics = NULL; }
    mf->n_points = mf->dim = 0;
}

/*
 * Inverse-distance-weighted interpolation of metric tensor.
 * Uses k nearest neighbors with Shepard weighting: w_i = 1/||x - x_i||^p
 */
void axgeo_metric_interpolate(const axgeo_metric_field_t *mf,
                              const double *x, double *g_out)
{
    int np = mf->n_points;
    int d = mf->dim;
    int dd = d * d;

    memset(g_out, 0, (uint64_t)dd * sizeof(double));

    double w_total = 0.0;
    int k_nearest = (np < 8) ? np : 8;  /* use up to 8 nearest neighbors */

    /* Find k nearest neighbors (simple linear scan — fine for hundreds of points) */
    double *dists = (double *)tensor_alloc((uint64_t)np * sizeof(double));
    int *idx = (int *)tensor_alloc((uint64_t)np * sizeof(int));
    if (!dists || !idx) {
        if (dists) tensor_free(dists);
        if (idx) tensor_free(idx);
        return;
    }

    for (int i = 0; i < np; i++) {
        const double *pt = mf->points + (uint64_t)i * d;
        double dist2 = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = x[j] - pt[j];
            dist2 += diff * diff;
        }
        dists[i] = dist2;
        idx[i] = i;
    }

    /* Partial sort: bring k_nearest smallest to front */
    for (int i = 0; i < k_nearest && i < np; i++) {
        int min_j = i;
        for (int j = i + 1; j < np; j++)
            if (dists[j] < dists[min_j]) min_j = j;
        if (min_j != i) {
            double td = dists[i]; dists[i] = dists[min_j]; dists[min_j] = td;
            int ti = idx[i]; idx[i] = idx[min_j]; idx[min_j] = ti;
        }
    }

    /* Weighted average */
    for (int i = 0; i < k_nearest; i++) {
        double dist = sqrt(dists[i]);
        if (dist < 1e-15) {
            /* Exact match — use this metric directly */
            memcpy(g_out, mf->metrics + (uint64_t)idx[i] * dd,
                   (uint64_t)dd * sizeof(double));
            tensor_free(dists);
            tensor_free(idx);
            return;
        }
        /* Shepard p=2 */
        double w = 1.0 / (dist * dist);
        w_total += w;
        const double *g_i = mf->metrics + (uint64_t)idx[i] * dd;
        for (int j = 0; j < dd; j++)
            g_out[j] += w * g_i[j];
    }

    if (w_total > 0.0) {
        for (int j = 0; j < dd; j++)
            g_out[j] /= w_total;
    }

    tensor_free(dists);
    tensor_free(idx);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Christoffel Symbols
 *
 * Γ^k_ij = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
 *
 * Metric derivatives estimated via centered finite differences between
 * nearest sample points.
 * ═══════════════════════════════════════════════════════════════════════════ */

axgeo_christoffel_t axgeo_christoffel_create(int n_points, int dim)
{
    axgeo_christoffel_t ch;
    ch.n_points = n_points;
    ch.dim = dim;
    uint64_t size = (uint64_t)n_points * dim * dim * dim * sizeof(double);
    ch.gamma = (double *)tensor_alloc(size);
    if (ch.gamma) memset(ch.gamma, 0, size);
    return ch;
}

axgeo_layer_christoffel_t axgeo_layer_christoffel_create(int n_layers, int dim)
{
    axgeo_layer_christoffel_t ch;
    ch.n_layers = n_layers;
    ch.dim = dim;
    ch.gamma = (double *)0;
    if (n_layers > 0 && dim > 0) {
        uint64_t size = (uint64_t)n_layers * dim * dim * dim * sizeof(double);
        ch.gamma = (double *)tensor_alloc(size);
        if (ch.gamma) memset(ch.gamma, 0, size);
    }
    return ch;
}

void axgeo_christoffel_destroy(axgeo_christoffel_t *ch)
{
    if (!ch) return;
    if (ch->gamma) { tensor_free(ch->gamma); ch->gamma = NULL; }
    ch->n_points = ch->dim = 0;
}

void axgeo_layer_christoffel_destroy(axgeo_layer_christoffel_t *ch)
{
    if (!ch) return;
    if (ch->gamma) { tensor_free(ch->gamma); ch->gamma = NULL; }
    ch->n_layers = ch->dim = 0;
}

/* Invert a small symmetric matrix (d×d) using Gauss-Jordan elimination */
static int invert_symmetric(const double *A, double *Ainv, int d)
{
    /* Augmented matrix [A | I] */
    int sz = d * 2 * d;
    double *aug = (double *)tensor_alloc((uint64_t)sz * sizeof(double));
    if (!aug) return -1;

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            double v = A[i * d + j];
            /* Replace NaN/Inf with identity matrix value */
            if (v != v || v > 1e30 || v < -1e30) v = (i == j) ? 1.0 : 0.0;
            aug[i * 2 * d + j] = v;
        }
        for (int j = 0; j < d; j++)
            aug[i * 2 * d + d + j] = (i == j) ? 1.0 : 0.0;
    }

    for (int i = 0; i < d; i++) {
        /* Find pivot */
        int piv = i;
        double piv_val = fabs(aug[i * 2 * d + i]);
        for (int k = i + 1; k < d; k++) {
            double v = fabs(aug[k * 2 * d + i]);
            if (v > piv_val) { piv = k; piv_val = v; }
        }
        if (piv_val < 1e-14) {
            /* Singular — add regularization */
            aug[i * 2 * d + i] += 1e-10;
            piv_val = fabs(aug[i * 2 * d + i]);
        }

        /* Swap rows */
        if (piv != i) {
            for (int j = 0; j < 2 * d; j++) {
                double tmp = aug[i * 2 * d + j];
                aug[i * 2 * d + j] = aug[piv * 2 * d + j];
                aug[piv * 2 * d + j] = tmp;
            }
        }

        /* Eliminate */
        double diag = aug[i * 2 * d + i];
        for (int j = 0; j < 2 * d; j++)
            aug[i * 2 * d + j] /= diag;

        for (int k = 0; k < d; k++) {
            if (k == i) continue;
            double factor = aug[k * 2 * d + i];
            for (int j = 0; j < 2 * d; j++)
                aug[k * 2 * d + j] -= factor * aug[i * 2 * d + j];
        }
    }

    /* Extract inverse */
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            Ainv[i * d + j] = aug[i * 2 * d + d + j];

    tensor_free(aug);
    return 0;
}

int axgeo_compute_christoffel(const axgeo_metric_field_t *mf,
                              axgeo_christoffel_t *ch)
{
    int np = mf->n_points;
    int d = mf->dim;
    if (np < 2 || d <= 0 || d > AXGEO_MAX_DIM) return -1;

    int dd = d * d;
    int ddd = d * d * d;

    /* Allocate temporary buffers */
    double *g_inv = (double *)tensor_alloc((uint64_t)dd * sizeof(double));
    double *dg = (double *)tensor_alloc((uint64_t)d * dd * sizeof(double));  /* ∂_m g_{ij} for each m */
    if (!g_inv || !dg) {
        if (g_inv) tensor_free(g_inv);
        if (dg) tensor_free(dg);
        return -1;
    }

    /* For each sample point, compute Christoffel symbols */
    for (int p = 0; p < np; p++) {
        const double *g_p = mf->metrics + (uint64_t)p * dd;
        const double *x_p = mf->points + (uint64_t)p * d;
        double *gamma_p = ch->gamma + (uint64_t)p * ddd;

        /* 1. Invert the metric tensor at this point: g^{kl} */
        invert_symmetric(g_p, g_inv, d);

        /* 2. Estimate metric derivatives ∂_m g_{ij} using finite differences
         *    between nearby sample points. */
        memset(dg, 0, (uint64_t)d * dd * sizeof(double));

        /* Find nearest neighbors for finite difference estimates */
        for (int m = 0; m < d; m++) {
            /* Find the sample point with the largest displacement in
             * direction m relative to this point */
            int best_plus = -1, best_minus = -1;
            double best_dp = 0.0, best_dm = 0.0;

            for (int q = 0; q < np; q++) {
                if (q == p) continue;
                const double *x_q = mf->points + (uint64_t)q * d;
                double delta_m = x_q[m] - x_p[m];

                /* Check that displacement is primarily in direction m */
                double delta2_total = 0.0;
                for (int k = 0; k < d; k++) {
                    double dk = x_q[k] - x_p[k];
                    delta2_total += dk * dk;
                }
                double delta_m2 = delta_m * delta_m;
                if (delta2_total > 0 && delta_m2 / delta2_total < 0.3)
                    continue;  /* too much off-axis displacement */

                if (delta_m > best_dp) {
                    best_dp = delta_m;
                    best_plus = q;
                }
                if (delta_m < -best_dm || (best_minus < 0 && delta_m < 0)) {
                    best_dm = -delta_m;
                    best_minus = q;
                }
            }

            /* Centered or single-sided finite difference */
            if (best_plus >= 0 && best_minus >= 0) {
                const double *g_plus = mf->metrics + (uint64_t)best_plus * dd;
                const double *g_minus = mf->metrics + (uint64_t)best_minus * dd;
                double h = best_dp + best_dm;
                if (h > 1e-15) {
                    for (int ij = 0; ij < dd; ij++)
                        dg[m * dd + ij] = (g_plus[ij] - g_minus[ij]) / h;
                }
            } else if (best_plus >= 0) {
                const double *g_plus = mf->metrics + (uint64_t)best_plus * dd;
                if (best_dp > 1e-15) {
                    for (int ij = 0; ij < dd; ij++)
                        dg[m * dd + ij] = (g_plus[ij] - g_p[ij]) / best_dp;
                }
            } else if (best_minus >= 0) {
                const double *g_minus = mf->metrics + (uint64_t)best_minus * dd;
                if (best_dm > 1e-15) {
                    for (int ij = 0; ij < dd; ij++)
                        dg[m * dd + ij] = (g_p[ij] - g_minus[ij]) / best_dm;
                }
            }
        }

        /* 3. Compute Γ^k_ij = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
         *    gamma_p[k * dd + i * d + j] = Γ^k_ij */
        for (int k = 0; k < d; k++) {
            for (int i = 0; i < d; i++) {
                for (int j = i; j < d; j++) {
                    double sum = 0.0;
                    for (int l = 0; l < d; l++) {
                        double dg_i_jl = dg[i * dd + j * d + l];
                        double dg_j_il = dg[j * dd + i * d + l];
                        double dg_l_ij = dg[l * dd + i * d + j];
                        sum += g_inv[k * d + l] *
                               (dg_i_jl + dg_j_il - dg_l_ij);
                    }
                    /* Clamp NaN/Inf to 0 — singular metrics produce garbage Christoffel */
                    double val = 0.5 * sum;
                    if (val != val || val > 1e15 || val < -1e15) val = 0.0;
                    gamma_p[k * dd + i * d + j] = val;
                    gamma_p[k * dd + j * d + i] = val; /* symmetric in i,j */
                }
            }
        }
    }

    tensor_free(g_inv);
    tensor_free(dg);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * axgeo_compute_christoffel_global
 *
 * Computes a single globally-valid Γ^k_ij by:
 *   1. Mean metric  ḡ_{ij}  = average over all sample points
 *   2. Global gradient  ∂_m ḡ_{ij}  via marginal OLS regression:
 *          slope_m = Σ_p (x_m(p) − x̄_m)(g_{ij}(p) − ḡ_{ij})
 *                  / Σ_p (x_m(p) − x̄_m)²
 *   3. ḡ^{kl}  from Gauss-Jordan inversion of ḡ
 *   4. Γ^k_ij = ½ ḡ^{kl}(∂_i ḡ_{jl} + ∂_j ḡ_{il} − ∂_l ḡ_{ij})  (d⁴)
 *
 * ch must be created with axgeo_christoffel_create(1, dim).
 * ═══════════════════════════════════════════════════════════════════════════ */
int axgeo_compute_christoffel_global(const axgeo_metric_field_t *mf,
                                     axgeo_christoffel_t *ch)
{
    int np = mf->n_points;
    int d  = mf->dim;
    if (np < 2 || d <= 0 || d > AXGEO_MAX_DIM) return -1;
    if (ch->n_points != 1 || ch->dim != d || !ch->gamma) return -1;

    int dd  = d * d;
    int ddd = d * d * d;

    double *g_mean = (double *)tensor_alloc((uint64_t)dd  * sizeof(double));
    double *dg     = (double *)tensor_alloc((uint64_t)d * dd * sizeof(double)); /* ∂_m ḡ_{ij} */
    double *g_inv  = (double *)tensor_alloc((uint64_t)dd  * sizeof(double));
    double *x_mean = (double *)tensor_alloc((uint64_t)d   * sizeof(double));
    double *var_m  = (double *)tensor_alloc((uint64_t)d   * sizeof(double));
    if (!g_mean || !dg || !g_inv || !x_mean || !var_m) {
        if (g_mean) tensor_free(g_mean);
        if (dg)     tensor_free(dg);
        if (g_inv)  tensor_free(g_inv);
        if (x_mean) tensor_free(x_mean);
        if (var_m)  tensor_free(var_m);
        return -1;
    }

    /* ── Step 1: mean metric and mean sample position ───────────────────── */
    memset(g_mean, 0, (uint64_t)dd * sizeof(double));
    memset(x_mean, 0, (uint64_t)d  * sizeof(double));
    for (int p = 0; p < np; p++) {
        const double *g_p = mf->metrics + (uint64_t)p * dd;
        const double *x_p = mf->points  + (uint64_t)p * d;
        for (int ij = 0; ij < dd; ij++) g_mean[ij] += g_p[ij];
        for (int j  = 0; j  < d;  j++)  x_mean[j]  += x_p[j];
    }
    double inv_np = 1.0 / (double)np;
    for (int ij = 0; ij < dd; ij++) g_mean[ij] *= inv_np;
    for (int j  = 0; j  < d;  j++)  x_mean[j]  *= inv_np;

    /* ── Step 2: per-direction variance  Σ (x_m − x̄_m)²  ─────────────── */
    memset(var_m, 0, (uint64_t)d * sizeof(double));
    for (int p = 0; p < np; p++) {
        const double *x_p = mf->points + (uint64_t)p * d;
        for (int m = 0; m < d; m++) {
            double dm = x_p[m] - x_mean[m];
            var_m[m] += dm * dm;
        }
    }

    /* ── Step 3: marginal OLS gradient  ∂_m ḡ_{ij}  ────────────────────── */
    memset(dg, 0, (uint64_t)d * dd * sizeof(double));
    for (int p = 0; p < np; p++) {
        const double *g_p = mf->metrics + (uint64_t)p * dd;
        const double *x_p = mf->points  + (uint64_t)p * d;
        for (int m = 0; m < d; m++) {
            double dm = x_p[m] - x_mean[m];
            if (var_m[m] < 1e-30) continue;
            double inv_var = dm / var_m[m];
            for (int ij = 0; ij < dd; ij++)
                dg[m * dd + ij] += inv_var * (g_p[ij] - g_mean[ij]);
        }
    }

    /* ── Step 4: invert mean metric  ḡ^{kl}  ───────────────────────────── */
    invert_symmetric(g_mean, g_inv, d);

    /* ── Step 5: Γ^k_ij = ½ ḡ^{kl}(∂_i ḡ_{jl} + ∂_j ḡ_{il} − ∂_l ḡ_{ij}) */
    double *gamma0 = ch->gamma; /* n_points=1, so offset = 0 */
    for (int k = 0; k < d; k++) {
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                double sum = 0.0;
                for (int l = 0; l < d; l++) {
                    double dg_i_jl = dg[i * dd + j * d + l];
                    double dg_j_il = dg[j * dd + i * d + l];
                    double dg_l_ij = dg[l * dd + i * d + j];
                    sum += g_inv[k * d + l] * (dg_i_jl + dg_j_il - dg_l_ij);
                }
                double val = 0.5 * sum;
                if (val != val || val > 1e15 || val < -1e15) val = 0.0;
                gamma0[k * dd + i * d + j] = val;
                gamma0[k * dd + j * d + i] = val; /* symmetric in i,j */
            }
        }
    }

    tensor_free(g_mean);
    tensor_free(dg);
    tensor_free(g_inv);
    tensor_free(x_mean);
    tensor_free(var_m);
    return 0;
}

void axgeo_christoffel_interpolate(const axgeo_christoffel_t *ch,
                                   const axgeo_metric_field_t *mf,
                                   const double *x, double *gamma_out)
{
    int np = ch->n_points;
    int d = ch->dim;
    int ddd = d * d * d;

    memset(gamma_out, 0, (uint64_t)ddd * sizeof(double));

    /* Same k-nearest inverse-distance weighting as metric interpolation */
    int k_nearest = (np < 8) ? np : 8;
    double *dists = (double *)tensor_alloc((uint64_t)np * sizeof(double));
    int *idx = (int *)tensor_alloc((uint64_t)np * sizeof(int));
    if (!dists || !idx) {
        if (dists) tensor_free(dists);
        if (idx) tensor_free(idx);
        return;
    }

    for (int i = 0; i < np; i++) {
        const double *pt = mf->points + (uint64_t)i * d;
        double dist2 = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = x[j] - pt[j];
            dist2 += diff * diff;
        }
        dists[i] = dist2;
        idx[i] = i;
    }

    for (int i = 0; i < k_nearest && i < np; i++) {
        int min_j = i;
        for (int j = i + 1; j < np; j++)
            if (dists[j] < dists[min_j]) min_j = j;
        if (min_j != i) {
            double td = dists[i]; dists[i] = dists[min_j]; dists[min_j] = td;
            int ti = idx[i]; idx[i] = idx[min_j]; idx[min_j] = ti;
        }
    }

    double w_total = 0.0;
    for (int i = 0; i < k_nearest; i++) {
        double dist = sqrt(dists[i]);
        if (dist < 1e-15) {
            memcpy(gamma_out, ch->gamma + (uint64_t)idx[i] * ddd,
                   (uint64_t)ddd * sizeof(double));
            tensor_free(dists);
            tensor_free(idx);
            return;
        }
        double w = 1.0 / (dist * dist);
        w_total += w;
        const double *g_i = ch->gamma + (uint64_t)idx[i] * ddd;
        for (int j = 0; j < ddd; j++)
            gamma_out[j] += w * g_i[j];
    }

    if (w_total > 0.0) {
        for (int j = 0; j < ddd; j++)
            gamma_out[j] /= w_total;
    }

    tensor_free(dists);
    tensor_free(idx);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Curvature
 *
 * Riemann:  R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij}
 *                      + Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij}
 *
 * We compute Ricci and scalar curvature from the Christoffel symbols.
 * Riemann tensor not stored explicitly (O(d⁴) memory) — only contracted.
 * ═══════════════════════════════════════════════════════════════════════════ */

axgeo_curvature_t axgeo_curvature_create(int n_points, int dim)
{
    axgeo_curvature_t c;
    c.n_points = n_points;
    c.dim = dim;
    c.ricci = (double *)tensor_alloc((uint64_t)n_points * dim * dim * sizeof(double));
    c.scalar_curv = (double *)tensor_alloc((uint64_t)n_points * sizeof(double));
    if (c.ricci) memset(c.ricci, 0,
                         (uint64_t)n_points * dim * dim * sizeof(double));
    if (c.scalar_curv) memset(c.scalar_curv, 0,
                               (uint64_t)n_points * sizeof(double));
    c.mean_curvature = 0.0;
    c.max_curvature = 0.0;
    return c;
}

void axgeo_curvature_destroy(axgeo_curvature_t *curv)
{
    if (!curv) return;
    if (curv->ricci) { tensor_free(curv->ricci); curv->ricci = NULL; }
    if (curv->scalar_curv) { tensor_free(curv->scalar_curv); curv->scalar_curv = NULL; }
    curv->n_points = curv->dim = 0;
}

int axgeo_compute_curvature(const axgeo_christoffel_t *ch,
                            const axgeo_metric_field_t *mf,
                            axgeo_curvature_t *curv)
{
    int np = ch->n_points;
    int d = ch->dim;
    int dd = d * d;
    int ddd = d * d * d;

    if (np < 1 || d <= 0) return -1;

    /* We compute Ricci tensor by contracting Riemann:
     * R_{ij} = R^k_{ikj} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik}
     *                     + Γ^k_{km} Γ^m_{ij} - Γ^k_{jm} Γ^m_{ik}
     *
     * Christoffel derivatives are numerically estimated from nearby sample
     * points (same finite-difference approach as metric derivatives). */

    /* Workspace for inverse metric (for scalar curvature) */
    double *g_inv = (double *)tensor_alloc((uint64_t)dd * sizeof(double));
    /* Workspace for Christoffel derivatives ∂_m Γ^k_ij */
    double *dGamma = (double *)tensor_alloc((uint64_t)d * ddd * sizeof(double));
    if (!g_inv || !dGamma) {
        if (g_inv) tensor_free(g_inv);
        if (dGamma) tensor_free(dGamma);
        return -1;
    }

    curv->mean_curvature = 0.0;
    curv->max_curvature = 0.0;

    for (int p = 0; p < np; p++) {
        const double *gamma_p = ch->gamma + (uint64_t)p * ddd;
        double *ricci_p = curv->ricci + (uint64_t)p * dd;
        const double *x_p = mf->points + (uint64_t)p * d;

        /* ── Estimate Christoffel derivatives ∂_m Γ^k_ij ──
         * Use finite differences between Christoffel symbols at nearby
         * sample points, same directional filtering as metric derivatives. */
        memset(dGamma, 0, (uint64_t)d * ddd * sizeof(double));

        for (int m = 0; m < d; m++) {
            int best_plus = -1, best_minus = -1;
            double best_dp = 0.0, best_dm = 0.0;

            for (int q = 0; q < np; q++) {
                if (q == p) continue;
                const double *x_q = mf->points + (uint64_t)q * d;
                double delta_m = x_q[m] - x_p[m];

                double delta2_total = 0.0;
                for (int kk = 0; kk < d; kk++) {
                    double dk = x_q[kk] - x_p[kk];
                    delta2_total += dk * dk;
                }
                double delta_m2 = delta_m * delta_m;
                if (delta2_total > 0 && delta_m2 / delta2_total < 0.3)
                    continue;

                if (delta_m > best_dp) { best_dp = delta_m; best_plus = q; }
                if (delta_m < -best_dm || (best_minus < 0 && delta_m < 0)) {
                    best_dm = -delta_m; best_minus = q;
                }
            }

            double *dG_m = dGamma + (uint64_t)m * ddd;  /* ∂_m Γ^k_ij */
            if (best_plus >= 0 && best_minus >= 0) {
                const double *g_plus  = ch->gamma + (uint64_t)best_plus * ddd;
                const double *g_minus = ch->gamma + (uint64_t)best_minus * ddd;
                double h = best_dp + best_dm;
                if (h > 1e-15) {
                    for (int idx = 0; idx < ddd; idx++)
                        dG_m[idx] = (g_plus[idx] - g_minus[idx]) / h;
                }
            } else if (best_plus >= 0) {
                const double *g_plus = ch->gamma + (uint64_t)best_plus * ddd;
                if (best_dp > 1e-15) {
                    for (int idx = 0; idx < ddd; idx++)
                        dG_m[idx] = (g_plus[idx] - gamma_p[idx]) / best_dp;
                }
            } else if (best_minus >= 0) {
                const double *g_minus = ch->gamma + (uint64_t)best_minus * ddd;
                if (best_dm > 1e-15) {
                    for (int idx = 0; idx < ddd; idx++)
                        dG_m[idx] = (gamma_p[idx] - g_minus[idx]) / best_dm;
                }
            }
        }

        /* ── Compute full Ricci tensor ──
         * R_{ij} = R^k_{ikj} = ∂_k Γ^k_{ij} - ∂_j Γ^k_{ik}
         *                     + Γ^k_{km} Γ^m_{ij} - Γ^k_{jm} Γ^m_{ik}
         */

        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                double r_ij = 0.0;

                for (int k = 0; k < d; k++) {
                    /* ∂_k Γ^k_{ij} — derivative term (from dGamma) */
                    r_ij += dGamma[k * ddd + k * dd + i * d + j];

                    /* -∂_j Γ^k_{ik} — derivative term */
                    r_ij -= dGamma[j * ddd + k * dd + i * d + k];

                    /* Γ^k_{km} Γ^m_{ij} — algebraic term */
                    for (int m = 0; m < d; m++) {
                        r_ij += gamma_p[k * dd + k * d + m] *
                                gamma_p[m * dd + i * d + j];
                    }
                    /* -Γ^k_{jm} Γ^m_{ik} — algebraic term */
                    for (int m = 0; m < d; m++) {
                        r_ij -= gamma_p[k * dd + j * d + m] *
                                gamma_p[m * dd + i * d + k];
                    }
                }

                ricci_p[i * d + j] = r_ij;
                ricci_p[j * d + i] = r_ij;  /* symmetric */
            }
        }

        /* Scalar curvature: R = g^{ij} R_{ij} */
        const double *g_p = mf->metrics + (uint64_t)p * dd;
        invert_symmetric(g_p, g_inv, d);

        double scalar = 0.0;
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                scalar += g_inv[i * d + j] * ricci_p[i * d + j];

        curv->scalar_curv[p] = scalar;
        curv->mean_curvature += scalar;
        if (fabs(scalar) > curv->max_curvature)
            curv->max_curvature = fabs(scalar);
    }

    if (np > 0)
        curv->mean_curvature /= (double)np;

    tensor_free(g_inv);
    tensor_free(dGamma);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Geodesic Solver (RK4)
 *
 * Integrates the geodesic equation:
 *   dx^μ/dλ = v^μ
 *   dv^μ/dλ = -Γ^μ_νρ v^ν v^ρ
 *
 * using classical 4th-order Runge-Kutta.
 * ═══════════════════════════════════════════════════════════════════════════ */

axgeo_geodesic_t axgeo_geodesic_init(int dim, const double *x0,
                                      const double *v0,
                                      double step_size, int max_steps,
                                      int record_trajectory)
{
    axgeo_geodesic_t g;
    memset(&g, 0, sizeof(g));
    g.dim = dim;
    g.step_size = step_size;
    g.max_steps = max_steps;
    g.steps = 0;
    g.lambda = 0.0;
    g.record = record_trajectory;

    g.x = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
    g.v = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
    if (g.x) memcpy(g.x, x0, (uint64_t)dim * sizeof(double));
    if (g.v) memcpy(g.v, v0, (uint64_t)dim * sizeof(double));

    if (record_trajectory && max_steps > 0) {
        g.trajectory = (double *)tensor_alloc((uint64_t)max_steps * dim * sizeof(double));
        if (g.trajectory)
            memcpy(g.trajectory, x0, (uint64_t)dim * sizeof(double));
    }

    return g;
}

void axgeo_geodesic_destroy(axgeo_geodesic_t *geo)
{
    if (!geo) return;
    if (geo->x) { tensor_free(geo->x); geo->x = NULL; }
    if (geo->v) { tensor_free(geo->v); geo->v = NULL; }
    if (geo->trajectory) { tensor_free(geo->trajectory); geo->trajectory = NULL; }
}

/*
 * Compute acceleration: dv^μ/dλ = -Γ^μ_νρ v^ν v^ρ
 *
 * gamma: [dim × dim × dim] Christoffel symbols at the current point
 * v:     [dim] velocity
 * acc:   [dim] output acceleration
 *
 * Item 3 (Christoffel sparsity): entries with |Γ| < sparse_thresh are skipped.
 * In the intrinsic subspace (dim≈22), the manifold has approximate symmetry in
 * most directions — typically 70-90% of entries are near-zero.  Skipping them
 * cuts the O(d³) inner loop cost by the sparsity fraction.
 */
#define GAMMA_SPARSE_THRESH 1e-9

static void geodesic_acceleration(const double *gamma, const double *v,
                                  double *acc, int dim)
{
    int dd = dim * dim;
    for (int mu = 0; mu < dim; mu++) {
        double a = 0.0;
        const double *gamma_mu = gamma + mu * dd;
        for (int nu = 0; nu < dim; nu++) {
            double vnu = v[nu];
            if (vnu == 0.0) continue;                      /* sparse v */
            const double *row = gamma_mu + nu * dim;
            for (int rho = 0; rho < dim; rho++) {
                double g = row[rho];
                if (g > GAMMA_SPARSE_THRESH || g < -GAMMA_SPARSE_THRESH)
                    a -= g * vnu * v[rho];
            }
        }
        acc[mu] = a;
    }
}

int axgeo_geodesic_integrate(axgeo_geodesic_t *geo,
                             const axgeo_christoffel_t *ch,
                             const axgeo_metric_field_t *mf,
                             int n_steps)
{
    int d = geo->dim;
    double h = geo->step_size;
    int dd = d * d;
    int ddd = d * d * d;

    /* RK4 workspace */
    double *kx1 = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *kv1 = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *kx2 = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *kv2 = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *kx3 = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *kv3 = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *kx4 = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *kv4 = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *xt  = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *vt  = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *gamma_local = (double *)tensor_alloc((uint64_t)ddd * sizeof(double));

    if (!kx1 || !kv1 || !kx2 || !kv2 || !kx3 || !kv3 ||
        !kx4 || !kv4 || !xt || !vt || !gamma_local) {
        /* cleanup partial allocs */
        if (kx1) tensor_free(kx1); if (kv1) tensor_free(kv1);
        if (kx2) tensor_free(kx2); if (kv2) tensor_free(kv2);
        if (kx3) tensor_free(kx3); if (kv3) tensor_free(kv3);
        if (kx4) tensor_free(kx4); if (kv4) tensor_free(kv4);
        if (xt) tensor_free(xt);   if (vt) tensor_free(vt);
        if (gamma_local) tensor_free(gamma_local);
        return -1;
    }

    for (int step = 0; step < n_steps; step++) {
        if (geo->steps >= geo->max_steps - 1) break;

        /* k1: evaluate at (x, v) */
        axgeo_christoffel_interpolate(ch, mf, geo->x, gamma_local);
        for (int i = 0; i < d; i++) kx1[i] = geo->v[i];
        geodesic_acceleration(gamma_local, geo->v, kv1, d);

        /* k2: evaluate at (x + h/2 * kx1, v + h/2 * kv1) */
        for (int i = 0; i < d; i++) {
            xt[i] = geo->x[i] + 0.5 * h * kx1[i];
            vt[i] = geo->v[i] + 0.5 * h * kv1[i];
        }
        axgeo_christoffel_interpolate(ch, mf, xt, gamma_local);
        for (int i = 0; i < d; i++) kx2[i] = vt[i];
        geodesic_acceleration(gamma_local, vt, kv2, d);

        /* k3: evaluate at (x + h/2 * kx2, v + h/2 * kv2) */
        for (int i = 0; i < d; i++) {
            xt[i] = geo->x[i] + 0.5 * h * kx2[i];
            vt[i] = geo->v[i] + 0.5 * h * kv2[i];
        }
        axgeo_christoffel_interpolate(ch, mf, xt, gamma_local);
        for (int i = 0; i < d; i++) kx3[i] = vt[i];
        geodesic_acceleration(gamma_local, vt, kv3, d);

        /* k4: evaluate at (x + h * kx3, v + h * kv3) */
        for (int i = 0; i < d; i++) {
            xt[i] = geo->x[i] + h * kx3[i];
            vt[i] = geo->v[i] + h * kv3[i];
        }
        axgeo_christoffel_interpolate(ch, mf, xt, gamma_local);
        for (int i = 0; i < d; i++) kx4[i] = vt[i];
        geodesic_acceleration(gamma_local, vt, kv4, d);

        /* Update: x += (h/6)(k1 + 2k2 + 2k3 + k4) */
        for (int i = 0; i < d; i++) {
            geo->x[i] += (h / 6.0) * (kx1[i] + 2*kx2[i] + 2*kx3[i] + kx4[i]);
            geo->v[i] += (h / 6.0) * (kv1[i] + 2*kv2[i] + 2*kv3[i] + kv4[i]);
        }

        geo->lambda += h;
        geo->steps++;

        /* Record trajectory */
        if (geo->record && geo->trajectory) {
            memcpy(geo->trajectory + (uint64_t)geo->steps * d,
                   geo->x, (uint64_t)d * sizeof(double));
        }

        /* Divergence check: velocity blowup */
        double vnorm = ax_vec_norm(geo->v, d);
        if (vnorm > 1e10 || vnorm != vnorm) {  /* nan check */
            tensor_free(kx1); tensor_free(kv1); tensor_free(kx2); tensor_free(kv2);
            tensor_free(kx3); tensor_free(kv3); tensor_free(kx4); tensor_free(kv4);
            tensor_free(xt);  tensor_free(vt);  tensor_free(gamma_local);
            return -1;
        }
    }

    tensor_free(kx1); tensor_free(kv1); tensor_free(kx2); tensor_free(kv2);
    tensor_free(kx3); tensor_free(kv3); tensor_free(kx4); tensor_free(kv4);
    tensor_free(xt);  tensor_free(vt);  tensor_free(gamma_local);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Adaptive RK45 Geodesic Integrator (Cash-Karp / Dormand-Prince coefficients)
 *
 * Item 2: Variable-step RK45.  The ODE system is:
 *   dx/dλ = v,   dv/dλ = -Γ(x) v⊗v
 *
 * Cash-Karp Butcher tableau (6-stage, 4th+5th order):
 *   RK4 estimate (y4) and RK5 estimate (y5) share all 6 function evals.
 *   Error = |y5 - y4|; step accepted if err < tol, rejected+halved otherwise.
 *
 * On a nearly-flat manifold (most of it, per the 2026 Nature paper), the
 * step size grows to h_max, so flat regions cost ~1 step per h_max stride
 * instead of n_steps fixed steps.  High-curvature sites shrink h automatically.
 *
 * Returns: number of accepted steps taken, or -1 on divergence/alloc failure.
 *          geo->x / geo->v updated to the final state.
 *          geo->lambda updated to total path length integrated.
 *          geo->steps = accepted step count.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Cash-Karp coefficients */
static const double CK_A21 =  1.0/5.0;
static const double CK_A31 =  3.0/40.0,  CK_A32 = 9.0/40.0;
static const double CK_A41 =  3.0/10.0,  CK_A42 = -9.0/10.0, CK_A43 = 6.0/5.0;
static const double CK_A51 = -11.0/54.0, CK_A52 = 5.0/2.0,   CK_A53 = -70.0/27.0, CK_A54 = 35.0/27.0;
static const double CK_A61 =  1631.0/55296.0, CK_A62 = 175.0/512.0, CK_A63 = 575.0/13824.0,
                    CK_A64 =  44275.0/110592.0, CK_A65 = 253.0/4096.0;
/* 4th-order weights */
static const double CK_B41 =  37.0/378.0, CK_B43 = 250.0/621.0, CK_B44 = 125.0/594.0, CK_B46 = 512.0/1771.0;
/* 5th-order weights */
static const double CK_B51 =  2825.0/27648.0, CK_B53 = 18575.0/48384.0,
                    CK_B54 =  13525.0/55296.0, CK_B55 = 277.0/14336.0, CK_B56 = 1.0/4.0;

int axgeo_geodesic_integrate_adaptive(axgeo_geodesic_t *geo,
                                      const axgeo_christoffel_t *ch,
                                      const axgeo_metric_field_t *mf,
                                      double lambda_end,
                                      double tol,
                                      double h_min,
                                      double h_max)
{
    int d = geo->dim;
    int ddd = d * d * d;

    /* Allocate 6 stage buffers for (x,v) derivatives + workspace */
    double *kx[6], *kv[6];
    int alloc_ok = 1;
    for (int s = 0; s < 6; s++) {
        kx[s] = (double *)tensor_alloc((uint64_t)d * sizeof(double));
        kv[s] = (double *)tensor_alloc((uint64_t)d * sizeof(double));
        if (!kx[s] || !kv[s]) { alloc_ok = 0; break; }
    }
    double *xt   = alloc_ok ? (double *)tensor_alloc((uint64_t)d * sizeof(double)) : NULL;
    double *vt   = alloc_ok ? (double *)tensor_alloc((uint64_t)d * sizeof(double)) : NULL;
    double *x4   = alloc_ok ? (double *)tensor_alloc((uint64_t)d * sizeof(double)) : NULL;
    double *v4   = alloc_ok ? (double *)tensor_alloc((uint64_t)d * sizeof(double)) : NULL;
    double *x5   = alloc_ok ? (double *)tensor_alloc((uint64_t)d * sizeof(double)) : NULL;
    double *v5   = alloc_ok ? (double *)tensor_alloc((uint64_t)d * sizeof(double)) : NULL;
    double *gl   = alloc_ok ? (double *)tensor_alloc((uint64_t)ddd * sizeof(double)) : NULL;
    if (!alloc_ok || !xt || !vt || !x4 || !v4 || !x5 || !v5 || !gl) {
        for (int s = 0; s < 6; s++) { if (kx[s]) tensor_free(kx[s]); if (kv[s]) tensor_free(kv[s]); }
        if (xt) tensor_free(xt); if (vt) tensor_free(vt);
        if (x4) tensor_free(x4); if (v4) tensor_free(v4);
        if (x5) tensor_free(x5); if (v5) tensor_free(v5);
        if (gl) tensor_free(gl);
        return -1;
    }

    double h    = geo->step_size > 0.0 ? geo->step_size : h_max;
    if (h > h_max) h = h_max;
    if (h < h_min) h = h_min;

    int accepted = 0;
    int rejected = 0;

    while (geo->lambda < lambda_end && geo->steps < geo->max_steps - 1) {
        if (geo->lambda + h > lambda_end) h = lambda_end - geo->lambda;
        if (h < h_min) h = h_min;

        /* ── Stage 1: k1 at (x, v) ── */
        axgeo_christoffel_interpolate(ch, mf, geo->x, gl);
        for (int i = 0; i < d; i++) kx[0][i] = geo->v[i];
        geodesic_acceleration(gl, geo->v, kv[0], d);

        /* ── Stage 2 ── */
        for (int i = 0; i < d; i++) { xt[i] = geo->x[i] + h*CK_A21*kx[0][i]; vt[i] = geo->v[i] + h*CK_A21*kv[0][i]; }
        axgeo_christoffel_interpolate(ch, mf, xt, gl);
        for (int i = 0; i < d; i++) kx[1][i] = vt[i];
        geodesic_acceleration(gl, vt, kv[1], d);

        /* ── Stage 3 ── */
        for (int i = 0; i < d; i++) { xt[i] = geo->x[i] + h*(CK_A31*kx[0][i]+CK_A32*kx[1][i]); vt[i] = geo->v[i] + h*(CK_A31*kv[0][i]+CK_A32*kv[1][i]); }
        axgeo_christoffel_interpolate(ch, mf, xt, gl);
        for (int i = 0; i < d; i++) kx[2][i] = vt[i];
        geodesic_acceleration(gl, vt, kv[2], d);

        /* ── Stage 4 ── */
        for (int i = 0; i < d; i++) { xt[i] = geo->x[i] + h*(CK_A41*kx[0][i]+CK_A42*kx[1][i]+CK_A43*kx[2][i]); vt[i] = geo->v[i] + h*(CK_A41*kv[0][i]+CK_A42*kv[1][i]+CK_A43*kv[2][i]); }
        axgeo_christoffel_interpolate(ch, mf, xt, gl);
        for (int i = 0; i < d; i++) kx[3][i] = vt[i];
        geodesic_acceleration(gl, vt, kv[3], d);

        /* ── Stage 5 ── */
        for (int i = 0; i < d; i++) { xt[i] = geo->x[i] + h*(CK_A51*kx[0][i]+CK_A52*kx[1][i]+CK_A53*kx[2][i]+CK_A54*kx[3][i]); vt[i] = geo->v[i] + h*(CK_A51*kv[0][i]+CK_A52*kv[1][i]+CK_A53*kv[2][i]+CK_A54*kv[3][i]); }
        axgeo_christoffel_interpolate(ch, mf, xt, gl);
        for (int i = 0; i < d; i++) kx[4][i] = vt[i];
        geodesic_acceleration(gl, vt, kv[4], d);

        /* ── Stage 6 ── */
        for (int i = 0; i < d; i++) { xt[i] = geo->x[i] + h*(CK_A61*kx[0][i]+CK_A62*kx[1][i]+CK_A63*kx[2][i]+CK_A64*kx[3][i]+CK_A65*kx[4][i]); vt[i] = geo->v[i] + h*(CK_A61*kv[0][i]+CK_A62*kv[1][i]+CK_A63*kv[2][i]+CK_A64*kv[3][i]+CK_A65*kv[4][i]); }
        axgeo_christoffel_interpolate(ch, mf, xt, gl);
        for (int i = 0; i < d; i++) kx[5][i] = vt[i];
        geodesic_acceleration(gl, vt, kv[5], d);

        /* ── 4th-order estimate ── */
        for (int i = 0; i < d; i++) {
            x4[i] = geo->x[i] + h*(CK_B41*kx[0][i]+CK_B43*kx[2][i]+CK_B44*kx[3][i]+CK_B46*kx[5][i]);
            v4[i] = geo->v[i] + h*(CK_B41*kv[0][i]+CK_B43*kv[2][i]+CK_B44*kv[3][i]+CK_B46*kv[5][i]);
        }
        /* ── 5th-order estimate ── */
        for (int i = 0; i < d; i++) {
            x5[i] = geo->x[i] + h*(CK_B51*kx[0][i]+CK_B53*kx[2][i]+CK_B54*kx[3][i]+CK_B55*kx[4][i]+CK_B56*kx[5][i]);
            v5[i] = geo->v[i] + h*(CK_B51*kv[0][i]+CK_B53*kv[2][i]+CK_B54*kv[3][i]+CK_B55*kv[4][i]+CK_B56*kv[5][i]);
        }

        /* ── Error estimate: max absolute component difference ── */
        double err = 0.0;
        for (int i = 0; i < d; i++) {
            double ex = fabs(x5[i] - x4[i]);
            double ev = fabs(v5[i] - v4[i]);
            if (ex > err) err = ex;
            if (ev > err) err = ev;
        }

        /* ── Step size control: h_new = h * (tol/err)^(1/5) * safety ── */
        double h_new;
        if (err < 1e-15) {
            h_new = h * 5.0;  /* essentially zero error — max growth */
        } else {
            double ratio = tol / err;
            /* 5th root: ratio^0.2, safety factor 0.9 */
            double scale = 0.9;
            /* Fast approximation: exp(0.2 * log(ratio)) */
            if (ratio > 1.0) {
                scale *= (ratio > 1e10) ? 5.0 : (1.0 + 0.2 * (ratio - 1.0) / (1.0 + 0.1 * (ratio - 1.0)));
            } else {
                scale *= (ratio < 0.1) ? 0.1 : ratio;
            }
            h_new = h * scale;
        }
        if (h_new > h_max) h_new = h_max;
        if (h_new < h_min) h_new = h_min;

        if (err <= tol || h <= h_min) {
            /* Accept step — take the 4th-order result (more conservative) */
            memcpy(geo->x, x4, (uint64_t)d * sizeof(double));
            memcpy(geo->v, v4, (uint64_t)d * sizeof(double));
            geo->lambda += h;
            geo->steps++;
            accepted++;

            /* Trajectory recording */
            if (geo->record && geo->trajectory && geo->steps < geo->max_steps)
                memcpy(geo->trajectory + (uint64_t)geo->steps * d, geo->x, (uint64_t)d * sizeof(double));

            /* Divergence check */
            double vnorm = ax_vec_norm(geo->v, d);
            if (vnorm > 1e10 || vnorm != vnorm) {
                for (int s = 0; s < 6; s++) { tensor_free(kx[s]); tensor_free(kv[s]); }
                tensor_free(xt); tensor_free(vt); tensor_free(x4); tensor_free(v4);
                tensor_free(x5); tensor_free(v5); tensor_free(gl);
                return -1;
            }
        } else {
            rejected++;
        }
        h = h_new;
        (void)rejected;
    }

    for (int s = 0; s < 6; s++) { tensor_free(kx[s]); tensor_free(kv[s]); }
    tensor_free(xt); tensor_free(vt); tensor_free(x4); tensor_free(v4);
    tensor_free(x5); tensor_free(v5); tensor_free(gl);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Geodesic Trajectory Cache — item 4
 *
 * Memoize solved geodesics keyed by (start_cluster, end_cluster).
 * Cluster assignment = argmin distance to centroid set.
 * Hit: return stored endpoint immediately (zero integration cost).
 * ═══════════════════════════════════════════════════════════════════════════ */

void axgeo_traj_cache_init(axgeo_traj_cache_t *tc, int dim)
{
    if (!tc) return;
    memset(tc, 0, sizeof(*tc));
    tc->dim = dim;
}

void axgeo_traj_cache_destroy(axgeo_traj_cache_t *tc)
{
    if (!tc) return;
    if (tc->centroids) { tensor_free(tc->centroids); tc->centroids = NULL; }
    for (int i = 0; i < AXGEO_TRAJ_CACHE_CAP; i++) {
        if (tc->entries[i].endpoint) { tensor_free(tc->entries[i].endpoint); tc->entries[i].endpoint = NULL; }
        if (tc->entries[i].velocity) { tensor_free(tc->entries[i].velocity); tc->entries[i].velocity = NULL; }
        tc->entries[i].valid = 0;
    }
    tc->count = 0;
    tc->n_clusters = 0;
}

void axgeo_traj_cache_flush(axgeo_traj_cache_t *tc)
{
    if (!tc) return;
    for (int i = 0; i < AXGEO_TRAJ_CACHE_CAP; i++) {
        if (tc->entries[i].endpoint) { tensor_free(tc->entries[i].endpoint); tc->entries[i].endpoint = NULL; }
        if (tc->entries[i].velocity) { tensor_free(tc->entries[i].velocity); tc->entries[i].velocity = NULL; }
        tc->entries[i].valid = 0;
    }
    tc->count = 0;
    tc->hits = tc->misses = 0;
}

static int traj_nearest_centroid(const axgeo_traj_cache_t *tc, const double *pt)
{
    if (tc->n_clusters == 0) return -1;
    int best = 0;
    double best_d2 = 1e300;
    int d = tc->dim;
    for (int c = 0; c < tc->n_clusters; c++) {
        const double *cen = tc->centroids + (uint64_t)c * d;
        double d2 = 0.0;
        for (int i = 0; i < d; i++) { double diff = pt[i] - cen[i]; d2 += diff*diff; }
        if (d2 < best_d2) { best_d2 = d2; best = c; }
    }
    return best;
}

int axgeo_traj_cache_add_centroid(axgeo_traj_cache_t *tc, const double *pt)
{
    if (!tc || !pt) return -1;
    if (tc->n_clusters >= AXGEO_TRAJ_CLUSTER_K) return traj_nearest_centroid(tc, pt);
    int d = tc->dim;
    double *new_cens = (double *)tensor_alloc((uint64_t)(tc->n_clusters + 1) * d * sizeof(double));
    if (!new_cens) return -1;
    if (tc->centroids) {
        memcpy(new_cens, tc->centroids, (uint64_t)tc->n_clusters * d * sizeof(double));
        tensor_free(tc->centroids);
    }
    memcpy(new_cens + (uint64_t)tc->n_clusters * d, pt, (uint64_t)d * sizeof(double));
    tc->centroids = new_cens;
    return tc->n_clusters++;
}

int axgeo_traj_cache_lookup(axgeo_traj_cache_t *tc,
                             const double *start, const double *end,
                             double *out_endpoint, double *out_vel)
{
    if (!tc || tc->count == 0 || tc->n_clusters == 0) { tc->misses++; return 0; }
    int cs = traj_nearest_centroid(tc, start);
    int ce = traj_nearest_centroid(tc, end);
    for (int i = 0; i < AXGEO_TRAJ_CACHE_CAP; i++) {
        const axgeo_traj_entry_t *e = &tc->entries[i];
        if (e->valid && e->start_cluster == cs && e->end_cluster == ce) {
            if (out_endpoint) memcpy(out_endpoint, e->endpoint, (uint64_t)e->dim * sizeof(double));
            if (out_vel)      memcpy(out_vel,      e->velocity, (uint64_t)e->dim * sizeof(double));
            tc->hits++;
            return 1;
        }
    }
    tc->misses++;
    return 0;
}

void axgeo_traj_cache_insert(axgeo_traj_cache_t *tc,
                              const double *start, const double *end,
                              const double *endpoint, const double *velocity,
                              int n_steps)
{
    if (!tc || !start || !end || !endpoint) return;
    int cs = axgeo_traj_cache_add_centroid(tc, start);
    int ce = axgeo_traj_cache_add_centroid(tc, end);
    /* Find an empty slot or overwrite oldest */
    int slot = tc->count % AXGEO_TRAJ_CACHE_CAP;
    axgeo_traj_entry_t *e = &tc->entries[slot];
    int d = tc->dim;
    if (!e->endpoint || e->dim != d) {
        if (e->endpoint) tensor_free(e->endpoint);
        if (e->velocity) tensor_free(e->velocity);
        e->endpoint = (double *)tensor_alloc((uint64_t)d * sizeof(double));
        e->velocity = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    }
    if (!e->endpoint || !e->velocity) return;
    memcpy(e->endpoint, endpoint, (uint64_t)d * sizeof(double));
    if (velocity) memcpy(e->velocity, velocity, (uint64_t)d * sizeof(double));
    else memset(e->velocity, 0, (uint64_t)d * sizeof(double));
    e->start_cluster = cs;
    e->end_cluster   = ce;
    e->dim           = d;
    e->n_steps       = n_steps;
    e->valid         = 1;
    tc->count++;
}

double axgeo_geodesic_length(const axgeo_geodesic_t *geo,
                             const axgeo_metric_field_t *mf)
{
    if (!geo->trajectory || geo->steps < 2) return 0.0;

    int d = geo->dim;
    double total = 0.0;
    double *g = (double *)tensor_alloc((uint64_t)d * d * sizeof(double));
    if (!g) return 0.0;

    for (int s = 1; s <= geo->steps; s++) {
        const double *p0 = geo->trajectory + (uint64_t)(s - 1) * d;
        const double *p1 = geo->trajectory + (uint64_t)s * d;

        /* Midpoint metric */
        double *mid = (double *)tensor_alloc((uint64_t)d * sizeof(double));
        if (!mid) break;
        for (int i = 0; i < d; i++) mid[i] = 0.5 * (p0[i] + p1[i]);
        axgeo_metric_interpolate(mf, mid, g);

        /* ds² = g_ij dx^i dx^j */
        double ds2 = 0.0;
        for (int i = 0; i < d; i++) {
            double dxi = p1[i] - p0[i];
            for (int j = 0; j < d; j++) {
                double dxj = p1[j] - p0[j];
                ds2 += g[i * d + j] * dxi * dxj;
            }
        }
        if (ds2 > 0.0) total += sqrt(ds2);
        tensor_free(mid);
    }

    tensor_free(g);
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Fisher Information Metric
 * ═══════════════════════════════════════════════════════════════════════════ */

axgeo_fisher_t axgeo_fisher_create(int dim)
{
    axgeo_fisher_t f;
    f.dim = dim;
    f.matrix = (double *)tensor_alloc((uint64_t)dim * dim * sizeof(double));
    if (f.matrix) memset(f.matrix, 0,
                          (uint64_t)dim * dim * sizeof(double));
    f.trace = 0.0;
    f.det_log = 0.0;
    return f;
}

void axgeo_fisher_destroy(axgeo_fisher_t *f)
{
    if (!f) return;
    if (f->matrix) { tensor_free(f->matrix); f->matrix = NULL; }
    f->dim = 0;
}

int axgeo_compute_fisher(axgeo_fisher_t *f,
                         const axgeo_metric_field_t *mf,
                         int point_idx)
{
    int d = mf->dim;
    int dd = d * d;
    if (point_idx < 0 || point_idx >= mf->n_points) return -1;
    if (!f->matrix || f->dim != d) return -1;

    /* Fisher ≈ g⁻¹ (inverse of the covariance metric tensor).
     * For a Gaussian distribution this is exact; for the embedding
     * manifold it approximates the information-geometric metric. */
    const double *g = mf->metrics + (uint64_t)point_idx * dd;
    if (invert_symmetric(g, f->matrix, d) != 0) return -1;

    /* Trace and log-determinant */
    f->trace = 0.0;
    for (int i = 0; i < d; i++)
        f->trace += f->matrix[i * d + i];

    /* Log-det(F) = -log-det(g).  Compute log-det(g) from diagonal of
     * the Cholesky-like factorization used during inversion.
     * Simpler: det_log = Σ log(eigenvalues of F).
     * Approximate via Tr(log(F)) ≈ log(det(F))/d... too rough.
     * Instead compute det(g) via the product of pivots from our
     * Gauss-Jordan.  For now use a simple LU-like approach. */
    double *tmp = (double *)tensor_alloc((uint64_t)dd * sizeof(double));
    if (!tmp) { f->det_log = 0.0; return 0; }
    memcpy(tmp, g, (uint64_t)dd * sizeof(double));

    double log_det_g = 0.0;
    for (int i = 0; i < d; i++) {
        /* Partial pivoting */
        int piv = i;
        for (int k = i + 1; k < d; k++)
            if (fabs(tmp[k * d + i]) > fabs(tmp[piv * d + i])) piv = k;
        if (piv != i) {
            for (int j = 0; j < d; j++) {
                double t = tmp[i * d + j];
                tmp[i * d + j] = tmp[piv * d + j];
                tmp[piv * d + j] = t;
            }
        }
        double diag = tmp[i * d + i];
        if (fabs(diag) < 1e-300) diag = 1e-300;
        log_det_g += log(fabs(diag));
        for (int k = i + 1; k < d; k++) {
            double factor = tmp[k * d + i] / diag;
            for (int j = i + 1; j < d; j++)
                tmp[k * d + j] -= factor * tmp[i * d + j];
        }
    }
    tensor_free(tmp);
    f->det_log = -log_det_g;  /* det(F) = 1/det(g) */

    return 0;
}

void axgeo_metric_blend_fisher(axgeo_metric_field_t *mf,
                               const axgeo_fisher_t *f,
                               int point_idx, double alpha)
{
    int d = mf->dim;
    int dd = d * d;
    if (point_idx < 0 || point_idx >= mf->n_points) return;
    if (!f->matrix || f->dim != d) return;
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;

    double *g = mf->metrics + (uint64_t)point_idx * dd;
    double inv_alpha = 1.0 - alpha;
    for (int i = 0; i < dd; i++)
        g[i] = inv_alpha * g[i] + alpha * f->matrix[i];
}

int axgeo_apply_local_warp(axgeo_christoffel_t *ch,
                           const axgeo_metric_field_t *mf,
                           const double *p,
                           const double *phi,
                           double alpha,
                           double sigma)
{
    if (!ch || !mf || !p || !phi || !ch->gamma || !mf->points || sigma <= 0.0)
        return 0;
    if (ch->dim != mf->dim || ch->n_points != mf->n_points)
        return 0;

    int np = ch->n_points;
    int d = ch->dim;
    int ddd = d * d * d;
    int touched = 0;

    double sigma2 = sigma * sigma;
    double *g_local = (double *)tensor_alloc((uint64_t)d * d * sizeof(double));
    if (!g_local) return 0;

    for (int q = 0; q < np; q++) {
        const double *xq = mf->points + (uint64_t)q * d;
        double *gamma_q = ch->gamma + (uint64_t)q * ddd;

        axgeo_metric_interpolate(mf, xq, g_local);

        /* Approximate d_g(xq,p)^2 via local quadratic form. */
        double dist2 = 0.0;
        for (int i = 0; i < d; i++) {
            double di = xq[i] - p[i];
            for (int j = 0; j < d; j++) {
                double dj = xq[j] - p[j];
                dist2 += g_local[i * d + j] * di * dj;
            }
        }
        if (dist2 < 0.0) dist2 = -dist2;

        double w = exp(-dist2 / (2.0 * sigma2));
        if (w < 1e-8) continue;

        double scale = alpha * w;
        for (int idx = 0; idx < ddd; idx++)
            gamma_q[idx] += scale * phi[idx];

        touched++;
    }

    tensor_free(g_local);
    return touched;
}

int axgeo_apply_local_warp_many(axgeo_christoffel_t *ch,
                                const axgeo_metric_field_t *mf,
                                const double *points,
                                const double *phis,
                                int n_points,
                                double alpha,
                                double sigma)
{
    if (!points || !phis || n_points <= 0 || !ch || !mf) return 0;

    int d = ch->dim;
    int ddd = d * d * d;
    int total_touched = 0;

    for (int n = 0; n < n_points; n++) {
        const double *p = points + (uint64_t)n * d;
        const double *phi = phis + (uint64_t)n * ddd;
        total_touched += axgeo_apply_local_warp(ch, mf, p, phi, alpha, sigma);
    }

    return total_touched;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight-Derived Pullback Metric (Step 1 of OTT k⁴ plan)
 *
 * Build a Riemannian metric field from the actual transformer weight matrices.
 * The pullback metric G = (W·U)^T (W·U) captures the geometry that each
 * weight matrix imposes on the k-dimensional PCA subspace U.
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * axgeo_dequant_row_f64 — thin wrapper kept for axiom_geo.c internal use.
 * Delegates to ax_dequant_row from axiom_linalg.h.
 */
int axgeo_dequant_row_f64(const void *row_ptr, double *out, int dim, int type)
{
    return ax_dequant_row(row_ptr, out, dim, (ggml_type_t)type);
}

/*
 * axgeo_pullback_metric — compute G = (W·U)^T (W·U) ∈ R^{k×k}.
 *
 * For each row w_i ∈ R^d of W, compute the projected row:
 *   w̃_i = U^T w_i  ∈ R^k
 * Then accumulate:
 *   G += w̃_i ⊗ w̃_i   (outer product)
 *
 * The result G is a positive semi-definite k×k matrix representing the
 * pullback of the Euclidean metric on the output space through U.
 *
 * We normalise by n_rows so that G is the *mean* outer product — this makes
 * the metric scale-invariant to layer width.
 */
int axgeo_pullback_metric(const void *W_ptr, int w_type,
                          int n_rows, int n_cols,
                          const double *U, int k,
                          double *G_out,
                          double *row_buf, double *wu_buf)
{
    if (!W_ptr || !U || !G_out || !row_buf || !wu_buf) return -1;
    if (n_rows <= 0 || n_cols <= 0 || k <= 0 || k > AXGEO_MAX_DIM) return -1;

    /* Compute bytes per row for pointer arithmetic */
    uint64_t row_stride;
    switch (w_type) {
        case 10: /* GGML_TYPE_Q4_0 */
            row_stride = (uint64_t)(n_cols / 32) * 18;
            break;
        case 8:  /* GGML_TYPE_Q8_0 */
            row_stride = (uint64_t)(n_cols / 32) * 34;
            break;
        case 0:  /* GGML_TYPE_F32 */
            row_stride = (uint64_t)n_cols * 4;
            break;
        case 1:  /* GGML_TYPE_F16 */
            row_stride = (uint64_t)n_cols * 2;
            break;
        default:
            return -1;
    }

    /* Zero output metric */
    memset(G_out, 0, (uint64_t)k * k * sizeof(double));

    const uint8_t *wbase = (const uint8_t *)W_ptr;

    /* Process rows in blocks to amortize allocation cost.
     * wu_buf must hold at least n_rows × k doubles. */
    for (int r = 0; r < n_rows; r++) {
        const void *row_ptr = wbase + (uint64_t)r * row_stride;

        /* Dequantize row r → row_buf ∈ R^d */
        if (ax_dequant_row(row_ptr, row_buf, n_cols, (ggml_type_t)w_type) != 0)
            continue;

        /* Project onto PCA subspace: w̃ = U^T w ∈ R^k
         * U is stored as U[col][dim_idx]: U[j*n_cols + i] = U_{ij}
         * (column-major: each column of U is a basis vector in R^d)
         */
        double *w_tilde = wu_buf + (uint64_t)r * k;
        for (int j = 0; j < k; j++) {
            double dot = 0.0;
            const double *u_col = U + (uint64_t)j * n_cols;
            for (int i = 0; i < n_cols; i++)
                dot += u_col[i] * row_buf[i];
            w_tilde[j] = dot;
        }

        /* Accumulate outer product w̃ ⊗ w̃ into G */
        for (int a = 0; a < k; a++) {
            for (int b = a; b < k; b++) {
                double v = w_tilde[a] * w_tilde[b];
                G_out[a * k + b] += v;
                if (b != a) G_out[b * k + a] += v;
            }
        }
    }

    /* Normalise by n_rows */
    double inv_n = (n_rows > 0) ? 1.0 / (double)n_rows : 1.0;
    for (int i = 0; i < k * k; i++)
        G_out[i] *= inv_n;

    /* Add small regularization to the diagonal for numerical stability */
    for (int i = 0; i < k; i++)
        G_out[i * k + i] += 1e-8;

    return 0;
}

/*
 * axgeo_build_metric_from_weights — construct a metric field from the actual
 * transformer weight matrices across all layers.
 *
 * Strategy:
 * 1. For each sample point p (index into mf->points):
 *    - Determine which layer L this point is "closest to" by mapping p index
 *      uniformly across [0, n_layers].
 *    - Compute per-layer pullback metric G_L = (W_L · U)^T (W_L · U)
 *    - Blend with adjacent layers for smoothness.
 * 2. Set mf->metrics[p] = blended G.
 * 3. If sample_pts is provided, copy the sample points into mf->points.
 *    Otherwise, leave mf->points as-is (caller should set them up).
 */
int axgeo_build_metric_from_weights(axgeo_metric_field_t *mf,
                                    const double *U, int k, int d,
                                    const void **layer_weights,
                                    const int *layer_types,
                                    const int *n_rows_each,
                                    int n_layers,
                                    const double *sample_pts)
{
    if (!mf || !U || k <= 0 || d <= 0 || !layer_weights || !layer_types || !n_rows_each || n_layers <= 0)
        return -1;
    if (mf->n_points <= 0 || mf->dim != k) return -1;

    int np = mf->n_points;

    /* Copy sample points if provided */
    if (sample_pts) {
        memcpy(mf->points, sample_pts, (uint64_t)np * k * sizeof(double));
    }

    /* Allocate scratch buffers */
    int max_rows = 0;
    for (int L = 0; L < n_layers; L++)
        if (n_rows_each[L] > max_rows) max_rows = n_rows_each[L];

    double *row_buf = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    double *wu_buf  = (double *)tensor_alloc((uint64_t)max_rows * k * sizeof(double));
    double *G_layer = (double *)tensor_alloc((uint64_t)k * k * sizeof(double));
    double *G_blend = (double *)tensor_alloc((uint64_t)k * k * sizeof(double));

    if (!row_buf || !wu_buf || !G_layer || !G_blend) {
        if (row_buf)  tensor_free(row_buf);
        if (wu_buf)   tensor_free(wu_buf);
        if (G_layer)  tensor_free(G_layer);
        if (G_blend)  tensor_free(G_blend);
        return -1;
    }

    int kk = k * k;

    /* Precompute per-layer pullback metrics.
     * Store in a temporary array so we can blend adjacent layers. */
    double *all_G = (double *)tensor_alloc((uint64_t)n_layers * kk * sizeof(double));
    if (!all_G) {
        tensor_free(row_buf); tensor_free(wu_buf);
        tensor_free(G_layer); tensor_free(G_blend);
        return -1;
    }

    for (int L = 0; L < n_layers; L++) {
        if (!layer_weights[L]) {
            /* No weight for this layer — use identity metric */
            double *G_L = all_G + (uint64_t)L * kk;
            memset(G_L, 0, (uint64_t)kk * sizeof(double));
            for (int i = 0; i < k; i++) G_L[i * k + i] = 1.0;
            continue;
        }

        int rc = axgeo_pullback_metric(layer_weights[L], layer_types[L],
                                       n_rows_each[L], d,
                                       U, k,
                                       all_G + (uint64_t)L * kk,
                                       row_buf, wu_buf);
        if (rc != 0) {
            /* Fallback: identity */
            double *G_L = all_G + (uint64_t)L * kk;
            memset(G_L, 0, (uint64_t)kk * sizeof(double));
            for (int i = 0; i < k; i++) G_L[i * k + i] = 1.0;
        }
    }

    /* Assign metrics to sample points.
     * Each sample point p ∈ [0, np) maps to a fractional layer index
     * f = p * (n_layers - 1) / (np - 1). Blend the two neighboring layers. */
    for (int p = 0; p < np; p++) {
        double f = (np > 1) ? (double)p * (double)(n_layers - 1) / (double)(np - 1) : 0.0;
        int L0 = (int)f;
        int L1 = L0 + 1;
        if (L0 < 0) L0 = 0;
        if (L1 >= n_layers) L1 = n_layers - 1;
        double t = f - (double)L0;  /* blend weight [0,1] */

        const double *G0 = all_G + (uint64_t)L0 * kk;
        const double *G1 = all_G + (uint64_t)L1 * kk;
        double *G_p = axgeo_metric_at(mf, p);

        for (int i = 0; i < kk; i++)
            G_p[i] = (1.0 - t) * G0[i] + t * G1[i];
    }

    tensor_free(all_G);
    tensor_free(row_buf);
    tensor_free(wu_buf);
    tensor_free(G_layer);
    tensor_free(G_blend);
    return 0;
}

/*
 * axgeo_apply_rmsnorm_connection — apply the Levi-Civita connection correction
 * for RMSNorm on the (k-1)-sphere.
 *
 * For RMSNorm(p) = p / ||p||, the Christoffel correction implements the
 * diffeomorphism ϕ that absorbs the normalization into the connection:
 *
 *   ΔΓ^μ_νρ = -(δ^μ_ρ p_ν + δ^μ_ν p_ρ)/||p||² + 2 p^μ p_ν p_ρ/||p||⁴
 *
 * This is the pullback of the sphere connection onto the ambient R^k space.
 * Blended by alpha ∈ [0,1] for gradual activation.
 */
void axgeo_apply_rmsnorm_connection(double *gamma, const double *p,
                                    int k, double alpha)
{
    if (!gamma || !p || k <= 0 || alpha <= 0.0) return;

    double norm2 = 0.0;
    for (int i = 0; i < k; i++) norm2 += p[i] * p[i];
    if (norm2 < 1e-20) return;  /* zero vector — no correction */

    double inv_n2 = 1.0 / norm2;
    double inv_n4 = inv_n2 * inv_n2;

    /* ΔΓ^μ_νρ = alpha * [-(δ^μ_ρ p_ν + δ^μ_ν p_ρ)/||p||² + 2 p^μ p_ν p_ρ/||p||⁴] */
    for (int mu = 0; mu < k; mu++) {
        for (int nu = 0; nu < k; nu++) {
            for (int rho = 0; rho < k; rho++) {
                double delta_mu_rho = (mu == rho) ? 1.0 : 0.0;
                double delta_mu_nu  = (mu == nu)  ? 1.0 : 0.0;

                double corr = -(delta_mu_rho * p[nu] + delta_mu_nu * p[rho]) * inv_n2
                              + 2.0 * p[mu] * p[nu] * p[rho] * inv_n4;

                gamma[mu * k * k + nu * k + rho] += alpha * corr;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * axgeo_estimate_injectivity_radius
 *
 * Proper curvature-based injectivity radius estimate at q_bar.
 * Uses the Frobenius norm of the Christoffel tensor as a proxy for sectional
 * curvature:  K_est ≈ ||Γ||²_F / (k(k-1)+1)
 * Inj radius ≥ π / sqrt(K_max) (standard Riemannian comparison theorem).
 * ═══════════════════════════════════════════════════════════════════════════ */
double axgeo_estimate_injectivity_radius(const axgeo_christoffel_t *ch,
                                          const axgeo_metric_field_t *mf,
                                          const double *q_bar, int k)
{
    if (!ch || !mf || !q_bar || k <= 0) return 0.05;

    int kkk = k * k * k;
    double *gamma = (double *)tensor_alloc((uint64_t)kkk * sizeof(double));
    if (!gamma) return 0.05;

    axgeo_christoffel_interpolate(ch, mf, q_bar, gamma);

    double frob2 = 0.0;
    for (int i = 0; i < kkk; i++) frob2 += gamma[i] * gamma[i];
    tensor_free(gamma);

    double denom = (double)(k * (k - 1) + 1);
    double K_est = frob2 / denom;

    double rho;
    if (K_est > 1e-10)
        rho = 3.14159265358979323846 / sqrt(K_est);
    else
        rho = 5.0;  /* flat region: large validity radius */

    /* Clamp to practical range */
    if (rho < 0.05) rho = 0.05;
    if (rho > 5.0)  rho = 5.0;
    return rho;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * axgeo_compute_jacobi_propagator (defined below)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════
 * axgeo_compute_jacobi_propagator
 *
 * Integrate the Jacobi ODE along a precomputed geodesic trajectory to obtain
 * J(λ_f) ∈ R^{k×k}.
 *
 * The Jacobi equation for a deviation field J^α (one column):
 *   dJ/dλ = Z
 *   dZ/dλ = -K(λ) · J
 *
 * where K^α_β(λ) = R^α_μβν v^μ v^ν is the geodesic tidal operator.
 * We approximate K from the Christoffel symbols via:
 *   K^α_β ≈ (Γ^α_μγ Γ^γ_νβ - Γ^α_νγ Γ^γ_μβ) v^μ v^ν   (flat-space Riemann)
 *
 * For k ≤ 30, this is O(k^3) per waypoint — negligible.
 * Initial conditions: J(0) = I (identity), Z(0) = 0.
 * ═══════════════════════════════════════════════════════════════════════════
 */
int axgeo_compute_jacobi_propagator(const double *trajectory,
                                     const double *velocities,
                                     int n_wp, int k,
                                     const axgeo_christoffel_t *ch,
                                     const axgeo_metric_field_t *mf,
                                     double *J_out)
{
    if (!trajectory || !J_out || n_wp < 2 || k <= 0 || !ch || !mf) return -1;

    int kk = k * k;

    /* J[k×k] and Z[k×k] (dJ/dλ) — stored row-major J[α*k + β] */
    double *J = (double *)tensor_alloc((uint64_t)kk * sizeof(double));
    double *Z = (double *)tensor_alloc((uint64_t)kk * sizeof(double));
    double *K = (double *)tensor_alloc((uint64_t)kk * sizeof(double));
    double *gamma_wp = (double *)tensor_alloc((uint64_t)k * kk * sizeof(double));
    double *v_est    = (double *)tensor_alloc((uint64_t)k * sizeof(double));

    if (!J || !Z || !K || !gamma_wp || !v_est) {
        if (J)        tensor_free(J);
        if (Z)        tensor_free(Z);
        if (K)        tensor_free(K);
        if (gamma_wp) tensor_free(gamma_wp);
        if (v_est)    tensor_free(v_est);
        return -1;
    }

    /* Initial conditions: J = identity, Z = 0 */
    memset(J, 0, (uint64_t)kk * sizeof(double));
    memset(Z, 0, (uint64_t)kk * sizeof(double));
    for (int i = 0; i < k; i++) J[i * k + i] = 1.0;

    double dt = 1.0 / (double)(n_wp - 1);  /* affine parameter step */

    for (int wi = 0; wi < n_wp - 1; wi++) {
        const double *pos = trajectory + (uint64_t)wi * k;
        const double *pos_next = trajectory + (uint64_t)(wi + 1) * k;

        /* Velocity at this waypoint */
        const double *v;
        if (velocities) {
            v = velocities + (uint64_t)wi * k;
        } else {
            /* Estimate from finite difference */
            for (int i = 0; i < k; i++)
                v_est[i] = (pos_next[i] - pos[i]) / dt;
            v = v_est;
        }

        /* Interpolate Christoffel symbols Γ^α_μν at this waypoint */
        axgeo_christoffel_interpolate(ch, mf, pos, gamma_wp);

        /* Compute tidal operator K^α_β = R^α_μβν v^μ v^ν
         * Approximation via Christoffel products (flat-space Riemann):
         *   K^α_β ≈ Σ_{μ,ν,γ} (Γ^α_μγ Γ^γ_νβ - Γ^α_νγ Γ^γ_μβ) v^μ v^ν
         */
        memset(K, 0, (uint64_t)kk * sizeof(double));
        for (int alpha = 0; alpha < k; alpha++) {
            for (int beta = 0; beta < k; beta++) {
                double kval = 0.0;
                for (int mu = 0; mu < k; mu++) {
                    for (int nu = 0; nu < k; nu++) {
                        double vv = v[mu] * v[nu];
                        if (vv == 0.0) continue;
                        for (int gamma_idx = 0; gamma_idx < k; gamma_idx++) {
                            /* Γ^α_μγ * Γ^γ_νβ */
                            kval += gamma_wp[alpha * kk + mu * k + gamma_idx]
                                  * gamma_wp[gamma_idx * kk + nu * k + beta] * vv;
                            /* - Γ^α_νγ * Γ^γ_μβ */
                            kval -= gamma_wp[alpha * kk + nu * k + gamma_idx]
                                  * gamma_wp[gamma_idx * kk + mu * k + beta] * vv;
                        }
                    }
                }
                K[alpha * k + beta] = kval;
            }
        }

        /* Euler step: Z_new = Z - K·J·dt, J_new = J + Z·dt */
        double *J_new = (double *)tensor_alloc((uint64_t)kk * sizeof(double));
        double *Z_new = (double *)tensor_alloc((uint64_t)kk * sizeof(double));
        if (!J_new || !Z_new) {
            if (J_new) tensor_free(J_new);
            if (Z_new) tensor_free(Z_new);
            break;
        }

        /* Z_new^α_i = Z^α_i - (K·J)^α_i · dt */
        for (int alpha = 0; alpha < k; alpha++) {
            for (int i = 0; i < k; i++) {
                double kj = 0.0;
                for (int beta = 0; beta < k; beta++)
                    kj += K[alpha * k + beta] * J[beta * k + i];
                Z_new[alpha * k + i] = Z[alpha * k + i] - kj * dt;
            }
        }
        /* J_new^α_i = J^α_i + Z^α_i · dt */
        for (int alpha = 0; alpha < k; alpha++) {
            for (int i = 0; i < k; i++)
                J_new[alpha * k + i] = J[alpha * k + i] + Z[alpha * k + i] * dt;
        }

        tensor_free(J); tensor_free(Z);
        J = J_new; Z = Z_new;
    }

    memcpy(J_out, J, (uint64_t)kk * sizeof(double));

    tensor_free(J); tensor_free(Z); tensor_free(K);
    tensor_free(gamma_wp); tensor_free(v_est);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GRC Library — ANN bucket index + AttnRes waypoints
 * ═══════════════════════════════════════════════════════════════════════════ */

axgeo_grc_library_t axgeo_grc_library_create(int k, int cap)
{
    axgeo_grc_library_t lib;
    memset(&lib, 0, sizeof(lib));
    if (k <= 0 || cap <= 0) return lib;

    lib.k         = k;
    lib.cap       = cap;
    lib.write_idx = 0;

    int kk  = k * k;
    int nwp = AXGEO_GRC_N_WP;

    lib.q_bars           = (double *)tensor_alloc((uint64_t)cap * k      * sizeof(double));
    lib.Js               = (double *)tensor_alloc((uint64_t)cap * kk     * sizeof(double));
    lib.x_ends           = (double *)tensor_alloc((uint64_t)cap * k      * sizeof(double));
    lib.rhos             = (double *)tensor_alloc((uint64_t)cap          * sizeof(double));
    lib.best_toks        = (int    *)tensor_alloc((uint64_t)cap          * sizeof(int));
    lib.x_wps            = (double *)tensor_alloc((uint64_t)cap * nwp * k* sizeof(double));
    lib.bucket_centroids = (double *)tensor_alloc((uint64_t)AXGEO_GRC_N_BUCKETS * k * sizeof(double));
    lib.bucket_assign    = (int    *)tensor_alloc((uint64_t)cap          * sizeof(int));

    if (!lib.q_bars || !lib.Js || !lib.x_ends || !lib.rhos || !lib.best_toks
            || !lib.x_wps || !lib.bucket_centroids || !lib.bucket_assign) {
        axgeo_grc_library_destroy(&lib);
        memset(&lib, 0, sizeof(lib));
        return lib;
    }

    memset(lib.bucket_assign,    -1, (uint64_t)cap          * sizeof(int));
    memset(lib.bucket_counts,     0, sizeof(lib.bucket_counts));
    memset(lib.bucket_centroids,  0, (uint64_t)AXGEO_GRC_N_BUCKETS * k * sizeof(double));
    return lib;
}

void axgeo_grc_library_destroy(axgeo_grc_library_t *lib)
{
    if (!lib) return;
    if (lib->q_bars)           tensor_free(lib->q_bars);
    if (lib->Js)               tensor_free(lib->Js);
    if (lib->x_ends)           tensor_free(lib->x_ends);
    if (lib->rhos)             tensor_free(lib->rhos);
    if (lib->best_toks)        tensor_free(lib->best_toks);
    if (lib->x_wps)            tensor_free(lib->x_wps);
    if (lib->bucket_centroids) tensor_free(lib->bucket_centroids);
    if (lib->bucket_assign)    tensor_free(lib->bucket_assign);
    memset(lib, 0, sizeof(*lib));
}

/* ─── ANN bucket helpers ── */

static int grc_nearest_bucket(const axgeo_grc_library_t *lib, const double *q)
{
    int k = lib->k;
    int best = 0;
    double best_d2 = 1e300;
    for (int b = 0; b < lib->n_buckets_used; b++) {
        const double *c = lib->bucket_centroids + (uint64_t)b * k;
        double d2 = 0.0;
        for (int i = 0; i < k; i++) { double diff = q[i] - c[i]; d2 += diff * diff; }
        if (d2 < best_d2) { best_d2 = d2; best = b; }
    }
    return best;
}

static void grc_update_centroid(axgeo_grc_library_t *lib, int b, const double *q)
{
    int k = lib->k;
    double *c = lib->bucket_centroids + (uint64_t)b * k;
    int cnt = lib->bucket_counts[b];
    /* Online mean update: c = c*(cnt-1)/cnt + q/cnt */
    if (cnt <= 0) cnt = 1;
    double inv = 1.0 / (double)cnt;
    double scale = (double)(cnt - 1) * inv;
    for (int i = 0; i < k; i++)
        c[i] = c[i] * scale + q[i] * inv;
}

void axgeo_grc_insert(axgeo_grc_library_t *lib,
                      const double *q_bar, const double *J,
                      const double *x_end, double rho, int best_tok,
                      const double *x_wps_in)
{
    if (!lib || !lib->q_bars || lib->k <= 0 || !q_bar || !J || !x_end) return;

    int k    = lib->k;
    int kk   = k * k;
    int nwp  = AXGEO_GRC_N_WP;
    int nb   = AXGEO_GRC_N_BUCKETS;

    /* Ring-buffer slot selection */
    int slot = lib->write_idx;
    lib->write_idx = (lib->write_idx + 1) % lib->cap;
    if (lib->count < lib->cap) lib->count++;

    /* If this slot was occupied, decrement old bucket count */
    if (lib->bucket_assign[slot] >= 0 && lib->count > 1) {
        int old_b = lib->bucket_assign[slot];
        if (old_b < nb && lib->bucket_counts[old_b] > 0)
            lib->bucket_counts[old_b]--;
    }

    /* Store record */
    memcpy(lib->q_bars + (uint64_t)slot * k,  q_bar, (uint64_t)k  * sizeof(double));
    memcpy(lib->Js     + (uint64_t)slot * kk, J,     (uint64_t)kk * sizeof(double));
    memcpy(lib->x_ends + (uint64_t)slot * k,  x_end, (uint64_t)k  * sizeof(double));
    lib->rhos[slot]      = rho;
    lib->best_toks[slot] = best_tok;

    if (x_wps_in) {
        memcpy(lib->x_wps + (uint64_t)slot * nwp * k, x_wps_in,
               (uint64_t)nwp * k * sizeof(double));
    } else {
        memset(lib->x_wps + (uint64_t)slot * nwp * k, 0,
               (uint64_t)nwp * k * sizeof(double));
    }

    /* Assign bucket (ANN index) */
    int bucket;
    if (lib->n_buckets_used < nb) {
        /* Seed a new centroid */
        bucket = lib->n_buckets_used++;
        memcpy(lib->bucket_centroids + (uint64_t)bucket * k, q_bar,
               (uint64_t)k * sizeof(double));
        lib->bucket_counts[bucket] = 0;
    } else {
        bucket = grc_nearest_bucket(lib, q_bar);
    }
    lib->bucket_assign[slot] = bucket;
    lib->bucket_counts[bucket]++;
    grc_update_centroid(lib, bucket, q_bar);
}

/* ─── Core lookup (shared between lookup and lookup_with_summaries) ── */

static int grc_find_best(axgeo_grc_library_t *lib, const double *q,
                          int *best_slot_out)
{
    if (!lib || !lib->q_bars || lib->count <= 0) return 0;

    int k  = lib->k;
    int nb = lib->n_buckets_used;

    /* Stage 1: find the 2 nearest buckets */
    int   top_b[2]  = {0, 0};
    double top_d2[2] = {1e300, 1e300};
    for (int b = 0; b < nb; b++) {
        const double *c = lib->bucket_centroids + (uint64_t)b * k;
        double d2 = 0.0;
        for (int i = 0; i < k; i++) { double diff = q[i] - c[i]; d2 += diff * diff; }
        if (d2 < top_d2[0]) {
            top_d2[1] = top_d2[0]; top_b[1] = top_b[0];
            top_d2[0] = d2;        top_b[0] = b;
        } else if (d2 < top_d2[1]) {
            top_d2[1] = d2; top_b[1] = b;
        }
    }

    /* Stage 2: scan only records in those 2 buckets (+ overflow: also scan
     * any record whose bucket was evicted so bucket_assign is stale) */
    int    best_slot = -1;
    double best_dist = 1e300;
    int n = lib->count;
    for (int i = 0; i < n; i++) {
        int ba = lib->bucket_assign[i];
        if (ba != top_b[0] && ba != top_b[1] && nb > 2) continue;
        const double *qb = lib->q_bars + (uint64_t)i * k;
        double d2 = 0.0;
        for (int j = 0; j < k; j++) {
            double diff = q[j] - qb[j]; d2 += diff * diff;
        }
        if (d2 < best_dist) { best_dist = d2; best_slot = i; }
    }

    if (best_slot < 0) return 0;

    double dist = sqrt(best_dist);
    double rho  = lib->rhos[best_slot];
    if (dist >= rho) return 0;

    *best_slot_out = best_slot;
    return 1;
}

int axgeo_grc_lookup(axgeo_grc_library_t *lib,
                     const double *q, int k,
                     double *x_end_out,
                     int *best_tok_out)
{
    if (!x_end_out || !best_tok_out || !lib || lib->k != k) return 0;

    int slot = -1;
    if (!grc_find_best(lib, q, &slot)) { lib->misses++; return 0; }

    int kk = k * k;
    const double *J    = lib->Js     + (uint64_t)slot * kk;
    const double *xbar = lib->x_ends + (uint64_t)slot * k;
    const double *qbar = lib->q_bars + (uint64_t)slot * k;

    for (int alpha = 0; alpha < k; alpha++) {
        double dx = 0.0;
        for (int beta = 0; beta < k; beta++)
            dx += J[alpha * k + beta] * (q[beta] - qbar[beta]);
        x_end_out[alpha] = xbar[alpha] + dx;
    }
    *best_tok_out = lib->best_toks[slot];
    lib->hits++;
    return 1;
}

int axgeo_grc_lookup_with_summaries(axgeo_grc_library_t *lib,
                                    const double *q, int k,
                                    double *x_end_out, int *best_tok_out,
                                    double *block_summaries_out,
                                    int *n_summaries_out)
{
    if (n_summaries_out) *n_summaries_out = 0;
    if (!x_end_out || !best_tok_out || !lib || lib->k != k) return 0;

    int slot = -1;
    if (!grc_find_best(lib, q, &slot)) { lib->misses++; return 0; }

    int kk  = k * k;
    int nwp = AXGEO_GRC_N_WP;
    const double *J     = lib->Js     + (uint64_t)slot * kk;
    const double *xbar  = lib->x_ends + (uint64_t)slot * k;
    const double *qbar  = lib->q_bars + (uint64_t)slot * k;
    const double *wps   = lib->x_wps  + (uint64_t)slot * nwp * k;

    /* Compute base Jacobi correction δx = J·δq */
    double *dx = (double *)tensor_alloc((uint64_t)k * sizeof(double));
    if (!dx) { lib->misses++; return 0; }

    for (int alpha = 0; alpha < k; alpha++) {
        double d = 0.0;
        for (int beta = 0; beta < k; beta++)
            d += J[alpha * k + beta] * (q[beta] - qbar[beta]);
        dx[alpha] = d;
    }

    /* Terminal endpoint */
    for (int i = 0; i < k; i++)
        x_end_out[i] = xbar[i] + dx[i];
    *best_tok_out = lib->best_toks[slot];

    /* AttnRes block summaries (§6): b_n(q) ≈ x_wp_n(q̄) + J·δq·(n+1)/N_WP
     * The Jacobi correction at fraction t of the path is J(λ_f)·δq·t
     * (linear interpolation — see paper §6.2: "same propagator at shallower λ") */
    if (block_summaries_out) {
        for (int n = 0; n < nwp; n++) {
            double t = (double)(n + 1) / (double)nwp;
            const double *wp_n = wps + (uint64_t)n * k;
            double *out_n = block_summaries_out + (uint64_t)n * k;
            for (int i = 0; i < k; i++)
                out_n[i] = wp_n[i] + dx[i] * t;
        }
        if (n_summaries_out) *n_summaries_out = nwp;
    }

    tensor_free(dx);
    lib->hits++;
    return 1;
}

/* ── GRC Disk Persistence ─────────────────────────────────────────────── */
#define AXGEO_GRC_MAGIC   0x4752434C4942ULL  /* "GRCLIB" */
#define AXGEO_GRC_FVER    1

int axgeo_grc_save(const axgeo_grc_library_t *lib, const char *path)
{
    if (!lib || !path || lib->k <= 0 || lib->count <= 0) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int ok = 1;
    uint64_t magic   = AXGEO_GRC_MAGIC;
    int      ver     = AXGEO_GRC_FVER;
    int      k       = lib->k;
    int      count   = lib->count;
    int      cap     = lib->cap;
    int      write_idx = lib->write_idx;
    int      hits    = lib->hits;
    int      misses  = lib->misses;
    int      nbu     = lib->n_buckets_used;

#define GS_WRITE(p,n) do { if (fwrite((p),1,(n),f)!=(n)){ok=0;goto gs_done;} } while(0)
    GS_WRITE(&magic,    sizeof(magic));
    GS_WRITE(&ver,      sizeof(ver));
    GS_WRITE(&k,        sizeof(k));
    GS_WRITE(&count,    sizeof(count));
    GS_WRITE(&cap,      sizeof(cap));
    GS_WRITE(&write_idx,sizeof(write_idx));
    GS_WRITE(&hits,     sizeof(hits));
    GS_WRITE(&misses,   sizeof(misses));
    GS_WRITE(&nbu,      sizeof(nbu));
    GS_WRITE(lib->bucket_counts, sizeof(lib->bucket_counts));

    GS_WRITE(lib->q_bars,           (uint64_t)cap * k * sizeof(double));
    GS_WRITE(lib->Js,               (uint64_t)cap * k * k * sizeof(double));
    GS_WRITE(lib->x_ends,           (uint64_t)cap * k * sizeof(double));
    GS_WRITE(lib->rhos,             (uint64_t)cap * sizeof(double));
    GS_WRITE(lib->best_toks,        (uint64_t)cap * sizeof(int));
    GS_WRITE(lib->x_wps,            (uint64_t)cap * AXGEO_GRC_N_WP * k * sizeof(double));
    GS_WRITE(lib->bucket_centroids, (uint64_t)AXGEO_GRC_N_BUCKETS * k * sizeof(double));
    GS_WRITE(lib->bucket_assign,    (uint64_t)cap * sizeof(int));
#undef GS_WRITE

gs_done:
    fclose(f);
    return ok ? 0 : -1;
}

int axgeo_grc_load(axgeo_grc_library_t *lib, const char *path)
{
    if (!lib || !path) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    int ok = 1;
    uint64_t magic = 0;
    int ver = 0, k = 0, count = 0, cap = 0, write_idx = 0, hits = 0, misses = 0, nbu = 0;

#define GS_READ(p,n) do { if (fread((p),1,(n),f)!=(n)){ok=0;goto gl_done;} } while(0)
    GS_READ(&magic,    sizeof(magic));
    GS_READ(&ver,      sizeof(ver));
    GS_READ(&k,        sizeof(k));
    GS_READ(&count,    sizeof(count));
    GS_READ(&cap,      sizeof(cap));
    GS_READ(&write_idx,sizeof(write_idx));
    GS_READ(&hits,     sizeof(hits));
    GS_READ(&misses,   sizeof(misses));
    GS_READ(&nbu,      sizeof(nbu));

    if (magic != AXGEO_GRC_MAGIC || ver != AXGEO_GRC_FVER
        || k <= 0 || count < 0 || cap <= 0) {
        ok = 0; goto gl_done;
    }

    /* Destroy old library if it has different k or capacity */
    if (lib->k != k || lib->cap != cap) {
        axgeo_grc_library_destroy(lib);
        *lib = axgeo_grc_library_create(k, cap);
        if (!lib->q_bars) { ok = 0; goto gl_done; }
    }

    GS_READ(lib->bucket_counts, sizeof(lib->bucket_counts));
    GS_READ(lib->q_bars,           (uint64_t)cap * k * sizeof(double));
    GS_READ(lib->Js,               (uint64_t)cap * k * k * sizeof(double));
    GS_READ(lib->x_ends,           (uint64_t)cap * k * sizeof(double));
    GS_READ(lib->rhos,             (uint64_t)cap * sizeof(double));
    GS_READ(lib->best_toks,        (uint64_t)cap * sizeof(int));
    GS_READ(lib->x_wps,            (uint64_t)cap * AXGEO_GRC_N_WP * k * sizeof(double));
    GS_READ(lib->bucket_centroids, (uint64_t)AXGEO_GRC_N_BUCKETS * k * sizeof(double));
    GS_READ(lib->bucket_assign,    (uint64_t)cap * sizeof(int));

    lib->count         = count;
    lib->write_idx     = write_idx;
    lib->hits          = hits;
    lib->misses        = misses;
    lib->n_buckets_used = nbu;
#undef GS_READ

gl_done:
    fclose(f);
    return ok ? 0 : -1;
}

/* ── Per-QKV Head Pullback Metric ──────────────────────────────────────── */
/*
 * Compute a blended pullback metric from Q, K, V weight matrices.
 * For each weight matrix W [n_rows × n_cols]:
 *   w̃_i = U · W[i,:]^T  (project row into PCA subspace, d-vector)
 *   G   += w̃_i ⊗ w̃_i   (outer product accumulation)
 * The three contributions are summed and the caller can normalise.
 * This gives a more accurate geometric picture than using W alone because
 * Q and K have spherical geometry (via RoPE normalisation) while V is linear.
 */
void axgeo_pullback_metric_qkv(const void *W_Q, int nq_rows, int nq_cols,
                                ggml_type_t qtype,
                                const void *W_K, int nk_rows, int nk_cols,
                                ggml_type_t ktype,
                                const void *W_V, int nv_rows, int nv_cols,
                                ggml_type_t vtype,
                                const double *U, int d, int dim,
                                double *G_out)
{
    if (!G_out || !U || d <= 0 || dim <= 0) return;

    double *row_buf = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
    double *proj    = (double *)tensor_alloc((uint64_t)d   * sizeof(double));
    if (!row_buf || !proj) {
        if (row_buf) tensor_free(row_buf);
        if (proj)    tensor_free(proj);
        return;
    }

    /* Helper: accumulate outer product from one weight matrix */
#define BYTES_PER_ROW(ncols_, type_) \
    ((type_) == GGML_TYPE_Q4_0 ? ((size_t)(ncols_) / 32 * 18) : \
     (type_) == GGML_TYPE_Q8_0 ? ((size_t)(ncols_) / 32 * 34) : \
     (size_t)(ncols_) * sizeof(float))

#define ACCUM_PULLBACK(W_, nrows_, ncols_, type_) \
    do { \
        if ((W_) && (nrows_) > 0 && (ncols_) > 0) { \
            size_t row_bytes = BYTES_PER_ROW((ncols_), (type_)); \
            for (int ri = 0; ri < (nrows_); ri++) { \
                const void *rptr = (const char *)(W_) + (size_t)ri * row_bytes; \
                axgeo_dequant_row_f64(rptr, row_buf, dim, (int)(type_)); \
                /* proj = U · row_buf */ \
                for (int a = 0; a < d; a++) { \
                    double s = 0.0; \
                    for (int j = 0; j < dim; j++) \
                        s += U[(uint64_t)a * dim + j] * row_buf[j]; \
                    proj[a] = s; \
                } \
                /* G_out += proj ⊗ proj */ \
                for (int a = 0; a < d; a++) \
                    for (int b = 0; b < d; b++) \
                        G_out[(uint64_t)a * d + b] += proj[a] * proj[b]; \
            } \
        } \
    } while(0)

    ACCUM_PULLBACK(W_Q, nq_rows, nq_cols, qtype);
    ACCUM_PULLBACK(W_K, nk_rows, nk_cols, ktype);
    ACCUM_PULLBACK(W_V, nv_rows, nv_cols, vtype);

#undef ACCUM_PULLBACK
#undef BYTES_PER_ROW
    tensor_free(row_buf);
    tensor_free(proj);
}
