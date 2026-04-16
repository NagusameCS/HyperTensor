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

void axgeo_christoffel_destroy(axgeo_christoffel_t *ch)
{
    if (!ch) return;
    if (ch->gamma) { tensor_free(ch->gamma); ch->gamma = NULL; }
    ch->n_points = ch->dim = 0;
}

/* Invert a small symmetric matrix (d×d) using Gauss-Jordan elimination */
static int invert_symmetric(const double *A, double *Ainv, int d)
{
    /* Augmented matrix [A | I] */
    int sz = d * 2 * d;
    double *aug = (double *)tensor_alloc((uint64_t)sz * sizeof(double));
    if (!aug) return -1;

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++)
            aug[i * 2 * d + j] = A[i * d + j];
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
                    gamma_p[k * dd + i * d + j] = 0.5 * sum;
                    gamma_p[k * dd + j * d + i] = 0.5 * sum; /* symmetric in i,j */
                }
            }
        }
    }

    tensor_free(g_inv);
    tensor_free(dg);
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
