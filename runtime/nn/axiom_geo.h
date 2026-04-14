/*
 * Geodessical — Axiomatic Differential Geometry Engine
 *
 * Riemannian geometry on neural manifolds: metric tensors, Christoffel
 * symbols, curvature tensors, and geodesic integration.
 *
 * All computations operate in a projected subspace of dimension ≤ 512
 * (the intrinsic dimension found by PCA/TwoNN), making O(d⁴) curvature
 * computation tractable.
 *
 * Key types:
 *   axgeo_metric_field_t  — sampled metric tensor field g_ij(x) at N points
 *   axgeo_christoffel_t   — Christoffel symbols Γ^k_ij at N points
 *   axgeo_curvature_t     — Riemann/Ricci curvature at N points
 *   axgeo_geodesic_t      — Geodesic path state for ODE integration
 */

#ifndef GEODESSICAL_AXIOM_GEO_H
#define GEODESSICAL_AXIOM_GEO_H

#include "runtime/nn/axiom_linalg.h"
#include <stdint.h>

/* Maximum subspace dimension for geometric computation.
 * Christoffel symbols are O(d³), curvature is O(d⁴). Keep d ≤ 256. */
#define AXGEO_MAX_DIM 256

/* ─── Metric tensor field ─── */

/*
 * A sampled Riemannian metric field: g_ij(x) defined at N sample points
 * in a d-dimensional subspace.
 *
 * points[k]  — the k-th sample point in subspace coordinates [d]
 * metrics[k] — the metric tensor at point k, as a symmetric d×d matrix
 */
typedef struct {
    int     n_points;   /* number of sample points */
    int     dim;        /* subspace dimension */
    double *points;     /* [n_points × dim] sample locations */
    double *metrics;    /* [n_points × dim × dim] metric tensors (symmetric) */
} axgeo_metric_field_t;

axgeo_metric_field_t axgeo_metric_field_create(int n_points, int dim);
void axgeo_metric_field_destroy(axgeo_metric_field_t *mf);

/* Get a pointer to the metric tensor at sample point k */
static inline double *axgeo_metric_at(axgeo_metric_field_t *mf, int k) {
    return mf->metrics + (uint64_t)k * mf->dim * mf->dim;
}

/* Get a pointer to the sample point k */
static inline double *axgeo_point_at(axgeo_metric_field_t *mf, int k) {
    return mf->points + (uint64_t)k * mf->dim;
}

/* Interpolate metric tensor at an arbitrary point x using inverse-distance
 * weighting from the nearest sample points. */
void axgeo_metric_interpolate(const axgeo_metric_field_t *mf,
                              const double *x, double *g_out);

/* ─── Christoffel symbols ─── */

/*
 * Christoffel symbols of the second kind: Γ^k_ij
 * Computed from the metric field via numerical differentiation:
 *   Γ^k_ij = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
 *
 * Stored as [n_points × dim × dim × dim] — Γ^k_ij for each sample point.
 */
typedef struct {
    int     n_points;
    int     dim;
    double *gamma;      /* [n_points × dim × dim × dim] */
} axgeo_christoffel_t;

axgeo_christoffel_t axgeo_christoffel_create(int n_points, int dim);
void axgeo_christoffel_destroy(axgeo_christoffel_t *ch);

/* Get Γ^k_ij at sample point p: gamma[p][k][i][j] */
static inline double *axgeo_gamma_at(axgeo_christoffel_t *ch, int p) {
    return ch->gamma + (uint64_t)p * ch->dim * ch->dim * ch->dim;
}

/* Compute Christoffel symbols from metric field.
 * Uses centered finite differences for metric derivatives. */
int axgeo_compute_christoffel(const axgeo_metric_field_t *mf,
                              axgeo_christoffel_t *ch);

/* Interpolate Christoffel symbols at arbitrary point x */
void axgeo_christoffel_interpolate(const axgeo_christoffel_t *ch,
                                   const axgeo_metric_field_t *mf,
                                   const double *x, double *gamma_out);

/* ─── Curvature ─── */

typedef struct {
    int     n_points;
    int     dim;
    double *ricci;          /* [n_points × dim × dim] Ricci tensor R_ij */
    double *scalar_curv;    /* [n_points] scalar curvature R */
    double  mean_curvature; /* mean scalar curvature across points */
    double  max_curvature;  /* max |R| across points */
} axgeo_curvature_t;

axgeo_curvature_t axgeo_curvature_create(int n_points, int dim);
void axgeo_curvature_destroy(axgeo_curvature_t *curv);

/*
 * Compute Riemann → Ricci → scalar curvature from Christoffel symbols.
 * Riemann: R^l_{ijk} = ∂_j Γ^l_{ik} - ∂_k Γ^l_{ij}
 *                     + Γ^l_{jm} Γ^m_{ik} - Γ^l_{km} Γ^m_{ij}
 * Ricci:   R_{ij} = R^k_{ikj}
 * Scalar:  R = g^{ij} R_{ij}
 */
int axgeo_compute_curvature(const axgeo_christoffel_t *ch,
                            const axgeo_metric_field_t *mf,
                            axgeo_curvature_t *curv);

/* ─── Geodesic Solver ─── */

/*
 * Geodesic equation:  d²x^μ/dλ² + Γ^μ_νρ dx^ν/dλ dx^ρ/dλ = 0
 *
 * Rewritten as first-order system:
 *   dx^μ/dλ = v^μ
 *   dv^μ/dλ = -Γ^μ_νρ v^ν v^ρ
 *
 * Integrated with RK4.
 */
typedef struct {
    int     dim;
    double *x;      /* [dim] current position */
    double *v;      /* [dim] current velocity (tangent vector) */
    double  lambda;  /* path parameter */
    int     steps;   /* steps taken */
    int     max_steps;
    double  step_size;
    /* Trajectory recording */
    double *trajectory;  /* [max_steps × dim] if non-NULL */
    int     record;      /* 1 = record trajectory */
} axgeo_geodesic_t;

/* Initialize geodesic state with starting position and velocity. */
axgeo_geodesic_t axgeo_geodesic_init(int dim, const double *x0,
                                      const double *v0,
                                      double step_size, int max_steps,
                                      int record_trajectory);
void axgeo_geodesic_destroy(axgeo_geodesic_t *geo);

/*
 * Integrate geodesic for n_steps using RK4.
 * Christoffel symbols are interpolated from the precomputed field.
 * Returns 0 on success, -1 if geodesic diverges.
 */
int axgeo_geodesic_integrate(axgeo_geodesic_t *geo,
                             const axgeo_christoffel_t *ch,
                             const axgeo_metric_field_t *mf,
                             int n_steps);

/* Compute geodesic distance (arc length) along the integrated path. */
double axgeo_geodesic_length(const axgeo_geodesic_t *geo,
                             const axgeo_metric_field_t *mf);

/* ─── Fisher Information Metric ─── */

/*
 * Estimate the Fisher Information Matrix at a point in embedding space
 * by numerical perturbation of the model's output distribution.
 *
 * F_ij ≈ (1/N) Σ_k [∂log p_k/∂x_i · ∂log p_k/∂x_j]
 *
 * Uses finite differences with step size ε in each subspace direction.
 */
typedef struct {
    int    dim;
    double *matrix;     /* [dim × dim] Fisher Information Matrix */
    double trace;       /* Tr(F) */
    double det_log;     /* log det(F) */
} axgeo_fisher_t;

axgeo_fisher_t axgeo_fisher_create(int dim);
void axgeo_fisher_destroy(axgeo_fisher_t *f);

/*
 * Compute Fisher Information Matrix at a metric field sample point.
 * Approximates F ≈ g⁻¹ (inverse covariance metric) — exact for Gaussian
 * distributions and a principled information-theoretic metric otherwise.
 *
 * point_idx: index into the metric field's sample points.
 * Sets f->matrix, f->trace, f->det_log.
 * Returns 0 on success, -1 on failure.
 */
int axgeo_compute_fisher(axgeo_fisher_t *f,
                         const axgeo_metric_field_t *mf,
                         int point_idx);

/*
 * Blend Fisher metric into a metric field point:
 *   g_blended = (1 - alpha) * g_covariance + alpha * F
 * alpha ∈ [0,1] controls the blend.  Overwrites the metric at point_idx.
 */
void axgeo_metric_blend_fisher(axgeo_metric_field_t *mf,
                               const axgeo_fisher_t *f,
                               int point_idx, double alpha);

#endif /* GEODESSICAL_AXIOM_GEO_H */
