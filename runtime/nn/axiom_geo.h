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

/*  Metric tensor field  */

/*
 * A sampled Riemannian metric field: g_ij(x) defined at N sample points
 * in a d-dimensional subspace.
 *
 * points[k]  — the k-th sample point in subspace coordinates [d]
 * metrics[k] — the metric tensor at point k, as a symmetric dd matrix
 */
typedef struct {
    int     n_points;   /* number of sample points */
    int     dim;        /* subspace dimension */
    double *points;     /* [n_points  dim] sample locations */
    double *metrics;    /* [n_points  dim  dim] metric tensors (symmetric) */
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

/*  Christoffel symbols  */

/*
 * Christoffel symbols of the second kind: Γ^k_ij
 * Computed from the metric field via numerical differentiation:
 *   Γ^k_ij = ½ g^{kl} (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
 *
 * Stored as [n_points  dim  dim  dim] — Γ^k_ij for each sample point.
 */
typedef struct {
    int     n_points;
    int     dim;
    double *gamma;      /* [n_points  dim  dim  dim] */
} axgeo_christoffel_t;

/* Layer-stratified global Christoffel cache: one Γ^k_ij tensor per layer. */
typedef struct {
    int     n_layers;
    int     dim;
    double *gamma;      /* [n_layers  dim  dim  dim] */
} axgeo_layer_christoffel_t;

axgeo_christoffel_t axgeo_christoffel_create(int n_points, int dim);
void axgeo_christoffel_destroy(axgeo_christoffel_t *ch);
axgeo_layer_christoffel_t axgeo_layer_christoffel_create(int n_layers, int dim);
void axgeo_layer_christoffel_destroy(axgeo_layer_christoffel_t *ch);

/* Get Γ^k_ij at sample point p: gamma[p][k][i][j] */
static inline double *axgeo_gamma_at(axgeo_christoffel_t *ch, int p) {
    return ch->gamma + (uint64_t)p * ch->dim * ch->dim * ch->dim;
}

static inline double *axgeo_layer_gamma_at(axgeo_layer_christoffel_t *ch, int layer) {
    return ch->gamma + (uint64_t)layer * ch->dim * ch->dim * ch->dim;
}

/* Compute Christoffel symbols from metric field.
 * Uses centered finite differences for metric derivatives. */
int axgeo_compute_christoffel(const axgeo_metric_field_t *mf,
                              axgeo_christoffel_t *ch);

/* Compute a single global Christoffel tensor valid everywhere on the manifold.
 * Fits a global linear metric gradient via marginal OLS regression over all
 * sample points, then computes Γ^k_ij = ½ ḡ^{kl}(∂_i ḡ_{jl} + ∂_j ḡ_{il}
 * - ∂_l ḡ_{ij}) once.  ch must be created with n_points=1.
 * Cost: O(n_mp  d²) regression + O(d⁴) Γ tensor — one-time on the manifold.
 * The resulting ch can be passed directly to axgeo_christoffel_interpolate;
 * with n_points=1 it always returns the single global Γ. */
int axgeo_compute_christoffel_global(const axgeo_metric_field_t *mf,
                                     axgeo_christoffel_t *ch);

/* Interpolate Christoffel symbols at arbitrary point x */
void axgeo_christoffel_interpolate(const axgeo_christoffel_t *ch,
                                   const axgeo_metric_field_t *mf,
                                   const double *x, double *gamma_out);

/*  Curvature  */

typedef struct {
    int     n_points;
    int     dim;
    double *ricci;          /* [n_points  dim  dim] Ricci tensor R_ij */
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

/*  Geodesic Solver  */

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
    double *trajectory;  /* [max_steps  dim] if non-NULL */
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

/*
 * Adaptive RK45 (Cash-Karp) geodesic integrator — item 2.
 * Integrates until geo->lambda >= lambda_end or max_steps reached.
 * tol:     local error tolerance per step (e.g. 1e-6)
 * h_min:   minimum allowed step size
 * h_max:   maximum allowed step size (= 1/geo_steps is a good starting point)
 * Returns 0 on success, -1 on divergence.
 */
int axgeo_geodesic_integrate_adaptive(axgeo_geodesic_t *geo,
                                      const axgeo_christoffel_t *ch,
                                      const axgeo_metric_field_t *mf,
                                      double lambda_end,
                                      double tol,
                                      double h_min,
                                      double h_max);

/*
 * Geodesic trajectory cache — item 4.
 * Memoizes solved geodesic endpoints keyed by (start_cluster, end_cluster).
 * Cluster index is computed by nearest-neighbor among n_clusters centroids.
 * On a cache hit, the stored endpoint is returned directly (zero integration cost).
 */
#define AXGEO_TRAJ_CACHE_CAP  256
#define AXGEO_TRAJ_CLUSTER_K   16

typedef struct {
    int     start_cluster;
    int     end_cluster;
    double *endpoint;      /* [dim] geodesic endpoint in PCA subspace */
    double *velocity;      /* [dim] final velocity (for waypoint extraction) */
    int     dim;
    int     n_steps;       /* steps used when solved */
    int     valid;
} axgeo_traj_entry_t;

typedef struct {
    int                  n_clusters;   /* current centroid count */
    int                  dim;
    double              *centroids;    /* [n_clusters  dim] */
    axgeo_traj_entry_t   entries[AXGEO_TRAJ_CACHE_CAP];
    int                  count;
    int                  hits;
    int                  misses;
} axgeo_traj_cache_t;

void  axgeo_traj_cache_init(axgeo_traj_cache_t *tc, int dim);
void  axgeo_traj_cache_destroy(axgeo_traj_cache_t *tc);
void  axgeo_traj_cache_flush(axgeo_traj_cache_t *tc);

/* Add a centroid; returns its cluster index. */
int   axgeo_traj_cache_add_centroid(axgeo_traj_cache_t *tc, const double *pt);

/* Lookup: returns 1 + fills endpoint/velocity in out_geo on hit, 0 on miss. */
int   axgeo_traj_cache_lookup(axgeo_traj_cache_t *tc,
                               const double *start, const double *end,
                               double *out_endpoint, double *out_vel);

/* Insert solved geodesic into cache. */
void  axgeo_traj_cache_insert(axgeo_traj_cache_t *tc,
                               const double *start, const double *end,
                               const double *endpoint, const double *velocity,
                               int n_steps);

/* Compute geodesic distance (arc length) along the integrated path. */
double axgeo_geodesic_length(const axgeo_geodesic_t *geo,
                             const axgeo_metric_field_t *mf);

/*  Fisher Information Metric  */

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
    double *matrix;     /* [dim  dim] Fisher Information Matrix */
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

/*  OTT Knowledge Injection (local curvature warp) 
 * Apply a local Christoffel perturbation around point p:
 *   Gamma_tilde = Gamma + alpha * Phi(p) * exp(-d_g(x,p)^2 / (2*sigma^2))
 *
 * Returns number of sample points where a non-trivial warp was applied.
 */
int axgeo_apply_local_warp(axgeo_christoffel_t *ch,
                           const axgeo_metric_field_t *mf,
                           const double *p,
                           const double *phi,
                           double alpha,
                           double sigma);

/* Apply multiple local warps in superposition. */
int axgeo_apply_local_warp_many(axgeo_christoffel_t *ch,
                                const axgeo_metric_field_t *mf,
                                const double *points,
                                const double *phis,
                                int n_points,
                                double alpha,
                                double sigma);

/*  Weight-Derived Pullback Metric (Step 1 of OTT k⁴ plan) 
 *
 * Build a Riemannian metric field from the actual transformer weight matrices
 * rather than from token embedding covariance.
 *
 * For a layer with weight matrix W ∈ R^{outd}, the pullback metric on the
 * k-dimensional PCA subspace (spanned by U ∈ R^{dk}) is:
 *
 *   G = (WU)^T (WU) ∈ R^{kk}              (Fisher/Gram metric)
 *
 * This is the exact local geometry that the layer "sees" when operating on
 * an input in the PCA subspace. It is independent of sample distribution and
 * reflects the actual parameter structure of the model.
 *
 * For multiple layers, the metric varies with depth — we average across a
 * sliding window of layers to obtain a smooth per-point metric field.
 *  */

/*
 * Dequantize one row of a weight matrix into a float64 buffer.
 * type: GGML_TYPE_Q4_0 (10) or GGML_TYPE_Q8_0 (8).
 * Returns 0 on success.
 */
int axgeo_dequant_row_f64(const void *row_ptr, double *out, int dim, int type);

/*
 * Compute the pullback metric G = (W_sub)^T W_sub in R^{kk},
 * where W_sub = W · U (projection of weight rows onto PCA subspace).
 *
 * W_ptr:    pointer to weight matrix data (Q4_0 or F32)
 * w_type:   GGML_type of the weight
 * n_rows:   number of rows in W (output dimension)
 * n_cols:   number of columns in W (input dimension = d)
 * U:        PCA basis matrix U ∈ R^{dk}, column-major (d rows, k cols)
 * k:        subspace dimension
 * G_out:    output metric G ∈ R^{kk}, row-major
 * row_buf:  scratch buffer of size ≥ n_cols doubles
 * wu_buf:   scratch buffer of size ≥ n_rows  k doubles
 *
 * Returns 0 on success.
 */
int axgeo_pullback_metric(const void *W_ptr, int w_type,
                          int n_rows, int n_cols,
                          const double *U, int k,
                          double *G_out,
                          double *row_buf, double *wu_buf);

/*
 * Build a full metric field from weight matrices across layers.
 *
 * mf:       pre-allocated metric field (n_points sample points, dim=k)
 * U:        PCA basis U ∈ R^{dk} (row-major: U[i*k + j] = U_{ij})
 * k:        subspace dimension
 * d:        full model dimension
 * layer_weights: array of n_layers weight pointers (QKV concatenated or Q only)
 * layer_types:   array of n_layers GGML types
 * n_rows_each:   array of n_layers row counts
 * n_layers:  number of layers
 * sample_pts: optional pre-projected sample points [n_points  k];
 *             if NULL, sample points are set to equally-spaced grid
 *
 * Returns 0 on success.
 */
int axgeo_build_metric_from_weights(axgeo_metric_field_t *mf,
                                    const double *U, int k, int d,
                                    const void **layer_weights,
                                    const int *layer_types,
                                    const int *n_rows_each,
                                    int n_layers,
                                    const double *sample_pts);

/*
 * Apply RMSNorm sphere correction to Christoffel symbols at a point.
 *
 * The connection correction for RMSNorm(x) on the (d-1)-sphere:
 *   ΔΓ^k_ij = -(δ^k_j p_i + δ^k_i p_j)/||p||² + 2 p^k p_i p_j/||p||⁴
 *
 * This absorbs the RMSNorm nonlinearity into the Riemannian connection,
 * implementing the diffeomorphism ϕ described in the OTT paper §11.
 *
 * gamma: Christoffel tensor at one point, size kkk (index: [μ*k*k + ν*k + ρ])
 * p:     position vector in PCA subspace, size k
 * k:     subspace dimension
 * alpha: blend factor (0 = no correction, 1 = full correction)
 */
void axgeo_apply_rmsnorm_connection(double *gamma, const double *p,
                                    int k, double alpha);

/* 
 * Geodesic Resonance Caching (GRC) — OTT paper §4
 * 
 *
 * The Jacobi propagator J(λ_f) ∈ R^{kk} maps an initial perturbation δq
 * at the query point to the corresponding perturbation δx(λ_f) at the
 * geodesic endpoint:
 *
 *   δx(λ_f) ≈ J(λ_f) · δq                        O(k²) correction
 *
 * J is computed by integrating k independent Jacobi ODEs along a stored
 * geodesic trajectory:
 *
 *   D²J^α_i/dλ² + K^α_β(λ) J^β_i = 0
 *
 * where K^α_β = R^α_μβν (dx̄^μ/dλ)(dx̄^ν/dλ) is the geodesic tidal operator,
 * approximated from finite differences of the Christoffel symbols.
 */

/*
 * Compute the Jacobi propagator matrix J ∈ R^{kk} along a precomputed
 * geodesic trajectory.
 *
 * trajectory:  [n_wp  k] waypoints (row-major), from axgeo_geodesic_t
 * velocities:  [n_wp  k] tangent vectors at each waypoint (or NULL to
 *              estimate by finite difference from trajectory)
 * n_wp:        number of waypoints (≥ 2)
 * k:           subspace dimension
 * ch:          precomputed Christoffel symbols
 * mf:          metric field (for interpolation)
 * J_out:       output propagator [k  k], row-major (J_out[α*k + β] = J^α_β)
 *
 * Returns 0 on success, -1 on failure.
 */
int axgeo_compute_jacobi_propagator(const double *trajectory,
                                     const double *velocities,
                                     int n_wp, int k,
                                     const axgeo_christoffel_t *ch,
                                     const axgeo_metric_field_t *mf,
                                     double *J_out);

/*
 * GRC library — stores geodesic records for Jacobi-based fast correction.
 *
 * Each record encodes:
 *   q_bar[k]   — query embedding (PCA-projected)
 *   J[kk]     — Jacobi propagator at terminal waypoint
 *   x_end[k]   — geodesic endpoint (PCA-projected)
 *   rho        — injectivity radius estimate (curvature-based)
 *   best_tok   — nearest-vocab token at endpoint (-1 = unknown)
 *   x_wps[N_WP  k] — intermediate waypoints (block summaries, §6)
 */
#define AXGEO_GRC_CAP      4096  /* max stored records */
#define AXGEO_GRC_N_BUCKETS  16  /* ANN bucket count (O(cap/B) lookup) */
#define AXGEO_GRC_N_WP        4  /* intermediate waypoints (AttnRes blocks) */

/*
 * Estimate the injectivity radius at q_bar using the Christoffel norm:
 *
 *   K_est  = ||Γ(q_bar)||²_F / (k·(k-1) + 1)      ← bounding sectional K
 *   ρ(q̄) = π / sqrt(K_est)    clamped to [0.05, 5.0]
 *
 * This replaces the ad-hoc 0.5·||v_final|| heuristic with a proper
 * differential-geometry bound (inj_rad ≥ π/√K_max for sectional K_max).
 */
double axgeo_estimate_injectivity_radius(const axgeo_christoffel_t *ch,
                                          const axgeo_metric_field_t *mf,
                                          const double *q_bar, int k);

typedef struct {
    int     k;                      /* subspace dimension */
    int     count;                  /* records stored */
    int     cap;                    /* allocated capacity */
    double *q_bars;                 /* [cap  k] query centers */
    double *Js;                     /* [cap  k  k] Jacobi propagators */
    double *x_ends;                 /* [cap  k] geodesic endpoints */
    double *rhos;                   /* [cap] injectivity radii */
    int    *best_toks;              /* [cap] nearest vocab tokens at endpoint */
    double *x_wps;                  /* [cap  N_WP  k] intermediate waypoints */
    /* ANN bucket index */
    double *bucket_centroids;       /* [N_BUCKETS  k] cluster means */
    int    *bucket_assign;          /* [cap] bucket id per record */
    int     bucket_counts[AXGEO_GRC_N_BUCKETS]; /* records per bucket */
    int     n_buckets_used;
    int     write_idx;              /* next write slot (ring buffer) */
    int     hits;
    int     misses;
} axgeo_grc_library_t;

/* Initialise / destroy library. */
axgeo_grc_library_t axgeo_grc_library_create(int k, int cap);
void                axgeo_grc_library_destroy(axgeo_grc_library_t *lib);

/*
 * Insert a geodesic record.
 * q_bar[k], J[kk], x_end[k], rho, best_tok.
 * x_wps: optional [N_WP  k] intermediate waypoints for block summaries.
 *        Pass NULL to skip (waypoints stored as zeros).
 * If the library is full, the oldest entry is overwritten (ring buffer).
 */
void axgeo_grc_insert(axgeo_grc_library_t *lib,
                      const double *q_bar, const double *J,
                      const double *x_end, double rho, int best_tok,
                      const double *x_wps);

/*
 * Look up the nearest stored record to query q[k].
 * Uses two-stage ANN: find nearest buckets, scan only those records.
 * If dist(q, q_bar*) < rho:
 *   fills x_end_out[k] = x_end + J·δq  (Jacobi correction)
 *   *best_tok_out = stored best_tok
 *   returns 1 (hit)
 * Otherwise returns 0 (miss).
 */
int axgeo_grc_lookup(axgeo_grc_library_t *lib,
                     const double *q, int k,
                     double *x_end_out,
                     int *best_tok_out);

/*
 * Lookup with AttnRes block summaries (OTT paper §6).
 * On hit: fills x_end_out AND block_summaries_out[N_WP  k] with
 * Jacobi-corrected intermediate waypoints:
 *   b_n(q) ≈ x_wp_n(q̄) + J·δq · (n+1)/N_WP
 * n_summaries_out = AXGEO_GRC_N_WP on hit, 0 on miss.
 */
int axgeo_grc_lookup_with_summaries(axgeo_grc_library_t *lib,
                                    const double *q, int k,
                                    double *x_end_out, int *best_tok_out,
                                    double *block_summaries_out,
                                    int *n_summaries_out);

/*
 * Persist the GRC library to / from a binary file.
 * Format: magic + version + k + count + all arrays.
 * Returns 0 on success, -1 on error.
 */
int axgeo_grc_save(const axgeo_grc_library_t *lib, const char *path);
int axgeo_grc_load(axgeo_grc_library_t *lib, const char *path);

/*  Per-QKV head pullback metric  */
/*
 * Compute separate pullback metrics for Q, K, and V weight matrices of a
 * single attention layer, then blend them (equal weights) into a single
 * dd metric.  More accurate than using only one weight matrix because
 * Q, K, V have different geometry (Q/K live on sphere via RoPE, V is linear).
 *
 * W_Q [dim  n_heads*head_dim],  type: quantised
 * W_K [dim  n_kv_heads*head_dim], type: quantised
 * W_V [dim  n_kv_heads*head_dim], type: quantised
 * U   [d  dim]   PCA projection (d rows, dim cols)
 * out G_out[dd]  accumulated (summed) — caller should divide by n_layers
 */
void axgeo_pullback_metric_qkv(const void *W_Q, int nq_rows, int nq_cols,
                                ggml_type_t qtype,
                                const void *W_K, int nk_rows, int nk_cols,
                                ggml_type_t ktype,
                                const void *W_V, int nv_rows, int nv_cols,
                                ggml_type_t vtype,
                                const double *U, int d, int dim,
                                double *G_out);

#endif /* GEODESSICAL_AXIOM_GEO_H */
