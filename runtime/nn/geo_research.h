/*
 * geo_research.h - Four research compression extensions
 *
 * Feature 1: Real sectional / Ricci curvature compression signal
 *   Uses the existing axgeo machinery to compute a true sectional curvature
 *   from hidden-state samples and uses it as the per-layer rank budget signal.
 *   Currently axiom_exploit.c uses Frobenius^2 x spectral_gap x depth as a
 *   proxy.  This computes the actual differential-geometric quantity by
 *   collecting hidden-state samples per layer and running axgeo_compute_curvature.
 *
 * Feature 2: Cross-model basis transfer (small -> large sibling)
 *   Transfers a PCA basis calibrated on a small model (e.g. Gemma 2B) to
 *   accelerate compression of a larger model in the same family (7B, 27B).
 *   Caller passes arrays of axpca_t for each model.  The function aligns
 *   dimensions, maps basis vectors, and validates with held-out samples.
 *
 * Feature 3: Phase-conditional compression plans
 *   Detects the current generation phase from the KV/attention entropy pattern
 *   (retrieval, reasoning, generation) and selects a different compression
 *   plan per phase.  Each plan trades quality vs cost differently.
 *
 * Feature 4: Differentiable compression plan selection
 *   Parameterises the compression plan as a softmax-over-rank-levels per layer
 *   and optimises it end-to-end against a perplexity proxy.  Uses a straight-
 *   through estimator so the rank selection step is differentiable.
 */

#ifndef GEODESSICAL_GEO_RESEARCH_H
#define GEODESSICAL_GEO_RESEARCH_H

#include "runtime/nn/axiom_geo.h"
#include "runtime/nn/axiom_linalg.h"
#include "runtime/nn/axiom_beta.h"
#include "runtime/nn/axiom_exploit.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Feature 1: True sectional curvature compression signal
 * =========================================================================
 *
 * The current Ricci proxy in axiom_exploit.c is:
 *   proxy = ||W||_F^2 * spectral_gap * (layer_depth / n_layers)
 * which is just activation covariance importance - the same signal used by
 * ESPACE, ASVD, FLAT-LLM.  It has nothing to do with differential geometry.
 *
 * The true signal: for a pair of tangent vectors u, v at a point p on the
 * manifold, the sectional curvature is:
 *
 *   K(u,v) = R(u,v,u,v) / (g(u,u)*g(v,v) - g(u,v)^2)
 *
 * where R is the Riemann curvature tensor.  The mean sectional curvature
 * (averaged over all 2-planes at a sample point) gives a per-layer scalar
 * that is genuinely differential-geometric.
 *
 * This is computable from the axgeo_curvature_t struct which is already
 * populated by axiom_beta Phase 3.  The mean sectional curvature per layer
 * is a more honest rank signal than the Frobenius proxy.
 *
 * Implementation note: for dimension d, there are d*(d-1)/2 two-planes.
 * We evaluate K on a random sample of planes (default 32) to keep cost O(d).
 */

#define GRCURV_MAX_LAYERS 256
#define GRCURV_PLANES_PER_LAYER 32

typedef struct {
    int     n_layers;
    double  sectional[GRCURV_MAX_LAYERS];   /* mean sectional curvature per layer */
    double  ricci_diag[GRCURV_MAX_LAYERS];  /* Ric(e_1, e_1) along first principal dir */
    double  scalar[GRCURV_MAX_LAYERS];      /* scalar curvature R = g^{ij} R_{ij} */
    double  proxy[GRCURV_MAX_LAYERS];       /* old Frobenius proxy for comparison */
    double  correlation;                     /* Pearson corr(sectional, proxy) */
} grcurv_per_layer_t;

/*
 * Per-layer curvature input: caller provides one of these per transformer layer.
 *
 * samples:    row-major F32 hidden states [n_samples x dim] collected from that
 *             layer during a calibration forward pass.  Compute with
 *             axiom_beta_probe_all_layer_states().  May be NULL if n_samples == 0.
 * dim:        hidden state dimension (full, not projected).
 * n_samples:  number of sample rows in `samples`.  Minimum 4; recommended 16-64.
 *
 * proxy_ricci: the Frobenius-proxy importance score already computed by
 *              axiom_exploit (pass axex_offload_plan_t.entries[l].ricci_scalar).
 *              Used for the correlation comparison only; may be 0.0 if unknown.
 */
typedef struct {
    const float *samples;   /* [n_samples x dim] F32 row-major, may be NULL */
    int          n_samples;
    int          dim;
    double       proxy_ricci; /* Frobenius proxy value from axiom_exploit */
} grcurv_layer_input_t;

/*
 * Compute true sectional curvature per layer.
 *
 * inputs:      array of per-layer inputs [n_layers]
 * n_layers:    number of layers
 * out:         filled with real and proxy curvature per layer
 *
 * For each layer:
 *   - Builds a Riemannian metric field from the sample covariances
 *   - Computes Christoffel symbols and Riemann/Ricci/scalar curvature
 *   - Records mean_sectional = scalar_R / (d*(d-1))
 *
 * Layers with n_samples < 4 fall back to 0.0 curvature.
 *
 * Returns 0 on success, -1 on bad input, -2 on allocation failure.
 */
int grcurv_compute(const grcurv_layer_input_t *inputs,
                   int n_layers,
                   grcurv_per_layer_t *out);

/*
 * Translate per-layer curvature to per-layer SVD rank budgets.
 *
 * Higher curvature -> higher rank (more information density, keep more SVs).
 * The total rank budget across all layers is constrained to total_rank_budget.
 *
 * curv:               per-layer curvature from grcurv_compute()
 * min_rank, max_rank: hard bounds per layer
 * total_rank_budget:  sum of all layer ranks must not exceed this
 * ranks_out[]:        output array [n_layers]
 */
void grcurv_to_rank_budget(const grcurv_per_layer_t *curv,
                           int min_rank, int max_rank,
                           int total_rank_budget,
                           int *ranks_out);

/*
 * Print a comparison table of real vs proxy curvature per layer.
 * Useful for the research question: do they correlate?
 */
void grcurv_print_comparison(const grcurv_per_layer_t *curv);


/* =========================================================================
 * Feature 2: Cross-model basis transfer
 * =========================================================================
 *
 * A PCA basis calibrated on a small sibling model can bootstrap the basis
 * for a large model in the same family, saving the ~17-minute calibration
 * cost.  This works because same-family models share:
 *   - Vocabulary and embedding space
 *   - Residual stream semantics (trained on same data distribution)
 *   - Layer structure (dim ratios differ but principal directions align)
 *
 * The transfer procedure:
 *   1. Align dimensions: the small model may have dim_s != dim_l.
 *      We use the shared embedding matrix to find a linear map A: R^{d_s} -> R^{d_l}
 *      via the k nearest vocabulary rows (in embedding space).
 *   2. Map basis vectors: P_l = A @ P_s @ A^+  (pseudoinverse)
 *   3. Validate: run N validation passes on the large model with the transferred
 *      basis.  Compute explained variance.  If > threshold, accept; else fall
 *      back to full calibration.
 *
 * The transfer saves calibration time proportional to (1 - validation_fraction).
 * With 32 validation passes instead of 512, the saving is ~94%.
 */

#define XFER_MAX_LAYERS 256

typedef struct {
    int    n_layers;
    /* Transferred bases: basis_out[l] = [k_l x dim_l] F32 row-major */
    float *bases[XFER_MAX_LAYERS];   /* allocated here, caller must free */
    int    ks[XFER_MAX_LAYERS];      /* rank per layer after transfer */
    int    dims[XFER_MAX_LAYERS];    /* target dim per layer */

    /* Validation results */
    double variance_explained[XFER_MAX_LAYERS]; /* fraction explained by transferred basis */
    int    accepted[XFER_MAX_LAYERS];            /* 1 = transfer accepted, 0 = needs recal */
    int    n_accepted;                           /* count of accepted layers */
    int    n_rejected;                           /* count of layers needing recalibration */

    double mean_variance_explained;
    double acceptance_threshold;    /* variance threshold used (e.g. 0.90) */
} xfer_result_t;

/*
 * Transfer PCA bases from a small model to a large model in the same family.
 *
 * small_pcas:   per-layer PCA structs from the small model [small_n_layers].
 *               These are the axpca_t values filled by the layerwise calibration
 *               in axiom_exploit.  Pass NULL elements for uncalibrated layers.
 * small_n_layers: count of entries in small_pcas.
 * large_n_layers: target layer count for the large model.
 * large_layer_dims: hidden state dimension per layer in the large model [large_n_layers].
 * large_layer_samples: optional validation samples per layer.
 *               Pass as float *[large_n_layers] where each pointer is
 *               [n_validation_samples x dim] F32 row-major.  May be NULL or
 *               individual entries may be NULL (those layers get ev=0, rejected).
 * n_validation_samples: rows in each large_layer_samples[l] array (e.g. 32).
 * acceptance_variance_threshold: minimum explained variance to accept transfer.
 *               Good default: 0.90.  Use 0.95 for conservative transfer.
 *
 * Fills result with transferred bases and validation results.
 * Returns 0 on success, negative on error.
 * Caller must free result->bases[l] for each layer (or call xfer_result_destroy).
 */
int xfer_basis_small_to_large(const axpca_t *small_pcas,
                               int small_n_layers,
                               int large_n_layers,
                               const int *large_layer_dims,
                               float * const *large_layer_samples,
                               int n_validation_samples,
                               double acceptance_variance_threshold,
                               xfer_result_t *result);

/*
 * Free all bases allocated inside a xfer_result_t.
 */
void xfer_result_destroy(xfer_result_t *result);

/*
 * Print a transfer summary.
 */
void xfer_result_print(const xfer_result_t *result);


/* =========================================================================
 * Feature 3: Phase-conditional compression plans
 * =========================================================================
 *
 * A single generation often has three distinct phases:
 *
 *   RETRIEVAL:  Early tokens. Model is scanning context, attention is broad,
 *               entropy is high, most layers are active.  Prefer higher rank
 *               to preserve recall quality.
 *
 *   REASONING:  Middle tokens (common in thinking models).  Attention focuses
 *               on a small working set, hidden state entropy drops.  Moderate
 *               rank is fine; speed matters more than recall.
 *
 *   GENERATION: Final tokens.  Attention is very focused, entropy is low,
 *               output is predictable.  Aggressive compression is safe here.
 *
 * Phase is detected from the KV attention entropy pattern:
 *   - Mean attention entropy over all heads and layers (from flash_attn stats)
 *   - Rate of change of entropy (smoothed over a short window)
 *   - KV cache hit rate (from axex_kv_ctx_t compression ratio)
 *
 * Each phase has a different compression plan: a set of per-layer ranks.
 * The plans are pre-computed at startup from the axiom_beta result and stored
 * in a phased_plan_t.  Switching plans at decode time is cheap: it only
 * changes which W_proj pointers are used, not the weights themselves.
 */

typedef enum {
    PHASE_UNKNOWN    = 0,
    PHASE_RETRIEVAL  = 1,
    PHASE_REASONING  = 2,
    PHASE_GENERATION = 3,
} gen_phase_t;

#define PHASEPLAN_MAX_LAYERS 256

/* Per-phase compression ranks */
typedef struct {
    int ranks[PHASEPLAN_MAX_LAYERS];    /* SVD rank per layer for this phase */
    int n_layers;
    float quality_floor;                /* target quality for this phase */
    const char *name;
} phase_plan_t;

typedef struct {
    phase_plan_t retrieval;
    phase_plan_t reasoning;
    phase_plan_t generation;

    /* Phase detector state */
    gen_phase_t current_phase;
    float attn_entropy_ema;         /* exponential moving average of attention entropy */
    float entropy_change_rate;      /* smoothed dH/dt */
    int   tokens_in_phase;          /* tokens generated in current phase */
    int   min_tokens_per_phase;     /* minimum before a phase switch (hysteresis) */

    /* Thresholds for phase transitions */
    float retrieval_entropy_min;    /* entropy above this = retrieval phase */
    float generation_entropy_max;   /* entropy below this = generation phase */
} phased_plan_t;

/*
 * Build three compression plans.
 *
 * offload_plan:       curvature-guided offload plan from axex_layer_offload_plan().
 *                     Used to read per-layer importance (ricci_scalar) and
 *                     derive per-layer base ranks.
 * n_layers:           number of transformer layers.
 * retrieval_quality:  quality floor for retrieval phase (e.g. 0.95)
 * reasoning_quality:  quality floor for reasoning phase (e.g. 0.90)
 * generation_quality: quality floor for generation phase (e.g. 0.80)
 *
 * Returns 0 on success.
 */
int phaseplan_build(phased_plan_t *pp,
                    const axex_offload_plan_t *offload_plan,
                    int n_layers,
                    float retrieval_quality,
                    float reasoning_quality,
                    float generation_quality);

/*
 * Update phase detector with one decode step's attention entropy.
 *
 * attn_entropy: mean attention entropy this step (from flash_attn_stats_t or
 *               computed from the raw softmax output over the KV positions).
 * kv_merge_rate: fraction of KV positions merged this step (from axex_kv_ctx).
 *
 * Returns the current phase (may have changed).
 */
gen_phase_t phaseplan_update(phased_plan_t *pp,
                             float attn_entropy,
                             float kv_merge_rate);

/*
 * Get the active plan for the current phase.
 */
const phase_plan_t *phaseplan_active(const phased_plan_t *pp);

/*
 * Get rank for a specific layer in the current phase.
 */
static inline int phaseplan_rank(const phased_plan_t *pp, int layer) {
    const phase_plan_t *p = phaseplan_active(pp);
    if (!p || layer < 0 || layer >= p->n_layers) return 64;
    return p->ranks[layer];
}

void phaseplan_print(const phased_plan_t *pp);


/* =========================================================================
 * Feature 4: Differentiable compression plan selection
 * =========================================================================
 *
 * The compression plan (which layers to compress, at what rank) is currently
 * chosen by a heuristic: curvature rank budget.  This feature makes the plan
 * selection differentiable by parameterising it as a soft rank allocation
 * and optimising it against a perplexity proxy.
 *
 * Parameterisation
 * ----------------
 * For each layer l and each rank level r in {r_0, r_1, ..., r_R}:
 *   theta[l][r] = unnormalized logit
 *   p[l][r]     = softmax(theta[l])[r]
 *   rank_soft[l] = sum_r( p[l][r] * r )   <- expected rank
 *
 * The straight-through estimator makes the hard rank selection differentiable:
 *   rank_hard[l] = argmax_r(p[l]) at forward time
 *   gradient flows through rank_soft[l] at backward time
 *
 * Objective
 * ---------
 * L = perplexity_proxy(compressed_model, calibration_tokens)
 *   + lambda * sum_l(rank_soft[l])   <- L1 regularisation on total rank
 *
 * perplexity_proxy is approximated by:
 *   proxy = sum_l( ||W_l - W_l_compressed||_F / ||W_l||_F )
 * which is cheap (no forward pass needed after the SVD).
 *
 * The gradient of the proxy w.r.t. theta[l][r] is computed analytically:
 *   d(proxy)/d(theta[l][r]) = d(frob_err_l)/d(rank_l) * d(rank_l)/d(theta[l][r])
 *
 * d(frob_err_l)/d(rank_l) is approximated by the slope of the singular value
 * decay curve at the current rank.
 *
 * Optimiser: vanilla SGD with momentum, ~100 iterations, negligible cost.
 *
 * Prior art gap: WeLore, CorDA, EoRA learn compression magnitudes given a fixed
 * plan structure.  This learns the structure (which layers, what rank) directly.
 */

#define DIFFPLAN_MAX_LAYERS  256
#define DIFFPLAN_N_LEVELS      8  /* rank levels: 8, 16, 32, 64, 128, 192, 256, 512 */

static const int DIFFPLAN_RANK_LEVELS[DIFFPLAN_N_LEVELS] = {8, 16, 32, 64, 128, 192, 256, 512};

typedef struct {
    int    n_layers;
    int    n_levels;

    float  theta[DIFFPLAN_MAX_LAYERS][DIFFPLAN_N_LEVELS];  /* logits */
    float  p[DIFFPLAN_MAX_LAYERS][DIFFPLAN_N_LEVELS];      /* softmax probs */
    float  rank_soft[DIFFPLAN_MAX_LAYERS];                  /* expected rank */
    int    rank_hard[DIFFPLAN_MAX_LAYERS];                  /* argmax rank */

    /* Singular value decay slopes per layer (needed for gradient) */
    float  sv_slope[DIFFPLAN_MAX_LAYERS];     /* d(frob_err)/d(rank) at current rank */
    float  frob_err[DIFFPLAN_MAX_LAYERS];     /* current reconstruction error per layer */

    /* Training state */
    float  momentum[DIFFPLAN_MAX_LAYERS][DIFFPLAN_N_LEVELS];
    float  lr;          /* learning rate (default 0.01) */
    float  momentum_coef; /* momentum coeff (default 0.9) */
    float  lambda;      /* L1 rank regularisation strength */
    int    n_iter;      /* iterations run so far */
    float  loss_history[256]; /* loss at each iteration, up to 256 */

    /* Config */
    float  quality_floor;     /* minimum acceptable frob quality per layer */
    int    max_iter;          /* iteration limit */
    float  convergence_tol;   /* stop if |dL| < tol */
} diffplan_t;

/*
 * Per-layer SVD data needed to initialise the differentiable plan.
 * The caller fills this from axex_compressed_weight_t after an initial
 * full-rank SVD pass.
 *
 * frob_err:   ||W - U S Vt||_F / ||W||_F at current rank (from axex_compressed_weight_t.frobenius_err)
 * sv_slope:   d(frob_err)/d(rank) at the current rank.
 *             Approximated as (sv[rank] / sv[0]) — the relative magnitude of
 *             the first truncated singular value.  Negative (higher rank -> lower err).
 */
typedef struct {
    float frob_err;
    float sv_slope;
    int   current_rank;
} diffplan_layer_data_t;

/*
 * Initialise from per-layer SVD data.
 *
 * layer_data:    array [n_layers] with SVD stats per layer.
 *                Pass NULL to use synthetic defaults (frob_err=0.05, sv_slope=-0.001).
 * n_layers:      number of layers.
 * lambda:        rank regularisation strength (0.001 is a good start).
 * quality_floor: per-layer minimum quality (0.90 is reasonable).
 *
 * Returns 0 on success.
 */
int diffplan_init(diffplan_t *dp,
                  const diffplan_layer_data_t *layer_data,
                  int n_layers,
                  float lambda,
                  float quality_floor);

/*
 * Run one optimisation step.
 * Updates theta, p, rank_soft, rank_hard.
 * Returns current loss.
 */
float diffplan_step(diffplan_t *dp);

/*
 * Run n_iter optimisation steps to convergence.
 * Returns final loss.
 */
float diffplan_optimise(diffplan_t *dp, int n_iter);

/*
 * Extract the optimised hard rank plan into an array.
 * ranks_out[l] = rank_hard[l] for each layer.
 */
void diffplan_get_ranks(const diffplan_t *dp, int *ranks_out);

/*
 * Print a summary of the plan and loss curve.
 */
void diffplan_print(const diffplan_t *dp);

#ifdef __cplusplus
}
#endif

#endif /* GEODESSICAL_GEO_RESEARCH_H */
