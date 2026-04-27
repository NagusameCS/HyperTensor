/*
 * mcr_compress.h - Mix-Compress-Refine phase detection & attention-sink bypass
 *
 * Feature 1: MCR-aware per-phase rank allocation
 *   Queipo-de-Llano et al. (Oct 2025, LeCun/Bronstein co-authors) proved that
 *   transformers process tokens in three measurable phases driven by massive
 *   activations on the BOS token creating attention sinks:
 *     Mix phase:      early layers — broad token mixing, high activation variance.
 *                     Compression here is destructive because this is where the
 *                     model builds its cross-token representation.
 *     Compress phase: middle layers — the model ALREADY self-compresses, attentional
 *                     mass collapses onto a small set of directions, activation
 *                     variance drops to a global minimum.  Low-rank projection
 *                     here is nearly free — we're just replicating what the model
 *                     already does.
 *     Refine phase:   late layers — selective, task-specific feature extraction.
 *                     Rank matters again for different reasons.
 *
 *   Current axex pipeline treats all layers uniformly (one compression rank per
 *   layer, or the Ricci-budgeted variant).  MCR-aware allocation detects phase
 *   boundaries empirically from the hidden-state activation variance profile
 *   and assigns non-uniform rank budgets: high rank to Mix/Refine, low rank
 *   to Compress.  This gives quality-at-fixed-memory gains without changing
 *   the compression pipeline itself — just the rank numbers.
 *
 * Feature 2: Attention-sink bypass
 *   BOS and other "sink" tokens have residual-stream L2 norms several standard
 *   deviations above the mean.  Their large activations may land outside the
 *   PCA subspace, degrading reconstruction quality non-linearly past some
 *   compression ratio.  StreamingLLM preserves sink tokens for KV cache but
 *   nobody does it for weight/activation compression.
 *
 *   We detect the dominant high-norm direction in the calibration samples and
 *   check whether the existing PCA basis covers it.  If not, the caller can
 *   inject it as an extra basis vector before axex_manifold_init_layer().
 *   This preserves sink-direction fidelity with minimal rank cost.
 */

#ifndef GEODESSICAL_MCR_COMPRESS_H
#define GEODESSICAL_MCR_COMPRESS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Feature 1: MCR phase detection and rank allocation
 * ========================================================================= */

#define MCR_MAX_LAYERS 512

typedef enum {
    MCR_PHASE_UNKNOWN  = 0,
    MCR_PHASE_MIX      = 1,   /* early: broad mixing, high act variance         */
    MCR_PHASE_COMPRESS = 2,   /* middle: model self-compressed, low act variance */
    MCR_PHASE_REFINE   = 3,   /* late: selective refinement, variance rises      */
} mcr_phase_t;

/*
 * Per-layer statistics provided by the caller from hidden-state calibration.
 * Fill act_variance from the captured all_hs matrix; bos_attn_mass is optional
 * (pass 0.0 if attention weights are not captured).
 */
typedef struct {
    float act_variance;  /* mean squared deviation of hidden states / dim */
    float bos_attn_mass; /* fraction of attention mass on token 0; 0 = unknown */
} mcr_layer_stats_t;

/*
 * Phase detection result.
 */
typedef struct {
    mcr_phase_t phase[MCR_MAX_LAYERS];
    int         n_layers;
    int         mix_end;        /* inclusive last Mix layer   (-1 = no mix phase)  */
    int         compress_start; /* inclusive first Compress layer                  */
    int         compress_end;   /* inclusive last Compress layer                   */
    int         refine_start;   /* inclusive first Refine layer (= n_layers if none) */
    float       smoothed_var[MCR_MAX_LAYERS]; /* 3-tap moving-average */
    float       var_min;        /* minimum smoothed variance  */
    float       var_max;        /* maximum smoothed variance  */
    int         phases_valid;   /* 0 = flat profile, uniform treatment recommended  */
} mcr_result_t;

/*
 * Detect MCR phase boundaries from per-layer statistics.
 *
 * stats         [n_layers]: per-layer stats (fill act_variance from all_hs)
 * n_layers:     number of transformer layers
 * compress_thr: layers with smoothed_var <= var_min * compress_thr labelled
 *               Compress (default 1.5 = "within 50% of the global minimum")
 *
 * Returns 0 on success.
 */
int mcr_detect_phases(mcr_result_t *r,
                      const mcr_layer_stats_t *stats,
                      int n_layers,
                      float compress_thr);

/*
 * Allocate per-layer rank budget given MCR phases and a total rank budget.
 *
 * Scale factors are multiplied on base_rank = total_budget / n_layers:
 *   mix_scale      > 1.0   e.g. 1.5  (protect mixing layers)
 *   compress_scale < 1.0   e.g. 0.35 (cheap to project; model already compressed)
 *   refine_scale   ~ 1.2   e.g. 1.2  (moderate protection for task features)
 *
 * Results are clamped to [min_rank, max_rank]; the total is then re-normalised
 * to the requested budget.  ranks_out[n_layers] is filled.
 */
void mcr_rank_budget(const mcr_result_t *r,
                     int total_budget,
                     int min_rank, int max_rank,
                     float mix_scale, float compress_scale, float refine_scale,
                     int *ranks_out);

void mcr_print(const mcr_result_t *r);

/* =========================================================================
 * Feature 2: Attention-sink bypass
 * ========================================================================= */

#define SINK_MAX_ITEMS 128

typedef struct {
    int   indices[SINK_MAX_ITEMS]; /* sample/position indices identified as sinks */
    float norms[SINK_MAX_ITEMS];   /* L2 norm at each sink index                  */
    int   n_sinks;
    float norm_mean;
    float norm_std;
    float sigma_threshold;
    int   valid;
} sink_ctx_t;

/*
 * Detect sink items from an array of L2 norms.
 * An item is a sink if its norm > mean + sigma_threshold * std.
 * Index 0 is always included if its norm >= mean + 1.5 * std (BOS heuristic).
 *
 * norms[n]:        L2 norm per item (sample index or token position)
 * n:               number of items
 * sigma_threshold: typically 3.0
 */
int sink_detect(sink_ctx_t *ctx, const float *norms, int n, float sigma_threshold);

/*
 * Check whether the dominant sink direction is well-represented by the
 * existing PCA basis.
 *
 * mean_sink_hs[dim]: mean hidden state of sink items at a representative layer
 * basis[n_comp*dim]: PCA components row-major double [n_comp × dim]
 * cos_threshold:     if max |cosine| < this, the direction is underrepresented
 *                    (typical: 0.5)
 *
 * If underrepresented, extra_dir[dim] (caller-allocated) is filled with the
 * unit sink direction and the function returns 1.
 * Returns 0 if adequately covered, -1 on error.
 */
int sink_check_basis_coverage(const sink_ctx_t *ctx,
                               const float  *mean_sink_hs,
                               const double *basis,
                               int n_comp, int dim,
                               float cos_threshold,
                               float *extra_dir);

void sink_print(const sink_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GEODESSICAL_MCR_COMPRESS_H */
