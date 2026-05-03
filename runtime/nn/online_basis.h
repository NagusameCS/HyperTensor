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
 * online_basis.h - Online PCA basis update triggered by speculative-decode rejections
 *
 * Background
 * ----------
 * During speculative decoding the draft model proposes a batch of tokens and
 * the target model verifies them.  When the target model rejects a draft token
 * the hidden state at that position diverged far enough from the draft's
 * prediction to matter.  That divergence is a sharp signal that the current
 * PCA basis is stale for the present input distribution.
 *
 * This file implements an online basis updater that hooks into the speculative
 * decode rejection path.  Each rejection fires an Oja's-rule rank-1 update to
 * the stored PCA basis using the residual (target hidden state minus draft
 * hidden state) as the new sample direction.
 *
 * Oja's rule for one principal component:
 *
 *   w_{t+1} = w_t + eta * x * (x^T w_t)
 *   w_{t+1} = w_{t+1} / ||w_{t+1}||
 *
 * For k components (Gram-Schmidt orthogonalised Oja):
 *
 *   for i in 0..k-1:
 *     w_i += eta * x * (x^T w_i)  - deflation from already-updated components
 *     w_i /= ||w_i||
 *
 * The learning rate eta decays as eta_0 / sqrt(t) to guarantee convergence.
 *
 * Integration points
 * ------------------
 * 1. After every speculative batch, call onb_record_rejection() with the
 *    layer index, the target hidden state, and the draft hidden state.
 *    The residual is added to a pending update queue.
 *
 * 2. onb_apply_pending() is called between decode steps (not inside the hot
 *    path).  It drains the queue and fires Oja updates for each pending sample.
 *
 * 3. The caller (llm.c GP path) reads the updated basis via onb_get_basis()
 *    and re-projects w_proj on the fly for the layers that were updated.
 *    Basis staleness is tracked per layer via onb_basis_version[].
 *
 * References
 * ----------
 * Oja (1982) — A simplified neuron model as a principal component analyzer.
 * OjaKV (Sep 2025) — Online adaptive PCA for KV cache via Oja's rule.
 * RAP (2024) — RL-based adaptive pruning per request.
 */

#ifndef GEODESSICAL_ONLINE_BASIS_H
#define GEODESSICAL_ONLINE_BASIS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum layers and subspace dimension supported. */
#define ONB_MAX_LAYERS  256
#define ONB_MAX_DIM    8192   /* full hidden dim upper bound */
#define ONB_MAX_K       512   /* max PCA rank to maintain online */

/* Maximum pending rejection samples before forced flush. */
#define ONB_QUEUE_CAP  256

/* -------------------------------------------------------------------------
 * Per-layer online basis state
 * -------------------------------------------------------------------------
 */
typedef struct {
    int    dim;           /* hidden state dimension */
    int    k;             /* current subspace rank */
    float *W;             /* [k x dim] basis matrix, rows are principal directions */
    double eta0;          /* initial learning rate (default 0.01) */
    uint64_t t;           /* sample count seen so far (for eta decay) */
    uint32_t version;     /* incremented on every update; callers track staleness */
    int    active;        /* 1 once first sample received */
} onb_layer_state_t;

/* -------------------------------------------------------------------------
 * Pending rejection sample (lives in the queue)
 * -------------------------------------------------------------------------
 */
typedef struct {
    int    layer;
    float  residual[ONB_MAX_DIM]; /* target_hidden - draft_hidden, truncated to dim */
} onb_pending_t;

/* -------------------------------------------------------------------------
 * Global online basis context (one per model)
 * -------------------------------------------------------------------------
 */
typedef struct {
    int    n_layers;
    onb_layer_state_t layers[ONB_MAX_LAYERS];

    /* Rejection queue */
    onb_pending_t queue[ONB_QUEUE_CAP];
    int           queue_len;

    /* Aggregate stats */
    uint64_t total_rejections;  /* rejection events received */
    uint64_t total_updates;     /* Oja steps applied */
    uint32_t basis_version[ONB_MAX_LAYERS]; /* latest version per layer */

    /* Config */
    double   eta0;              /* global initial LR override (0 = use per-layer default) */
    int      updates_per_rejection; /* Oja steps to run per queued sample (default 1) */
    int      min_rejections_before_update; /* gate: skip updates if too few data (default 4) */
} onb_ctx_t;

/* -------------------------------------------------------------------------
 * Lifecycle
 * -------------------------------------------------------------------------
 */

/*
 * Initialise an online basis context.
 *
 * Copies initial bases from existing_bases[layer][k x dim] (F32 row-major).
 * If existing_bases is NULL, all layers start with an identity basis of rank k.
 *
 * n_layers:        number of transformer layers
 * dims[]:          hidden dimension per layer (usually all the same)
 * ks[]:            subspace rank per layer (from current GP calibration)
 * existing_bases[]: pointer array [n_layers], each float* [k x dim]
 *                   (may be NULL; individual pointers may also be NULL)
 * eta0:            initial Oja learning rate (0 = use default 0.01)
 */
int onb_init(onb_ctx_t *ctx,
             int n_layers,
             const int *dims,
             const int *ks,
             float * const *existing_bases,
             double eta0);

/* Destroy and free all memory. */
void onb_destroy(onb_ctx_t *ctx);

/* Reset sample counts and queue but keep basis vectors. */
void onb_reset_stats(onb_ctx_t *ctx);

/* -------------------------------------------------------------------------
 * Recording rejections (called from the speculative decode verify loop)
 * -------------------------------------------------------------------------
 */

/*
 * Record one speculative-decode rejection event.
 *
 * layer:         which transformer layer the hidden states are from
 * target_hidden: hidden state from the target (verifier) model [dim floats]
 * draft_hidden:  hidden state from the draft model at the same position [dim floats]
 *
 * Adds residual = (target_hidden - draft_hidden) to the pending queue.
 * Does NOT run Oja immediately (to keep rejection path latency near zero).
 * Thread-unsafe: call from the decode thread only.
 *
 * Returns 0 on success, -1 if queue full (sample dropped, stats updated).
 */
int onb_record_rejection(onb_ctx_t *ctx,
                         int layer,
                         const float *target_hidden,
                         const float *draft_hidden);

/*
 * Same as onb_record_rejection but takes the residual directly instead of
 * computing it.  Use this when the caller already has the residual.
 */
int onb_record_residual(onb_ctx_t *ctx,
                        int layer,
                        const float *residual);

/* -------------------------------------------------------------------------
 * Applying pending updates (called between decode steps)
 * -------------------------------------------------------------------------
 */

/*
 * Drain the pending queue and run Oja updates.
 *
 * Should be called once per decode step, outside the hot path.
 * Returns number of Oja update steps actually applied.
 *
 * After this call, any layer whose basis changed has its
 * ctx->basis_version[layer] incremented.
 */
int onb_apply_pending(onb_ctx_t *ctx);

/*
 * Force a single Oja update for a given layer with an explicit sample.
 * Primarily for testing.
 */
void onb_oja_update(onb_layer_state_t *ls, const float *x);

/* -------------------------------------------------------------------------
 * Basis access
 * -------------------------------------------------------------------------
 */

/*
 * Get a pointer to the current basis for layer l: [k x dim] F32 row-major.
 * The caller should not cache this pointer across onb_apply_pending() calls
 * without checking basis_version.
 */
static inline const float *onb_get_basis(const onb_ctx_t *ctx, int l) {
    if (l < 0 || l >= ctx->n_layers) return 0;
    return ctx->layers[l].W;
}

static inline int onb_get_k(const onb_ctx_t *ctx, int l) {
    if (l < 0 || l >= ctx->n_layers) return 0;
    return ctx->layers[l].k;
}

static inline uint32_t onb_get_version(const onb_ctx_t *ctx, int l) {
    if (l < 0 || l >= ctx->n_layers) return 0;
    return ctx->basis_version[l];
}

/* -------------------------------------------------------------------------
 * GP weight re-projection after a basis update
 * -------------------------------------------------------------------------
 *
 * When a layer's basis changes, its W_proj = W @ P must be recomputed.
 * These helpers do that in place using the updated basis.
 *
 * W_orig:    original weight matrix [m x dim] F32 (dequantized or already F32)
 * W_proj:    output projected weight [m x k] F32
 * basis:     new basis [k x dim] F32 (from onb_get_basis)
 * m, dim, k: matrix dimensions
 */
void onb_reproject_weight(const float *W_orig, float *W_proj,
                          const float *basis,
                          int m, int dim, int k);

/* -------------------------------------------------------------------------
 * Statistics
 * -------------------------------------------------------------------------
 */

typedef struct {
    uint64_t total_rejections;
    uint64_t total_updates;
    uint64_t queue_drops;       /* samples dropped due to full queue */
    int      layers_updated;    /* layers that received at least one update */
    double   mean_update_magnitude; /* mean ||delta_W||_F across all updates */
} onb_stats_t;

void onb_get_stats(const onb_ctx_t *ctx, onb_stats_t *stats);
void onb_print_stats(const onb_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GEODESSICAL_ONLINE_BASIS_H */
