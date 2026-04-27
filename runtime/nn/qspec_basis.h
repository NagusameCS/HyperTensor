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
 * qspec_basis.h - Cross-quantization shared basis test (Feature 5)
 *                 Failure-mode-targeted rank allocation   (Feature 6)
 *
 * Feature 5: QSPEC-adjacent shared basis analysis
 *   QSPEC (EMNLP 2025) shows weight sharing across quantization levels in a
 *   speculator/verifier pair.  Nobody has asked whether a *single PCA/low-rank
 *   basis* works across quantization levels of the same weight tensor.
 *
 *   Operationalisation: if the activation-space PCA basis (Pt, used in the
 *   manifold GP compression) captures the same energy as the weight matrix's
 *   own top singular vectors (SVD), then quantizing the weight matrix doesn't
 *   change which subspace is important.  One basis computed on the fp16/bfloat16
 *   activations is also near-optimal for the int4 version of the same weight.
 *
 *   Measurement: for each compressed layer slot,
 *     proj_energy   = ||W_proj||_F^2 / ||W||_F^2     (from axex_manifold_weight_t)
 *     svd_explained = 1 - frobenius_err^2             (from axex_compressed_weight_t)
 *     alignment     = proj_energy / svd_explained
 *
 *   If alignment ≈ 1.0, the PCA basis is as good as the SVD basis for that
 *   weight → cross-quant basis sharing is viable.
 *   If alignment << 1.0, the PCA basis misses some of the weight's structure →
 *   different quant levels would benefit from separate basis computation.
 *
 * Feature 6: ErrorAtlas failure-mode-targeted rank allocation
 *   ErrorAtlas (2024) identifies 17 distinct LLM failure modes.  Compression
 *   literature reports "accuracy drop" as a single number; nobody checks which
 *   failure class the compressed model enters first.  Our hypothesis:
 *     Early layers (l < n/3)      high frob_err → factual recall failures
 *     Middle layers (n/3..2n/3)   high frob_err → reasoning/inference failures
 *     Late layers (l > 2n/3)      high frob_err → output coherence failures
 *     Uniform high frob_err       all zones      → long-context recall failures
 *
 *   This module detects the dominant failure zone and returns per-layer rank
 *   multipliers that concentrate the rank budget on the at-risk layers, giving
 *   targeted quality protection without increasing total memory.
 */

#ifndef GEODESSICAL_QSPEC_BASIS_H
#define GEODESSICAL_QSPEC_BASIS_H

#include "runtime/nn/axiom_exploit.h"  /* axex_get_compressed_layer, axex_get_manifold_layer */
#include "runtime/nn/geo_research.h"   /* DIFFPLAN_MAX_LAYERS                                 */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Feature 5: Cross-quantization shared basis test
 * ========================================================================= */

#define QSPEC_MAX_LAYERS 256
#define QSPEC_MAX_SLOTS    8

typedef struct {
    int   layer;
    int   slot;           /* weight slot (0=FFN_down, 1=Q, 2=K, 3=V, 4=O, …) */
    float proj_energy;    /* ||W_proj||_F^2 / ||W||_F^2 from manifold weight    */
    float svd_explained;  /* 1 - frobenius_err^2 from SVD-compressed weight     */
    float alignment;      /* proj_energy / svd_explained ∈ [0,∞); 1.0 = perfect */
    int   shared_ok;      /* 1 if alignment >= share_threshold                   */
} qspec_entry_t;

typedef struct {
    qspec_entry_t entries[QSPEC_MAX_LAYERS * QSPEC_MAX_SLOTS];
    int           n_entries;
    float         mean_alignment;
    float         min_alignment;
    float         share_threshold;
    int           n_shared_ok;
} qspec_result_t;

/*
 * Test shared-basis quality across all compressed layers.
 *
 * Evaluates every (layer, slot) pair where both axex_get_compressed_layer()
 * and axex_get_manifold_layer() return non-NULL.  For each pair the alignment
 * score is computed as described above.
 *
 * n_layers:        number of transformer layers
 * share_threshold: minimum alignment to call a slot "shared OK" (e.g. 0.80)
 *
 * Returns number of valid entries evaluated, -1 on error.
 */
int  qspec_test_shared_basis(qspec_result_t *result, int n_layers, float share_threshold);

void qspec_print(const qspec_result_t *result);

/* =========================================================================
 * Feature 6: Failure-mode-targeted rank allocation
 * ========================================================================= */

typedef enum {
    FMODE_NONE      = 0,
    FMODE_FACTUAL   = 1,  /* early-layer frob_err peak → hallucination risk     */
    FMODE_REASONING = 2,  /* mid-layer frob_err peak  → reasoning failures      */
    FMODE_COHERENCE = 3,  /* late-layer frob_err peak → output incoherence      */
    FMODE_CONTEXT   = 4,  /* uniformly high frob_err  → long-context recall     */
} fail_mode_t;

typedef struct {
    fail_mode_t dominant_mode;
    float       early_err;   /* mean frob_err layers [0, n/3)   */
    float       mid_err;     /* mean frob_err layers [n/3, 2n/3)*/
    float       late_err;    /* mean frob_err layers [2n/3, n)  */
    float       global_err;  /* mean frob_err across all layers  */
    /* Per-layer rank multipliers to focus budget on the at-risk zone */
    float       rank_scale[DIFFPLAN_MAX_LAYERS];
    int         n_layers;
    int         valid;
} frank_result_t;

/*
 * Build a failure-mode rank profile from the per-layer Frobenius error array.
 *
 * frob_err[n_layers]: reconstruction error per layer (from axex_get_compressed_layer)
 * n_layers:          number of transformer layers
 * dominant_boost:    rank multiplier for the at-risk zone (e.g. 1.8)
 * decay:             factor applied to layers just outside the at-risk zone (e.g. 0.6)
 *                    (layers in the other zones get rank_scale 1.0)
 *
 * Returns 0 on success, -1 on error.
 */
int frank_build(frank_result_t *r,
                const float *frob_err, int n_layers,
                float dominant_boost, float decay);

/*
 * Apply frank rank scales to an existing per-layer rank array.
 * Multiplies ranks[l] by rank_scale[l] then clamps to [min_rank, max_rank].
 */
void frank_apply(const frank_result_t *r, int *ranks, int n_layers,
                 int min_rank, int max_rank);

void frank_print(const frank_result_t *r);

#ifdef __cplusplus
}
#endif

#endif /* GEODESSICAL_QSPEC_BASIS_H */
