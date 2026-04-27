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

/* =============================================================================
 * Speculative Neural Execution (SNE) — public interface
 *
 * Five inference optimizations: adaptive precision cascade, speculative layer
 * fusion, entropy-aware neuron pruning, DAG scheduling, confidence-gated early
 * exit.  See speculative.c for implementation details.
 * =============================================================================*/

#ifndef TENSOROS_NN_SPECULATIVE_H
#define TENSOROS_NN_SPECULATIVE_H

#include <stdint.h>
#include "runtime/nn/inference.h"
#include "runtime/nn/quantize.h"

/* =============================================================================
 * Technique 1: Adaptive Precision Cascade (APC)
 * =============================================================================*/

/* Entropy threshold: if output entropy exceeds this, escalate to FP32.
 * Tuned empirically: 0.5 bits = high confidence, escalation rare.
 * Range: 0.0 (always FP32) to 2.0 (always INT16, never escalate). */
#define APC_ENTROPY_THRESHOLD 0.5f

/* Statistics for APC */
typedef struct {
    uint64_t total_inferences;
    uint64_t int16_hits;         /* Ran INT16 only (fast path) */
    uint64_t fp32_escalations;   /* Had to escalate to FP32 */
    uint64_t cycles_saved;       /* Estimated cycles saved by INT16 fast path */
} apc_stats_t;

/* Run inference with adaptive precision cascade.
 * Returns actual outputs in `output`. Automatically selects INT16 or FP32. */
void nn_apc_forward(nn_model_t *fp_model, nn_qmodel_t *q_model,
                    float *output, const float *input, apc_stats_t *stats);

/* =============================================================================
 * Technique 2: Speculative Layer Fusion (SLF)
 * =============================================================================*/

/* Maximum tracked layers */
#define SLF_MAX_LAYERS 16

/* Activation signature: compact representation of a layer's output distribution.
 * Uses mean + variance + L1-norm as a 3-number "fingerprint." */
typedef struct {
    float mean;
    float variance;
    float l1_norm;
} activation_sig_t;

/* Speculative layer cache: stores previous layer outputs for reuse */
typedef struct {
    activation_sig_t prev_sig[SLF_MAX_LAYERS];  /* Previous signatures */
    float *cached_output[SLF_MAX_LAYERS];        /* Cached layer outputs */
    int cached_dim[SLF_MAX_LAYERS];              /* Dims of cached outputs */
    int valid[SLF_MAX_LAYERS];                   /* Cache line validity */
    float similarity_threshold;                   /* How similar to reuse */
    uint64_t hits;
    uint64_t misses;
    uint64_t total;
} slf_cache_t;

/* Initialize speculative layer cache */
void slf_cache_init(slf_cache_t *cache, float similarity_threshold);

/* Run inference with speculative layer fusion */
void nn_slf_forward(nn_model_t *model, float *output, const float *input,
                    slf_cache_t *cache);

/* =============================================================================
 * Technique 3: Entropy-Aware Neuron Pruning (EANP)
 * =============================================================================*/

/* Histogram bins for activation distribution tracking */
#define EANP_HIST_BINS 8
#define EANP_MAX_NEURONS 256

/* Per-neuron entropy tracker */
typedef struct {
    uint32_t hist[EANP_MAX_NEURONS][EANP_HIST_BINS]; /* Activation histograms */
    float entropy[EANP_MAX_NEURONS];                   /* Shannon entropy per neuron */
    uint8_t pruned[EANP_MAX_NEURONS];                  /* 1 = pruned (low entropy) */
    int num_neurons;
    uint32_t sample_count;
    float prune_threshold;   /* Entropy below this → prune */
    int num_pruned;
} eanp_tracker_t;

/* Initialize neuron pruning tracker for a layer */
void eanp_init(eanp_tracker_t *tracker, int num_neurons, float prune_threshold);

/* Update statistics based on observed activations */
void eanp_observe(eanp_tracker_t *tracker, const float *activations, int n);

/* Recompute entropy and update prune masks. Called periodically. */
void eanp_update_masks(eanp_tracker_t *tracker);

/* Apply pruning: zero out pruned neuron contributions to skip compute.
 * Returns number of active (non-pruned) neurons. */
int eanp_apply(const eanp_tracker_t *tracker, float *weights, int in_dim,
               int out_dim);

/* =============================================================================
 * Technique 4: Kernel-Level DAG Scheduling
 * =============================================================================*/

/* Operation types in the tensor DAG */
#define DAG_OP_MATMUL   0
#define DAG_OP_BIAS     1
#define DAG_OP_RELU     2
#define DAG_OP_SOFTMAX  3
#define DAG_OP_CONV2D   4

#define DAG_MAX_OPS 64

/* Single node in the compute DAG */
typedef struct {
    int op_type;        /* DAG_OP_* */
    int deps[4];        /* Dependencies (indices into dag, -1 = none) */
    int num_deps;
    int out_size;       /* Output buffer size in floats */
    int scheduled;      /* 1 = already executed */
    int priority;       /* Topological priority (lower = earlier) */
    uint64_t est_cycles;/* Estimated cycle cost */
} dag_node_t;

/* Compute DAG for a model */
typedef struct {
    dag_node_t nodes[DAG_MAX_OPS];
    int num_ops;
    int order[DAG_MAX_OPS];     /* Execution order after scheduling */
    int num_scheduled;
    uint64_t total_est_cycles;
} compute_dag_t;

/* Build a compute DAG from a neural network model */
void dag_build(compute_dag_t *dag, const nn_model_t *model);

/* Schedule DAG using topological sort with priority ordering.
 * Applies Coffman-Graham-style scheduling for optimal ordering. */
void dag_schedule(compute_dag_t *dag);

/* =============================================================================
 * Technique 5: Confidence-Gated Early Exit
 * =============================================================================*/

/* Early exit statistics */
typedef struct {
    uint64_t total_inferences;
    uint64_t exits_per_layer[NN_MAX_LAYERS]; /* How many exited at each layer */
    uint64_t total_layers_saved;
    float avg_exit_layer;
} early_exit_stats_t;

/* Run inference with confidence-gated early exit.
 * After each hidden layer, computes max activation as confidence proxy.
 * If confidence > threshold, returns early.
 * confidence_threshold: 0.0 = always early exit, 1.0 = never early exit */
void nn_early_exit_forward(nn_model_t *model, float *output, const float *input,
                           float confidence_threshold, early_exit_stats_t *stats);

/* =============================================================================
 * Master API: Speculative Neural Execution (SNE)
 *
 * Combines ALL five techniques into a unified inference call.
 * The OS automatically selects the optimal strategy per-input:
 *   - INT16 for easy/confident inputs (APC)
 *   - Cached layers for similar inputs (SLF)  
 *   - Pruned neurons for redundant features (EANP)
 *   - Reordered ops for cache efficiency (DAG)
 *   - Early exits for obvious inputs (confidence gating)
 * =============================================================================*/

typedef struct {
    apc_stats_t apc;
    slf_cache_t slf;
    eanp_tracker_t eanp[NN_MAX_LAYERS];  /* Per-layer trackers */
    early_exit_stats_t exits;
    compute_dag_t dag;
    int initialized;
    uint64_t total_cycles;
    uint64_t total_inferences;
} sne_engine_t;

/* Initialize the Speculative Neural Execution engine */
void sne_init(sne_engine_t *engine, nn_model_t *model);

/* Run a single inference through the full SNE pipeline */
void sne_forward(sne_engine_t *engine, nn_model_t *fp_model, nn_qmodel_t *q_model,
                 float *output, const float *input);

/* Print SNE statistics */
void sne_print_stats(const sne_engine_t *engine);

/* Run all speculative inference demos and benchmarks */
void sne_run_demos(void);

#endif /* TENSOROS_NN_SPECULATIVE_H */
