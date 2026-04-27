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
 * TensorOS — Braniac: Brain-Inspired Predictive Coding + Gradient Refinement
 *
 * A biologically-plausible training algorithm combining:
 *   Phase 1: Local Hebbian learning via predictive coding (cortical settling)
 *   Phase 2: Global gradient refinement on residual errors (neuromodulation)
 *   Phase 3: Lateral inhibition / sparsification (cortical competition)
 *
 * Based on neuroscience of cortical columns, predictive coding (Rao & Ballard
 * 1999, Friston 2005), and the PC–backprop equivalence (Whittington & Bogacz
 * 2017). Implements a dual-phase architecture where fast local Hebbian updates
 * handle most learning, and a weak global gradient signal fine-tunes residuals.
 *
 * JIT-compiled hot paths: PC inference inner loop and Hebbian outer product
 * are compiled to native x86_64 SSE2 code via the TensorOS JIT subsystem.
 * =============================================================================*/

#ifndef TENSOROS_BRANIAC_H
#define TENSOROS_BRANIAC_H

#include "kernel/core/kernel.h"

/* =============================================================================
 * Limits
 * =============================================================================*/

#define BRANIAC_MAX_COLUMNS     8       /* Max cortical columns (layers - 1) */
#define BRANIAC_MAX_LAYERS      9       /* BRANIAC_MAX_COLUMNS + 1 */
#define BRANIAC_MAX_DIM         512     /* Max neurons per layer */
#define BRANIAC_MAX_BATCH       64      /* Max batch size */
#define BRANIAC_MAX_WEIGHTS     (BRANIAC_MAX_DIM * BRANIAC_MAX_DIM)

/* Activation types */
#define BRANIAC_ACT_RELU        0
#define BRANIAC_ACT_TANH        1
#define BRANIAC_ACT_GELU        2

/* Beta schedule types */
#define BRANIAC_SCHED_COSINE    0
#define BRANIAC_SCHED_LINEAR    1
#define BRANIAC_SCHED_CONSTANT  2

/* =============================================================================
 * Configuration
 * =============================================================================*/

typedef struct {
    /* Architecture */
    int   layer_sizes[BRANIAC_MAX_LAYERS];
    int   num_layers;                   /* Including input layer */

    /* Phase 1: Predictive Coding */
    float pc_lr;                        /* Value neuron settling rate = 0.05 */
    float gen_lr;                       /* Generative weight Hebbian rate = 0.005 */
    float precision_lr;                 /* Precision adaptation rate = 0.0005 */
    int   pc_iterations;                /* Settling iterations = 5 */

    /* Phase 2: Gradient Refinement */
    float refine_lr;                    /* Max refinement rate = 0.0005 */
    int   refine_schedule;              /* BRANIAC_SCHED_* */
    int   refine_warmup;                /* Steps before activation */

    /* Phase 3: Lateral Inhibition */
    float sparsity_k;                   /* Active fraction = 0.2 */

    /* Flags */
    int   activation;                   /* BRANIAC_ACT_* */
    int   use_precision;                /* Learn precision weights */
    int   use_lateral;                  /* Apply sparsification */
    int   use_refinement;               /* Apply Phase 2 */

    /* Training schedule */
    int   total_steps;                  /* For beta schedule */
} braniac_config_t;

/* =============================================================================
 * Cortical Column — One layer of the hierarchical predictive coding network
 *
 * Contains:
 *   - Forward weights W: bottom-up sensory processing (Layer 4 → 2/3)
 *   - Generative weights G: top-down predictions (Layer 5/6 feedback)
 *   - Precision weights π: learned confidence (astrocyte-like metaplasticity)
 * =============================================================================*/

typedef struct __attribute__((aligned(64))) {
    float W_forward[BRANIAC_MAX_WEIGHTS];   /* [out_dim × in_dim] */
    float W_bias[BRANIAC_MAX_DIM];          /* [out_dim] */
    float G[BRANIAC_MAX_WEIGHTS];           /* [in_dim × out_dim] */
    float G_bias[BRANIAC_MAX_DIM];          /* [in_dim] */
    float log_precision[BRANIAC_MAX_DIM];   /* [in_dim] log-space */
    int   in_dim;
    int   out_dim;
    int   _pad[14];                         /* Pad to 64-byte boundary */
} braniac_column_t;

/* =============================================================================
 * JIT-compiled Braniac kernel function types
 * =============================================================================*/

/* Fused PC column step: prediction + error + feedback in one SSE2 kernel */
typedef void (*braniac_jit_col_fn)(
    float *error_out,           /* [in_dim] prediction error */
    float *feedback_out,        /* [out_dim] feedback for value update */
    const float *target,        /* [in_dim] value neurons below */
    const float *r,             /* [out_dim] this column's value neurons */
    const float *G_weights,     /* [in_dim × out_dim] generative weights */
    const float *G_bias,        /* [in_dim] */
    const float *precision      /* [in_dim] exp(log_precision) */
);

/* JIT-compiled Hebbian outer-product update kernel */
typedef void (*braniac_jit_hebb_fn)(
    float *G_weights,           /* [in_dim × out_dim] updated in-place */
    float *G_bias,              /* [in_dim] updated in-place */
    const float *error,         /* [in_dim] prediction error */
    const float *r,             /* [out_dim] value neurons */
    float lr                    /* Learning rate */
);

/* =============================================================================
 * Network — Hierarchical stack of BraniacColumns
 * =============================================================================*/

typedef struct {
    braniac_column_t columns[BRANIAC_MAX_COLUMNS];
    int              num_columns;
    int              layer_sizes[BRANIAC_MAX_LAYERS];
    int              num_layers;
    int              activation;

    /* JIT-compiled per-column step kernels */
    braniac_jit_col_fn  jit_col_step[BRANIAC_MAX_COLUMNS];
    braniac_jit_hebb_fn jit_hebb[BRANIAC_MAX_COLUMNS];
    int                 jit_ready;
} braniac_network_t;

/* =============================================================================
 * Work Buffers — Preallocated scratch for PC dynamics
 * =============================================================================*/

/*
 * WARNING: ~3.4 MB struct — must be static or heap-allocated, NEVER on stack.
 */
typedef struct __attribute__((aligned(64))) {
    /* Value neurons per layer: [batch × dim] */
    float values[BRANIAC_MAX_LAYERS][BRANIAC_MAX_BATCH * BRANIAC_MAX_DIM];
    /* Prediction errors per column (indexed 0..num_columns-1) */
    float errors[BRANIAC_MAX_COLUMNS][BRANIAC_MAX_BATCH * BRANIAC_MAX_DIM];
    /* Per-column feedback signals */
    float feedback[BRANIAC_MAX_COLUMNS][BRANIAC_MAX_BATCH * BRANIAC_MAX_DIM];
    /* Cached exp(log_precision) per column */
    float precision[BRANIAC_MAX_COLUMNS][BRANIAC_MAX_DIM];
    /* Scratch: gradient refinement delta + backprop staging */
    float delta[BRANIAC_MAX_BATCH * BRANIAC_MAX_DIM];
    float delta_swap[BRANIAC_MAX_BATCH * BRANIAC_MAX_DIM];
} braniac_buffers_t;

/* =============================================================================
 * Trainer — Orchestrates the three-phase learning protocol
 * =============================================================================*/

typedef struct {
    braniac_network_t  *network;
    braniac_config_t    config;
    braniac_buffers_t  *buffers;

    int   step_count;
    float current_beta;

    /* Running metrics */
    float last_pc_energy;
    float last_task_loss;
    float last_residual_loss;
    float last_sparsity;
} braniac_trainer_t;

/* Training step metrics */
typedef struct {
    float pc_energy;
    float task_loss;
    float residual_loss;
    float total_loss;
    float beta;
    float avg_sparsity;
    int   step;
} braniac_metrics_t;

/* =============================================================================
 * API
 * =============================================================================*/

/* Create default config (sensible defaults from the Braniac spec) */
braniac_config_t braniac_default_config(void);

/* Initialize network with given config */
int  braniac_init(braniac_network_t *net, const braniac_config_t *config);

/* Initialize trainer (call after braniac_init) */
void braniac_trainer_init(braniac_trainer_t *trainer, braniac_network_t *net,
                          braniac_buffers_t *bufs, const braniac_config_t *config);

/* Forward pass: bottom-up only (fast inference, no PC dynamics)
 * NOTE: uses a static internal buffer — not reentrant. */
void braniac_forward(const braniac_network_t *net, float *output,
                     const float *input, int batch_size);

/* Full PC inference: forward + settling dynamics (accurate but slower) */
void braniac_pc_infer(braniac_trainer_t *trainer, const float *input,
                      float *output, int batch_size);

/* One training step: all three phases */
int  braniac_train_step(braniac_trainer_t *trainer,
                        const float *input, const float *targets,
                        int batch_size, int output_dim,
                        braniac_metrics_t *metrics);

/* Evaluate on a batch (no weight updates) */
void braniac_evaluate(braniac_trainer_t *trainer,
                      const float *input, const float *targets,
                      int batch_size, int output_dim,
                      braniac_metrics_t *metrics);

/* JIT-compile all hot path kernels for the current network topology */
int  braniac_jit_compile(braniac_network_t *net);

/* Print network diagnostics */
void braniac_print_diagnostics(const braniac_trainer_t *trainer);

/* Self-test: create small network, train, verify convergence */
int  braniac_selftest(void);

#endif /* TENSOROS_BRANIAC_H */
