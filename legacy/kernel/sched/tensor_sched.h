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
 * TensorOS - Tensor-Aware Scheduler Header
 *
 * Unlike traditional OS schedulers that treat all processes equally, this
 * scheduler understands AI workloads natively. It makes decisions based on:
 *  - Tensor operation type (matmul, conv, attention)
 *  - Memory access patterns (sequential weight loading vs random)
 *  - Device affinity (GPU/TPU/CPU)
 *  - Batch size and throughput requirements
 *  - Energy efficiency targets
 * =============================================================================*/

#ifndef TENSOROS_TENSOR_SCHED_H
#define TENSOROS_TENSOR_SCHED_H

#include "kernel/core/kernel.h"

/* =============================================================================
 * Scheduler Configuration
 * =============================================================================*/

#define SCHED_MAX_MEUS           256    /* Max concurrent model execution units */
#define SCHED_MAX_GPUS           16     /* Max GPUs managed */
#define SCHED_MAX_TPUS           8      /* Max TPUs managed */
#define SCHED_TIMESLICE_MS       10     /* Base timeslice in milliseconds */
#define SCHED_AI_TIMESLICE_MS    50     /* Extended timeslice for AI tasks */
#define SCHED_PREEMPT_THRESHOLD  100    /* Tensor ops before preemption check */

/* Scheduling policies */
typedef enum {
    SCHED_POLICY_THROUGHPUT  = 0,  /* Maximize tensor ops/sec */
    SCHED_POLICY_LATENCY    = 1,  /* Minimize inference latency */
    SCHED_POLICY_EFFICIENCY = 2,  /* Minimize energy per inference */
    SCHED_POLICY_FAIR       = 3,  /* Fair sharing across models */
} sched_policy_t;

/* =============================================================================
 * Tensor Operation Hints - Used for scheduling decisions
 * =============================================================================*/

typedef enum {
    TOP_MATMUL       = 0,   /* Matrix multiplication - GPU heavy */
    TOP_CONV2D       = 1,   /* 2D convolution */
    TOP_ATTENTION    = 2,   /* Attention mechanism - memory bound */
    TOP_SOFTMAX      = 3,   /* Softmax - compute bound */
    TOP_LAYERNORM    = 4,   /* Layer normalization */
    TOP_EMBEDDING    = 5,   /* Embedding lookup - memory bound */
    TOP_ACTIVATION   = 6,   /* Activation functions */
    TOP_REDUCE       = 7,   /* Reduction operations */
    TOP_ELEMENTWISE  = 8,   /* Element-wise operations */
    TOP_CUSTOM       = 9,
} tensor_op_type_t;

/* Operation profile for scheduling decisions */
typedef struct {
    tensor_op_type_t op_type;
    uint64_t         flop_estimate;     /* Estimated FLOPs */
    uint64_t         memory_bytes;      /* Memory access estimate */
    float            compute_intensity; /* FLOPs / byte accessed */
    bool             requires_gpu;
    uint32_t         preferred_device;
} tensor_op_profile_t;

/* =============================================================================
 * GPU/TPU Device State (tracked by scheduler)
 * =============================================================================*/

typedef struct {
    uint32_t    device_id;
    bool        active;
    uint32_t    utilization_pct;    /* 0-100 */
    uint64_t    vram_total;
    uint64_t    vram_free;
    uint64_t    ops_per_sec;        /* Current throughput */
    uint32_t    temperature_c;      /* Thermal state */
    uint32_t    power_watts;        /* Power consumption */
    uint32_t    meu_count;          /* Number of MEUs assigned */
    model_exec_unit_t *meu_list;    /* MEUs assigned to this device */
} device_state_t;

/* =============================================================================
 * Scheduler Run Queues
 * =============================================================================*/

typedef struct {
    model_exec_unit_t *head;
    model_exec_unit_t *tail;
    uint32_t           count;
} sched_queue_t;

typedef struct {
    /* Per-priority run queues */
    sched_queue_t      queues[6];  /* One per priority level */

    /* Device tracking */
    device_state_t     gpus[SCHED_MAX_GPUS];
    device_state_t     tpus[SCHED_MAX_TPUS];
    uint32_t           gpu_count;
    uint32_t           tpu_count;

    /* Currently running MEUs (per CPU) */
    model_exec_unit_t *current[64]; /* Max 64 CPUs */

    /* Scheduling policy */
    sched_policy_t     policy;

    /* Statistics */
    uint64_t           total_dispatches;
    uint64_t           total_preemptions;
    uint64_t           total_migrations;    /* Device-to-device moves */
    uint64_t           total_tensor_ops;
    uint64_t           avg_latency_us;      /* Average dispatch latency */

    /* Batch coalescing state */
    uint32_t           coalesce_window_ms;  /* Window for batching requests */
    uint32_t           pending_batch_count;
} tensor_scheduler_t;

extern tensor_scheduler_t g_scheduler;

/* =============================================================================
 * Scheduler API
 * =============================================================================*/

/* Initialization */
void tensor_sched_init(void);

/* MEU lifecycle */
model_exec_unit_t *meu_create(const char *name, meu_type_t type,
                               meu_priority_t priority);
void meu_destroy(model_exec_unit_t *meu);
model_exec_unit_t *meu_find_by_id(uint64_t meu_id);
int  meu_set_model(model_exec_unit_t *meu, uint64_t model_hash,
                    uint64_t param_count, tensor_dtype_t dtype);
int  meu_set_resource_budget(model_exec_unit_t *meu, uint64_t mem_bytes,
                              uint64_t vram_bytes);

/* Scheduling operations */
void tensor_sched_enqueue(model_exec_unit_t *meu);
void tensor_sched_dequeue(model_exec_unit_t *meu);
bool tensor_sched_has_pending(void);
void tensor_sched_dispatch(void);
void tensor_sched_yield(model_exec_unit_t *meu);
void tensor_sched_block(model_exec_unit_t *meu, const char *reason);
void tensor_sched_unblock(model_exec_unit_t *meu);

/* Tensor operation hints - tell scheduler what's coming */
void tensor_sched_hint_op(model_exec_unit_t *meu, tensor_op_profile_t *profile);
void tensor_sched_hint_batch_size(model_exec_unit_t *meu, uint32_t batch_size);

/* Device management */
int  tensor_sched_assign_device(model_exec_unit_t *meu, uint32_t device_id);
int  tensor_sched_migrate_device(model_exec_unit_t *meu, uint32_t new_device);
void tensor_sched_balance_devices(void);

/* Policy */
void tensor_sched_set_policy(sched_policy_t policy);

/* Batch coalescing - combine small inference requests */
void tensor_sched_coalesce_enable(uint32_t window_ms);
void tensor_sched_coalesce_flush(void);

/* Statistics */
void tensor_sched_get_stats(uint64_t *ops, uint64_t *dispatches,
                             uint64_t *avg_latency_us);

#endif /* TENSOROS_TENSOR_SCHED_H */
