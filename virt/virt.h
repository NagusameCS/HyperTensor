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
 * TensorOS - Near-Zero-Cost Virtualization Layer
 *
 * Provides lightweight virtualization that adds minimal overhead:
 *
 * Strategy for near-zero cost:
 * 1. Use hardware VT-x/VT-d extensions when available
 * 2. Second-level address translation (EPT/NPT) avoids expensive page walks
 * 3. GPU passthrough via IOMMU/VT-d for direct hardware access
 * 4. Shared tensor memory pages between host and guest (no copies)
 * 5. Paravirtualized AI ops - guest knows it's virtualized and uses
 *    hypercalls for tensor operations instead of emulating GPU
 * 6. Lightweight containers (cgroups-style) for simple isolation
 *
 * Three isolation levels:
 * - VIRT_LEVEL_NONE:      Direct execution, no isolation
 * - VIRT_LEVEL_CONTAINER:  Namespace isolation, shared kernel (<1% overhead)
 * - VIRT_LEVEL_VM:         Full VM with EPT/VT-x (~2-3% overhead)
 * =============================================================================*/

#ifndef TENSOROS_VIRT_H
#define TENSOROS_VIRT_H

#include "kernel/core/kernel.h"

/* Virtualization capabilities */
#define VIRT_CAP_VTX        (1 << 0)  /* Intel VT-x */
#define VIRT_CAP_AMD_V      (1 << 1)  /* AMD-V */
#define VIRT_CAP_EPT        (1 << 2)  /* Extended Page Tables */
#define VIRT_CAP_NPT        (1 << 3)  /* Nested Page Tables */
#define VIRT_CAP_IOMMU      (1 << 4)  /* I/O MMU (VT-d / AMD-Vi) */
#define VIRT_CAP_SR_IOV     (1 << 5)  /* SR-IOV for GPU sharing */

typedef enum {
    VIRT_LEVEL_NONE      = 0,  /* No isolation */
    VIRT_LEVEL_CONTAINER = 1,  /* Lightweight container */
    VIRT_LEVEL_VM        = 2,  /* Full virtual machine */
} virt_level_t;

/* =============================================================================
 * Container (Lightweight Isolation)
 * Namespace-based isolation with near-zero overhead
 * =============================================================================*/

#define CONTAINER_MAX  64

typedef struct {
    uint32_t    id;
    char        name[64];
    bool        active;
    virt_level_t level;

    /* Resource limits */
    uint64_t    mem_limit;          /* Memory limit in bytes */
    uint64_t    mem_used;
    uint32_t    cpu_shares;         /* CPU time shares (relative) */
    uint64_t    gpu_mem_limit;      /* GPU memory limit */
    uint32_t    gpu_compute_pct;    /* GPU compute % limit */

    /* Namespace isolation */
    uint32_t    pid_namespace;      /* PID namespace ID */
    uint32_t    net_namespace;      /* Network namespace ID */
    uint32_t    fs_namespace;       /* Filesystem namespace ID */

    /* Tensor-specific */
    uint32_t    max_meus;           /* Max concurrent MEUs */
    uint32_t    active_meus;
    uint64_t    tensor_ops_limit;   /* Max tensor ops/sec */

    /* GPU passthrough */
    int32_t     gpu_assigned;       /* -1 = shared, >=0 = dedicated */
    bool        gpu_passthrough;    /* Direct GPU access (VT-d) */
} virt_container_t;

/* =============================================================================
 * Hypercall Interface
 * Guest-to-host calls for paravirtualized tensor operations
 * Avoids the overhead of emulating GPU hardware
 * =============================================================================*/

typedef enum {
    HCALL_TENSOR_ALLOC      = 0x100,
    HCALL_TENSOR_FREE       = 0x101,
    HCALL_TENSOR_MATMUL     = 0x102,
    HCALL_TENSOR_ATTENTION  = 0x103,
    HCALL_TENSOR_SOFTMAX    = 0x104,
    HCALL_TENSOR_TRANSFER   = 0x105,
    HCALL_MODEL_LOAD        = 0x200,
    HCALL_MODEL_INFER       = 0x201,
    HCALL_MODEL_TRAIN_STEP  = 0x202,
    HCALL_GIT_COMMIT        = 0x300,
    HCALL_GIT_PUSH          = 0x301,
} hypercall_t;

typedef struct {
    hypercall_t call;
    uint64_t    args[6];        /* Up to 6 arguments */
    uint64_t    ret;            /* Return value */
} hypercall_frame_t;

/* =============================================================================
 * Shared Memory Region for Zero-Copy Tensor Transfer
 * =============================================================================*/

typedef struct {
    uint64_t    host_phys;      /* Host physical address */
    uint64_t    guest_phys;     /* Guest physical address */
    uint64_t    size;           /* Region size */
    bool        writable;       /* Guest can write */
    uint32_t    container_id;   /* Owning container */
} shared_mem_region_t;

#define SHARED_MEM_MAX  128

/* =============================================================================
 * Virtualization API
 * =============================================================================*/

/* Initialization */
int  virt_layer_init(void);
uint64_t virt_get_capabilities(void);

/* Container lifecycle */
virt_container_t *virt_container_create(const char *name, virt_level_t level);
int  virt_container_destroy(uint32_t container_id);
int  virt_container_start(uint32_t container_id);
int  virt_container_stop(uint32_t container_id);

/* Resource limits */
int  virt_container_set_mem_limit(uint32_t container_id, uint64_t bytes);
int  virt_container_set_cpu_shares(uint32_t container_id, uint32_t shares);
int  virt_container_set_gpu_limit(uint32_t container_id, uint64_t mem_bytes,
                                    uint32_t compute_pct);
int  virt_container_set_meu_limit(uint32_t container_id, uint32_t max_meus);

/* GPU passthrough */
int  virt_container_assign_gpu(uint32_t container_id, uint32_t gpu_id,
                                 bool passthrough);

/* Shared memory for zero-copy tensors */
int  virt_shared_mem_create(uint32_t container_id, uint64_t size,
                              shared_mem_region_t *region);
int  virt_shared_mem_destroy(shared_mem_region_t *region);

/* Hypercall handling */
int  virt_handle_hypercall(uint32_t container_id, hypercall_frame_t *frame);

/* Monitoring */
int  virt_container_get_stats(uint32_t container_id, uint64_t *mem_used,
                                uint64_t *cpu_time, uint64_t *tensor_ops);

#endif /* TENSOROS_VIRT_H */
