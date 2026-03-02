/* =============================================================================
 * TensorOS - Security Sandbox for Model Isolation
 *
 * Every model runs in an isolated sandbox that enforces:
 * 1. Memory isolation: Models cannot read other models' weights
 * 2. Device isolation: GPU access controlled per-sandbox
 * 3. Network isolation: Models can't phone home without permission
 * 4. Filesystem isolation: Models only see their own directory
 * 5. Auditable execution: Every tensor op is logged with provenance
 * 6. Reproducibility: Deterministic execution for debugging
 * =============================================================================*/

#ifndef TENSOROS_SANDBOX_H
#define TENSOROS_SANDBOX_H

#include "kernel/core/kernel.h"

/* =============================================================================
 * Sandbox Policy Definitions
 * =============================================================================*/

typedef enum {
    SANDBOX_POLICY_STRICT    = 0,  /* No external access at all */
    SANDBOX_POLICY_STANDARD  = 1,  /* Filesystem + limited network */
    SANDBOX_POLICY_PERMISSIVE = 2, /* Full access with auditing */
} sandbox_policy_t;

/* Fine-grained permissions */
#define SANDBOX_PERM_READ_MODEL     (1 << 0)   /* Read model weights */
#define SANDBOX_PERM_WRITE_MODEL    (1 << 1)   /* Write/modify weights */
#define SANDBOX_PERM_GPU_ACCESS     (1 << 2)   /* Use GPU compute */
#define SANDBOX_PERM_NETWORK_LOCAL  (1 << 3)   /* Local network only */
#define SANDBOX_PERM_NETWORK_REMOTE (1 << 4)   /* Internet access */
#define SANDBOX_PERM_FS_READ        (1 << 5)   /* Read filesystem */
#define SANDBOX_PERM_FS_WRITE       (1 << 6)   /* Write filesystem */
#define SANDBOX_PERM_IPC            (1 << 7)   /* Inter-model communication */
#define SANDBOX_PERM_SPAWN          (1 << 8)   /* Spawn sub-models */
#define SANDBOX_PERM_GIT            (1 << 9)   /* Git operations */
#define SANDBOX_PERM_DEPLOY         (1 << 10)  /* Deploy as service */

/* =============================================================================
 * Audit Log Entry
 * Every tensor operation is recorded for provenance tracking
 * =============================================================================*/

typedef enum {
    AUDIT_TENSOR_ALLOC,
    AUDIT_TENSOR_FREE,
    AUDIT_TENSOR_OP,
    AUDIT_MODEL_LOAD,
    AUDIT_MODEL_SAVE,
    AUDIT_NETWORK_ACCESS,
    AUDIT_FS_ACCESS,
    AUDIT_IPC_SEND,
    AUDIT_IPC_RECV,
    AUDIT_PERMISSION_DENIED,
    AUDIT_SANDBOX_VIOLATION,
} audit_event_type_t;

typedef struct {
    audit_event_type_t type;
    uint64_t           timestamp;
    uint64_t           sandbox_id;
    uint64_t           meu_id;
    char               description[256];
    /* For tensor ops */
    uint64_t           op_flops;
    uint64_t           op_bytes;
    tensor_dtype_t     op_dtype;
} audit_entry_t;

#define AUDIT_LOG_SIZE  4096

/* =============================================================================
 * Sandbox Instance
 * =============================================================================*/

#define SANDBOX_MAX  64

typedef struct {
    uint64_t            id;
    char                name[64];
    bool                active;
    sandbox_policy_t    policy;
    uint32_t            permissions;

    /* Resource accounting */
    uint64_t            mem_allocated;
    uint64_t            mem_limit;
    uint64_t            gpu_mem_allocated;
    uint64_t            gpu_mem_limit;
    uint64_t            tensor_ops_count;
    uint64_t            tensor_ops_limit; /* Max ops before forced stop */

    /* Isolation state */
    uint64_t            page_table_root;   /* Separate page table */
    uint32_t            fs_root_inode;     /* Chroot equivalent */

    /* Audit log */
    audit_entry_t       audit_log[256];    /* Ring buffer */
    uint32_t            audit_head;
    uint32_t            audit_count;
    bool                audit_enabled;

    /* Reproducibility */
    uint64_t            random_seed;        /* Fixed seed for determinism */
    bool                deterministic_mode;
} sandbox_t;

/* =============================================================================
 * Sandbox API
 * =============================================================================*/

/* Initialization */
void sandbox_init(void);

/* Lifecycle */
sandbox_t *sandbox_create(const char *name, sandbox_policy_t policy);
int  sandbox_destroy(uint64_t sandbox_id);
int  sandbox_activate(uint64_t sandbox_id);
int  sandbox_deactivate(uint64_t sandbox_id);

/* Permission management */
int  sandbox_grant_permission(uint64_t sandbox_id, uint32_t permissions);
int  sandbox_revoke_permission(uint64_t sandbox_id, uint32_t permissions);
bool sandbox_check_permission(uint64_t sandbox_id, uint32_t permission);

/* Resource limits */
int  sandbox_set_mem_limit(uint64_t sandbox_id, uint64_t bytes);
int  sandbox_set_gpu_limit(uint64_t sandbox_id, uint64_t bytes);
int  sandbox_set_ops_limit(uint64_t sandbox_id, uint64_t max_ops);

/* Enforcement (called by kernel subsystems) */
bool sandbox_allow_tensor_op(uint64_t sandbox_id, tensor_desc_t *tensor);
bool sandbox_allow_network(uint64_t sandbox_id, const char *host, uint16_t port);
bool sandbox_allow_fs_access(uint64_t sandbox_id, const char *path, bool write);
bool sandbox_allow_ipc(uint64_t sandbox_id, uint64_t target_sandbox_id);

/* Auditing */
void sandbox_audit_log(uint64_t sandbox_id, audit_event_type_t type,
                        const char *description);
int  sandbox_audit_dump(uint64_t sandbox_id, audit_entry_t *entries,
                         uint32_t max, uint32_t *count);

/* Reproducibility */
int  sandbox_set_deterministic(uint64_t sandbox_id, bool deterministic,
                                 uint64_t seed);

#endif /* TENSOROS_SANDBOX_H */
