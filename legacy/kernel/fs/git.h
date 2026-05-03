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
 * TensorOS - Native Git Subsystem
 *
 * Git is built directly into the kernel as a first-class subsystem.
 * This means:
 * - No external git binary needed
 * - Model weights and code are version-controlled natively
 * - Atomic snapshot/rollback of entire model + data states
 * - Efficient binary diff for model weight changes
 * - Built-in LFS-like support for large tensor files
 *
 * Objects are stored in a kernel-managed memory region and persisted
 * to TensorFS. The git protocol is available over the network stack
 * for distributed collaboration.
 * =============================================================================*/

#ifndef TENSOROS_GIT_H
#define TENSOROS_GIT_H

#include "kernel/core/kernel.h"

/* =============================================================================
 * Git Object Types
 * =============================================================================*/

typedef enum {
    GIT_OBJ_BLOB    = 1,
    GIT_OBJ_TREE    = 2,
    GIT_OBJ_COMMIT  = 3,
    GIT_OBJ_TAG     = 4,
    GIT_OBJ_TENSOR  = 5,   /* TensorOS extension: native tensor object */
    GIT_OBJ_MODEL   = 6,   /* TensorOS extension: model checkpoint */
} git_obj_type_t;

/* SHA-256 hash (256 bits = 32 bytes) */
typedef struct {
    uint8_t bytes[32];
} git_hash_t;

/* Git object header */
typedef struct {
    git_obj_type_t  type;
    uint64_t        size;
    git_hash_t      hash;
} git_obj_header_t;

/* =============================================================================
 * Git Blob - File content
 * =============================================================================*/
typedef struct {
    git_obj_header_t header;
    uint8_t         *data;
} git_blob_t;

/* =============================================================================
 * Git Tree - Directory listing
 * =============================================================================*/

#define GIT_TREE_MAX_ENTRIES 256

typedef struct {
    uint32_t    mode;       /* File mode (0100644, 040000, etc.) */
    char        name[256];  /* Entry name */
    git_hash_t  hash;       /* Hash of blob/tree */
} git_tree_entry_t;

typedef struct {
    git_obj_header_t    header;
    uint32_t            entry_count;
    git_tree_entry_t    entries[GIT_TREE_MAX_ENTRIES];
} git_tree_t;

/* =============================================================================
 * Git Commit
 * =============================================================================*/
typedef struct {
    git_obj_header_t header;
    git_hash_t       tree;          /* Root tree hash */
    git_hash_t       parent;        /* Parent commit (zero = initial) */
    git_hash_t       parent2;       /* Second parent (for merges) */
    char             author[128];
    char             committer[128];
    uint64_t         timestamp;     /* Unix timestamp */
    char             message[1024];
} git_commit_t;

/* =============================================================================
 * TensorOS Extensions: Tensor Object
 * Stores tensor metadata + data with efficient delta encoding
 * =============================================================================*/
typedef struct {
    git_obj_header_t header;
    tensor_desc_t    desc;          /* Tensor descriptor */
    uint64_t         data_offset;   /* Offset into data region */
    uint64_t         data_size;     /* Compressed size */
    uint64_t         original_size; /* Uncompressed size */
    uint8_t          compression;   /* 0=none, 1=zstd, 2=delta */
} git_tensor_obj_t;

/* =============================================================================
 * TensorOS Extensions: Model Checkpoint Object
 * Snapshot of an entire model's state
 * =============================================================================*/
typedef struct {
    git_obj_header_t header;
    char             model_name[64];
    uint64_t         param_count;
    tensor_dtype_t   dtype;
    uint32_t         layer_count;
    git_hash_t       weight_hashes[512]; /* Hash of each layer's weights */
    git_hash_t       optimizer_state;    /* Optimizer state snapshot */
    uint64_t         training_step;
    float            loss;
    float            accuracy;
} git_model_checkpoint_t;

/* =============================================================================
 * Git Reference (branch/tag pointer)
 * =============================================================================*/
typedef struct {
    char        name[128];      /* e.g., "refs/heads/main" */
    git_hash_t  target;         /* Commit hash */
    bool        is_symbolic;    /* Symbolic ref (like HEAD) */
    char        symref[128];    /* Target of symbolic ref */
} git_ref_t;

#define GIT_MAX_REFS 128

/* =============================================================================
 * Git Repository State
 * =============================================================================*/
typedef struct {
    char        path[256];              /* Repository path */
    git_ref_t   head;                   /* HEAD reference */
    git_ref_t   refs[GIT_MAX_REFS];     /* All references */
    uint32_t    ref_count;
    uint64_t    object_count;           /* Total objects in store */
    uint64_t    pack_count;             /* Packfiles */
} git_repo_t;

/* =============================================================================
 * Git Index (staging area)
 * =============================================================================*/

#define GIT_INDEX_MAX_ENTRIES 1024

typedef struct {
    char        path[256];
    git_hash_t  hash;
    uint32_t    mode;
    uint64_t    size;
    uint64_t    mtime;
    bool        modified;
} git_index_entry_t;

typedef struct {
    git_index_entry_t entries[GIT_INDEX_MAX_ENTRIES];
    uint32_t          count;
} git_index_t;

/* =============================================================================
 * Native Git API
 * =============================================================================*/

/* Subsystem initialization */
void git_subsystem_init(void);

/* Repository operations */
int  git_repo_init(const char *path, git_repo_t *repo);
int  git_repo_open(const char *path, git_repo_t *repo);

/* Object operations */
int  git_obj_write(git_repo_t *repo, git_obj_type_t type,
                    const void *data, uint64_t size, git_hash_t *out_hash);
int  git_obj_read(git_repo_t *repo, const git_hash_t *hash,
                   void *buf, uint64_t buf_size, git_obj_header_t *header);
bool git_obj_exists(git_repo_t *repo, const git_hash_t *hash);

/* Hashing */
void git_hash_compute(const void *data, uint64_t size, git_hash_t *out);
bool git_hash_equal(const git_hash_t *a, const git_hash_t *b);
void git_hash_to_hex(const git_hash_t *hash, char *buf); /* buf must be 65 bytes */

/* Tree operations */
int  git_tree_create(git_repo_t *repo, git_tree_t *tree, git_hash_t *out_hash);
int  git_tree_add_entry(git_tree_t *tree, const char *name, uint32_t mode,
                         const git_hash_t *hash);

/* Commit operations */
int  git_commit_create(git_repo_t *repo, const git_hash_t *tree,
                        const git_hash_t *parent, const char *author,
                        const char *message, git_hash_t *out_hash);
int  git_commit_read(git_repo_t *repo, const git_hash_t *hash,
                      git_commit_t *commit);

/* Index (staging) operations */
int  git_index_add(git_index_t *index, const char *path,
                    const git_hash_t *hash, uint32_t mode);
int  git_index_remove(git_index_t *index, const char *path);
int  git_index_write_tree(git_repo_t *repo, git_index_t *index,
                           git_hash_t *out_tree);

/* Reference operations */
int  git_ref_create(git_repo_t *repo, const char *name, const git_hash_t *target);
int  git_ref_update(git_repo_t *repo, const char *name, const git_hash_t *target);
int  git_ref_resolve(git_repo_t *repo, const char *name, git_hash_t *out);
int  git_ref_list(git_repo_t *repo, git_ref_t *refs, uint32_t max, uint32_t *count);

/* High-level operations */
int  git_add(git_repo_t *repo, const char *path);
int  git_commit(git_repo_t *repo, const char *message);
int  git_log(git_repo_t *repo, git_commit_t *commits, uint32_t max, uint32_t *count);
int  git_diff(git_repo_t *repo, const git_hash_t *a, const git_hash_t *b);
int  git_branch_create(git_repo_t *repo, const char *name);
int  git_branch_delete(git_repo_t *repo, const char *name);
int  git_checkout(git_repo_t *repo, const char *ref_name);
int  git_merge(git_repo_t *repo, const char *branch);

/* TensorOS Extensions */
int  git_tensor_store(git_repo_t *repo, const tensor_desc_t *tensor,
                       const void *data, git_hash_t *out_hash);
int  git_tensor_load(git_repo_t *repo, const git_hash_t *hash,
                      tensor_desc_t *tensor, void *buf, uint64_t buf_size);
int  git_model_checkpoint(git_repo_t *repo, const char *model_name,
                           const git_model_checkpoint_t *checkpoint,
                           git_hash_t *out_hash);
int  git_model_restore(git_repo_t *repo, const git_hash_t *hash,
                        git_model_checkpoint_t *checkpoint);

/* Network operations (for distributed) */
int  git_clone(const char *url, const char *path, git_repo_t *repo);
int  git_fetch(git_repo_t *repo, const char *remote);
int  git_push(git_repo_t *repo, const char *remote, const char *ref);
int  git_pull(git_repo_t *repo, const char *remote);

#endif /* TENSOROS_GIT_H */
