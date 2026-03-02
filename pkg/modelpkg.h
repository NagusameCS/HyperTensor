/* =============================================================================
 * TensorOS - Model Package Manager
 *
 * Like apt/npm but for AI models. Manages:
 * - Model discovery and installation from registries
 * - Dependency resolution (tokenizers, configs, base models)
 * - Version management with git-backed storage
 * - Model verification (checksums, signatures)
 * - Automatic quantization/optimization for target hardware
 * =============================================================================*/

#ifndef TENSOROS_MODELPKG_H
#define TENSOROS_MODELPKG_H

#include "kernel/core/kernel.h"

/* =============================================================================
 * Model Package Manifest
 * =============================================================================*/

typedef struct {
    char        name[64];          /* e.g., "llama-3-8b" */
    char        version[16];       /* e.g., "1.0.0" */
    char        author[64];
    char        license[32];       /* e.g., "MIT", "Apache-2.0" */
    char        description[256];
    char        architecture[64];  /* e.g., "transformer-decoder" */

    /* Model specifications */
    uint64_t    param_count;
    tensor_dtype_t dtype;
    uint64_t    context_length;
    uint64_t    vocab_size;
    uint64_t    hidden_dim;
    uint32_t    num_layers;
    uint32_t    num_heads;

    /* File information */
    uint64_t    total_size;        /* Total download size */
    uint32_t    file_count;
    char        format[16];        /* "safetensors", "gguf", etc. */
    char        checksum[65];      /* SHA-256 hex */

    /* Dependencies */
    uint32_t    dep_count;
    char        dependencies[8][64]; /* Required models/tokenizers */

    /* Hardware requirements */
    uint64_t    min_ram_bytes;
    uint64_t    min_vram_bytes;
    uint32_t    min_compute_capability;

    /* Registry info */
    char        registry_url[256];
    char        download_url[256];
} model_manifest_t;

/* =============================================================================
 * Package Registry
 * =============================================================================*/

#define MODELPKG_MAX_INSTALLED  128
#define MODELPKG_MAX_REGISTRIES 8

typedef struct {
    char        name[64];
    char        url[256];
    bool        enabled;
    uint64_t    last_sync;
} model_registry_t;

typedef enum {
    PKG_STATUS_AVAILABLE    = 0,  /* In registry, not installed */
    PKG_STATUS_DOWNLOADING  = 1,
    PKG_STATUS_INSTALLING   = 2,
    PKG_STATUS_INSTALLED    = 3,
    PKG_STATUS_OPTIMIZING   = 4,  /* Being quantized/optimized */
    PKG_STATUS_ERROR        = 5,
} pkg_status_t;

typedef struct {
    model_manifest_t manifest;
    pkg_status_t     status;
    char             install_path[256];
    uint64_t         install_time;
    bool             optimized;         /* Has been quantized for local hw */
    tensor_dtype_t   optimized_dtype;   /* Quantized dtype */
} installed_model_t;

/* =============================================================================
 * Package Manager State
 * =============================================================================*/

typedef struct {
    model_registry_t    registries[MODELPKG_MAX_REGISTRIES];
    uint32_t            registry_count;
    installed_model_t   installed[MODELPKG_MAX_INSTALLED];
    uint32_t            installed_count;
} modelpkg_state_t;

extern modelpkg_state_t g_modelpkg;

/* =============================================================================
 * Package Manager API
 * =============================================================================*/

/* Initialization */
int  modelpkg_init(void);

/* Registry management */
int  modelpkg_add_registry(const char *name, const char *url);
int  modelpkg_remove_registry(const char *name);
int  modelpkg_sync_registry(const char *name);  /* Fetch latest index */

/* Search and discovery */
int  modelpkg_search(const char *query, model_manifest_t *results,
                      uint32_t max, uint32_t *count);
int  modelpkg_info(const char *name, model_manifest_t *manifest);

/* Installation */
int  modelpkg_install(const char *name, const char *version);
int  modelpkg_uninstall(const char *name);
int  modelpkg_update(const char *name);
int  modelpkg_update_all(void);

/* Listing */
int  modelpkg_list_installed(installed_model_t *models,
                              uint32_t max, uint32_t *count);

/* Optimization */
int  modelpkg_optimize(const char *name, tensor_dtype_t target_dtype);
int  modelpkg_auto_optimize(const char *name); /* Auto-detect best dtype */

/* Verification */
int  modelpkg_verify(const char *name);  /* Check integrity */

/* Loading (returns path to model files) */
const char *modelpkg_get_path(const char *name);
int  modelpkg_load_to_cache(const char *name);  /* Pre-load into model cache */

#endif /* TENSOROS_MODELPKG_H */
