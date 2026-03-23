/* =============================================================================
 * TensorOS - Flash Attention Kernel Interface
 * Tiled attention with O(N) memory instead of O(N²)
 * Based on the FlashAttention algorithm (Dao et al., 2022)
 * =============================================================================*/

#ifndef TENSOROS_FLASH_ATTN_H
#define TENSOROS_FLASH_ATTN_H

#include "kernel/core/kernel.h"

/* Flash attention configuration */
typedef struct {
    int     head_dim;       /* Dimension per head (e.g., 64, 128) */
    int     n_heads;        /* Number of attention heads */
    int     n_kv_heads;     /* Number of KV heads (for GQA) */
    int     seq_len;        /* Current sequence length */
    int     block_size_q;   /* Tile size for Q (Br) */
    int     block_size_kv;  /* Tile size for K/V (Bc) */
    float   scale;          /* 1/sqrt(d_k) */
    bool    causal;         /* Causal (autoregressive) mask */
    bool    use_alibi;      /* Use ALiBi positional encoding */
} flash_attn_config_t;

/* Flash attention statistics */
typedef struct {
    uint64_t total_flops;
    uint64_t peak_memory_bytes;
    uint64_t tiling_passes;
    uint64_t time_us;
} flash_attn_stats_t;

/* Core API */
int flash_attn_init(void);

int flash_attn_forward(
    float *output,                  /* [seq_len, n_heads, head_dim] */
    const float *Q,                 /* [seq_len, n_heads, head_dim] */
    const float *K,                 /* [seq_len, n_kv_heads, head_dim] */
    const float *V,                 /* [seq_len, n_kv_heads, head_dim] */
    const flash_attn_config_t *cfg,
    flash_attn_stats_t *stats       /* optional */
);

/* Single-query variant for autoregressive decoding */
int flash_attn_decode_step(
    float *output,                  /* [1, n_heads, head_dim] */
    const float *q,                 /* [1, n_heads, head_dim] */
    const float *k_cache,           /* [cache_len, n_kv_heads, head_dim] */
    const float *v_cache,           /* [cache_len, n_kv_heads, head_dim] */
    int cache_len,
    const flash_attn_config_t *cfg
);

void flash_attn_selftest(void);

#endif /* TENSOROS_FLASH_ATTN_H */
