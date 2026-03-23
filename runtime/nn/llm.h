/* =============================================================================
 * TensorOS - Real LLM Inference Engine
 *
 * Loads and runs real pre-trained language models from GGUF files.
 * Supports any LLaMA-architecture model: Qwen, Gemma, LLaMA, SmolLM,
 * Mistral, TinyLlama, Phi, etc.
 *
 * Features:
 *   - GGUF model loading from virtio-blk disk
 *   - Q4_0 / Q8_0 / F16 / F32 quantized weight support
 *   - Grouped Query Attention (GQA)
 *   - Rotary Position Embeddings (RoPE)
 *   - KV-cache for efficient autoregressive decoding
 *   - RMSNorm + SwiGLU FFN (LLaMA/GPT architecture)
 *   - BPE tokenizer (from GGUF vocabulary)
 *   - Greedy / temperature-sampled text generation
 *   - Built-in math evaluation benchmark
 *
 * All running bare-metal in ring-0 with SSE2 SIMD acceleration.
 * =============================================================================*/

#ifndef TENSOROS_LLM_H
#define TENSOROS_LLM_H

#include <stdint.h>
#include "runtime/nn/gguf.h"

/* ─── Limits ─── */
/* No compile-time dimension caps: buffers are allocated dynamically from
 * the tensor heap after the model's actual dimensions are known.  The only
 * hard limit is available physical memory. */
#define LLM_MAX_VOCAB     160000   /* logits buffer; overridden at runtime */

/* ─── Hash-table tokenizer ─── */
#define LLM_HASH_BITS     17       /* 131072 slots */
#define LLM_HASH_SIZE     (1 << LLM_HASH_BITS)

/* ─── Layer Weights (pointers into GGUF data) ─── */
typedef struct {
    const void *attn_norm;       /* F32 [dim] - pre-attention RMSNorm/LayerNorm */
    const void *q_weight;        /* Quantized [dim × n_heads*head_dim] */
    const void *k_weight;        /* Quantized [dim × n_kv_heads*head_dim] */
    const void *v_weight;        /* Quantized [dim × n_kv_heads*head_dim] */
    const void *o_weight;        /* Quantized [n_heads*head_dim × dim] */
    const void *ffn_norm;        /* F32 [dim] - pre-FFN RMSNorm/LayerNorm */
    const void *ffn_gate;        /* Quantized [dim × ff_dim] (W1) — NULL for Phi-2 */
    const void *ffn_up;          /* Quantized [dim × ff_dim] (W3) */
    const void *ffn_down;        /* Quantized [ff_dim × dim] (W2) */
    ggml_type_t q_type, k_type, v_type, o_type;
    ggml_type_t gate_type, up_type, down_type;
    ggml_type_t attn_norm_type, ffn_norm_type;
    /* Bias vectors (Phi-2, NULL for most LLaMA-derived architectures) */
    const void *attn_norm_bias;  /* F32 [dim] */
    const void *q_bias;          /* F32 [n_heads*head_dim] */
    const void *k_bias;          /* F32 [n_kv_heads*head_dim] */
    const void *v_bias;          /* F32 [n_kv_heads*head_dim] */
    const void *o_bias;          /* F32 [dim] */
    const void *ffn_norm_bias;   /* F32 [dim] */
    const void *ffn_up_bias;     /* F32 [ff_dim] */
    const void *ffn_down_bias;   /* F32 [dim] */
} llm_layer_t;

/* ─── Vocab Entry ─── */
typedef struct {
    const char *str;    /* Pointer into GGUF data (NOT null-terminated) */
    uint16_t    len;    /* String length */
} llm_vocab_entry_t;

/* ─── Full Model Context ─── */
typedef struct {
    /* Architecture parameters */
    int dim;                /* Model/embedding dimension */
    int n_layers;           /* Number of transformer layers */
    int n_heads;            /* Number of attention heads */
    int n_kv_heads;         /* Number of KV heads (GQA) */
    int head_dim;           /* dim / n_heads */
    int ff_dim;             /* Feed-forward hidden dimension */
    int vocab_size;         /* Vocabulary size */
    int max_seq;            /* Maximum sequence length */
    float rope_base;        /* RoPE frequency base */
    int   rope_dim;         /* Partial RoPE dimension (0 = full head_dim) */
    int   rope_orig_ctx;    /* Original context length (for longrope scaling) */
    const float *rope_factors_short; /* [head_dim/2] longrope short factors (NULL if none) */
    const float *rope_factors_long;  /* [head_dim/2] longrope long factors (NULL if none) */
    float rms_eps;          /* RMSNorm epsilon (e.g. 1e-5) */
    int   use_layernorm;    /* 1 = LayerNorm (Phi-2), 0 = RMSNorm */
    int   use_gelu;         /* 1 = GELU FFN (Phi-2), 0 = SwiGLU */
    char arch[64];          /* Architecture name (e.g. "qwen2", "llama", "phi3") */
    char name[128];         /* Model name */

    /* Per-layer weight pointers (dynamically allocated) */
    llm_layer_t *layers;
    int          n_layers_alloc;

    /* Global weights */
    const void *token_embd;     /* Embedding matrix */
    ggml_type_t token_embd_type;
    const void *output_norm;    /* Final RMSNorm / LayerNorm */
    ggml_type_t output_norm_type;
    const void *output_norm_bias; /* LayerNorm bias (NULL for RMSNorm) */
    const void *output_weight;  /* LM head (or NULL if tied to token_embd) */
    ggml_type_t output_type;

    /* Tokenizer */
    int n_vocab;                /* actual vocab count from tokenizer */
    int bos_id;                 /* Beginning of sequence token */
    int eos_id;                 /* End of sequence token */
    int byte_tokens[256];       /* Byte fallback token IDs */

    /* GGUF context (for metadata access) */
    gguf_ctx_t *gguf;

    /* KV cache (allocated at runtime) */
    float *k_cache;    /* [n_layers * max_seq * n_kv_heads * head_dim] */
    float *v_cache;
    int    cache_len;  /* Current cached sequence length */

    /* Vocab lookup (allocated after GGUF data) */
    llm_vocab_entry_t *vocab;   /* [vocab_size] */
    float             *vocab_scores; /* [vocab_size] merge scores */

    /* Hash table for O(1) token lookup by string */
    int32_t           *vocab_ht_slot; /* [LLM_HASH_SIZE] → vocab index or -1 */

    /* Tokenizer type: 0 = GPT-2 BPE, 1 = SentencePiece (LLaMA/Phi-3) */
    int use_spm;

    /* Model data buffer */
    void    *data_buf;          /* Loaded GGUF file */
    uint64_t data_size;         /* Size of loaded data */
} llm_model_t;

/* ─── API ─── */

/**
 * Boot-time loader: detect disk, load model into RAM. Fast — no eval.
 */
void llm_boot_load(void);

/**
 * Full eval: load model + run benchmark + math eval (legacy, slow).
 */
void llm_run_eval(void);

/**
 * Run benchmark + math eval on already-loaded model.
 */
void llm_run_full_eval(void);

/**
 * Interactive prompt: tokenize user text, generate response.
 * Returns number of generated tokens, or -1 on error.
 */
int llm_prompt(const char *user_text, char *output, int max_output);

/**
 * Like llm_prompt, but with an explicit max-token cap (for quick smoke tests).
 */
int llm_prompt_n(const char *user_text, char *output, int max_output, int max_tokens);

/**
 * Check if an LLM model is currently loaded.
 */
int llm_is_loaded(void);

/**
 * Return the name of the currently loaded model (or "(none)").
 */
const char *llm_model_name(void);

/**
 * Return total parameter count of the loaded model, or 0 if none loaded.
 */
uint64_t llm_param_count(void);

/**
 * Reset the KV cache (for starting a new conversation).
 */
void llm_reset_cache(void);

#endif /* TENSOROS_LLM_H */
