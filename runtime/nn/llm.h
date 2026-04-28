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
#define LLM_HASH_BITS     20       /* 1048576 slots — must exceed vocab_size */
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

    /* Gemma4: per-head Q/K normalization */
    const void *q_norm;          /* F32 [head_dim_layer] - per-head Q norm */
    const void *k_norm;          /* F32 [head_dim_layer] - per-head K norm */
    /* Gemma4: post-attention and post-FFW norms */
    const void *post_attn_norm;  /* F32 [dim] */
    const void *post_ffw_norm;   /* F32 [dim] */
    /* Gemma4 ISWA: per-layer injection */
    const void *iswa_inp_gate;   /* blk.N.inp_gate.weight [1536, 256] Q4_0 */
    ggml_type_t iswa_inp_gate_type;
    const void *iswa_proj;       /* blk.N.proj.weight [256, 1536] Q4_0 */
    ggml_type_t iswa_proj_type;
    const void *iswa_post_norm;  /* blk.N.post_norm.weight [dim] F32 */
    const void *iswa_out_scale;  /* blk.N.layer_output_scale.weight [1] F32 */
    /* Derived per-layer dimension */
    int head_dim_layer;          /* head_dim for this layer (SWA vs full) */
    int kv_reuse_layer;          /* -1=own KV, >=0=reuse this layer's KV cache */
    int ff_dim_layer;            /* FFN hidden dim for this layer (may differ) */
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
    int   use_geglu;        /* 1 = GeGLU FFN (Gemma): GELU(gate) * up */
    float embed_scale;      /* Embedding scaling factor (sqrt(dim) for Gemma, 1.0 otherwise) */
    float logit_softcap;    /* Final logit softcapping (0 = disabled, 30.0 for Gemma 4) */
    char arch[64];          /* Architecture name (e.g. "qwen2", "llama", "phi3") */
    char name[128];         /* Model name */

    /* Gemma4 SWA/ISWA parameters */
    int   head_dim_swa;     /* SWA head dim (256), 0=not Gemma4 */
    float rope_base_swa;    /* RoPE base for SWA layers (10000) */
    int   n_layer_kv_start; /* First n layers have own KV cache, rest reuse */
    int   is_gemma4;        /* 1 = Gemma4 architecture */

    /* Gemma4 ISWA global weights */
    const void *iswa_tok_embd;        /* per_layer_token_embd.weight */
    ggml_type_t iswa_tok_embd_type;
    const void *iswa_model_proj;      /* per_layer_model_proj.weight */
    ggml_type_t iswa_model_proj_type;
    const void *iswa_proj_norm;       /* per_layer_proj_norm.weight */
    int         iswa_n_embd;          /* n_embd_per_layer (256) */
    const float *rope_freqs;          /* rope_freqs.weight [head_dim/2] for full-attn layers */

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
 * Hosted-mode loader: load model from a pre-mapped memory buffer.
 * The buffer must contain a valid GGUF file. Returns 0 on success.
 */
int llm_load_from_buffer(void *data, uint64_t size);

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

/** Return model architecture name (e.g. "gemma4", "llama", "phi3"). */
const char *llm_model_arch(void);

/** Return number of transformer layers. */
int llm_model_layers(void);

/** Return hidden dimension. */
int llm_model_dim(void);

/** Return vocabulary size. */
int llm_model_vocab(void);

/**
 * Return total parameter count of the loaded model, or 0 if none loaded.
 */
uint64_t llm_param_count(void);

/* Execution backend selection (CPU today, CUDA/MLIR staged). */
typedef enum {
    LLM_BACKEND_CPU = 0,
    LLM_BACKEND_CUDA = 1,
    LLM_BACKEND_MLIR = 2
} llm_backend_t;

/* Select backend. Returns 0 on success, negative if unsupported/invalid. */
int llm_set_backend(llm_backend_t backend);

/* Get active backend. */
llm_backend_t llm_get_backend(void);

/* Human-readable backend name. */
const char *llm_backend_name(void);

/**
 * Reset the KV cache (for starting a new conversation).
 */
void llm_reset_cache(void);

/* ── Streaming token callback ──────────────────────────────── */

/**
 * Callback type: called after each generated token, with the decoded text
 * fragment and its byte length. Used to stream output character-by-character.
 */
typedef void (*llm_token_cb_t)(const char *text, int len, void *userdata);

/**
 * Register a streaming token callback.  Pass NULL to disable.
 * The callback is called from the inference thread — do not block in it.
 */
void llm_set_stream_cb(llm_token_cb_t cb, void *userdata);

/**
 * Callback type for token-native tool/code execution.
 * Implementations consume token IDs directly and emit token IDs directly,
 * avoiding text serialization at machine-to-machine boundaries.
 */
typedef int (*llm_token_exec_fn)(const int *input_tokens, int n_input_tokens,
                                 int *output_tokens, int max_output_tokens,
                                 void *userdata);

/* ── Token-native API ───────────────────────────────────────── */

/** Tokenize text into token IDs using the loaded model tokenizer. */
int llm_tokenize_text(const char *text, int *tokens, int max_tokens);

/** Decode token IDs into text in one final pass. */
int llm_decode_tokens(const int *tokens, int n_tokens, char *output, int max_output);

/**
 * Generate visible output tokens directly from a prompt token buffer.
 * Internal thinking/control tokens remain inside the model loop unless the
 * caller explicitly asks for them through a lower-level integration.
 */
int llm_generate_tokens(const int *prompt_tokens, int n_prompt,
                        int *output_tokens, int max_output_tokens,
                        int max_gen, float temperature, int continue_cache);

/** Prompt the model and return visible output as token IDs. */
int llm_prompt_tokens(const char *user_text, int *output_tokens,
                      int max_output_tokens, int max_gen, float temperature);

/** Multi-turn chat variant that keeps the response in token space. */
int llm_chat_turn_tokens(const char *user_text, int *output_tokens,
                         int max_output_tokens, int max_gen, float temperature);

/**
 * Evaluate per-token log-probabilities for an input text sequence.
 * Runs the full forward pass for each token in the sequence, collecting
 * log P(token[i+1] | token[0..i]) into out_logprobs[i].
 *
 * @param text         Input text to evaluate.
 * @param out_logprobs Caller-allocated buffer, receives n_tokens-1 logprob values.
 * @param max_logprobs Capacity of out_logprobs.
 * @return Number of logprob values written (n_tokens-1), or negative on error.
 */
int llm_eval_sequence_logprobs(const char *text, float *out_logprobs, int max_logprobs);

/** Execute a token-native tool/program callback without a text boundary. */
int llm_execute_token_program(const int *input_tokens, int n_input_tokens,
                              llm_token_exec_fn executor, void *userdata,
                              int *output_tokens, int max_output_tokens);

/**
 * Token-native code execution loop contract.
 *
 * 1) Feed prompt_tokens to the model and generate up to max_gen output tokens.
 * 2) Pass output tokens to the executor callback.
 * 3) If the executor returns >0 output tokens, reinject them as a new prompt
 *    and repeat (up to max_rounds).  Token IDs stay in token space the
 *    entire time — no text serialization at any internal boundary.
 * 4) Final model output is written to output_tokens.
 *
 * Returns total tokens written to output_tokens, negative on error.
 */
int llm_execute_token_loop(const int *prompt_tokens, int n_prompt,
                           llm_token_exec_fn executor, void *userdata,
                           int *output_tokens, int max_output_tokens,
                           int max_gen, float temperature,
                           int max_rounds);

/**
 * Validate whether token IDs form syntactically valid JSON without
 * detokenizing the full sequence into an intermediate string.
 * Returns 1 for valid JSON, 0 for invalid, negative on runtime error.
 */
int llm_validate_json_tokens(const int *tokens, int n_tokens);

/**
 * Validate markdown fenced-code structure (```...```) directly on token IDs.
 * Returns 1 for valid fenced block, 0 for invalid, negative on runtime error.
 */
int llm_validate_code_fence_tokens(const int *tokens, int n_tokens);

/**
 * Validate basic XML well-formedness directly on token IDs.
 * Returns 1 for valid XML, 0 for invalid, negative on runtime error.
 */
int llm_validate_xml_tokens(const int *tokens, int n_tokens);

/**
 * Validate schema-lite key:value structure directly on token IDs.
 *
 * Accepts lines of the form:  KEY SEPARATOR VALUE
 *   - keys:       alphanumeric or underscore/dash/dot
 *   - separators: ':', '=', or '::'
 *   - values:     non-empty remainder until newline
 *   - blank lines and '#'-prefixed comment lines are allowed
 *
 * Returns 1 for valid, 0 for invalid, negative on runtime error.
 */
int llm_validate_key_value_tokens(const int *tokens, int n_tokens);

/**
 * Snapshot the current KV cache prefix under a token-prefix key.
 * Returns captured prefix length on success, negative on error.
 */
int llm_kv_snapshot_prefix(const int *prefix_tokens, int n_prefix_tokens);

/**
 * Restore a previously snapshotted KV cache by exact token-prefix key match.
 * Returns restored length, negative if no match or on error.
 */
int llm_kv_restore_prefix(const int *prefix_tokens, int n_prefix_tokens);

/**
 * Restore a KV snapshot AND run the last context token to prime llm_logits.
 * After this call: cache_len == n_ctx, llm_logits_pos == n_ctx - 1.
 * Enables the zero-cost skip-all path in the speculative verifier.
 * Returns n_ctx on success, -1 if no snapshot exists.
 */
int llm_kv_restore_and_prime(const int *ctx, int n_ctx);

/**
 * Cheaply prime llm_logits for n_ctx - 1 when the KV cache is exactly one
 * token behind the requested context.
 * Returns 0 when logits are primed for position n_ctx - 1, -1 otherwise.
 */
int llm_prime_logits_fast(const int *ctx, int n_ctx);

/**
 * Return a read-only pointer to the current logit buffer if the logits are
 * primed for position n_ctx - 1 (i.e. llm_logits_pos == n_ctx - 1).
 * Returns NULL when stale.  Used by geodesic_step_fast to bypass the
 * PCA-space nearest-neighbour search with actual transformer logits.
 */
const float *llm_get_logits_primed(int n_ctx);

/**
 * Return the cached greedy argmax token for the last primed context position.
 * This is O(1) — the argmax is computed once during llm_prime_logits_fast and
 * cached.  Returns -1 if logits are not primed for n_ctx-1.
 * Use this instead of a manual O(vocab) scan when you only need the top-1 token.
 */
int llm_get_primed_greedy_token(int n_ctx);

/**
 * Get the last context-conditioned hidden state (final-layer RMSNorm output).
 * This is the model's true manifold position for the current token in context.
 * Returns NULL if no forward pass has been run yet this call.
 * Buffer is dim floats (float32, CPU-side).
 */
const float *llm_get_last_hidden_state(void);

/**
 * Exact greedy rollout from the current context using the transformer's own
 * logits. This is an upper-bound drafting path: every emitted token is the
 * model's exact greedy continuation, so no verifier pass is required.
 *
 * out_tokens[0..n_out-1] receives up to n_steps tokens.
 * out_margin[0..n_out-1], when non-NULL, receives the top-1 minus top-2 logit
 * margin for each emitted token.
 *
 * Returns number of emitted tokens (0..n_steps), or -1 on error.
 */
int llm_rollout_exact_greedy(const int *context_tokens, int n_context,
                             int n_steps,
                             int *out_tokens,
                             float *out_margin,
                             int *n_out);

/**
 * Reset a persistent agent token context slot.
 * Returns 0 on success, negative on error.
 */
int llm_agent_ctx_reset(int ctx_id);

/**
 * Append token IDs to a persistent agent context slot.
 * Returns new context length on success, negative on error.
 */
int llm_agent_ctx_append_tokens(int ctx_id, const int *tokens, int n_tokens);

/**
 * Generate from a persistent agent context in token space only.
 * Returns number of generated tokens, negative on error.
 */
int llm_agent_ctx_generate(int ctx_id, int *output_tokens, int max_output_tokens,
                           int max_gen, float temperature);

/**
 * Set a token-native RAG prefix as embedding vectors [n_prefix x dim].
 * Returns number of stored prefix vectors, negative on error.
 */
int llm_rag_set_prefix_embeddings(const float *embeddings, int n_prefix, int dim);

/** Clear the current token-native RAG prefix embeddings. */
void llm_rag_clear_prefix_embeddings(void);

/**
 * Generic vector-prefix ingress (e.g. multimodal encoders).
 */
int llm_set_vector_prefix(const float *vectors, int n_vectors, int dim);

/** Clear generic vector-prefix state. */
void llm_clear_vector_prefix(void);

/**
 * Verify a draft token sequence directly in token space against the current model.
 * Returns number of accepted draft tokens.
 */
int llm_speculative_verify_tokens(const int *context_tokens, int n_context,
                                  const int *draft_tokens, int n_draft);

/**
 * Like llm_speculative_verify_tokens but also outputs the transformer's greedy
 * correction token at the first rejection point (or the bonus next token when
 * all drafts are accepted).  correction_tok_out may be NULL.
 * Returns number of accepted draft tokens (0..n_draft), or -1 on error.
 */
int llm_speculative_verify_with_correction(const int *context_tokens, int n_context,
                                           const int *draft_tokens,  int n_draft,
                                           int *correction_tok_out);

/**
 * Like llm_speculative_verify_with_correction but uses relaxed top-K acceptance:
 * a draft token is accepted if its logit is within `topk_margin` nats of the
 * greedy argmax logit (i.e., its probability is at least exp(-topk_margin) of
 * the top token's probability).  topk_margin=2.3 ≈ top-10%, topk_margin=4.6 ≈ top-1%.
 * When a draft token is accepted this way, the KV cache still advances through
 * that position and `correction_tok_out` reflects the true greedy continuation
 * AFTER the last accepted position.  This preserves output quality while raising
 * the acceptance rate substantially for imperfect draft models.
 * Returns number of accepted draft tokens (0..n_draft), or -1 on error.
 */
int llm_speculative_verify_topk(const int *context_tokens, int n_context,
                                const int *draft_tokens,   int n_draft,
                                float topk_margin,
                                int *correction_tok_out);

/**
 * Return the top-1 token from the most-recent logits (set by the preceding
 * llm_speculative_verify_* / llm_prime_logits_fast call), excluding any token
 * id in exclude[0..n_exclude).  Returns -1 if no logits are currently valid.
 * Performs no forward pass.
 */
int llm_topk_excluding(const int *exclude, int n_exclude);

/**
 * OD-SWARM speculative verify: like llm_speculative_verify_topk (topk_margin
 * acceptance) but with an additional SWARM acceptance path: if the primary
 * draft at slot i fails the margin test, the verifier's greedy token is
 * checked against swarm_tokens[i*swarm_k .. i*swarm_k+swarm_k-1].  When
 * a SWARM hit occurs, the KV-cache advances with the verifier's greedy token
 * (preserving correctness).  swarm_k=0 degrades to plain verify_topk.
 * Returns accepted count (0..n_draft), or -1 on error.
 */
int llm_speculative_verify_swarm(const int *context_tokens, int n_context,
                                 const int *draft_tokens,   int n_draft,
                                 float      topk_margin,
                                 const int *swarm_tokens,   int swarm_k,
                                 int *correction_tok_out);

/**
 * Arm online PCA basis update on speculative-decode rejections.
 * Pass NULL to disarm.  The context must stay valid for the lifetime of
 * inference (typically a static g_onb_ctx in main).
 * Requires #include "runtime/nn/online_basis.h" in the caller.
 */
void llm_set_online_basis_ctx(void *ctx);

/** Maximum number of rollout steps retained in token-native trace buffer. */
#define LLM_ROLLOUT_MAX_STEPS 4096

/**
 * Canonical token-native RL rollout step.
 */
#ifndef LLM_ROLLOUT_STEP_FWDDECL
#define LLM_ROLLOUT_STEP_FWDDECL
typedef struct {
    int token_id;
    float logprob;
    float value;
    float reward;
    int done;
} llm_rollout_step_t;
#endif

/** Reset rollout trace state. Returns 0 on success. */
int llm_rollout_reset(void);

/** Append one rollout step. Returns new step count, negative on error. */
int llm_rollout_append_step(int token_id, float logprob, float value);

/** Set reward/done for an existing step index. Returns 0 on success. */
int llm_rollout_set_step_reward(int step_idx, float reward, int done);

/** Copy rollout steps into caller buffer. Returns copied step count. */
int llm_rollout_get_steps(llm_rollout_step_t *out_steps, int max_steps);

/**
 * Compute discounted returns per step into out_returns.
 * Returns number of computed returns, negative on error.
 */
int llm_rollout_compute_returns(float gamma, float *out_returns, int max_returns);

/* ── Multi-turn chat API ────────────────────────────────────── */

/**
 * Generate one assistant reply for the given user message, maintaining
 * KV-cache context across multiple calls.
 *
 * @param user_text   The user's message (plain text, no template needed).
 * @param output      Buffer to receive the full reply (also streamed if cb set).
 * @param max_output  Size of @output in bytes.
 * @param max_tokens  Maximum new tokens to generate (0 → default 256).
 * @param temperature Sampling temperature (e.g. 0.7).
 * @return Number of tokens generated, or negative on error.
 *
 * Call llm_chat_reset() to start a new conversation.
 */
int  llm_chat_turn(const char *user_text, char *output, int max_output,
                   int max_tokens, float temperature);

/** Reset the KV cache, starting a fresh conversation. */
void llm_chat_reset(void);

/* ── Thinking Mode ──────────────────────────────────────────── */

/**
 * Enable or disable thinking.  When enabled (default=1), the model may
 * emit <|think|> tokens; the runtime detects them, runs the thinking
 * phase purely in token space (no detokenization), and resumes normal
 * output after the second <|think|> marker.
 *
 * Set enable=0 to suppress thinking (strip <|think|> blocks).
 * Set enable=2 to force-inject <|think|> into prompts (extended thinking).
 */
void llm_set_thinking(int enable);

/**
 * If set, thinking tokens are detokenized and included in the output
 * (wrapped in <think>...</think> tags).  Default: 0 (thinking hidden).
 */
void llm_set_show_thinking(int show);

/**
 * After generation, returns the number of thinking tokens consumed.
 */
int  llm_thinking_tokens(void);

/** Set sampling controls for top-k and top-p nucleus filtering. */
void llm_set_sampling_params(int top_k, float top_p);

/**
 * Override the GPU KV-cache context window before loading a model.
 * Default (0) caps the context at 2048 tokens to keep VRAM usage ~1 GB.
 * Use llm_set_max_ctx(8192) to restore the full Gemma4 context window.
 */
void llm_set_max_ctx(int n);

/**
 * Enable/disable AttnRes-inspired depth stabilization.
 * strength is clamped to [0,1], where higher values apply stronger damping.
 */
void llm_set_attention_residuals(int enable, float strength);

/**
 * Enable/disable content-dependent depth-wise residual attention over prior layers.
 * strength is clamped to [0,1], window to [2,64].
 */
void llm_set_depth_residual_attention(int enable, float strength, int window);

/** Return the number of KV-cache positions currently occupied. */
int  llm_chat_context_tokens(void);

/** Return the total context-window capacity of the loaded model. */
int  llm_chat_context_max(void);

/** Last-generation performance stats (updated after each llm_generate). */
float llm_last_prefill_ms(void);
float llm_last_tok_per_sec(void);
int   llm_last_vram_usage_mb(void);
/** KV prefix-cache hit rate since model load (0.0 = no hits, 1.0 = all hits). */
float llm_kv_cache_hit_rate(void);
/** Shannon entropy of the last-computed softmax distribution (from llm_prime_logits_fast).
 *  High value (~8-10) = high uncertainty (retrieval/reasoning phase);
 *  Low value (<4) = peaked distribution (generation phase). */
float llm_last_logit_entropy(void);

/* ── Test-facing API (exposes internal tokenizer for unit tests) ─── */

/** Tokenize text into token IDs. Returns number of tokens. */
int llm_test_tokenize(const char *text, int text_len, int *tokens, int max_tokens);

/** Decode a token ID into text. Returns byte length. */
int llm_test_decode_token(int token_id, char *buf, int max_len);

/* ── Model Accessor API (for geometric / axiomatic analysis) ─── */

/**
 * Access the loaded model structure for read-only geometric analysis.
 * Returns NULL if no model is loaded.
 */
const llm_model_t *llm_get_model(void);

/**
 * Get the dequantized embedding vector for a token ID.
 * Writes dim floats to out. Returns 0 on success.
 */
int llm_get_embedding_vec(int token_id, float *out, int dim);

/**
 * Get a pointer to a K vector in the live CPU KV cache.
 * layer: transformer layer index (0-based)
 * head:  KV head index (0-based, < n_kv_heads)
 * pos:   sequence position (0-based, < cache_len)
 * Returns pointer to head_dim floats, or NULL if invalid / not available.
 * The pointer is valid until the next forward pass.
 */
const float *llm_get_kv_k(int layer, int head, int pos);

/**
 * Layer-granular GPU weight management for curvature-guided offload.
 *
 * llm_gpu_layer_ensure: upload layer l's weights to GPU if not already there.
 *   Returns 0 on success, -1 if GPU not available or layer out of range.
 *
 * llm_gpu_layer_release: free GPU copy of layer l (keeps CPU copy).
 *   Returns 0 on success, -1 on error.
 *
 * llm_gpu_vram_available: return free VRAM in bytes (0 if no GPU).
 */
int      llm_gpu_layer_ensure(int layer);
int      llm_gpu_layer_release(int layer);
uint64_t llm_gpu_vram_available(void);
void     llm_gpu_upload_compressed_weights(void);
void     llm_gpu_set_compress_mode(int enable);
void     llm_gpu_set_attn_compress_mode(int enable);
void     llm_gpu_upload_ffn_fallback(void);
void     llm_set_temperature(float t);

/* ── Tensor Bridge API (hidden-state injection / daisy-chaining) ─── */
#include "runtime/nn/tensor_bridge.h"
tensor_bridge_t *llm_get_bridge(void);

/** One-step logits prime: runs a single forward pass when cache_len == n_ctx-1.
 *  Returns 0 if logits are now primed for n_ctx-1, -1 if the fast path could
 *  not be used (caller should fall back to llm_kv_restore_and_prime). */
int llm_prime_logits_fast(const int *ctx, int n_ctx);

/**
 * Geodesic Speculative Decode — the HyperTensor turbo path.
 *
 * Generates up to max_gen tokens into output_tokens using geodesic rollout
 * as a speculative draft, verified by the transformer at batch cost:
 *
 *   loop:
 *     1. axiom_beta_geodesic_rollout(ctx, n_ctx, n_draft) → draft[0..k-1]
 *     2. llm_speculative_verify_with_correction → accept n_accepted tokens
 *     3. axiom_beta_grc_feedback on first rejection (manifold learning)
 *     4. Append accepted tokens + correction, advance context
 *
 * Expected speedup: 2.5–4× over autoregressive when GRC library is warm.
 * Falls back to standard autoregressive if geodesic cache is not ready.
 *
 * Returns the number of tokens written to output_tokens, or -1 on error.
 */
int llm_generate_geodesic_speculative(const int *prompt_tokens, int n_prompt,
                                      int *output_tokens, int max_output_tokens,
                                      int max_gen, float temperature,
                                      int n_draft);

/* ── Session API — per-session inference for parallel serving ─────────────
 *
 * Each session is an independent context backed by a KV snapshot.
 * The parallel api_server worker steps N sessions round-robin.
 *
 *   sid = llm_session_create(prompt_tokens, n_prompt, max_gen, temp);
 *   while ((tok = llm_session_step(sid)) >= 0) { emit(tok); }
 *   llm_session_destroy(sid);
 */
#define LLM_SESSION_DONE  (-10)
#define LLM_SESSION_ERR   (-11)
#define LLM_SESSION_MAX     8

int  llm_session_create(const int *prompt_tokens, int n_prompt,
                        int max_gen, float temperature);
int  llm_session_step(int sid);
void llm_session_destroy(int sid);
int  llm_session_is_done(int sid);
int  llm_session_get_output(int sid, int *token_buf, int max_tokens);

#endif /* TENSOROS_LLM_H */
