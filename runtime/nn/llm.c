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
 * TensorOS - Real LLM Inference Engine Implementation
 *
 * Complete pipeline: disk → GGUF parse → tensor map → tokenize → forward →
 * generate text → evaluate mathematical reasoning.
 *
 * Supported architectures (all LLaMA-family):
 *   - Qwen2 / Qwen2.5  (qwen2)
 *   - LLaMA / LLaMA2/3  (llama)
 *   - Gemma / Gemma2    (gemma)
 *   - SmolLM / SmolLM2  (llama)
 *   - TinyLlama         (llama)
 *   - Mistral           (llama)
 *   - Phi-3             (phi3)
 *
 * Quantization formats: Q4_0, Q8_0, F16, F32
 * All running bare-metal with SSE2 SIMD, zero OS overhead.
 * =============================================================================*/

#include "runtime/nn/llm.h"
#include "runtime/nn/backend.h"
#include "runtime/nn/model_meta.h"
#include "runtime/nn/tensor_bridge.h"
#include "runtime/nn/blt.h"
#include "runtime/nn/flash_attn.h"
#include "runtime/nn/axiom_beta.h"
#include "runtime/nn/axiom_exploit.h"
#include "runtime/nn/online_basis.h"
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/core/perf.h"
#include "runtime/nn/gguf.h"
#include "runtime/jit/x86_jit.h"
#include "kernel/security/crypto.h"
#ifdef GEODESSICAL_HOSTED
#include <stdlib.h>
#include <string.h>
#include <math.h>
#endif
#ifndef __aarch64__
#include "kernel/drivers/blk/virtio_blk.h"
#include "kernel/core/smp.h"
#else
#include <arm_neon.h>
#endif

/* ─────────────────────────────────────────────────────────────────────────── */
/*  SSE2 SIMD type                                                             */
/* ─────────────────────────────────────────────────────────────────────────── */
typedef float v4f __attribute__((vector_size(16)));


#ifndef __aarch64__
/* AVX2 8-wide float vector */
typedef float v8f __attribute__((vector_size(32)));
#include "kernel/core/cpu_features.h"
#endif

/* Forward declarations */
static void llm_build_hash_table(const llm_model_t *m);
static uint64_t llm_row_bytes(int in_dim, ggml_type_t type);

/* CUDA fused kernel wrappers (defined in backend_cuda.c) */
#ifdef ENABLE_CUDA
void cuda_fused_qk_norm_rope(float *Q, float *K,
    const float *q_norm_w, const float *k_norm_w,
    int n_heads, int n_kv_heads, int head_dim,
    int pos, float rope_base, const float *rope_freqs,
    float eps, int rope_dim);
void cuda_v_norm(float *V, int n_kv_heads, int head_dim, float eps);

/* Batched prefill: replaces per-token attention loop */

void cuda_batch_fused_qk_norm_rope(float *Q, float *K,

    const float *q_norm_w, const float *k_norm_w,

    int n_heads, int n_kv_heads, int head_dim,

    int n, int start_pos, float rope_base, const float *rope_freqs,

    float eps, int rope_dim);

void cuda_batch_v_norm(float *V, int n_kv_heads, int head_dim, int n, float eps);

void cuda_batch_kv_update(float *K_cache, float *V_cache,

    const float *K_new, const float *V_new,

    int n_kv_heads, int head_dim, int n, int start_pos, int max_seq);

void cuda_prefill_attn_batched(float *O, const float *Q,

    const float *K_cache, const float *V_cache,

    int n_heads, int n_kv_heads, int head_dim,

    int n, int start_pos, int max_seq, float scale, float softcap);

int  cuda_have_batch_attn(void);
int  cuda_have_sgemm_batched_f32(void);
void cuda_fused_geglu(float *gate, const float *up, int n);
void cuda_fused_swiglu(float *gate, const float *up, int n);
void cuda_batched_rmsnorm(float *data, const float *w,
                           int n_slices, int slice_dim, float eps);
void cuda_iswa_combine(float *out, const float *tok_embd,
                        const float *proj, float scale, int n);
void cuda_add_rmsnorm(float *norm_out, float *x_inout,
                       const float *residual, const float *norm_w,
                       int dim, float eps);
void cuda_rmsnorm_add(float *x_inout, const float *data,
                       const float *norm_w, int dim, float eps);
void cuda_gelu_mul(float *a, const float *b, int n);
int cuda_gemv_dual_q4_0(float *out_a, float *out_b,
                         const void *W_a, const void *W_b,
                         const float *x, int out_dim, int in_dim);
int cuda_gemv_dual_q8_0(float *out_a, float *out_b,
                         const void *W_a, const void *W_b,
                         const float *x, int out_dim, int in_dim);
/* cuBLAS batched GEMV wrapper (defined in backend_cuda.c) */
void cuda_sgemm_batched_f32(int M, int K,
                              const float * const *d_Aarray,
                              const float * const *d_xarray,
                              float * const       *d_yarray,
                              int batch_count);
int cuda_gemv_triple_q4_0(float *out_q, float *out_k, float *out_v,
                            const void *W_q, const void *W_k, const void *W_v,
                            const float *x,
                            int q_dim, int k_dim, int v_dim, int in_dim);
int cuda_fused_rmsnorm_triple_q4_0(
        float *out_q, float *out_k, float *out_v,
        const void *W_q, const void *W_k, const void *W_v,
        const float *x, const float *norm_w, float eps,
        int q_dim, int k_dim, int v_dim, int in_dim);
int cuda_graph_begin_capture(void);
int cuda_graph_end_capture(void);
int cuda_graph_launch(void);
void cuda_graph_destroy(void);
void cuda_set_decode_pos(int pos, int seq_len);
int cuda_argmax(const float *data, int n);
/* Batch Prefill */
void cuda_prefill_batch_presized(int max_batch, int max_dim);
void cuda_prefill_batch_quant(const float *X, int batch, int in_dim);
int  cuda_prefill_batch_gemv_q4(float *C, const void *W, int out_dim, int in_dim, int batch);
void cuda_batched_rmsnorm_out(float *out, const float *in, const float *w, int n, int d, float eps);

/* Capture-once CUDA Graph state */
int cuda_graph_decode_ready = 0;  /* set after prefill by llm_generate */
int cuda_graph_captured = 0;      /* 1 after graph successfully captured */
int cuda_graph_tried = 0;         /* 1 after first capture attempt */
int cuda_graph_ctx_len = -1;      /* context length at capture time; -1 = unknown */
int gpu_skip_logits_download = 0; /* 1 = caller will use GPU argmax, skip D2H */
#endif

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Debug/Trace Logging Macros                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */
#ifndef LOG_DBG
#define LOG_DBG(tag, fmt, ...)  kprintf("[" tag "] " fmt "\n", ##__VA_ARGS__)
#endif
#ifndef LOG_INFO
#define LOG_INFO(tag, fmt, ...) kprintf("[" tag "] " fmt "\n", ##__VA_ARGS__)
#endif
#ifndef LOG_TRC
#define LOG_TRC(tag, fmt, ...)  do {} while (0)  /* Trace: compiled out */
#endif

static int llm_emit_benchmark_debug(void) {
    static int initialized = 0;
    static int enabled = 0;
    if (!initialized) {
        const char *env = getenv("GD_BENCH_DEBUG");
        enabled = (env && env[0] && env[0] != '0') ? 1 : 0;
        initialized = 1;
    }
    return enabled;
}

static int llm_disable_pt_q4(void) {
    static int initialized = 0;
    static int disabled = 0;
    if (!initialized) {
        const char *env = getenv("AXEX_DISABLE_PT_Q4");
        disabled = (env && env[0] && env[0] != '0') ? 1 : 0;
        initialized = 1;
    }
    return disabled;
}

static int llm_disable_fused_kv(void) {
    static int initialized = 0;
    static int disabled = 0;
    if (!initialized) {
        const char *env = getenv("AXEX_DISABLE_FUSED_KV");
        disabled = (env && env[0] && env[0] != '0') ? 1 : 0;
        initialized = 1;
    }
    return disabled;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Static Allocations                                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

/* GGUF parsing context */
static gguf_ctx_t llm_gguf_ctx;

/* Model descriptor */
static llm_model_t llm_model;

/* Inference serialization: prevent concurrent use of static buffers */
static volatile int llm_inference_active = 0;

/* Online basis update context (NULL = disabled).
 * Set via llm_set_online_basis_ctx() from main after model load. */
static onb_ctx_t *g_onb_ctx = (onb_ctx_t *)0;

void llm_set_online_basis_ctx(void *ctx)
{
    g_onb_ctx = (onb_ctx_t *)ctx;
}

/* Streaming callback: called per-token during generation (NULL = disabled) */
typedef void (*llm_token_cb_t)(const char *text, int len, void *userdata);
static llm_token_cb_t llm_stream_cb   = (llm_token_cb_t)0;
static void          *llm_stream_cb_ud = (void *)0;
static llm_backend_t  llm_backend        = LLM_BACKEND_CPU;
static int            llm_last_vram_mb   = 0;
static float          llm_last_prefill_ms_val = 0.0f;
static float          llm_last_tok_per_sec_val = 0.0f;
static int            g_logged_dual_q8_kv = 0;
static uint64_t       g_kv_snap_lookups = 0;  /* total restore attempts */
static uint64_t       g_kv_snap_hits    = 0;  /* successful restore hits */
static float          g_last_logit_entropy = 0.0f; /* Shannon entropy of last softmax */

/* Tensor bridge for hidden-state injection / daisy-chaining */
static tensor_bridge_t llm_bridge;

void llm_set_stream_cb(llm_token_cb_t cb, void *userdata) {
    llm_stream_cb    = cb;
    llm_stream_cb_ud = userdata;
}

static char *llm_strstr(const char *haystack, const char *needle);
static int llm_decode_token(const llm_model_t *m, int token_id, char *buf, int max_len);

static int llm_decode_tokens_to_text(const llm_model_t *m, const int *tokens,
                                     int n_tokens, char *output_text, int max_text_len)
{
    int out_pos = 0;

    if (!output_text || max_text_len <= 0) return 0;
    output_text[0] = '\0';
    if (!m || !tokens || n_tokens <= 0) return 0;

    for (int i = 0; i < n_tokens && out_pos < max_text_len - 1; i++) {
        char tok_buf[128];
        int tok_len = llm_decode_token(m, tokens[i], tok_buf, sizeof(tok_buf));
        if (tok_len <= 0) continue;
        for (int j = 0; j < tok_len && out_pos < max_text_len - 1; j++)
            output_text[out_pos++] = tok_buf[j];
    }

    output_text[out_pos] = '\0';

    {
        static const char *stop_seqs[] = {
            "<|im_end|>", "<|endoftext|>", "<|end|>",
            "<|eot_id|>", "<|end_of_turn|>", "<end_of_turn>", NULL
        };
        for (int si = 0; stop_seqs[si]; si++) {
            char *found = llm_strstr(output_text, stop_seqs[si]);
            if (found) {
                *found = '\0';
                out_pos = (int)(found - output_text);
                break;
            }
        }
        while (out_pos > 0 && (output_text[out_pos - 1] == ' ' || output_text[out_pos - 1] == '\n'))
            output_text[--out_pos] = '\0';
    }

    return out_pos;
}

static int llm_ends_with(const char *s, int len, const char *suffix)
{
    int slen = (int)kstrlen(suffix);
    if (slen <= 0 || len < slen) return 0;
    const char *tail = s + (len - slen);
    for (int i = 0; i < slen; ++i) {
        if (tail[i] != suffix[i]) return 0;
    }
    return 1;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Dynamic scratch arena (allocated from tensor heap after model load)         */
/* ─────────────────────────────────────────────────────────────────────────── */

static float *llm_kv_k;        /* [n_layers * max_seq * n_kv_heads * head_dim] */
static float *llm_kv_v;

static float *llm_x;           /* [dim]    hidden state */
static float *llm_xn;          /* [dim]    normalized */
static float *llm_last_hs;     /* [dim]    last forward-pass hidden state (context-conditioned) */
static int    llm_last_hs_valid = 0;
static float *llm_q;           /* [dim]    query */
static float *llm_k_buf;       /* [dim]    key (current token) */
static float *llm_v_buf;       /* [dim]    value (current token) */
static float *llm_attn_out;    /* [dim]    attention output */
static float *llm_ffn_g;       /* [ff_dim] FFN gate */
static float *llm_ffn_u;       /* [ff_dim] FFN up */
static float *llm_ffn_d;       /* [dim]    FFN down */
static float *llm_head_buf;    /* [dim]    LM head scratch */
static float *llm_attn_scores; /* [max_seq] attention scores */
static float *llm_logits;          /* [vocab_size] output logits */
static int    llm_logits_pos = -1; /* position of last valid llm_logits (-1 = stale) */
static int    llm_primed_greedy  = -1; /* cached argmax of llm_logits when primed */
static int   *llm_tokens;      /* [max_tokens] token buffer */
static float *llm_rope_freqs_buf; /* [head_dim/2] precomputed RoPE */
static float *llm_iswa_per_layer; /* [iswa_n_embd * n_layers] per-layer embeddings */
static float *llm_iswa_tmp;       /* [iswa_n_embd] ISWA gating scratch */
static float *llm_depth_hist;     /* [n_layers * dim] layer history for theorem mode */
static float *llm_depth_scores;   /* [n_layers] depth-attention score scratch */

/* Sizes cached from the loaded model (for bounds checking) */
static int llm_alloc_dim;
static int llm_alloc_ff;
static int llm_alloc_seq;
static int llm_alloc_vocab;
static int llm_alloc_tokens;
static int64_t llm_alloc_kv_floats; /* int64_t: layers×seq×kv_heads×hd can exceed 2^31 on 70B+ models */

/* Prefix-keyed KV snapshots (token-native boundary for cache reuse). */
#define LLM_KV_SNAPSHOT_SLOTS 16
typedef struct {
    int in_use;
    uint32_t prefix_hash;
    int n_prefix_tokens;
    int cached_len;
    int n_layers;
    int max_seq;
    int n_kv_heads;
    int head_dim;
    int *prefix_tokens;
    float *k_cache;
    float *v_cache;
} llm_kv_snapshot_slot_t;

static llm_kv_snapshot_slot_t llm_kv_snapshots[LLM_KV_SNAPSHOT_SLOTS];
static int llm_kv_snapshot_evict;

#define LLM_AGENT_CTX_SLOTS 4
#define LLM_AGENT_CTX_MAX_TOKENS 32768
typedef struct {
    int in_use;
    int n_tokens;
    int tokens[LLM_AGENT_CTX_MAX_TOKENS];
} llm_agent_ctx_slot_t;

static llm_agent_ctx_slot_t llm_agent_ctx[LLM_AGENT_CTX_SLOTS];

#define LLM_RAG_PREFIX_MAX 32
typedef struct {
    int active;
    int n_prefix;
    int dim;
    float *emb;
} llm_rag_prefix_t;

static llm_rag_prefix_t llm_rag_prefix;

typedef struct {
    int n_steps;
    llm_rollout_step_t steps[LLM_ROLLOUT_MAX_STEPS];
} llm_rollout_trace_t;

static llm_rollout_trace_t llm_rollout_trace;

static void llm_rag_prefix_clear_internal(void)
{
    if (llm_rag_prefix.emb) tensor_free(llm_rag_prefix.emb);
    kmemset(&llm_rag_prefix, 0, sizeof(llm_rag_prefix));
}

static uint32_t llm_hash_tokens(const int *tokens, int n)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)tokens[i];
        h ^= (v & 0xFFu);
        h *= 16777619u;
        h ^= ((v >> 8) & 0xFFu);
        h *= 16777619u;
        h ^= ((v >> 16) & 0xFFu);
        h *= 16777619u;
        h ^= ((v >> 24) & 0xFFu);
        h *= 16777619u;
    }
    return h;
}

static void llm_kv_snapshot_clear_slot(llm_kv_snapshot_slot_t *s)
{
    if (!s) return;
    if (s->prefix_tokens) tensor_free(s->prefix_tokens);
    if (s->k_cache) tensor_free(s->k_cache);
    if (s->v_cache) tensor_free(s->v_cache);
    kmemset(s, 0, sizeof(*s));
}

static int llm_kv_snapshot_prefix_match(const llm_kv_snapshot_slot_t *s,
                                        const int *prefix_tokens,
                                        int n_prefix_tokens,
                                        uint32_t prefix_hash)
{
    if (!s || !s->in_use) return 0;
    if (s->prefix_hash != prefix_hash) return 0;
    if (s->n_prefix_tokens != n_prefix_tokens) return 0;
    if (!s->prefix_tokens || !prefix_tokens) return 0;
    for (int i = 0; i < n_prefix_tokens; i++) {
        if (s->prefix_tokens[i] != prefix_tokens[i]) return 0;
    }
    return 1;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  GPU Weight Offload (CUDA)                                                  */
/*                                                                             */
/*  Pre-uploads all weight matrices to GPU VRAM at model load time.            */
/*  During inference, llm_gemv() intercepts calls and dispatches to CUDA       */
/*  GEMV when a GPU copy of the weight exists.                                 */
/* ─────────────────────────────────────────────────────────────────────────── */

#ifdef ENABLE_CUDA

#define GPU_WEIGHT_MAP_MAX 512
#define GPU_MAX_LAYERS     128

typedef struct {
    const void *host_ptr;   /* mmap'd host weight pointer (lookup key) */
    void       *dev_ptr;    /* GPU device pointer */
    uint64_t    size;       /* Tensor size in bytes */
    ggml_type_t dev_type;   /* Type of data as stored on GPU (may differ from host after dequant) */
} gpu_weight_entry_t;

/* Per-layer GPU-resident norm weights and KV cache */
typedef struct {
    float *d_attn_norm;        /* [dim] attention RMSNorm weight (F32 on GPU) */
    float *d_ffn_norm;         /* [dim] FFN RMSNorm weight */
    float *d_post_attn_norm;   /* [dim] post-attention norm (Gemma4, NULL if absent) */
    float *d_post_ffw_norm;    /* [dim] post-FFW norm (Gemma4, NULL if absent) */
    float *d_q_norm;           /* [n_heads * lhd] per-head Q norm (NULL if absent) */
    float *d_k_norm;           /* [n_kv * lhd] per-head K norm (NULL if absent) */
    float *d_k_cache;          /* [n_kv * max_seq * lhd] KV cache K */
    float *d_v_cache;          /* [n_kv * max_seq * lhd] KV cache V */
    float *d_iswa_post_norm;   /* [dim] ISWA post-norm weight (Gemma4, NULL if absent) */
    int    lhd;                /* actual head dim for this layer */
    int    on_gpu;             /* 1 = weights uploaded, 0 = CPU fallback for this layer */
} gpu_layer_t;

static struct {
    int active;                            /* 1 if GPU offload is live */
    gpu_weight_entry_t map[GPU_WEIGHT_MAP_MAX];
    int map_count;
    float *d_x;                            /* Device input scratch */
    float *d_out;                          /* Device output scratch */
    int max_in_dim;
    int max_out_dim;

    /* GPU-resident forward pass state */
    int gpu_fwd;                /* 1 if GPU-resident forward pass is ready */
    float *d_xn;                /* [dim] normalized hidden state */
    float *d_q;                 /* [max_q_dim] Q projection output */
    float *d_k;                 /* [max_kv_dim] K buffer */
    float *d_v;                 /* [max_kv_dim] V buffer */
    float *d_attn;              /* [max_q_dim] attention output */
    float *d_ffn_g;             /* [max_ff] FFN gate */
    float *d_ffn_u;             /* [max_ff] FFN up */
    float *d_ffn_d;             /* [dim] FFN down / scratch */
    float *d_compress_tmp;      /* [AXEX_COMPRESS_MAX_RANK] scratch for GPU two-GEMV */
    float *d_x_sub;             /* [AXEX_MANIFOLD_K_MAX] GP manifold projection scratch */
    float *d_output_norm;       /* [dim] final output norm weights */
    float *d_rope_freqs;        /* RoPE freq factors for full-attn layers (NULL if unused) */
    gpu_layer_t layers[GPU_MAX_LAYERS];

    /* Cached model dimensions for GPU forward pass */
    int dim;
    int max_seq;
    int n_heads;
    int n_kv;
    int head_dim;               /* max head dim */
    int head_dim_swa;           /* SWA head dim (Gemma4) */
    int max_ff;
    int vocab_size;
    float rms_eps;
    int n_layers;
    int is_gemma4;
    int use_geglu;
    int use_gelu;
    int use_layernorm;
    int n_layer_kv_start;       /* Gemma4: first layer with own KV */
    float rope_base;
    float rope_base_swa;

    /* ISWA GPU-resident state (Gemma4) */
    float *d_iswa_tmp;          /* [iswa_n_embd] ISWA gating scratch */
    float *d_iswa_per_layer;    /* [iswa_n_embd * n_layers] precomputed per-token */
    float *d_iswa_proj;         /* [iswa_n_embd * n_layers] model projection output */
    float *d_iswa_proj_norm;    /* [iswa_n_embd] shared norm weights */
    void  *d_iswa_tok_embd;     /* [vocab × iswa_total] Q6_K token embd on GPU */
    int iswa_n_embd;
    int iswa_tok_embd_type;     /* quant type of ISWA token embedding */

    /* Batch Prefill scratch — lazy-allocated, freed on context reset */
    float *d_pfx;               /* [prefill_max_n × dim] */
    float *d_pfxn;              /* [prefill_max_n × dim] */
    float *d_pfq;               /* [prefill_max_n × max_q_dim] */
    float *d_pfk;               /* [prefill_max_n × max_kv_dim] */
    float *d_pfv;               /* [prefill_max_n × max_kv_dim] */
    float *d_pfattn;            /* [prefill_max_n × max_q_dim] */
    float *d_pffg;              /* [prefill_max_n × max_ff] */
    float *d_pffu;              /* [prefill_max_n × max_ff] */
    float *d_pffd;              /* [prefill_max_n × dim] */
    float *d_pfiswa;            /* [prefill_max_n × iswa_total] tok_embd+final */
    float *d_pfiswa_proj;       /* [prefill_max_n × iswa_total] model_proj output */
    int    prefill_max_n;       /* current allocation batch size */
} gpu_ctx;

/* Look up device pointer for a host weight pointer */
static void *llm_gpu_lookup(const void *host_ptr) {
    for (int i = 0; i < gpu_ctx.map_count; i++) {
        if (gpu_ctx.map[i].host_ptr == host_ptr)
            return gpu_ctx.map[i].dev_ptr;
    }
    return (void *)0;
}

/* Look up device pointer and return GPU-resident type (may differ from host type) */
static void *llm_gpu_lookup_with_type(const void *host_ptr, ggml_type_t *out_type) {
    for (int i = 0; i < gpu_ctx.map_count; i++) {
        if (gpu_ctx.map[i].host_ptr == host_ptr) {
            if (out_type) *out_type = gpu_ctx.map[i].dev_type;
            return gpu_ctx.map[i].dev_ptr;
        }
    }
    return (void *)0;
}

/* Upload one weight tensor to GPU and register in the map */
static int llm_gpu_register_weight(const void *host, uint64_t size, ggml_type_t type) {
    if (!host || size == 0) return 0;
    if (gpu_ctx.map_count >= GPU_WEIGHT_MAP_MAX) return -1;
    const backend_t *be = backend_get_by_id(BACKEND_CUDA);
    if (!be) return -1;

    void *dev = be->mem.alloc(size);
    if (!dev) return -2;
    be->mem.upload(dev, host, size);

    gpu_ctx.map[gpu_ctx.map_count].host_ptr = host;
    gpu_ctx.map[gpu_ctx.map_count].dev_ptr  = dev;
    gpu_ctx.map[gpu_ctx.map_count].size     = size;
    gpu_ctx.map[gpu_ctx.map_count].dev_type = type;
    gpu_ctx.map_count++;
    return 0;
}

/* Helper: actual bytes we will store in VRAM for a weight of the given type.
 * Q2_K is dequantized to F32 on upload, so cost is n * sizeof(float).
 * IQ2_XS now has a native kernel — stored at its packed size (74B/256 elements). */
static uint64_t llm_gpu_actual_bytes(int rows, int cols, ggml_type_t type) {
    if (type == GGML_TYPE_Q2_K)
        return (uint64_t)rows * cols * sizeof(float);
    return llm_row_bytes(cols, type) * (uint64_t)rows;
}

/* Estimate bytes needed to upload all weights for layer L to VRAM.
 * Used by the budget check so we can predict before committing. */
static uint64_t llm_layer_vram_bytes(const llm_model_t *m, int L)
{
    const llm_layer_t *layer = &m->layers[L];
    int lhd     = layer->head_dim_layer ? layer->head_dim_layer : m->head_dim;
    int lq_dim  = m->n_heads * lhd;
    int lkv_dim = m->n_kv_heads * lhd;
    int lff     = layer->ff_dim_layer ? layer->ff_dim_layer : m->ff_dim;
    int has_kv  = (layer->kv_reuse_layer < 0);
    uint64_t b = 0;
    b += llm_gpu_actual_bytes(lq_dim,  m->dim, layer->q_type);
    if (has_kv) {
        b += llm_gpu_actual_bytes(lkv_dim, m->dim, layer->k_type);
        b += llm_gpu_actual_bytes(lkv_dim, m->dim, layer->v_type);
    }
    b += llm_gpu_actual_bytes(m->dim,  lq_dim, layer->o_type);
    if (layer->ffn_gate)
        b += llm_gpu_actual_bytes(lff,     m->dim, layer->gate_type);
    b += llm_gpu_actual_bytes(lff,     m->dim, layer->up_type);
    b += llm_gpu_actual_bytes(m->dim,  lff,    layer->down_type);
    return b;
}

/* VRAM estimate for attention-only upload (compress mode: FFN not uploaded raw).
 * Includes attention weights + norm F32 + compressed FFN F32 if available. */
static uint64_t llm_layer_vram_bytes_attn_only(const llm_model_t *m, int L)
{
    const llm_layer_t *layer = &m->layers[L];
    int lhd     = layer->head_dim_layer ? layer->head_dim_layer : m->head_dim;
    int lq_dim  = m->n_heads * lhd;
    int lkv_dim = m->n_kv_heads * lhd;
    int has_kv  = (layer->kv_reuse_layer < 0);
    uint64_t b = 0;
    b += llm_row_bytes(m->dim, layer->q_type) * (uint64_t)lq_dim;
    if (has_kv) {
        b += llm_row_bytes(m->dim, layer->k_type) * (uint64_t)lkv_dim;
        b += llm_row_bytes(m->dim, layer->v_type) * (uint64_t)lkv_dim;
    }
    b += llm_row_bytes(lq_dim, layer->o_type) * (uint64_t)m->dim;
    b += (uint64_t)m->dim * sizeof(float) * 4; /* 4 norm vectors */
    /* Add compressed FFN F32 footprint (if already computed, 0 if not yet) */
    for (int slot = 0; slot < AXEX_COMPRESS_WHICH_MAX; slot++) {
        const axex_compressed_weight_t *cw = axex_get_compressed_layer(L, slot);
        if (cw) b += (uint64_t)cw->rank * (cw->m + cw->n) * sizeof(float);
    }
    return b;
}

/* VRAM estimate for FFN + K/V upload (attn-SVD mode: Q/O replaced by SVD factors,
 * but K and V are still uploaded raw for the KV cache). */
static uint64_t llm_layer_vram_bytes_kv_and_ffn(const llm_model_t *m, int L)
{
    const llm_layer_t *layer = &m->layers[L];
    int lhd     = layer->head_dim_layer ? layer->head_dim_layer : m->head_dim;
    int lkv_dim = m->n_kv_heads * lhd;
    int lff     = layer->ff_dim_layer ? layer->ff_dim_layer : m->ff_dim;
    int has_kv  = (layer->kv_reuse_layer < 0);
    uint64_t b = 0;
    if (has_kv) {
        b += llm_row_bytes(m->dim, layer->k_type) * (uint64_t)lkv_dim;
        b += llm_row_bytes(m->dim, layer->v_type) * (uint64_t)lkv_dim;
    }
    if (layer->ffn_gate)
        b += llm_row_bytes(m->dim, layer->gate_type) * (uint64_t)lff;
    b += llm_row_bytes(m->dim, layer->up_type)   * (uint64_t)lff;
    b += llm_row_bytes(lff,    layer->down_type) * (uint64_t)m->dim;
    b += (uint64_t)m->dim * sizeof(float) * 4; /* 4 norm vectors */
    return b;
}

/* VRAM estimate for FFN-only upload (full GP manifold path: Q/K/V/O all CPU-side).
 * Used when attention is GP-compressed to CPU manifold path; FFN stays on GPU. */
static uint64_t llm_layer_vram_bytes_ffn_only(const llm_model_t *m, int L)
{
    const llm_layer_t *layer = &m->layers[L];
    int lff = layer->ff_dim_layer ? layer->ff_dim_layer : m->ff_dim;
    uint64_t b = 0;
    if (layer->ffn_gate)
        b += llm_row_bytes(m->dim, layer->gate_type) * (uint64_t)lff;
    b += llm_row_bytes(m->dim, layer->up_type)   * (uint64_t)lff;
    b += llm_row_bytes(lff,    layer->down_type) * (uint64_t)m->dim;
    b += (uint64_t)m->dim * sizeof(float) * 4; /* 4 norm vectors */
    return b;
}

/* VRAM estimate when both attention and FFN are compressed (full GP compress mode).
 * Only norms remain on GPU. */
static uint64_t llm_layer_vram_bytes_norms_only(const llm_model_t *m, int L)
{
    (void)L;
    return (uint64_t)m->dim * sizeof(float) * 4; /* 4 norm vectors per layer */
}

/* Upload a weight matrix: out_dim rows, each llm_row_bytes(in_dim, type) */
static int llm_gpu_upload_weight_mat(const void *weight, int out_dim, int in_dim,
                                     ggml_type_t type) {
    if (!weight) return 0;

    /* Q2_K: not directly supported by CUDA GEMV kernels.
     * Dequantize to F32 on host and upload as F32 so the GPU can handle them.
     * NOTE: IQ2_XS now has a native CUDA kernel — handled below. */
    if (type == GGML_TYPE_Q2_K) {
#ifdef ENABLE_CUDA
        uint64_t n_floats = (uint64_t)out_dim * in_dim;
        float *f32_buf = (float *)malloc(n_floats * sizeof(float));
        if (!f32_buf) return -1;
        /* Dequantize row by row using the CPU backend dequantize function */
        const backend_t *cpu_be = backend_get_by_id(BACKEND_CPU);
        uint64_t rb = llm_row_bytes(in_dim, type);
        for (int r = 0; r < out_dim; r++) {
            const uint8_t *src = (const uint8_t *)weight + (uint64_t)r * rb;
            cpu_be->compute.dequantize(f32_buf + (uint64_t)r * in_dim, src, in_dim, type);
        }
        const backend_t *be = backend_get_by_id(BACKEND_CUDA);
        if (!be || gpu_ctx.map_count >= GPU_WEIGHT_MAP_MAX) { free(f32_buf); return -1; }
        uint64_t sz = n_floats * sizeof(float);
        void *dev = be->mem.alloc(sz);
        if (!dev) { free(f32_buf); return -2; }
        be->mem.upload(dev, f32_buf, sz);
        free(f32_buf);
        gpu_ctx.map[gpu_ctx.map_count].host_ptr = weight;
        gpu_ctx.map[gpu_ctx.map_count].dev_ptr  = dev;
        gpu_ctx.map[gpu_ctx.map_count].size     = sz;
        gpu_ctx.map[gpu_ctx.map_count].dev_type = GGML_TYPE_F32;
        gpu_ctx.map_count++;
        return 0;
#else
        return 0;
#endif
    }

    /* Only upload types the CUDA GEMV kernel can handle.
     * Q3_K, Q5_K, IQ4_NL, IQ4_XS have no CUDA kernels — skip upload so
     * llm_gemv falls back to CPU (which handles all types correctly). */
    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1 && type != GGML_TYPE_Q8_0 &&
        type != GGML_TYPE_F32  && type != GGML_TYPE_F16  && type != GGML_TYPE_Q4_K &&
        type != GGML_TYPE_Q6_K && type != GGML_TYPE_BF16 && type != GGML_TYPE_IQ2_XS)
        return 0;

#ifdef ENABLE_CUDA
    /* Q4_0 → SOA layout: DISABLED for A/B test */
#if 0
    if (type == GGML_TYPE_Q4_0) {
        int nb = in_dim / 32;
        uint64_t row_bytes = (uint64_t)nb * 18;
        uint64_t total_size = (uint64_t)out_dim * row_bytes;

        uint8_t *soa = (uint8_t *)malloc((size_t)total_size);
        if (!soa) {
            /* Fallback: upload AOS as-is */
            return llm_gpu_register_weight(weight, total_size, type);
        }

        const uint8_t *aos = (const uint8_t *)weight;
        int q_off = nb * 2;  /* quants offset within SOA row */

        for (int r = 0; r < out_dim; r++) {
            const uint8_t *src = aos + (int64_t)r * row_bytes;
            uint8_t *dst = soa + (int64_t)r * row_bytes;
            /* Pack all scales contiguously */
            for (int b = 0; b < nb; b++) {
                dst[b * 2 + 0] = src[b * 18 + 0];
                dst[b * 2 + 1] = src[b * 18 + 1];
            }
            /* Pack all quants contiguously after scales */
            for (int b = 0; b < nb; b++)
                memcpy(dst + q_off + b * 16, src + b * 18 + 2, 16);
        }

        /* Upload SOA data, register with original host pointer as key */
        const backend_t *be = backend_get_by_id(BACKEND_CUDA);
        if (!be || gpu_ctx.map_count >= GPU_WEIGHT_MAP_MAX) {
            free(soa);
            return llm_gpu_register_weight(weight, total_size, type);
        }
        void *dev = be->mem.alloc(total_size);
        if (!dev) { free(soa); return -2; }
        be->mem.upload(dev, soa, total_size);
        free(soa);

        gpu_ctx.map[gpu_ctx.map_count].host_ptr = weight;
        gpu_ctx.map[gpu_ctx.map_count].dev_ptr  = dev;
        gpu_ctx.map[gpu_ctx.map_count].size     = total_size;
        gpu_ctx.map[gpu_ctx.map_count].dev_type = type;
        gpu_ctx.map_count++;
        return 0;
    }
#endif /* SOA disabled */
#endif /* ENABLE_CUDA */

    uint64_t rb = llm_row_bytes(in_dim, type);
    uint64_t size = (uint64_t)out_dim * rb;
    return llm_gpu_register_weight(weight, size, type);
}

/* Upload a norm weight vector to GPU as F32.
 * For F32 norm weights: direct upload.
 * For non-F32: skip (returns NULL) — caller falls back to CPU for that norm. */
static float *llm_gpu_upload_norm(const void *weight, int dim, ggml_type_t type) {
    if (!weight || dim <= 0) return (float *)0;
    const backend_t *be = backend_get_by_id(BACKEND_CUDA);
    if (!be) return (float *)0;

    if (type == GGML_TYPE_F32) {
        float *d = (float *)be->mem.alloc((uint64_t)dim * sizeof(float));
        if (d) be->mem.upload(d, weight, (uint64_t)dim * sizeof(float));
        return d;
    }
    /* Non-F32 norm weights: not supported on GPU path */
    return (float *)0;
}

/* When set, llm_gpu_init() skips raw FFN uploads — compression will supply GPU FFN.
 * Call llm_gpu_set_compress_mode(1) before llm_gpu_init() to activate. */
static int gpu_ffn_skip_for_compress = 0;

/* When set, llm_gpu_init() skips raw Q/K/V/O uploads — manifold GP path supplies them.
 * In attention-only compress mode (default), set this to save ~1.3 GB VRAM on 8B models.
 * Call llm_gpu_set_attn_compress_mode(1) before llm_gpu_init() to activate. */
static int gpu_attn_skip_for_compress = 0;

void llm_gpu_set_compress_mode(int enable) { gpu_ffn_skip_for_compress = enable; }
void llm_gpu_set_attn_compress_mode(int enable) { gpu_attn_skip_for_compress = enable; }

/*
 * llm_gpu_upload_ffn_fallback — re-upload raw FFN weights when compression was
 * fully skipped (e.g., all matrices exceeded --axex-compress-max-err threshold).
 * gpu_ffn_skip_for_compress was set before llm_gpu_init so raw FFN weights were
 * not uploaded. Call this after axex_init if compress_layers == 0.
 */
void llm_gpu_upload_ffn_fallback(void) {
    llm_model_t *m = llm_get_model();
    if (!m || !gpu_ctx.active) return;
    int uploaded = 0;
    for (int L = 0; L < m->n_layers; L++) {
        llm_layer_t *layer = &m->layers[L];
        gpu_layer_t *gl = &gpu_ctx.layers[L];
        if (!gl->on_gpu) continue;
        int dim = m->dim;
        int lff = (layer->ff_dim_layer > 0) ? layer->ff_dim_layer : m->ff_dim;
        if (layer->ffn_gate && !llm_gpu_lookup(layer->ffn_gate)) {
            llm_gpu_upload_weight_mat(layer->ffn_gate, lff, dim, layer->gate_type);
            uploaded++;
        }
        if (layer->ffn_up && !llm_gpu_lookup(layer->ffn_up)) {
            llm_gpu_upload_weight_mat(layer->ffn_up,   lff, dim, layer->up_type);
            uploaded++;
        }
        if (layer->ffn_down && !llm_gpu_lookup(layer->ffn_down)) {
            llm_gpu_upload_weight_mat(layer->ffn_down, dim, lff, layer->down_type);
            uploaded++;
        }
    }
    kprintf("[GPU] FFN fallback: uploaded %d raw FFN tensors to GPU\n", uploaded);
    /* Verify all on_gpu layers now have FFN weights resident.  If so, clear
     * gpu_ffn_skip_for_compress so the fast batch-prefill path runs — safe
     * because we just confirmed no null weight pointers exist. */
    int all_ok = 1;
    for (int L = 0; L < m->n_layers && all_ok; L++) {
        llm_layer_t *layer = &m->layers[L];
        gpu_layer_t *gl    = &gpu_ctx.layers[L];
        if (!gl->on_gpu) continue;
        if ((layer->ffn_gate && !llm_gpu_lookup(layer->ffn_gate)) ||
            (layer->ffn_up   && !llm_gpu_lookup(layer->ffn_up))   ||
            (layer->ffn_down && !llm_gpu_lookup(layer->ffn_down)))
            all_ok = 0;
    }
    if (all_ok) {
        gpu_ffn_skip_for_compress = 0;
        kprintf("[GPU] FFN fallback: all weights verified — compress-skip cleared, batch-prefill enabled\n");
    }
}

/* Upload all model weights to GPU. Called after model load if CUDA available. */
static int llm_gpu_init(llm_model_t *m) {
    const backend_t *be = backend_get_by_id(BACKEND_CUDA);
    if (!be) return -1;

    kmemset(&gpu_ctx, 0, sizeof(gpu_ctx));

    int dim = m->dim;
    int hd  = m->head_dim;
    int n_heads = m->n_heads;
    int n_kv = m->n_kv_heads;
    int ff = m->ff_dim;
    for (int i = 0; i < m->n_layers; i++)
        if (m->layers[i].ff_dim_layer > ff) ff = m->layers[i].ff_dim_layer;

    int q_dim  = n_heads * hd;
    int kv_dim = n_kv * hd;

    /* Compute max input/output dimensions for scratch buffers */
    gpu_ctx.max_in_dim = dim > ff ? dim : ff;
    if (q_dim > gpu_ctx.max_in_dim) gpu_ctx.max_in_dim = q_dim;
    gpu_ctx.max_out_dim = m->vocab_size;
    if (q_dim > gpu_ctx.max_out_dim) gpu_ctx.max_out_dim = q_dim;
    if (ff > gpu_ctx.max_out_dim) gpu_ctx.max_out_dim = ff;

    /* Check if model fits in GPU memory (leave 256 MB headroom) */
    uint64_t free_mem = be->get_free_memory(0);
    uint64_t scratch_bytes = (uint64_t)(gpu_ctx.max_in_dim + gpu_ctx.max_out_dim) * sizeof(float);
    kprintf("[GPU] VRAM free: %llu MB, scratch: %llu KB\n",
            (unsigned long long)(free_mem / (1024 * 1024)),
            (unsigned long long)(scratch_bytes / 1024));

    /* Allocate GPU scratch buffers */
    gpu_ctx.d_x   = (float *)be->mem.alloc((uint64_t)gpu_ctx.max_in_dim * sizeof(float));
    gpu_ctx.d_out  = (float *)be->mem.alloc((uint64_t)gpu_ctx.max_out_dim * sizeof(float));
    if (!gpu_ctx.d_x || !gpu_ctx.d_out) {
        kprintf("[GPU] Failed to allocate scratch buffers\n");
        return -1;
    }

    /* Upload per-layer weights — stop when VRAM budget is exhausted.
     * Reserve 1.5 GB headroom for activation buffers, KV cache, and scratch.
     * Layers that don't fit are marked on_gpu=0 and will run on CPU. */
    #define VRAM_LAYER_RESERVE (1536ULL * 1024 * 1024)
    int vram_layers_ok = 1;  /* becomes 0 when we first run out of VRAM */
    for (int L = 0; L < m->n_layers; L++) {
        llm_layer_t *layer = &m->layers[L];
        gpu_layer_t *gl    = &gpu_ctx.layers[L];
        int lhd     = layer->head_dim_layer ? layer->head_dim_layer : hd;
        int lq_dim  = n_heads * lhd;
        int lkv_dim = n_kv * lhd;
        int lff     = layer->ff_dim_layer ? layer->ff_dim_layer : ff;

        if (vram_layers_ok) {
            /* Budget estimation depends on which weights will actually land on GPU:
             *   - attn-only compress: only FFN + norms (attn is manifold CPU path)
             *   - full compress: only norms (attn + FFN both manifold CPU path)
             *   - no compress: full layer (attn + FFN + norms) */
            uint64_t layer_need;
            if (gpu_attn_skip_for_compress && gpu_ffn_skip_for_compress)
                layer_need = llm_layer_vram_bytes_norms_only(m, L);
            else if (gpu_attn_skip_for_compress)
                layer_need = llm_layer_vram_bytes_kv_and_ffn(m, L);
            else if (gpu_ffn_skip_for_compress)
                layer_need = llm_layer_vram_bytes_attn_only(m, L);
            else
                layer_need = llm_layer_vram_bytes(m, L);
            uint64_t vram_free  = be->get_free_memory(0);
            if (vram_free < layer_need + VRAM_LAYER_RESERVE) {
                vram_layers_ok = 0;
                kprintf("[GPU] Layer %d: VRAM tight (%lluMB free, %lluMB needed) → CPU offload\n",
                        L, (unsigned long long)(vram_free >> 20),
                           (unsigned long long)(layer_need >> 20));
            }
        }

        gl->on_gpu = vram_layers_ok;

        if (!gl->on_gpu) continue;  /* skip weight upload for CPU layers */

        if (!gpu_attn_skip_for_compress) {
            /* Only upload raw Q/O when not using SVD/GP compressed attention */
            llm_gpu_upload_weight_mat(layer->q_weight, lq_dim, dim, layer->q_type);
            llm_gpu_upload_weight_mat(layer->o_weight, dim, lq_dim, layer->o_type);
        }
        /* K and V are always uploaded: needed for KV cache even in attn-SVD mode */
        if (layer->kv_reuse_layer < 0) {
            llm_gpu_upload_weight_mat(layer->k_weight, lkv_dim, dim, layer->k_type);
            llm_gpu_upload_weight_mat(layer->v_weight, lkv_dim, dim, layer->v_type);
        }
        if (!gpu_ffn_skip_for_compress) {
            /* Only upload raw FFN when not going to compress → saves PCIe BW */
            if (layer->ffn_gate)
                llm_gpu_upload_weight_mat(layer->ffn_gate, lff, dim, layer->gate_type);
            llm_gpu_upload_weight_mat(layer->ffn_up,   lff, dim, layer->up_type);
            llm_gpu_upload_weight_mat(layer->ffn_down, dim, lff, layer->down_type);
        }

        /* ISWA injection weights (Gemma4) */
        if (layer->iswa_inp_gate)
            llm_gpu_upload_weight_mat(layer->iswa_inp_gate, m->iswa_n_embd,
                                      dim, layer->iswa_inp_gate_type);
        if (layer->iswa_proj)
            llm_gpu_upload_weight_mat(layer->iswa_proj, dim, m->iswa_n_embd,
                                      layer->iswa_proj_type);

        /* Post-upload verification: if any mandatory weight failed to upload
         * (e.g. dequant-on-upload OOM'd), mark layer as CPU-offloaded so the
         * GPU forward pass doesn't invoke CUDA with null weight pointers. */
        if (!gpu_attn_skip_for_compress) {
            if ((layer->q_weight   && !llm_gpu_lookup(layer->q_weight)) ||
                (layer->o_weight   && !llm_gpu_lookup(layer->o_weight))) {
                gl->on_gpu = 0;
            }
        }
        /* K and V verification always runs (uploaded regardless of compress mode) */
        if (gl->on_gpu && layer->kv_reuse_layer < 0) {
            if ((layer->k_weight && !llm_gpu_lookup(layer->k_weight)) ||
                (layer->v_weight && !llm_gpu_lookup(layer->v_weight)))
                gl->on_gpu = 0;
        }
        if (!gpu_ffn_skip_for_compress && gl->on_gpu) {
            if ((layer->ffn_gate && !llm_gpu_lookup(layer->ffn_gate)) ||
                (layer->ffn_up   && !llm_gpu_lookup(layer->ffn_up))   ||
                (layer->ffn_down && !llm_gpu_lookup(layer->ffn_down))) {
                gl->on_gpu = 0;
            }
        }
    }

    /* Upload global weights */
    llm_gpu_upload_weight_mat(m->token_embd, m->vocab_size, dim, m->token_embd_type);
    if (m->output_weight)
        llm_gpu_upload_weight_mat(m->output_weight, m->vocab_size, dim, m->output_type);
    if (m->iswa_model_proj)
        llm_gpu_upload_weight_mat(m->iswa_model_proj,
                                  m->iswa_n_embd * m->n_layers, dim,
                                  m->iswa_model_proj_type);

    be->mem.sync();

    /* Report total uploaded */
    uint64_t total = 0;
    for (int i = 0; i < gpu_ctx.map_count; i++)
        total += gpu_ctx.map[i].size;

    gpu_ctx.active = 1;
    llm_last_vram_mb = (int)(total / (1024 * 1024));
    llm_backend = LLM_BACKEND_CUDA;
    kprintf("[GPU] Uploaded %d weight tensors (%llu MB) to VRAM\n",
            gpu_ctx.map_count, (unsigned long long)(total / (1024 * 1024)));

#ifdef ENABLE_CUDA
    /* Q4_0 dequant flag — disabled while using native Q4_0 GEMV */
    /* cuda_set_q4_dequant_flag(1); */
#endif

    /* ── GPU-Resident Forward Pass Setup ── */
    /* Allocate activation buffers and KV cache on GPU.
     * Upload norm weights so the entire forward pass stays on device. */

    gpu_ctx.dim        = dim;
    gpu_ctx.max_seq    = m->max_seq;
    gpu_ctx.n_heads    = n_heads;
    gpu_ctx.n_kv       = n_kv;
    gpu_ctx.head_dim   = hd;
    gpu_ctx.head_dim_swa = m->is_gemma4 ? m->head_dim_swa : hd;
    gpu_ctx.max_ff     = ff;
    gpu_ctx.vocab_size = m->vocab_size;
    gpu_ctx.rms_eps    = m->rms_eps;
    gpu_ctx.n_layers   = m->n_layers;
    gpu_ctx.is_gemma4  = m->is_gemma4;
    gpu_ctx.use_geglu  = m->use_geglu;
    gpu_ctx.use_gelu   = m->use_gelu;
    gpu_ctx.use_layernorm = m->use_layernorm;
    gpu_ctx.n_layer_kv_start = m->n_layer_kv_start;
    gpu_ctx.rope_base  = m->rope_base;
    gpu_ctx.rope_base_swa = m->is_gemma4 ? m->rope_base_swa : m->rope_base;

    /* Activation scratch buffers */
    gpu_ctx.d_xn    = (float *)be->mem.alloc((uint64_t)dim * sizeof(float));
    gpu_ctx.d_q     = (float *)be->mem.alloc((uint64_t)q_dim * sizeof(float));
    gpu_ctx.d_k     = (float *)be->mem.alloc((uint64_t)kv_dim * sizeof(float));
    gpu_ctx.d_v     = (float *)be->mem.alloc((uint64_t)kv_dim * sizeof(float));
    gpu_ctx.d_attn  = (float *)be->mem.alloc((uint64_t)q_dim * sizeof(float));
    gpu_ctx.d_ffn_g = (float *)be->mem.alloc((uint64_t)ff * sizeof(float));
    gpu_ctx.d_ffn_u = (float *)be->mem.alloc((uint64_t)ff * sizeof(float));
    gpu_ctx.d_ffn_d = (float *)be->mem.alloc((uint64_t)dim * sizeof(float));
    /* Scratch for GPU compressed-weight two-GEMV intermediate (rank-sized) */
    gpu_ctx.d_compress_tmp = (float *)be->mem.alloc(
        (uint64_t)AXEX_COMPRESS_MAX_RANK * sizeof(float));
    gpu_ctx.d_x_sub = (float *)be->mem.alloc(
        (uint64_t)AXEX_MANIFOLD_K_MAX * sizeof(float));

    int fwd_ok = gpu_ctx.d_xn && gpu_ctx.d_q && gpu_ctx.d_k && gpu_ctx.d_v &&
                 gpu_ctx.d_attn && gpu_ctx.d_ffn_g && gpu_ctx.d_ffn_u && gpu_ctx.d_ffn_d;

    /* Per-layer KV cache + norm weights */
    if (fwd_ok && m->n_layers <= GPU_MAX_LAYERS) {
        for (int L = 0; L < m->n_layers && fwd_ok; L++) {
            llm_layer_t *layer = &m->layers[L];
            int lhd = layer->head_dim_layer ? layer->head_dim_layer : hd;
            uint64_t kv_layer_size = (uint64_t)n_kv * m->max_seq * lhd * sizeof(float);
            gpu_layer_t *gl = &gpu_ctx.layers[L];
            gl->lhd = lhd;

            /* CPU-fallback layers: skip KV/norm GPU allocation */
            if (!gl->on_gpu) {
                gl->d_k_cache        = NULL;
                gl->d_v_cache        = NULL;
                gl->d_attn_norm      = NULL;
                gl->d_ffn_norm       = NULL;
                gl->d_post_attn_norm = NULL;
                gl->d_post_ffw_norm  = NULL;
                continue;
            }
            if (layer->kv_reuse_layer < 0) {
                gl->d_k_cache = (float *)be->mem.alloc(kv_layer_size);
                gl->d_v_cache = (float *)be->mem.alloc(kv_layer_size);
                if (!gl->d_k_cache || !gl->d_v_cache) { fwd_ok = 0; break; }
                /* Zero-init KV cache */
                be->mem.upload(gl->d_k_cache, llm_kv_k, kv_layer_size); /* zeros from alloc_scratch */
                be->mem.upload(gl->d_v_cache, llm_kv_v, kv_layer_size);
            }

            /* Norm weights */
            gl->d_attn_norm = llm_gpu_upload_norm(layer->attn_norm, dim, layer->attn_norm_type);
            gl->d_ffn_norm  = llm_gpu_upload_norm(layer->ffn_norm, dim, layer->ffn_norm_type);
            gl->d_post_attn_norm = llm_gpu_upload_norm(layer->post_attn_norm, dim, GGML_TYPE_F32);
            gl->d_post_ffw_norm  = llm_gpu_upload_norm(layer->post_ffw_norm, dim, GGML_TYPE_F32);

            /* Per-head Q/K norm (Gemma4): weights are [head_dim] shared across heads */
            if (layer->q_norm)
                gl->d_q_norm = llm_gpu_upload_norm(layer->q_norm, lhd, GGML_TYPE_F32);
            if (layer->k_norm)
                gl->d_k_norm = llm_gpu_upload_norm(layer->k_norm, lhd, GGML_TYPE_F32);

            /* ISWA post-norm (Gemma4) */
            if (layer->iswa_post_norm)
                gl->d_iswa_post_norm = llm_gpu_upload_norm(layer->iswa_post_norm, dim, GGML_TYPE_F32);

            if (!gl->d_attn_norm || !gl->d_ffn_norm) { fwd_ok = 0; break; }
        }

        /* If no layers ended up on GPU (e.g. VRAM=0, CUDA broken), fall back
         * to CPU-only mode — avoids GPU decode loop with broken device memory. */
        if (fwd_ok) {
            int any_on_gpu = 0;
            for (int L = 0; L < m->n_layers; L++) {
                if (gpu_ctx.layers[L].on_gpu) { any_on_gpu = 1; break; }
            }
            if (!any_on_gpu) {
                kprintf("[GPU] No layers on GPU (VRAM too small or CUDA unavailable), "
                        "using CPU-only mode\n");
                fwd_ok = 0;
            }
        }
    } else {
        fwd_ok = 0;
    }

    /* Output norm */
    if (fwd_ok) {
        gpu_ctx.d_output_norm = llm_gpu_upload_norm(m->output_norm, dim, m->output_norm_type);
        if (!gpu_ctx.d_output_norm) fwd_ok = 0;
    }

    /* RoPE freq factors (Gemma4: precomputed for full-attention layers) */
    if (fwd_ok && m->rope_freqs) {
        int half_hd = hd / 2;
        gpu_ctx.d_rope_freqs = (float *)be->mem.alloc((uint64_t)half_hd * sizeof(float));
        if (gpu_ctx.d_rope_freqs)
            be->mem.upload(gpu_ctx.d_rope_freqs, m->rope_freqs, (uint64_t)half_hd * sizeof(float));
    }

    /* ISWA GPU scratch (Gemma4: eliminate per-layer CPU round-trip) */
    if (fwd_ok && m->is_gemma4 && m->iswa_n_embd > 0) {
        int iswa_d = m->iswa_n_embd;
        int iswa_total = iswa_d * m->n_layers;
        gpu_ctx.iswa_n_embd = iswa_d;
        gpu_ctx.d_iswa_tmp = (float *)be->mem.alloc((uint64_t)iswa_d * sizeof(float));
        gpu_ctx.d_iswa_per_layer = (float *)be->mem.alloc(
            (uint64_t)iswa_total * sizeof(float));
        /* Projection output buffer (reusable scratch for ISWA precompute) */
        gpu_ctx.d_iswa_proj = (float *)be->mem.alloc(
            (uint64_t)iswa_total * sizeof(float));
        /* Upload ISWA proj norm weights to GPU */
        if (m->iswa_proj_norm) {
            gpu_ctx.d_iswa_proj_norm = (float *)be->mem.alloc(
                (uint64_t)iswa_d * sizeof(float));
            if (gpu_ctx.d_iswa_proj_norm)
                be->mem.upload(gpu_ctx.d_iswa_proj_norm, m->iswa_proj_norm,
                               (uint64_t)iswa_d * sizeof(float));
        }
        if (!gpu_ctx.d_iswa_tmp || !gpu_ctx.d_iswa_per_layer) {
            gpu_ctx.d_iswa_tmp = NULL;
            gpu_ctx.d_iswa_per_layer = NULL;
        }
        /* ISWA token embedding GPU upload disabled — too large, no perf benefit */
    }

    /* Set up shared KV cache pointers for layers that reuse */
    if (fwd_ok) {
        for (int L = 0; L < m->n_layers; L++) {
            llm_layer_t *layer = &m->layers[L];
            if (layer->kv_reuse_layer >= 0 && layer->kv_reuse_layer < m->n_layers) {
                gpu_ctx.layers[L].d_k_cache = gpu_ctx.layers[layer->kv_reuse_layer].d_k_cache;
                gpu_ctx.layers[L].d_v_cache = gpu_ctx.layers[layer->kv_reuse_layer].d_v_cache;
            }
        }
    }

    be->mem.sync();

    if (fwd_ok) {
        gpu_ctx.gpu_fwd = 1; /* GPU traces */
        /* Re-tally total VRAM */
        total = 0;
        for (int i = 0; i < gpu_ctx.map_count; i++) total += gpu_ctx.map[i].size;
        uint64_t used = free_mem - be->get_free_memory(0);
        llm_last_vram_mb = (int)(used / (1024 * 1024));
        kprintf("[GPU] GPU-resident forward pass ready (total VRAM: %d MB)\n", llm_last_vram_mb);
    } else {
        kprintf("[GPU] GPU-resident forward pass not available, using upload/download per GEMV\n");
    }

    return 0;
}

#endif /* ENABLE_CUDA */

#ifndef ENABLE_CUDA
/* CPU-only build stubs — no GPU hardware present, all GPU calls are no-ops */
void llm_gpu_set_compress_mode(int enable)      { (void)enable; }
void llm_gpu_set_attn_compress_mode(int enable) { (void)enable; }
void llm_gpu_upload_ffn_fallback(void)          { }
#endif /* !ENABLE_CUDA */

/* Allocate all inference scratch buffers from the tensor heap.
 * Called once after the model's dimensions are known. */
static int llm_alloc_scratch(const llm_model_t *m)
{
    int dim      = m->dim;
    int ff       = m->ff_dim;
    /* Find max ff_dim across all layers (Gemma4: layers 15+ have doubled FFN) */
    for (int i = 0; i < m->n_layers; i++) {
        if (m->layers[i].ff_dim_layer > ff) ff = m->layers[i].ff_dim_layer;
    }
    int seq      = m->max_seq;
    int vocab    = m->vocab_size;
    int hd       = m->head_dim;
    int q_dim    = m->n_heads * hd;     /* Q/attn output dimension (may differ from dim) */
    int kv_dim   = m->n_kv_heads * hd;  /* K/V dimension per token */
    int64_t kv_total = (int64_t)m->n_layers * seq * m->n_kv_heads * hd; /* int64_t: overflows int for 70B+ models */
    int max_tok  = seq * 2;  /* generous token buffer */

    /* Compute total bytes needed (64-byte aligned per buffer) */
    #define ALIGN64(x) (((x) + 63) & ~63ULL)
    uint64_t total = 0;
    total += ALIGN64((uint64_t)kv_total * sizeof(float)) * 2; /* k+v cache */
    total += ALIGN64((uint64_t)dim * sizeof(float)) * 4;      /* x, xn, last_hs, ffn_d */
    total += ALIGN64((uint64_t)q_dim * sizeof(float)) * 2;    /* q, attn_out */
    total += ALIGN64((uint64_t)kv_dim * sizeof(float)) * 2;   /* k_buf, v_buf */
    total += ALIGN64((uint64_t)hd * sizeof(float));            /* head_buf */
    total += ALIGN64((uint64_t)ff * sizeof(float)) * 2;       /* ffn_g, ffn_u */
    total += ALIGN64((uint64_t)seq * sizeof(float));           /* attn_scores */
    total += ALIGN64((uint64_t)vocab * sizeof(float));         /* logits */
    total += ALIGN64((uint64_t)max_tok * sizeof(int));         /* tokens */
    total += ALIGN64((uint64_t)(hd / 2) * sizeof(float));     /* rope freqs */
    /* ISWA scratch */
    int iswa_n_ = m->iswa_n_embd > 0 ? m->iswa_n_embd : 1;
    total += ALIGN64((uint64_t)(iswa_n_ * m->n_layers) * sizeof(float)); /* per_layer */
    total += ALIGN64((uint64_t)iswa_n_ * sizeof(float));                  /* tmp */
    total += ALIGN64((uint64_t)m->n_layers * dim * sizeof(float));        /* depth hist */
    total += ALIGN64((uint64_t)m->n_layers * sizeof(float));              /* depth scores */

    kprintf("[LLM] Allocating %lu MB scratch arena\n",
            (unsigned long)(total / (1024 * 1024)));

    uint8_t *arena = (uint8_t *)tensor_alloc(total);
    if (!arena) {
        kprintf("[LLM] ERROR: Failed to allocate %lu MB for inference buffers\n",
                (unsigned long)(total / (1024 * 1024)));
        return -1;
    }
    kmemset(arena, 0, total);

    /* Carve out buffers from the arena */
    #define ARENA_NEXT(ptr, type, count) do { \
        ptr = (type *)arena; \
        arena += ALIGN64((uint64_t)(count) * sizeof(type)); \
    } while (0)

    ARENA_NEXT(llm_kv_k,        float, kv_total);
    ARENA_NEXT(llm_kv_v,        float, kv_total);
    ARENA_NEXT(llm_x,           float, dim);
    ARENA_NEXT(llm_xn,          float, dim);
    ARENA_NEXT(llm_last_hs,     float, dim);
    ARENA_NEXT(llm_q,           float, q_dim);
    ARENA_NEXT(llm_k_buf,       float, kv_dim);
    ARENA_NEXT(llm_v_buf,       float, kv_dim);
    ARENA_NEXT(llm_attn_out,    float, q_dim);
    ARENA_NEXT(llm_ffn_g,       float, ff);
    ARENA_NEXT(llm_ffn_u,       float, ff);
    ARENA_NEXT(llm_ffn_d,       float, dim);
    ARENA_NEXT(llm_head_buf,    float, hd);
    ARENA_NEXT(llm_attn_scores, float, seq);
    ARENA_NEXT(llm_logits,      float, vocab);
    ARENA_NEXT(llm_tokens,      int,   max_tok);
    ARENA_NEXT(llm_rope_freqs_buf, float, hd / 2);
    /* ISWA scratch (Gemma4) */
    int iswa_n = m->iswa_n_embd > 0 ? m->iswa_n_embd : 1;
    int iswa_total = iswa_n * m->n_layers;
    ARENA_NEXT(llm_iswa_per_layer, float, iswa_total);
    ARENA_NEXT(llm_iswa_tmp,       float, iswa_n);
    ARENA_NEXT(llm_depth_hist,     float, m->n_layers * dim);
    ARENA_NEXT(llm_depth_scores,   float, m->n_layers);

    #undef ARENA_NEXT
    #undef ALIGN64

    /* Cache sizes for runtime bounds checks */
    llm_alloc_dim      = dim;
    llm_alloc_ff       = ff;
    llm_alloc_seq      = seq;
    llm_alloc_vocab    = vocab;
    llm_alloc_tokens   = max_tok;
    llm_alloc_kv_floats = kv_total;

    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Float16 (half precision) conversion                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* Positive/negative zero */
            float f;
            kmemcpy(&f, &sign, 4);
            return f;
        }
        /* Denormalized — normalize it */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        /* Inf / NaN */
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        float f;
        kmemcpy(&f, &bits, 4);
        return f;
    }

    exp = exp + 127 - 15;
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f;
    kmemcpy(&f, &bits, 4);
    return f;
}

/* BF16 to FP32: just shift left by 16 (BF16 is upper 16 bits of FP32) */
static float bf16_to_fp32(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    kmemcpy(&f, &bits, 4);
    return f;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Math helpers                                                                */
/* ─────────────────────────────────────────────────────────────────────────── */

static float llm_fabsf(float x) { return x < 0 ? -x : x; }

static float llm_sqrtf(float x)
{
    float r;
#if defined(__aarch64__)
    __asm__("fsqrt %s0, %s1" : "=w"(r) : "w"(x));
#else
    __asm__("sqrtss %1, %0" : "=x"(r) : "x"(x));
#endif
    return r;
}

static float llm_expf(float x)
{
    if (x > 88.7f) return 3.4028235e38f;
    if (x < -88.7f) return 0.0f;
    /* Range reduction: exp(x) = 2^n * exp(r), r in [-ln2/2, ln2/2]
     * n = round(x / ln2), r = x - n * ln2 */
    float n = (float)(int)(x * 1.4426950408f + (x >= 0 ? 0.5f : -0.5f)); /* round(x/ln2) */
    float r = x - n * 0.693145751953125f;   /* Cody-Waite high part */
    r -= n * 1.428606765330187e-06f;         /* Cody-Waite low part  */
    /* Minimax polynomial for exp(r), r in [-ln2/2, ln2/2], max err < 2e-7 */
    float r2 = r * r;
    float r3 = r2 * r;
    float p = 1.0f + r + r2 * 0.5f + r3 * 0.16666666f
            + r2 * r2 * 0.041666666f + r2 * r3 * 0.008333333f
            + r3 * r3 * 0.001388889f;
    /* Reconstruct: multiply by 2^n via IEEE 754 bit manipulation */
    int ni = (int)n;
    union { float f; unsigned int u; } scale;
    scale.u = (unsigned int)(127 + ni) << 23;
    return p * scale.f;
}

/* Accurate sinf/cosf using Cody-Waite range reduction + minimax polynomials.
 * Reduces to [-pi/4, pi/4] using quadrant decomposition where Taylor series
 * converges rapidly (max |x| = 0.785, so x^8/8! ~ 5e-7). */
static float llm_sinf(float x)
{
    /* Range reduction to [-pi, pi] */
    float pi  = 3.14159265358979f;
    float hpi = 1.57079632679490f;
    float itp = 0.15915494309190f; /* 1/(2*pi) */
    x = x - (float)(int)(x * itp + (x >= 0 ? 0.5f : -0.5f)) * (2.0f * pi);

    /* Quadrant decomposition: reduce to [-pi/4, pi/4] */
    int negate = 0;
    if (x < 0) { x = -x; negate = 1; }
    /* x now in [0, pi] */
    int use_cos = 0;
    if (x > hpi) { x = pi - x; }       /* mirror: sin(pi-x) = sin(x) */
    if (x > hpi * 0.5f) { x = hpi - x; use_cos = 1; } /* sin(pi/2-x) = cos(x) */

    float x2 = x * x;
    float r;
    if (use_cos) {
        /* cos(x) = 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320 */
        r = 1.0f - x2 * (0.5f - x2 * (0.041666666f - x2 * (0.001388889f - x2 * 2.48016e-05f)));
    } else {
        /* sin(x) = x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880 */
        r = x * (1.0f - x2 * (0.16666667f - x2 * (0.008333333f - x2 * (0.000198413f - x2 * 2.7558e-06f))));
    }
    return negate ? -r : r;
}

static float llm_cosf(float x)
{
    /* cos(x) = sin(x + pi/2) */
    return llm_sinf(x + 1.57079632679490f);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Quantized Tensor Operations (GGML format)                                  */
/*                                                                             */
/*  Q4_0: 32 elements → 2 bytes (fp16 scale) + 16 bytes (nibbles) = 18 bytes  */
/*  Q8_0: 32 elements → 2 bytes (fp16 scale) + 32 bytes (int8)    = 34 bytes  */
/*  F16:  1 element  → 2 bytes (half float)                                    */
/*  F32:  1 element  → 4 bytes (float)                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

/* GGML Q4_0 block: {fp16 scale, uint8[16]} = 18 bytes for 32 elements */
typedef struct { uint16_t d; uint8_t qs[16]; } __attribute__((packed)) ggml_q4_0_t;

/* GGML Q4_1 block: {fp16 scale, fp16 min, uint8[16]} = 20 bytes for 32 elements
 * Dequant: val = d * nibble + m  (unsigned nibble 0-15, m is minimum value) */
typedef struct { uint16_t d; uint16_t m; uint8_t qs[16]; } __attribute__((packed)) ggml_q4_1_t;

/* GGML Q8_0 block: {fp16 scale, int8[32]}  = 34 bytes for 32 elements */
typedef struct { uint16_t d; int8_t  qs[32]; } __attribute__((packed)) ggml_q8_0_t;

/* GGML Q6_K super-block: 256 elements in 210 bytes
 * ql[128] — lower 4 bits of each 6-bit quant (2 quants per byte for first half,
 *           then 2 quants per byte for second half)
 * qh[64]  — upper 2 bits of each quant (4 quants packed per byte)
 * scales[16] — int8 scale per 16-element sub-block
 * d — fp16 super-block scale
 */
typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
} __attribute__((packed)) ggml_q6_k_t;

/* Dot product: Q4_0 block · float[32] — SSE2 optimized */
static float q4_0_dot32(const ggml_q4_0_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    const uint8_t *qs = block->qs;

#ifndef __aarch64__
    /* SSE2: standard GGML layout — lo nibbles map to x[0..15], hi to x[16..31] */
    v4f acc_lo0 = {0, 0, 0, 0};
    v4f acc_lo1 = {0, 0, 0, 0};
    v4f acc_hi0 = {0, 0, 0, 0};
    v4f acc_hi1 = {0, 0, 0, 0};

    for (int j = 0; j < 16; j += 4) {
        uint8_t b0 = qs[j], b1 = qs[j+1], b2 = qs[j+2], b3 = qs[j+3];
        v4f qlo = {(float)((int)(b0 & 0xF) - 8), (float)((int)(b1 & 0xF) - 8),
                   (float)((int)(b2 & 0xF) - 8), (float)((int)(b3 & 0xF) - 8)};
        v4f qhi = {(float)((int)(b0 >> 4) - 8), (float)((int)(b1 >> 4) - 8),
                   (float)((int)(b2 >> 4) - 8), (float)((int)(b3 >> 4) - 8)};
        v4f xlo = *(const v4f *)(x + j);       /* x[j..j+3] */
        v4f xhi = *(const v4f *)(x + j + 16);  /* x[j+16..j+19] */
        acc_lo0 += qlo * xlo;
        acc_hi0 += qhi * xhi;
    }

    v4f acc = acc_lo0 + acc_hi0;
    union { v4f v; float f[4]; } u = { .v = acc };
    return (u.f[0] + u.f[1] + u.f[2] + u.f[3]) * d;
#else
    /* NEON: process 8 packed bytes per iteration = 16 lo + 16 hi Q4 values */
    uint8x8_t mask4  = vdup_n_u8(0x0F);
    int8x8_t  off8   = vdup_n_s8(8);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (int j = 0; j < 16; j += 8) {
        uint8x8_t packed = vld1_u8(qs + j);
        int8x8_t  qlo8   = vsub_s8(vreinterpret_s8_u8(vand_u8(packed, mask4)), off8);
        int8x8_t  qhi8   = vsub_s8(vreinterpret_s8_u8(vshr_n_u8(packed, 4)), off8);
        int16x8_t qlo16  = vmovl_s8(qlo8);
        int16x8_t qhi16  = vmovl_s8(qhi8);

        float32x4_t qlo_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo16)));
        float32x4_t qlo_f1 = vcvtq_f32_s32(vmovl_high_s16(qlo16));
        float32x4_t qhi_f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi16)));
        float32x4_t qhi_f1 = vcvtq_f32_s32(vmovl_high_s16(qhi16));

        float32x4_t xlo_0 = vld1q_f32(x + j);
        float32x4_t xlo_1 = vld1q_f32(x + j + 4);
        float32x4_t xhi_0 = vld1q_f32(x + j + 16);
        float32x4_t xhi_1 = vld1q_f32(x + j + 20);

        acc0 = vmlaq_f32(acc0, qlo_f0, xlo_0);
        acc1 = vmlaq_f32(acc1, qlo_f1, xlo_1);
        acc2 = vmlaq_f32(acc2, qhi_f0, xhi_0);
        acc3 = vmlaq_f32(acc3, qhi_f1, xhi_1);
    }
    float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
    return vaddvq_f32(acc) * d;
#endif
}

/* Dot product: Q4_1 block · float[32]
 * Each nibble is unsigned (0-15), dequant: d * nibble + m
 * dot = sum_i (d * nibble_i + m) * x_i = d * sum(nibble_i * x_i) + m * sum(x_i) */
static float q4_1_dot32(const ggml_q4_1_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    float m = fp16_to_fp32(block->m);
    const uint8_t *qs = block->qs;

#ifndef __aarch64__
    /* Standard GGML layout: lo nibbles → x[0..15], hi nibbles → x[16..31] */
    v4f qxacc_lo = {0, 0, 0, 0};
    v4f qxacc_hi = {0, 0, 0, 0};
    v4f xacc_lo = {0, 0, 0, 0};
    v4f xacc_hi = {0, 0, 0, 0};

    for (int j = 0; j < 16; j += 4) {
        uint8_t b0 = qs[j], b1 = qs[j+1], b2 = qs[j+2], b3 = qs[j+3];
        v4f qlo = {(float)(b0 & 0xF), (float)(b1 & 0xF),
                   (float)(b2 & 0xF), (float)(b3 & 0xF)};
        v4f qhi = {(float)(b0 >> 4), (float)(b1 >> 4),
                   (float)(b2 >> 4), (float)(b3 >> 4)};
        v4f xlo = *(const v4f *)(x + j);       /* x[j..j+3] */
        v4f xhi = *(const v4f *)(x + j + 16);  /* x[j+16..j+19] */
        qxacc_lo += qlo * xlo;
        qxacc_hi += qhi * xhi;
        xacc_lo += xlo;
        xacc_hi += xhi;
    }

    v4f qx = qxacc_lo + qxacc_hi;
    v4f xs = xacc_lo + xacc_hi;
    union { v4f v; float f[4]; } uq = { .v = qx };
    union { v4f v; float f[4]; } ux = { .v = xs };
    float qxsum = uq.f[0] + uq.f[1] + uq.f[2] + uq.f[3];
    float xsum  = ux.f[0] + ux.f[1] + ux.f[2] + ux.f[3];
    return d * qxsum + m * xsum;
#else
    float qxsum = 0.0f, xsum = 0.0f;
    for (int j = 0; j < 16; j++) {
        float lo = (float)(qs[j] & 0xF);
        float hi = (float)(qs[j] >> 4);
        qxsum += lo * x[j] + hi * x[j + 16];
        xsum  += x[j] + x[j + 16];
    }
    return d * qxsum + m * xsum;
#endif
}

/* Dot product: Q8_0 block · float[32] — SSE2 optimized */
static float q8_0_dot32(const ggml_q8_0_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    const int8_t *qs = block->qs;

#ifndef __aarch64__
    /* SSE2: dual v4f accumulators, 4 iterations of 8 elements */
    v4f acc0 = {0, 0, 0, 0};
    v4f acc1 = {0, 0, 0, 0};

    for (int j = 0; j < 32; j += 8) {
        v4f q0 = {(float)qs[j], (float)qs[j+1], (float)qs[j+2], (float)qs[j+3]};
        v4f q1 = {(float)qs[j+4], (float)qs[j+5], (float)qs[j+6], (float)qs[j+7]};
        v4f x0 = *(const v4f *)(x + j);
        v4f x1 = *(const v4f *)(x + j + 4);
        acc0 += q0 * x0;
        acc1 += q1 * x1;
    }

    v4f acc = acc0 + acc1;
    union { v4f v; float f[4]; } u = { .v = acc };
    return (u.f[0] + u.f[1] + u.f[2] + u.f[3]) * d;
#else
    /* NEON: process 8 int8 weights at a time → sign-extend → fmla */
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    for (int j = 0; j < 32; j += 8) {
        int8x8_t  q8  = vld1_s8(qs + j);
        int16x8_t q16 = vmovl_s8(q8);
        float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        float32x4_t qf1 = vcvtq_f32_s32(vmovl_high_s16(q16));
        float32x4_t x0  = vld1q_f32(x + j);
        float32x4_t x1  = vld1q_f32(x + j + 4);
        acc0 = vmlaq_f32(acc0, qf0, x0);
        acc1 = vmlaq_f32(acc1, qf1, x1);
    }
    float32x4_t acc = vaddq_f32(acc0, acc1);
    return vaddvq_f32(acc) * d;
#endif
}

/* Q6_K super-block dot product: 256 elements
 * Dequant: val = d * scales[sub] * q6 where q6 is 6-bit signed (-32..31)
 * Lower 4 bits in ql[], upper 2 bits in qh[]
 * Layout follows ggml reference: 2 halves of 128, each with 4 groups of 32.
 * ql[0..63]: lower nibbles for quants at offsets 0..31 (lo nibble) and 64..95 (hi nibble)
 *            in the first half, ql[32..63] for offsets 32..63 and 96..127
 * ql[64..127]: same pattern for second half (quants 128..255)
 * qh[0..31]: 2-bit high parts for first half (4 quants per byte, shifts 0/2/4/6)
 * qh[32..63]: same for second half
 * scales[0..15]: int8 scales, 2 per group of 32: sc[0]=grp0, sc[2]=grp1, etc. */
static float q6_k_dot256(const ggml_q6_k_t *block, const float *x)
{
    float d = fp16_to_fp32(block->d);
    float sum = 0.0f;
    const uint8_t *ql = block->ql;
    const uint8_t *qh = block->qh;
    const int8_t *sc = block->scales;

    /* Process in 2 halves of 128 quants each */
    for (int half = 0; half < 2; half++) {
        const uint8_t *ql_h = ql + half * 64;
        const uint8_t *qh_h = qh + half * 32;
        const int8_t *sc_h = sc + half * 8;
        const float *x_h = x + half * 128;

        for (int l = 0; l < 32; l++) {
            /* 4 quants per iteration from the 4 groups of 32 in this half */
            int q0 = (int)( ql_h[l]      & 0xF) | (int)(((qh_h[l] >> 0) & 3) << 4);
            int q1 = (int)( ql_h[l + 32] & 0xF) | (int)(((qh_h[l] >> 2) & 3) << 4);
            int q2 = (int)( ql_h[l]      >>  4) | (int)(((qh_h[l] >> 4) & 3) << 4);
            int q3 = (int)( ql_h[l + 32] >>  4) | (int)(((qh_h[l] >> 6) & 3) << 4);

            /* Scale mapping: each group of 32 has 2 sub-blocks of 16.
             * sc_h[0] for l<16 in group 0, sc_h[1] for l>=16 in group 0, etc. */
            int si0 = l / 16;        /* 0 or 1 → scale index within group 0 */
            sum += (float)sc_h[0 + si0] * (float)(q0 - 32) * x_h[l];
            sum += (float)sc_h[2 + si0] * (float)(q1 - 32) * x_h[l + 32];
            sum += (float)sc_h[4 + si0] * (float)(q2 - 32) * x_h[l + 64];
            sum += (float)sc_h[6 + si0] * (float)(q3 - 32) * x_h[l + 96];
        }
    }

    return sum * d;
}

/* Generic vector dot product: quantized weight row · float input
 * Returns the dot product of n elements.
 * The weight pointer must be to the start of the row's block data. */
static float llm_vec_dot(const void *weight, const float *x, int n, ggml_type_t type)
{
    float sum = 0.0f;
    int nb = n / 32;  /* number of blocks (both Q4_0 and Q8_0 use block_size=32) */

    switch (type) {
    case GGML_TYPE_Q4_0: {
        const ggml_q4_0_t *blocks = (const ggml_q4_0_t *)weight;
        for (int b = 0; b < nb; b++)
            sum += q4_0_dot32(&blocks[b], x + b * 32);
        break;
    }
    case GGML_TYPE_Q4_1: {
        const ggml_q4_1_t *blocks = (const ggml_q4_1_t *)weight;
        for (int b = 0; b < nb; b++)
            sum += q4_1_dot32(&blocks[b], x + b * 32);
        break;
    }
    case GGML_TYPE_Q8_0: {
        const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)weight;
        for (int b = 0; b < nb; b++)
            sum += q8_0_dot32(&blocks[b], x + b * 32);
        break;
    }
    case GGML_TYPE_F16: {
        const uint16_t *f16 = (const uint16_t *)weight;
#ifndef __aarch64__
        v4f acc = {0, 0, 0, 0};
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            v4f w = {fp16_to_fp32(f16[i]), fp16_to_fp32(f16[i+1]),
                     fp16_to_fp32(f16[i+2]), fp16_to_fp32(f16[i+3])};
            v4f vx = *(const v4f *)(x + i);
            acc += w * vx;
        }
        union { v4f v; float f[4]; } u = { .v = acc };
        sum += u.f[0] + u.f[1] + u.f[2] + u.f[3];
        for (; i < n; i++)
            sum += fp16_to_fp32(f16[i]) * x[i];
#else
        /* NEON F16: convert pairs via vcvt_f32_f16 (requires float16x4_t) */
        float32x4_t vacc = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float16x4_t h = vld1_f16((const __fp16 *)(f16 + i));
            float32x4_t wv = vcvt_f32_f16(h);
            float32x4_t xv = vld1q_f32(x + i);
            vacc = vmlaq_f32(vacc, wv, xv);
        }
        sum += vaddvq_f32(vacc);
        for (; i < n; i++)
            sum += fp16_to_fp32(f16[i]) * x[i];
#endif
        break;
    }
    case GGML_TYPE_F32: {
        const float *f32 = (const float *)weight;
#ifndef __aarch64__
        v4f acc0 = {0, 0, 0, 0};
        v4f acc1 = {0, 0, 0, 0};
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            v4f w0 = *(const v4f *)(f32 + i);
            v4f w1 = *(const v4f *)(f32 + i + 4);
            v4f x0 = *(const v4f *)(x + i);
            v4f x1 = *(const v4f *)(x + i + 4);
            acc0 += w0 * x0;
            acc1 += w1 * x1;
        }
        v4f a = acc0 + acc1;
        union { v4f v; float f[4]; } u = { .v = a };
        sum += u.f[0] + u.f[1] + u.f[2] + u.f[3];
        for (; i < n; i++)
            sum += f32[i] * x[i];
#else
        /* NEON F32: dual accumulators, 8 elements per iteration */
        float32x4_t vacc0 = vdupq_n_f32(0.0f);
        float32x4_t vacc1 = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            float32x4_t w0 = vld1q_f32(f32 + i);
            float32x4_t w1 = vld1q_f32(f32 + i + 4);
            float32x4_t x0 = vld1q_f32(x + i);
            float32x4_t x1 = vld1q_f32(x + i + 4);
            vacc0 = vmlaq_f32(vacc0, w0, x0);
            vacc1 = vmlaq_f32(vacc1, w1, x1);
        }
        sum += vaddvq_f32(vaddq_f32(vacc0, vacc1));
        for (; i < n; i++)
            sum += f32[i] * x[i];
#endif
        break;
    }
    case GGML_TYPE_Q6_K: {
        const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)weight;
        int nsb = n / 256;  /* Q6_K uses super-blocks of 256 elements */
        for (int b = 0; b < nsb; b++)
            sum += q6_k_dot256(&blocks[b], x + b * 256);
        break;
    }
    case GGML_TYPE_Q4_K: {
        /* Q4_K (K-quant): 256 elements per super-block, 8 sub-blocks of 32.
         * Block layout: d(fp16) + dmin(fp16) + scales[12] + qs[128] = 144 bytes.
         * value = d * scale_j * nibble - dmin * min_j */
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh, dmh;
            kmemcpy(&dh,  p,     2);
            kmemcpy(&dmh, p + 2, 2);
            float d    = fp16_to_fp32(dh);
            float dmin = fp16_to_fp32(dmh);
            const uint8_t *scales = p + 4;
            const uint8_t *qs     = p + 16;  /* 4 + 12 = 16 */
            p += 144;

            const float *xb = x + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                /* Extract scale/min for two sub-blocks (is, is+1) */
                uint8_t sc0, m0, sc1, m1;
                /* get_scale_min_k4 for sub-block is+0 */
                if (is < 4) {
                    sc0 = scales[is] & 63;
                    m0  = scales[is + 4] & 63;
                } else {
                    sc0 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                    m0  = (scales[is + 4] >> 4)  | ((scales[is + 0] >> 6) << 4);
                }
                /* get_scale_min_k4 for sub-block is+1 */
                int is1 = is + 1;
                if (is1 < 4) {
                    sc1 = scales[is1] & 63;
                    m1  = scales[is1 + 4] & 63;
                } else {
                    sc1 = (scales[is1 + 4] & 0xF) | ((scales[is1 - 4] >> 6) << 4);
                    m1  = (scales[is1 + 4] >> 4)  | ((scales[is1 + 0] >> 6) << 4);
                }
                float d1 = d * (float)sc0;  float m1f = dmin * (float)m0;
                float d2 = d * (float)sc1;  float m2f = dmin * (float)m1;

                /* First 32 from lower nibbles, second 32 from upper nibbles */
                const uint8_t *qj = qs + (j / 2);
                for (int l = 0; l < 32; l++)
                    sum += xb[j + l]      * (d1 * (float)(qj[l] & 0xF) - m1f);
                for (int l = 0; l < 32; l++)
                    sum += xb[j + 32 + l] * (d2 * (float)(qj[l] >> 4)  - m2f);
                is += 2;
            }
        }
        break;
    }
    case GGML_TYPE_BF16: {
        const uint16_t *bf = (const uint16_t *)weight;
        for (int i = 0; i < n; i++)
            sum += bf16_to_fp32(bf[i]) * x[i];
        break;
    }
    case GGML_TYPE_IQ4_NL: {
        /* IQ4_NL: 32 elements/block, 18 bytes. Layout: d(fp16) + qs[16].
         * Values decoded via non-linear lookup table (16 levels). */
        static const int8_t iq4nl_tbl[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
        };
        const uint8_t *p = (const uint8_t *)weight;
        int nb = n / 32;
        for (int b = 0; b < nb; b++) {
            uint16_t dh;
            kmemcpy(&dh, p, 2);
            float d = fp16_to_fp32(dh);
            const uint8_t *qs = p + 2;
            const float *xb = x + b * 32;
            for (int l = 0; l < 16; l++) {
                sum += xb[2*l + 0] * d * (float)iq4nl_tbl[qs[l] & 0xF];
                sum += xb[2*l + 1] * d * (float)iq4nl_tbl[qs[l] >> 4];
            }
            p += 18;
        }
        break;
    }
    case GGML_TYPE_IQ4_XS: {
        /* IQ4_XS: 256 elements/super-block, 136 bytes.
         * Layout: d(fp16) + scales_h(uint16) + scales_l[4] + qs[128].
         * 8 sub-blocks of 32, 6-bit scales, same IQ4NL lookup table. */
        static const int8_t iq4xs_tbl[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
        };
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh, scales_h_u;
            kmemcpy(&dh,       p,     2);
            kmemcpy(&scales_h_u, p + 2, 2);
            float d = fp16_to_fp32(dh);
            const uint8_t *scales_l = p + 4;   /* [4] */
            const uint8_t *qs       = p + 8;   /* [128] */
            p += 136;
            const float *xb = x + b * 256;
            for (int ib = 0; ib < 8; ib++) {
                uint8_t ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF)
                             | (((scales_h_u >> (2 * ib)) & 3) << 4);
                float dl = d * (float)((int)ls - 32);
                const uint8_t *qb = qs + ib * 16;
                const float *xsb = xb + ib * 32;
                for (int j = 0; j < 16; j++) {
                    sum += xsb[2*j + 0] * dl * (float)iq4xs_tbl[qb[j] & 0xF];
                    sum += xsb[2*j + 1] * dl * (float)iq4xs_tbl[qb[j] >> 4];
                }
            }
        }
        break;
    }
    case GGML_TYPE_Q2_K: {
        /* Q2_K: 256 elements/super-block, 84 bytes.
         * Layout: scales[16](nibble-packed) + qs[64] + d(fp16) + dmin(fp16).
         * Matches ggml dequantize_row_q2_K exactly. */
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            const uint8_t *scales = p;      /* [16]: low nibble=d_scale, high=min_scale */
            const uint8_t *qs     = p + 16; /* [64] */
            uint16_t dh, dmh;
            kmemcpy(&dh,  p + 80, 2);
            kmemcpy(&dmh, p + 82, 2);
            float d    = fp16_to_fp32(dh);
            float dmin = fp16_to_fp32(dmh);
            p += 84;
            const float *xb = x + b * 256;
            int y_idx = 0, is = 0;
            const uint8_t *qb = qs;
            for (int half = 0; half < 2; half++) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0xF);
                    float ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        sum += xb[y_idx++] * (dl * (float)((qb[l] >> shift) & 3) - ml);
                    sc = scales[is++];
                    dl = d * (float)(sc & 0xF);
                    ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        sum += xb[y_idx++] * (dl * (float)((qb[l + 16] >> shift) & 3) - ml);
                    shift += 2;
                }
                qb += 32;
            }
        }
        break;
    }
    case GGML_TYPE_Q3_K: {
        /* Q3_K: 256 elements/super-block, 110 bytes.
         * Layout: hmask[32] + qs[64] + scales[12] + d(fp16).
         * hmask provides bit 2 of each 3-bit quant. */
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            const uint8_t *hmask  = p;        /* [32]  high bits */
            const uint8_t *qs     = p + 32;   /* [64]  low 2 bits */
            const uint8_t *scales = p + 96;   /* [12]  6-bit scales */
            uint16_t dh;
            kmemcpy(&dh, p + 108, 2);
            float d = fp16_to_fp32(dh);
            p += 110;
            const float *xb = x + b * 256;
            /* 8 sub-blocks of 32 elements */
            for (int sb = 0; sb < 8; sb++) {
                /* Extract 6-bit scale for this sub-block (packed like Q4_K) */
                int is = sb;
                uint8_t sc;
                if (is < 4) {
                    sc = (scales[is] & 0xF) | ((scales[is + 8] & 3) << 4);
                } else {
                    sc = (scales[is + 4] >> 4) | ((scales[is + 4] & 3) << 4);
                }
                /* scale is stored as (sc-32) offset: actual = sc - 32 */
                float d_sc = d * (float)((int)sc - 32);
                int base = sb * 32;
                for (int l = 0; l < 32; l++) {
                    int idx = base + l;
                    int qlo = (qs[idx >> 2] >> (2 * (idx & 3))) & 3;
                    int qhi = (hmask[idx >> 3] >> (idx & 7)) & 1;
                    int q   = qlo | (qhi << 2);  /* 3-bit value 0..7 */
                    sum += xb[idx] * d_sc * (float)(q - 4);
                }
            }
        }
        break;
    }
    case GGML_TYPE_Q5_K: {
        /* Q5_K: 256 elements/super-block, 176 bytes. */
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh, dmh;
            kmemcpy(&dh,  p,     2);
            kmemcpy(&dmh, p + 2, 2);
            float d    = fp16_to_fp32(dh);
            float dmin = fp16_to_fp32(dmh);
            const uint8_t *scales = p + 4;   /* [12] */
            const uint8_t *qh     = p + 16;  /* [32] */
            const uint8_t *qs     = p + 48;  /* [128] */
            p += 176;
            const float *xb = x + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                if (is < 4) {
                    sc0 = scales[is] & 63;      m0 = scales[is + 4] & 63;
                } else {
                    sc0 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                    m0  = (scales[is + 4] >> 4)  | ((scales[is + 0] >> 6) << 4);
                }
                int is1 = is + 1;
                if (is1 < 4) {
                    sc1 = scales[is1] & 63;     m1 = scales[is1 + 4] & 63;
                } else {
                    sc1 = (scales[is1 + 4] & 0xF) | ((scales[is1 - 4] >> 6) << 4);
                    m1  = (scales[is1 + 4] >> 4)  | ((scales[is1 + 0] >> 6) << 4);
                }
                float d1 = d * (float)sc0;  float m1f = dmin * (float)m0;
                float d2 = d * (float)sc1;  float m2f = dmin * (float)m1;
                const uint8_t *qj = qs + (j / 2);
                /* bit 4 from qh: byte (j/8 + l/8), bit (j/32)*4 + bit within */
                for (int l = 0; l < 32; l++) {
                    int pos = j + l;
                    int hi  = (qh[pos >> 3] >> ((pos >> 5) * 4 + (pos & 7) / 8)) & 1;
                    int q   = (qj[l] & 0xF) | (hi << 4);
                    sum += xb[j + l] * (d1 * (float)q - m1f);
                }
                for (int l = 0; l < 32; l++) {
                    int pos = j + 32 + l;
                    int hi  = (qh[pos >> 3] >> ((pos >> 5) * 4 + (pos & 7) / 8)) & 1;
                    int q   = (qj[l] >> 4) | (hi << 4);
                    sum += xb[j + 32 + l] * (d2 * (float)q - m2f);
                }
                is += 2;
            }
        }
        break;
    }
    case GGML_TYPE_IQ2_XS: {
        /* IQ2_XS: 256 elements/super-block, 74 bytes.
         * Layout: d(fp16) + qs[32](uint16_t) + scales[8](uint8_t).
         * scales[] are packed 2 nibbles per byte: group ib32 uses nibble
         *   (scales[ib32/2] >> 4*(ib32%2)) & 0xf.
         * Each qs[k]: bits[8:0]=grid_idx (9-bit), bits[15:9]=7 sign bits.
         * kgrid[idx] encodes 8 x 2-bit indices; map v -> {8,25,43,60}.
         * 8th sign from even-parity: s8 = s7|(popcount(s7)&1)<<7. */
        static const uint8_t  iq2xs_vals[4] = {1, 3, 5, 7};
        static const uint16_t kgrid[512] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
          260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
          352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
          640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
         1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
         1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
         2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
         2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
         4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
         4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
         5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
         5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
         8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
         8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
        10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
        16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
        16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
        16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
        17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
        18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
        20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
        21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
        22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
        24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
        32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
        33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
        33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
        35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
        37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
        40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
        42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
        };
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh;
            kmemcpy(&dh, p, 2);
            float d = fp16_to_fp32(dh);
            const uint16_t *qs = (const uint16_t *)(p + 2);
            const uint8_t  *sc = p + 66;
            p += 74;
            const float *xb = x + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                /* Two scale nibbles per 32-element group: low for l=0,1; high for l=2,3 */
                float dl0 = d * (0.5f + (sc[ib32] & 0xf)) * 0.25f;
                float dl1 = d * (0.5f + (sc[ib32] >> 4))  * 0.25f;
                for (int l = 0; l < 4; l++) {
                    float dl = (l < 2) ? dl0 : dl1;
                    uint16_t qv = qs[ib32 * 4 + l];
                    uint16_t gi = qv & 511;
                    uint8_t  s7 = (uint8_t)(qv >> 9);
                    uint8_t  s8 = s7 | (uint8_t)((__builtin_popcount(s7) & 1) << 7);
                    uint16_t gv = kgrid[gi];
                    const float *xs = xb + ib32 * 32 + l * 8;
                    for (int k = 0; k < 8; k++) {
                        float gk = (float)iq2xs_vals[(gv >> (2 * k)) & 3];
                        if (s8 & (1u << k)) gk = -gk;
                        sum += xs[k] * dl * gk;
                    }
                }
            }
        }
        break;
    }
    case GGML_TYPE_IQ2_XXS: {
        /* IQ2_XXS: 256 elements/super-block, 66 bytes.
         * Layout: d(fp16) + uint16_t qs[32] (=64 bytes).
         * Each 32-element sub-block (ib32=0..7): 8 bytes read as 2 uint32:
         *   aux32[0]: bytes 0-3 = 4 grid indices (8-bit each, one per 8-elem group)
         *   aux32[1]: bits[27:0]=four 7-bit sign groups; bits[31:28]=scale nibble.
         * Scale: db = d * (0.5 + nibble) * 0.25.
         * Grid: kgrid_iq2xxs[idx] = uint16_t, bits[2k..2k+1]=l -> val=2l+1 ∈ {1,3,5,7}.
         * 8th sign via even parity: s8 = s7 | (popcount(s7)&1)<<7. */
        static const uint16_t kgrid_iq2xxs[256] = {
                0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
              100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
             1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
             1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
             2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
             4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
             5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
             8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
            10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
            16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
            17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
            20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
            22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
            25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
            33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
            37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
        };
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh;
            kmemcpy(&dh, p, 2);
            float d = fp16_to_fp32(dh);
            const uint8_t *qs = p + 2;
            p += 66;
            const float *xb = x + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                uint32_t aux32[2];
                kmemcpy(aux32, qs + 8 * ib32, 8);
                const uint8_t *a8 = (const uint8_t *)aux32;
                float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; l++) {
                    uint16_t gv = kgrid_iq2xxs[a8[l]];
                    uint8_t  s7 = (uint8_t)((aux32[1] >> (7 * l)) & 127);
                    uint8_t  s8 = s7 | (uint8_t)((__builtin_popcount(s7) & 1) << 7);
                    const float *xs = xb + ib32 * 32 + l * 8;
                    for (int k = 0; k < 8; k++) {
                        float w = (float)(2 * ((gv >> (2 * k)) & 3) + 1);
                        if (s8 & (1u << k)) w = -w;
                        sum += xs[k] * db * w;
                    }
                }
            }
        }
        break;
    }
    case GGML_TYPE_IQ3_XXS: {
        /* IQ3_XXS: 256 elements/super-block, 98 bytes.
         * Layout: d(fp16) + qs[96](uint8_t).
         * qs[0..63]: 8 grid indices per 32-element sub-block (×8 sub-blocks).
         * qs[64..95]: 4-byte scale+sign per sub-block (scales_and_signs[8]):
         *   bits[27:0] = four 7-bit sign groups; bits[31:28] = scale nibble.
         * Scale: db = d * (0.5 + nibble) * 0.5.
         * Grid: kgrid_iq3xxs[idx], bits[3k..3k+2]=l -> val=2l+1 ∈ {1..15}. */
        static const uint16_t kgrid_iq3xxs[256] = {
               0,    2,    4,    9,   11,   15,   16,   18,   25,   34,   59,   61,   65,   67,   72,   74,
              81,   85,   88,   90,   97,  108,  120,  128,  130,  132,  137,  144,  146,  153,  155,  159,
             169,  175,  189,  193,  199,  200,  202,  213,  248,  267,  287,  292,  303,  315,  317,  321,
             327,  346,  362,  413,  436,  456,  460,  462,  483,  497,  513,  515,  520,  522,  529,  531,
             536,  538,  540,  551,  552,  576,  578,  585,  592,  594,  641,  643,  648,  650,  657,  664,
             698,  704,  706,  720,  729,  742,  758,  769,  773,  808,  848,  852,  870,  889,  901,  978,
             992, 1024, 1026, 1033, 1035, 1040, 1042, 1046, 1049, 1058, 1089, 1091, 1093, 1096, 1098, 1105,
            1112, 1139, 1143, 1144, 1152, 1154, 1161, 1167, 1168, 1170, 1183, 1184, 1197, 1217, 1224, 1228,
            1272, 1276, 1309, 1323, 1347, 1367, 1377, 1404, 1473, 1475, 1486, 1509, 1537, 1544, 1546, 1553,
            1555, 1576, 1589, 1594, 1600, 1602, 1616, 1625, 1636, 1638, 1665, 1667, 1672, 1685, 1706, 1722,
            1737, 1755, 1816, 1831, 1850, 1856, 1862, 1874, 1901, 1932, 1950, 1971, 2011, 2032, 2052, 2063,
            2077, 2079, 2091, 2095, 2172, 2192, 2207, 2208, 2224, 2230, 2247, 2277, 2308, 2345, 2356, 2389,
            2403, 2424, 2501, 2504, 2506, 2520, 2570, 2593, 2616, 2624, 2630, 2646, 2669, 2700, 2714, 2746,
            2754, 2795, 2824, 2835, 2839, 2874, 2882, 2905, 2984, 3028, 3042, 3092, 3108, 3110, 3124, 3153,
            3185, 3215, 3252, 3288, 3294, 3364, 3397, 3434, 3483, 3523, 3537, 3587, 3589, 3591, 3592, 3610,
            3626, 3670, 3680, 3722, 3749, 3754, 3776, 3789, 3803, 3824, 3857, 3873, 3904, 3906, 3924, 3992,
        };
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh;
            kmemcpy(&dh, p, 2);
            float d = fp16_to_fp32(dh);
            const uint8_t *qs  = p + 2;        /* grid indices [64] */
            const uint8_t *sas = p + 2 + 64;   /* scales_and_signs [32] */
            p += 98;
            const float *xb = x + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                uint32_t aux32;
                kmemcpy(&aux32, sas + 4 * ib32, 4);
                float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
                for (int l = 0; l < 4; l++) {
                    uint8_t  s7  = (uint8_t)((aux32 >> (7 * l)) & 127);
                    uint8_t  s8  = s7 | (uint8_t)((__builtin_popcount(s7) & 1) << 7);
                    uint16_t gv1 = kgrid_iq3xxs[qs[2 * l + 0]];
                    uint16_t gv2 = kgrid_iq3xxs[qs[2 * l + 1]];
                    const float *xs = xb + ib32 * 32 + l * 8;
                    for (int k = 0; k < 4; k++) {
                        float w1 = (float)(2 * ((gv1 >> (3 * k)) & 7) + 1);
                        float w2 = (float)(2 * ((gv2 >> (3 * k)) & 7) + 1);
                        if (s8 & (1u <<  k))    w1 = -w1;
                        if (s8 & (1u << (k+4))) w2 = -w2;
                        sum += xs[k]   * db * w1;
                        sum += xs[k+4] * db * w2;
                    }
                }
                qs += 8;
            }
        }
        break;
    }
    case GGML_TYPE_IQ3_S: {
        /* IQ3_S: 256 elements/super-block, 110 bytes.
         * Layout: d(fp16) + qs[64] + qh[8] + signs[32] + scales[4].
         * 9-bit grid index: qs[k] | (high-bit-from-qh << 8).
         * kgrid_iq3s[idx]: bits[3k..3k+2]=l -> val=2l+1 ∈ {1..15}.
         * Scale: db = d * (1 + 2*nibble). Processed in pairs of sub-blocks. */
        static const uint16_t kgrid_iq3s[512] = {
               0,    1,    2,    5,    7,    8,    9,   10,   12,   14,   16,   17,   21,   27,   32,   34,
              37,   39,   41,   43,   48,   50,   57,   60,   63,   64,   65,   66,   68,   72,   73,   77,
              80,   83,   87,   89,   93,  100,  113,  117,  122,  128,  129,  133,  135,  136,  139,  142,
             145,  149,  152,  156,  162,  165,  167,  169,  171,  184,  187,  195,  201,  205,  208,  210,
             217,  219,  222,  228,  232,  234,  247,  249,  253,  256,  267,  271,  273,  276,  282,  288,
             291,  297,  312,  322,  324,  336,  338,  342,  347,  353,  357,  359,  374,  379,  390,  393,
             395,  409,  426,  441,  448,  450,  452,  464,  466,  470,  475,  488,  492,  512,  513,  514,
             516,  520,  521,  523,  525,  527,  528,  530,  537,  540,  542,  556,  558,  561,  570,  576,
             577,  579,  582,  584,  588,  593,  600,  603,  609,  616,  618,  632,  638,  640,  650,  653,
             655,  656,  660,  666,  672,  675,  685,  688,  698,  705,  708,  711,  712,  715,  721,  727,
             728,  732,  737,  754,  760,  771,  773,  778,  780,  793,  795,  802,  806,  808,  812,  833,
             840,  843,  849,  856,  858,  873,  912,  916,  919,  932,  934,  961,  963,  968,  970,  977,
             989,  993, 1010, 1016, 1024, 1025, 1027, 1029, 1031, 1032, 1034, 1036, 1038, 1041, 1043, 1047,
            1048, 1050, 1057, 1059, 1061, 1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103,
            1106, 1109, 1113, 1116, 1122, 1129, 1153, 1156, 1159, 1169, 1171, 1176, 1183, 1185, 1195, 1199,
            1209, 1212, 1216, 1218, 1221, 1225, 1234, 1236, 1241, 1243, 1250, 1256, 1270, 1281, 1287, 1296,
            1299, 1306, 1309, 1313, 1338, 1341, 1348, 1353, 1362, 1375, 1376, 1387, 1400, 1408, 1410, 1415,
            1425, 1453, 1457, 1477, 1481, 1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549, 1551, 1554, 1561,
            1563, 1565, 1570, 1572, 1575, 1577, 1587, 1593, 1601, 1603, 1605, 1612, 1617, 1619, 1632, 1648,
            1658, 1662, 1664, 1674, 1680, 1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761,
            1763, 1767, 1773, 1787, 1795, 1801, 1806, 1810, 1817, 1834, 1840, 1844, 1857, 1864, 1866, 1877,
            1882, 1892, 1902, 1915, 1934, 1953, 1985, 1987, 2000, 2002, 2013, 2048, 2052, 2058, 2064, 2068,
            2071, 2074, 2081, 2088, 2104, 2114, 2119, 2121, 2123, 2130, 2136, 2141, 2147, 2153, 2157, 2177,
            2179, 2184, 2189, 2193, 2203, 2208, 2223, 2226, 2232, 2244, 2249, 2251, 2256, 2258, 2265, 2269,
            2304, 2306, 2324, 2335, 2336, 2361, 2373, 2375, 2385, 2418, 2443, 2460, 2480, 2504, 2509, 2520,
            2531, 2537, 2562, 2568, 2572, 2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634,
            2641, 2650, 2682, 2688, 2697, 2707, 2712, 2718, 2731, 2754, 2759, 2760, 2775, 2788, 2793, 2805,
            2811, 2817, 2820, 2832, 2842, 2854, 2890, 2902, 2921, 2923, 2978, 3010, 3012, 3026, 3081, 3083,
            3085, 3097, 3099, 3120, 3136, 3152, 3159, 3188, 3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276,
            3281, 3296, 3349, 3363, 3378, 3392, 3395, 3420, 3440, 3461, 3488, 3529, 3531, 3584, 3588, 3591,
            3600, 3602, 3614, 3616, 3628, 3634, 3650, 3657, 3668, 3683, 3685, 3713, 3716, 3720, 3726, 3729,
            3736, 3753, 3778, 3802, 3805, 3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
        };
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh;
            kmemcpy(&dh, p, 2);
            float d = fp16_to_fp32(dh);
            const uint8_t *qs    = p + 2;    /* grid low bytes [64]  */
            const uint8_t *qh    = p + 66;   /* grid high bits [8]   */
            const uint8_t *signs = p + 74;   /* sign bytes     [32]  */
            const uint8_t *sc    = p + 106;  /* scale nibbles  [4]   */
            p += 110;
            const float *xb = x + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32 += 2) {
                float db1 = d * (float)(1 + 2 * (sc[ib32 / 2] & 0xf));
                float db2 = d * (float)(1 + 2 * (sc[ib32 / 2] >>  4));
                /* First 32-element sub-block (uses qh[0]) */
                for (int l = 0; l < 4; l++) {
                    uint16_t idx1 = (uint16_t)qs[2*l+0] | (uint16_t)(((unsigned)qh[0] << (8 - 2*l)) & 256u);
                    uint16_t idx2 = (uint16_t)qs[2*l+1] | (uint16_t)(((unsigned)qh[0] << (7 - 2*l)) & 256u);
                    uint16_t gv1  = kgrid_iq3s[idx1];
                    uint16_t gv2  = kgrid_iq3s[idx2];
                    uint8_t  sn   = signs[l];
                    const float *xs = xb + ib32 * 32 + l * 8;
                    for (int k = 0; k < 4; k++) {
                        float w1 = (float)(2 * ((gv1 >> (3 * k)) & 7) + 1);
                        float w2 = (float)(2 * ((gv2 >> (3 * k)) & 7) + 1);
                        if (sn & (1u <<  k))    w1 = -w1;
                        if (sn & (1u << (k+4))) w2 = -w2;
                        sum += xs[k]   * db1 * w1;
                        sum += xs[k+4] * db1 * w2;
                    }
                }
                qs += 8; signs += 4;
                /* Second 32-element sub-block (uses qh[1]) */
                for (int l = 0; l < 4; l++) {
                    uint16_t idx1 = (uint16_t)qs[2*l+0] | (uint16_t)(((unsigned)qh[1] << (8 - 2*l)) & 256u);
                    uint16_t idx2 = (uint16_t)qs[2*l+1] | (uint16_t)(((unsigned)qh[1] << (7 - 2*l)) & 256u);
                    uint16_t gv1  = kgrid_iq3s[idx1];
                    uint16_t gv2  = kgrid_iq3s[idx2];
                    uint8_t  sn   = signs[l];
                    const float *xs = xb + (ib32 + 1) * 32 + l * 8;
                    for (int k = 0; k < 4; k++) {
                        float w1 = (float)(2 * ((gv1 >> (3 * k)) & 7) + 1);
                        float w2 = (float)(2 * ((gv2 >> (3 * k)) & 7) + 1);
                        if (sn & (1u <<  k))    w1 = -w1;
                        if (sn & (1u << (k+4))) w2 = -w2;
                        sum += xs[k]   * db2 * w1;
                        sum += xs[k+4] * db2 * w2;
                    }
                }
                qh += 2; qs += 8; signs += 4;
            }
        }
        break;
    }
    case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M: {
        /* IQ1_S: 256 elements/super-block, 50 bytes.
         * Layout: fp16 d (2) + qs[32] + qh_u16[8] (16).
         * 8 mini-blocks of 32 elements each.
         * dl = d * (2*((qh[ib]>>12)&7) + 1)
         * delta = (qh[ib]&0x8000) ? -0.125f : +0.125f
         * 4 groups of 8 (l=0..3):
         *   grid_idx = qs[4*ib+l] | (((qh[ib]>>(3*(3-l)))&7)<<8)
         *   int8_t grid[8] = kgrid_iq1s[grid_idx] bytes (-1/0/+1 encoded as 0xff/0x00/0x01)
         *   y[j] = dl * ((float)grid[j] + delta)
         * IQ1_M: 256 elements/super-block, 56 bytes.
         * Layout: qs[32] + qh[16] + scales[8]. No explicit fp16 d.
         * d packed in top 4 bits of each scales uint16_t pair.
         * Uses same kgrid_iq1s[2048] lookup table. */
        static const uint64_t kgrid_iq1s[2048] = {
        0xffffffffffffffff, 0xffffffffffffff01, 0xffffffffffff0000, 0xffffffffffff01ff,
        0xffffffffffff0101, 0xffffffffff00ff00, 0xffffffffff000000, 0xffffffffff01ffff,
        0xffffffffff01ff01, 0xffffffffff0101ff, 0xffffffffff010101, 0xffffffff00ff0000,
        0xffffffff0000ff00, 0xffffffff000000ff, 0xffffffff00000001, 0xffffffff00010000,
        0xffffffff01ffffff, 0xffffffff01ffff01, 0xffffffff01ff01ff, 0xffffffff01ff0101,
        0xffffffff01000000, 0xffffffff0101ffff, 0xffffffff0101ff01, 0xffffffff010101ff,
        0xffffffff01010101, 0xffffff00ffff00ff, 0xffffff00ffff0000, 0xffffff00ff00ff00,
        0xffffff00ff0000ff, 0xffffff00ff000001, 0xffffff00ff000100, 0xffffff00ff000101,
        0xffffff00ff010000, 0xffffff0000ffff00, 0xffffff0000ff0001, 0xffffff0000ff0100,
        0xffffff000000ff01, 0xffffff0000000000, 0xffffff0000000101, 0xffffff000001ff00,
        0xffffff00000100ff, 0xffffff0000010001, 0xffffff00000101ff, 0xffffff0001ff0000,
        0xffffff000100ff00, 0xffffff00010000ff, 0xffffff0001000001, 0xffffff0001010000,
        0xffffff01ffffffff, 0xffffff01ffffff01, 0xffffff01ffff01ff, 0xffffff01ffff0101,
        0xffffff01ff000000, 0xffffff01ff01ffff, 0xffffff01ff01ff01, 0xffffff01ff0101ff,
        0xffffff01ff010101, 0xffffff0100ff0000, 0xffffff010000ff00, 0xffffff0100000100,
        0xffffff01000100ff, 0xffffff0100010100, 0xffffff0101ffffff, 0xffffff0101ffff01,
        0xffffff0101ff01ff, 0xffffff0101ff0101, 0xffffff010100ff00, 0xffffff0101000000,
        0xffffff0101000100, 0xffffff010101ffff, 0xffffff010101ff01, 0xffffff01010101ff,
        0xffffff0101010101, 0xffff00ffff00ff00, 0xffff00ffff0000ff, 0xffff00ffff000001,
        0xffff00ffff010000, 0xffff00ff00ffff00, 0xffff00ff00ff0100, 0xffff00ff00000000,
        0xffff00ff00000101, 0xffff00ff000100ff, 0xffff00ff00010000, 0xffff00ff0100ff00,
        0xffff00ff01000100, 0xffff00ff01010000, 0xffff0000ffffff00, 0xffff0000ffff00ff,
        0xffff0000ffff0000, 0xffff0000ffff0001, 0xffff0000ff000000, 0xffff0000ff0001ff,
        0xffff0000ff000101, 0xffff0000ff010100, 0xffff000000ffffff, 0xffff000000ff0000,
        0xffff000000ff0101, 0xffff00000000ffff, 0xffff00000000ff00, 0xffff0000000000ff,
        0xffff000000000000, 0xffff000000000001, 0xffff000000000100, 0xffff00000001ffff,
        0xffff00000001ff01, 0xffff000000010000, 0xffff0000000101ff, 0xffff000000010101,
        0xffff000001ffff00, 0xffff00000100ff00, 0xffff000001000000, 0xffff0000010001ff,
        0xffff000001000101, 0xffff00000101ff00, 0xffff0000010100ff, 0xffff000001010000,
        0xffff000001010001, 0xffff000001010100, 0xffff0001ff0000ff, 0xffff0001ff000100,
        0xffff000100ffff00, 0xffff000100ff00ff, 0xffff00010000ffff, 0xffff00010000ff01,
        0xffff000100000000, 0xffff0001000001ff, 0xffff00010001ffff, 0xffff00010001ff00,
        0xffff000100010001, 0xffff000100010100, 0xffff000101ff0000, 0xffff00010100ff00,
        0xffff0001010000ff, 0xffff000101000100, 0xffff01ffffffffff, 0xffff01ffffffff01,
        0xffff01ffffff01ff, 0xffff01ffffff0101, 0xffff01ffff000000, 0xffff01ffff01ffff,
        0xffff01ffff01ff01, 0xffff01ffff0101ff, 0xffff01ffff010101, 0xffff01ff00ff0000,
        0xffff01ff0000ff00, 0xffff01ff00000001, 0xffff01ff00010000, 0xffff01ff01ffffff,
        0xffff01ff01ffff01, 0xffff01ff01ff01ff, 0xffff01ff01ff0101, 0xffff01ff01000000,
        0xffff01ff0101ffff, 0xffff01ff0101ff01, 0xffff01ff010101ff, 0xffff01ff01010101,
        0xffff0100ffff0000, 0xffff0100ff00ff00, 0xffff0100ff0000ff, 0xffff0100ff000100,
        0xffff0100ff0100ff, 0xffff0100ff010000, 0xffff010000ffff00, 0xffff01000000ffff,
        0xffff01000000ff00, 0xffff010000000000, 0xffff01000001ff00, 0xffff0100000100ff,
        0xffff010000010100, 0xffff01000100ff00, 0xffff0100010000ff, 0xffff010001000001,
        0xffff010001000100, 0xffff010001010000, 0xffff0101ffffffff, 0xffff0101ffffff01,
        0xffff0101ffff01ff, 0xffff0101ffff0101, 0xffff0101ff000000, 0xffff0101ff01ffff,
        0xffff0101ff01ff01, 0xffff0101ff0101ff, 0xffff0101ff010101, 0xffff010100ff0000,
        0xffff01010000ff00, 0xffff010100000100, 0xffff01010001ff00, 0xffff010100010000,
        0xffff010101ffffff, 0xffff010101ffff01, 0xffff010101ff0000, 0xffff010101ff01ff,
        0xffff010101ff0101, 0xffff010101000000, 0xffff01010101ffff, 0xffff01010101ff01,
        0xffff0101010101ff, 0xffff010101010101, 0xff00ffffff00ffff, 0xff00ffffff00ff00,
        0xff00ffffff0000ff, 0xff00ffffff000100, 0xff00ffffff0100ff, 0xff00ffffff010000,
        0xff00ffff00ffff00, 0xff00ffff00ff00ff, 0xff00ffff0000ffff, 0xff00ffff00000000,
        0xff00ffff000001ff, 0xff00ffff0001ff00, 0xff00ffff000100ff, 0xff00ffff00010000,
        0xff00ffff00010100, 0xff00ffff0100ff00, 0xff00ffff010000ff, 0xff00ffff01000001,
        0xff00ffff0101ff00, 0xff00ffff01010000, 0xff00ff00ffffff00, 0xff00ff00ffff00ff,
        0xff00ff00ffff0001, 0xff00ff00ffff0100, 0xff00ff00ff00ffff, 0xff00ff00ff00ff01,
        0xff00ff00ff000000, 0xff00ff00ff0001ff, 0xff00ff00ff01ff00, 0xff00ff00ff0100ff,
        0xff00ff00ff010100, 0xff00ff0000ff0000, 0xff00ff0000ff0101, 0xff00ff000000ffff,
        0xff00ff000000ff00, 0xff00ff000000ff01, 0xff00ff00000000ff, 0xff00ff0000000000,
        0xff00ff0000000001, 0xff00ff0000000100, 0xff00ff000001ffff, 0xff00ff0000010000,
        0xff00ff0001ff00ff, 0xff00ff000100ff01, 0xff00ff0001000000, 0xff00ff000101ff00,
        0xff00ff00010100ff, 0xff00ff01ff00ff00, 0xff00ff01ff0000ff, 0xff00ff01ff000001,
        0xff00ff01ff010000, 0xff00ff0100ffffff, 0xff00ff0100ff0001, 0xff00ff0100ff0100,
        0xff00ff010000ff01, 0xff00ff0100000000, 0xff00ff01000001ff, 0xff00ff0100000101,
        0xff00ff01000100ff, 0xff00ff0100010001, 0xff00ff0101ff0000, 0xff00ff010100ff00,
        0xff00ff01010000ff, 0xff00ff0101000001, 0xff00ff0101010000, 0xff0000ffffffff00,
        0xff0000ffffff0001, 0xff0000ffffff0100, 0xff0000ffff0000ff, 0xff0000ffff000000,
        0xff0000ffff0001ff, 0xff0000ffff000100, 0xff0000ffff01ff00, 0xff0000ffff010001,
        0xff0000ff00ffff00, 0xff0000ff00ff0000, 0xff0000ff00ff0001, 0xff0000ff00ff01ff,
        0xff0000ff00ff0101, 0xff0000ff0000ff00, 0xff0000ff000000ff, 0xff0000ff00000000,
        0xff0000ff00000001, 0xff0000ff00000100, 0xff0000ff0001ff01, 0xff0000ff00010000,
        0xff0000ff000101ff, 0xff0000ff01ff00ff, 0xff0000ff01ff0100, 0xff0000ff0100ffff,
        0xff0000ff010000ff, 0xff0000ff01000000, 0xff0000ff010001ff, 0xff0000ff01000100,
        0xff0000ff01000101, 0xff0000ff0101ff00, 0xff0000ff010100ff, 0xff0000ff01010000,
        0xff0000ff01010100, 0xff000000ffffff01, 0xff000000ffff0000, 0xff000000ffff0101,
        0xff000000ff00ff00, 0xff000000ff0000ff, 0xff000000ff000000, 0xff000000ff000001,
        0xff000000ff000100, 0xff000000ff01ffff, 0xff000000ff01ff01, 0xff000000ff010000,
        0xff000000ff0101ff, 0xff000000ff010101, 0xff00000000ffff00, 0xff00000000ff00ff,
        0xff00000000ff0000, 0xff00000000ff0001, 0xff0000000000ff00, 0xff0000000000ff01,
        0xff000000000000ff, 0xff00000000000000, 0xff00000000000001, 0xff00000000000100,
        0xff00000000000101, 0xff0000000001ff00, 0xff000000000100ff, 0xff00000000010000,
        0xff00000000010001, 0xff00000000010100, 0xff00000001ffffff, 0xff00000001ffff01,
        0xff00000001ff00ff, 0xff00000001ff0000, 0xff00000001ff01ff, 0xff00000001ff0101,
        0xff0000000100ffff, 0xff0000000100ff00, 0xff000000010000ff, 0xff00000001000000,
        0xff00000001000001, 0xff00000001000100, 0xff00000001000101, 0xff0000000101ffff,
        0xff0000000101ff01, 0xff00000001010000, 0xff000001ffffff00, 0xff000001ffff00ff,
        0xff000001ffff0000, 0xff000001ffff0001, 0xff000001ff000000, 0xff000001ff000001,
        0xff000001ff0001ff, 0xff000001ff000101, 0xff000001ff01ff00, 0xff000001ff010001,
        0xff00000100ffffff, 0xff00000100ffff01, 0xff00000100ff00ff, 0xff00000100ff0000,
        0xff00000100ff01ff, 0xff00000100ff0101, 0xff0000010000ff00, 0xff00000100000000,
        0xff00000100000001, 0xff000001000001ff, 0xff00000100000100, 0xff0000010001ff00,
        0xff000001000100ff, 0xff00000100010000, 0xff000001000101ff, 0xff00000100010100,
        0xff00000100010101, 0xff00000101ff0001, 0xff00000101ff0101, 0xff0000010100ff01,
        0xff00000101000000, 0xff000001010100ff, 0xff00000101010100, 0xff0001ffff00ff00,
        0xff0001ffff000001, 0xff0001ffff010000, 0xff0001ff00ffff00, 0xff0001ff00ff00ff,
        0xff0001ff00ff0001, 0xff0001ff00ff0100, 0xff0001ff0000ffff, 0xff0001ff00000000,
        0xff0001ff000001ff, 0xff0001ff00000101, 0xff0001ff0001ffff, 0xff0001ff0001ff00,
        0xff0001ff000100ff, 0xff0001ff00010001, 0xff0001ff00010100, 0xff0001ff01ff0000,
        0xff0001ff0100ff00, 0xff0001ff010000ff, 0xff0001ff01010000, 0xff000100ff00ffff,
        0xff000100ff00ff01, 0xff000100ff000000, 0xff000100ff000101, 0xff000100ff01ff00,
        0xff000100ff010000, 0xff00010000ffff01, 0xff00010000ff00ff, 0xff00010000ff0000,
        0xff00010000ff01ff, 0xff0001000000ff00, 0xff000100000000ff, 0xff00010000000000,
        0xff00010000000001, 0xff00010000000100, 0xff00010000000101, 0xff0001000001ffff,
        0xff00010000010000, 0xff00010000010101, 0xff00010001ff0100, 0xff0001000100ff00,
        0xff0001000100ff01, 0xff00010001000000, 0xff000100010001ff, 0xff0001000101ff00,
        0xff00010001010001, 0xff00010001010100, 0xff000101ffff0100, 0xff000101ff000001,
        0xff000101ff0100ff, 0xff000101ff010001, 0xff00010100ff00ff, 0xff00010100ff0001,
        0xff00010100ff0100, 0xff0001010000ffff, 0xff0001010000ff01, 0xff00010100000000,
        0xff000101000001ff, 0xff0001010001ff00, 0xff00010100010001, 0xff00010100010100,
        0xff00010101ff0000, 0xff0001010100ff00, 0xff00010101000001, 0xff00010101000101,
        0xff01ffffffffffff, 0xff01ffffffffff01, 0xff01ffffffff01ff, 0xff01ffffffff0101,
        0xff01ffffff000000, 0xff01ffffff01ffff, 0xff01ffffff01ff01, 0xff01ffffff010000,
        0xff01ffffff0101ff, 0xff01ffffff010101, 0xff01ffff00ff0000, 0xff01ffff0000ff00,
        0xff01ffff00000100, 0xff01ffff0001ff00, 0xff01ffff00010000, 0xff01ffff01ffffff,
        0xff01ffff01ffff01, 0xff01ffff01ff01ff, 0xff01ffff01ff0101, 0xff01ffff01000000,
        0xff01ffff0101ffff, 0xff01ffff0101ff01, 0xff01ffff01010000, 0xff01ffff010101ff,
        0xff01ffff01010101, 0xff01ff00ffff0000, 0xff01ff00ff00ff00, 0xff01ff00ff0000ff,
        0xff01ff00ff000100, 0xff01ff00ff010000, 0xff01ff0000ffff01, 0xff01ff0000ff00ff,
        0xff01ff0000ff0100, 0xff01ff0000000000, 0xff01ff00000001ff, 0xff01ff0000000101,
        0xff01ff000001ff00, 0xff01ff00000100ff, 0xff01ff0000010000, 0xff01ff0000010001,
        0xff01ff0001ff0000, 0xff01ff000100ffff, 0xff01ff0001000001, 0xff01ff0001000100,
        0xff01ff0001010000, 0xff01ff01ffffff00, 0xff01ff01ffff01ff, 0xff01ff01ffff0101,
        0xff01ff01ff00ff00, 0xff01ff01ff000000, 0xff01ff01ff01ffff, 0xff01ff01ff01ff01,
        0xff01ff01ff0101ff, 0xff01ff01ff010101, 0xff01ff0100ff0000, 0xff01ff010000ff00,
        0xff01ff0100000001, 0xff01ff0100000100, 0xff01ff0100010000, 0xff01ff0101ffff00,
        0xff01ff0101ff01ff, 0xff01ff0101ff0101, 0xff01ff010100ff00, 0xff01ff0101000000,
        0xff01ff010101ffff, 0xff01ff010101ff01, 0xff01ff01010101ff, 0xff01ff0101010101,
        0xff0100ffffff0000, 0xff0100ffff0000ff, 0xff0100ffff000001, 0xff0100ffff000100,
        0xff0100ffff010000, 0xff0100ff00ff00ff, 0xff0100ff00ff0000, 0xff0100ff00ff0001,
        0xff0100ff00ff0100, 0xff0100ff0000ff01, 0xff0100ff00000000, 0xff0100ff000001ff,
        0xff0100ff00000101, 0xff0100ff00010001, 0xff0100ff01ff0000, 0xff0100ff0100ff00,
        0xff0100ff010000ff, 0xff0100ff01000100, 0xff0100ff0101ff00, 0xff0100ff01010000,
        0xff010000ffff0100, 0xff010000ff000000, 0xff010000ff01ff00, 0xff010000ff010100,
        0xff01000000ffffff, 0xff01000000ff0000, 0xff01000000ff01ff, 0xff0100000000ff00,
        0xff010000000000ff, 0xff01000000000000, 0xff01000000000100, 0xff0100000001ff01,
        0xff01000000010000, 0xff010000000101ff, 0xff01000001ff0100, 0xff0100000100ffff,
        0xff010000010000ff, 0xff01000001000000, 0xff010000010001ff, 0xff01000001000101,
        0xff0100000101ff00, 0xff010000010100ff, 0xff01000001010001, 0xff01000001010100,
        0xff010001ffff0000, 0xff010001ff00ffff, 0xff010001ff00ff01, 0xff010001ff000100,
        0xff010001ff010000, 0xff01000100ffff00, 0xff01000100ff0100, 0xff01000100000000,
        0xff0100010001ffff, 0xff0100010001ff00, 0xff01000100010100, 0xff01000101ff00ff,
        0xff01000101ff0001, 0xff0100010100ffff, 0xff01000101000101, 0xff0101ffffffffff,
        0xff0101ffffffff01, 0xff0101ffffff01ff, 0xff0101ffffff0101, 0xff0101ffff000000,
        0xff0101ffff01ffff, 0xff0101ffff01ff01, 0xff0101ffff0101ff, 0xff0101ffff010101,
        0xff0101ff00ff0000, 0xff0101ff0000ff00, 0xff0101ff000000ff, 0xff0101ff00010000,
        0xff0101ff01ffffff, 0xff0101ff01ffff01, 0xff0101ff01ff01ff, 0xff0101ff01ff0101,
        0xff0101ff0101ffff, 0xff0101ff0101ff01, 0xff0101ff010101ff, 0xff0101ff01010101,
        0xff010100ffff0100, 0xff010100ff00ff00, 0xff010100ff0000ff, 0xff010100ff000100,
        0xff010100ff010000, 0xff01010000ff0001, 0xff01010000ff0100, 0xff0101000000ff01,
        0xff01010000000000, 0xff0101000001ff00, 0xff010100000100ff, 0xff01010000010001,
        0xff01010000010100, 0xff01010001ff0000, 0xff0101000100ffff, 0xff01010001000001,
        0xff01010001000100, 0xff010100010100ff, 0xff01010001010000, 0xff010101ffffffff,
        0xff010101ffffff01, 0xff010101ffff01ff, 0xff010101ffff0101, 0xff010101ff01ffff,
        0xff010101ff01ff01, 0xff010101ff0101ff, 0xff010101ff010101, 0xff01010100ff0000,
        0xff0101010000ff00, 0xff01010100000001, 0xff01010100000100, 0xff01010100010000,
        0xff01010101ffffff, 0xff01010101ffff01, 0xff01010101ff01ff, 0xff01010101ff0101,
        0xff01010101000000, 0xff0101010101ffff, 0xff0101010101ff01, 0xff010101010101ff,
        0xff01010101010101, 0x00ffffffffff0000, 0x00ffffffff00ff00, 0x00ffffffff000001,
        0x00ffffffff010000, 0x00ffffff00ff0100, 0x00ffffff0000ff01, 0x00ffffff00000000,
        0x00ffffff000001ff, 0x00ffffff00000101, 0x00ffffff0001ff00, 0x00ffffff000100ff,
        0x00ffffff00010001, 0x00ffffff010000ff, 0x00ffffff01000100, 0x00ffffff0101ff00,
        0x00ffffff01010001, 0x00ffff00ffffffff, 0x00ffff00ffffff00, 0x00ffff00ffff00ff,
        0x00ffff00ffff0001, 0x00ffff00ffff0100, 0x00ffff00ff00ff01, 0x00ffff00ff000000,
        0x00ffff00ff000001, 0x00ffff00ff0001ff, 0x00ffff00ff000101, 0x00ffff00ff01ff00,
        0x00ffff00ff010001, 0x00ffff00ff010100, 0x00ffff0000ff0000, 0x00ffff0000ff01ff,
        0x00ffff0000ff0101, 0x00ffff000000ff00, 0x00ffff00000000ff, 0x00ffff0000000000,
        0x00ffff0000000001, 0x00ffff0000000100, 0x00ffff0000000101, 0x00ffff0000010000,
        0x00ffff00000101ff, 0x00ffff0000010101, 0x00ffff0001ffff00, 0x00ffff0001ff00ff,
        0x00ffff0001ff0001, 0x00ffff000100ffff, 0x00ffff000100ff01, 0x00ffff0001000000,
        0x00ffff000101ffff, 0x00ffff000101ff00, 0x00ffff000101ff01, 0x00ffff01ffff0000,
        0x00ffff01ff00ff00, 0x00ffff01ff0000ff, 0x00ffff01ff000001, 0x00ffff01ff010000,
        0x00ffff0100ffff00, 0x00ffff010000ff01, 0x00ffff0100000000, 0x00ffff0100000101,
        0x00ffff01000100ff, 0x00ffff0100010100, 0x00ffff0101ff0100, 0x00ffff01010000ff,
        0x00ffff0101010000, 0x00ff00ffffffff00, 0x00ff00ffff000000, 0x00ff00ffff000100,
        0x00ff00ffff010100, 0x00ff00ff00ff0000, 0x00ff00ff00ff01ff, 0x00ff00ff00ff0101,
        0x00ff00ff0000ff00, 0x00ff00ff000000ff, 0x00ff00ff00000000, 0x00ff00ff00000001,
        0x00ff00ff0001ff00, 0x00ff00ff0001ff01, 0x00ff00ff00010000, 0x00ff00ff000101ff,
        0x00ff00ff00010101, 0x00ff00ff01ffff00, 0x00ff00ff01ff0001, 0x00ff00ff01ff0100,
        0x00ff00ff0100ffff, 0x00ff00ff0100ff01, 0x00ff00ff01000000, 0x00ff00ff0101ffff,
        0x00ff00ff0101ff00, 0x00ff00ff01010100, 0x00ff0000ffffff00, 0x00ff0000ffffff01,
        0x00ff0000ffff0000, 0x00ff0000ffff0101, 0x00ff0000ff00ff00, 0x00ff0000ff0000ff,
        0x00ff0000ff000000, 0x00ff0000ff000001, 0x00ff0000ff000100, 0x00ff0000ff01ffff,
        0x00ff0000ff010000, 0x00ff0000ff010101, 0x00ff000000ffff00, 0x00ff000000ff00ff,
        0x00ff000000ff0000, 0x00ff000000ff0001, 0x00ff000000ff0100, 0x00ff00000000ffff,
        0x00ff00000000ff00, 0x00ff0000000000ff, 0x00ff000000000000, 0x00ff000000000001,
        0x00ff0000000001ff, 0x00ff000000000100, 0x00ff00000001ff00, 0x00ff0000000100ff,
        0x00ff000000010000, 0x00ff000000010001, 0x00ff000000010100, 0x00ff000001ffff01,
        0x00ff000001ff00ff, 0x00ff000001ff0000, 0x00ff000001ff01ff, 0x00ff00000100ff00,
        0x00ff0000010000ff, 0x00ff000001000000, 0x00ff000001000001, 0x00ff000001000100,
        0x00ff000001000101, 0x00ff000001010000, 0x00ff0000010101ff, 0x00ff000001010101,
        0x00ff0001ffffff00, 0x00ff0001ffff0000, 0x00ff0001ffff0100, 0x00ff0001ff0000ff,
        0x00ff0001ff000000, 0x00ff0001ff0001ff, 0x00ff0001ff000101, 0x00ff0001ff01ff00,
        0x00ff0001ff0100ff, 0x00ff0001ff010100, 0x00ff000100ffffff, 0x00ff000100ffff01,
        0x00ff000100ff0000, 0x00ff000100ff01ff, 0x00ff00010000ffff, 0x00ff00010000ff00,
        0x00ff00010000ff01, 0x00ff000100000000, 0x00ff000100000001, 0x00ff000100000100,
        0x00ff00010001ff01, 0x00ff000100010000, 0x00ff0001000101ff, 0x00ff000101ffff00,
        0x00ff000101ff0000, 0x00ff000101ff0101, 0x00ff0001010000ff, 0x00ff000101000000,
        0x00ff00010101ff00, 0x00ff0001010100ff, 0x00ff000101010001, 0x00ff01ffffff0000,
        0x00ff01ffff00ff00, 0x00ff01ffff000000, 0x00ff01ffff000101, 0x00ff01ffff010000,
        0x00ff01ff00ffff01, 0x00ff01ff00ff0100, 0x00ff01ff0000ffff, 0x00ff01ff00000000,
        0x00ff01ff000001ff, 0x00ff01ff0001ff00, 0x00ff01ff000100ff, 0x00ff01ff00010001,
        0x00ff01ff00010100, 0x00ff01ff01ff0000, 0x00ff01ff0100ff00, 0x00ff01ff010000ff,
        0x00ff01ff01000001, 0x00ff01ff01000100, 0x00ff01ff01010000, 0x00ff0100ffffff00,
        0x00ff0100ffff0000, 0x00ff0100ffff0001, 0x00ff0100ffff0101, 0x00ff0100ff00ffff,
        0x00ff0100ff0000ff, 0x00ff0100ff000000, 0x00ff0100ff0001ff, 0x00ff0100ff01ff00,
        0x00ff0100ff0100ff, 0x00ff0100ff010001, 0x00ff010000ffffff, 0x00ff010000ff0000,
        0x00ff010000ff0101, 0x00ff01000000ff00, 0x00ff01000000ff01, 0x00ff0100000000ff,
        0x00ff010000000000, 0x00ff010000000001, 0x00ff010000000100, 0x00ff01000001ffff,
        0x00ff01000001ff01, 0x00ff010000010000, 0x00ff010000010001, 0x00ff010000010101,
        0x00ff010001ff0001, 0x00ff010001ff0100, 0x00ff01000100ff01, 0x00ff010001000000,
        0x00ff010001000001, 0x00ff0100010001ff, 0x00ff01000101ff00, 0x00ff0100010100ff,
        0x00ff010001010001, 0x00ff010001010100, 0x00ff0101ff000001, 0x00ff010100ff00ff,
        0x00ff010100ff0001, 0x00ff010100ff0100, 0x00ff010100000000, 0x00ff0101000001ff,
        0x00ff010100000101, 0x00ff0101000100ff, 0x00ff010100010100, 0x00ff0101010000ff,
        0x00ff010101010000, 0x0000ffffffffff00, 0x0000ffffffff00ff, 0x0000ffffffff0000,
        0x0000ffffffff0001, 0x0000ffffffff0100, 0x0000ffffff00ff01, 0x0000ffffff000000,
        0x0000ffffff000101, 0x0000ffffff01ff00, 0x0000ffffff0100ff, 0x0000ffffff010100,
        0x0000ffff00ffffff, 0x0000ffff00ff0000, 0x0000ffff00ff01ff, 0x0000ffff0000ff00,
        0x0000ffff000000ff, 0x0000ffff00000000, 0x0000ffff00000001, 0x0000ffff00000100,
        0x0000ffff00010000, 0x0000ffff000101ff, 0x0000ffff01ff0001, 0x0000ffff01ff0100,
        0x0000ffff01000000, 0x0000ffff010001ff, 0x0000ffff0101ffff, 0x0000ffff0101ff00,
        0x0000ffff01010001, 0x0000ffff01010100, 0x0000ff00ffff0000, 0x0000ff00ffff01ff,
        0x0000ff00ffff0100, 0x0000ff00ffff0101, 0x0000ff00ff00ff00, 0x0000ff00ff0000ff,
        0x0000ff00ff000000, 0x0000ff00ff000001, 0x0000ff00ff0001ff, 0x0000ff00ff000100,
        0x0000ff00ff01ffff, 0x0000ff00ff010000, 0x0000ff00ff010001, 0x0000ff00ff0101ff,
        0x0000ff00ff010101, 0x0000ff0000ffff00, 0x0000ff0000ff00ff, 0x0000ff0000ff0000,
        0x0000ff0000ff0001, 0x0000ff0000ff0100, 0x0000ff000000ffff, 0x0000ff000000ff00,
        0x0000ff000000ff01, 0x0000ff00000000ff, 0x0000ff0000000000, 0x0000ff0000000001,
        0x0000ff00000001ff, 0x0000ff0000000100, 0x0000ff0000000101, 0x0000ff000001ff00,
        0x0000ff00000100ff, 0x0000ff0000010000, 0x0000ff0000010001, 0x0000ff0000010100,
        0x0000ff0001ffff01, 0x0000ff0001ff0000, 0x0000ff000100ff00, 0x0000ff00010000ff,
        0x0000ff0001000000, 0x0000ff0001000001, 0x0000ff0001000100, 0x0000ff000101ffff,
        0x0000ff0001010000, 0x0000ff0001010101, 0x0000ff01ffffff00, 0x0000ff01ffff0001,
        0x0000ff01ff00ff01, 0x0000ff01ff000000, 0x0000ff01ff000101, 0x0000ff01ff01ff00,
        0x0000ff01ff0100ff, 0x0000ff0100ffff01, 0x0000ff0100ff0000, 0x0000ff0100ff0101,
        0x0000ff010000ff00, 0x0000ff01000000ff, 0x0000ff0100000000, 0x0000ff0100000001,
        0x0000ff0100000100, 0x0000ff010001ff01, 0x0000ff0100010000, 0x0000ff0101ff0000,
        0x0000ff010100ffff, 0x0000ff010100ff01, 0x0000ff0101000000, 0x0000ff0101000100,
        0x0000ff0101000101, 0x0000ff01010100ff, 0x000000ffffff00ff, 0x000000ffffff0000,
        0x000000ffff00ff00, 0x000000ffff0000ff, 0x000000ffff000000, 0x000000ffff000001,
        0x000000ffff0001ff, 0x000000ffff000100, 0x000000ffff01ff00, 0x000000ffff010000,
        0x000000ffff0101ff, 0x000000ffff010101, 0x000000ff00ffff00, 0x000000ff00ff00ff,
        0x000000ff00ff0000, 0x000000ff00ff0001, 0x000000ff00ff0100, 0x000000ff00ff0101,
        0x000000ff0000ffff, 0x000000ff0000ff00, 0x000000ff000000ff, 0x000000ff00000000,
        0x000000ff00000001, 0x000000ff000001ff, 0x000000ff00000100, 0x000000ff00000101,
        0x000000ff0001ff00, 0x000000ff0001ff01, 0x000000ff000100ff, 0x000000ff00010000,
        0x000000ff00010001, 0x000000ff00010100, 0x000000ff01ffffff, 0x000000ff01ff01ff,
        0x000000ff01ff0101, 0x000000ff0100ff00, 0x000000ff010000ff, 0x000000ff01000000,
        0x000000ff01000001, 0x000000ff01000100, 0x000000ff0101ff00, 0x000000ff010100ff,
        0x000000ff01010000, 0x000000ff01010101, 0x00000000ffffff00, 0x00000000ffffff01,
        0x00000000ffff00ff, 0x00000000ffff0000, 0x00000000ffff0001, 0x00000000ffff0100,
        0x00000000ff00ffff, 0x00000000ff00ff00, 0x00000000ff00ff01, 0x00000000ff0000ff,
        0x00000000ff000000, 0x00000000ff000001, 0x00000000ff000100, 0x00000000ff000101,
        0x00000000ff01ff00, 0x00000000ff0100ff, 0x00000000ff010000, 0x00000000ff010001,
        0x00000000ff010100, 0x0000000000ffffff, 0x0000000000ffff00, 0x0000000000ffff01,
        0x0000000000ff00ff, 0x0000000000ff0000, 0x0000000000ff0001, 0x0000000000ff01ff,
        0x0000000000ff0100, 0x000000000000ffff, 0x000000000000ff00, 0x000000000000ff01,
        0x00000000000000ff, 0x0000000000000000, 0x0000000000000001, 0x00000000000001ff,
        0x0000000000000100, 0x0000000000000101, 0x000000000001ffff, 0x000000000001ff00,
        0x00000000000100ff, 0x0000000000010000, 0x0000000000010001, 0x00000000000101ff,
        0x0000000000010100, 0x0000000000010101, 0x0000000001ffff00, 0x0000000001ff00ff,
        0x0000000001ff0000, 0x0000000001ff0100, 0x0000000001ff0101, 0x000000000100ffff,
        0x000000000100ff00, 0x00000000010000ff, 0x0000000001000000, 0x0000000001000001,
        0x00000000010001ff, 0x0000000001000100, 0x000000000101ff00, 0x00000000010100ff,
        0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x00000001ffffffff,
        0x00000001ffffff00, 0x00000001ffffff01, 0x00000001ffff00ff, 0x00000001ffff0001,
        0x00000001ffff01ff, 0x00000001ffff0100, 0x00000001ff00ff00, 0x00000001ff0000ff,
        0x00000001ff000000, 0x00000001ff0001ff, 0x00000001ff000100, 0x00000001ff01ffff,
        0x00000001ff01ff00, 0x00000001ff01ff01, 0x00000001ff0100ff, 0x00000001ff010000,
        0x00000001ff010001, 0x00000001ff0101ff, 0x00000001ff010100, 0x0000000100ffff00,
        0x0000000100ff0000, 0x0000000100ff0001, 0x0000000100ff01ff, 0x0000000100ff0100,
        0x0000000100ff0101, 0x000000010000ffff, 0x000000010000ff00, 0x000000010000ff01,
        0x00000001000000ff, 0x0000000100000000, 0x0000000100000001, 0x00000001000001ff,
        0x0000000100000100, 0x0000000100000101, 0x000000010001ff00, 0x00000001000100ff,
        0x0000000100010000, 0x0000000100010100, 0x0000000101ffff01, 0x0000000101ff0000,
        0x0000000101ff0001, 0x0000000101ff01ff, 0x0000000101ff0100, 0x0000000101ff0101,
        0x000000010100ff00, 0x0000000101000000, 0x0000000101000101, 0x000000010101ff01,
        0x0000000101010000, 0x0000000101010001, 0x00000001010101ff, 0x0000000101010100,
        0x000001ffffff00ff, 0x000001ffffff0000, 0x000001ffffff0001, 0x000001ffffff0100,
        0x000001ffff00ffff, 0x000001ffff000000, 0x000001ffff0001ff, 0x000001ffff01ff00,
        0x000001ffff010101, 0x000001ff00ff0000, 0x000001ff00ff01ff, 0x000001ff00ff0101,
        0x000001ff0000ff00, 0x000001ff000000ff, 0x000001ff00000000, 0x000001ff00000001,
        0x000001ff000001ff, 0x000001ff00000100, 0x000001ff0001ffff, 0x000001ff0001ff01,
        0x000001ff000100ff, 0x000001ff00010000, 0x000001ff01ffff01, 0x000001ff01ff0100,
        0x000001ff0100ffff, 0x000001ff0100ff01, 0x000001ff01000000, 0x000001ff010001ff,
        0x000001ff0101ff00, 0x000001ff01010100, 0x00000100ffffff00, 0x00000100ffffff01,
        0x00000100ffff0000, 0x00000100ffff0101, 0x00000100ff00ff00, 0x00000100ff0000ff,
        0x00000100ff000000, 0x00000100ff000001, 0x00000100ff000100, 0x00000100ff010000,
        0x0000010000ffff00, 0x0000010000ff00ff, 0x0000010000ff0000, 0x0000010000ff0001,
        0x0000010000ff0100, 0x000001000000ffff, 0x000001000000ff00, 0x000001000000ff01,
        0x00000100000000ff, 0x0000010000000000, 0x0000010000000001, 0x00000100000001ff,
        0x0000010000000100, 0x0000010000000101, 0x000001000001ff00, 0x00000100000100ff,
        0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010001ffff00,
        0x0000010001ff0000, 0x0000010001ff0100, 0x000001000100ff00, 0x00000100010000ff,
        0x0000010001000000, 0x0000010001000001, 0x00000100010001ff, 0x0000010001000100,
        0x0000010001010000, 0x00000101ffff00ff, 0x00000101ffff01ff, 0x00000101ff000000,
        0x00000101ff000101, 0x00000101ff01ffff, 0x00000101ff010000, 0x00000101ff010001,
        0x00000101ff010100, 0x0000010100ff0000, 0x0000010100ff01ff, 0x0000010100ff0100,
        0x000001010000ff00, 0x0000010100000000, 0x0000010100000001, 0x00000101000001ff,
        0x0000010100000100, 0x000001010001ff01, 0x0000010100010000, 0x00000101000101ff,
        0x0000010100010101, 0x0000010101ffff00, 0x0000010101ff0101, 0x000001010100ff01,
        0x0000010101000000, 0x0000010101000001, 0x00000101010001ff, 0x0000010101000101,
        0x000001010101ff00, 0x0001ffffffff0000, 0x0001ffffff0000ff, 0x0001ffffff000001,
        0x0001ffffff000100, 0x0001ffffff010000, 0x0001ffff00ff00ff, 0x0001ffff0000ffff,
        0x0001ffff00000000, 0x0001ffff00000001, 0x0001ffff000001ff, 0x0001ffff00000101,
        0x0001ffff0001ff00, 0x0001ffff000100ff, 0x0001ffff00010001, 0x0001ffff00010100,
        0x0001ffff01ffff00, 0x0001ffff01000001, 0x0001ffff01010000, 0x0001ff00ffffff00,
        0x0001ff00ffff00ff, 0x0001ff00ffff0001, 0x0001ff00ffff0100, 0x0001ff00ff00ff01,
        0x0001ff00ff000000, 0x0001ff00ff01ff00, 0x0001ff00ff01ff01, 0x0001ff00ff010001,
        0x0001ff00ff010100, 0x0001ff0000ff0000, 0x0001ff0000ff0100, 0x0001ff000000ff00,
        0x0001ff0000000000, 0x0001ff0000000001, 0x0001ff0000000100, 0x0001ff0000010000,
        0x0001ff0000010001, 0x0001ff0000010101, 0x0001ff0001ff00ff, 0x0001ff0001ff0101,
        0x0001ff000100ff01, 0x0001ff0001000000, 0x0001ff000101ff00, 0x0001ff0001010001,
        0x0001ff0001010100, 0x0001ff01ff00ff00, 0x0001ff01ff000001, 0x0001ff01ff000100,
        0x0001ff0100ffffff, 0x0001ff0100ffff00, 0x0001ff0100ff0001, 0x0001ff0100000000,
        0x0001ff0100000001, 0x0001ff01000001ff, 0x0001ff010001ffff, 0x0001ff0101ff0000,
        0x0001ff010100ff00, 0x0001ff0101000001, 0x0001ff0101010000, 0x000100ffff00ff00,
        0x000100ffff00ff01, 0x000100ffff000000, 0x000100ffff000001, 0x000100ffff000101,
        0x000100ffff01ff00, 0x000100ffff010001, 0x000100ffff010100, 0x000100ff00ffffff,
        0x000100ff00ffff01, 0x000100ff00ff0000, 0x000100ff00ff01ff, 0x000100ff00ff0101,
        0x000100ff0000ff00, 0x000100ff000000ff, 0x000100ff00000000, 0x000100ff00000001,
        0x000100ff00000100, 0x000100ff00000101, 0x000100ff0001ffff, 0x000100ff0001ff01,
        0x000100ff00010000, 0x000100ff01ff00ff, 0x000100ff01ff0000, 0x000100ff01ff0100,
        0x000100ff0100ffff, 0x000100ff0100ff01, 0x000100ff010000ff, 0x000100ff01000000,
        0x000100ff01000001, 0x000100ff010001ff, 0x000100ff01000101, 0x000100ff0101ff00,
        0x000100ff010100ff, 0x000100ff01010100, 0x00010000ffff0000, 0x00010000ffff01ff,
        0x00010000ffff0101, 0x00010000ff00ff00, 0x00010000ff000000, 0x00010000ff000001,
        0x00010000ff000100, 0x0001000000ff00ff, 0x0001000000ff0000, 0x0001000000ff0001,
        0x0001000000ff0100, 0x000100000000ffff, 0x000100000000ff00, 0x00010000000000ff,
        0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x000100000001ff00,
        0x00010000000100ff, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100,
        0x0001000001ff0001, 0x0001000001ff0100, 0x0001000001ff0101, 0x000100000100ff00,
        0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101,
        0x000100000101ff01, 0x0001000001010000, 0x0001000001010001, 0x00010000010101ff,
        0x00010001ffffff01, 0x00010001ffff0100, 0x00010001ff000000, 0x00010001ff01ffff,
        0x00010001ff010001, 0x00010001ff0101ff, 0x00010001ff010100, 0x0001000100ffffff,
        0x0001000100ff0000, 0x0001000100ff01ff, 0x0001000100ff0101, 0x000100010000ff00,
        0x00010001000000ff, 0x0001000100000000, 0x0001000100000001, 0x00010001000001ff,
        0x0001000100000101, 0x000100010001ffff, 0x0001000100010000, 0x00010001000101ff,
        0x0001000101ffffff, 0x0001000101ffff01, 0x0001000101ff0000, 0x0001000101ff0101,
        0x00010001010000ff, 0x0001000101000001, 0x00010001010001ff, 0x0001000101000100,
        0x000100010101ffff, 0x00010001010100ff, 0x0001000101010001, 0x0001000101010101,
        0x000101ffff000001, 0x000101ffff000100, 0x000101ffff010000, 0x000101ff00ffff00,
        0x000101ff0000ff01, 0x000101ff00000000, 0x000101ff00000101, 0x000101ff0001ff00,
        0x000101ff00010100, 0x000101ff01ff0000, 0x000101ff0100ff00, 0x000101ff010001ff,
        0x000101ff01010001, 0x00010100ffffff00, 0x00010100ffff00ff, 0x00010100ff00ffff,
        0x00010100ff000000, 0x00010100ff01ff00, 0x00010100ff0100ff, 0x00010100ff010001,
        0x00010100ff010100, 0x0001010000ffffff, 0x0001010000ffff00, 0x0001010000ff0000,
        0x0001010000ff0001, 0x0001010000ff01ff, 0x000101000000ff00, 0x00010100000000ff,
        0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x000101000001ffff,
        0x0001010000010000, 0x0001010000010101, 0x0001010001ffff01, 0x0001010001ff00ff,
        0x0001010001ff0101, 0x0001010001000000, 0x000101000101ff00, 0x00010100010100ff,
        0x0001010001010000, 0x0001010001010100, 0x00010101ff00ff00, 0x00010101ff000001,
        0x00010101ff0001ff, 0x0001010100ffff00, 0x0001010100ff00ff, 0x0001010100ff0100,
        0x000101010000ffff, 0x0001010100000000, 0x00010101000001ff, 0x0001010100000101,
        0x00010101000100ff, 0x0001010100010000, 0x0001010100010100, 0x0001010101ff0001,
        0x00010101010000ff, 0x00010101010001ff, 0x0001010101000101, 0x0001010101010001,
        0x01ffffffffffffff, 0x01ffffffffffff01, 0x01ffffffffff01ff, 0x01ffffffffff0101,
        0x01ffffffff01ffff, 0x01ffffffff01ff01, 0x01ffffffff0101ff, 0x01ffffffff010101,
        0x01ffffff00ff0000, 0x01ffffff0000ffff, 0x01ffffff0000ff00, 0x01ffffff000000ff,
        0x01ffffff00000001, 0x01ffffff00000100, 0x01ffffff00010000, 0x01ffffff01ffffff,
        0x01ffffff01ffff01, 0x01ffffff01ff01ff, 0x01ffffff01ff0101, 0x01ffffff01000000,
        0x01ffffff0101ffff, 0x01ffffff0101ff01, 0x01ffffff010101ff, 0x01ffffff01010101,
        0x01ffff00ffff0000, 0x01ffff00ff00ff00, 0x01ffff00ff0000ff, 0x01ffff00ff000001,
        0x01ffff00ff000100, 0x01ffff00ff010000, 0x01ffff0000ffff00, 0x01ffff0000ff00ff,
        0x01ffff0000ff0100, 0x01ffff000000ffff, 0x01ffff000000ff01, 0x01ffff0000000000,
        0x01ffff0000000001, 0x01ffff00000001ff, 0x01ffff0000000100, 0x01ffff00000100ff,
        0x01ffff0000010001, 0x01ffff0000010100, 0x01ffff0001ff0000, 0x01ffff0001ff0100,
        0x01ffff00010000ff, 0x01ffff0001000001, 0x01ffff0001000100, 0x01ffff0001010000,
        0x01ffff01ffffffff, 0x01ffff01ffffff01, 0x01ffff01ffff01ff, 0x01ffff01ffff0101,
        0x01ffff01ff000000, 0x01ffff01ff01ffff, 0x01ffff01ff01ff01, 0x01ffff01ff0101ff,
        0x01ffff01ff010101, 0x01ffff010000ff00, 0x01ffff01000000ff, 0x01ffff0100000100,
        0x01ffff0100010000, 0x01ffff0101ffffff, 0x01ffff0101ffff01, 0x01ffff0101ff01ff,
        0x01ffff0101ff0101, 0x01ffff0101000000, 0x01ffff010101ffff, 0x01ffff010101ff01,
        0x01ffff01010101ff, 0x01ffff0101010101, 0x01ff00ffff0000ff, 0x01ff00ffff000100,
        0x01ff00ff00ffff00, 0x01ff00ff00ff00ff, 0x01ff00ff0000ff00, 0x01ff00ff00000000,
        0x01ff00ff00000101, 0x01ff00ff0001ff00, 0x01ff00ff000100ff, 0x01ff00ff00010100,
        0x01ff00ff010000ff, 0x01ff00ff01000100, 0x01ff0000ffffff00, 0x01ff0000ffff0100,
        0x01ff0000ff00ff01, 0x01ff0000ff000000, 0x01ff0000ff000101, 0x01ff0000ff010001,
        0x01ff0000ff010100, 0x01ff000000ffffff, 0x01ff000000ffff00, 0x01ff000000ff0000,
        0x01ff000000ff01ff, 0x01ff00000000ff00, 0x01ff0000000000ff, 0x01ff000000000000,
        0x01ff000000000001, 0x01ff000000000100, 0x01ff000000000101, 0x01ff000000010000,
        0x01ff000000010001, 0x01ff0000000101ff, 0x01ff000000010101, 0x01ff000001ffff00,
        0x01ff000001ff00ff, 0x01ff000001ff0001, 0x01ff000001ff0100, 0x01ff00000100ffff,
        0x01ff00000100ff01, 0x01ff000001000000, 0x01ff0000010001ff, 0x01ff000001010001,
        0x01ff0001ff00ff00, 0x01ff0001ff000001, 0x01ff0001ff000100, 0x01ff0001ff010000,
        0x01ff000100ffff00, 0x01ff000100ff00ff, 0x01ff000100ff0100, 0x01ff000100ff0101,
        0x01ff00010000ffff, 0x01ff000100000000, 0x01ff000100000100, 0x01ff000100000101,
        0x01ff00010001ff00, 0x01ff000100010001, 0x01ff000100010101, 0x01ff000101ff0000,
        0x01ff00010100ff00, 0x01ff000101000101, 0x01ff0001010100ff, 0x01ff01ffffffffff,
        0x01ff01ffffffff01, 0x01ff01ffffff01ff, 0x01ff01ffffff0101, 0x01ff01ffff000000,
        0x01ff01ffff01ffff, 0x01ff01ffff01ff01, 0x01ff01ffff0101ff, 0x01ff01ffff010101,
        0x01ff01ff00ffff00, 0x01ff01ff00ff0000, 0x01ff01ff0000ff00, 0x01ff01ff000000ff,
        0x01ff01ff00000100, 0x01ff01ff00010000, 0x01ff01ff00010100, 0x01ff01ff01ffffff,
        0x01ff01ff01ffff01, 0x01ff01ff01ff01ff, 0x01ff01ff01ff0101, 0x01ff01ff01000000,
        0x01ff01ff0101ffff, 0x01ff01ff0101ff01, 0x01ff01ff010101ff, 0x01ff01ff01010101,
        0x01ff0100ffff0000, 0x01ff0100ffff0001, 0x01ff0100ff00ff00, 0x01ff0100ff0000ff,
        0x01ff0100ff000001, 0x01ff0100ff010000, 0x01ff010000ffff00, 0x01ff010000ff00ff,
        0x01ff010000ff0001, 0x01ff010000ff0100, 0x01ff01000000ffff, 0x01ff01000000ff01,
        0x01ff010000000000, 0x01ff010000000101, 0x01ff01000001ff00, 0x01ff0100000100ff,
        0x01ff010001ff0000, 0x01ff010001000001, 0x01ff010001000100, 0x01ff010001010000,
        0x01ff0101ffffffff, 0x01ff0101ffffff01, 0x01ff0101ffff01ff, 0x01ff0101ffff0101,
        0x01ff0101ff000000, 0x01ff0101ff01ffff, 0x01ff0101ff01ff01, 0x01ff0101ff0101ff,
        0x01ff0101ff010101, 0x01ff010100ff0000, 0x01ff01010000ff00, 0x01ff0101000000ff,
        0x01ff010100000001, 0x01ff010101ffffff, 0x01ff010101ffff01, 0x01ff010101ff01ff,
        0x01ff010101ff0101, 0x01ff010101000000, 0x01ff01010101ffff, 0x01ff01010101ff01,
        0x01ff0101010101ff, 0x01ff010101010101, 0x0100ffffffff0000, 0x0100ffffff00ff00,
        0x0100ffffff000001, 0x0100ffffff0001ff, 0x0100ffffff000100, 0x0100ffffff010000,
        0x0100ffff00ffff00, 0x0100ffff00ff0001, 0x0100ffff00ff0100, 0x0100ffff00000000,
        0x0100ffff000001ff, 0x0100ffff00000101, 0x0100ffff00010100, 0x0100ffff00010101,
        0x0100ffff01ff0000, 0x0100ffff0100ff00, 0x0100ffff010000ff, 0x0100ffff01000001,
        0x0100ffff01000100, 0x0100ffff01010000, 0x0100ff00ffffff00, 0x0100ff00ffff00ff,
        0x0100ff00ffff0001, 0x0100ff00ffff0100, 0x0100ff00ff00ffff, 0x0100ff00ff000000,
        0x0100ff00ff0001ff, 0x0100ff00ff000101, 0x0100ff00ff01ff00, 0x0100ff00ff0100ff,
        0x0100ff00ff010001, 0x0100ff00ff010100, 0x0100ff0000ffffff, 0x0100ff0000ff0000,
        0x0100ff000000ffff, 0x0100ff000000ff00, 0x0100ff00000000ff, 0x0100ff0000000000,
        0x0100ff0000000001, 0x0100ff0000000100, 0x0100ff000001ff01, 0x0100ff0000010000,
        0x0100ff0001ff00ff, 0x0100ff0001ff0001, 0x0100ff000100ff01, 0x0100ff0001000000,
        0x0100ff00010001ff, 0x0100ff000101ff00, 0x0100ff00010100ff, 0x0100ff0001010001,
        0x0100ff0001010100, 0x0100ff01ffff0000, 0x0100ff01ff00ff00, 0x0100ff01ff0000ff,
        0x0100ff01ff000100, 0x0100ff01ff010000, 0x0100ff0100ff00ff, 0x0100ff0100ff0001,
        0x0100ff0100ff0100, 0x0100ff010000ffff, 0x0100ff010000ff01, 0x0100ff0100000000,
        0x0100ff01000001ff, 0x0100ff0100010001, 0x0100ff0100010100, 0x0100ff0101ff0000,
        0x0100ff01010000ff, 0x0100ff0101000001, 0x0100ff0101010100, 0x010000ffffffff00,
        0x010000ffffff00ff, 0x010000ffffff0001, 0x010000ffff00ffff, 0x010000ffff000000,
        0x010000ffff0001ff, 0x010000ffff010001, 0x010000ff00ffffff, 0x010000ff00ff0101,
        0x010000ff0000ff00, 0x010000ff000000ff, 0x010000ff00000000, 0x010000ff00000001,
        0x010000ff000001ff, 0x010000ff00000100, 0x010000ff0001ffff, 0x010000ff0001ff00,
        0x010000ff0001ff01, 0x010000ff00010000, 0x010000ff01ff00ff, 0x010000ff01ff0001,
        0x010000ff0100ff01, 0x010000ff010000ff, 0x010000ff01000000, 0x010000ff010001ff,
        0x010000ff0101ff00, 0x010000ff01010100, 0x01000000ffffffff, 0x01000000ffff0000,
        0x01000000ffff01ff, 0x01000000ffff0101, 0x01000000ff00ffff, 0x01000000ff00ff00,
        0x01000000ff0000ff, 0x01000000ff000000, 0x01000000ff000001, 0x01000000ff000100,
        0x01000000ff01ff00, 0x01000000ff010000, 0x01000000ff010100, 0x01000000ff010101,
        0x0100000000ffff00, 0x0100000000ff00ff, 0x0100000000ff0000, 0x0100000000ff0001,
        0x0100000000ff0100, 0x010000000000ffff, 0x010000000000ff00, 0x010000000000ff01,
        0x01000000000000ff, 0x0100000000000000, 0x0100000000000001, 0x01000000000001ff,
        0x0100000000000100, 0x0100000000000101, 0x010000000001ff00, 0x01000000000100ff,
        0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000001ffff00,
        0x0100000001ff0000, 0x0100000001ff01ff, 0x010000000100ff00, 0x010000000100ff01,
        0x01000000010000ff, 0x0100000001000000, 0x0100000001000001, 0x0100000001000100,
        0x0100000001000101, 0x010000000101ffff, 0x010000000101ff01, 0x0100000001010000,
        0x01000000010101ff, 0x0100000001010101, 0x01000001ffffff00, 0x01000001ffff00ff,
        0x01000001ff00ffff, 0x01000001ff000000, 0x01000001ff000100, 0x01000001ff01ffff,
        0x01000001ff010001, 0x01000001ff010100, 0x0100000100ff0000, 0x0100000100ff01ff,
        0x0100000100ff0100, 0x010000010000ff00, 0x010000010000ff01, 0x0100000100000000,
        0x0100000100000001, 0x0100000100000100, 0x0100000100010000, 0x01000001000101ff,
        0x0100000101ffff01, 0x0100000101ff00ff, 0x0100000101ff0100, 0x0100000101ff0101,
        0x010000010100ff01, 0x01000001010000ff, 0x0100000101000000, 0x01000001010100ff,
        0x0100000101010001, 0x0100000101010100, 0x010001ffffff0000, 0x010001ffff000001,
        0x010001ffff000100, 0x010001ffff010000, 0x010001ff00ffff00, 0x010001ff00ff0001,
        0x010001ff0000ffff, 0x010001ff0000ff01, 0x010001ff00000000, 0x010001ff00000001,
        0x010001ff00000101, 0x010001ff000100ff, 0x010001ff00010000, 0x010001ff01ff0000,
        0x010001ff0100ff00, 0x010001ff01000001, 0x010001ff01000100, 0x010001ff01010000,
        0x01000100ffff00ff, 0x01000100ffff0001, 0x01000100ffff0100, 0x01000100ff00ffff,
        0x01000100ff00ff01, 0x01000100ff000000, 0x01000100ff0001ff, 0x01000100ff000101,
        0x01000100ff01ffff, 0x01000100ff01ff00, 0x01000100ff0100ff, 0x01000100ff010001,
        0x0100010000ffffff, 0x0100010000ffff01, 0x0100010000ff0000, 0x0100010000ff01ff,
        0x0100010000ff0101, 0x010001000000ff00, 0x01000100000000ff, 0x0100010000000000,
        0x0100010000000001, 0x0100010000000100, 0x010001000001ff01, 0x0100010000010000,
        0x0100010000010001, 0x0100010000010101, 0x0100010001ffff00, 0x0100010001ff00ff,
        0x010001000100ffff, 0x010001000100ff01, 0x0100010001000000, 0x0100010001000101,
        0x010001000101ff00, 0x0100010001010001, 0x01000101ffff0000, 0x01000101ff000000,
        0x01000101ff010000, 0x0100010100ff00ff, 0x0100010100ff0001, 0x0100010100ff0100,
        0x010001010000ffff, 0x0100010100000000, 0x01000101000001ff, 0x010001010001ff00,
        0x0100010101ff0000, 0x010001010100ff00, 0x01000101010000ff, 0x0100010101000000,
        0x0100010101000001, 0x0101ffffffffffff, 0x0101ffffffffff01, 0x0101ffffffff01ff,
        0x0101ffffffff0101, 0x0101ffffff000000, 0x0101ffffff01ffff, 0x0101ffffff01ff01,
        0x0101ffffff0101ff, 0x0101ffffff010101, 0x0101ffff00ff0000, 0x0101ffff0000ff00,
        0x0101ffff000000ff, 0x0101ffff00000001, 0x0101ffff00000100, 0x0101ffff01ffffff,
        0x0101ffff01ffff01, 0x0101ffff01ff01ff, 0x0101ffff01ff0101, 0x0101ffff01000000,
        0x0101ffff0101ffff, 0x0101ffff0101ff01, 0x0101ffff010101ff, 0x0101ffff01010101,
        0x0101ff00ffff0000, 0x0101ff00ffff0100, 0x0101ff00ff00ff00, 0x0101ff00ff0000ff,
        0x0101ff00ff000001, 0x0101ff00ff000100, 0x0101ff00ff000101, 0x0101ff0000ff0001,
        0x0101ff0000ff0100, 0x0101ff000000ff00, 0x0101ff0000000000, 0x0101ff00000001ff,
        0x0101ff0000000101, 0x0101ff000001ff00, 0x0101ff00000100ff, 0x0101ff0001ff0000,
        0x0101ff000100ffff, 0x0101ff000100ff01, 0x0101ff0001000001, 0x0101ff0001000100,
        0x0101ff01ffffff01, 0x0101ff01ffff01ff, 0x0101ff01ffff0101, 0x0101ff01ff00ffff,
        0x0101ff01ff000100, 0x0101ff01ff01ff01, 0x0101ff01ff0101ff, 0x0101ff01ff010101,
        0x0101ff0100ff0000, 0x0101ff010000ff00, 0x0101ff0100000001, 0x0101ff0100000100,
        0x0101ff0100010000, 0x0101ff0101ffffff, 0x0101ff0101ffff01, 0x0101ff0101ff01ff,
        0x0101ff0101ff0101, 0x0101ff0101000000, 0x0101ff010101ffff, 0x0101ff010101ff01,
        0x0101ff01010101ff, 0x0101ff0101010101, 0x010100ffff000100, 0x010100ffff010000,
        0x010100ff00ffff00, 0x010100ff00ff00ff, 0x010100ff0000ffff, 0x010100ff000000ff,
        0x010100ff00000000, 0x010100ff000001ff, 0x010100ff00000101, 0x010100ff0001ff00,
        0x010100ff00010000, 0x010100ff00010001, 0x010100ff000101ff, 0x010100ff00010100,
        0x010100ff01ff0000, 0x01010000ffff0001, 0x01010000ffff0100, 0x01010000ff00ffff,
        0x01010000ff00ff01, 0x01010000ff000000, 0x01010000ff0001ff, 0x01010000ff010001,
        0x01010000ff010100, 0x0101000000ffff01, 0x0101000000ff0000, 0x010100000000ff00,
        0x01010000000000ff, 0x0101000000000000, 0x0101000000000001, 0x0101000000000100,
        0x0101000000010000, 0x0101000000010101, 0x0101000001ffff00, 0x0101000001ff00ff,
        0x0101000001ff0000, 0x0101000001ff0001, 0x0101000001ff0100, 0x010100000100ff01,
        0x0101000001000000, 0x01010000010001ff, 0x01010001ffff0000, 0x01010001ff00ff00,
        0x01010001ff000001, 0x01010001ff000101, 0x01010001ff01ff00, 0x01010001ff010000,
        0x0101000100ff00ff, 0x0101000100ff0001, 0x0101000100ff0101, 0x010100010000ff01,
        0x0101000100000000, 0x0101000100000001, 0x01010001000001ff, 0x010100010001ffff,
        0x010100010001ff01, 0x0101000101ff0001, 0x010100010100ffff, 0x0101000101000000,
        0x0101000101000001, 0x0101000101000100, 0x010100010101ff00, 0x01010001010100ff,
        0x0101000101010001, 0x010101ffffffffff, 0x010101ffffffff01, 0x010101ffffff01ff,
        0x010101ffffff0101, 0x010101ffff01ffff, 0x010101ffff01ff01, 0x010101ffff0101ff,
        0x010101ffff010101, 0x010101ff0000ff00, 0x010101ff000000ff, 0x010101ff00000001,
        0x010101ff00000100, 0x010101ff01ffffff, 0x010101ff01ffff01, 0x010101ff01ff01ff,
        0x010101ff01ff0101, 0x010101ff01000000, 0x010101ff0101ffff, 0x010101ff0101ff01,
        0x010101ff010101ff, 0x010101ff01010101, 0x01010100ffff0000, 0x01010100ff0000ff,
        0x01010100ff000100, 0x01010100ff01ff00, 0x01010100ff010000, 0x0101010000ffff00,
        0x010101000000ffff, 0x0101010000000000, 0x0101010000000101, 0x010101000001ff00,
        0x0101010000010001, 0x0101010000010100, 0x010101000100ffff, 0x0101010001000001,
        0x01010101ffffffff, 0x01010101ffffff01, 0x01010101ffff01ff, 0x01010101ffff0101,
        0x01010101ff01ffff, 0x01010101ff01ff01, 0x01010101ff0101ff, 0x01010101ff010101,
        0x010101010000ff00, 0x01010101000000ff, 0x0101010100000001, 0x0101010101ffffff,
        0x0101010101ffff01, 0x0101010101ff01ff, 0x0101010101ff0101, 0x0101010101000000,
        0x010101010101ffff, 0x010101010101ff01, 0x01010101010101ff, 0x0101010101010101,
        };
        if (type == GGML_TYPE_IQ1_S) {
        const uint8_t *p = (const uint8_t *)weight;
        int nsb = n / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh; kmemcpy(&dh, p, 2);
            float d = fp16_to_fp32(dh);
            const uint8_t  *qs     = p + 2;
            const uint8_t  *qh_raw = p + 34;
            p += 50;
            const float *xb = x + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                uint16_t qh_val; kmemcpy(&qh_val, qh_raw + ib32 * 2, 2);
                float dl    = d * (float)(2 * ((qh_val >> 12) & 7) + 1);
                float delta = (qh_val & 0x8000) ? -0.125f : 0.125f;
                for (int l = 0; l < 4; l++) {
                    unsigned grid_idx = (unsigned)qs[4*ib32 + l]
                                      | ((unsigned)((qh_val >> (3*(3-l))) & 7) << 8);
                    const int8_t *gv = (const int8_t *)(kgrid_iq1s + grid_idx);
                    const float  *xs = xb + ib32 * 32 + l * 8;
                    for (int j = 0; j < 8; j++)
                        sum += xs[j] * dl * ((float)gv[j] + delta);
                }
            }
        }
        } else {
        /* IQ1_M: 256 elements/super-block, 56 bytes. No explicit fp16 d.
         * Layout: qs[32] + qh[16] + scales[8].
         * d packed into top 4 bits of each scales uint16_t (viewed as uint16_t[4]).
         * dl1/dl2 use 3-bit scale nibbles; uses same kgrid_iq1s[2048] lookup. */
        const uint8_t *p_m = (const uint8_t *)weight;
        int nsb_m = n / 256;
        for (int b = 0; b < nsb_m; b++) {
            const uint8_t  *qs_m = p_m;
            const uint8_t  *qh_m = p_m + 32;
            const uint16_t *sc_m = (const uint16_t *)(p_m + 48);
            p_m += 56;
            uint16_t scale_u16 = (uint16_t)((sc_m[0] >> 12)
                               | ((sc_m[1] >> 8) & 0x00f0u)
                               | ((sc_m[2] >> 4) & 0x0f00u)
                               | (sc_m[3] & 0xf000u));
            float d_m = fp16_to_fp32(scale_u16);
            const float *xb_m = x + b * 256;
            for (int ib = 0; ib < 8; ib++) {
                float dl1 = d_m * (float)(2 * ((sc_m[ib/2] >> (6*(ib%2)+0)) & 7u) + 1);
                float dl2 = d_m * (float)(2 * ((sc_m[ib/2] >> (6*(ib%2)+3)) & 7u) + 1);
                uint16_t idx0 = (uint16_t)qs_m[0] | (uint16_t)(((uint16_t)qh_m[0] << 8) & 0x700u);
                uint16_t idx1 = (uint16_t)qs_m[1] | (uint16_t)(((uint16_t)qh_m[0] << 4) & 0x700u);
                uint16_t idx2 = (uint16_t)qs_m[2] | (uint16_t)(((uint16_t)qh_m[1] << 8) & 0x700u);
                uint16_t idx3 = (uint16_t)qs_m[3] | (uint16_t)(((uint16_t)qh_m[1] << 4) & 0x700u);
                float delta0 = (qh_m[0] & 0x08) ? -0.125f : 0.125f;
                float delta1 = (qh_m[0] & 0x80) ? -0.125f : 0.125f;
                float delta2 = (qh_m[1] & 0x08) ? -0.125f : 0.125f;
                float delta3 = (qh_m[1] & 0x80) ? -0.125f : 0.125f;
                const uint16_t idxs[4] = {idx0, idx1, idx2, idx3};
                const float    dlts[4] = {delta0, delta1, delta2, delta3};
                for (int l = 0; l < 2; l++) {
                    const int8_t *gv_m = (const int8_t *)(kgrid_iq1s + idxs[l]);
                    const float  *xs_m = xb_m + ib * 32 + l * 8;
                    for (int j = 0; j < 8; j++)
                        sum += xs_m[j] * dl1 * ((float)gv_m[j] + dlts[l]);
                }
                for (int l = 2; l < 4; l++) {
                    const int8_t *gv_m = (const int8_t *)(kgrid_iq1s + idxs[l]);
                    const float  *xs_m = xb_m + ib * 32 + l * 8;
                    for (int j = 0; j < 8; j++)
                        sum += xs_m[j] * dl2 * ((float)gv_m[j] + dlts[l]);
                }
                qs_m += 4;
                qh_m += 2;
            }
        }
        }
        break;
    }
    case GGML_TYPE_Q5_0: {
        /* Q5_0: 32 elements/block, 22 bytes.
         * Layout: fp16 d (2) + qh uint32 (4) + qs[16].
         * qh bit j = high bit of element j (j=0..15), bit j+16 = high bit of element j+16.
         * Element j:   q = ((qs[j]&0xF) | (((qh>>j)&1)<<4)) - 16
         * Element j+16:q = ((qs[j]>>4) | (((qh>>(j+16))&1)<<4)) - 16 */
        const uint8_t *p = (const uint8_t *)weight;
        int nb32 = n / 32;
        for (int b = 0; b < nb32; b++) {
            uint16_t dh; kmemcpy(&dh, p, 2);
            float d = fp16_to_fp32(dh);
            uint32_t qh; kmemcpy(&qh, p + 2, 4);
            const uint8_t *qs = p + 6;
            p += 22;
            const float *xb = x + b * 32;
            float bs = 0.0f;
            for (int j = 0; j < 16; j++) {
                int q_lo = (int)((qs[j] & 0xF) | (((qh >> j) & 1u) << 4)) - 16;
                int q_hi = (int)((qs[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                bs += xb[j] * (float)q_lo + xb[j + 16] * (float)q_hi;
            }
            sum += d * bs;
        }
        break;
    }
    case GGML_TYPE_Q5_1: {
        /* Q5_1: 32 elements/block, 24 bytes.
         * Layout: fp16 d (2) + fp16 m (2) + qh uint32 (4) + qs[16].
         * Values are unsigned 5-bit + min offset m.
         * Element j:   q = (qs[j]&0xF) | (((qh>>j)&1)<<4)
         * Element j+16:q = (qs[j]>>4)  | (((qh>>(j+16))&1)<<4)
         * y = d*q + m */
        const uint8_t *p = (const uint8_t *)weight;
        int nb32 = n / 32;
        for (int b = 0; b < nb32; b++) {
            uint16_t dh, mh; kmemcpy(&dh, p, 2); kmemcpy(&mh, p + 2, 2);
            float d = fp16_to_fp32(dh), m = fp16_to_fp32(mh);
            uint32_t qh; kmemcpy(&qh, p + 4, 4);
            const uint8_t *qs = p + 8;
            p += 24;
            const float *xb = x + b * 32;
            float bs = 0.0f, xs = 0.0f;
            for (int j = 0; j < 16; j++) {
                int q_lo = (int)((qs[j] & 0xF) | (((qh >> j) & 1u) << 4));
                int q_hi = (int)((qs[j] >> 4)  | (((qh >> (j + 16)) & 1u) << 4));
                bs += xb[j] * (float)q_lo + xb[j + 16] * (float)q_hi;
                xs += xb[j] + xb[j + 16];
            }
            sum += d * bs + m * xs;
        }
        break;
    }
    case GGML_TYPE_Q8_1: {
        /* Q8_1: 32 elements/block, 36 bytes.
         * Layout: fp16 d (2) + fp16 s (2) + int8_t qs[32].
         * s = d * sum(qs) is a cached sum, not needed for dequant.
         * y = d * qs[j] */
        const uint8_t *p = (const uint8_t *)weight;
        int nb32 = n / 32;
        for (int b = 0; b < nb32; b++) {
            uint16_t dh; kmemcpy(&dh, p, 2);
            float d = fp16_to_fp32(dh);
            const int8_t *qs = (const int8_t *)(p + 4); /* skip d(2) + s(2) */
            p += 36;
            const float *xb = x + b * 32;
            float bs = 0.0f;
            for (int j = 0; j < 32; j++) bs += xb[j] * (float)qs[j];
            sum += d * bs;
        }
        break;
    }
    case GGML_TYPE_IQ2_S: {
        /* IQ2_S: 256 elements/super-block, 82 bytes.
         * Layout: fp16 d(2) + qs[64] + qh[8] + scales[8].
         * qs[0..31]=grid-idx low bytes (4 per ib32), qs[32..63]=sign bytes.
         * qh[ib32] has 4x2 high-bit pairs; scales[ib32] = 4-bit scale pair.
         * grid_idx = qs[l] | ((qh[ib32]<<(8-2*l))&0x300).
         * Grid values via kgrid_iq2s (kgrid_2bit_1024), decoded as {1,3,5,7}. */
        static const uint8_t  iq2s_vals[4] = {1, 3, 5, 7};
        static const uint16_t kgrid_iq2s[1024] = {
            0,   2,   5,   8,  10,  17,  20,  22,  25,  32,  34,  37,  40,  65,  68,  70,
           73,  80,  82,  85,  88,  97, 100, 102, 105, 128, 130, 133, 136, 145, 148, 160,
          165, 170, 257, 260, 262, 265, 272, 274, 277, 280, 289, 292, 320, 322, 325, 328,
          337, 340, 342, 345, 352, 357, 360, 385, 388, 400, 402, 405, 417, 420, 512, 514,
          517, 520, 529, 532, 544, 554, 577, 580, 582, 585, 592, 597, 640, 645, 650, 660,
          674,1025,1028,1030,1033,1040,1042,1045,1048,1057,1060,1062,1065,1088,1090,1093,
         1096,1098,1105,1108,1110,1113,1120,1122,1125,1153,1156,1158,1161,1168,1173,1176,
         1185,1188,1280,1282,1285,1288,1290,1297,1300,1302,1305,1312,1317,1320,1345,1348,
         1350,1353,1360,1362,1365,1368,1377,1380,1408,1410,1413,1416,1425,1428,1440,1537,
         1540,1542,1545,1552,1557,1600,1605,1608,1617,1620,1632,1665,1668,1680,2048,2050,
         2053,2056,2065,2068,2070,2073,2080,2085,2090,2113,2116,2118,2121,2128,2130,2133,
         2136,2145,2148,2176,2181,2196,2218,2305,2308,2320,2322,2325,2328,2337,2368,2373,
         2376,2385,2388,2400,2433,2448,2560,2577,2580,2594,2600,2602,2640,2713,4097,4100,
         4102,4105,4112,4114,4117,4120,4129,4132,4134,4160,4162,4165,4168,4177,4180,4182,
         4185,4192,4194,4197,4200,4225,4228,4230,4240,4245,4248,4257,4260,4352,4354,4357,
         4360,4362,4369,4372,4374,4377,4384,4386,4389,4392,4417,4420,4422,4425,4432,4434,
         4437,4440,4449,4452,4480,4482,4485,4488,4497,4500,4609,4612,4617,4624,4629,4641,
         4644,4672,4677,4689,4692,4737,4740,4752,5120,5122,5125,5128,5137,5140,5142,5145,
         5152,5157,5160,5185,5188,5190,5193,5200,5202,5205,5208,5217,5220,5248,5250,5253,
         5256,5265,5268,5280,5377,5380,5382,5385,5392,5394,5397,5400,5409,5412,5440,5442,
         5445,5448,5457,5460,5472,5505,5508,5520,5632,5637,5640,5649,5652,5664,5697,5700,
         5712,5760,5802,6145,6148,6150,6153,6160,6165,6168,6177,6208,6210,6213,6216,6225,
         6228,6240,6273,6276,6400,6402,6405,6408,6417,6420,6432,6465,6468,6480,6505,6562,
         6660,6672,6720,6742,8192,8194,8197,8200,8209,8212,8214,8217,8224,8229,8234,8257,
         8260,8272,8274,8277,8292,8320,8330,8340,8362,8449,8452,8464,8466,8469,8481,8512,
         8514,8517,8529,8532,8544,8577,8580,8592,8704,8714,8738,8744,8746,8772,8784,8840,
         8842,8872,9217,9220,9222,9225,9232,9237,9240,9249,9252,9280,9282,9285,9288,9297,
         9300,9312,9345,9348,9360,9472,9477,9480,9489,9492,9504,9537,9540,9552,9574,9600,
         9729,9732,9744,9792,9817,10240,10245,10257,10260,10305,10308,10320,10378,10410,10497,10500,
        10512,10645,10762,10786,10852,10888,10890,
        16385,16388,16390,16393,16400,16402,16405,16408,16410,16417,16420,16422,16448,16450,16453,16456,
        16458,16465,16468,16470,16473,16480,16482,16485,16513,16516,16528,16533,16536,16545,16548,16640,
        16642,16645,16648,16657,16660,16662,16665,16672,16674,16677,16705,16708,16710,16713,16720,16722,
        16725,16728,16737,16740,16768,16770,16773,16776,16785,16788,16800,16897,16900,16912,16914,16917,
        16920,16932,16960,16965,16968,16977,16980,16992,17025,17028,
        17408,17410,17413,17416,17418,17425,17428,17430,17433,17440,17442,17445,17448,17473,17476,17478,
        17481,17488,17490,17493,17496,17505,17508,17536,17538,17541,17544,17553,17556,17568,17665,17668,
        17670,17673,17680,17682,17685,17688,17697,17700,17728,17730,17733,17736,17745,17748,17760,17770,
        17793,17796,17808,17920,17922,17925,17928,17937,17940,17952,17985,17988,18000,18048,18085,
        18433,18436,18441,18448,18450,18453,18456,18465,18468,18496,18498,18501,18504,18513,18516,18528,
        18564,18576,18688,18690,18693,18696,18705,18708,18720,18753,18756,18768,18816,18838,18945,18948,
        18960,19008,
        20480,20482,20485,20488,20497,20500,20502,20505,20512,20514,20517,20520,20545,20548,20550,20553,
        20560,20562,20565,20568,20577,20580,20608,20610,20613,20616,20625,20628,20737,20740,20742,20745,
        20752,20754,20757,20760,20769,20772,20800,20802,20805,20808,20817,20820,20832,20865,20868,20880,
        20992,20997,21000,21009,21012,21024,21057,21060,21072,21097,21120,
        21505,21508,21510,21513,21520,21522,21525,21528,21537,21540,21568,21570,21573,21576,21585,21588,
        21600,21633,21636,21648,21760,21762,21765,21768,21777,21780,21792,21825,21828,21840,21888,22017,
        22020,22032,22054,22080,
        22528,22530,22533,22536,22545,22548,22560,22593,22596,22608,22618,22656,22785,22788,22800,22848,
        23040,23065,23173,23208,
        24577,24580,24582,24592,24594,24597,24600,24609,24612,24640,24645,24648,24657,24660,24672,24708,
        24720,24832,24834,24837,24840,24849,24852,24864,24897,24900,24912,24960,24985,25092,25104,25152,
        25174,25249,
        25600,25605,25608,25617,25620,25632,25665,25668,25680,25728,25857,25860,25872,25920,25930,25960,
        26002,26112,26260,
        26625,26628,26640,26725,26776,26880,26922,27202,27297,
        32768,32770,32773,32776,32785,32788,32793,32800,32805,32833,32836,32848,32850,32853,32856,32865,
        32896,32901,32913,32916,33025,33028,33033,33040,33042,33045,33048,33057,33060,33088,33090,33093,
        33096,33105,33108,33153,33156,33168,33193,33280,33285,33290,33297,33300,33345,33348,33360,
        33793,33796,33798,33801,33808,33810,33813,33816,33825,33856,33858,33861,33864,33873,33876,33888,
        33921,33924,33936,34048,34050,34053,34056,34065,34068,34080,34113,34116,34128,34176,34186,
        34305,34308,34320,34345,34368,34816,34821,34833,34836,34881,34884,34896,34978,35073,35076,35136,
        35173,35362,35416,35418,35458,35490,
        36865,36868,36873,36880,36882,36885,36888,36900,36928,36930,36933,36936,36945,36948,36960,36993,
        36996,37008,37120,37125,37137,37140,37185,37188,37200,37210,37377,37380,37392,37440,37542,
        37888,37890,37893,37896,37905,37908,37920,37953,37956,37968,38016,38038,38145,38148,38160,38208,
        38296,38305,38400,38470,38500,
        38913,38916,38928,38950,38976,39081,39168,39241,39250,39568,
        40960,40965,40970,40980,40994,41002,41025,41028,41040,41122,41130,41280,41317,41474,41482,41506,
        41512,41514,41602,41608,41610,41640,41985,41988,42000,42048,42121,42148,42240,42265,42577,43018,
        43048,43170,43348,43398,43528,43530,43552,43554,43560,43656,43690,
        };
        const uint8_t *p_s = (const uint8_t *)weight;
        int nsb_s = n / 256;
        for (int b = 0; b < nsb_s; b++) {
            uint16_t dh; kmemcpy(&dh, p_s, 2);
            float d = fp16_to_fp32(dh);
            const uint8_t *qs_s  = p_s + 2;   /* grid low bytes [0..31] */
            const uint8_t *sn_s  = p_s + 34;  /* sign bytes [32..63] = qs+32 */
            const uint8_t *qh_s  = p_s + 66;  /* high 2 bits per group */
            const uint8_t *sc_s  = p_s + 74;  /* 4-bit scale pairs */
            p_s += 82;
            const float *xb_s = x + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                float db0 = d * (0.5f + (float)(sc_s[ib32] & 0xf)) * 0.25f;
                float db1 = d * (0.5f + (float)(sc_s[ib32] >> 4))  * 0.25f;
                for (int l = 0; l < 4; l++) {
                    float dl = (l < 2) ? db0 : db1;
                    unsigned gi = (unsigned)qs_s[l]
                                | ((unsigned)(qh_s[ib32] << (8 - 2*l)) & 0x300u);
                    uint16_t gv = kgrid_iq2s[gi];
                    uint8_t  sn = sn_s[l];
                    const float *xs_s = xb_s + ib32 * 32 + l * 8;
                    for (int j = 0; j < 8; j++) {
                        float w = (float)iq2s_vals[(gv >> (2*j)) & 3];
                        if (sn & (1u << j)) w = -w;
                        sum += xs_s[j] * dl * w;
                    }
                }
                qs_s += 4;
                sn_s += 4;
            }
        }
        break;
    }
    default:
        break;
    }
    return sum;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  AVX2+FMA Optimized Dot Products (2× throughput vs SSE2)                    */
/* ─────────────────────────────────────────────────────────────────────────── */

#ifndef __aarch64__

__attribute__((target("avx2,fma")))
static inline v8f v8f_setzero(void) { return (v8f){0,0,0,0,0,0,0,0}; }

__attribute__((target("avx2,fma")))
static inline v8f v8f_set1(float x) { return (v8f){x,x,x,x,x,x,x,x}; }

__attribute__((target("avx2,fma")))
static inline float v8f_reduce(v8f v) {
    union { v8f vec; float f[8]; } u;
    u.vec = v;
    return (u.f[0]+u.f[1]+u.f[2]+u.f[3])+(u.f[4]+u.f[5]+u.f[6]+u.f[7]);
}

/* AVX2 min/max per-element */
__attribute__((target("avx2,fma")))
static inline v8f v8f_min(v8f a, v8f b) {
    v8f r; __asm__("vminps %2, %1, %0" : "=x"(r) : "x"(a), "x"(b));
    return r;
}
__attribute__((target("avx2,fma")))
static inline v8f v8f_max(v8f a, v8f b) {
    v8f r; __asm__("vmaxps %2, %1, %0" : "=x"(r) : "x"(a), "x"(b));
    return r;
}

/* AVX2 vectorized exp: range reduction exp(x)=2^n * exp(r), minimax poly */
__attribute__((target("avx2,fma")))
static inline v8f v8f_expf(v8f x) {
    x = v8f_max(x, v8f_set1(-86.0f));
    x = v8f_min(x, v8f_set1(86.0f));
    /* n = round(x / ln2) */
    v8f nf;
    __asm__("vroundps $0, %1, %0" : "=x"(nf) : "x"(x * v8f_set1(1.4426950408f)));
    /* r = x - n*ln2 (Cody-Waite two-part) */
    v8f r = x - nf * v8f_set1(0.693145751953125f);
    r = r - nf * v8f_set1(1.428606765330187e-06f);
    /* Polynomial: exp(r) for r in [-ln2/2, ln2/2] */
    v8f r2 = r * r;
    v8f r3 = r2 * r;
    v8f p = v8f_set1(1.0f) + r + r2 * v8f_set1(0.5f)
          + r3 * v8f_set1(0.16666666f) + r2 * r2 * v8f_set1(0.041666666f)
          + r2 * r3 * v8f_set1(0.008333333f) + r3 * r3 * v8f_set1(0.001388889f);
    /* 2^n: convert n to int, add 127 bias, shift into IEEE754 exponent field */
    union { int i[8]; v8f v; } ni, scale;
    __asm__("vcvtps2dq %1, %0" : "=x"(ni.v) : "x"(nf));
    for (int j = 0; j < 8; j++)
        scale.i[j] = (ni.i[j] + 127) << 23;
    return p * scale.v;
}

/* SIMD int8→float: vpmovsxbd (sign-extend 8×i8→8×i32) + vcvtdq2ps
 * Replaces 8 scalar (float)cast operations with 2 SIMD instructions */
__attribute__((target("avx2,fma")))
static inline v8f v8f_load_i8(const int8_t *src)
{
    v8f result;
    __asm__ volatile(
        "vpmovsxbd (%[s]), %[r]\n\t"
        "vcvtdq2ps %[r], %[r]"
        : [r] "=x"(result) : [s] "r"(src)
    );
    return result;
}

/* Fast branchless fp16→fp32 for quantization scales (normal values only).
 * ~8 ops vs ~20 for the full conversion. Safe for Q8_0/Q4_0 scale values
 * which are always normal fp16 (never denorm/inf/nan in practice). */
static inline float fp16_to_fp32_fast(uint16_t h)
{
    uint32_t bits = ((uint32_t)(h & 0x8000) << 16) |
                    ((((uint32_t)(h >> 10) & 0x1F) + 112u) << 23) |
                    ((uint32_t)(h & 0x3FF) << 13);
    float f;
    __builtin_memcpy(&f, &bits, 4);
    return f;
}

/* ─── Integer Q8×Q8 dot product (avoids ALL int→float conversion) ─── */

/* Temporary quantized input block (float scale, not fp16, for speed) */
typedef struct { float d; int32_t isum; int8_t qs[32]; } q8_input_t;

/* Pre-quantized input buffer — max 512 blocks = 16384 elements (covers 9216-dim FFN) */
#define Q8_MAX_BLOCKS 512
static q8_input_t llm_xq_buf[Q8_MAX_BLOCKS];

/* AVX2 quantize 32 floats → Q8_0 block (8-wide SIMD abs-max + rounding) */
__attribute__((target("avx2,fma")))
static void q8_quantize_block(q8_input_t *out, const float *x)
{
    /* Find absmax with 8-wide AVX2 */
    v8f sign_mask = v8f_set1(-0.0f);
    v8f mx = v8f_setzero();
    for (int i = 0; i < 32; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v8f av;
        __asm__("vandnps %2, %1, %0" : "=x"(av) : "x"(sign_mask), "x"(v)); /* abs */
        mx = v8f_max(mx, av);
    }
    /* Horizontal max of 8 lanes */
    union { v8f vec; float f[8]; } mu = { .vec = mx };
    float amax = mu.f[0];
    for (int j = 1; j < 8; j++) if (mu.f[j] > amax) amax = mu.f[j];

    if (amax < 1e-30f) {
        out->d = 0.0f;
        out->isum = 0;
        __builtin_memset(out->qs, 0, 32);
        return;
    }
    out->d = amax / 127.0f;
    float id = 127.0f / amax;
    v8f vid = v8f_set1(id);
    /* Quantize 32 floats with rounding via vcvtps2dq (banker's rounding) */
    for (int i = 0; i < 32; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v8f scaled = v * vid;
        /* vcvtps2dq rounds to nearest-even (same as llama.cpp) */
        typedef int v8i __attribute__((vector_size(32)));
        v8i iv;
        __asm__("vcvtps2dq %1, %0" : "=x"(iv) : "x"(scaled));
        /* Pack int32→int16→int8: vpackssdw + vpacksswb */
        /* For simplicity, extract and clamp (compiler optimizes well) */
        union { v8i vec; int i[8]; } u = { .vec = iv };
        for (int j = 0; j < 8; j++) {
            int q = u.i[j];
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            out->qs[i + j] = (int8_t)q;
        }
    }
    /* Cache sum of quantized values for Q4×Q8 bias correction */
    {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) s += out->qs[j];
        out->isum = s;
    }
}

/* Quantize a full row of n floats to q8 blocks */
__attribute__((target("avx2,fma")))
static void q8_quantize_row(q8_input_t *out, const float *x, int n)
{
    int nb = n / 32;
    for (int b = 0; b < nb; b++)
        q8_quantize_block(&out[b], x + b * 32);
}

/* Integer dot: Q8_0 weight block x q8_input block
 * Uses v8f_load_i8 (vpmovsxbd+vcvtdq2ps) for BOTH weight and input
 * Avoids float input loads — input stays as int8, converted in-place */
__attribute__((target("avx2,fma")))
static float q8_0_dot_q8_avx2(const ggml_q8_0_t *w, const q8_input_t *xq)
{
    /* Load both sides from int8 using SIMD dequant (proven to work) */
    v8f w0 = v8f_load_i8(w->qs);      v8f x0 = v8f_load_i8(xq->qs);
    v8f w1 = v8f_load_i8(w->qs + 8);  v8f x1 = v8f_load_i8(xq->qs + 8);
    v8f w2 = v8f_load_i8(w->qs + 16); v8f x2 = v8f_load_i8(xq->qs + 16);
    v8f w3 = v8f_load_i8(w->qs + 24); v8f x3 = v8f_load_i8(xq->qs + 24);

    v8f acc = w0 * x0;
    acc += w1 * x1;
    acc += w2 * x2;
    acc += w3 * x3;

    return v8f_reduce(acc) * fp16_to_fp32(w->d) * xq->d;
}

/* Integer Q8 vec dot: weight row × pre-quantized input */
__attribute__((target("avx2,fma")))
static float llm_vec_dot_avx2_q8(const void *weight, const q8_input_t *xq, int n)
{
    int nb = n / 32;
    const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)weight;
    float sum = 0.0f;
    for (int b = 0; b < nb; b++)
        sum += q8_0_dot_q8_avx2(&blocks[b], &xq[b]);
    return sum;
}

/* AVX2 Q8_0 dot: process 32 elements per block using SIMD dequant */
__attribute__((target("avx2,fma")))
static float q8_0_dot32_avx2(const ggml_q8_0_t *block, const float *x)
{
    float d = fp16_to_fp32_fast(block->d);
    const int8_t *qs = block->qs;

    /* SIMD int8→float: 2 ops per 8 elements instead of 8 scalar casts */
    v8f q0 = v8f_load_i8(qs);
    v8f q1 = v8f_load_i8(qs + 8);
    v8f q2 = v8f_load_i8(qs + 16);
    v8f q3 = v8f_load_i8(qs + 24);

    v8f x0, x1, x2, x3;
    __builtin_memcpy(&x0, x, 32);
    __builtin_memcpy(&x1, x+8, 32);
    __builtin_memcpy(&x2, x+16, 32);
    __builtin_memcpy(&x3, x+24, 32);

    /* FMA-friendly accumulation */
    v8f acc = q0 * x0;
    acc += q1 * x1;
    acc += q2 * x2;
    acc += q3 * x3;

    return v8f_reduce(acc) * d;
}

/* AVX2 Q4_1 dot: process 32 elements per block using 8-wide
 * Q4_1: dequant = d * nibble + m (unsigned nibbles 0-15)
 * dot = sum(d * nibble_i * x_i + m * x_i) = d * sum(nibble_i * x_i) + m * sum(x_i) */
__attribute__((target("avx2,fma")))
static float q4_1_dot32_avx2(const ggml_q4_1_t *block, const float *x)
{
    float d = fp16_to_fp32_fast(block->d);
    float m = fp16_to_fp32_fast(block->m);
    const uint8_t *qs = block->qs;

    /* Standard GGML layout: lo nibbles → x[0..15], hi nibbles → x[16..31] */
    v8f qxacc_lo = v8f_setzero();
    v8f qxacc_hi = v8f_setzero();
    v8f xacc_lo = v8f_setzero();
    v8f xacc_hi = v8f_setzero();

    for (int j = 0; j < 16; j += 8) {
        uint8_t b0=qs[j], b1=qs[j+1], b2=qs[j+2], b3=qs[j+3];
        uint8_t b4=qs[j+4], b5=qs[j+5], b6=qs[j+6], b7=qs[j+7];
        v8f qlo = {(float)(b0&0xF), (float)(b1&0xF),
                   (float)(b2&0xF), (float)(b3&0xF),
                   (float)(b4&0xF), (float)(b5&0xF),
                   (float)(b6&0xF), (float)(b7&0xF)};
        v8f qhi = {(float)(b0>>4), (float)(b1>>4),
                   (float)(b2>>4), (float)(b3>>4),
                   (float)(b4>>4), (float)(b5>>4),
                   (float)(b6>>4), (float)(b7>>4)};
        v8f xlo; __builtin_memcpy(&xlo, x + j, 32);       /* x[j..j+7] */
        v8f xhi; __builtin_memcpy(&xhi, x + j + 16, 32);  /* x[j+16..j+23] */
        qxacc_lo += qlo * xlo;
        qxacc_hi += qhi * xhi;
        xacc_lo += xlo;
        xacc_hi += xhi;
    }

    float qxsum = v8f_reduce(qxacc_lo + qxacc_hi);
    float xsum  = v8f_reduce(xacc_lo + xacc_hi);
    return d * qxsum + m * xsum;
}

/* AVX2 Q4_0 dot: process 32 elements per block using 8-wide */
__attribute__((target("avx2,fma")))
static float q4_0_dot32_avx2(const ggml_q4_0_t *block, const float *x)
{
    float d = fp16_to_fp32_fast(block->d);
    const uint8_t *qs = block->qs;

    /* Standard GGML layout: lo nibbles → x[0..15], hi nibbles → x[16..31] */
    v8f acc_lo = v8f_setzero();
    v8f acc_hi = v8f_setzero();

    for (int j = 0; j < 16; j += 8) {
        uint8_t b0=qs[j], b1=qs[j+1], b2=qs[j+2], b3=qs[j+3];
        uint8_t b4=qs[j+4], b5=qs[j+5], b6=qs[j+6], b7=qs[j+7];
        v8f qlo = {(float)((int)(b0&0xF)-8), (float)((int)(b1&0xF)-8),
                   (float)((int)(b2&0xF)-8), (float)((int)(b3&0xF)-8),
                   (float)((int)(b4&0xF)-8), (float)((int)(b5&0xF)-8),
                   (float)((int)(b6&0xF)-8), (float)((int)(b7&0xF)-8)};
        v8f qhi = {(float)((int)(b0>>4)-8), (float)((int)(b1>>4)-8),
                   (float)((int)(b2>>4)-8), (float)((int)(b3>>4)-8),
                   (float)((int)(b4>>4)-8), (float)((int)(b5>>4)-8),
                   (float)((int)(b6>>4)-8), (float)((int)(b7>>4)-8)};
        v8f xlo; __builtin_memcpy(&xlo, x + j, 32);       /* x[j..j+7] */
        v8f xhi; __builtin_memcpy(&xhi, x + j + 16, 32);  /* x[j+16..j+23] */
        acc_lo += qlo * xlo;
        acc_hi += qhi * xhi;
    }

    return v8f_reduce(acc_lo + acc_hi) * d;
}

/* AVX2 generic vec_dot: dispatches to AVX2 block kernels */
__attribute__((target("avx2,fma")))
static float llm_vec_dot_avx2(const void *weight, const float *x, int n, ggml_type_t type)
{
    int nb = n / 32;
    switch (type) {
    case GGML_TYPE_Q8_0: {
        const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)weight;
        v8f acc = v8f_setzero();
        /* Unroll 2 blocks at a time for ILP */
        int b = 0;
        float sum = 0.0f;
        for (; b + 1 < nb; b += 2) {
            sum += q8_0_dot32_avx2(&blocks[b], x + b * 32);
            sum += q8_0_dot32_avx2(&blocks[b+1], x + (b+1) * 32);
        }
        for (; b < nb; b++)
            sum += q8_0_dot32_avx2(&blocks[b], x + b * 32);
        return sum;
    }
    case GGML_TYPE_Q4_0: {
        const ggml_q4_0_t *blocks = (const ggml_q4_0_t *)weight;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++)
            sum += q4_0_dot32_avx2(&blocks[b], x + b * 32);
        return sum;
    }
    case GGML_TYPE_Q4_1: {
        const ggml_q4_1_t *blocks = (const ggml_q4_1_t *)weight;
        float sum = 0.0f;
        for (int b = 0; b < nb; b++)
            sum += q4_1_dot32_avx2(&blocks[b], x + b * 32);
        return sum;
    }
    case GGML_TYPE_F32: {
        const float *f32 = (const float *)weight;
        v8f s0 = v8f_setzero(), s1 = v8f_setzero();
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            v8f a0, a1, b0, b1;
            __builtin_memcpy(&a0, f32+i, 32);
            __builtin_memcpy(&a1, f32+i+8, 32);
            __builtin_memcpy(&b0, x+i, 32);
            __builtin_memcpy(&b1, x+i+8, 32);
            s0 = s0 + a0 * b0;
            s1 = s1 + a1 * b1;
        }
        float sum = v8f_reduce(s0 + s1);
        for (; i < n; i++)
            sum += f32[i] * x[i];
        return sum;
    }
    case GGML_TYPE_Q6_K: {
        const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)weight;
        int nsb = n / 256;
        float sum = 0.0f;
        for (int b = 0; b < nsb; b++)
            sum += q6_k_dot256(&blocks[b], x + b * 256);
        return sum;
    }
    default:
        return llm_vec_dot(weight, x, n, type);
    }
}

/* ─── Fused Q8_0 GEMV: eliminates per-block horizontal reduction ─── *
 * Key optimizations vs the generic path:
 * 1. Single v8f accumulator per row — v8f_reduce called ONCE per row
 *    instead of once per block (saves ~15 ops × nb per row)
 * 2. Shared x loads across 4 rows (load x once, reuse for 4 weight rows)
 * 3. Branchless fp16_to_fp32_fast (~8 ops vs ~20)
 * 4. Scale broadcast + v8f multiply instead of scalar mul after reduce
 * 5. FMA-friendly instruction ordering
 */
__attribute__((target("avx2,fma")))
static void llm_gemv_q8_fused_avx2(float *out, const void *weight,
                                     const float *x, int out_dim, int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 34;
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    /* 4-row batched: shared x loads, fused accumulation */
    for (; i + 3 < out_dim; i += 4) {
        const ggml_q8_0_t *r0 = (const ggml_q8_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q8_0_t *r1 = (const ggml_q8_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q8_0_t *r2 = (const ggml_q8_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q8_0_t *r3 = (const ggml_q8_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            /* Load x ONCE — shared across all 4 rows */
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);

            /* Row 0 */
            {
                v8f d0 = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = v8f_load_i8(r0[b].qs) * x0;
                s += v8f_load_i8(r0[b].qs + 8)  * x1;
                s += v8f_load_i8(r0[b].qs + 16) * x2;
                s += v8f_load_i8(r0[b].qs + 24) * x3;
                a0 += s * d0;
            }
            /* Row 1 */
            {
                v8f d1 = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = v8f_load_i8(r1[b].qs) * x0;
                s += v8f_load_i8(r1[b].qs + 8)  * x1;
                s += v8f_load_i8(r1[b].qs + 16) * x2;
                s += v8f_load_i8(r1[b].qs + 24) * x3;
                a1 += s * d1;
            }
            /* Row 2 */
            {
                v8f d2 = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = v8f_load_i8(r2[b].qs) * x0;
                s += v8f_load_i8(r2[b].qs + 8)  * x1;
                s += v8f_load_i8(r2[b].qs + 16) * x2;
                s += v8f_load_i8(r2[b].qs + 24) * x3;
                a2 += s * d2;
            }
            /* Row 3 */
            {
                v8f d3 = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = v8f_load_i8(r3[b].qs) * x0;
                s += v8f_load_i8(r3[b].qs + 8)  * x1;
                s += v8f_load_i8(r3[b].qs + 16) * x2;
                s += v8f_load_i8(r3[b].qs + 24) * x3;
                a3 += s * d3;
            }
        }

        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    /* Tail rows */
    for (; i < out_dim; i++) {
        const ggml_q8_0_t *row = (const ggml_q8_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);
            v8f s = v8f_load_i8(row[b].qs) * x0;
            s += v8f_load_i8(row[b].qs + 8)  * x1;
            s += v8f_load_i8(row[b].qs + 16) * x2;
            s += v8f_load_i8(row[b].qs + 24) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* Fused Q8_0 GEMV for a range of rows (used by parallel workers) */
__attribute__((target("avx2,fma")))
static void llm_gemv_q8_fused_range_avx2(float *out, const void *weight,
                                           const float *x, int row_start,
                                           int row_end, int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 34;
    const uint8_t *base = (const uint8_t *)weight;
    int i = row_start;

    for (; i + 3 < row_end; i += 4) {
        const ggml_q8_0_t *r0 = (const ggml_q8_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q8_0_t *r1 = (const ggml_q8_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q8_0_t *r2 = (const ggml_q8_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q8_0_t *r3 = (const ggml_q8_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < row_end) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);

            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = v8f_load_i8(r0[b].qs) * x0;
                s += v8f_load_i8(r0[b].qs + 8)  * x1;
                s += v8f_load_i8(r0[b].qs + 16) * x2;
                s += v8f_load_i8(r0[b].qs + 24) * x3;
                a0 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = v8f_load_i8(r1[b].qs) * x0;
                s += v8f_load_i8(r1[b].qs + 8)  * x1;
                s += v8f_load_i8(r1[b].qs + 16) * x2;
                s += v8f_load_i8(r1[b].qs + 24) * x3;
                a1 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = v8f_load_i8(r2[b].qs) * x0;
                s += v8f_load_i8(r2[b].qs + 8)  * x1;
                s += v8f_load_i8(r2[b].qs + 16) * x2;
                s += v8f_load_i8(r2[b].qs + 24) * x3;
                a2 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = v8f_load_i8(r3[b].qs) * x0;
                s += v8f_load_i8(r3[b].qs + 8)  * x1;
                s += v8f_load_i8(r3[b].qs + 16) * x2;
                s += v8f_load_i8(r3[b].qs + 24) * x3;
                a3 += s * d;
            }
        }

        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    for (; i < row_end; i++) {
        const ggml_q8_0_t *row = (const ggml_q8_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);
            v8f s = v8f_load_i8(row[b].qs) * x0;
            s += v8f_load_i8(row[b].qs + 8)  * x1;
            s += v8f_load_i8(row[b].qs + 16) * x2;
            s += v8f_load_i8(row[b].qs + 24) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* ─── Q4_0 × Q8_0 integer dot product (llama.cpp approach) ─── */
/* Uses vpmaddubsw (unsigned×signed→int16) + vpmaddwd (int16→int32).
 * 32 MACs in ~4 SIMD instructions vs ~16 for the float path.
 * Q4 nibbles are treated as unsigned [0,15], bias of 8 is folded into sum. */
typedef int   v8i  __attribute__((vector_size(32)));
typedef short v16s __attribute__((vector_size(32)));
typedef char  v32b __attribute__((vector_size(32)));

__attribute__((target("avx2,fma")))
static inline float q4_0_q8_dot_avx2(const ggml_q4_0_t *w, const q8_input_t *xq)
{
    /* Scalar nibble unpack: 16 iterations, well-predicted by OOO hardware.
     * Main improvement over old path: avoids the 32-iteration scalar xsum loop
     * by using the precomputed xq->isum from q8_quantize_block. */
    uint8_t q4u[32];
    for (int j = 0; j < 16; j++) {
        q4u[j]      = w->qs[j] & 0x0F;
        q4u[j + 16] = w->qs[j] >> 4;
    }

    v32b q4_unsigned; __builtin_memcpy(&q4_unsigned, q4u, 32);
    v32b xq_bytes;    __builtin_memcpy(&xq_bytes,    xq->qs, 32);

    /* vpmaddubsw: unsigned Q4 [0,15] × signed Q8 → 16 × int16 */
    v16s prod16;
    __asm__("vpmaddubsw %2, %1, %0" : "=x"(prod16) : "x"(q4_unsigned), "x"(xq_bytes));

    /* vpmaddwd with ones → 8 × int32 */
    const v16s ones = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    v8i sum32;
    __asm__("vpmaddwd %2, %1, %0" : "=x"(sum32) : "x"(prod16), "x"(ones));

    /* Horizontal sum of 8 int32s */
    union { v8i vec; int i[8]; } su = { .vec = sum32 };
    int dot = (su.i[0]+su.i[1]+su.i[2]+su.i[3]) + (su.i[4]+su.i[5]+su.i[6]+su.i[7]);

    /* Bias correction using cached isum — avoids the old 32-scalar-op xsum loop */
    float wd = fp16_to_fp32_fast(w->d);
    return wd * xq->d * (float)(dot - 8 * xq->isum);
}

/* Fused Q4_0×Q8_0 integer GEMV: 4 rows at a time with shared Q8 input */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_q8_fused_avx2(float *out, const void *weight,
                                       const q8_input_t *xq, int out_dim,
                                       int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18; /* Q4_0 row bytes */
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    for (; i + 3 < out_dim; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int b = 0; b < nb; b++) {
            s0 += q4_0_q8_dot_avx2(&r0[b], &xq[b]);
            s1 += q4_0_q8_dot_avx2(&r1[b], &xq[b]);
            s2 += q4_0_q8_dot_avx2(&r2[b], &xq[b]);
            s3 += q4_0_q8_dot_avx2(&r3[b], &xq[b]);
        }
        out[i]   = s0;
        out[i+1] = s1;
        out[i+2] = s2;
        out[i+3] = s3;
    }
    for (; i < out_dim; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        float s = 0.0f;
        for (int b = 0; b < nb; b++)
            s += q4_0_q8_dot_avx2(&row[b], &xq[b]);
        out[i] = s;
    }
}

/* Fused Q4_0×Q8_0 integer GEMV for a range of rows (SMP workers) */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_q8_fused_range_avx2(float *out, const void *weight,
                                             const q8_input_t *xq,
                                             int row_start, int row_end,
                                             int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18;
    const uint8_t *base = (const uint8_t *)weight;
    int i = row_start;

    for (; i + 3 < row_end; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < row_end) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int b = 0; b < nb; b++) {
            s0 += q4_0_q8_dot_avx2(&r0[b], &xq[b]);
            s1 += q4_0_q8_dot_avx2(&r1[b], &xq[b]);
            s2 += q4_0_q8_dot_avx2(&r2[b], &xq[b]);
            s3 += q4_0_q8_dot_avx2(&r3[b], &xq[b]);
        }
        out[i]   = s0;
        out[i+1] = s1;
        out[i+2] = s2;
        out[i+3] = s3;
    }
    for (; i < row_end; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        float s = 0.0f;
        for (int b = 0; b < nb; b++)
            s += q4_0_q8_dot_avx2(&row[b], &xq[b]);
        out[i] = s;
    }
}

/* ─── Q4_0 nibble unpacking helpers for standard GGML layout ─── */
/* Extract 8 lo nibbles from 8 consecutive bytes → 8 floats.
 * Uses VPMOVZXBD (8-byte load → 8×int32) + AND + SUB + VCVTDQ2PS (1 cycle each). */
__attribute__((target("avx2,fma")))
static inline v8f q4_unpack_lo_v8f(const uint8_t *qs)
{
    typedef unsigned char v16ub __attribute__((vector_size(16)));
    typedef int v8i __attribute__((vector_size(32)));
    v16ub tmp = {0}; __builtin_memcpy(&tmp, qs, 8);
    v8i i32; __asm__("vpmovzxbd %1, %0" : "=x"(i32) : "x"(tmp));
    v8i mask = (v8i){0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF};
    v8i bias = (v8i){8,8,8,8,8,8,8,8};
    v8i signed_n = (i32 & mask) - bias;
    v8f result; __asm__("vcvtdq2ps %1, %0" : "=x"(result) : "x"(signed_n));
    return result;
}

/* Extract 8 hi nibbles from 8 consecutive bytes → 8 floats. */
__attribute__((target("avx2,fma")))
static inline v8f q4_unpack_hi_v8f(const uint8_t *qs)
{
    typedef unsigned char v16ub __attribute__((vector_size(16)));
    typedef int v8i __attribute__((vector_size(32)));
    v16ub tmp = {0}; __builtin_memcpy(&tmp, qs, 8);
    v8i i32; __asm__("vpmovzxbd %1, %0" : "=x"(i32) : "x"(tmp));
    v8i mask = (v8i){0xF,0xF,0xF,0xF,0xF,0xF,0xF,0xF};
    v8i bias = (v8i){8,8,8,8,8,8,8,8};
    v8i shifted;
    /* VPSRLD $4: logical right shift each int32 element by 4 bits */
    __asm__("vpsrld $4, %1, %0" : "=x"(shifted) : "x"(i32));
    v8i signed_n = (shifted & mask) - bias;
    v8f result; __asm__("vcvtdq2ps %1, %0" : "=x"(result) : "x"(signed_n));
    return result;
}

/* ─── Fused Q4_0 GEMV: 4-row batched with shared x loads ───
 * Standard GGML layout: lo nibbles → elements 0..15, hi → 16..31
 * For each block: qs[0..7] lo → x[0..7], qs[8..15] lo → x[8..15],
 *                 qs[0..7] hi → x[16..23], qs[8..15] hi → x[24..31]
 */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_fused_avx2(float *out, const void *weight,
                                     const float *x, int out_dim, int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18;
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    for (; i + 3 < out_dim; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);       /* x[0..7]   → lo qs[0..7] */
            __builtin_memcpy(&x1, xp + 8, 32);   /* x[8..15]  → lo qs[8..15] */
            __builtin_memcpy(&x2, xp + 16, 32);  /* x[16..23] → hi qs[0..7] */
            __builtin_memcpy(&x3, xp + 24, 32);  /* x[24..31] → hi qs[8..15] */

            /* Row 0 */
            {
                v8f d0 = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = q4_unpack_lo_v8f(r0[b].qs) * x0;
                s += q4_unpack_lo_v8f(r0[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r0[b].qs) * x2;
                s += q4_unpack_hi_v8f(r0[b].qs + 8) * x3;
                a0 += s * d0;
            }
            /* Row 1 */
            {
                v8f d1 = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = q4_unpack_lo_v8f(r1[b].qs) * x0;
                s += q4_unpack_lo_v8f(r1[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r1[b].qs) * x2;
                s += q4_unpack_hi_v8f(r1[b].qs + 8) * x3;
                a1 += s * d1;
            }
            /* Row 2 */
            {
                v8f d2 = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = q4_unpack_lo_v8f(r2[b].qs) * x0;
                s += q4_unpack_lo_v8f(r2[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r2[b].qs) * x2;
                s += q4_unpack_hi_v8f(r2[b].qs + 8) * x3;
                a2 += s * d2;
            }
            /* Row 3 */
            {
                v8f d3 = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = q4_unpack_lo_v8f(r3[b].qs) * x0;
                s += q4_unpack_lo_v8f(r3[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r3[b].qs) * x2;
                s += q4_unpack_hi_v8f(r3[b].qs + 8) * x3;
                a3 += s * d3;
            }
        }

        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    /* Tail rows */
    for (; i < out_dim; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);
            v8f s = q4_unpack_lo_v8f(row[b].qs) * x0;
            s += q4_unpack_lo_v8f(row[b].qs + 8) * x1;
            s += q4_unpack_hi_v8f(row[b].qs) * x2;
            s += q4_unpack_hi_v8f(row[b].qs + 8) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* Fused Q4_0 GEMV for a range of rows (used by parallel workers) */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_fused_range_avx2(float *out, const void *weight,
                                           const float *x, int row_start,
                                           int row_end, int in_dim)
{
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18;
    const uint8_t *base = (const uint8_t *)weight;
    int i = row_start;

    for (; i + 3 < row_end; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < row_end) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);

            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = q4_unpack_lo_v8f(r0[b].qs) * x0;
                s += q4_unpack_lo_v8f(r0[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r0[b].qs) * x2;
                s += q4_unpack_hi_v8f(r0[b].qs + 8) * x3;
                a0 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = q4_unpack_lo_v8f(r1[b].qs) * x0;
                s += q4_unpack_lo_v8f(r1[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r1[b].qs) * x2;
                s += q4_unpack_hi_v8f(r1[b].qs + 8) * x3;
                a1 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = q4_unpack_lo_v8f(r2[b].qs) * x0;
                s += q4_unpack_lo_v8f(r2[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r2[b].qs) * x2;
                s += q4_unpack_hi_v8f(r2[b].qs + 8) * x3;
                a2 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = q4_unpack_lo_v8f(r3[b].qs) * x0;
                s += q4_unpack_lo_v8f(r3[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r3[b].qs) * x2;
                s += q4_unpack_hi_v8f(r3[b].qs + 8) * x3;
                a3 += s * d;
            }
        }

        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    for (; i < row_end; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            const float *xp = x + b * 32;
            v8f x0, x1, x2, x3;
            __builtin_memcpy(&x0, xp, 32);
            __builtin_memcpy(&x1, xp + 8, 32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);
            v8f s = q4_unpack_lo_v8f(row[b].qs) * x0;
            s += q4_unpack_lo_v8f(row[b].qs + 8) * x1;
            s += q4_unpack_hi_v8f(row[b].qs) * x2;
            s += q4_unpack_hi_v8f(row[b].qs + 8) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* ─── Fused RMSNorm + Q4_0 GEMV (AVX2) ────────────────────────────────────
 * Eliminates the intermediate write of the normalized-x buffer (llm_xn).
 *
 * Traditional path: llm_rmsnorm writes llm_xn → N separate GEMVs each read
 * llm_xn.  For Q/K/V (3 projections) this costs: 1 write + 3 reads = 4 streams
 * of in_dim floats through the cache hierarchy.
 *
 * Fused path: inv_rms scalar computed once in pass 1, then applied inline
 * during each Q4_0 dot-product loop — norm weights are fused with x load.
 * Cost: 1 read of x for ss-pass + N × 1 read of x during GEMV = (N+1) reads,
 * zero writes of llm_xn.  For N=3 that saves 1 write + 0 redundant reads.
 *
 * void fused(out, weight, x, norm_w, out_dim, in_dim, eps)
 * ─────────────────────────────────────────────────────────────────────────── */
__attribute__((target("avx2,fma")))
static void llm_gemv_q4_fused_rmsnorm_avx2(
        float *out, const void *weight, const float *x, const float *norm_w,
        int out_dim, int in_dim, float eps)
{
    /* Pass 1: sum of squares using AVX2 eight-wide accumulation */
    v8f acc_sq = v8f_setzero();
    int vi;
    for (vi = 0; vi + 7 < in_dim; vi += 8) {
        v8f v;
        __builtin_memcpy(&v, x + vi, 32);
        acc_sq += v * v;
    }
    float ss = v8f_reduce(acc_sq);
    for (; vi < in_dim; vi++) ss += x[vi] * x[vi];
    const float inv_rms = 1.0f / llm_sqrtf(ss / (float)in_dim + eps);
    const v8f v_inv_rms = v8f_set1(inv_rms);

    /* Pass 2: Q4_0 GEMV with inline normalization: xn[i] = x[i]*inv_rms*norm_w[i]
     * computed per 32-element block, never written to memory. */
    const int nb = in_dim / 32;
    const uint64_t rb = (uint64_t)nb * 18;
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    for (; i + 3 < out_dim; i += 4) {
        const ggml_q4_0_t *r0 = (const ggml_q4_0_t *)(base + (uint64_t)(i)   * rb);
        const ggml_q4_0_t *r1 = (const ggml_q4_0_t *)(base + (uint64_t)(i+1) * rb);
        const ggml_q4_0_t *r2 = (const ggml_q4_0_t *)(base + (uint64_t)(i+2) * rb);
        const ggml_q4_0_t *r3 = (const ggml_q4_0_t *)(base + (uint64_t)(i+3) * rb);

        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 1);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 1);
        }

        v8f a0 = v8f_setzero(), a1 = v8f_setzero();
        v8f a2 = v8f_setzero(), a3 = v8f_setzero();

        for (int b = 0; b < nb; b++) {
            const float *xp = x      + b * 32;
            const float *wp = norm_w + b * 32;

            v8f w0, w1, w2, w3, x0, x1, x2, x3;
            __builtin_memcpy(&w0, wp,      32);
            __builtin_memcpy(&w1, wp + 8,  32);
            __builtin_memcpy(&w2, wp + 16, 32);
            __builtin_memcpy(&w3, wp + 24, 32);
            __builtin_memcpy(&x0, xp,      32);
            __builtin_memcpy(&x1, xp + 8,  32);
            __builtin_memcpy(&x2, xp + 16, 32);
            __builtin_memcpy(&x3, xp + 24, 32);

            /* Apply RMSNorm inline: xn = x * inv_rms * norm_w */
            x0 = x0 * v_inv_rms * w0;
            x1 = x1 * v_inv_rms * w1;
            x2 = x2 * v_inv_rms * w2;
            x3 = x3 * v_inv_rms * w3;

            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r0[b].d));
                v8f s = q4_unpack_lo_v8f(r0[b].qs) * x0;
                s += q4_unpack_lo_v8f(r0[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r0[b].qs) * x2;
                s += q4_unpack_hi_v8f(r0[b].qs + 8) * x3;
                a0 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r1[b].d));
                v8f s = q4_unpack_lo_v8f(r1[b].qs) * x0;
                s += q4_unpack_lo_v8f(r1[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r1[b].qs) * x2;
                s += q4_unpack_hi_v8f(r1[b].qs + 8) * x3;
                a1 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r2[b].d));
                v8f s = q4_unpack_lo_v8f(r2[b].qs) * x0;
                s += q4_unpack_lo_v8f(r2[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r2[b].qs) * x2;
                s += q4_unpack_hi_v8f(r2[b].qs + 8) * x3;
                a2 += s * d;
            }
            {
                v8f d = v8f_set1(fp16_to_fp32_fast(r3[b].d));
                v8f s = q4_unpack_lo_v8f(r3[b].qs) * x0;
                s += q4_unpack_lo_v8f(r3[b].qs + 8) * x1;
                s += q4_unpack_hi_v8f(r3[b].qs) * x2;
                s += q4_unpack_hi_v8f(r3[b].qs + 8) * x3;
                a3 += s * d;
            }
        }
        out[i]   = v8f_reduce(a0);
        out[i+1] = v8f_reduce(a1);
        out[i+2] = v8f_reduce(a2);
        out[i+3] = v8f_reduce(a3);
    }

    /* Tail rows */
    for (; i < out_dim; i++) {
        const ggml_q4_0_t *row = (const ggml_q4_0_t *)(base + (uint64_t)i * rb);
        v8f acc = v8f_setzero();
        for (int b = 0; b < nb; b++) {
            const float *xp = x      + b * 32;
            const float *wp = norm_w + b * 32;
            v8f w0, w1, w2, w3, x0, x1, x2, x3;
            __builtin_memcpy(&w0, wp,      32);  __builtin_memcpy(&w1, wp + 8,  32);
            __builtin_memcpy(&w2, wp + 16, 32);  __builtin_memcpy(&w3, wp + 24, 32);
            __builtin_memcpy(&x0, xp,      32);  __builtin_memcpy(&x1, xp + 8,  32);
            __builtin_memcpy(&x2, xp + 16, 32);  __builtin_memcpy(&x3, xp + 24, 32);
            x0 = x0 * v_inv_rms * w0;
            x1 = x1 * v_inv_rms * w1;
            x2 = x2 * v_inv_rms * w2;
            x3 = x3 * v_inv_rms * w3;
            v8f d = v8f_set1(fp16_to_fp32_fast(row[b].d));
            v8f s = q4_unpack_lo_v8f(row[b].qs) * x0;
            s += q4_unpack_lo_v8f(row[b].qs + 8) * x1;
            s += q4_unpack_hi_v8f(row[b].qs) * x2;
            s += q4_unpack_hi_v8f(row[b].qs + 8) * x3;
            acc += s * d;
        }
        out[i] = v8f_reduce(acc);
    }
}

/* AVX2 GEMV: 4-row batched for better cache utilization */
__attribute__((target("avx2,fma")))
static void llm_gemv_avx2(float *out, const void *weight, const float *x,
                           int out_dim, int in_dim, ggml_type_t type)
{
    /* Q8_0 fast path: fully fused GEMV (no per-block reduce, shared x loads) */
    if (type == GGML_TYPE_Q8_0) {
        llm_gemv_q8_fused_avx2(out, weight, x, out_dim, in_dim);
        return;
    }
    /* Q4_0 fast path: float fused GEMV */
    if (type == GGML_TYPE_Q4_0) {
        llm_gemv_q4_fused_avx2(out, weight, x, out_dim, in_dim);
        return;
    }
    uint64_t rb = llm_row_bytes(in_dim, type);
    const uint8_t *base = (const uint8_t *)weight;
    int i = 0;

    /* Q8_0 pre-quantized path disabled (slower under QEMU TCG) */
    if (0) {
        int nb = in_dim / 32;
        q8_input_t xq[Q8_MAX_BLOCKS];
        q8_quantize_row(xq, x, in_dim);

        for (; i + 3 < out_dim; i += 4) {
            if (i + 7 < out_dim) {
                __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 3);
                __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 3);
            }
            out[i]   = llm_vec_dot_avx2_q8(base + (uint64_t)(i)   * rb, xq, in_dim);
            out[i+1] = llm_vec_dot_avx2_q8(base + (uint64_t)(i+1) * rb, xq, in_dim);
            out[i+2] = llm_vec_dot_avx2_q8(base + (uint64_t)(i+2) * rb, xq, in_dim);
            out[i+3] = llm_vec_dot_avx2_q8(base + (uint64_t)(i+3) * rb, xq, in_dim);
        }
        for (; i < out_dim; i++)
            out[i] = llm_vec_dot_avx2_q8(base + (uint64_t)i * rb, xq, in_dim);
        return;
    }

    /* Generic float path for other types */
    for (; i + 3 < out_dim; i += 4) {
        if (i + 7 < out_dim) {
            __builtin_prefetch(base + (uint64_t)(i+4) * rb, 0, 3);
            __builtin_prefetch(base + (uint64_t)(i+5) * rb, 0, 3);
        }
        const void *r0 = base + (uint64_t)(i)   * rb;
        const void *r1 = base + (uint64_t)(i+1) * rb;
        const void *r2 = base + (uint64_t)(i+2) * rb;
        const void *r3 = base + (uint64_t)(i+3) * rb;
        out[i]   = llm_vec_dot_avx2(r0, x, in_dim, type);
        out[i+1] = llm_vec_dot_avx2(r1, x, in_dim, type);
        out[i+2] = llm_vec_dot_avx2(r2, x, in_dim, type);
        out[i+3] = llm_vec_dot_avx2(r3, x, in_dim, type);
    }
    for (; i < out_dim; i++) {
        const void *row = base + (uint64_t)i * rb;
        out[i] = llm_vec_dot_avx2(row, x, in_dim, type);
    }
}

#endif /* !__aarch64__ */

/* Bytes per row for a quantized matrix [out_dim × in_dim] */
static uint64_t llm_row_bytes(int in_dim, ggml_type_t type)
{
    switch (type) {
    case GGML_TYPE_Q4_0: return (uint64_t)(in_dim / 32) * 18;
    case GGML_TYPE_Q4_1: return (uint64_t)(in_dim / 32) * 20;
    case GGML_TYPE_Q8_0: return (uint64_t)(in_dim / 32) * 34;
    /* K-quant super-block types (256 elements each) */
    case GGML_TYPE_Q2_K: return (uint64_t)(in_dim / 256) * 84;   /* 2+2+16+64 bytes */
    case GGML_TYPE_Q3_K: return (uint64_t)(in_dim / 256) * 110;  /* 2+12+64+32 bytes */
    case GGML_TYPE_Q4_K: return (uint64_t)(in_dim / 256) * 144;  /* 2+2+12+128 bytes */
    case GGML_TYPE_Q5_K: return (uint64_t)(in_dim / 256) * 176;  /* 2+2+12+128+64 bytes */
    case GGML_TYPE_Q6_K: return (uint64_t)(in_dim / 256) * 210;
    /* IQ (importance quant) types */
    case GGML_TYPE_IQ4_NL: return (uint64_t)(in_dim / 32) * 18;   /* fp16+qs[16], 32-elem blocks */
    case GGML_TYPE_IQ4_XS: return (uint64_t)(in_dim / 256) * 136; /* fp16+sc_h+sc_l[4]+qs[128] */
    /* IQ2 / IQ3 super-block types */
    case GGML_TYPE_IQ2_XXS: return (uint64_t)(in_dim / 256) * 66;  /* fp16+qs[32] (uint16_t) */
    case GGML_TYPE_IQ2_XS:  return (uint64_t)(in_dim / 256) * 74;  /* fp16+qs[32]+scales[8] */
    case GGML_TYPE_IQ3_XXS: return (uint64_t)(in_dim / 256) * 98;  /* fp16+qs[32]+scales[16] */
    case GGML_TYPE_IQ3_S:   return (uint64_t)(in_dim / 256) * 110; /* same byte count as Q3_K */
    case GGML_TYPE_Q5_0:  return (uint64_t)(in_dim / 32) * 22;   /* fp16 d + qh[4] + qs[16] */
    case GGML_TYPE_Q5_1:  return (uint64_t)(in_dim / 32) * 24;   /* fp16 d + fp16 m + qh[4] + qs[16] */
    case GGML_TYPE_Q8_1:  return (uint64_t)(in_dim / 32) * 36;   /* fp16 d + fp16 s + qs[32] */
    case GGML_TYPE_IQ1_S: return (uint64_t)(in_dim / 256) * 50;  /* fp16 d + qs[32] + qh[16] */
    case GGML_TYPE_IQ2_S: return (uint64_t)(in_dim / 256) * 82;  /* fp16 d + qs[64] + qh[8] + scales[8] */
    case GGML_TYPE_IQ1_M: return (uint64_t)(in_dim / 256) * 56;  /* qs[32] + qh[16] + scales[8], no d field */
    case GGML_TYPE_F16:  return (uint64_t)in_dim * 2;
    case GGML_TYPE_BF16: return (uint64_t)in_dim * 2;
    case GGML_TYPE_F32:  return (uint64_t)in_dim * 4;
    default:             return (uint64_t)in_dim * 4;
    }
}

/* GEMV: out[out_dim] = weight[out_dim x in_dim] . x[in_dim]
 * weight is in quantized GGML format (row-major)
 * Tries JIT-compiled Q8_0 kernel first, falls back to vec_dot loop. */

/* =============================================================================
 * LRU GEMV Kernel Cache
 *
 * Unified cache for all GEMV kernel types with LRU eviction.
 * Replaces the previous fixed 8-slot caches with a larger 64-slot adaptive cache.
 * =============================================================================*/
#define LRU_GEMV_CACHE_SIZE 64

typedef enum {
    LRU_GEMV_SSE2_Q8 = 0,
    LRU_GEMV_AVX2_Q8,
    LRU_GEMV_AVX2_Q4Q8,
    LRU_GEMV_AVX512_Q8,
    LRU_GEMV_TYPE_COUNT
} lru_gemv_type_t;

typedef struct {
    int rows;
    int cols;
    lru_gemv_type_t type;
    jit_gemv_q8_fn fn;
    uint32_t last_used;  /* Monotonic counter for LRU eviction */
} lru_gemv_entry_t;

static lru_gemv_entry_t lru_gemv_cache[LRU_GEMV_CACHE_SIZE];
static int lru_gemv_cache_n = 0;
static uint32_t lru_gemv_clock = 0;

/* Find an entry in the LRU cache, update last_used if found */
static jit_gemv_q8_fn lru_gemv_lookup(int rows, int cols, lru_gemv_type_t type)
{
    for (int i = 0; i < lru_gemv_cache_n; i++) {
        if (lru_gemv_cache[i].rows == rows &&
            lru_gemv_cache[i].cols == cols &&
            lru_gemv_cache[i].type == type) {
            lru_gemv_cache[i].last_used = ++lru_gemv_clock;
            return lru_gemv_cache[i].fn;
        }
    }
    return NULL;
}

/* Store an entry in the LRU cache, evicting oldest if full */
static void lru_gemv_store(int rows, int cols, lru_gemv_type_t type, jit_gemv_q8_fn fn)
{
    if (!fn) return;

    /* Find slot: empty or evict LRU */
    int slot;
    if (lru_gemv_cache_n < LRU_GEMV_CACHE_SIZE) {
        slot = lru_gemv_cache_n++;
    } else {
        /* Find least recently used entry */
        slot = 0;
        uint32_t oldest = lru_gemv_cache[0].last_used;
        for (int i = 1; i < LRU_GEMV_CACHE_SIZE; i++) {
            if (lru_gemv_cache[i].last_used < oldest) {
                oldest = lru_gemv_cache[i].last_used;
                slot = i;
            }
        }
        /* Note: we don't free the old JIT buffer since it's in the JIT pool */
    }

    lru_gemv_cache[slot].rows = rows;
    lru_gemv_cache[slot].cols = cols;
    lru_gemv_cache[slot].type = type;
    lru_gemv_cache[slot].fn = fn;
    lru_gemv_cache[slot].last_used = ++lru_gemv_clock;
}

/* Legacy API wrappers using LRU cache */
static jit_gemv_q8_fn llm_get_jit_gemv_avx2(int rows, int cols)
{
    jit_gemv_q8_fn fn = lru_gemv_lookup(rows, cols, LRU_GEMV_AVX2_Q8);
    if (fn) return fn;

    fn = jit_compile_q8_gemv_avx2(rows, cols);
    lru_gemv_store(rows, cols, LRU_GEMV_AVX2_Q8, fn);
    return fn;
}

static jit_gemv_q8_fn llm_get_jit_gemv(int rows, int cols)
{
    jit_gemv_q8_fn fn = lru_gemv_lookup(rows, cols, LRU_GEMV_SSE2_Q8);
    if (fn) return fn;

    fn = jit_compile_q8_gemv(rows, cols);
    lru_gemv_store(rows, cols, LRU_GEMV_SSE2_Q8, fn);
    return fn;
}


static jit_gemv_q8_fn llm_get_jit_gemv_q4q8(int rows, int cols)
{
    jit_gemv_q8_fn fn = lru_gemv_lookup(rows, cols, LRU_GEMV_AVX2_Q4Q8);
    if (fn) return fn;

    fn = jit_compile_q4_q8_gemv_avx2(rows, cols);
    lru_gemv_store(rows, cols, LRU_GEMV_AVX2_Q4Q8, fn);
    return fn;
}

/* AVX-512 Q8_0 GEMV via LRU JIT cache — EVEX P0 byte (0x62 opcode prefix) verified. */
static jit_gemv_q8_fn llm_get_jit_gemv_avx512(int rows, int cols)
{
    jit_gemv_q8_fn fn = lru_gemv_lookup(rows, cols, LRU_GEMV_AVX512_Q8);
    if (fn) return fn;

    fn = jit_compile_q8_gemv_avx512(rows, cols);
    lru_gemv_store(rows, cols, LRU_GEMV_AVX512_Q8, fn);
    return fn;
}

/* ─── Parallel GEMV work item for SMP dispatch ─── */
#ifndef __aarch64__
typedef struct {
    float       *out;
    const void  *weight;
    const float *x;
    const q8_input_t *xq; /* Pre-quantized input for Q8_0 integer path */
    int          out_dim;
    int          in_dim;
    ggml_type_t  type;
    int          row_start;
    int          row_end;
} gemv_work_t;

static gemv_work_t gemv_work_items[MAX_CPUS];

__attribute__((target("avx2,fma")))
static void gemv_worker_avx2(void *arg)
{
    gemv_work_t *w = (gemv_work_t *)arg;

    /* Q8_0 fast path: fully fused (no per-block reduce, shared x loads) */
    if (w->type == GGML_TYPE_Q8_0) {
        llm_gemv_q8_fused_range_avx2(w->out, w->weight, w->x,
                                      w->row_start, w->row_end, w->in_dim);
        return;
    }

    /* Q4_0 integer path: use pre-quantized Q8 input if available */
    if (w->type == GGML_TYPE_Q4_0 && w->xq) {
        llm_gemv_q4_q8_fused_range_avx2(w->out, w->weight, w->xq,
                                         w->row_start, w->row_end, w->in_dim);
        return;
    }

    /* Q4_0 float fallback */
    if (w->type == GGML_TYPE_Q4_0) {
        llm_gemv_q4_fused_range_avx2(w->out, w->weight, w->x,
                                      w->row_start, w->row_end, w->in_dim);
        return;
    }

    uint64_t rb = llm_row_bytes(w->in_dim, w->type);
    const uint8_t *base = (const uint8_t *)w->weight;

    for (int i = w->row_start; i < w->row_end; i++) {
        if (i + 1 < w->row_end)
            __builtin_prefetch(base + (uint64_t)(i + 1) * rb, 0, 3);
        const void *row = base + (uint64_t)i * rb;
        w->out[i] = llm_vec_dot_avx2(row, w->x, w->in_dim, w->type);
    }
}
#endif

static void llm_gemv(float *out, const void *weight, const float *x,
                     int out_dim, int in_dim, ggml_type_t type)
{
#ifdef ENABLE_CUDA
    /* GPU-accelerated GEMV: look up pre-uploaded device weight.
     * Use the GPU-resident type (may differ from host type, e.g. IQ2_XS uploaded as F32). */
    if (gpu_ctx.active) {
        ggml_type_t gpu_type = type;
        void *d_weight = llm_gpu_lookup_with_type(weight, &gpu_type);
        if (d_weight &&
            (gpu_type == GGML_TYPE_Q4_0 || gpu_type == GGML_TYPE_Q4_1 ||
             gpu_type == GGML_TYPE_Q8_0 ||
             gpu_type == GGML_TYPE_F32  || gpu_type == GGML_TYPE_F16  ||
             gpu_type == GGML_TYPE_BF16 ||
             gpu_type == GGML_TYPE_Q6_K || gpu_type == GGML_TYPE_Q4_K ||
             gpu_type == GGML_TYPE_IQ2_XS)) {
            const backend_t *be = backend_get_by_id(BACKEND_CUDA);
            /* Upload input vector to GPU scratch */
            be->mem.upload(gpu_ctx.d_x, x, (uint64_t)in_dim * sizeof(float));
            /* Launch CUDA GEMV kernel with GPU-resident type */
            be->compute.gemv(gpu_ctx.d_out, d_weight, gpu_ctx.d_x,
                             out_dim, in_dim, gpu_type);
            /* Download result (cudaMemcpy D2H is synchronous — waits for kernel) */
            be->mem.download(out, gpu_ctx.d_out, (uint64_t)out_dim * sizeof(float));
            return;
        }
    }
#endif

#ifndef __aarch64__
    /* JIT AVX-512 GEMV: 16-wide ZMM GEMV; EVEX encoding verified (P0 fix applied).
     * Processes 64 weights/block vs 32 for AVX2, 4-row batched — ~1.4× vs AVX2 on Zen4. */
    if (cpu_features.has_avx512f && type == GGML_TYPE_Q8_0 && in_dim >= 64) {
        jit_gemv_q8_fn jfn = llm_get_jit_gemv_avx512(out_dim, in_dim);
        if (jfn) {
            jfn(out, weight, x, out_dim, in_dim);
            return;
        }
    }

    /* JIT AVX2+FMA GEMV: best single-core Q8_0 performance */
    if (cpu_features.avx2_usable && type == GGML_TYPE_Q8_0) {
        jit_gemv_q8_fn jfn = llm_get_jit_gemv_avx2(out_dim, in_dim);
        if (jfn) {
            jfn(out, weight, x, out_dim, in_dim);
            return;
        }
    }

    /* Q4×Q8 integer GEMV: pre-quantize input to Q8, then use integer dot */
    if (cpu_features.avx2_usable && type == GGML_TYPE_Q4_0) {
        int nb = in_dim / 32;
        if (nb <= Q8_MAX_BLOCKS) {
            /* Pre-quantize input vector to Q8 for integer dot path */
            q8_quantize_row(llm_xq_buf, x, in_dim);

            /* Parallel Q4xQ8 integer GEMV across all CPUs */
            uint32_t ncpu = smp.ap_started + 1;
            if (ncpu > 1 && out_dim >= 64) {
                int rows_per_cpu = out_dim / ncpu;
                int remainder    = out_dim % ncpu;
                int row = 0;
                int bsp_chunk = rows_per_cpu + (remainder > 0 ? 1 : 0);
                int bsp_start = 0;
                int bsp_end   = bsp_chunk;
                row = bsp_end;

                for (uint32_t c = 1; c < ncpu; c++) {
                    int chunk = rows_per_cpu + ((int)c < remainder ? 1 : 0);
                    gemv_work_items[c].out      = out;
                    gemv_work_items[c].weight   = weight;
                    gemv_work_items[c].x        = x;
                    gemv_work_items[c].xq       = llm_xq_buf;
                    gemv_work_items[c].out_dim  = out_dim;
                    gemv_work_items[c].in_dim   = in_dim;
                    gemv_work_items[c].type     = type;
                    gemv_work_items[c].row_start = row;
                    gemv_work_items[c].row_end   = row + chunk;
                    smp_dispatch(c, gemv_worker_avx2, &gemv_work_items[c]);
                    row += chunk;
                }
                /* BSP does its share using integer path */
                llm_gemv_q4_q8_fused_range_avx2(out, weight, llm_xq_buf,
                                                 bsp_start, bsp_end, in_dim);
                smp_wait_all();
                return;
            }

            /* Single-core fallback: fused integer GEMV */
            llm_gemv_q4_q8_fused_avx2(out, weight, llm_xq_buf, out_dim, in_dim);
            return;
        }
    }

    /* Use AVX2+FMA GEMV when available */
    if (cpu_features.avx2_usable) {
        uint32_t ncpu = smp.ap_started + 1; /* Only use actually-running CPUs */
        /* Parallel GEMV: split rows across all CPUs when worth it.
         * Q8_0 and Q4_0 use optimised fused paths; IQ2_XS uses the generic
         * llm_vec_dot_avx2 fallback in gemv_worker_avx2, which is thread-safe
         * (no mutable shared state — only const-static kgrid and local vars).
         * All other types fall through to single-threaded llm_gemv_avx2. */
        if (ncpu > 1 && out_dim >= 64 &&
            (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 ||
             type == GGML_TYPE_IQ2_XS)) {
            int rows_per_cpu = out_dim / ncpu;
            int remainder    = out_dim % ncpu;
            int row = 0;
            /* BSP gets chunk 0, APs get chunks 1..ncpu-1 */
            int bsp_chunk = rows_per_cpu + (remainder > 0 ? 1 : 0);
            int bsp_start = 0;
            int bsp_end   = bsp_chunk;
            row = bsp_end;

            /* Q4×Q8 integer path pending correctness fix — use float path */
            const q8_input_t *xq_ptr = (void *)0;

            /* Dispatch to APs (cpu 1..ncpu-1) */
            for (uint32_t c = 1; c < ncpu; c++) {
                int chunk = rows_per_cpu + ((int)c < remainder ? 1 : 0);
                gemv_work_items[c].out      = out;
                gemv_work_items[c].weight   = weight;
                gemv_work_items[c].x        = x;
                gemv_work_items[c].xq       = xq_ptr;
                gemv_work_items[c].out_dim  = out_dim;
                gemv_work_items[c].in_dim   = in_dim;
                gemv_work_items[c].type     = type;
                gemv_work_items[c].row_start = row;
                gemv_work_items[c].row_end   = row + chunk;
                smp_dispatch(c, gemv_worker_avx2, &gemv_work_items[c]);
                row += chunk;
            }
            /* BSP does its share — use fused path for Q8_0/Q4_0 */
            if (type == GGML_TYPE_Q8_0) {
                llm_gemv_q8_fused_range_avx2(out, weight, x,
                                              bsp_start, bsp_end, in_dim);
            } else if (type == GGML_TYPE_Q4_0) {
                llm_gemv_q4_fused_range_avx2(out, weight, x,
                                              bsp_start, bsp_end, in_dim);
            } else {
                uint64_t rb = llm_row_bytes(in_dim, type);
                const uint8_t *base = (const uint8_t *)weight;
                for (int i = bsp_start; i < bsp_end; i++) {
                    if (i + 1 < bsp_end)
                        __builtin_prefetch(base + (uint64_t)(i + 1) * rb, 0, 3);
                    const void *r = base + (uint64_t)i * rb;
                    out[i] = llm_vec_dot_avx2(r, x, in_dim, type);
                }
            }
            /* Wait for all APs */
            smp_wait_all();
            return;
        }
        llm_gemv_avx2(out, weight, x, out_dim, in_dim, type);
        return;
    }

    /* Try JIT-compiled Q8_0 GEMV kernel */
    if (type == GGML_TYPE_Q8_0) {
        jit_gemv_q8_fn jfn = llm_get_jit_gemv(out_dim, in_dim);
        if (jfn) {
            jfn(out, weight, x, out_dim, in_dim);
            return;
        }
    }
#endif

    uint64_t rb = llm_row_bytes(in_dim, type);

    for (int i = 0; i < out_dim; i++) {
        const void *row = (const uint8_t *)weight + (uint64_t)i * rb;
        out[i] = llm_vec_dot(row, x, in_dim, type);
    }
}

/* Batched GEMV — processes batch_size input vectors against the
 * same weight matrix in a single kernel invocation.  Used by speculative
 * decode verification when multiple draft tokens share the same weight layers.
 *
 * out  [batch_size][out_dim]   x [batch_size][in_dim]   weight [out_dim][in_dim/Q8]
 *
 * Falls back to repeated llm_gemv calls when the JIT kernel is unavailable. */
static void llm_gemv_batch(float *out, const void *weight, const float *x,
                           int out_dim, int in_dim, ggml_type_t type, int batch_size)
{
    if (batch_size <= 1) {
        llm_gemv(out, weight, x, out_dim, in_dim, type);
        return;
    }

#ifndef __aarch64__
    if (cpu_features.avx2_usable && type == GGML_TYPE_Q8_0 && in_dim % 32 == 0) {
        jit_batched_gemv_fn jfn = jit_compile_batched_q8_gemv(out_dim, in_dim, batch_size);
        if (jfn) {
            jfn(out, weight, x, out_dim, in_dim, batch_size);
            return;
        }
    }
#endif

    /* Scalar fallback: loop over batch */
    for (int b = 0; b < batch_size; b++) {
        llm_gemv(out + (uint64_t)b * out_dim, weight,
                 x   + (uint64_t)b * in_dim,
                 out_dim, in_dim, type);
    }
}

/* Get a single float from a (possibly quantized) 1D vector at index idx */
static float llm_get_f(const void *data, int idx, ggml_type_t type)
{
    switch (type) {
    case GGML_TYPE_F32:
        return ((const float *)data)[idx];
    case GGML_TYPE_F16:
        return fp16_to_fp32(((const uint16_t *)data)[idx]);
    case GGML_TYPE_BF16:
        return bf16_to_fp32(((const uint16_t *)data)[idx]);
    default:
        return 0.0f;
    }
}

/* Embedding lookup: copy the embedding vector for token_id into out[dim] */
static void llm_embed(float *out, const llm_model_t *m, int token_id)
{
    int dim = m->dim;
    uint64_t rb = llm_row_bytes(dim, m->token_embd_type);
    const uint8_t *row = (const uint8_t *)m->token_embd + (uint64_t)token_id * rb;

    switch (m->token_embd_type) {
    case GGML_TYPE_F32: {
        const float *f = (const float *)row;
        for (int i = 0; i < dim; i++) out[i] = f[i];
        break;
    }
    case GGML_TYPE_F16: {
        const uint16_t *h = (const uint16_t *)row;
        for (int i = 0; i < dim; i++) out[i] = fp16_to_fp32(h[i]);
        break;
    }
    case GGML_TYPE_Q4_0: {
        /* Dequantize Q4_0 blocks (standard GGML layout: lo→[0..15], hi→[16..31]) */
        const ggml_q4_0_t *blocks = (const ggml_q4_0_t *)row;
        int nb = dim / 32;
        for (int b = 0; b < nb; b++) {
            float d = fp16_to_fp32(blocks[b].d);
            for (int j = 0; j < 16; j++) {
                uint8_t packed = blocks[b].qs[j];
                out[b * 32 + j]      = (float)((int)(packed & 0x0F) - 8) * d;
                out[b * 32 + j + 16] = (float)((int)(packed >> 4)   - 8) * d;
            }
        }
        break;
    }
    case GGML_TYPE_Q8_0: {
        const ggml_q8_0_t *blocks = (const ggml_q8_0_t *)row;
        int nb = dim / 32;
        for (int b = 0; b < nb; b++) {
            float d = fp16_to_fp32(blocks[b].d);
            for (int j = 0; j < 32; j++)
                out[b * 32 + j] = (float)blocks[b].qs[j] * d;
        }
        break;
    }
    case GGML_TYPE_Q6_K: {
        /* Q6_K: 256 elements per super-block, 210 bytes each */
        const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)row;
        int nsb = dim / 256;
        for (int sb = 0; sb < nsb; sb++) {
            const ggml_q6_k_t *b = &blocks[sb];
            float d = fp16_to_fp32(b->d);
            const uint8_t *ql = b->ql;
            const uint8_t *qh = b->qh;
            const int8_t *sc = b->scales;
            float *o = out + sb * 256;
            for (int half = 0; half < 2; half++) {
                const uint8_t *ql_h = ql + half * 64;
                const uint8_t *qh_h = qh + half * 32;
                const int8_t *sc_h = sc + half * 8;
                for (int l = 0; l < 32; l++) {
                    int q0 = (int)(ql_h[l] & 0xF)      | (int)(((qh_h[l] >> 0) & 3) << 4);
                    int q1 = (int)(ql_h[l + 32] & 0xF)  | (int)(((qh_h[l] >> 2) & 3) << 4);
                    int q2 = (int)(ql_h[l] >> 4)         | (int)(((qh_h[l] >> 4) & 3) << 4);
                    int q3 = (int)(ql_h[l + 32] >> 4)    | (int)(((qh_h[l] >> 6) & 3) << 4);
                    int si = l / 16;
                    o[half*128 + l]      = d * (float)sc_h[0 + si] * (float)(q0 - 32);
                    o[half*128 + l + 32] = d * (float)sc_h[2 + si] * (float)(q1 - 32);
                    o[half*128 + l + 64] = d * (float)sc_h[4 + si] * (float)(q2 - 32);
                    o[half*128 + l + 96] = d * (float)sc_h[6 + si] * (float)(q3 - 32);
                }
            }
        }
        break;
    }
    case GGML_TYPE_Q4_K: {
        /* Q4_K: 256 elements per super-block, 144 bytes each.
         * Layout: d(fp16) + dmin(fp16) + scales[12] + qs[128]. */
        const uint8_t *p = row;
        int nsb = dim / 256;
        for (int b = 0; b < nsb; b++) {
            uint16_t dh, dmh;
            kmemcpy(&dh,  p,     2);
            kmemcpy(&dmh, p + 2, 2);
            float d    = fp16_to_fp32(dh);
            float dmin = fp16_to_fp32(dmh);
            const uint8_t *scales = p + 4;
            const uint8_t *qs     = p + 16;
            p += 144;

            float *o = out + b * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0, mn0, sc1, mn1;
                if (is < 4) {
                    sc0 = scales[is] & 63;
                    mn0 = scales[is + 4] & 63;
                } else {
                    sc0 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                    mn0 = (scales[is + 4] >> 4)  | ((scales[is + 0] >> 6) << 4);
                }
                int is1 = is + 1;
                if (is1 < 4) {
                    sc1 = scales[is1] & 63;
                    mn1 = scales[is1 + 4] & 63;
                } else {
                    sc1 = (scales[is1 + 4] & 0xF) | ((scales[is1 - 4] >> 6) << 4);
                    mn1 = (scales[is1 + 4] >> 4)  | ((scales[is1 + 0] >> 6) << 4);
                }
                float d1 = d * (float)sc0;
                float m1f = dmin * (float)mn0;
                float d2 = d * (float)sc1;
                float m2f = dmin * (float)mn1;

                const uint8_t *qj = qs + (j / 2);
                for (int l = 0; l < 32; l++)
                    o[j + l] = d1 * (float)(qj[l] & 0xF) - m1f;
                for (int l = 0; l < 32; l++)
                    o[j + 32 + l] = d2 * (float)(qj[l] >> 4) - m2f;
                is += 2;
            }
        }
        break;
    }
    case GGML_TYPE_IQ2_XS: {
        static const uint8_t  iq2xs_vals_e[4] = {1, 3, 5, 7};
        static const uint16_t kgrid_e[512] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
          260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
          352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
          640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
         1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
         1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
         2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
         2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
         4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
         4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
         5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
         5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
         8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
         8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
        10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
        16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
        16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
        16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
        17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
        18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
        20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
        21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
        22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
        24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
        32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
        33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
        33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
        35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
        37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
        40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
        42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
        };
        const uint8_t *bptr = row;
        int nb = dim / 256;
        float *y = out;
        for (int b = 0; b < nb; b++) {
            uint16_t dh;
            kmemcpy(&dh, bptr, 2);
            float d = fp16_to_fp32(dh);
            const uint16_t *qs = (const uint16_t *)(bptr + 2);
            const uint8_t  *sc = bptr + 66;
            bptr += 74;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                float dl0 = d * (0.5f + (sc[ib32] & 0xf)) * 0.25f;
                float dl1 = d * (0.5f + (sc[ib32] >> 4))  * 0.25f;
                for (int l = 0; l < 4; l++) {
                    float dl = (l < 2) ? dl0 : dl1;
                    uint16_t qv = qs[ib32 * 4 + l];
                    uint16_t gi = qv & 511;
                    uint8_t  s7 = (uint8_t)(qv >> 9);
                    uint8_t  s8 = s7 | (uint8_t)((__builtin_popcount(s7) & 1) << 7);
                    uint16_t gv = kgrid_e[gi];
                    for (int k = 0; k < 8; k++) {
                        float gk = (float)iq2xs_vals_e[(gv >> (2 * k)) & 3];
                        if (s8 & (1u << k)) gk = -gk;
                        *y++ = dl * gk;
                    }
                }
            }
        }
        break;
    }
    case GGML_TYPE_Q2_K: {
        /* Q2_K: 256 elements/super-block, 84 bytes.
         * Layout: scales[16](nibble-packed) + qs[64] + d(fp16) + dmin(fp16). */
        const uint8_t *p = row;
        int nsb = dim / 256;
        float *outp = out;
        for (int b = 0; b < nsb; b++) {
            const uint8_t *scales = p;
            const uint8_t *qs     = p + 16;
            uint16_t dh, dmh;
            kmemcpy(&dh,  p + 80, 2);
            kmemcpy(&dmh, p + 82, 2);
            float d    = fp16_to_fp32(dh);
            float dmin = fp16_to_fp32(dmh);
            p += 84;
            int is = 0;
            const uint8_t *qb = qs;
            for (int half = 0; half < 2; half++) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0xF);
                    float ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        *outp++ = dl * (float)((qb[l] >> shift) & 3) - ml;
                    sc = scales[is++];
                    dl = d * (float)(sc & 0xF);
                    ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        *outp++ = dl * (float)((qb[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                qb += 32;
            }
        }
        break;
    }
    default:
        for (int i = 0; i < dim; i++) out[i] = 0.0f;
        break;
    }

    /* Gemma models scale embeddings by sqrt(dim) */
    if (m->embed_scale != 1.0f) {
        float s = m->embed_scale;
        for (int i = 0; i < dim; i++) out[i] *= s;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  RMSNorm                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

/* AVX2 RMSNorm for F32 weights (8-wide, 2× unroll) */
__attribute__((target("avx2,fma")))
static void llm_rmsnorm_avx2_f32(float *out, const float *x, const float *wf, int dim, float eps)
{
    v8f ss0 = v8f_setzero(), ss1 = v8f_setzero();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        v8f v0, v1;
        __builtin_memcpy(&v0, x + i, 32);
        __builtin_memcpy(&v1, x + i + 8, 32);
        ss0 = ss0 + v0 * v0;
        ss1 = ss1 + v1 * v1;
    }
    float ss = v8f_reduce(ss0 + ss1);
    for (; i < dim; i++)
        ss += x[i] * x[i];
    ss = 1.0f / llm_sqrtf(ss / (float)dim + eps);

    v8f ssv = v8f_set1(ss);
    for (i = 0; i + 8 <= dim; i += 8) {
        v8f xv, wv;
        __builtin_memcpy(&xv, x + i, 32);
        __builtin_memcpy(&wv, wf + i, 32);
        v8f r = xv * ssv * wv;
        __builtin_memcpy(out + i, &r, 32);
    }
    for (; i < dim; i++)
        out[i] = x[i] * ss * wf[i];
}

static void llm_rmsnorm(float *out, const float *x, const void *w,
                        int dim, ggml_type_t wtype, float eps)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable && wtype == GGML_TYPE_F32) {
        llm_rmsnorm_avx2_f32(out, x, (const float *)w, dim, eps);
        return;
    }

    /* SSE2 fallback for non-AVX2 or non-F32 weights */
    v4f ss_vec = {0, 0, 0, 0};
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        v4f v = *(const v4f *)(x + i);
        ss_vec += v * v;
    }
    union { v4f v; float f[4]; } u = { .v = ss_vec };
    float ss = u.f[0] + u.f[1] + u.f[2] + u.f[3];
    for (; i < dim; i++)
        ss += x[i] * x[i];
    ss = 1.0f / llm_sqrtf(ss / (float)dim + eps);

    /* Vectorized normalize + weight multiply */
    v4f ss_v = {ss, ss, ss, ss};
    for (i = 0; i + 4 <= dim; i += 4) {
        v4f xv = *(const v4f *)(x + i);
        v4f wv = {llm_get_f(w, i, wtype), llm_get_f(w, i+1, wtype),
                  llm_get_f(w, i+2, wtype), llm_get_f(w, i+3, wtype)};
        *(v4f *)(out + i) = xv * ss_v * wv;
    }
    for (; i < dim; i++)
        out[i] = x[i] * ss * llm_get_f(w, i, wtype);
#else
    /* NEON: vectorized sum-of-squares */
    float32x4_t vss = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vss = vmlaq_f32(vss, v, v);
    }
    float ss = vaddvq_f32(vss);
    for (; i < dim; i++) ss += x[i] * x[i];
    ss = 1.0f / llm_sqrtf(ss / (float)dim + eps);

    float32x4_t vscale = vdupq_n_f32(ss);
    for (i = 0; i + 4 <= dim && wtype == GGML_TYPE_F32; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32((const float *)w + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(xv, vscale), wv));
    }
    for (; i < dim; i++)
        out[i] = x[i] * ss * llm_get_f(w, i, wtype);
#endif
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  LayerNorm (Phi-2): subtract mean, normalize, scale + bias                  */
/*  out[i] = w[i] * (x[i] - mean) / sqrt(var + eps) + b[i]                   */
/* ─────────────────────────────────────────────────────────────────────────── */

__attribute__((target("avx2,fma")))
static void llm_layernorm_avx2_f32(float *out, const float *x, const float *wf,
                                    const float *bf, int dim)
{
    /* Compute mean */
    v8f sum0 = v8f_setzero(), sum1 = v8f_setzero();
    int i = 0;
    for (; i + 16 <= dim; i += 16) {
        v8f v0, v1;
        __builtin_memcpy(&v0, x + i, 32);
        __builtin_memcpy(&v1, x + i + 8, 32);
        sum0 = sum0 + v0;
        sum1 = sum1 + v1;
    }
    float mean = v8f_reduce(sum0 + sum1);
    for (; i < dim; i++) mean += x[i];
    mean /= (float)dim;

    /* Compute variance */
    v8f mv = v8f_set1(mean);
    v8f var0 = v8f_setzero(), var1 = v8f_setzero();
    for (i = 0; i + 16 <= dim; i += 16) {
        v8f v0, v1;
        __builtin_memcpy(&v0, x + i, 32);
        __builtin_memcpy(&v1, x + i + 8, 32);
        v8f d0 = v0 - mv, d1 = v1 - mv;
        var0 = var0 + d0 * d0;
        var1 = var1 + d1 * d1;
    }
    float var = v8f_reduce(var0 + var1);
    for (; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv_std = 1.0f / llm_sqrtf(var / (float)dim + 1e-5f);

    /* Normalize + scale + bias */
    v8f inv_v = v8f_set1(inv_std);
    for (i = 0; i + 8 <= dim; i += 8) {
        v8f xv, wv, bv;
        __builtin_memcpy(&xv, x + i, 32);
        __builtin_memcpy(&wv, wf + i, 32);
        v8f r = (xv - mv) * inv_v * wv;
        if (bf) {
            __builtin_memcpy(&bv, bf + i, 32);
            r = r + bv;
        }
        __builtin_memcpy(out + i, &r, 32);
    }
    for (; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * wf[i];
        if (bf) out[i] += bf[i];
    }
}

static void llm_layernorm(float *out, const float *x, const void *w,
                           const void *bias, int dim, ggml_type_t wtype)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable && wtype == GGML_TYPE_F32) {
        llm_layernorm_avx2_f32(out, x, (const float *)w,
                                bias ? (const float *)bias : NULL, dim);
        return;
    }
#endif
    /* Scalar fallback */
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv_std = 1.0f / llm_sqrtf(var / (float)dim + 1e-5f);

    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * llm_get_f(w, i, wtype);
        if (bias) out[i] += llm_get_f(bias, i, GGML_TYPE_F32);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  GELU activation (Phi-2 FFN uses GELU instead of SiLU/SwiGLU)              */
/*  GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))       */
/* ─────────────────────────────────────────────────────────────────────────── */

#ifndef __aarch64__
__attribute__((target("avx2,fma")))
static void llm_gelu_avx2(float *x, int n)
{
    v8f half   = v8f_set1(0.5f);
    v8f one    = v8f_set1(1.0f);
    v8f coeff  = v8f_set1(0.044715f);
    v8f sqrt2p = v8f_set1(0.7978845608f); /* sqrt(2/pi) */
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v8f t = sqrt2p * (v + coeff * v * v * v);
        /* tanh(t) ≈ 1 - 2/(1+exp(2t)) */
        v8f e2t = v8f_expf(t + t);
        v8f tanh_t = (e2t - one) / (e2t + one);
        v8f r = v * half * (one + tanh_t);
        __builtin_memcpy(x + i, &r, 32);
    }
    for (; i < n; i++) {
        float v = x[i];
        float t = 0.7978845608f * (v + 0.044715f * v * v * v);
        float e = llm_expf(2.0f * t);
        float tanh_val = (e - 1.0f) / (e + 1.0f);
        x[i] = v * 0.5f * (1.0f + tanh_val);
    }
}
#endif

static void llm_gelu(float *x, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        llm_gelu_avx2(x, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = 0.7978845608f * (v + 0.044715f * v * v * v);
        float e = llm_expf(2.0f * t);
        float tanh_val = (e - 1.0f) / (e + 1.0f);
        x[i] = v * 0.5f * (1.0f + tanh_val);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Bias addition helper — add F32 bias to output vector                       */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_add_bias(float *out, const void *bias, int n)
{
    if (!bias) return;
    const float *b = (const float *)bias;
    for (int i = 0; i < n; i++)
        out[i] += b[i];
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Rotary Position Embeddings (RoPE) — Optimized                              */
/*  Precomputed frequency table + fast sin/cos via range-reduced polynomial.   */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Precomputed inverse-frequency table: freq[i] = base^(-2i/dim)
 * Uses the dynamically allocated llm_rope_freqs_buf from scratch arena. */
static int   llm_rope_freqs_ready = 0;
static float llm_rope_base_cached = 0.0f;
static int   llm_rope_hdim_cached = 0;

/* Per-dimension RoPE JIT kernel pointer (set at model load) and dim cache.
 * Declared early because llm_rope() and llm_rope_precompute() reference them. */
static jit_rope_fn jit_fwd_rope_hd = NULL;
#define JIT_ROPE_CACHE_SIZE 8
static struct { int dim; jit_rope_fn fn; } jit_rope_dim_cache[JIT_ROPE_CACHE_SIZE];
static int jit_rope_dim_cache_n = 0;

static float llm_log_approx(float x)
{
    /* log(x) via IEEE754: extract exponent + polynomial on mantissa.
     * x = 2^e * m where m in [1,2). log(x) = e*ln2 + log(m).
     * log(m) ≈ Remez polynomial on [1,2). */
    union { float f; unsigned int u; } u = { .f = x };
    int e = (int)((u.u >> 23) & 0xFF) - 127;
    u.u = (u.u & 0x007FFFFFu) | 0x3F800000u; /* m in [1,2) */
    float m = u.f;
    float m1 = m - 1.0f;
    /* Minimax log(1+m1) for m1 in [0,1): 4th order Remez */
    float logm = m1 * (0.9999964f - m1 * (0.4999899f - m1 * (0.3334508f - m1 * 0.2414937f)));
    return (float)e * 0.69314718f + logm;
}

static void llm_rope_precompute(float base, int head_dim, const float *factors)
{
    if (llm_rope_freqs_ready && llm_rope_base_cached == base &&
        llm_rope_hdim_cached == head_dim)
        return;
    if (!llm_rope_freqs_buf) return; /* not yet allocated */
    float log_base = llm_log_approx(base);
    for (int i = 0; i < head_dim / 2; i++) {
        float exponent = -(float)(2 * i) / (float)head_dim;
        float freq = llm_expf(exponent * log_base);
        /* Longrope: divide frequency by scaling factor */
        if (factors)
            freq /= factors[i];
        llm_rope_freqs_buf[i] = freq;
    }
    llm_rope_base_cached = base;
    llm_rope_hdim_cached = head_dim;
    llm_rope_freqs_ready = 1;
}

static void llm_rope(float *vec, int pos, int head_dim, float base,
                     const float *factors)
{
    llm_rope_precompute(base, head_dim, factors);

#ifndef __aarch64__
    /* JIT RoPE: processes 2 pairs (4 floats) per SIMD iteration using
     * polynomial sin/cos — ~2× faster than the scalar pair-by-pair path.
     * Only valid when: no custom scaling (factors==NULL) and head_dim
     * matches the compiled kernel dim. Gemma4 partial RoPE handled by
     * jit_rope_dim_cache which has per-rope_dim compiled kernels. */
    if (factors == NULL && jit_fwd_rope_hd && head_dim == llm_rope_hdim_cached) {
        jit_fwd_rope_hd(vec, pos, head_dim, llm_rope_freqs_buf);
        return;
    }
    /* Partial RoPE cache (Gemma4: rope_dim != head_dim) */
    if (factors == NULL && head_dim == llm_rope_hdim_cached) {
        for (int ci = 0; ci < jit_rope_dim_cache_n; ci++) {
            if (jit_rope_dim_cache[ci].dim == head_dim && jit_rope_dim_cache[ci].fn) {
                jit_rope_dim_cache[ci].fn(vec, pos, head_dim, llm_rope_freqs_buf);
                return;
            }
        }
    }
#endif

    for (int i = 0; i < head_dim; i += 2) {
        float theta = (float)pos * llm_rope_freqs_buf[i / 2];
        float cos_t = llm_cosf(theta);
        float sin_t = llm_sinf(theta);

        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i]     = v0 * cos_t - v1 * sin_t;
        vec[i + 1] = v0 * sin_t + v1 * cos_t;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Softmax                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

/* AVX2 softmax: 8-wide max, exp, and divide */
__attribute__((target("avx2,fma")))
static void llm_softmax_avx2(float *x, int n)
{
    /* 8-wide find-max */
    v8f mv = v8f_set1(x[0]);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        mv = v8f_max(mv, v);
    }
    union { v8f vec; float f[8]; } mu = { .vec = mv };
    float max_val = mu.f[0];
    for (int j = 1; j < 8; j++)
        if (mu.f[j] > max_val) max_val = mu.f[j];
    for (; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    /* Vectorized exp + sum */
    v8f sv = v8f_setzero();
    v8f mv8 = v8f_set1(max_val);
    for (i = 0; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v8f e = v8f_expf(v - mv8);
        __builtin_memcpy(x + i, &e, 32);
        sv = sv + e;
    }
    float sum = v8f_reduce(sv);
    for (; i < n; i++) {
        x[i] = llm_expf(x[i] - max_val);
        sum += x[i];
    }

    /* Vectorized divide */
    float inv = 1.0f / (sum + 1e-10f);
    v8f inv_v = v8f_set1(inv);
    for (i = 0; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        v = v * inv_v;
        __builtin_memcpy(x + i, &v, 32);
    }
    for (; i < n; i++)
        x[i] *= inv;
}

static void llm_softmax(float *x, int n)
{
    if (n <= 0) return;

#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        llm_softmax_avx2(x, n);
        return;
    }
    /* SSE2-only path: use JIT variable-length softmax dispatcher.
     * Dispatches to the smallest bucket >= n, compiled at init time.
     * Falls back to scalar if n > 8192 or no bucket found. */
    {
        jit_softmax_var_fn sfn = jit_get_softmax_var_kernel();
        if (sfn) { sfn(x, n); return; }
    }
#endif

    /* Scalar fallback */
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = llm_expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < n; i++)
        x[i] *= inv;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  SiLU (Sigmoid Linear Unit) — used in SwiGLU FFN — Vectorized              */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Fast vectorized exp for small batches (Padé with better range reduction) */
static inline float llm_fast_sigmoid(float x)
{
    /* Clamp to avoid overflow */
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    float e = llm_expf(-x);
    return 1.0f / (1.0f + e);
}

#ifndef __aarch64__
__attribute__((target("avx2,fma")))
static void llm_silu_avx2(float *x, int n)
{
    v8f one = v8f_set1(1.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f v; __builtin_memcpy(&v, x + i, 32);
        /* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
        v8f e = v8f_expf(v8f_setzero() - v);
        v8f r = v / (one + e);
        __builtin_memcpy(x + i, &r, 32);
    }
    for (; i < n; i++) {
        float s = llm_fast_sigmoid(x[i]);
        x[i] *= s;
    }
}
#endif

static void llm_silu(float *x, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        llm_silu_avx2(x, n);
        return;
    }
    /* SSE2 4-wide unrolled */
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float s0 = llm_fast_sigmoid(x[i]);
        float s1 = llm_fast_sigmoid(x[i+1]);
        float s2 = llm_fast_sigmoid(x[i+2]);
        float s3 = llm_fast_sigmoid(x[i+3]);
        x[i] *= s0; x[i+1] *= s1; x[i+2] *= s2; x[i+3] *= s3;
    }
    for (; i < n; i++) {
        float s = llm_fast_sigmoid(x[i]);
        x[i] *= s;
    }
#else
    for (int i = 0; i < n; i++) {
        float s = 1.0f / (1.0f + llm_expf(-x[i]));
        x[i] = x[i] * s;
    }
#endif
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Vectorized helpers for attention inner loops (SSE2 + AVX2 dispatch)        */
/* ─────────────────────────────────────────────────────────────────────────── */

#ifndef __aarch64__
/* AVX2 dot product: 8-wide with 2× unroll */
__attribute__((target("avx2,fma")))
static float llm_dot_f32_avx2(const float *a, const float *b, int n)
{
    v8f s0 = v8f_setzero(), s1 = v8f_setzero();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        v8f a0, a1, b0, b1;
        __builtin_memcpy(&a0, a+i, 32);
        __builtin_memcpy(&a1, a+i+8, 32);
        __builtin_memcpy(&b0, b+i, 32);
        __builtin_memcpy(&b1, b+i+8, 32);
        s0 = s0 + a0 * b0;
        s1 = s1 + a1 * b1;
    }
    float sum = v8f_reduce(s0 + s1);
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/* AVX2 axpy: dst += scale * src */
__attribute__((target("avx2,fma")))
static void llm_axpy_f32_avx2(float *dst, float scale, const float *src, int n)
{
    v8f sv = v8f_set1(scale);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v8f d, s;
        __builtin_memcpy(&d, dst+i, 32);
        __builtin_memcpy(&s, src+i, 32);
        d = d + sv * s;
        __builtin_memcpy(dst+i, &d, 32);
    }
    for (; i < n; i++)
        dst[i] += scale * src[i];
}
#endif /* !__aarch64__ */

/* Dot product of two float arrays */
static float llm_dot_f32(const float *a, const float *b, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable)
        return llm_dot_f32_avx2(a, b, n);
    v4f acc0 = {0, 0, 0, 0};
    v4f acc1 = {0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        v4f a0 = *(const v4f *)(a + i);
        v4f a1 = *(const v4f *)(a + i + 4);
        v4f b0 = *(const v4f *)(b + i);
        v4f b1 = *(const v4f *)(b + i + 4);
        acc0 += a0 * b0;
        acc1 += a1 * b1;
    }
    v4f acc = acc0 + acc1;
    union { v4f v; float f[4]; } u = { .v = acc };
    float sum = u.f[0] + u.f[1] + u.f[2] + u.f[3];
    for (; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
#endif
}

/* Scaled add: dst[i] += scale * src[i] */
static void llm_axpy_f32(float *dst, float scale, const float *src, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        llm_axpy_f32_avx2(dst, scale, src, n);
        return;
    }
    v4f sv = {scale, scale, scale, scale};
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        v4f d = *(const v4f *)(dst + i);
        v4f s = *(const v4f *)(src + i);
        *(v4f *)(dst + i) = d + sv * s;
    }
    for (; i < n; i++)
        dst[i] += scale * src[i];
#else
    for (int i = 0; i < n; i++)
        dst[i] += scale * src[i];
#endif
}

/* Vector add: dst[i] += src[i] */
static void llm_vadd_f32(float *dst, const float *src, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            v8f d, s;
            __builtin_memcpy(&d, dst+i, 32);
            __builtin_memcpy(&s, src+i, 32);
            d = d + s;
            __builtin_memcpy(dst+i, &d, 32);
        }
        for (; i < n; i++) dst[i] += src[i];
        return;
    }
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        v4f d = *(const v4f *)(dst + i);
        v4f s = *(const v4f *)(src + i);
        *(v4f *)(dst + i) = d + s;
    }
    for (; i < n; i++)
        dst[i] += src[i];
#else
    for (int i = 0; i < n; i++)
        dst[i] += src[i];
#endif
}

/* Element-wise multiply: dst[i] *= src[i] */
static void llm_vmul_f32(float *dst, const float *src, int n)
{
#ifndef __aarch64__
    if (cpu_features.avx2_usable) {
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            v8f d, s;
            __builtin_memcpy(&d, dst+i, 32);
            __builtin_memcpy(&s, src+i, 32);
            d = d * s;
            __builtin_memcpy(dst+i, &d, 32);
        }
        for (; i < n; i++) dst[i] *= src[i];
        return;
    }
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        v4f d = *(const v4f *)(dst + i);
        v4f s = *(const v4f *)(src + i);
        *(v4f *)(dst + i) = d * s;
    }
    for (; i < n; i++)
        dst[i] *= src[i];
#else
    for (int i = 0; i < n; i++)
        dst[i] *= src[i];
#endif
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Transformer Forward Pass — Single Token                                    */
/*                                                                             */
/*  Process one token through the full transformer stack (incremental decode).  */
/*  Updates KV-cache and returns logits[vocab_size].                           */
/* ─────────────────────────────────────────────────────────────────────────── */

/* === JIT kernel cache for the forward pass (lazy-compiled once) === */
static jit_fused_silu_mul_fn jit_fwd_fused_silu = NULL;
static jit_ewise_fn          jit_fwd_vadd_dim   = NULL;
static jit_ewise_fn          jit_fwd_vmul_ff    = NULL;
static jit_silu_fn           jit_fwd_gelu       = NULL;
static jit_dot_fn            jit_fwd_dot_hd     = NULL;
static jit_axpy_fn           jit_fwd_axpy_hd    = NULL;
static jit_rmsnorm_fn        jit_fwd_rmsnorm    = NULL;
static jit_rmsnorm_fn        jit_fwd_rmsnorm_lhd = NULL;  /* compiled for head_dim */
static jit_rmsnorm_scale_fn  jit_fwd_rmsnorm_sc = NULL;  /* fused RMSNorm+scale, baked at model dim */
static jit_softmax_fn        jit_fwd_softmax    = NULL;
static int                   jit_fwd_ready      = 0;

static jit_rope_fn jit_get_rope_for_dim(int dim)
{
    /* Check cache */
    for (int i = 0; i < jit_rope_dim_cache_n; i++) {
        if (jit_rope_dim_cache[i].dim == dim)
            return jit_rope_dim_cache[i].fn;
    }
    /* Compile new kernel */
    jit_rope_fn fn = jit_compile_rope_kernel(dim);
    if (fn && jit_rope_dim_cache_n < JIT_ROPE_CACHE_SIZE) {
        jit_rope_dim_cache[jit_rope_dim_cache_n].dim = dim;
        jit_rope_dim_cache[jit_rope_dim_cache_n].fn = fn;
        jit_rope_dim_cache_n++;
    }
    return fn;
}

/* Forward declarations: configured later with defaults and setters. */
static int                   llm_attnres_enabled;
static float                 llm_attnres_strength;
static int                   llm_depth_attn_enabled;
static float                 llm_depth_attn_strength;
static int                   llm_depth_attn_window;

/* ═══════════════════════════════════════════════════════════════════════════
 * Batch Prefill — allocate GPU scratch and run all N prompt tokens at once
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifdef ENABLE_CUDA
static void llm_forward_token(llm_model_t *m, float *logits, int token_id, int pos);  /* forward decl */
static int llm_ensure_prefill_scratch(llm_model_t *m, int n) {
    if (n <= 0 || !gpu_ctx.gpu_fwd) return -1;
    if (gpu_ctx.prefill_max_n >= n && gpu_ctx.d_pfx) return 0;
    const backend_t *be = backend_get_by_id(BACKEND_CUDA);
    if (!be) return -1;

#define PF_FREE(p) do { if (gpu_ctx.p) { be->mem.free(gpu_ctx.p); gpu_ctx.p = NULL; } } while(0)
    PF_FREE(d_pfx); PF_FREE(d_pfxn); PF_FREE(d_pfq);
    PF_FREE(d_pfk); PF_FREE(d_pfv); PF_FREE(d_pfattn);
    PF_FREE(d_pffg); PF_FREE(d_pffu); PF_FREE(d_pffd);
    PF_FREE(d_pfiswa); PF_FREE(d_pfiswa_proj);
    gpu_ctx.prefill_max_n = 0;
#undef PF_FREE

    int dim        = m->dim;
    int max_q_dim  = m->n_heads * m->head_dim;
    int max_kv_dim = m->n_kv_heads * m->head_dim;
    int max_ff     = gpu_ctx.max_ff;
    int iswa_d     = m->iswa_n_embd;
    int iswa_total = (iswa_d > 0) ? iswa_d * m->n_layers : 0;

    uint64_t sz_x    = (uint64_t)n * dim       * sizeof(float);
    uint64_t sz_q    = (uint64_t)n * max_q_dim  * sizeof(float);
    uint64_t sz_kv   = (uint64_t)n * max_kv_dim * sizeof(float);
    uint64_t sz_ff   = (uint64_t)n * max_ff     * sizeof(float);
    uint64_t sz_iswa = (iswa_total > 0) ? (uint64_t)n * iswa_total * sizeof(float) : 0;

    gpu_ctx.d_pfx    = (float *)be->mem.alloc(sz_x);
    gpu_ctx.d_pfxn   = (float *)be->mem.alloc(sz_x);
    gpu_ctx.d_pfq    = (float *)be->mem.alloc(sz_q);
    gpu_ctx.d_pfk    = (float *)be->mem.alloc(sz_kv);
    gpu_ctx.d_pfv    = (float *)be->mem.alloc(sz_kv);
    gpu_ctx.d_pfattn = (float *)be->mem.alloc(sz_q);
    gpu_ctx.d_pffg   = (float *)be->mem.alloc(sz_ff);
    gpu_ctx.d_pffu   = (float *)be->mem.alloc(sz_ff);
    gpu_ctx.d_pffd   = (float *)be->mem.alloc(sz_x);
    if (iswa_total > 0) {
        gpu_ctx.d_pfiswa      = (float *)be->mem.alloc(sz_iswa);
        gpu_ctx.d_pfiswa_proj = (float *)be->mem.alloc(sz_iswa);
    }

    int ok = gpu_ctx.d_pfx && gpu_ctx.d_pfxn && gpu_ctx.d_pfq &&
             gpu_ctx.d_pfk && gpu_ctx.d_pfv  && gpu_ctx.d_pfattn &&
             gpu_ctx.d_pffg && gpu_ctx.d_pffu && gpu_ctx.d_pffd;
    if (iswa_total > 0)
        ok = ok && gpu_ctx.d_pfiswa && gpu_ctx.d_pfiswa_proj;

    if (!ok) {
#define PF_FREE2(p) do { if (gpu_ctx.p) { be->mem.free(gpu_ctx.p); gpu_ctx.p = NULL; } } while(0)
        PF_FREE2(d_pfx); PF_FREE2(d_pfxn); PF_FREE2(d_pfq);
        PF_FREE2(d_pfk); PF_FREE2(d_pfv); PF_FREE2(d_pfattn);
        PF_FREE2(d_pffg); PF_FREE2(d_pffu); PF_FREE2(d_pffd);
        PF_FREE2(d_pfiswa); PF_FREE2(d_pfiswa_proj);
#undef PF_FREE2
        return -1;
    }

    cuda_prefill_batch_presized(n, max_ff > dim ? max_ff : dim);
    gpu_ctx.prefill_max_n = n;
    uint64_t total_mb = (sz_x * 2 + sz_q * 2 + sz_kv * 2 + sz_ff * 2 + sz_x + sz_iswa * 2) >> 20;
    kprintf("[GPU] batch-prefill scratch: n=%d dim=%d ff=%d (~%llu MB)\n",
            n, dim, max_ff, (unsigned long long)total_mb);
    return 0;
}

/* Per-token compressed GEMV for prefill batch.
 * Used when raw GPU FFN weights are freed but compressed (U/S/Vt) are on GPU.
 * _out: [n × _rows] output, _in: [n × _cols] input, both in device memory. */
#define PF_CMPR_GEMV_N(_out, _in, _n, _rows, _cols, _cw) \
    do { for (int _pfi = 0; _pfi < (_n); _pfi++) { \
        be->compute.gemv(gpu_ctx.d_compress_tmp, (_cw)->d_Vt, \
                         (_in)  + (int64_t)(_pfi) * (_cols), \
                         (_cw)->rank, (_cw)->n, GGML_TYPE_F32); \
        be->compute.gemv((_out) + (int64_t)(_pfi) * (_rows), \
                         (_cw)->d_U, gpu_ctx.d_compress_tmp, \
                         (_cw)->m, (_cw)->rank, GGML_TYPE_F32); \
    } } while (0)

/* Run a batched GPU prefill for N tokens starting at start_pos.             */
static void llm_forward_prefill_batch_gpu(
    llm_model_t *m,
    const int   *tokens,
    int          n,
    int          start_pos)
{
    const backend_t *be = backend_get_by_id(BACKEND_CUDA);
    int dim    = m->dim;
    int n_heads = m->n_heads;
    int n_kv   = m->n_kv_heads;
    int hd     = m->head_dim;
    int ff     = m->ff_dim;
    int max_seq = gpu_ctx.max_seq;

    float *d_pfx    = gpu_ctx.d_pfx;
    float *d_pfxn   = gpu_ctx.d_pfxn;
    float *d_pfq    = gpu_ctx.d_pfq;
    float *d_pfk    = gpu_ctx.d_pfk;
    float *d_pfv    = gpu_ctx.d_pfv;
    float *d_pfattn = gpu_ctx.d_pfattn;
    float *d_pffg   = gpu_ctx.d_pffg;
    float *d_pffu   = gpu_ctx.d_pffu;
    float *d_pffd   = gpu_ctx.d_pffd;

    /* 1. Embed all tokens */
    void *d_embd = llm_gpu_lookup(m->token_embd);
    int embed_on_gpu = d_embd && (m->token_embd_type == GGML_TYPE_Q4_0 ||
                                   m->token_embd_type == GGML_TYPE_F32);
    for (int i = 0; i < n; i++) {
        float *xi = d_pfx + (int64_t)i * dim;
        if (embed_on_gpu) {
            be->compute.embed_lookup(xi, d_embd, tokens[i], dim, m->token_embd_type);
            if (m->embed_scale != 0.0f && m->embed_scale != 1.0f)
                be->compute.scale(xi, xi, m->embed_scale, dim);
        } else {
            llm_embed(llm_x, m, tokens[i]);
            be->mem.upload(xi, llm_x, (uint64_t)dim * sizeof(float));
        }
    }

    /* 2. ISWA batch precompute */
    if (m->is_gemma4 && m->iswa_tok_embd && m->iswa_model_proj &&
        gpu_ctx.d_pfiswa && gpu_ctx.d_pfiswa_proj) {
        int iswa_d     = m->iswa_n_embd;
        int iswa_total = iswa_d * m->n_layers;
        float inv_sqrt_dim = 1.0f / llm_sqrtf((float)dim);
        float sqrt_iswa_d  = llm_sqrtf((float)iswa_d);
        float inv_sqrt_2   = 1.0f / llm_sqrtf(2.0f);

        if (gpu_ctx.d_iswa_tok_embd) {
            for (int i = 0; i < n; i++) {
                float *dst = gpu_ctx.d_pfiswa + (int64_t)i * iswa_total;
                be->compute.embed_lookup(dst, gpu_ctx.d_iswa_tok_embd,
                                         tokens[i], iswa_total,
                                         gpu_ctx.iswa_tok_embd_type);
                be->compute.scale(dst, dst, sqrt_iswa_d, iswa_total);
            }
        } else {
            uint64_t emb_rb = llm_row_bytes(iswa_total, m->iswa_tok_embd_type);
            for (int i = 0; i < n; i++) {
                float *dst = gpu_ctx.d_pfiswa + (int64_t)i * iswa_total;
                if (m->iswa_tok_embd_type == GGML_TYPE_Q6_K) {
                    const uint8_t *emb_row = (const uint8_t *)m->iswa_tok_embd
                                             + (uint64_t)tokens[i] * emb_rb;
                    const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)emb_row;
                    int nsb = iswa_total / 256;
                    for (int sb = 0; sb < nsb; sb++) {
                        const ggml_q6_k_t *b = &blocks[sb];
                        float d_val = fp16_to_fp32(b->d);
                        float *o = llm_iswa_per_layer + sb * 256;
                        for (int half = 0; half < 2; half++) {
                            const uint8_t *ql_h = b->ql + half * 64;
                            const uint8_t *qh_h = b->qh + half * 32;
                            const int8_t  *sc_h = b->scales + half * 8;
                            for (int l = 0; l < 32; l++) {
                                int q0 = (int)(ql_h[l]    & 0xF) | (int)(((qh_h[l]>>0)&3)<<4);
                                int q1 = (int)(ql_h[l+32] & 0xF) | (int)(((qh_h[l]>>2)&3)<<4);
                                int q2 = (int)(ql_h[l]    >> 4)  | (int)(((qh_h[l]>>4)&3)<<4);
                                int q3 = (int)(ql_h[l+32] >> 4)  | (int)(((qh_h[l]>>6)&3)<<4);
                                int si = l / 16;
                                o[half*128 + l]      = d_val*(float)sc_h[0+si]*(float)(q0-32);
                                o[half*128 + l + 32] = d_val*(float)sc_h[2+si]*(float)(q1-32);
                                o[half*128 + l + 64] = d_val*(float)sc_h[4+si]*(float)(q2-32);
                                o[half*128 + l + 96] = d_val*(float)sc_h[6+si]*(float)(q3-32);
                            }
                        }
                    }
                    for (int k = 0; k < iswa_total; k++) llm_iswa_per_layer[k] *= sqrt_iswa_d;
                } else {
                    for (int k = 0; k < iswa_total; k++)
                        llm_iswa_per_layer[k] = llm_get_f(m->iswa_tok_embd, k,
                                                           m->iswa_tok_embd_type) * sqrt_iswa_d;
                }
                be->mem.upload(dst, llm_iswa_per_layer, (uint64_t)iswa_total * sizeof(float));
            }
        }

        void *d_model_proj = llm_gpu_lookup(m->iswa_model_proj);
        if (d_model_proj) {
            int batch_ok = 0;
            if (m->iswa_model_proj_type == GGML_TYPE_Q4_0) {
                cuda_prefill_batch_quant(d_pfx, n, dim);
                batch_ok = cuda_prefill_batch_gemv_q4(gpu_ctx.d_pfiswa_proj,
                                                       d_model_proj,
                                                       iswa_total, dim, n);
            }
            if (!batch_ok) {
                for (int i = 0; i < n; i++)
                    be->compute.gemv(gpu_ctx.d_pfiswa_proj + (int64_t)i * iswa_total,
                                      d_model_proj,
                                      d_pfx + (int64_t)i * dim,
                                      iswa_total, dim, m->iswa_model_proj_type);
            }
            be->compute.scale(gpu_ctx.d_pfiswa_proj, gpu_ctx.d_pfiswa_proj,
                               inv_sqrt_dim, n * iswa_total);
            if (gpu_ctx.d_iswa_proj_norm)
                cuda_batched_rmsnorm(gpu_ctx.d_pfiswa_proj, gpu_ctx.d_iswa_proj_norm,
                                      n * m->n_layers, iswa_d, m->rms_eps);
            cuda_iswa_combine(gpu_ctx.d_pfiswa, gpu_ctx.d_pfiswa,
                               gpu_ctx.d_pfiswa_proj, inv_sqrt_2, n * iswa_total);
        }
    }

    /* 3. Per-layer transform */
#define BGEMV4(out, W, od, id, type) do { \
    if ((type) == GGML_TYPE_Q4_0) { \
        cuda_prefill_batch_quant(d_pfxn, n, (id)); \
        cuda_prefill_batch_gemv_q4((out), (W), (od), (id), n); \
    } else { \
        for (int _i = 0; _i < n; _i++) \
            be->compute.gemv((out) + (int64_t)_i*(od), (W), \
                             d_pfxn + (int64_t)_i*(id), (od), (id), (type)); \
    } \
} while(0)

    for (int L = 0; L < m->n_layers; L++) {
        llm_layer_t *layer = &m->layers[L];
        gpu_layer_t *gl    = &gpu_ctx.layers[L];
        int lhd     = layer->head_dim_layer ? layer->head_dim_layer : hd;
        int lkv_dim = n_kv * lhd;
        int lq_dim  = n_heads * lhd;
        int lff     = layer->ff_dim_layer  ? layer->ff_dim_layer  : ff;
        int has_own_kv = (layer->kv_reuse_layer < 0);

        /* Pre-attn batched RMSNorm (out-of-place) */
        cuda_batched_rmsnorm_out(d_pfxn, d_pfx, gl->d_attn_norm, n, dim, gpu_ctx.rms_eps);

        /* QKV projections — SVD-compressed fallback for Q if raw weight freed */
        {
            void *d_qw = llm_gpu_lookup(layer->q_weight);
            const axex_compressed_weight_t *_pf_cq = axex_get_compressed_layer(L, 1);
            if (!d_qw && _pf_cq && _pf_cq->d_Vt) {
                PF_CMPR_GEMV_N(d_pfq, d_pfxn, n, lq_dim, dim, _pf_cq);
            } else {
                BGEMV4(d_pfq, d_qw, lq_dim, dim, layer->q_type);
            }
        }
        if (has_own_kv) {
            void *d_kw = llm_gpu_lookup(layer->k_weight);
            void *d_vw = llm_gpu_lookup(layer->v_weight);
            BGEMV4(d_pfk, d_kw, lkv_dim, dim, layer->k_type);
            BGEMV4(d_pfv, d_vw, lkv_dim, dim, layer->v_type);
        }

        /* 3c. Batched: RoPE + KV update + attention (all n tokens in parallel) */
        {
            float layer_rope_base = m->rope_base;
            if (m->is_gemma4 && lhd == m->head_dim_swa)
                layer_rope_base = m->rope_base_swa;
            float *d_rope_f = (m->is_gemma4 && m->rope_freqs && lhd != m->head_dim_swa)
                               ? gpu_ctx.d_rope_freqs : (float *)0;
            int rdim = m->rope_dim > 0 ? m->rope_dim : lhd;
            float attn_scale = m->is_gemma4 ? 1.0f : (1.0f / llm_sqrtf((float)lhd));

            /* Use batched attention kernels if available (loaded from DLL) */
            int have_batched_attn = cuda_have_batch_attn();

            if (have_batched_attn) {
                /* Batched path: one launch per step instead of n launches */
                if (layer->q_norm && has_own_kv) {
                    cuda_batch_fused_qk_norm_rope(d_pfq, d_pfk,
                        gl->d_q_norm, gl->d_k_norm,
                        n_heads, n_kv, lhd, n, start_pos,
                        layer_rope_base, d_rope_f, m->rms_eps, rdim);
                    cuda_batch_v_norm(d_pfv, n_kv, lhd, n, m->rms_eps);
                } else if (layer->q_norm) {
                    cuda_batch_fused_qk_norm_rope(d_pfq, (float *)0,
                        gl->d_q_norm, (float *)0,
                        n_heads, 0, lhd, n, start_pos,
                        layer_rope_base, d_rope_f, m->rms_eps, rdim);
                } else {
                    if (has_own_kv)
                        cuda_batch_fused_qk_norm_rope(d_pfq, d_pfk,
                            (float *)0, (float *)0,
                            n_heads, n_kv, lhd, n, start_pos,
                            layer_rope_base, d_rope_f, m->rms_eps, rdim);
                    else
                        cuda_batch_fused_qk_norm_rope(d_pfq, (float *)0,
                            (float *)0, (float *)0,
                            n_heads, 0, lhd, n, start_pos,
                            layer_rope_base, d_rope_f, m->rms_eps, rdim);
                }

                if (has_own_kv)
                    cuda_batch_kv_update(gl->d_k_cache, gl->d_v_cache,
                                         d_pfk, d_pfv, n_kv, lhd,
                                         n, start_pos, max_seq);

                {
                    int kv_src = has_own_kv ? L : layer->kv_reuse_layer;
                    gpu_layer_t *kv_gl = &gpu_ctx.layers[kv_src];
                    cuda_prefill_attn_batched(d_pfattn, d_pfq,
                        kv_gl->d_k_cache, kv_gl->d_v_cache,
                        n_heads, n_kv, lhd,
                        n, start_pos, max_seq, attn_scale, 0.0f);
                }
            } else {
                /* Fallback: sequential per-token loop */
                for (int i = 0; i < n; i++) {
                    int    pos_i = start_pos + i;
                    float *dqi   = d_pfq    + (int64_t)i * lq_dim;
                    float *dki   = d_pfk    + (int64_t)i * lkv_dim;
                    float *dvi   = d_pfv    + (int64_t)i * lkv_dim;
                    float *dai   = d_pfattn + (int64_t)i * lq_dim;
                    if (layer->q_norm && has_own_kv) {
                        cuda_fused_qk_norm_rope(dqi, dki, gl->d_q_norm, gl->d_k_norm,
                            n_heads, n_kv, lhd, pos_i, layer_rope_base, d_rope_f,
                            m->rms_eps, rdim);
                        cuda_v_norm(dvi, n_kv, lhd, m->rms_eps);
                    } else if (layer->q_norm) {
                        cuda_fused_qk_norm_rope(dqi, dqi, gl->d_q_norm, (float *)0,
                            n_heads, 0, lhd, pos_i, layer_rope_base, d_rope_f,
                            m->rms_eps, rdim);
                    } else {
                        if (has_own_kv)
                            be->compute.rope(dqi, dki, lhd, n_heads, n_kv,
                                             pos_i, layer_rope_base, d_rope_f);
                        else
                            be->compute.rope(dqi, dqi, lhd, n_heads, 0,
                                             pos_i, layer_rope_base, d_rope_f);
                    }
                    if (has_own_kv)
                        be->compute.kv_update(gl->d_k_cache, gl->d_v_cache,
                                               dki, dvi, n_kv, lhd, pos_i, max_seq, L);
                    int kv_src = has_own_kv ? L : layer->kv_reuse_layer;
                    gpu_layer_t *kv_gl = &gpu_ctx.layers[kv_src];
                    be->compute.attention(dai, dqi, kv_gl->d_k_cache, kv_gl->d_v_cache,
                                           n_heads, n_kv, lhd,
                                           pos_i + 1, max_seq, attn_scale, 0.0f);
                }
            }
        }
        /* O projection — SVD-compressed fallback if raw weight freed */
        {
            void *d_ow = llm_gpu_lookup(layer->o_weight);
            const axex_compressed_weight_t *_pf_co = axex_get_compressed_layer(L, 4);
            if (!d_ow && _pf_co && _pf_co->d_Vt) {
                PF_CMPR_GEMV_N(d_pffd, d_pfattn, n, dim, lq_dim, _pf_co);
            } else if (layer->o_type == GGML_TYPE_Q4_0) {
                cuda_prefill_batch_quant(d_pfattn, n, lq_dim);
                cuda_prefill_batch_gemv_q4(d_pffd, d_ow, dim, lq_dim, n);
            } else {
                for (int i = 0; i < n; i++)
                    be->compute.gemv(d_pffd + (int64_t)i * dim, d_ow,
                                     d_pfattn + (int64_t)i * lq_dim,
                                     dim, lq_dim, layer->o_type);
            }
        }

        /* Post-attn residual */
        if (layer->post_attn_norm && gl->d_post_attn_norm) {
            for (int i = 0; i < n; i++)
                cuda_rmsnorm_add(d_pfx + (int64_t)i * dim,
                                  d_pffd + (int64_t)i * dim,
                                  gl->d_post_attn_norm, dim, gpu_ctx.rms_eps);
        } else {
            be->compute.add(d_pfx, d_pfx, d_pffd, n * dim);
        }

        /* Pre-FFN batched RMSNorm */
        cuda_batched_rmsnorm_out(d_pfxn, d_pfx, gl->d_ffn_norm, n, dim, gpu_ctx.rms_eps);

        /* FFN — compressed-weight aware: if raw GPU tensor freed, use U/S/Vt path */
        if (m->use_gelu || !layer->ffn_gate) {
            void *d_upw   = llm_gpu_lookup(layer->ffn_up);
            void *d_downw = llm_gpu_lookup(layer->ffn_down);
            const axex_compressed_weight_t *_pf_cu = axex_get_compressed_layer(L, 5);
            const axex_compressed_weight_t *_pf_cd = axex_get_compressed_layer(L, 0);
            if (!d_upw && _pf_cu && _pf_cu->d_Vt) {
                PF_CMPR_GEMV_N(d_pffu, d_pfxn, n, lff, dim, _pf_cu);
            } else {
                BGEMV4(d_pffu, d_upw, lff, dim, layer->up_type);
            }
            be->compute.gelu(d_pffu, n * lff);
            if (!d_downw && _pf_cd && _pf_cd->d_Vt) {
                PF_CMPR_GEMV_N(d_pffd, d_pffu, n, dim, lff, _pf_cd);
            } else if (layer->down_type == GGML_TYPE_Q4_0) {
                cuda_prefill_batch_quant(d_pffu, n, lff);
                cuda_prefill_batch_gemv_q4(d_pffd, d_downw, dim, lff, n);
            } else {
                for (int i = 0; i < n; i++)
                    be->compute.gemv(d_pffd + (int64_t)i * dim, d_downw,
                                     d_pffu + (int64_t)i * lff, dim, lff, layer->down_type);
            }
        } else if (m->use_geglu) {
            void *d_gatew = llm_gpu_lookup(layer->ffn_gate);
            void *d_upw   = llm_gpu_lookup(layer->ffn_up);
            void *d_downw = llm_gpu_lookup(layer->ffn_down);
            const axex_compressed_weight_t *_pf_cg = axex_get_compressed_layer(L, 6);
            const axex_compressed_weight_t *_pf_cu = axex_get_compressed_layer(L, 5);
            const axex_compressed_weight_t *_pf_cd = axex_get_compressed_layer(L, 0);
            if (!d_gatew && _pf_cg && _pf_cg->d_Vt) {
                PF_CMPR_GEMV_N(d_pffg, d_pfxn, n, lff, dim, _pf_cg);
            } else {
                BGEMV4(d_pffg, d_gatew, lff, dim, layer->gate_type);
            }
            if (!d_upw && _pf_cu && _pf_cu->d_Vt) {
                PF_CMPR_GEMV_N(d_pffu, d_pfxn, n, lff, dim, _pf_cu);
            } else {
                BGEMV4(d_pffu, d_upw, lff, dim, layer->up_type);
            }
            cuda_fused_geglu(d_pffg, d_pffu, n * lff);
            if (!d_downw && _pf_cd && _pf_cd->d_Vt) {
                PF_CMPR_GEMV_N(d_pffd, d_pffg, n, dim, lff, _pf_cd);
            } else if (layer->down_type == GGML_TYPE_Q4_0) {
                cuda_prefill_batch_quant(d_pffg, n, lff);
                cuda_prefill_batch_gemv_q4(d_pffd, d_downw, dim, lff, n);
            } else {
                for (int i = 0; i < n; i++)
                    be->compute.gemv(d_pffd + (int64_t)i * dim, d_downw,
                                     d_pffg + (int64_t)i * lff, dim, lff, layer->down_type);
            }
        } else {
            void *d_gatew = llm_gpu_lookup(layer->ffn_gate);
            void *d_upw   = llm_gpu_lookup(layer->ffn_up);
            void *d_downw = llm_gpu_lookup(layer->ffn_down);
            const axex_compressed_weight_t *_pf_cg = axex_get_compressed_layer(L, 6);
            const axex_compressed_weight_t *_pf_cu = axex_get_compressed_layer(L, 5);
            const axex_compressed_weight_t *_pf_cd = axex_get_compressed_layer(L, 0);
            if (!d_gatew && _pf_cg && _pf_cg->d_Vt) {
                PF_CMPR_GEMV_N(d_pffg, d_pfxn, n, lff, dim, _pf_cg);
            } else {
                BGEMV4(d_pffg, d_gatew, lff, dim, layer->gate_type);
            }
            if (!d_upw && _pf_cu && _pf_cu->d_Vt) {
                PF_CMPR_GEMV_N(d_pffu, d_pfxn, n, lff, dim, _pf_cu);
            } else {
                BGEMV4(d_pffu, d_upw, lff, dim, layer->up_type);
            }
            cuda_fused_swiglu(d_pffg, d_pffu, n * lff);
            if (!d_downw && _pf_cd && _pf_cd->d_Vt) {
                PF_CMPR_GEMV_N(d_pffd, d_pffg, n, dim, lff, _pf_cd);
            } else if (layer->down_type == GGML_TYPE_Q4_0) {
                cuda_prefill_batch_quant(d_pffg, n, lff);
                cuda_prefill_batch_gemv_q4(d_pffd, d_downw, dim, lff, n);
            } else {
                for (int i = 0; i < n; i++)
                    be->compute.gemv(d_pffd + (int64_t)i * dim, d_downw,
                                     d_pffg + (int64_t)i * lff, dim, lff, layer->down_type);
            }
        }

        /* Post-FFW residual */
        if (layer->post_ffw_norm && gl->d_post_ffw_norm) {
            for (int i = 0; i < n; i++)
                cuda_rmsnorm_add(d_pfx + (int64_t)i * dim,
                                  d_pffd + (int64_t)i * dim,
                                  gl->d_post_ffw_norm, dim, gpu_ctx.rms_eps);
        } else {
            be->compute.add(d_pfx, d_pfx, d_pffd, n * dim);
        }

        /* ISWA injection per token */
        if (m->is_gemma4 && layer->iswa_inp_gate && gpu_ctx.d_pfiswa) {
            int iswa_d = m->iswa_n_embd;
            void *d_inp_gate = llm_gpu_lookup(layer->iswa_inp_gate);
            void *d_proj     = llm_gpu_lookup(layer->iswa_proj);
            if (d_inp_gate && d_proj && gpu_ctx.d_iswa_tmp) {
                int iswa_total = iswa_d * m->n_layers;
                for (int i = 0; i < n; i++) {
                    float *xi      = d_pfx + (int64_t)i * dim;
                    float *iswa_pl = gpu_ctx.d_pfiswa
                                     + (int64_t)i * iswa_total + (int64_t)L * iswa_d;
                    float *ffd_i   = d_pffd + (int64_t)i * dim;
                    be->compute.gemv(gpu_ctx.d_iswa_tmp, d_inp_gate, xi,
                                     iswa_d, dim, layer->iswa_inp_gate_type);
                    cuda_gelu_mul(gpu_ctx.d_iswa_tmp, iswa_pl, iswa_d);
                    be->compute.gemv(ffd_i, d_proj, gpu_ctx.d_iswa_tmp,
                                     dim, iswa_d, layer->iswa_proj_type);
                    if (gl->d_iswa_post_norm)
                        cuda_rmsnorm_add(xi, ffd_i, gl->d_iswa_post_norm,
                                          dim, gpu_ctx.rms_eps);
                    else
                        be->compute.add(xi, xi, ffd_i, dim);
                }
            }
        }

        /* Per-layer output scale */
        if (layer->iswa_out_scale) {
            float s = ((const float *)layer->iswa_out_scale)[0];
            be->compute.scale(d_pfx, d_pfx, s, n * dim);
        }
    }
#undef BGEMV4

    /* The caller will run token n-1 sequentially via llm_forward_token for
     * numerically-exact logits.  We are done: KV[0..n-1] are now populated. */
}

#endif /* ENABLE_CUDA */
/* ─── CPU single-layer forward for mixed GPU/CPU mode ───────────────────────
 * Processes one transformer layer entirely on CPU, in-place on llm_x.
 * Called from the GPU decode path when gl->on_gpu == 0.
 * Supports: SwiGLU / GeGLU / GELU FFN, GQA, RoPE, KV cache, compression.
 * ─────────────────────────────────────────────────────────────────────────*/
static void llm_cpu_layer(llm_model_t *m, int L, int pos)
{
    int dim     = m->dim;
    int n_heads = m->n_heads;
    int n_kv    = m->n_kv_heads;
    int hd      = m->head_dim;
    int ff      = m->ff_dim;
    int kv_dim  = n_kv * hd;

    llm_layer_t *layer = &m->layers[L];
    int lhd     = layer->head_dim_layer ? layer->head_dim_layer : hd;
    int lkv_dim = n_kv * lhd;
    int lq_dim  = n_heads * lhd;
    int lff     = layer->ff_dim_layer ? layer->ff_dim_layer : ff;
    int has_own_kv = (layer->kv_reuse_layer < 0);

    llm_rmsnorm(llm_xn, llm_x, layer->attn_norm, dim, layer->attn_norm_type, m->rms_eps);

    /* If per-layer GP bases are active, switch to this layer's Pt basis. */
    if (axex_manifold_has_layer_bases())
        axex_manifold_select_layer(L);

    /* Manifold projection: if any manifold-compressed weights exist for this
     * layer, compute x_sub = P^T @ xn once and reuse for Q, K, V, gate, up.
     * Stack-allocated (512 floats = 2 KB) — no heap cost on the hot path. */
    float x_sub[AXEX_MANIFOLD_K_MAX];
    int   x_sub_ready = 0;
    {
        const axex_manifold_weight_t *_mq = axex_get_manifold_layer(L, 1);
        const axex_manifold_weight_t *_mg = axex_get_manifold_layer(L, 6);
        if (_mq || _mg) {
            axex_manifold_project_x(llm_xn, x_sub);
            x_sub_ready = 1;
        }
    }

    /* Q/K/V: use manifold GEMV if available, otherwise full GEMV */
    {
        const axex_manifold_weight_t *_mq = axex_get_manifold_layer(L, 1);
        if (_mq && x_sub_ready)
            axex_manifold_weight_gemv(_mq, x_sub, llm_q);
        else
            llm_gemv(llm_q, layer->q_weight, llm_xn, lq_dim, dim, layer->q_type);
    }
    if (has_own_kv) {
        const axex_manifold_weight_t *_mk = axex_get_manifold_layer(L, 2);
        const axex_manifold_weight_t *_mv = axex_get_manifold_layer(L, 3);
        if (_mk && x_sub_ready)
            axex_manifold_weight_gemv(_mk, x_sub, llm_k_buf);
        else
            llm_gemv(llm_k_buf, layer->k_weight, llm_xn, lkv_dim, dim, layer->k_type);
        if (_mv && x_sub_ready)
            axex_manifold_weight_gemv(_mv, x_sub, llm_v_buf);
        else
            llm_gemv(llm_v_buf, layer->v_weight, llm_xn, lkv_dim, dim, layer->v_type);
    }

    {
        float rope_base = m->rope_base;
        int   rdim      = m->rope_dim > 0 ? m->rope_dim : lhd;
        for (int h = 0; h < n_heads; h++)
            llm_rope(llm_q + h * lhd, pos, rdim, rope_base, NULL);
        if (has_own_kv)
            for (int h = 0; h < n_kv; h++)
                llm_rope(llm_k_buf + h * lhd, pos, rdim, rope_base, NULL);
    }

    int64_t kv_stride = (int64_t)m->max_seq * kv_dim; /* int64_t: overflows int for large seq*kv_dim products */
    if (has_own_kv) {
        float *kc = m->k_cache + (int64_t)L * kv_stride + pos * kv_dim;
        float *vc = m->v_cache + (int64_t)L * kv_stride + pos * kv_dim;
        if (lkv_dim < kv_dim) {
            kmemset(kc, 0, kv_dim * sizeof(float));
            kmemset(vc, 0, kv_dim * sizeof(float));
        }
        axex_kv_ctx_t *_kv = axex_get_global_kv_ctx();
        for (int h = 0; h < n_kv; h++) {
            const float *k_head = llm_k_buf + h * lhd;
            if (_kv && pos > 0) {
                int store = axex_kv_should_store(_kv, k_head, lhd, L, h, pos);
                if (!store) {
                    int16_t tgt = _kv->heads[L * _kv->n_kv_heads + h].merge_into[pos];
                    if (tgt >= 0 && tgt < pos) {
                        kmemcpy(kc + h*hd, m->k_cache + (uint64_t)L*kv_stride + tgt*kv_dim + h*hd, lhd*sizeof(float));
                        kmemcpy(vc + h*hd, m->v_cache + (uint64_t)L*kv_stride + tgt*kv_dim + h*hd, lhd*sizeof(float));
                        continue;
                    }
                }
            }
            kmemcpy(kc + h*hd, k_head,            lhd*sizeof(float));
            kmemcpy(vc + h*hd, llm_v_buf + h*lhd, lhd*sizeof(float));
        }
    }

    kmemset(llm_attn_out, 0, lq_dim * sizeof(float));
    int kv_src = has_own_kv ? L : layer->kv_reuse_layer;
    {
        flash_attn_config_t fa;
        fa.head_dim      = lhd;
        fa.n_heads       = n_heads;
        fa.n_kv_heads    = n_kv;
        fa.seq_len       = m->max_seq;
        fa.block_size_q  = DEFAULT_BR;
        fa.block_size_kv = DEFAULT_BC;
        fa.scale         = 1.0f / llm_sqrtf((float)lhd);
        fa.causal        = true;
        fa.use_alibi     = false;
        flash_attn_decode_strided(
            llm_attn_out, llm_q,
            m->k_cache + (uint64_t)kv_src * kv_stride,
            m->v_cache + (uint64_t)kv_src * kv_stride,
            pos + 1, kv_dim, &fa);
    }

    /* O-proj: use GP manifold slot 4 when lq_dim == gp_n (residual stream match) */
    {
        const axex_manifold_weight_t *_gp_o = axex_get_manifold_layer(L, 4);
        if (_gp_o && _gp_o->W_proj && lq_dim == axex_manifold_n()) {
            float x_sub_o[AXEX_MANIFOLD_K_MAX];
            axex_manifold_project_x(llm_attn_out, x_sub_o);
            axex_manifold_weight_gemv(_gp_o, x_sub_o, llm_ffn_d);
        } else {
            llm_gemv(llm_ffn_d, layer->o_weight, llm_attn_out, dim, lq_dim, layer->o_type);
        }
    }
    llm_vadd_f32(llm_x, llm_ffn_d, dim);

    llm_rmsnorm(llm_xn, llm_x, layer->ffn_norm, dim, layer->ffn_norm_type, m->rms_eps);

    /* Recompute x_sub for FFN norm output if gate/up are manifold-compressed.
     * (xn is now the FFN-normed residual, which may differ from attn-normed xn) */
    if (!x_sub_ready) {
        const axex_manifold_weight_t *_mg6 = axex_get_manifold_layer(L, 6);
        const axex_manifold_weight_t *_mu5 = axex_get_manifold_layer(L, 5);
        if (_mg6 || _mu5) {
            axex_manifold_project_x(llm_xn, x_sub);
            x_sub_ready = 1;
        }
    } else {
        /* attn-norm and ffn-norm can differ — always re-project for FFN */
        const axex_manifold_weight_t *_mg6 = axex_get_manifold_layer(L, 6);
        const axex_manifold_weight_t *_mu5 = axex_get_manifold_layer(L, 5);
        if (_mg6 || _mu5)
            axex_manifold_project_x(llm_xn, x_sub);
    }

    if (m->use_gelu || !layer->ffn_gate) {
        llm_gemv(llm_ffn_u, layer->ffn_up, llm_xn, lff, dim, layer->up_type);
        llm_add_bias(llm_ffn_u, layer->ffn_up_bias, lff);
        if (jit_fwd_gelu && lff == ff) jit_fwd_gelu(llm_ffn_u, lff);
        else llm_gelu(llm_ffn_u, lff);
        const axex_compressed_weight_t *_cw = axex_get_compressed_layer(L, 0);
        if (_cw) axex_compressed_weight_gemv(_cw, llm_ffn_u, llm_ffn_d);
        else     llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_u, dim, lff, layer->down_type);
        llm_add_bias(llm_ffn_d, layer->ffn_down_bias, dim);
    } else if (m->use_geglu) {
        llm_gemv(llm_ffn_g, layer->ffn_gate, llm_xn, lff, dim, layer->gate_type);
        llm_gemv(llm_ffn_u, layer->ffn_up,   llm_xn, lff, dim, layer->up_type);
        if (jit_fwd_gelu && lff == ff) jit_fwd_gelu(llm_ffn_g, lff);
        else llm_gelu(llm_ffn_g, lff);
        if (jit_fwd_vmul_ff && lff == ff)
            jit_fwd_vmul_ff(llm_ffn_g, llm_ffn_u, lff);
        else
            llm_vmul_f32(llm_ffn_g, llm_ffn_u, lff);
        const axex_compressed_weight_t *_cw = axex_get_compressed_layer(L, 0);
        if (_cw) axex_compressed_weight_gemv(_cw, llm_ffn_g, llm_ffn_d);
        else     llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_g, dim, lff, layer->down_type);
    } else {
        /* SwiGLU */
        {
            const axex_manifold_weight_t *_mg = axex_get_manifold_layer(L, 6);
            const axex_compressed_weight_t *_cw_g = axex_get_compressed_layer(L, 6);
            if      (_mg   && x_sub_ready) axex_manifold_weight_gemv(_mg, x_sub, llm_ffn_g);
            else if (_cw_g)                axex_compressed_weight_gemv(_cw_g, llm_xn, llm_ffn_g);
            else                           llm_gemv(llm_ffn_g, layer->ffn_gate, llm_xn, lff, dim, layer->gate_type);
        }
        {
            const axex_manifold_weight_t *_mu = axex_get_manifold_layer(L, 5);
            const axex_compressed_weight_t *_cw_u = axex_get_compressed_layer(L, 5);
            if      (_mu   && x_sub_ready) axex_manifold_weight_gemv(_mu, x_sub, llm_ffn_u);
            else if (_cw_u)                axex_compressed_weight_gemv(_cw_u, llm_xn, llm_ffn_u);
            else                           llm_gemv(llm_ffn_u, layer->ffn_up,  llm_xn, lff, dim, layer->up_type);
        }
        if (jit_fwd_fused_silu && lff == ff) {
            jit_fwd_fused_silu(llm_ffn_g, llm_ffn_u, lff);
        } else {
            llm_silu(llm_ffn_g, lff);
            if (jit_fwd_vmul_ff && lff == ff)
                jit_fwd_vmul_ff(llm_ffn_g, llm_ffn_u, lff);
            else
                llm_vmul_f32(llm_ffn_g, llm_ffn_u, lff);
        }
        {
            const axex_compressed_weight_t *_cw = axex_get_compressed_layer(L, 0);
            if (_cw) axex_compressed_weight_gemv(_cw, llm_ffn_g, llm_ffn_d);
            else     llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_g, dim, lff, layer->down_type);
        }
    }
    if (jit_fwd_vadd_dim)
        jit_fwd_vadd_dim(llm_x, llm_ffn_d, dim);
    else
        llm_vadd_f32(llm_x, llm_ffn_d, dim);
}

static void llm_forward_token(llm_model_t *m, float *logits, int token_id, int pos)
{
    int dim = m->dim;
    int n_heads = m->n_heads;
    int n_kv = m->n_kv_heads;
    int hd = m->head_dim;
    int ff = m->ff_dim;
    int kv_dim = n_kv * hd;
    int want_logits = (logits != NULL);

    /* Lazy-compile JIT kernels on first call (dimensions fixed per model) */
#ifndef __aarch64__
    if (!jit_fwd_ready) {
        /* Compile ALL JIT kernels — baked dimensions eliminate loop overhead.
         * SSE2 JIT with constants often beats generic AVX2 C for small ops. */
        jit_fwd_vadd_dim   = jit_compile_vadd_kernel(dim);
        jit_fwd_vmul_ff    = jit_compile_vmul_kernel(ff);
        jit_fwd_fused_silu  = jit_compile_fused_silu_mul_kernel(ff);
        jit_fwd_gelu       = jit_compile_gelu_kernel(ff);
        jit_fwd_rope_hd    = jit_compile_rope_kernel(hd);
        jit_fwd_rmsnorm    = jit_compile_rmsnorm_kernel(dim);
        jit_fwd_rmsnorm_lhd = jit_compile_rmsnorm_kernel(hd);  /* per-head Q/K norm */
        jit_fwd_softmax    = NULL; /* softmax has variable length (seq_len) — skip */
        jit_fwd_ready = 1;
    }
#endif

    /* 1. Embedding lookup */

#ifdef ENABLE_CUDA
    /* ═════════════════════════════════════════════════════════════════════
     * GPU-Resident Forward Pass
     *
     * When gpu_ctx.gpu_fwd is set, the entire forward pass runs on GPU.
     * Activations stay in VRAM between operations — no PCIe round-trips
     * except: ISWA injection (small tensors) and final logits download.
     * ═════════════════════════════════════════════════════════════════════ */
    if (gpu_ctx.gpu_fwd &&
        /* Bridge capture requires CPU layer loop (no bridge support in GPU path).
         * Fall through to CPU path when capture mode is active so hidden-state
         * captures (OTT axiom probes) get real transformer activations.
         * BRIDGE_MODE_CAP_ALL (multi-layer calibration capture) and
         * BRIDGE_MODE_CAP_FFN (FFN activation capture for FFN-down GP compression)
         * also require CPU so the capture blocks at the end of the CPU layer loop
         * fire correctly. */
        !(llm_bridge.mode & (BRIDGE_MODE_CAPTURE | BRIDGE_MODE_CAP_ALL | BRIDGE_MODE_CAP_FFN))) {
        const backend_t *be = backend_get_by_id(BACKEND_CUDA);
        int max_seq = gpu_ctx.max_seq;
        int seq_len = pos + 1;
        int kv_stride_host = m->max_seq * n_kv * hd;
        int kv_dim_host = n_kv * hd;

        /* Profiling counters (accumulated per forward).
         * Opt-in via GD_PROFILE=1 and GD_PROFILE_DETAIL=1 to avoid contaminating normal benchmarks. */
        static int prof_init = 0;
        static int prof_enabled = 0;
        static int prof_detail = 0;
        if (!prof_init) {
            const char *env_prof = getenv("GD_PROFILE");
            const char *env_detail = getenv("GD_PROFILE_DETAIL");
            prof_enabled = (env_prof && env_prof[0] && strcmp(env_prof, "0") != 0) ? 1 : 0;
            prof_detail = (env_detail && env_detail[0] && strcmp(env_detail, "0") != 0) ? 1 : 0;
            prof_init = 1;
        }
        double t_embed = 0, t_iswa_pre = 0, t_layers = 0, t_lmhead = 0;
        double t_l_rmsnorm=0, t_l_qkv=0, t_l_rope=0, t_l_attn=0, t_l_oproj=0, t_l_ffn=0, t_l_iswa=0;
        uint64_t _pt0, _pt1;
        if (prof_enabled) { be->mem.sync(); _pt0 = hal_timer_us(); }

        /* 1. Embedding on GPU: GPU embed only handles Q4_0 and F32.
         * For other types (e.g. Q6_K), fall back to CPU dequant + upload. */
        uint64_t _emb0 = hal_timer_us();
        void *d_embd = llm_gpu_lookup(m->token_embd);
        int embed_on_gpu = (m->token_embd_type == GGML_TYPE_Q4_0 ||
                            m->token_embd_type == GGML_TYPE_F32);
        if (d_embd && embed_on_gpu) {
            be->compute.embed_lookup(gpu_ctx.d_x, d_embd, token_id, dim, m->token_embd_type);
            /* Gemma: scale embedding by sqrt(dim) — GPU embed_lookup only dequantizes */
            if (m->embed_scale != 0.0f && m->embed_scale != 1.0f)
                be->compute.scale(gpu_ctx.d_x, gpu_ctx.d_x, m->embed_scale, dim);
        } else {
            /* Fallback: embed on CPU (handles all types + embed_scale), upload */
            llm_embed(llm_x, m, token_id);
            be->mem.upload(gpu_ctx.d_x, llm_x, (uint64_t)dim * sizeof(float));
        }
        uint64_t _emb1 = hal_timer_us();

        /* === ISWA precomputation === */
        if (prof_enabled) { be->mem.sync(); _pt1 = hal_timer_us(); t_embed = (_pt1 - _pt0) / 1000.0; _pt0 = _pt1; }
        if (m->is_gemma4 && m->iswa_tok_embd && m->iswa_model_proj) {
            int iswa_d = m->iswa_n_embd;
            int iswa_total = iswa_d * m->n_layers;
            float inv_sqrt_dim = 1.0f / llm_sqrtf((float)dim);
            float sqrt_iswa_d  = llm_sqrtf((float)iswa_d);
            float inv_sqrt_2   = 1.0f / llm_sqrtf(2.0f);

            /* ISWA token embedding dequant: prefer GPU if tensor is uploaded */
            if (gpu_ctx.d_iswa_tok_embd) {
                /* GPU path: dequant one row directly on GPU — no CPU/disk access */
                be->compute.embed_lookup(gpu_ctx.d_iswa_per_layer,
                                         gpu_ctx.d_iswa_tok_embd, token_id,
                                         iswa_total, gpu_ctx.iswa_tok_embd_type);
                be->compute.scale(gpu_ctx.d_iswa_per_layer,
                                  gpu_ctx.d_iswa_per_layer, sqrt_iswa_d, iswa_total);
            } else {
                /* CPU fallback: dequant from mmap'd file + upload */
                uint64_t emb_rb = llm_row_bytes(iswa_total, m->iswa_tok_embd_type);
                const uint8_t *emb_row = (const uint8_t *)m->iswa_tok_embd
                                         + (uint64_t)token_id * emb_rb;
                if (m->iswa_tok_embd_type == GGML_TYPE_Q6_K) {
                    const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)emb_row;
                    int nsb = iswa_total / 256;
                    for (int sb = 0; sb < nsb; sb++) {
                        const ggml_q6_k_t *b = &blocks[sb];
                        float d_val = fp16_to_fp32(b->d);
                        float *o = llm_iswa_per_layer + sb * 256;
                        for (int half = 0; half < 2; half++) {
                            const uint8_t *ql_h = b->ql + half * 64;
                            const uint8_t *qh_h = b->qh + half * 32;
                            const int8_t *sc_h = b->scales + half * 8;
                            for (int l = 0; l < 32; l++) {
                                int q0 = (int)(ql_h[l] & 0xF)      | (int)(((qh_h[l] >> 0) & 3) << 4);
                                int q1 = (int)(ql_h[l + 32] & 0xF)  | (int)(((qh_h[l] >> 2) & 3) << 4);
                                int q2 = (int)(ql_h[l] >> 4)         | (int)(((qh_h[l] >> 4) & 3) << 4);
                                int q3 = (int)(ql_h[l + 32] >> 4)    | (int)(((qh_h[l] >> 6) & 3) << 4);
                                int si = l / 16;
                                o[half*128 + l]      = d_val * (float)sc_h[0 + si] * (float)(q0 - 32);
                                o[half*128 + l + 32] = d_val * (float)sc_h[2 + si] * (float)(q1 - 32);
                                o[half*128 + l + 64] = d_val * (float)sc_h[4 + si] * (float)(q2 - 32);
                                o[half*128 + l + 96] = d_val * (float)sc_h[6 + si] * (float)(q3 - 32);
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < iswa_total; i++)
                        llm_iswa_per_layer[i] = llm_get_f(m->iswa_tok_embd, i, m->iswa_tok_embd_type);
                }
                for (int i = 0; i < iswa_total; i++)
                    llm_iswa_per_layer[i] *= sqrt_iswa_d;

                be->mem.upload(gpu_ctx.d_iswa_per_layer, llm_iswa_per_layer,
                               (uint64_t)iswa_total * sizeof(float));
            }

            /* 2. Model projection GEMV: d_x → d_iswa_proj (fully on GPU) */
            if (prof_enabled) { be->mem.sync(); uint64_t _iswa_split = hal_timer_us(); _pt0 = _iswa_split; }
            void *d_model_proj = llm_gpu_lookup(m->iswa_model_proj);
            if (d_model_proj && gpu_ctx.d_iswa_proj) {
                be->compute.gemv(gpu_ctx.d_iswa_proj, d_model_proj, gpu_ctx.d_x,
                                 iswa_total, dim, m->iswa_model_proj_type);
                /* 3. Scale by 1/sqrt(dim) */
                be->compute.scale(gpu_ctx.d_iswa_proj, gpu_ctx.d_iswa_proj,
                                  inv_sqrt_dim, iswa_total);
                /* 4. Batched RMSNorm on projection (35 × 256 on GPU) */
                if (gpu_ctx.d_iswa_proj_norm)
                    cuda_batched_rmsnorm(gpu_ctx.d_iswa_proj, gpu_ctx.d_iswa_proj_norm,
                                         m->n_layers, iswa_d, m->rms_eps);
                /* 5. Combine: d_iswa_per_layer = (tok_embd + proj) * inv_sqrt_2 */
                cuda_iswa_combine(gpu_ctx.d_iswa_per_layer,
                                  gpu_ctx.d_iswa_per_layer, gpu_ctx.d_iswa_proj,
                                  inv_sqrt_2, iswa_total);
            } else {
                /* Fallback: CPU-hybrid if model_proj not on GPU */
                float *proj_buf = llm_ffn_g;
                be->mem.download(llm_x, gpu_ctx.d_x, (uint64_t)dim * sizeof(float));
                llm_gemv(proj_buf, m->iswa_model_proj, llm_x, iswa_total, dim,
                         m->iswa_model_proj_type);
                for (int i = 0; i < iswa_total; i++)
                    proj_buf[i] *= inv_sqrt_dim;
                if (m->iswa_proj_norm) {
                    const float *norm_w = (const float *)m->iswa_proj_norm;
                    for (int ll = 0; ll < m->n_layers; ll++) {
                        float *sl = proj_buf + ll * iswa_d;
                        float ss = 0.0f;
                        for (int i = 0; i < iswa_d; i++) ss += sl[i] * sl[i];
                        float rms = 1.0f / llm_sqrtf(ss / (float)iswa_d + m->rms_eps);
                        for (int i = 0; i < iswa_d; i++) sl[i] = sl[i] * rms * norm_w[i];
                    }
                }
                for (int i = 0; i < iswa_total; i++)
                    llm_iswa_per_layer[i] = (llm_iswa_per_layer[i] + proj_buf[i]) * inv_sqrt_2;
                be->mem.upload(gpu_ctx.d_iswa_per_layer, llm_iswa_per_layer,
                               (uint64_t)iswa_total * sizeof(float));
            }
        }
        uint64_t _iswa1 = hal_timer_us();

        /* === Per-layer transformer loop (GPU-resident) === */
        if (prof_enabled) { be->mem.sync(); _pt1 = hal_timer_us(); t_iswa_pre = (_pt1 - _pt0) / 1000.0; _pt0 = _pt1; }

        /* CUDA Graph: capture-once with device-side pos/seq_len vars.
         * First decode token: capture layer loop + rmsnorm + lmhead.
         * Subsequent tokens: replay graph (skip all compute dispatch). */
        const void *lm_head = m->output_weight ? m->output_weight : m->token_embd;
        ggml_type_t lm_type = m->output_weight ? m->output_type : m->token_embd_type;
        ggml_type_t d_lm_gpu_type = lm_type;  /* may be promoted to F32 if dequant-on-upload */
        void *d_lm = llm_gpu_lookup_with_type(lm_head, &d_lm_gpu_type);

        /* Graph REPLAY fast-path: skip entire compute.
         * Disabled when bridge capture is active (CAP_ALL): calibration needs
         * the per-layer hidden state downloads in the layerwise path. */
        if (want_logits && cuda_graph_captured && cuda_graph_decode_ready && d_lm &&
            !(llm_bridge.mode & BRIDGE_MODE_CAP_ALL)) {
            cuda_set_decode_pos(pos, pos + 1);
            cuda_graph_launch();
            if (!gpu_skip_logits_download) {
                be->mem.download(logits, gpu_ctx.d_out,
                                 (uint64_t)m->vocab_size * sizeof(float));
                be->mem.sync();
                /* Capture hidden state for context-conditioned manifold lookup */
                if (llm_last_hs)
                    be->mem.download(llm_last_hs, gpu_ctx.d_xn,
                                     (uint64_t)dim * sizeof(float));
                llm_last_hs_valid = (llm_last_hs != NULL);
            }
            return;
        }

        /* Graph CAPTURE: first decode token */
        int capturing = 0;
        /* GP manifold host calls (axex_get_manifold_layer, axex_manifold_select_layer)
         * are pure CPU pointer lookups — they do not issue CUDA API calls and are
         * therefore compatible with stream capture.  The device pointers they return
         * (d_W_proj, d_Pt) are static GPU allocations that never move, so the captured
         * kernel arguments remain valid on every replay. */
        int gp_active = (axex_manifold_compressed_count() > 0);
        if (want_logits && cuda_graph_decode_ready && !cuda_graph_tried && d_lm && !prof_enabled) {
            /* Pre-verify: ALL layer weights must be GPU-resident (raw tensor, GP manifold
             * d_W_proj, or SVD d_Vt) to avoid CPU fallback inside stream capture.
             * Types like IQ2_XS are dequantized to F32 on upload — all on GPU. */
            int iswa_ok = 1;
            for (int L2 = 0; L2 < m->n_layers && iswa_ok; L2++) {
                llm_layer_t *lyr = &m->layers[L2];
                /* GP manifold: Q/K/V/O/gate/up raw GPU tensors are freed after W_proj
                 * upload — accept GP d_W_proj as a valid GPU-resident path. */
                const axex_manifold_weight_t *_gm1 = axex_get_manifold_layer(L2, 1);
                const axex_manifold_weight_t *_gm2 = axex_get_manifold_layer(L2, 2);
                const axex_manifold_weight_t *_gm3 = axex_get_manifold_layer(L2, 3);
                const axex_manifold_weight_t *_gm4 = axex_get_manifold_layer(L2, 4);
                const axex_manifold_weight_t *_gm5 = axex_get_manifold_layer(L2, 5);
                const axex_manifold_weight_t *_gm6 = axex_get_manifold_layer(L2, 6);
                #define GP_COVERS(ptr_, gmw_) \
                    (!(ptr_) || llm_gpu_lookup(ptr_) || ((gmw_) && (gmw_)->d_W_proj))
                if (!GP_COVERS(lyr->q_weight, _gm1) ||
                    !GP_COVERS(lyr->k_weight, _gm2) ||
                    !GP_COVERS(lyr->v_weight, _gm3) ||
                    !GP_COVERS(lyr->o_weight, _gm4) ||
                    !GP_COVERS(lyr->ffn_gate, _gm6) ||
                    !GP_COVERS(lyr->ffn_up,   _gm5) ||
                    (lyr->ffn_down && !llm_gpu_lookup(lyr->ffn_down)))
                    iswa_ok = 0;
                #undef GP_COVERS
                if (m->is_gemma4 && lyr->iswa_inp_gate) {
                    if (!llm_gpu_lookup(lyr->iswa_inp_gate) ||
                        !llm_gpu_lookup(lyr->iswa_proj))
                        iswa_ok = 0;
                }
            }
            cuda_graph_tried = 1;
            /* In compressed-FFN mode (gpu_ffn_skip_for_compress) or attn-SVD mode
             * (gpu_attn_skip_for_compress), raw tensors were never uploaded / were
             * freed after SVD upload.  Override iswa_ok if all SVD factors are on GPU. */
            if (!iswa_ok && (gpu_ffn_skip_for_compress || gpu_attn_skip_for_compress)) {
                iswa_ok = 1;
                for (int L2 = 0; L2 < m->n_layers && iswa_ok; L2++) {
                    /* FFN: accept SVD factors or skip check if FFN wasn't compressed */
                    if (gpu_ffn_skip_for_compress) {
                        const axex_compressed_weight_t *_cd = axex_get_compressed_layer(L2, 0);
                        const axex_compressed_weight_t *_cg = axex_get_compressed_layer(L2, 6);
                        const axex_compressed_weight_t *_cu = axex_get_compressed_layer(L2, 5);
                        /* Guard all FFN checks: only require compressed slot when the raw
                         * weight pointer is non-NULL.  Also accept raw GPU as fallback (Phase 2d
                         * may have uploaded the raw weight for degenerate SVD layers). */
                        if (m->layers[L2].ffn_down &&
                            !(_cd && _cd->d_Vt) &&
                            !llm_gpu_lookup(m->layers[L2].ffn_down)) {
                            kprintf("[GRAPH-DBG] layer %d: ffn_down not compressed/GPU\n", L2);
                            iswa_ok = 0;
                        }
                        if (iswa_ok && m->layers[L2].ffn_up &&
                            !(_cu && _cu->d_Vt) &&
                            !llm_gpu_lookup(m->layers[L2].ffn_up)) {
                            kprintf("[GRAPH-DBG] layer %d: ffn_up not compressed/GPU\n", L2);
                            iswa_ok = 0;
                        }
                        if (iswa_ok && m->layers[L2].ffn_gate &&
                            !(_cg && _cg->d_Vt) &&
                            !llm_gpu_lookup(m->layers[L2].ffn_gate)) {
                            kprintf("[GRAPH-DBG] layer %d: ffn_gate not compressed/GPU\n", L2);
                            iswa_ok = 0;
                        }
                    }
                    /* Attention Q/O: accept either raw GPU or SVD-compressed */
                    const axex_compressed_weight_t *_cq2 = axex_get_compressed_layer(L2, 1);
                    const axex_compressed_weight_t *_co2 = axex_get_compressed_layer(L2, 4);
                    if (!(_cq2 && _cq2->d_Vt) && !llm_gpu_lookup(m->layers[L2].q_weight)) {
                        kprintf("[GRAPH-DBG] layer %d: q_weight not compressed/GPU\n", L2);
                        iswa_ok = 0;
                    }
                    if (iswa_ok && !(_co2 && _co2->d_Vt) && !llm_gpu_lookup(m->layers[L2].o_weight)) {
                        kprintf("[GRAPH-DBG] layer %d: o_weight not compressed/GPU\n", L2);
                        iswa_ok = 0;
                    }
                    /* K/V: always uploaded if kv_reuse_layer < 0; guard the check
                     * so layers with NULL k/v pointers don't fail it. */
                    if (iswa_ok && m->layers[L2].k_weight &&
                        !llm_gpu_lookup(m->layers[L2].k_weight)) {
                        kprintf("[GRAPH-DBG] layer %d: k_weight not on GPU\n", L2);
                        iswa_ok = 0;
                    }
                    if (iswa_ok && m->layers[L2].v_weight &&
                        !llm_gpu_lookup(m->layers[L2].v_weight)) {
                        kprintf("[GRAPH-DBG] layer %d: v_weight not on GPU\n", L2);
                        iswa_ok = 0;
                    }
                }
            }
            if (iswa_ok) {
                cuda_set_decode_pos(pos, pos + 1);
                if (cuda_graph_begin_capture() == 0) {
                    capturing = 1;
                    if (llm_emit_benchmark_debug())
                        LOG_DBG("GRAPH", "capture started at pos=%d", pos);
                } else {
                    if (llm_emit_benchmark_debug())
                        kprintf("[GRAPH] begin_capture failed at pos=%d\n", pos);
                }
            } else {
                if (llm_emit_benchmark_debug())
                    kprintf("[GRAPH] Layer weights not fully GPU-resident - graph capture skipped (pos=%d)\n", pos);
            }
        } else if (want_logits && !cuda_graph_decode_ready) {
            /* graph not ready — will run layerwise */
        }

/* GPU two-GEMV using compressed weight matrices (SVD: U, S baked into Vt).
 * d_Vt has S baked in, so: tmp[rank] = d_Vt * src, then dst[m] = d_U * tmp.
 * Defined here (before the per-layer loop) so it is visible to Q/O dispatch
 * as well as the later FFN dispatch macros. */
#ifndef GPU_COMPRESSED_GEMV
#define GPU_COMPRESSED_GEMV(_dst, _src, _cw_) do { \
    be->compute.gemv(gpu_ctx.d_compress_tmp, (_cw_)->d_Vt, (_src), \
                     (_cw_)->rank, (_cw_)->n, GGML_TYPE_F32); \
    be->compute.gemv((_dst), (_cw_)->d_U, gpu_ctx.d_compress_tmp, \
                     (_cw_)->m, (_cw_)->rank, GGML_TYPE_F32); \
} while (0)
#endif

        for (int L = 0; L < m->n_layers; L++) {
            llm_layer_t *layer = &m->layers[L];
            gpu_layer_t *gl = &gpu_ctx.layers[L];
            int lhd = layer->head_dim_layer ? layer->head_dim_layer : hd;
            int lkv_dim = n_kv * lhd;
            int lq_dim  = n_heads * lhd;
            int has_own_kv = (layer->kv_reuse_layer < 0);
            uint64_t _lp0, _lp1;

            /* ── Mixed GPU/CPU: run CPU-offloaded layers on the host ── */
            if (!gl->on_gpu) {
                be->mem.sync();
                be->mem.download(llm_x, gpu_ctx.d_x, (uint64_t)dim * sizeof(float));
                llm_cpu_layer(m, L, pos);
                be->mem.upload(gpu_ctx.d_x, llm_x, (uint64_t)dim * sizeof(float));
                continue;
            }
            int detail_layer = (prof_detail && L == 10);  /* profile layer 10 */

            /* 2a. Pre-attention RMSNorm */
            if (detail_layer) { be->mem.sync(); _lp0 = hal_timer_us(); }

            /* Per-layer GP: select layer-specific Pt basis if available */
            if (axex_manifold_has_layer_bases())
                axex_manifold_select_layer(L);

            /* 2b. Q/K/V projections on GPU */
            ggml_type_t q_gpu_type = layer->q_type;
            ggml_type_t o_gpu_type = layer->o_type;
            ggml_type_t gate_gpu_type = layer->gate_type;
            ggml_type_t up_gpu_type   = layer->up_type;
            ggml_type_t down_gpu_type = layer->down_type;
            void *d_qw = llm_gpu_lookup_with_type(layer->q_weight, &q_gpu_type);
            /* Pre-fetch GPU types for o/gate/up/down (handles IQ2_XS → F32 promotion) */
            if (layer->o_weight)   llm_gpu_lookup_with_type(layer->o_weight,   &o_gpu_type);
            if (layer->ffn_gate)   llm_gpu_lookup_with_type(layer->ffn_gate,   &gate_gpu_type);
            if (layer->ffn_up)     llm_gpu_lookup_with_type(layer->ffn_up,     &up_gpu_type);
            if (layer->ffn_down)   llm_gpu_lookup_with_type(layer->ffn_down,   &down_gpu_type);

            /* If core attention weights aren't GPU-resident, skip — BUT allow the
             * SVD-compressed path (slot 1) since it replaces the raw q_weight. */
            if (layer->q_weight && !d_qw) {
                const axex_compressed_weight_t *_cq_chk = axex_get_compressed_layer(L, 1);
                const axex_manifold_weight_t   *_gp_chk = axex_get_manifold_layer(L, 1);
                const float *_gp_dPt_chk = axex_manifold_d_Pt();
                if (!(_cq_chk && _cq_chk->d_Vt) &&
                    !(_gp_chk && _gp_chk->d_W_proj && _gp_dPt_chk)) {
                    /* No raw, no SVD Q, and no GP manifold Q — layer is broken */
                    kprintf("[GPU] WARNING: q_weight not on GPU for layer %d, skipping\n", L);
                    continue;
                }
                /* SVD-compressed Q or GP manifold Q present — raw freed intentionally */
            }

            /* Geodesic Projection (GP) state for this layer */
            const axex_manifold_weight_t *_gp_q  = axex_get_manifold_layer(L, 1);
            const float    *_gp_dPt     = axex_manifold_d_Pt();
            const uint16_t *_gp_dPt_f16 = axex_manifold_d_Pt_f16();
            /* Prefer Q4_0 Pt (2.8x less bandwidth vs F16); fall back to F16 then F32 */
            const void     *_gp_dPt_q4  = axex_manifold_d_Pt_q4();
            int _disable_pt_q4 = llm_disable_pt_q4();
            const void *_gp_Pt_ptr  = (!_disable_pt_q4 && _gp_dPt_q4) ? _gp_dPt_q4
                                    : _gp_dPt_f16  ? (const void *)_gp_dPt_f16
                                                   : (const void *)_gp_dPt;
            ggml_type_t _gp_Pt_type = (!_disable_pt_q4 && _gp_dPt_q4) ? GGML_TYPE_Q4_0
                                    : _gp_dPt_f16  ? GGML_TYPE_F16 : GGML_TYPE_F32;
            int  _gp_k  = axex_manifold_k();
            int  _gp_n  = axex_manifold_n();
            int gpu_x_sub_qkv_ready = 0;
            /* SVD-compressed Q and O (slots 1, 4) — takes priority over GP path */
            const axex_compressed_weight_t *_cq_ = axex_get_compressed_layer(L, 1);
            const axex_compressed_weight_t *_co_ = axex_get_compressed_layer(L, 4);

            if (has_own_kv) {
                ggml_type_t k_gpu_type = layer->k_type;
                ggml_type_t v_gpu_type = layer->v_type;
                void *d_kw = llm_gpu_lookup_with_type(layer->k_weight, &k_gpu_type);
                void *d_vw = llm_gpu_lookup_with_type(layer->v_weight, &v_gpu_type);
                /* GP fast path: d_xn = rmsnorm(d_x), then x_sub = Pt@d_xn once */
                const axex_manifold_weight_t *_gp_k2 = axex_get_manifold_layer(L, 2);
                const axex_manifold_weight_t *_gp_v2 = axex_get_manifold_layer(L, 3);
                if (_cq_ && _cq_->d_Vt) {
                    /* SVD two-step: tmp=SVt@xn, Q=U@tmp — no PCA projection needed */
                    be->compute.rmsnorm(gpu_ctx.d_xn, gpu_ctx.d_x, gl->d_attn_norm,
                                        dim, gpu_ctx.rms_eps);
                    GPU_COMPRESSED_GEMV(gpu_ctx.d_q, gpu_ctx.d_xn, _cq_);
                    be->compute.gemv(gpu_ctx.d_k, d_kw, gpu_ctx.d_xn, lkv_dim, dim, k_gpu_type);
                    be->compute.gemv(gpu_ctx.d_v, d_vw, gpu_ctx.d_xn, lkv_dim, dim, v_gpu_type);
                } else if (_gp_q && _gp_q->d_W_proj && _gp_dPt && _gp_k > 0) {
                    be->compute.rmsnorm(gpu_ctx.d_xn, gpu_ctx.d_x, gl->d_attn_norm,
                                        dim, gpu_ctx.rms_eps);
                    /* Project: d_x_sub[k] = Pt[k×n] @ d_xn[n] (F16 Pt → 2× less bandwidth) */
                    be->compute.gemv(gpu_ctx.d_x_sub, _gp_Pt_ptr, gpu_ctx.d_xn,
                                     _gp_k, _gp_n, _gp_Pt_type);
                    gpu_x_sub_qkv_ready = 1;
                    be->compute.gemv(gpu_ctx.d_q, (void *)_gp_q->d_W_proj, gpu_ctx.d_x_sub,
                                     lq_dim, _gp_k, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32);
                    if (_gp_q->d_bias)
                        be->compute.add(gpu_ctx.d_q, gpu_ctx.d_q, _gp_q->d_bias, lq_dim);
                    if (_gp_k2 && _gp_k2->d_W_proj) {
                        int kv_fused = 0;
                        if (_gp_v2 && _gp_v2->d_W_proj && axex_manifold_wp_is_q8()) {
                            kv_fused = cuda_gemv_dual_q8_0(
                                gpu_ctx.d_k, gpu_ctx.d_v,
                                (void *)_gp_k2->d_W_proj, (void *)_gp_v2->d_W_proj,
                                gpu_ctx.d_x_sub, lkv_dim, _gp_k);
                            if (kv_fused && !g_logged_dual_q8_kv) {
                                kprintf("[GP] fused dual-Q8 K/V GEMV active (layer=%d, k=%d)\n", L, _gp_k);
                                g_logged_dual_q8_kv = 1;
                            }
                        }
                        if (!kv_fused) {
                            be->compute.gemv(gpu_ctx.d_k, (void *)_gp_k2->d_W_proj, gpu_ctx.d_x_sub,
                                             lkv_dim, _gp_k, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32);
                            if (_gp_v2 && _gp_v2->d_W_proj) {
                                be->compute.gemv(gpu_ctx.d_v, (void *)_gp_v2->d_W_proj, gpu_ctx.d_x_sub,
                                                 lkv_dim, _gp_k, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32);
                            } else {
                                be->compute.gemv(gpu_ctx.d_v, d_vw, gpu_ctx.d_xn, lkv_dim, dim, v_gpu_type);
                            }
                        }
                        if (_gp_k2->d_bias)
                            be->compute.add(gpu_ctx.d_k, gpu_ctx.d_k, _gp_k2->d_bias, lkv_dim);
                        if (_gp_v2 && _gp_v2->d_bias)
                            be->compute.add(gpu_ctx.d_v, gpu_ctx.d_v, _gp_v2->d_bias, lkv_dim);
                    } else
                        be->compute.gemv(gpu_ctx.d_k, d_kw, gpu_ctx.d_xn, lkv_dim, dim, k_gpu_type);
                    if (!(_gp_k2 && _gp_k2->d_W_proj)) {
                        if (_gp_v2 && _gp_v2->d_W_proj) {
                            be->compute.gemv(gpu_ctx.d_v, (void *)_gp_v2->d_W_proj, gpu_ctx.d_x_sub,
                                             lkv_dim, _gp_k, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32);
                            if (_gp_v2->d_bias)
                                be->compute.add(gpu_ctx.d_v, gpu_ctx.d_v, _gp_v2->d_bias, lkv_dim);
                        } else {
                            be->compute.gemv(gpu_ctx.d_v, d_vw, gpu_ctx.d_xn, lkv_dim, dim, v_gpu_type);
                        }
                    }
                } else {
                    /* Fast path: fused RMSNorm + triple GEMV eliminates d_xn write */
                    int used_triple = 0;
                    if (q_gpu_type == GGML_TYPE_Q4_0 &&
                        k_gpu_type == GGML_TYPE_Q4_0 &&
                        v_gpu_type == GGML_TYPE_Q4_0 &&
                        gl->d_attn_norm) {
                        used_triple = cuda_fused_rmsnorm_triple_q4_0(
                            gpu_ctx.d_q, gpu_ctx.d_k, gpu_ctx.d_v,
                            d_qw, d_kw, d_vw,
                            gpu_ctx.d_x, gl->d_attn_norm, gpu_ctx.rms_eps,
                            lq_dim, lkv_dim, lkv_dim, dim);
                    }
                    /* Fallback: separate rmsnorm + regular triple or individual GEMVs */
                    if (!used_triple) {
                        be->compute.rmsnorm(gpu_ctx.d_xn, gpu_ctx.d_x, gl->d_attn_norm,
                                            dim, gpu_ctx.rms_eps);
                        int triple_ok = 0;
                        if (q_gpu_type == GGML_TYPE_Q4_0 &&
                            k_gpu_type == GGML_TYPE_Q4_0 &&
                            v_gpu_type == GGML_TYPE_Q4_0) {
                            triple_ok = cuda_gemv_triple_q4_0(
                                gpu_ctx.d_q, gpu_ctx.d_k, gpu_ctx.d_v,
                                d_qw, d_kw, d_vw, gpu_ctx.d_xn,
                                lq_dim, lkv_dim, lkv_dim, dim);
                        }
                        if (!triple_ok) {
                            be->compute.gemv(gpu_ctx.d_q, d_qw, gpu_ctx.d_xn, lq_dim, dim, q_gpu_type);
                            be->compute.gemv(gpu_ctx.d_k, d_kw, gpu_ctx.d_xn, lkv_dim, dim, k_gpu_type);
                            be->compute.gemv(gpu_ctx.d_v, d_vw, gpu_ctx.d_xn, lkv_dim, dim, v_gpu_type);
                        }
                    }
                }
            } else {
                be->compute.rmsnorm(gpu_ctx.d_xn, gpu_ctx.d_x, gl->d_attn_norm,
                                    dim, gpu_ctx.rms_eps);
                if (_cq_ && _cq_->d_Vt) {
                    GPU_COMPRESSED_GEMV(gpu_ctx.d_q, gpu_ctx.d_xn, _cq_);
                } else if (_gp_q && _gp_q->d_W_proj && _gp_dPt && _gp_k > 0) {
                    be->compute.gemv(gpu_ctx.d_x_sub, _gp_Pt_ptr, gpu_ctx.d_xn,
                                     _gp_k, _gp_n, _gp_Pt_type);
                    gpu_x_sub_qkv_ready = 1;
                    be->compute.gemv(gpu_ctx.d_q, (void *)_gp_q->d_W_proj, gpu_ctx.d_x_sub,
                                     lq_dim, _gp_k, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32);
                    if (_gp_q->d_bias)
                        be->compute.add(gpu_ctx.d_q, gpu_ctx.d_q, _gp_q->d_bias, lq_dim);
                } else {
                    be->compute.gemv(gpu_ctx.d_q, d_qw, gpu_ctx.d_xn, lq_dim, dim, q_gpu_type);
                }
            }
            if (detail_layer) { be->mem.sync(); _lp1 = hal_timer_us(); t_l_qkv += (_lp1 - _lp0); _lp0 = _lp1; }

            /* Per-head Q/K normalization + RoPE: fully GPU-resident */
            float layer_rope_base = m->rope_base;
            if (m->is_gemma4 && lhd == m->head_dim_swa)
                layer_rope_base = m->rope_base_swa;
            const float *rope_f_host = (const float *)0;
            if (m->rope_factors_short || m->rope_factors_long)
                rope_f_host = (pos < m->rope_orig_ctx) ? m->rope_factors_short : m->rope_factors_long;
            if (m->is_gemma4 && m->rope_freqs && lhd != m->head_dim_swa)
                rope_f_host = m->rope_freqs;
            int rdim = m->rope_dim > 0 ? m->rope_dim : lhd;

            /* GPU-resident: norm + RoPE via fused kernel */
            if (layer->q_norm && has_own_kv) {
                float *d_rope_f = (rope_f_host == m->rope_freqs) ? gpu_ctx.d_rope_freqs : (float *)0;
                cuda_fused_qk_norm_rope(gpu_ctx.d_q, gpu_ctx.d_k,
                    gl->d_q_norm, gl->d_k_norm,
                    n_heads, n_kv, lhd,
                    pos, layer_rope_base, d_rope_f,
                    m->rms_eps, rdim);
                cuda_v_norm(gpu_ctx.d_v, n_kv, lhd, m->rms_eps);
            } else if (layer->q_norm) {
                /* No own KV but has norm — norm Q only, RoPE Q only */
                float *d_rope_f = (rope_f_host == m->rope_freqs) ? gpu_ctx.d_rope_freqs : (float *)0;
                cuda_fused_qk_norm_rope(gpu_ctx.d_q, gpu_ctx.d_q, /* K unused */
                    gl->d_q_norm, (float *)0,
                    n_heads, 0, lhd,
                    pos, layer_rope_base, d_rope_f,
                    m->rms_eps, rdim);
            } else {
                /* No norm — just RoPE (non-Gemma4 models) */
                float *d_rope_f = (rope_f_host == m->rope_freqs) ? gpu_ctx.d_rope_freqs : (float *)0;
                if (has_own_kv) {
                    be->compute.rope(gpu_ctx.d_q, gpu_ctx.d_k, lhd, n_heads, n_kv,
                                     pos, layer_rope_base, d_rope_f);
                } else {
                    be->compute.rope(gpu_ctx.d_q, gpu_ctx.d_q, lhd, n_heads, 0,
                                     pos, layer_rope_base, d_rope_f);
                }
            }

            /* KV cache update on GPU */
            if (has_own_kv) {
                be->compute.kv_update(gl->d_k_cache, gl->d_v_cache,
                                      gpu_ctx.d_k, gpu_ctx.d_v,
                                      n_kv, lhd, pos, max_seq, L);
            }
            if (detail_layer) { be->mem.sync(); _lp1 = hal_timer_us(); t_l_rope += (_lp1 - _lp0); _lp0 = _lp1; }

            /* 2e. GPU-resident attention */
            {
                int kv_src_layer = has_own_kv ? L : layer->kv_reuse_layer;
                gpu_layer_t *kv_gl = &gpu_ctx.layers[kv_src_layer];
                float attn_scale = m->is_gemma4 ? 1.0f : (1.0f / llm_sqrtf((float)lhd));
                be->compute.attention(gpu_ctx.d_attn, gpu_ctx.d_q,
                                      kv_gl->d_k_cache, kv_gl->d_v_cache,
                                      n_heads, n_kv, lhd,
                                      pos + 1, max_seq, attn_scale, 0.0f);
            }

            if (detail_layer) { be->mem.sync(); _lp1 = hal_timer_us(); t_l_attn += (_lp1 - _lp0); _lp0 = _lp1; }
            /* 2f. O projection + post-attn norm + residual (GPU) */
            /* Priority: SVD compressed O (slot 4) > GP manifold > raw GPU weight */
            {
                const axex_manifold_weight_t *_gp_o = axex_get_manifold_layer(L, 4);
                const uint16_t *_gp_dPt_o_f16 = axex_manifold_d_Pt_f16();
                const float    *_gp_dPt_o     = axex_manifold_d_Pt();
                const void     *_gp_dPt_o_q4  = axex_manifold_d_Pt_q4();
                const void     *_gp_o_Pt_ptr  = _gp_dPt_o_q4  ? _gp_dPt_o_q4
                                              : _gp_dPt_o_f16  ? (const void *)_gp_dPt_o_f16
                                                                : (const void *)_gp_dPt_o;
                ggml_type_t _gp_o_Pt_type = _gp_dPt_o_q4  ? GGML_TYPE_Q4_0
                                          : _gp_dPt_o_f16  ? GGML_TYPE_F16 : GGML_TYPE_F32;
                int _gp_k_o = axex_manifold_k();
                int _gp_n_o = axex_manifold_n();
                if (_co_ && _co_->d_Vt) {
                    /* SVD two-step O projection: tmp=SVt@attn, out=U@tmp */
                    GPU_COMPRESSED_GEMV(gpu_ctx.d_ffn_d, gpu_ctx.d_attn, _co_);
                } else if (_gp_o && _gp_o->d_W_proj && _gp_dPt_o && _gp_k_o > 0 &&
                    lq_dim == _gp_n_o) {
                    /* x_sub_o[k] = Pt[k×n] @ d_attn[n] (F16 Pt → 2× less bandwidth) */
                    be->compute.gemv(gpu_ctx.d_x_sub, _gp_o_Pt_ptr,
                                     gpu_ctx.d_attn, _gp_k_o, lq_dim, _gp_o_Pt_type);
                    be->compute.gemv(gpu_ctx.d_ffn_d, (void *)_gp_o->d_W_proj,
                                     gpu_ctx.d_x_sub, dim, _gp_k_o, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32);
                    if (_gp_o->d_bias)
                        be->compute.add(gpu_ctx.d_ffn_d, gpu_ctx.d_ffn_d, _gp_o->d_bias, dim);
                } else {
                    void *d_ow = llm_gpu_lookup(layer->o_weight);
                    be->compute.gemv(gpu_ctx.d_ffn_d, d_ow, gpu_ctx.d_attn, dim, lq_dim, o_gpu_type);
                }
            }

            if (layer->post_attn_norm && gl->d_post_attn_norm) {
                /* Fused: rmsnorm(d_ffn_d) + add to d_x */
                cuda_rmsnorm_add(gpu_ctx.d_x, gpu_ctx.d_ffn_d,
                                  gl->d_post_attn_norm, dim, gpu_ctx.rms_eps);
            } else {
                be->compute.add(gpu_ctx.d_x, gpu_ctx.d_x, gpu_ctx.d_ffn_d, dim);
            }
            if (llm_attnres_enabled) {
                float depth_frac = (float)(L + 1) / (float)(m->n_layers > 0 ? m->n_layers : 1);
                float gamma = 1.0f - (0.12f * llm_attnres_strength * depth_frac);
                if (gamma < 0.70f) gamma = 0.70f;
                be->compute.scale(gpu_ctx.d_x, gpu_ctx.d_x, gamma, dim);
            }

            if (detail_layer) { be->mem.sync(); _lp1 = hal_timer_us(); t_l_oproj += (_lp1 - _lp0); _lp0 = _lp1; }
            /* === FFN === */

            /* 2g. Fused add + Pre-FFN RMSNorm (when post-attn norm was fused above) */
            be->compute.rmsnorm(gpu_ctx.d_xn, gpu_ctx.d_x, gl->d_ffn_norm,
                                dim, gpu_ctx.rms_eps);

            int lff = layer->ff_dim_layer ? layer->ff_dim_layer : ff;

            /* FFN: always GPU (CPU fallback removed for perf) */
            /* Helper macro: if this layer has a compressed ffn_down, do the
             * double-GEMV on CPU (U/S/Vt are in RAM, not VRAM) and DMA the
             * result back.  The compressed matrices are ~34% the size of the
             * original quantized weight, so they were never uploaded to GPU —
             * this saves VRAM at the cost of one PCIe round-trip per token. */
/* GPU_COMPRESSED_GEMV defined earlier (before per-layer loop). */

#define GPU_FFN_DOWN_GEMV(_dst, _src, _dim, _lff) do { \
    const axex_manifold_weight_t *_md7_ = axex_get_manifold_layer(L, 7); \
    const float *_d_Qff_ = axex_manifold_d_Qff(L); \
    int _k_ff_dn_ = axex_manifold_qff_k(L); \
    if (_md7_ && _md7_->d_W_proj && _d_Qff_ && _k_ff_dn_ > 0) { \
        /* FFN down GP: x_sub[k_ff] = Qff[k_ff×lff] @ ffn_g[lff], then \
         * ffn_d[dim] = W_proj_down[dim×k_ff] @ x_sub[k_ff]. \
         * d_x_sub is safe to reuse here: gate/up are already computed. */ \
        be->compute.gemv(gpu_ctx.d_x_sub, (void *)_d_Qff_, (_src), _k_ff_dn_, (_lff), GGML_TYPE_F32); \
        be->compute.gemv((_dst), (void *)_md7_->d_W_proj, gpu_ctx.d_x_sub, (_dim), _k_ff_dn_, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32); \
    } else { \
    const axex_compressed_weight_t *_cw_ = axex_get_compressed_layer(L, 0); \
    if (_cw_ && _cw_->d_Vt) { \
        /* True GPU two-GEMV — no PCIe round-trip */ \
        GPU_COMPRESSED_GEMV((_dst), (_src), _cw_); \
    } else if (_cw_) { \
        /* Compressed but not on GPU yet — PCIe fallback */ \
        be->mem.download(llm_ffn_u, (_src), (_lff) * sizeof(float)); \
        axex_compressed_weight_gemv(_cw_, llm_ffn_u, llm_ffn_d); \
        be->mem.upload((_dst), llm_ffn_d, (_dim) * sizeof(float)); \
    } else { \
        void *_dw_ = llm_gpu_lookup(layer->ffn_down); \
        if (_dw_) { \
            be->compute.gemv((_dst), _dw_, (_src), (_dim), (_lff), down_gpu_type); \
        } else { \
            /* FFN weight not on GPU (VRAM OOM or compress mode): PCIe D2H→CPU GEMV→H2D */ \
            be->mem.download(llm_ffn_u, (_src), (uint64_t)(_lff) * sizeof(float)); \
            llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_u, (_dim), (_lff), layer->down_type); \
            be->mem.upload((_dst), llm_ffn_d, (uint64_t)(_dim) * sizeof(float)); \
        } \
    } \
    } \
} while (0)
/* Macros for compressed gate and up GEMVs (slots 6 and 5).
 * Priority: GP manifold (d_W_proj) > SVD compressed (d_Vt) > regular GPU GEMV.
 * GP path uses _gpu_x_sub which must already hold Pt@d_xn. */
#define GPU_FFN_GATE_GEMV(_dst, _src, _lff, _dim) do { \
    const axex_manifold_weight_t *_mg_ = axex_get_manifold_layer(L, 6); \
    if (_mg_ && _mg_->d_W_proj && _gpu_x_sub_ffn_ready) { \
        be->compute.gemv((_dst), (void *)_mg_->d_W_proj, gpu_ctx.d_x_sub, (_lff), _gp_k_ffn, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32); \
    } else { \
        const axex_compressed_weight_t *_cw_g_ = axex_get_compressed_layer(L, 6); \
        if (_cw_g_ && _cw_g_->d_Vt) { \
            GPU_COMPRESSED_GEMV((_dst), (_src), _cw_g_); \
        } else { \
            void *_gw_ = llm_gpu_lookup(layer->ffn_gate); \
            if (_gw_) { \
                be->compute.gemv((_dst), _gw_, (_src), (_lff), (_dim), gate_gpu_type); \
            } else { \
                /* FFN weight not on GPU (VRAM OOM or compress mode): PCIe D2H→CPU GEMV→H2D */ \
                be->mem.download(llm_xn, (_src), (uint64_t)(_dim) * sizeof(float)); \
                llm_gemv(llm_ffn_g, layer->ffn_gate, llm_xn, (_lff), (_dim), layer->gate_type); \
                be->mem.upload((_dst), llm_ffn_g, (uint64_t)(_lff) * sizeof(float)); \
            } \
        } \
    } \
} while (0)

#define GPU_FFN_UP_GEMV(_dst, _src, _lff, _dim) do { \
    const axex_manifold_weight_t *_mu_ = axex_get_manifold_layer(L, 5); \
    if (_mu_ && _mu_->d_W_proj && _gpu_x_sub_ffn_ready) { \
        be->compute.gemv((_dst), (void *)_mu_->d_W_proj, gpu_ctx.d_x_sub, (_lff), _gp_k_ffn, axex_manifold_wp_is_q8() ? GGML_TYPE_Q8_0 : GGML_TYPE_F32); \
    } else { \
        const axex_compressed_weight_t *_cw_u_ = axex_get_compressed_layer(L, 5); \
        if (_cw_u_ && _cw_u_->d_Vt) { \
            GPU_COMPRESSED_GEMV((_dst), (_src), _cw_u_); \
        } else { \
            void *_uw_ = llm_gpu_lookup(layer->ffn_up); \
            if (_uw_) { \
                be->compute.gemv((_dst), _uw_, (_src), (_lff), (_dim), up_gpu_type); \
            } else { \
                /* FFN weight not on GPU (VRAM OOM or compress mode): PCIe D2H→CPU GEMV→H2D */ \
                be->mem.download(llm_xn, (_src), (uint64_t)(_dim) * sizeof(float)); \
                llm_gemv(llm_ffn_u, layer->ffn_up, llm_xn, (_lff), (_dim), layer->up_type); \
                be->mem.upload((_dst), llm_ffn_u, (uint64_t)(_lff) * sizeof(float)); \
            } \
        } \
    } \
} while (0)

            /* GP: project FFN-normed d_xn → d_x_sub once, shared by gate and up */
            int _gpu_x_sub_ffn_ready = 0;
            int _gp_k_ffn = axex_manifold_k();
            {
                const axex_manifold_weight_t *_mg_gate = axex_get_manifold_layer(L, 6);
                const uint16_t *_gp_dPt_ffn_f16 = axex_manifold_d_Pt_f16();
                const float    *_gp_dPt_ffn     = axex_manifold_d_Pt();
                const void     *_gp_dPt_ffn_q4  = axex_manifold_d_Pt_q4();
                const void     *_gp_ffn_Pt_ptr  = _gp_dPt_ffn_q4  ? _gp_dPt_ffn_q4
                                                : _gp_dPt_ffn_f16  ? (const void *)_gp_dPt_ffn_f16
                                                                    : (const void *)_gp_dPt_ffn;
                ggml_type_t _gp_ffn_Pt_type = _gp_dPt_ffn_q4  ? GGML_TYPE_Q4_0
                                            : _gp_dPt_ffn_f16  ? GGML_TYPE_F16 : GGML_TYPE_F32;
                if (_mg_gate && _mg_gate->d_W_proj && _gp_dPt_ffn && _gp_k_ffn > 0 &&
                    layer->ffn_gate) {
                    /* d_xn is already set (FFN RMSNorm was applied just above) */
                    be->compute.gemv(gpu_ctx.d_x_sub, _gp_ffn_Pt_ptr, gpu_ctx.d_xn,
                                     _gp_k_ffn, axex_manifold_n(), _gp_ffn_Pt_type);
                    _gpu_x_sub_ffn_ready = 1;
                }
            }

            if (m->use_gelu || !layer->ffn_gate) {
                /* GELU FFN — GP does not apply (no gate slot) */
                GPU_FFN_UP_GEMV(gpu_ctx.d_ffn_u, gpu_ctx.d_xn, lff, dim);
                be->compute.gelu(gpu_ctx.d_ffn_u, lff);
                GPU_FFN_DOWN_GEMV(gpu_ctx.d_ffn_d, gpu_ctx.d_ffn_u, dim, lff);
            } else if (m->use_geglu) {
                /* GeGLU: GELU(gate) ⊙ up */
                /* Skip fused dual Q4_0 path when GP is active */
                const axex_compressed_weight_t *_cw_g6 = axex_get_compressed_layer(L, 6);
                const axex_compressed_weight_t *_cw_u5 = axex_get_compressed_layer(L, 5);
                int used_dual = 0;
                if (!_gpu_x_sub_ffn_ready &&
                    !(_cw_g6 && _cw_g6->d_Vt) && !(_cw_u5 && _cw_u5->d_Vt) &&
                    layer->gate_type == GGML_TYPE_Q4_0 &&
                    layer->up_type == GGML_TYPE_Q4_0) {
                    void *d_gatew = llm_gpu_lookup(layer->ffn_gate);
                    void *d_upw   = llm_gpu_lookup(layer->ffn_up);
                    used_dual = cuda_gemv_dual_q4_0(
                        gpu_ctx.d_ffn_g, gpu_ctx.d_ffn_u,
                        d_gatew, d_upw, gpu_ctx.d_xn, lff, dim);
                }
                if (!used_dual) {
                    int used_batched = 0;
                    if (!_gpu_x_sub_ffn_ready && _cw_g6 && _cw_u5 &&
                        _cw_g6->d_Aarray_step1 && cuda_have_sgemm_batched_f32()) {
                        cuda_sgemm_batched_f32(_cw_g6->rank, _cw_g6->n,
                            (const float * const *)_cw_g6->d_Aarray_step1,
                            (const float * const *)_cw_g6->d_xarray_step1,
                            (float * const *)_cw_g6->d_yarray_step1,
                            2);
                        cuda_sgemm_batched_f32(_cw_g6->m, _cw_g6->rank,
                            (const float * const *)_cw_g6->d_Aarray_step2,
                            (const float * const *)_cw_g6->d_xarray_step2,
                            (float * const *)_cw_g6->d_yarray_step2,
                            2);
                        used_batched = 1;
                    }
                    if (!used_batched) {
                        GPU_FFN_GATE_GEMV(gpu_ctx.d_ffn_g, gpu_ctx.d_xn, lff, dim);
                        GPU_FFN_UP_GEMV(gpu_ctx.d_ffn_u,   gpu_ctx.d_xn, lff, dim);
                    }
                }
                cuda_fused_geglu(gpu_ctx.d_ffn_g, gpu_ctx.d_ffn_u, lff);
                GPU_FFN_DOWN_GEMV(gpu_ctx.d_ffn_d, gpu_ctx.d_ffn_g, dim, lff);
            } else {
                /* SwiGLU: SiLU(gate) ⊙ up — skip fused dual path when GP is active */
                const axex_compressed_weight_t *_cw_g6 = axex_get_compressed_layer(L, 6);
                const axex_compressed_weight_t *_cw_u5 = axex_get_compressed_layer(L, 5);
                int used_dual = 0;
                if (!_gpu_x_sub_ffn_ready &&
                    !(_cw_g6 && _cw_g6->d_Vt) && !(_cw_u5 && _cw_u5->d_Vt) &&
                    layer->gate_type == GGML_TYPE_Q4_0 &&
                    layer->up_type == GGML_TYPE_Q4_0) {
                    void *d_gatew = llm_gpu_lookup(layer->ffn_gate);
                    void *d_upw   = llm_gpu_lookup(layer->ffn_up);
                    used_dual = cuda_gemv_dual_q4_0(
                        gpu_ctx.d_ffn_g, gpu_ctx.d_ffn_u,
                        d_gatew, d_upw, gpu_ctx.d_xn, lff, dim);
                }
                if (!used_dual) {
                    int used_batched = 0;
                    if (!_gpu_x_sub_ffn_ready && _cw_g6 && _cw_u5 &&
                        _cw_g6->d_Aarray_step1 && cuda_have_sgemm_batched_f32()) {
                        cuda_sgemm_batched_f32(_cw_g6->rank, _cw_g6->n,
                            (const float * const *)_cw_g6->d_Aarray_step1,
                            (const float * const *)_cw_g6->d_xarray_step1,
                            (float * const *)_cw_g6->d_yarray_step1,
                            2);
                        cuda_sgemm_batched_f32(_cw_g6->m, _cw_g6->rank,
                            (const float * const *)_cw_g6->d_Aarray_step2,
                            (const float * const *)_cw_g6->d_xarray_step2,
                            (float * const *)_cw_g6->d_yarray_step2,
                            2);
                        used_batched = 1;
                    }
                    if (!used_batched) {
                        GPU_FFN_GATE_GEMV(gpu_ctx.d_ffn_g, gpu_ctx.d_xn, lff, dim);
                        GPU_FFN_UP_GEMV(gpu_ctx.d_ffn_u,   gpu_ctx.d_xn, lff, dim);
                    }
                }
                cuda_fused_swiglu(gpu_ctx.d_ffn_g, gpu_ctx.d_ffn_u, lff);
                GPU_FFN_DOWN_GEMV(gpu_ctx.d_ffn_d, gpu_ctx.d_ffn_g, dim, lff);
            }
#undef GPU_FFN_GATE_GEMV
#undef GPU_FFN_UP_GEMV
#undef GPU_FFN_DOWN_GEMV
#undef GPU_COMPRESSED_GEMV

            /* Post-FFW norm (Gemma4) + Residual — fused when possible */
            if (layer->post_ffw_norm && gl->d_post_ffw_norm) {
                cuda_rmsnorm_add(gpu_ctx.d_x, gpu_ctx.d_ffn_d,
                                  gl->d_post_ffw_norm, dim, gpu_ctx.rms_eps);
            } else {
                be->compute.add(gpu_ctx.d_x, gpu_ctx.d_x, gpu_ctx.d_ffn_d, dim);
            }
            if (llm_attnres_enabled) {
                float depth_frac = (float)(L + 1) / (float)(m->n_layers > 0 ? m->n_layers : 1);
                float gamma = 1.0f - (0.18f * llm_attnres_strength * depth_frac);
                if (gamma < 0.65f) gamma = 0.65f;
                be->compute.scale(gpu_ctx.d_x, gpu_ctx.d_x, gamma, dim);
            }

            if (llm_depth_attn_enabled && llm_depth_hist && llm_depth_scores && L > 0) {
                const backend_t *be2 = backend_get_by_id(BACKEND_CUDA);
                be2->mem.download(llm_x, gpu_ctx.d_x, (uint64_t)dim * sizeof(float));
                be2->mem.sync();
                kmemcpy(llm_depth_hist + (uint64_t)L * dim, llm_x, (uint64_t)dim * sizeof(float));

                int start = L - llm_depth_attn_window;
                if (start < 0) start = 0;
                int cnt = 0;
                float qn = llm_sqrtf(llm_dot_f32(llm_x, llm_x, dim) + 1e-9f);
                for (int p = start; p < L; p++) {
                    float *hp = llm_depth_hist + (uint64_t)p * dim;
                    float hn = llm_sqrtf(llm_dot_f32(hp, hp, dim) + 1e-9f);
                    float cs = llm_dot_f32(llm_x, hp, dim) / (qn * hn + 1e-9f);
                    llm_depth_scores[cnt++] = cs;
                }
                float mx = llm_depth_scores[0];
                for (int i = 1; i < cnt; i++) if (llm_depth_scores[i] > mx) mx = llm_depth_scores[i];
                float sm = 0.0f;
                for (int i = 0; i < cnt; i++) {
                    llm_depth_scores[i] = llm_expf((llm_depth_scores[i] - mx) * 6.0f);
                    sm += llm_depth_scores[i];
                }
                if (sm < 1e-9f) sm = 1e-9f;
                float inv = 1.0f / sm;
                kmemset(llm_ffn_d, 0, (uint64_t)dim * sizeof(float));
                for (int i = 0; i < cnt; i++) {
                    int p = start + i;
                    float w = llm_depth_scores[i] * inv;
                    llm_axpy_f32(llm_ffn_d, w, llm_depth_hist + (uint64_t)p * dim, dim);
                }
                float depth_frac = (float)(L + 1) / (float)(m->n_layers > 0 ? m->n_layers : 1);
                float alpha = llm_depth_attn_strength * depth_frac;
                if (alpha > 0.85f) alpha = 0.85f;
                be2->mem.upload(gpu_ctx.d_ffn_d, llm_ffn_d, (uint64_t)dim * sizeof(float));
                be2->compute.scale(gpu_ctx.d_x, gpu_ctx.d_x, 1.0f - alpha, dim);
                be2->compute.scale(gpu_ctx.d_ffn_d, gpu_ctx.d_ffn_d, alpha, dim);
                be2->compute.add(gpu_ctx.d_x, gpu_ctx.d_x, gpu_ctx.d_ffn_d, dim);
            } else if (llm_depth_attn_enabled && llm_depth_hist && L == 0) {
                const backend_t *be2 = backend_get_by_id(BACKEND_CUDA);
                be2->mem.download(llm_x, gpu_ctx.d_x, (uint64_t)dim * sizeof(float));
                be2->mem.sync();
                kmemcpy(llm_depth_hist + (uint64_t)L * dim, llm_x, (uint64_t)dim * sizeof(float));
            }
            if (detail_layer) { be->mem.sync(); _lp1 = hal_timer_us(); t_l_ffn += (_lp1 - _lp0); _lp0 = _lp1; }

            /* === ISWA injection (Gemma4): GPU-resident === */
            if (m->is_gemma4 && layer->iswa_inp_gate) {
                int iswa_d = m->iswa_n_embd;
                void *d_inp_gate = llm_gpu_lookup(layer->iswa_inp_gate);
                void *d_proj     = llm_gpu_lookup(layer->iswa_proj);

                if (d_inp_gate && d_proj && gpu_ctx.d_iswa_tmp
                    && gpu_ctx.d_iswa_per_layer) {
                    /* All on GPU — gelu+mul fused, norm+add fused */
                    be->compute.gemv(gpu_ctx.d_iswa_tmp, d_inp_gate, gpu_ctx.d_x,
                                     iswa_d, dim, layer->iswa_inp_gate_type);
                    cuda_gelu_mul(gpu_ctx.d_iswa_tmp,
                                  gpu_ctx.d_iswa_per_layer + L * iswa_d, iswa_d);
                    be->compute.gemv(gpu_ctx.d_ffn_d, d_proj, gpu_ctx.d_iswa_tmp,
                                     dim, iswa_d, layer->iswa_proj_type);
                    if (gl->d_iswa_post_norm)
                        cuda_rmsnorm_add(gpu_ctx.d_x, gpu_ctx.d_ffn_d,
                                          gl->d_iswa_post_norm, dim, gpu_ctx.rms_eps);
                    else
                        be->compute.add(gpu_ctx.d_x, gpu_ctx.d_x, gpu_ctx.d_ffn_d, dim);
                } else {
                    /* Fallback: CPU hybrid (weight not on GPU) */
                    be->mem.download(llm_x, gpu_ctx.d_x, (uint64_t)dim * sizeof(float));
                    { int sa = gpu_ctx.active; gpu_ctx.active = 0;
                      llm_gemv(llm_iswa_tmp, layer->iswa_inp_gate, llm_x, iswa_d, dim,
                               layer->iswa_inp_gate_type);
                      gpu_ctx.active = sa; }
                    llm_gelu(llm_iswa_tmp, iswa_d);
                    float *pl = llm_iswa_per_layer + L * iswa_d;
                    for (int i = 0; i < iswa_d; i++) llm_iswa_tmp[i] *= pl[i];
                    { int sa = gpu_ctx.active; gpu_ctx.active = 0;
                      llm_gemv(llm_ffn_d, layer->iswa_proj, llm_iswa_tmp, dim, iswa_d,
                               layer->iswa_proj_type);
                      gpu_ctx.active = sa; }
                    if (layer->iswa_post_norm) {
                        if (jit_fwd_rmsnorm)
                            jit_fwd_rmsnorm(llm_ffn_d, llm_ffn_d, (const float *)layer->iswa_post_norm, dim);
                        else
                            llm_rmsnorm(llm_ffn_d, llm_ffn_d, layer->iswa_post_norm, dim,
                                        GGML_TYPE_F32, m->rms_eps);
                    }
                    be->mem.upload(gpu_ctx.d_ffn_d, llm_ffn_d, (uint64_t)dim * sizeof(float));
                    be->compute.add(gpu_ctx.d_x, gpu_ctx.d_x, gpu_ctx.d_ffn_d, dim);
                }
            }

            /* Per-layer output scaling (Gemma4) */
            if (layer->iswa_out_scale) {
                float s = ((const float *)layer->iswa_out_scale)[0];
                be->compute.scale(gpu_ctx.d_x, gpu_ctx.d_x, s, dim);
            }
            if (detail_layer) { be->mem.sync(); _lp1 = hal_timer_us(); t_l_iswa += (_lp1 - _lp0); }

            /* GPU multi-layer capture: download hidden state for per-layer PCA calibration.
             * Only fires when BRIDGE_MODE_CAP_ALL is active (not during normal inference).
             * CAP_ONCE guard prevents decode-step from overwriting prefill captures. */
            if ((llm_bridge.mode & BRIDGE_MODE_CAP_ALL) &&
                llm_bridge.multi_cap_bufs && llm_bridge.multi_cap_valid &&
                L >= 0 && L < llm_bridge.multi_cap_n &&
                llm_bridge.multi_cap_dim == dim) {
                int _already = (llm_bridge.mode & BRIDGE_MODE_CAP_ONCE)
                               && llm_bridge.multi_cap_valid[L];
                if (!_already) {
                    be->mem.sync();
                    float *_gpu_cap_dst = llm_bridge.multi_cap_bufs + (size_t)L * dim;
                    be->mem.download(_gpu_cap_dst, gpu_ctx.d_x, (uint64_t)dim * sizeof(float));
                    llm_bridge.multi_cap_valid[L] = 1;
                }
            }

        }

        /* 3. Final output RMSNorm on GPU (all models) */
        if (prof_enabled) { be->mem.sync(); _pt1 = hal_timer_us(); t_layers = (_pt1 - _pt0) / 1000.0; _pt0 = _pt1; }
        be->compute.rmsnorm(gpu_ctx.d_xn, gpu_ctx.d_x, gpu_ctx.d_output_norm,
                            dim, gpu_ctx.rms_eps);

        /* 4. LM head projection on GPU → logits */
        if (!want_logits) {
            return;
        } else if (d_lm) {
            be->compute.gemv(gpu_ctx.d_out, d_lm, gpu_ctx.d_xn,
                             m->vocab_size, dim, d_lm_gpu_type);
            /* 5. Logit softcapping on GPU (before download) */
            if (m->logit_softcap > 0.0f)
                be->compute.softcap(gpu_ctx.d_out, m->vocab_size, m->logit_softcap);

            /* End graph capture BEFORE download (cudaMemcpy is synchronous) */
            if (capturing) {
                uint64_t _cap_t0 = hal_timer_us();
                int _cap_rc = cuda_graph_end_capture();
                uint64_t _cap_t1 = hal_timer_us();
                if (_cap_rc == 0) {
                    cuda_graph_captured = 1;
                    cuda_graph_ctx_len  = pos; /* record position for validity check */
                    if (llm_emit_benchmark_debug())
                        kprintf("[GRAPH] captured at pos=%d in %llums\n", pos, (unsigned long long)(_cap_t1-_cap_t0)/1000);
                    cuda_graph_launch();
                } else {
                    /* Capture failed — re-execute without capture */
                    if (llm_emit_benchmark_debug())
                        kprintf("[CUDA Graph] end_capture failed at pos=%d\n", pos);
                    capturing = 0;
                    llm_forward_token(m, logits, token_id, pos);
                    return;
                }
            }

            /* Download logits (waits for graph/kernels to finish on stream 0) */
            be->mem.download(logits, gpu_ctx.d_out, (uint64_t)m->vocab_size * sizeof(float));
            be->mem.sync();
            /* Capture hidden state for context-conditioned manifold lookup */
            if (llm_last_hs)
                be->mem.download(llm_last_hs, gpu_ctx.d_xn,
                                 (uint64_t)dim * sizeof(float));
            llm_last_hs_valid = (llm_last_hs != NULL);
        } else {
            /* Fallback: download hidden state and run LM head on CPU */
            if (capturing) {
                cuda_graph_end_capture();  /* discard — can't include CPU ops */
                capturing = 0;
            }
            be->mem.download(llm_xn, gpu_ctx.d_xn, (uint64_t)dim * sizeof(float));
            if (pos >= 4 && pos <= 7) {
                kprintf("[DBG] pre-lmhead pos=%d xn[0..7]= %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", pos,
                    llm_xn[0],llm_xn[1],llm_xn[2],llm_xn[3],llm_xn[4],llm_xn[5],llm_xn[6],llm_xn[7]);
            }
            if (llm_last_hs) kmemcpy(llm_last_hs, llm_xn, (uint64_t)dim * sizeof(float));
            llm_last_hs_valid = (llm_last_hs != NULL);
            llm_gemv(logits, lm_head, llm_xn, m->vocab_size, dim, lm_type);
            if (pos == 0) {
                /* Print top-3 logits for diagnosis */
                int best=0, best2=1, best3=2;
                for (int i=0;i<m->vocab_size;i++) {
                    if (logits[i]>logits[best]) {best3=best2;best2=best;best=i;}
                    else if (logits[i]>logits[best2]) {best3=best2;best2=i;}
                    else if (logits[i]>logits[best3]) best3=i;
                }
                kprintf("[DBG] top logits pos=%d: [%d]=%.4f [%d]=%.4f [%d]=%.4f tok0=%.4f\n",
                    pos, best,logits[best],best2,logits[best2],best3,logits[best3],logits[0]);
            }
            /* Softcap on CPU fallback */
            if (m->logit_softcap > 0.0f) {
                float cap = m->logit_softcap;
                float inv_cap = 1.0f / cap;
                for (int i = 0; i < m->vocab_size; i++) {
                    float xv = logits[i] * inv_cap;
                    float e2x = llm_expf(2.0f * xv);
                    logits[i] = cap * (e2x - 1.0f) / (e2x + 1.0f);
                }
            }
        }

        if (prof_enabled) {
            be->mem.sync(); _pt1 = hal_timer_us();
            t_lmhead = (_pt1 - _pt0) / 1000.0;
            kprintf("[PROF] embed=%.2fms iswa=%.2fms layers=%.2fms lmhead=%.2fms total=%.2fms\n",
                    t_embed, t_iswa_pre, t_layers, t_lmhead,
                    t_embed + t_iswa_pre + t_layers + t_lmhead);
            if (prof_detail) {
                kprintf("[LAYER10] norm=%.0fus qkv=%.0fus rope=%.0fus attn=%.0fus oproj=%.0fus ffn=%.0fus iswa=%.0fus total=%.0fus\n",
                        t_l_rmsnorm, t_l_qkv, t_l_rope, t_l_attn, t_l_oproj, t_l_ffn, t_l_iswa,
                        t_l_rmsnorm+t_l_qkv+t_l_rope+t_l_attn+t_l_oproj+t_l_ffn+t_l_iswa);
            }
        }

        return;
    }
#endif /* ENABLE_CUDA GPU-resident forward */

    llm_embed(llm_x, m, token_id);

    /* 2. Process each transformer layer */
    int rope_dim = m->rope_dim > 0 ? m->rope_dim : hd; /* partial RoPE for Phi-2 */

    /* === Gemma4 ISWA: precompute per-layer embeddings === */
    if (m->is_gemma4 && m->iswa_tok_embd && m->iswa_model_proj) {
        int iswa_d = m->iswa_n_embd;   /* 256 */
        int iswa_total = iswa_d * m->n_layers; /* 8960 */
        float inv_sqrt_dim = 1.0f / llm_sqrtf((float)dim);
        float sqrt_iswa_d  = llm_sqrtf((float)iswa_d);
        float inv_sqrt_2   = 1.0f / llm_sqrtf(2.0f);

        /* Step 1: token embedding lookup from per_layer_token_embd → [8960] */
        uint64_t emb_rb = llm_row_bytes(iswa_total, m->iswa_tok_embd_type);
        const uint8_t *emb_row = (const uint8_t *)m->iswa_tok_embd
                                 + (uint64_t)token_id * emb_rb;
        /* Dequant Q6_K row into llm_iswa_per_layer */
        if (m->iswa_tok_embd_type == GGML_TYPE_Q6_K) {
            const ggml_q6_k_t *blocks = (const ggml_q6_k_t *)emb_row;
            int nsb = iswa_total / 256;
            for (int sb = 0; sb < nsb; sb++) {
                const ggml_q6_k_t *b = &blocks[sb];
                float d_val = fp16_to_fp32(b->d);
                const uint8_t *ql = b->ql;
                const uint8_t *qh = b->qh;
                const int8_t *sc = b->scales;
                float *o = llm_iswa_per_layer + sb * 256;
                for (int half = 0; half < 2; half++) {
                    const uint8_t *ql_h = ql + half * 64;
                    const uint8_t *qh_h = qh + half * 32;
                    const int8_t *sc_h = sc + half * 8;
                    for (int l = 0; l < 32; l++) {
                        int q0 = (int)(ql_h[l] & 0xF)      | (int)(((qh_h[l] >> 0) & 3) << 4);
                        int q1 = (int)(ql_h[l + 32] & 0xF)  | (int)(((qh_h[l] >> 2) & 3) << 4);
                        int q2 = (int)(ql_h[l] >> 4)         | (int)(((qh_h[l] >> 4) & 3) << 4);
                        int q3 = (int)(ql_h[l + 32] >> 4)    | (int)(((qh_h[l] >> 6) & 3) << 4);
                        int si = l / 16;
                        o[half*128 + l]      = d_val * (float)sc_h[0 + si] * (float)(q0 - 32);
                        o[half*128 + l + 32] = d_val * (float)sc_h[2 + si] * (float)(q1 - 32);
                        o[half*128 + l + 64] = d_val * (float)sc_h[4 + si] * (float)(q2 - 32);
                        o[half*128 + l + 96] = d_val * (float)sc_h[6 + si] * (float)(q3 - 32);
                    }
                }
            }
        } else {
            /* F32/F16/BF16 fallback */
            for (int i = 0; i < iswa_total; i++)
                llm_iswa_per_layer[i] = llm_get_f(m->iswa_tok_embd, i, m->iswa_tok_embd_type);
        }
        /* Scale by sqrt(iswa_d) */
        for (int i = 0; i < iswa_total; i++)
            llm_iswa_per_layer[i] *= sqrt_iswa_d;

        /* Step 2: project model input to per-layer space */
        /* per_layer_proj[8960] = per_layer_model_proj[1536, 8960] · inpL[1536] */
        /* We use llm_iswa_tmp as temp for the projection (need 8960 floats — reuse ffn_g) */
        float *proj_buf = llm_ffn_g; /* reuse FFN scratch (size >= 12288 > 8960) */
        llm_gemv(proj_buf, m->iswa_model_proj, llm_x, iswa_total, dim,
                 m->iswa_model_proj_type);
        /* Scale by 1/sqrt(dim) */
        for (int i = 0; i < iswa_total; i++)
            proj_buf[i] *= inv_sqrt_dim;

        /* Step 3: RMSNorm each [iswa_d] slice with per_layer_proj_norm */
        if (m->iswa_proj_norm) {
            const float *norm_w = (const float *)m->iswa_proj_norm;
            for (int l = 0; l < m->n_layers; l++) {
                float *sl = proj_buf + l * iswa_d;
                float ss = 0.0f;
                for (int i = 0; i < iswa_d; i++) ss += sl[i] * sl[i];
                float rms = 1.0f / llm_sqrtf(ss / (float)iswa_d + m->rms_eps);
                for (int i = 0; i < iswa_d; i++) sl[i] = sl[i] * rms * norm_w[i];
            }
        }

        /* Step 4: Add token embedding + projection, scale by 1/sqrt(2) */
        for (int i = 0; i < iswa_total; i++)
            llm_iswa_per_layer[i] = (llm_iswa_per_layer[i] + proj_buf[i]) * inv_sqrt_2;
    }

    for (int L = 0; L < m->n_layers; L++) {
        llm_layer_t *layer = &m->layers[L];

        /* Crash-hunt: log every layer when bridge-capture is active (first token only) */
        if ((llm_bridge.mode & BRIDGE_MODE_CAPTURE) && pos == 0)
            kprintf("[CPU-L] L=%d attn_norm=%p q=%p k=%p v=%p o=%p\n",
                    L, layer->attn_norm, layer->q_weight, layer->k_weight,
                    layer->v_weight, layer->o_weight);

        /* Bridge: inject hidden state before this layer */
        if (llm_bridge.mode & BRIDGE_MODE_INJECT) {
            int inj_layer = llm_bridge.inject_layer < 0 ? 0 : llm_bridge.inject_layer;
            if (L == inj_layer)
                tensor_bridge_inject(&llm_bridge, llm_x, dim, pos);
        }

        /* === Self-Attention === */

        /* 2a. Pre-attention norm (LayerNorm for Phi-2, RMSNorm for others) */
        if (m->use_layernorm)
            llm_layernorm(llm_xn, llm_x, layer->attn_norm,
                          layer->attn_norm_bias, dim, layer->attn_norm_type);
        else if (jit_fwd_rmsnorm && layer->attn_norm_type == GGML_TYPE_F32)
            jit_fwd_rmsnorm(llm_xn, llm_x, (const float *)layer->attn_norm, dim);
        else
            llm_rmsnorm(llm_xn, llm_x, layer->attn_norm, dim, layer->attn_norm_type, m->rms_eps);

        /* Per-layer head_dim for Gemma4 (SWA=256, full=512) */
        int lhd = layer->head_dim_layer ? layer->head_dim_layer : hd;
        int lkv_dim = n_kv * lhd;
        int lq_dim = n_heads * lhd;
        int has_own_kv = (layer->kv_reuse_layer < 0);

        /* 2b. Q/K/V projections.
         * Fast path: when norm weights are F32 and all projections are Q4_0,
         * use the fused RMSNorm+GEMV — inv_rms computed once and applied
         * inline per block, eliminating the llm_xn write + re-read.
         * Manifold path: if Geodesic Projection (GP) compressed weights exist,
         * compute x_sub = P^T @ xn once, then use m×k GEMVs for Q/K/V. */

        /* Per-layer GP: select layer-specific Pt basis if available */
        if (axex_manifold_has_layer_bases())
            axex_manifold_select_layer(L);

        /* Geodesic Projection: project xn into manifold subspace once */
        float x_sub_fwd[AXEX_MANIFOLD_K_MAX];
        int   x_sub_fwd_ready = 0;
        {
            const axex_manifold_weight_t *_mq = axex_get_manifold_layer(L, 1);
            if (_mq) { axex_manifold_project_x(llm_xn, x_sub_fwd); x_sub_fwd_ready = 1; }
        }

#ifdef __AVX2__
        if (!x_sub_fwd_ready &&       /* skip fused path when GP is active */
            !m->use_layernorm &&
            layer->attn_norm_type == GGML_TYPE_F32 &&
            layer->q_type == GGML_TYPE_Q4_0) {
            const float *attn_nw = (const float *)layer->attn_norm;
            llm_gemv_q4_fused_rmsnorm_avx2(llm_q, layer->q_weight,
                                            llm_x, attn_nw, lq_dim, dim, m->rms_eps);
            if (has_own_kv &&
                layer->k_type == GGML_TYPE_Q4_0 &&
                layer->v_type == GGML_TYPE_Q4_0) {
                llm_gemv_q4_fused_rmsnorm_avx2(llm_k_buf, layer->k_weight,
                                                llm_x, attn_nw, lkv_dim, dim, m->rms_eps);
                llm_gemv_q4_fused_rmsnorm_avx2(llm_v_buf, layer->v_weight,
                                                llm_x, attn_nw, lkv_dim, dim, m->rms_eps);
            } else if (has_own_kv) {
                llm_gemv(llm_k_buf, layer->k_weight, llm_xn, lkv_dim, dim, layer->k_type);
                llm_gemv(llm_v_buf, layer->v_weight, llm_xn, lkv_dim, dim, layer->v_type);
            }
        } else
#endif
        {
            /* Standard / GP path */
            if (x_sub_fwd_ready)
                axex_manifold_weight_gemv(axex_get_manifold_layer(L, 1), x_sub_fwd, llm_q);
            else
                llm_gemv(llm_q, layer->q_weight, llm_xn, lq_dim, dim, layer->q_type);
            if (has_own_kv) {
                const axex_manifold_weight_t *_mk = axex_get_manifold_layer(L, 2);
                const axex_manifold_weight_t *_mv = axex_get_manifold_layer(L, 3);
                if (_mk && x_sub_fwd_ready) axex_manifold_weight_gemv(_mk, x_sub_fwd, llm_k_buf);
                else llm_gemv(llm_k_buf, layer->k_weight, llm_xn, lkv_dim, dim, layer->k_type);
                if (_mv && x_sub_fwd_ready) axex_manifold_weight_gemv(_mv, x_sub_fwd, llm_v_buf);
                else llm_gemv(llm_v_buf, layer->v_weight, llm_xn, lkv_dim, dim, layer->v_type);
            }
        }

        /* Add bias if present (Phi-2) */
        llm_add_bias(llm_q,     layer->q_bias, lq_dim);
        if (has_own_kv) {
            llm_add_bias(llm_k_buf, layer->k_bias, lkv_dim);
            llm_add_bias(llm_v_buf, layer->v_bias, lkv_dim);
        }

        /* Gemma4: per-head Q/K normalization */
        if (layer->q_norm) {
            const float *qnw = (const float *)layer->q_norm;
            if (jit_fwd_rmsnorm_lhd) {
                for (int h = 0; h < n_heads; h++)
                    jit_fwd_rmsnorm_lhd(llm_q + h * lhd, llm_q + h * lhd, qnw, lhd);
            } else {
                for (int h = 0; h < n_heads; h++) {
                    float *qh = llm_q + h * lhd;
                    float ss = 0.0f;
                    for (int i = 0; i < lhd; i++) ss += qh[i] * qh[i];
                    float rms = 1.0f / llm_sqrtf(ss / (float)lhd + m->rms_eps);
                    for (int i = 0; i < lhd; i++) qh[i] = qh[i] * rms * qnw[i];
                }
            }
        }
        if (layer->k_norm && has_own_kv) {
            const float *knw = (const float *)layer->k_norm;
            if (jit_fwd_rmsnorm_lhd) {
                for (int h = 0; h < n_kv; h++)
                    jit_fwd_rmsnorm_lhd(llm_k_buf + h * lhd, llm_k_buf + h * lhd, knw, lhd);
            } else {
                for (int h = 0; h < n_kv; h++) {
                    float *kh = llm_k_buf + h * lhd;
                    float ss = 0.0f;
                    for (int i = 0; i < lhd; i++) ss += kh[i] * kh[i];
                    float rms = 1.0f / llm_sqrtf(ss / (float)lhd + m->rms_eps);
                    for (int i = 0; i < lhd; i++) kh[i] = kh[i] * rms * knw[i];
                }
            }
            /* V bare RMSNorm: normalize per-head without learned weights */
            for (int h = 0; h < n_kv; h++) {
                float *vh = llm_v_buf + h * lhd;
                float ss = 0.0f;
                for (int i = 0; i < lhd; i++) ss += vh[i] * vh[i];
                float rms = 1.0f / llm_sqrtf(ss / (float)lhd + m->rms_eps);
                for (int i = 0; i < lhd; i++) vh[i] *= rms;
            }
        }

        /* 2c. Apply RoPE to Q and K */
        /* Gemma4: different RoPE base for SWA vs full attention layers */
        float layer_rope_base = m->rope_base;
        int   layer_rope_dim  = lhd; /* Gemma4 uses full head_dim for RoPE */
        if (m->is_gemma4 && lhd == m->head_dim_swa)
            layer_rope_base = m->rope_base_swa;

        /* Select longrope factors: short if pos < orig_ctx, long otherwise */
        const float *rope_f = (const float *)0;
        if (m->rope_factors_short || m->rope_factors_long) {
            rope_f = (pos < m->rope_orig_ctx) ? m->rope_factors_short : m->rope_factors_long;
        }
        /* Gemma4: full-attention layers use precomputed rope_freqs as factors */
        if (m->is_gemma4 && m->rope_freqs && lhd != m->head_dim_swa) {
            rope_f = m->rope_freqs;
        }

        /* Use JIT RoPE with per-dimension cache (supports Gemma4 variable head dims) */
        int rdim = m->rope_dim > 0 ? m->rope_dim : layer_rope_dim;
        jit_rope_fn rope_jit = jit_get_rope_for_dim(rdim);
        if (rope_jit) {
            llm_rope_precompute(layer_rope_base, rdim, rope_f);
            for (int h = 0; h < n_heads; h++)
                rope_jit(llm_q + h * lhd, pos, rdim, llm_rope_freqs_buf);
            if (has_own_kv)
                for (int h = 0; h < n_kv; h++)
                    rope_jit(llm_k_buf + h * lhd, pos, rdim, llm_rope_freqs_buf);
        } else {
            for (int h = 0; h < n_heads; h++)
                llm_rope(llm_q + h * lhd, pos, rdim, layer_rope_base, rope_f);
            if (has_own_kv)
                for (int h = 0; h < n_kv; h++)
                    llm_rope(llm_k_buf + h * lhd, pos, rdim, layer_rope_base, rope_f);
        }

        /* 2d. Store K,V in cache */
        int64_t kv_stride = (int64_t)m->max_seq * kv_dim; /* int64_t: overflows int for large seq*kv_dim products */
        int kv_src_layer = has_own_kv ? L : layer->kv_reuse_layer;
        if (has_own_kv) {
            /* Write K/V to this layer's cache slot.
             * If lhd < hd, we write lkv_dim floats into a kv_dim slot — pad rest with 0. */
            float *kc = m->k_cache + (int64_t)L * kv_stride + pos * kv_dim;
            float *vc = m->v_cache + (int64_t)L * kv_stride + pos * kv_dim;

            /* Geodesic KV compression: check each head independently.
             * When axex_kv_should_store() returns 0, the new K is geodesically
             * redundant — copy the merge-target slot instead to save attention work. */
            axex_kv_ctx_t *_kv_ctx = axex_get_global_kv_ctx();

            if (lkv_dim < kv_dim) {
                kmemset(kc, 0, kv_dim * sizeof(float));
                kmemset(vc, 0, kv_dim * sizeof(float));
            }
            for (int h = 0; h < n_kv; h++) {
                const float *k_head = llm_k_buf + h * lhd;
                int store = 1;
                if (_kv_ctx) {
                    store = axex_kv_should_store(_kv_ctx, k_head, lhd, L, h, pos);
                    if (!store) {
                        /* Merge: copy the target slot (already written earlier) */
                        int16_t tgt = _kv_ctx->heads[L * _kv_ctx->n_kv_heads + h].merge_into[pos];
                        if (tgt >= 0 && tgt < pos) {
                            float *ktgt = m->k_cache + L * kv_stride + tgt * kv_dim + h * hd;
                            float *vtgt = m->v_cache + L * kv_stride + tgt * kv_dim + h * hd;
                            kmemcpy(kc + h * hd, ktgt, lhd * sizeof(float));
                            kmemcpy(vc + h * hd, vtgt, lhd * sizeof(float));
                            continue;
                        }
                        /* Fallback: target not valid, store normally */
                    }
                }
                kmemcpy(kc + h * hd, k_head,              lhd * sizeof(float));
                kmemcpy(vc + h * hd, llm_v_buf + h * lhd, lhd * sizeof(float));
            }
        }

        /* 2e. Multi-head attention with GQA */
        kmemset(llm_attn_out, 0, lq_dim * sizeof(float));

        int heads_per_kv = n_heads / n_kv;

        /* Gemma4: attention scaling = 1.0 (pre-normalized by Q/K norms) */
        float attn_scale = m->is_gemma4 ? 1.0f : (1.0f / llm_sqrtf((float)lhd));

        {
            /* Flash Attention decode: single-pass tiled online-softmax.
             * Avoids materializing the full N attention-score vector per head;
             * processes KV cache in Bc-sized tiles for better L1 utilisation.
             * kv_dim = n_kv * hd is the stride between successive positions. */
            flash_attn_config_t fa_cfg;
            fa_cfg.head_dim     = lhd;
            fa_cfg.n_heads      = n_heads;
            fa_cfg.n_kv_heads   = n_kv;
            fa_cfg.seq_len      = m->max_seq;
            fa_cfg.block_size_q = DEFAULT_BR;
            fa_cfg.block_size_kv = DEFAULT_BC;
            fa_cfg.scale        = attn_scale;
            fa_cfg.causal       = true;
            fa_cfg.use_alibi    = false;

            flash_attn_decode_strided(
                llm_attn_out,
                llm_q,
                m->k_cache + (uint64_t)kv_src_layer * kv_stride,
                m->v_cache + (uint64_t)kv_src_layer * kv_stride,
                pos + 1,
                kv_dim,
                &fa_cfg);
        }

        /* 2f. Output projection + residual */
        {
            const axex_manifold_weight_t *_gp_o = axex_get_manifold_layer(L, 4);
            if (_gp_o && _gp_o->W_proj && lq_dim == axex_manifold_n()) {
                float x_sub_o[AXEX_MANIFOLD_K_MAX];
                axex_manifold_project_x(llm_attn_out, x_sub_o);
                axex_manifold_weight_gemv(_gp_o, x_sub_o, llm_ffn_d);
            } else {
                llm_gemv(llm_ffn_d, layer->o_weight, llm_attn_out, dim, lq_dim, layer->o_type);
            }
        }

        /* Gemma4: post-attention RMSNorm */
        if (layer->post_attn_norm) {
            if (jit_fwd_rmsnorm)
                jit_fwd_rmsnorm(llm_ffn_d, llm_ffn_d, (const float *)layer->post_attn_norm, dim);
            else
                llm_rmsnorm(llm_ffn_d, llm_ffn_d, layer->post_attn_norm, dim,
                            GGML_TYPE_F32, m->rms_eps);
        }

        if (jit_fwd_vadd_dim)
            jit_fwd_vadd_dim(llm_x, llm_ffn_d, dim);
        else
            llm_vadd_f32(llm_x, llm_ffn_d, dim);
        if (llm_attnres_enabled) {
            float depth_frac = (float)(L + 1) / (float)(m->n_layers > 0 ? m->n_layers : 1);
            float gamma = 1.0f - (0.12f * llm_attnres_strength * depth_frac);
            if (gamma < 0.70f) gamma = 0.70f;
            for (int i = 0; i < dim; i++) llm_x[i] *= gamma;
        }

        /* === Feed-Forward Network === */

        /* Per-layer FFN dim (Gemma4: 6144 for early layers, 12288 for later) */
        int lff = layer->ff_dim_layer ? layer->ff_dim_layer : ff;

        /* 2g. Pre-FFN norm (LayerNorm for Phi-2, RMSNorm for others) */
        if (m->use_layernorm)
            llm_layernorm(llm_xn, llm_x, layer->ffn_norm,
                          layer->ffn_norm_bias, dim, layer->ffn_norm_type);
        else if (jit_fwd_rmsnorm && layer->ffn_norm_type == GGML_TYPE_F32)
            jit_fwd_rmsnorm(llm_xn, llm_x, (const float *)layer->ffn_norm, dim);
        else
            llm_rmsnorm(llm_xn, llm_x, layer->ffn_norm, dim, layer->ffn_norm_type, m->rms_eps);

        if (m->use_gelu || !layer->ffn_gate) {
            /* 2h-alt. GELU FFN (Phi-2): hidden = GELU(W_up · x); out = W_down · hidden */
            llm_gemv(llm_ffn_u, layer->ffn_up, llm_xn, lff, dim, layer->up_type);
            llm_add_bias(llm_ffn_u, layer->ffn_up_bias, lff);
            if (jit_fwd_gelu && lff == ff) jit_fwd_gelu(llm_ffn_u, lff);
            else llm_gelu(llm_ffn_u, lff);
            {
                const axex_compressed_weight_t *_cw = axex_get_compressed_layer(L, 0);
                if (_cw) axex_compressed_weight_gemv(_cw, llm_ffn_u, llm_ffn_d);
                else     llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_u, dim, lff, layer->down_type);
            }
            llm_add_bias(llm_ffn_d, layer->ffn_down_bias, dim);
        } else if (m->use_geglu) {
            /* 2h-geglu. GeGLU (Gemma): hidden = GELU(W_gate · x) ⊙ (W_up · x) */
            llm_gemv(llm_ffn_g, layer->ffn_gate, llm_xn, lff, dim, layer->gate_type);
            llm_gemv(llm_ffn_u, layer->ffn_up,   llm_xn, lff, dim, layer->up_type);

            if (jit_fwd_gelu && lff == ff) jit_fwd_gelu(llm_ffn_g, lff);
            else llm_gelu(llm_ffn_g, lff);

            if (jit_fwd_vmul_ff && lff == ff)
                jit_fwd_vmul_ff(llm_ffn_g, llm_ffn_u, lff);
            else
                llm_vmul_f32(llm_ffn_g, llm_ffn_u, lff);

            /* FFN capture for GeGLU branch */
            if ((llm_bridge.mode & BRIDGE_MODE_CAP_FFN) &&
                llm_bridge.ffn_cap_bufs && llm_bridge.ffn_cap_valid &&
                L >= 0 && L < llm_bridge.ffn_cap_n &&
                lff <= llm_bridge.ffn_cap_dim) {
                float *fdst = llm_bridge.ffn_cap_bufs + (size_t)L * llm_bridge.ffn_cap_dim;
                kmemcpy(fdst, llm_ffn_g, (size_t)lff * sizeof(float));
                if (lff < llm_bridge.ffn_cap_dim)
                    kmemset(fdst + lff, 0, (size_t)(llm_bridge.ffn_cap_dim - lff) * sizeof(float));
                llm_bridge.ffn_cap_valid[L] = 1;
            }

            {
                const axex_compressed_weight_t *_cw = axex_get_compressed_layer(L, 0);
                const axex_manifold_weight_t   *_md = axex_get_manifold_layer(L, 7);
                if (_md) {
                    float x_sub_ffn_q[AXEX_MANIFOLD_K_MAX];
                    axex_manifold_project_ffn(L, llm_ffn_g, x_sub_ffn_q);
                    axex_manifold_weight_gemv(_md, x_sub_ffn_q, llm_ffn_d);
                } else if (_cw) {
                    axex_compressed_weight_gemv(_cw, llm_ffn_g, llm_ffn_d);
                } else {
                    llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_g, dim, lff, layer->down_type);
                }
            }
        } else {
            /* 2h. SwiGLU: hidden = SiLU(W_gate · x) ⊙ (W_up · x)
             * Geodesic Projection (GP) path: if manifold-compressed gate/up
             * exist, project xn → x_sub once and use the compact W_proj GEMVs.
             * Fast fused-RMSNorm path: only when GP is not active. */

            /* GP: project FFN-normed xn into manifold subspace */
            float x_sub_ffn[AXEX_MANIFOLD_K_MAX];
            int   x_sub_ffn_ready = 0;
            {
                const axex_manifold_weight_t *_mg = axex_get_manifold_layer(L, 6);
                if (_mg) {
                    axex_manifold_project_x(llm_xn, x_sub_ffn);
                    x_sub_ffn_ready = 1;
                }
            }

#ifdef __AVX2__
            if (!x_sub_ffn_ready &&   /* skip fused path when GP is active */
                !m->use_layernorm &&
                layer->ffn_norm_type == GGML_TYPE_F32 &&
                layer->gate_type == GGML_TYPE_Q4_0 &&
                layer->up_type   == GGML_TYPE_Q4_0) {
                const float *ffn_nw = (const float *)layer->ffn_norm;
                llm_gemv_q4_fused_rmsnorm_avx2(llm_ffn_g, layer->ffn_gate,
                                                llm_x, ffn_nw, lff, dim, m->rms_eps);
                llm_gemv_q4_fused_rmsnorm_avx2(llm_ffn_u, layer->ffn_up,
                                                llm_x, ffn_nw, lff, dim, m->rms_eps);
            } else
#endif
            {
                /* GP path or standard path */
                const axex_manifold_weight_t *_mg = axex_get_manifold_layer(L, 6);
                const axex_manifold_weight_t *_mu = axex_get_manifold_layer(L, 5);
                if (_mg && x_sub_ffn_ready) axex_manifold_weight_gemv(_mg, x_sub_ffn, llm_ffn_g);
                else                        llm_gemv(llm_ffn_g, layer->ffn_gate, llm_xn, lff, dim, layer->gate_type);
                if (_mu && x_sub_ffn_ready) axex_manifold_weight_gemv(_mu, x_sub_ffn, llm_ffn_u);
                else                        llm_gemv(llm_ffn_u, layer->ffn_up,   llm_xn, lff, dim, layer->up_type);
            }
            if (jit_fwd_fused_silu && lff == ff) {
                jit_fwd_fused_silu(llm_ffn_g, llm_ffn_u, lff);
            } else {
                llm_silu(llm_ffn_g, lff);
                llm_vmul_f32(llm_ffn_g, llm_ffn_u, lff);
            }

            /* FFN intermediate capture (BRIDGE_MODE_CAP_FFN):
             * llm_ffn_g now holds SiLU(gate)⊙up — the ff_dim vector that feeds
             * into W_down.  Capture it here for per-layer FFN PCA calibration. */
            if ((llm_bridge.mode & BRIDGE_MODE_CAP_FFN) &&
                llm_bridge.ffn_cap_bufs && llm_bridge.ffn_cap_valid &&
                L >= 0 && L < llm_bridge.ffn_cap_n &&
                lff <= llm_bridge.ffn_cap_dim) {
                float *fdst = llm_bridge.ffn_cap_bufs + (size_t)L * llm_bridge.ffn_cap_dim;
                kmemcpy(fdst, llm_ffn_g, (size_t)lff * sizeof(float));
                /* zero-pad if lff < ffn_cap_dim (shouldn't happen but be safe) */
                if (lff < llm_bridge.ffn_cap_dim)
                    kmemset(fdst + lff, 0, (size_t)(llm_bridge.ffn_cap_dim - lff) * sizeof(float));
                llm_bridge.ffn_cap_valid[L] = 1;
            }

            {
                const axex_compressed_weight_t *_cw = axex_get_compressed_layer(L, 0);
                const axex_manifold_weight_t   *_md = axex_get_manifold_layer(L, 7);
                if (_md) {
                    /* GP down path: project ffn_hidden → sub-space then GEMV */
                    float x_sub_ffn_q[AXEX_MANIFOLD_K_MAX];
                    axex_manifold_project_ffn(L, llm_ffn_g, x_sub_ffn_q);
                    axex_manifold_weight_gemv(_md, x_sub_ffn_q, llm_ffn_d);
                } else if (_cw) {
                    axex_compressed_weight_gemv(_cw, llm_ffn_g, llm_ffn_d);
                } else {
                    llm_gemv(llm_ffn_d, layer->ffn_down, llm_ffn_g, dim, lff, layer->down_type);
                }
            }
        }

        /* Gemma4: post-FFW RMSNorm */
        if (layer->post_ffw_norm) {
            if (jit_fwd_rmsnorm)
                jit_fwd_rmsnorm(llm_ffn_d, llm_ffn_d, (const float *)layer->post_ffw_norm, dim);
            else
                llm_rmsnorm(llm_ffn_d, llm_ffn_d, layer->post_ffw_norm, dim,
                            GGML_TYPE_F32, m->rms_eps);
        }

        /* Residual: x = x + ffn_output */
        if (jit_fwd_vadd_dim)
            jit_fwd_vadd_dim(llm_x, llm_ffn_d, dim);
        else
            llm_vadd_f32(llm_x, llm_ffn_d, dim);
        if (llm_attnres_enabled) {
            float depth_frac = (float)(L + 1) / (float)(m->n_layers > 0 ? m->n_layers : 1);
            float gamma = 1.0f - (0.18f * llm_attnres_strength * depth_frac);
            if (gamma < 0.65f) gamma = 0.65f;
            for (int i = 0; i < dim; i++) llm_x[i] *= gamma;
        }

        if (llm_depth_attn_enabled && llm_depth_hist && llm_depth_scores && L > 0) {
            kmemcpy(llm_depth_hist + (uint64_t)L * dim, llm_x, (uint64_t)dim * sizeof(float));
            int start = L - llm_depth_attn_window;
            if (start < 0) start = 0;
            int cnt = 0;
            float qn = llm_sqrtf(llm_dot_f32(llm_x, llm_x, dim) + 1e-9f);
            for (int p = start; p < L; p++) {
                float *hp = llm_depth_hist + (uint64_t)p * dim;
                float hn = llm_sqrtf(llm_dot_f32(hp, hp, dim) + 1e-9f);
                float cs = llm_dot_f32(llm_x, hp, dim) / (qn * hn + 1e-9f);
                llm_depth_scores[cnt++] = cs;
            }
            float mx = llm_depth_scores[0];
            for (int i = 1; i < cnt; i++) if (llm_depth_scores[i] > mx) mx = llm_depth_scores[i];
            float sm = 0.0f;
            for (int i = 0; i < cnt; i++) {
                llm_depth_scores[i] = llm_expf((llm_depth_scores[i] - mx) * 6.0f);
                sm += llm_depth_scores[i];
            }
            if (sm < 1e-9f) sm = 1e-9f;
            float inv = 1.0f / sm;
            kmemset(llm_ffn_d, 0, (uint64_t)dim * sizeof(float));
            for (int i = 0; i < cnt; i++) {
                int p = start + i;
                float w = llm_depth_scores[i] * inv;
                llm_axpy_f32(llm_ffn_d, w, llm_depth_hist + (uint64_t)p * dim, dim);
            }
            float depth_frac = (float)(L + 1) / (float)(m->n_layers > 0 ? m->n_layers : 1);
            float alpha = llm_depth_attn_strength * depth_frac;
            if (alpha > 0.85f) alpha = 0.85f;
            for (int i = 0; i < dim; i++)
                llm_x[i] = llm_x[i] * (1.0f - alpha) + llm_ffn_d[i] * alpha;
        } else if (llm_depth_attn_enabled && llm_depth_hist && L == 0) {
            kmemcpy(llm_depth_hist + (uint64_t)L * dim, llm_x, (uint64_t)dim * sizeof(float));
        }

        /* === Gemma4 ISWA: Per-layer embedding injection === */
        if (m->is_gemma4 && layer->iswa_inp_gate) {
            int iswa_d = m->iswa_n_embd; /* 256 */

            /* 2j-a. Gate: tmp[256] = GELU(inp_gate[1536,256] · x[1536]) */
            llm_gemv(llm_iswa_tmp, layer->iswa_inp_gate, llm_x, iswa_d, dim,
                     layer->iswa_inp_gate_type);
            llm_gelu(llm_iswa_tmp, iswa_d);

            /* 2j-b. Element-wise multiply with per-layer embedding */
            float *pl = llm_iswa_per_layer + L * iswa_d;
            for (int i = 0; i < iswa_d; i++)
                llm_iswa_tmp[i] *= pl[i];

            /* 2j-c. Project back: ffn_d[1536] = proj[256,1536] · tmp[256] */
            llm_gemv(llm_ffn_d, layer->iswa_proj, llm_iswa_tmp, dim, iswa_d,
                     layer->iswa_proj_type);

            /* 2j-d. Post-norm on projected output */
            if (layer->iswa_post_norm) {
                if (jit_fwd_rmsnorm)
                    jit_fwd_rmsnorm(llm_ffn_d, llm_ffn_d, (const float *)layer->iswa_post_norm, dim);
                else
                    llm_rmsnorm(llm_ffn_d, llm_ffn_d, layer->iswa_post_norm, dim,
                                GGML_TYPE_F32, m->rms_eps);
            }

            /* 2j-e. Residual: x = x + iswa_output */
            if (jit_fwd_vadd_dim)
                jit_fwd_vadd_dim(llm_x, llm_ffn_d, dim);
            else
                llm_vadd_f32(llm_x, llm_ffn_d, dim);
        }

        /* === Gemma4: Per-layer output scaling === */
        if (layer->iswa_out_scale) {
            float s = ((const float *)layer->iswa_out_scale)[0];
            for (int i = 0; i < dim; i++)
                llm_x[i] *= s;
        }

        /* Bridge: capture hidden state after this layer */
        if (llm_bridge.mode & BRIDGE_MODE_CAPTURE) {
            int cap_layer = llm_bridge.capture_layer < 0 ? m->n_layers - 1 : llm_bridge.capture_layer;
            if (L == cap_layer)
                tensor_bridge_capture(&llm_bridge, llm_x, dim, pos);
        }
        /* Multi-layer capture: grab hidden state at every layer in one pass.
         * Used by the per-layer PCA calibration (32x fewer forward passes).
         * CAP_ONCE guard: don't overwrite with decode-step states (greedy decode
         * would overwrite all 512 prefill captures with the same token → rank-1). */
        if ((llm_bridge.mode & BRIDGE_MODE_CAP_ALL) &&
            llm_bridge.multi_cap_bufs && llm_bridge.multi_cap_valid &&
            L >= 0 && L < llm_bridge.multi_cap_n &&
            llm_bridge.multi_cap_dim == dim) {
            int _already = (llm_bridge.mode & BRIDGE_MODE_CAP_ONCE)
                           && llm_bridge.multi_cap_valid[L];
            if (!_already) {
                float *dst = llm_bridge.multi_cap_bufs + (size_t)L * dim;
                kmemcpy(dst, llm_x, (size_t)dim * sizeof(float));
                llm_bridge.multi_cap_valid[L] = 1;
            }
        }
    }

    /* 3. Final norm (LayerNorm for Phi-2, RMSNorm for others) */
    if (m->use_layernorm)
        llm_layernorm(llm_xn, llm_x, m->output_norm,
                      m->output_norm_bias, dim, m->output_norm_type);
    else if (jit_fwd_rmsnorm && m->output_norm_type == GGML_TYPE_F32)
        jit_fwd_rmsnorm(llm_xn, llm_x, (const float *)m->output_norm, dim);
    else
        llm_rmsnorm(llm_xn, llm_x, m->output_norm, dim, m->output_norm_type, m->rms_eps);

    /* 4. LM head: logits = output_weight · x */
    const void *lm_head = m->output_weight ? m->output_weight : m->token_embd;
    ggml_type_t lm_type = m->output_weight ? m->output_type : m->token_embd_type;

    if (!want_logits) {
        return;
    }

    llm_gemv(logits, lm_head, llm_xn, m->vocab_size, dim, lm_type);

    /* 5. Logit softcapping: logits = cap * tanh(logits / cap) */
    if (m->logit_softcap > 0.0f) {
        float cap = m->logit_softcap;
        float inv_cap = 1.0f / cap;
        for (int i = 0; i < m->vocab_size; i++) {
            float x = logits[i] * inv_cap;
            /* tanh approximation: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) */
            float e2x = llm_expf(2.0f * x);
            logits[i] = cap * (e2x - 1.0f) / (e2x + 1.0f);
        }
    }

}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Tokenizer: GGUF Vocabulary Parsing and BPE Encoding                        */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Walk a GGUF string array and extract the i-th string */
static int gguf_array_string_at(const gguf_kv_t *kv, int idx,
                                const char **out_str, int *out_len)
{
    if (!kv || kv->type != GGUF_TYPE_ARRAY ||
        kv->value.array.elem_type != GGUF_TYPE_STRING)
        return -1;
    if ((uint64_t)idx >= kv->value.array.count) return -1;

    const uint8_t *p = (const uint8_t *)kv->value.array.data;
    for (int i = 0; i <= idx; i++) {
        uint64_t len = (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
                       ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
                       ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
                       ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
        p += 8;
        if (i == idx) {
            *out_str = (const char *)p;
            *out_len = (int)len;
            return 0;
        }
        p += len;
    }
    return -1;
}

/* Walk a GGUF float32 array and get the i-th element */
static float gguf_array_f32_at(const gguf_kv_t *kv, int idx)
{
    if (!kv || kv->type != GGUF_TYPE_ARRAY ||
        kv->value.array.elem_type != GGUF_TYPE_FLOAT32)
        return 0.0f;
    if ((uint64_t)idx >= kv->value.array.count) return 0.0f;

    const float *f = (const float *)kv->value.array.data;
    return f[idx];
}

/* Build vocab lookup from GGUF metadata */
static int llm_build_vocab(llm_model_t *m, gguf_ctx_t *ctx)
{
    const gguf_kv_t *tok_kv = gguf_find_kv(ctx, "tokenizer.ggml.tokens");
    if (!tok_kv || tok_kv->type != GGUF_TYPE_ARRAY) {
        kprintf("[LLM] No tokenizer.ggml.tokens found\n");
        return -1;
    }

    int n_vocab = (int)tok_kv->value.array.count;
    if (m->vocab_size > 0 && n_vocab > m->vocab_size)
        n_vocab = m->vocab_size;
    m->n_vocab = n_vocab;

    /* Allocate vocab array */
    m->vocab = (llm_vocab_entry_t *)tensor_alloc(
        (uint64_t)n_vocab * sizeof(llm_vocab_entry_t));
    if (!m->vocab) {
        kprintf("[LLM] ERROR: Failed to allocate vocab (%d entries)\n", n_vocab);
        return -1;
    }

    /* Walk the string array and populate vocab entries */
    const uint8_t *p = (const uint8_t *)tok_kv->value.array.data;
    for (int i = 0; i < n_vocab; i++) {
        uint64_t len = (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
                       ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
                       ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
                       ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
        p += 8;
        m->vocab[i].str = (const char *)p;
        m->vocab[i].len = (uint16_t)(len > 65535 ? 65535 : len);
        p += len;
    }

    /* Get merge scores */
    const gguf_kv_t *scores_kv = gguf_find_kv(ctx, "tokenizer.ggml.scores");
    if (scores_kv && scores_kv->type == GGUF_TYPE_ARRAY &&
        scores_kv->value.array.elem_type == GGUF_TYPE_FLOAT32) {
        m->vocab_scores = (float *)scores_kv->value.array.data;
    } else {
        m->vocab_scores = NULL;
    }

    /* Get special token IDs */
    m->bos_id = (int)gguf_get_u32(ctx, "tokenizer.ggml.bos_token_id", 1);
    m->eos_id = (int)gguf_get_u32(ctx, "tokenizer.ggml.eos_token_id", 2);

    /* Build byte fallback table */
    for (int b = 0; b < 256; b++) m->byte_tokens[b] = -1;

    /* Check for byte tokens like <0x00>, <0x01>, ... or raw byte entries */
    for (int i = 0; i < n_vocab; i++) {
        if (m->vocab[i].len == 6 && m->vocab[i].str[0] == '<' &&
            m->vocab[i].str[1] == '0' && m->vocab[i].str[2] == 'x' &&
            m->vocab[i].str[5] == '>') {
            /* Parse <0xHH> */
            int hi = 0, lo = 0;
            char c3 = m->vocab[i].str[3];
            char c4 = m->vocab[i].str[4];
            if (c3 >= '0' && c3 <= '9') hi = c3 - '0';
            else if (c3 >= 'A' && c3 <= 'F') hi = c3 - 'A' + 10;
            else if (c3 >= 'a' && c3 <= 'f') hi = c3 - 'a' + 10;
            if (c4 >= '0' && c4 <= '9') lo = c4 - '0';
            else if (c4 >= 'A' && c4 <= 'F') lo = c4 - 'A' + 10;
            else if (c4 >= 'a' && c4 <= 'f') lo = c4 - 'a' + 10;
            m->byte_tokens[hi * 16 + lo] = i;
        } else if (m->vocab[i].len == 1) {
            /* Single-byte token */
            unsigned char ch = (unsigned char)m->vocab[i].str[0];
            if (m->byte_tokens[ch] < 0)
                m->byte_tokens[ch] = i;
        }
    }

    kprintf("[LLM] Vocab: %d tokens, BOS=%d, EOS=%d\n",
            n_vocab, m->bos_id, m->eos_id);

    /* Detect tokenizer type: SentencePiece if vocab contains U+2581 (▁) tokens */
    m->use_spm = 0;
    {
        int check_max = n_vocab < 1000 ? n_vocab : 1000;
        for (int i = 0; i < check_max; i++) {
            if (m->vocab[i].len >= 3) {
                const uint8_t *s = (const uint8_t *)m->vocab[i].str;
                for (int j = 0; j + 2 < m->vocab[i].len; j++) {
                    if (s[j] == 0xE2 && s[j+1] == 0x96 && s[j+2] == 0x81) {
                        m->use_spm = 1;
                        break;
                    }
                }
                if (m->use_spm) break;
            }
        }
        kprintf("[LLM] Tokenizer: %s\n", m->use_spm ? "SentencePiece" : "BPE");
    }

    /* Build hash table for O(1) token lookups */
    llm_build_hash_table(m);

    return 0;
}

/* Check if two strings match (str1 may not be null-terminated) */
static int llm_str_match(const char *a, int alen, const char *b, int blen)
{
    if (alen != blen) return 0;
    for (int i = 0; i < alen; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

/* Find first occurrence of needle in haystack (both null-terminated) */
static char *llm_strstr(const char *haystack, const char *needle)
{
    if (!needle[0]) return (char *)haystack;
    for (const char *h = haystack; *h; h++) {
        const char *a = h, *b = needle;
        while (*a && *b && *a == *b) { a++; b++; }
        if (!*b) return (char *)h;
    }
    return (char *)0;
}

/* FNV-1a hash for (string, length) pairs */
static uint32_t llm_hash_str(const char *str, int len)
{
    uint32_t h = 0x811c9dc5u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)str[i];
        h *= 0x01000193u;
    }
    return h;
}

/* Hash-table backed token lookup — O(1) average */
static int32_t llm_vocab_ht[LLM_HASH_SIZE];
static int llm_ht_ready = 0;

static void llm_build_hash_table(const llm_model_t *m)
{
    /* Initialize all slots to -1 (empty) */
    for (int i = 0; i < LLM_HASH_SIZE; i++)
        llm_vocab_ht[i] = -1;  /* int32_t: supports vocab up to 2^31 */

    /* Insert all vocab entries using open addressing (linear probing) */
    for (int i = 0; i < m->n_vocab; i++) {
        uint32_t h = llm_hash_str(m->vocab[i].str, m->vocab[i].len);
        uint32_t slot = h & (LLM_HASH_SIZE - 1);
        int probes = 0;
        while (llm_vocab_ht[slot] != -1 && probes < LLM_HASH_SIZE) {
            slot = (slot + 1) & (LLM_HASH_SIZE - 1);
            probes++;
        }
        if (probes < LLM_HASH_SIZE)
            llm_vocab_ht[slot] = (int32_t)i;
    }
    llm_ht_ready = 1;
}

/* Find exact token ID for a string. Hash table O(1) with fallback. */
static int llm_find_token(const llm_model_t *m, const char *str, int len)
{
    if (llm_ht_ready) {
        uint32_t h = llm_hash_str(str, len);
        uint32_t slot = h & (LLM_HASH_SIZE - 1);
        int probes = 0;
        while (probes < 64) { /* limit probe length */
            int32_t idx = llm_vocab_ht[slot];
            if (idx == -1) return -1; /* empty slot = not found */
            if (llm_str_match(m->vocab[idx].str, m->vocab[idx].len, str, len))
                return idx;
            slot = (slot + 1) & (LLM_HASH_SIZE - 1);
            probes++;
        }
        return -1;
    }
    /* Fallback: linear scan (before hash table is built) */
    for (int i = 0; i < m->n_vocab; i++) {
        if (llm_str_match(m->vocab[i].str, m->vocab[i].len, str, len))
            return i;
    }
    return -1;
}

/* Encode raw bytes to GPT-2 byte-level encoding (reverse of bpe_decode).
 * Maps control chars, space, etc. to their U+0100..U+0143 Unicode forms. */
static int llm_bpe_encode(const char *src, int slen, char *dst, int dmax)
{
    int di = 0;
    for (int si = 0; si < slen && di < dmax - 2; si++) {
        uint8_t b = (uint8_t)src[si];
        /* Passthrough bytes: ASCII printable (except space) + Latin-1 parts */
        if ((b >= 0x21 && b <= 0x7E) ||
            (b >= 0xA1 && b <= 0xAC) ||
            (b >= 0xAE && b <= 0xFF)) {
            dst[di++] = (char)b;
        } else {
            /* Remap to U+0100..U+0143 as UTF-8 */
            int idx;
            if (b <= 0x20)       idx = b;              /* 0x00..0x20 → 0..32 */
            else if (b == 0x7F)  idx = 33;
            else if (b <= 0xA0)  idx = 34 + (b - 0x80); /* 0x80..0xA0 → 34..66 */
            else                 idx = 67;              /* 0xAD → 67 */
            int cp = 0x100 + idx;
            dst[di++] = (char)(0xC0 | (cp >> 6));
            dst[di++] = (char)(0x80 | (cp & 0x3F));
        }
    }
    dst[di] = '\0';
    return di;
}

/* Tokenize text using greedy longest-match with BPE merge fallback.
 * Returns number of tokens produced. */
/* Helper: tokenize a text segment (no special tokens) */
static int llm_tokenize_segment(const llm_model_t *m, const char *text, int text_len,
                                 int *tokens, int max_tokens, int is_first_segment)
{
    int n = 0;
    static char enc_buf[4096];
    int enc_len;

    if (m->use_spm) {
        int di = 0;
        int dmax = (int)sizeof(enc_buf) - 4;
        /* Prepend ▁ for SentencePiece (only for first segment or after special token) */
        if (is_first_segment && text_len > 0 && text[0] != '\n') {
            enc_buf[di++] = (char)0xE2;
            enc_buf[di++] = (char)0x96;
            enc_buf[di++] = (char)0x81;
        }
        for (int si = 0; si < text_len && di < dmax; si++) {
            if (text[si] == ' ') {
                enc_buf[di++] = (char)0xE2;
                enc_buf[di++] = (char)0x96;
                enc_buf[di++] = (char)0x81;
            } else {
                enc_buf[di++] = text[si];
            }
        }
        enc_buf[di] = '\0';
        enc_len = di;
    } else {
        enc_len = llm_bpe_encode(text, text_len, enc_buf, sizeof(enc_buf));
    }

    /* Greedy longest-match tokenization */
    int pos = 0;
    while (pos < enc_len && n < max_tokens) {
        int best_len = 0;
        int best_id = -1;
        int max_try = enc_len - pos;
        if (max_try > 128) max_try = 128;
        for (int try_len = max_try; try_len >= 1; try_len--) {
            int id = llm_find_token(m, enc_buf + pos, try_len);
            if (id >= 0) { best_len = try_len; best_id = id; break; }
        }
        if (best_id >= 0) {
            tokens[n++] = best_id;
            pos += best_len;
        } else {
            /* BLT-enhanced byte fallback: handle full UTF-8 sequences */
            uint8_t b0 = (uint8_t)enc_buf[pos];
            int utf8_len = 1;
            if      (b0 >= 0xF0 && pos + 3 < enc_len) utf8_len = 4;
            else if (b0 >= 0xE0 && pos + 2 < enc_len) utf8_len = 3;
            else if (b0 >= 0xC0 && pos + 1 < enc_len) utf8_len = 2;

            /* Try multi-byte token match first (e.g. ▁, kanji, emoji) */
            if (utf8_len > 1) {
                int id = llm_find_token(m, enc_buf + pos, utf8_len);
                if (id >= 0) { tokens[n++] = id; pos += utf8_len; continue; }
            }

            /* Emit each byte of the UTF-8 sequence as individual byte tokens */
            for (int bi = 0; bi < utf8_len && n < max_tokens; bi++) {
                uint8_t bv = (uint8_t)enc_buf[pos + bi];
                if (m->byte_tokens[bv] >= 0) {
                    tokens[n++] = m->byte_tokens[bv];
                } else {
                    tokens[n++] = 0; /* UNK */
                }
            }
            pos += utf8_len;
        }
    }

    /* BPE merge pass */
    if (m->vocab_scores && n > 1) {
        int changed = 1;
        while (changed) {
            changed = 0;
            float best_score = -1e30f;
            int best_idx = -1, best_merged = -1;
            for (int i = 0; i < n - 1; i++) {
                int len1 = m->vocab[tokens[i]].len;
                int len2 = m->vocab[tokens[i + 1]].len;
                if (len1 + len2 > 128) continue;
                char merged[128];
                kmemcpy(merged, m->vocab[tokens[i]].str, len1);
                kmemcpy(merged + len1, m->vocab[tokens[i + 1]].str, len2);
                int mid = llm_find_token(m, merged, len1 + len2);
                if (mid >= 0 && m->vocab_scores[mid] > best_score) {
                    best_score = m->vocab_scores[mid];
                    best_idx = i;
                    best_merged = mid;
                }
            }
            if (best_idx >= 0) {
                tokens[best_idx] = best_merged;
                for (int i = best_idx + 1; i < n - 1; i++)
                    tokens[i] = tokens[i + 1];
                n--;
                changed = 1;
            }
        }
    }
    return n;
}

static int llm_tokenize(const llm_model_t *m, const char *text,
                        int *tokens, int max_tokens)
{
    int n = 0;
    int text_len = 0;
    for (const char *p = text; *p; p++) text_len++;

    /* Pre-split at special token boundaries.
     * Detect both <|...|> patterns (GPT/Phi) and <..._of_...> patterns (Gemma). */
    int pos = 0;
    int first_seg = 1;
    while (pos < text_len && n < max_tokens) {
        /* Scan for next special token pattern */
        int spec_start = -1, spec_end = -1;
        for (int i = pos; i < text_len - 2; i++) {
            if (text[i] == '<') {
                /* Try <|...|> pattern */
                if (i + 3 < text_len && text[i+1] == '|') {
                    for (int j = i + 2; j < text_len - 1 && j < i + 32; j++) {
                        if (text[j] == '|' && text[j+1] == '>') {
                            spec_start = i;
                            spec_end = j + 2;
                            break;
                        }
                    }
                    if (spec_start >= 0) break;
                }
                /* Try <...> pattern — look up in vocab to confirm it's a special token */
                for (int j = i + 1; j < text_len && j < i + 32; j++) {
                    if (text[j] == '>') {
                        int cand_len = j + 1 - i;
                        int cand_id = llm_find_token(m, text + i, cand_len);
                        if (cand_id >= 0) {
                            spec_start = i;
                            spec_end = j + 1;
                        }
                        break;
                    }
                }
                if (spec_start >= 0) break;
            }
        }

        if (spec_start < 0) {
            /* No more special tokens — tokenize remaining text */
            int seg_n = llm_tokenize_segment(m, text + pos, text_len - pos,
                                              tokens + n, max_tokens - n, first_seg);
            n += seg_n;
            break;
        }

        /* Tokenize text before special token */
        if (spec_start > pos) {
            int seg_n = llm_tokenize_segment(m, text + pos, spec_start - pos,
                                              tokens + n, max_tokens - n, first_seg);
            n += seg_n;
            first_seg = 0;
        }

        /* Look up the special token directly in vocab */
        int spec_len = spec_end - spec_start;
        int spec_id = llm_find_token(m, text + spec_start, spec_len);
        if (spec_id >= 0 && n < max_tokens) {
            tokens[n++] = spec_id;
        } else if (n < max_tokens) {
            /* Special token not in vocab — tokenize as regular text */
            int seg_n = llm_tokenize_segment(m, text + spec_start, spec_len,
                                              tokens + n, max_tokens - n, 0);
            n += seg_n;
        }
        first_seg = 0;
        pos = spec_end;
    }

    return n;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  GPT-2 BPE Byte-Level Decode                                                */
/*  Reverses the byte_encoder mapping used by GPT-2/SmolLM/LLaMA tokenizers    */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Reverse table: index i → original byte for unicode U+0100+i */
static const uint8_t bpe_rev[68] = {
    /* 0-32: U+0100..U+0120 → byte 0x00..0x20 */
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
    0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
    0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
    0x20, /* space */
    /* 33: U+0121 → 0x7F */
    0x7F,
    /* 34-66: U+0122..U+0142 → 0x80..0xA0 */
    0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8A,0x8B,0x8C,0x8D,0x8E,0x8F,
    0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,
    0x98,0x99,0x9A,0x9B,0x9C,0x9D,0x9E,0x9F,
    0xA0,
    /* 67: U+0143 → 0xAD */
    0xAD
};

static int llm_bpe_decode(const char *src, int slen, char *dst, int dmax)
{
    int di = 0;
    for (int si = 0; si < slen && di < dmax - 1; ) {
        uint8_t c = (uint8_t)src[si];
        if (c == 0xC4 && si + 1 < slen) {
            uint8_t c2 = (uint8_t)src[si + 1];
            if (c2 >= 0x80 && c2 <= 0xBF) {
                int idx = c2 - 0x80;
                if (idx < 68) { dst[di++] = (char)bpe_rev[idx]; si += 2; continue; }
            }
        } else if (c == 0xC5 && si + 1 < slen) {
            uint8_t c2 = (uint8_t)src[si + 1];
            if (c2 >= 0x80 && c2 <= 0x83) {
                int idx = 64 + (c2 - 0x80);
                if (idx < 68) { dst[di++] = (char)bpe_rev[idx]; si += 2; continue; }
            }
        }
        dst[di++] = src[si++];
    }
    dst[di] = '\0';
    return di;
}

/* Decode token to text with GPT-2 BPE byte-level reverse mapping */
static int llm_decode_token(const llm_model_t *m, int token_id,
                            char *buf, int max_len)
{
    if (token_id < 0 || token_id >= m->n_vocab) return 0;
    int len = m->vocab[token_id].len;
    if (len > 255) len = 255;
    char tmp[256];
    kmemcpy(tmp, m->vocab[token_id].str, len);
    tmp[len] = '\0';
    return llm_bpe_decode(tmp, len, buf, max_len);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Text Generation                                                            */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Argmax over logits */
static int llm_argmax(const float *logits, int n)
{
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/* Sort indices by descending logit value (insertion sort for top-k) */
static void llm_partial_sort_desc(int *indices, const float *vals, int n, int k)
{
    /* Initialize first k indices */
    for (int i = 0; i < k && i < n; i++)
        indices[i] = i;

    /* Sort first k by insertion sort */
    for (int i = 1; i < k && i < n; i++) {
        int key = indices[i];
        float key_val = vals[key];
        int j = i - 1;
        while (j >= 0 && vals[indices[j]] < key_val) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }

    /* For remaining elements, insert if larger than smallest in top-k */
    for (int i = k; i < n; i++) {
        int last = (k < n) ? k - 1 : n - 1;
        if (vals[i] > vals[indices[last]]) {
            indices[last] = i;
            /* Bubble up */
            int j = last - 1;
            while (j >= 0 && vals[indices[j]] < vals[indices[j + 1]]) {
                int tmp = indices[j]; indices[j] = indices[j + 1]; indices[j + 1] = tmp;
                j--;
            }
        }
    }
}

/* ─── Repetition penalty state ─── */
#define LLM_REP_WINDOW  64   /* track last N tokens */
#define LLM_REP_PENALTY 1.15f  /* multiplicative penalty for repeated tokens */
#define LLM_FREQ_PENALTY 0.1f  /* additive penalty scaled by occurrence count */
static int   llm_rep_ring[LLM_REP_WINDOW];
static int   llm_rep_count = 0;  /* total tokens pushed */

static void llm_rep_reset(void) { llm_rep_count = 0; kmemset(llm_rep_ring, 0, sizeof(llm_rep_ring)); }
static void llm_rep_push(int token) { llm_rep_ring[llm_rep_count % LLM_REP_WINDOW] = token; llm_rep_count++; }

/* ─── Thinking mode state ─── */
#define LLM_THINK_TOKEN    98   /* <|think|> — toggle start/end thinking */
#define LLM_CHANNEL_START 100   /* <|channel> — Gemma 4 thinking start */
#define LLM_CHANNEL_END   101   /* <channel|> — Gemma 4 thinking end */
static int  llm_think_enabled   = 1;   /* 0=off, 1=reactive, 2=force-inject */
static int  llm_think_show      = 0;   /* 1 = include thinking text in output */
static int  llm_think_count     = 0;   /* thinking tokens in last generation */
static int  llm_think_active    = 0;   /* currently inside thinking block */
static int  llm_sample_top_k    = 40;
static float llm_sample_top_p   = 0.9f;
static float llm_default_temperature = 0.7f; /* used by llm_prompt_n / llm_prompt_tokens */
static int  llm_attnres_enabled = 0;
static float llm_attnres_strength = 0.35f;
static int  llm_depth_attn_enabled = 0;
static float llm_depth_attn_strength = 0.55f;
static int  llm_depth_attn_window = 16;

/* Fast sampler RNG: seeded once from crypto_random, then xorshift32 per token. */
static uint32_t llm_sample_rng_state = 0;

static uint32_t llm_sample_rand_u32(void)
{
    if (llm_sample_rng_state == 0) {
        uint32_t seed = 0;
        crypto_random(&seed, sizeof(seed));
        if (seed == 0) {
            seed = (uint32_t)rdtsc_fenced();
            if (seed == 0) seed = 0xA341316Cu;
        }
        llm_sample_rng_state = seed;
    }

    /* xorshift32 */
    uint32_t x = llm_sample_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    llm_sample_rng_state = x;
    return x;
}

/* Optional override for GPU KV-cache context window.  0 = use model default. */
static int llm_max_ctx_override = 0;
void llm_set_max_ctx(int n) { llm_max_ctx_override = n; }
void llm_set_temperature(float t) { if (t >= 0.0f && t <= 2.0f) llm_default_temperature = t; }

void llm_set_thinking(int enable)      { llm_think_enabled = enable; }
void llm_set_show_thinking(int show)   { llm_think_show = show; }
int  llm_thinking_tokens(void)         { return llm_think_count; }
void llm_set_sampling_params(int top_k, float top_p)
{
    if (top_k > 0 && top_k <= 128) llm_sample_top_k = top_k;
    if (top_p > 0.0f && top_p <= 1.0f) llm_sample_top_p = top_p;
}

void llm_set_attention_residuals(int enable, float strength)
{
    llm_attnres_enabled = enable ? 1 : 0;
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    llm_attnres_strength = strength;
}

void llm_set_depth_residual_attention(int enable, float strength, int window)
{
    llm_depth_attn_enabled = enable ? 1 : 0;
    if (strength < 0.0f) strength = 0.0f;
    if (strength > 1.0f) strength = 1.0f;
    llm_depth_attn_strength = strength;
    if (window < 2) window = 2;
    if (window > 64) window = 64;
    llm_depth_attn_window = window;
}

/* Apply repetition + frequency penalty to logits in-place */
static void llm_apply_rep_penalty(float *logits, int vocab_size)
{
    int window = llm_rep_count < LLM_REP_WINDOW ? llm_rep_count : LLM_REP_WINDOW;
    if (window == 0) return;

    /* Count occurrences of each token in the window */
    for (int i = 0; i < window; i++) {
        int tok = llm_rep_ring[(llm_rep_count - 1 - i) % LLM_REP_WINDOW];
        if (tok < 0 || tok >= vocab_size) continue;

        /* Multiplicative penalty: shrink logit toward zero */
        if (logits[tok] > 0)
            logits[tok] /= LLM_REP_PENALTY;
        else
            logits[tok] *= LLM_REP_PENALTY;

        /* Additive frequency penalty based on recency */
        logits[tok] -= LLM_FREQ_PENALTY;
    }
}

/* Top-k + top-p (nucleus) sampling with temperature + repetition penalty */
static int llm_sample(float *logits, int vocab_size, float temperature)
{
    /* True greedy path: do not mutate logits with repetition penalties. */
    if (temperature <= 0.001f)
        return llm_argmax(logits, vocab_size);

    /* Sampling path applies repetition/frequency penalties. */
    llm_apply_rep_penalty(logits, vocab_size);

    /* Top-k filtering: keep only top 40 candidates */
    #define LLM_TOP_K_MAX 128
    static int top_indices[LLM_TOP_K_MAX];
    static float top_probs[LLM_TOP_K_MAX];

    int k = llm_sample_top_k;
    if (k < 1) k = 1;
    if (k > LLM_TOP_K_MAX) k = LLM_TOP_K_MAX;
    if (k > vocab_size) k = vocab_size;

    llm_partial_sort_desc(top_indices, logits, vocab_size, k);

    /* Apply temperature and compute softmax over top-k */
    float max_val = logits[top_indices[0]];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        top_probs[i] = llm_expf((logits[top_indices[i]] - max_val) / temperature);
        sum += top_probs[i];
    }
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int i = 0; i < k; i++)
        top_probs[i] *= inv_sum;

    /* Top-p (nucleus) filtering: keep tokens until cumulative prob >= 0.9 */
    float top_p = llm_sample_top_p;
    float cum = 0.0f;
    int p_cutoff = k;
    for (int i = 0; i < k; i++) {
        cum += top_probs[i];
        if (cum >= top_p) {
            p_cutoff = i + 1;
            break;
        }
    }

    /* Re-normalize after top-p cutoff */
    if (p_cutoff < k) {
        sum = 0.0f;
        for (int i = 0; i < p_cutoff; i++)
            sum += top_probs[i];
        inv_sum = 1.0f / (sum + 1e-10f);
        for (int i = 0; i < p_cutoff; i++)
            top_probs[i] *= inv_sum;
    }

    /* Sample using fast local RNG (seeded once from CSPRNG). */
    float r = (float)(llm_sample_rand_u32() >> 8) * (1.0f / 16777216.0f);

    float cdf = 0.0f;
    for (int i = 0; i < p_cutoff; i++) {
        cdf += top_probs[i];
        if (cdf >= r) return top_indices[i];
    }
    return top_indices[p_cutoff - 1];
    #undef LLM_TOP_K_MAX
}

/* Generate text tokens autoregressively.
 * prompt_tokens: input token IDs (n_prompt tokens)
 * output_text: buffer for generated text
 * max_gen: maximum tokens to generate
 * temperature: sampling temperature (0.0 = greedy)
 * continue_cache: if nonzero, don't reset KV cache (multi-turn)
 * Returns: number of generated tokens */
static int llm_generate(llm_model_t *m, const int *prompt_tokens, int n_prompt,
                        char *output_text, int max_text_len,
                        int max_gen, float temperature, int continue_cache)
{
    int start_pos = 0;

    if (!continue_cache) {
        /* Logical reset only: attention reads are bounded by seq_len (pos+1),
         * so stale KV values outside active prefix are never consumed. */
        m->cache_len = 0;
        llm_rep_reset();
    } else {
        start_pos = m->cache_len;
    }

    int out_pos = 0;
    int gen_count = 0;
    output_text[0] = '\0';

    /* Reset thinking state for this generation */
    llm_think_count  = 0;
    llm_think_active = 0;

    /* Seed repetition window with prompt tokens */
    for (int i = 0; i < n_prompt; i++)
        llm_rep_push(prompt_tokens[i]);

    /* Process prompt tokens (prefill) */
    uint64_t t_prefill_start = rdtsc_fenced();
    int n_prefill = n_prompt;
    int max_prefill = (m->max_seq - 1) - start_pos;
    if (n_prefill > max_prefill) n_prefill = max_prefill;
    if (n_prefill < 0) n_prefill = 0;

#ifdef ENABLE_CUDA
    int did_batch_prefill = 0;
    /* Skip batch-prefill when any layer is CPU-offloaded: the batch GPU path
     * doesn't handle mixed CPU/GPU layers and will crash the CUDA context. */
    int all_layers_on_gpu = 1;
    for (int _L = 0; _L < m->n_layers; _L++) {
        if (!gpu_ctx.layers[_L].on_gpu) { all_layers_on_gpu = 0; break; }
    }
    if (gpu_ctx.gpu_fwd && n_prefill >= 2 &&
        !llm_attnres_enabled && !llm_depth_attn_enabled &&
        !gpu_ffn_skip_for_compress &&
        all_layers_on_gpu &&
        axex_manifold_compressed_count() == 0) {  /* skip when GP active (raw weights freed) */
        /* Batch fills KV for tokens 0..n_prefill-2; caller runs last token
         * sequentially for numerically-exact logits (avoids Q8-quant accumulation). */
        if (gpu_ctx.prefill_max_n < n_prefill - 1)
            llm_ensure_prefill_scratch(m, n_prefill - 1);
        if (gpu_ctx.prefill_max_n >= n_prefill - 1 && gpu_ctx.d_pfx && n_prefill >= 3) {
            llm_forward_prefill_batch_gpu(m, prompt_tokens, n_prefill - 1, start_pos);
            /* Last token: sequential for exact logits */
            llm_forward_token(m, llm_logits,
                              prompt_tokens[n_prefill - 1],
                              start_pos + n_prefill - 1);
            did_batch_prefill = 1;
        }
    }
    if (!did_batch_prefill)
#endif
    {
        for (int i = 0; i < n_prefill; i++) {
            float *prefill_logits = (i == n_prefill - 1) ? llm_logits : NULL;
            llm_forward_token(m, prefill_logits, prompt_tokens[i], start_pos + i);
        }
    }
    uint64_t t_prefill_end = rdtsc_fenced();

#ifdef ENABLE_CUDA
    int use_gpu_greedy = 0;
    /* Enable capture-once CUDA Graphs for decode tokens.
     * Graph captures the layer loop + rmsnorm + lmhead on first decode,
     * then replays with updated device-side pos/seq_len on subsequent tokens. */
    extern int cuda_graph_decode_ready;
    extern int cuda_graph_captured;
    extern int cuda_graph_tried;
    extern int gpu_skip_logits_download;
    int allow_graph_capture = (max_gen >= 8);
    /* Mixed CPU/GPU layers: can't CUDA-graph across CPU calls */
    if (allow_graph_capture) {
        for (int _L = 0; _L < m->n_layers; _L++) {
            if (!gpu_ctx.layers[_L].on_gpu) { allow_graph_capture = 0; break; }
        }
    }
    cuda_graph_decode_ready = allow_graph_capture ? 1 : 0;
    cuda_graph_captured = 0;
    cuda_graph_tried = 0;
    /* Greedy decode can consume GPU-side argmax and skip replay logits D2H. */
    use_gpu_greedy = allow_graph_capture && (temperature <= 0.001f && gpu_ctx.gpu_fwd);
    gpu_skip_logits_download = use_gpu_greedy ? 1 : 0;
#endif

    /* Generate new tokens */
    int last_token = (n_prompt > 0) ? prompt_tokens[n_prompt - 1] : m->bos_id;
    int pos = start_pos + n_prompt;

    /* Get next token from the last forward pass */
    {
        int dbg_k = (m->vocab_size < 5) ? m->vocab_size : 5;
        int dbg_idx[5] = {0, 0, 0, 0, 0};
        float eos_logit = 0.0f;
        if (dbg_k > 0)
            llm_partial_sort_desc(dbg_idx, llm_logits, m->vocab_size, dbg_k);
        if (m->eos_id >= 0 && m->eos_id < m->vocab_size)
            eos_logit = llm_logits[m->eos_id];
        if (dbg_k >= 3 && llm_emit_benchmark_debug()) {
            kprintf("[GEN-DBG] first-step eos=%d logit=%f top=%d(%f) %d(%f) %d(%f)\n",
                    m->eos_id, (double)eos_logit,
                    dbg_idx[0], (double)llm_logits[dbg_idx[0]],
                    dbg_idx[1], (double)llm_logits[dbg_idx[1]],
                    dbg_idx[2], (double)llm_logits[dbg_idx[2]]);
        }
    }
    int next = llm_sample(llm_logits, m->vocab_size, temperature);
    LOG_DBG("GEN", "first sampled token: %d (eos=%d)", next, m->eos_id);

    uint64_t t_gen_start = rdtsc_fenced();
    int consec_nl = 0;  /* consecutive newline counter */
    /* Gemma4 thinking state machine (uses proper tok IDs in the vocab):
     *  tok=106 = <start_of_turn>  tok=107 = <end_of_turn> (also '\n'!)
     *  Mode 0 = pre/direct  1 = thinking(suppress)  5 = tentative-end(saw \n in thinking)
     *  Mode 2 = inter-turn  3 = model-header(suppress to \n)  4 = response(emit)
     *  Key insight: tok=107='\n' appears INSIDE thinking content too, so we need
     *  lookahead: thinking only truly ends when tok=107 is FOLLOWED by tok=106.
     */
    int g4_mode = 0;

    for (int g = 0; g < max_gen && pos < m->max_seq; g++) {
        if (g < 6 && llm_emit_benchmark_debug())
            kprintf("[GEN-DBG] step=%d next=%d eos=%d\n", g, next, m->eos_id);
        /* Check for EOS BEFORE decoding its text */
        if (next == m->eos_id) {
            if (llm_emit_benchmark_debug())
                kprintf("[GEN-DBG] stop=eos step=%d tok=%d\n", g, next);
            break;
        }

        /* Gemma4 state machine */
        if (m->is_gemma4) {
            if (g4_mode == 1) {  /* thinking block: suppress */
                llm_forward_token(m, llm_logits, next, pos++);
                if (next == 107) {
                    g4_mode = 5;  /* tentative thinking-end (saw \n) */
                }
                next = llm_sample(llm_logits, m->vocab_size, temperature);
                continue;
            }
            if (g4_mode == 5) {  /* tentative end: confirm if followed by tok=106 */
                if (next == 106) {
                    /* Confirmed end of thinking — skip header "model\n" */
                    llm_forward_token(m, llm_logits, next, pos++);
                    g4_mode = 3;
                } else if (next == 107) {
                    /* Another newline — still inter-turn or multi-\n */
                    llm_forward_token(m, llm_logits, next, pos++);
                    /* stay in mode=5 */
                } else {
                    /* False alarm: the \n was content, resume thinking */
                    llm_forward_token(m, llm_logits, next, pos++);
                    g4_mode = 1;
                }
                next = llm_sample(llm_logits, m->vocab_size, temperature);
                continue;
            }
            if (g4_mode == 3) {  /* model-turn header: suppress until \n */
                char hbuf[32];
                int hlen = llm_decode_token(m, next, hbuf, sizeof(hbuf));
                int has_nl = 0;
                for (int i = 0; i < hlen; i++) if (hbuf[i] == '\n') has_nl = 1;
                llm_forward_token(m, llm_logits, next, pos++);
                if (has_nl) g4_mode = 4;
                next = llm_sample(llm_logits, m->vocab_size, temperature);
                continue;
            }
            if (g4_mode == 4) {  /* actual response: emit until tok=107 */
                if (next == 107 || next == 106) { llm_forward_token(m, llm_logits, next, pos++); break; }
                /* fall through to normal emit */
            }
            /* g4_mode == 0 */
            if (g4_mode == 0) {
                if (next == 107) { llm_forward_token(m, llm_logits, next, pos++); break; }
                if (next == 106) {  /* <start_of_turn> = thinking block start */
#ifdef ENABLE_CUDA
                    use_gpu_greedy = 0; gpu_skip_logits_download = 0; cuda_graph_decode_ready = 0;
#endif
                    llm_forward_token(m, llm_logits, next, pos++);
                    g4_mode = 1;
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                    continue;
                }
                /* direct response (no thinking) — fall through to normal emit */
            }
        } else {
            /* Non-Gemma models should only stop on EOS/stop-sequence checks.
             * Token IDs 106/107 are Gemma-specific control IDs and can be
             * ordinary tokens in other vocabularies. */
        }

        /* ── Thinking mode: detect <|think|> toggle (token 98) ────────── */
        if (next == LLM_THINK_TOKEN && llm_think_enabled) {
            if (!llm_think_active) {
                /* Entering thinking block */
                llm_think_active = 1;
                LOG_INFO("THINK", "thinking started at token %d", gen_count);
                if (llm_think_show) {
                    const char *tag = "<think>\n";
                    int tl = 8;
                    for (int i = 0; i < tl && out_pos < max_text_len - 1; i++)
                        output_text[out_pos++] = tag[i];
                    output_text[out_pos] = '\0';
                    if (llm_stream_cb)
                        llm_stream_cb(tag, tl, llm_stream_cb_ud);
                }
            } else {
                /* Exiting thinking block */
                llm_think_active = 0;
                LOG_INFO("THINK", "thinking ended: %d thinking tokens", llm_think_count);
                if (llm_think_show) {
                    const char *tag = "\n</think>\n";
                    int tl = 10;
                    for (int i = 0; i < tl && out_pos < max_text_len - 1; i++)
                        output_text[out_pos++] = tag[i];
                    output_text[out_pos] = '\0';
                    if (llm_stream_cb)
                        llm_stream_cb(tag, tl, llm_stream_cb_ud);
                }
            }
            /* Forward the think token through the transformer (keeps KV cache consistent) */
            llm_forward_token(m, llm_logits, next, pos);
            pos++;
            gen_count++;
            llm_rep_push(next);
#ifdef ENABLE_CUDA
            if (use_gpu_greedy && cuda_graph_captured)
            {
                int gpu_next = cuda_argmax((const float *)gpu_ctx.d_out, m->vocab_size);
                if (gpu_next >= 0 && gpu_next < m->vocab_size) next = gpu_next;
                else {
                    use_gpu_greedy = 0;
                    gpu_skip_logits_download = 0;
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                }
            }
            else
#endif
            next = llm_sample(llm_logits, m->vocab_size, temperature);
            continue;
        }

        /* ── Gemma channel thinking: <|channel> (100) starts, <channel|> (101) ends ── */
        if (next == LLM_CHANNEL_START && llm_think_enabled && !llm_think_active) {
            llm_think_active = 1;
            LOG_INFO("THINK", "channel thinking started at token %d", gen_count);
            if (llm_think_show) {
                const char *tag = "<think>\n";
                int tl = 8;
                for (int i = 0; i < tl && out_pos < max_text_len - 1; i++)
                    output_text[out_pos++] = tag[i];
                output_text[out_pos] = '\0';
                if (llm_stream_cb)
                    llm_stream_cb(tag, tl, llm_stream_cb_ud);
            }
            llm_forward_token(m, llm_logits, next, pos);
            pos++;
            gen_count++;
            llm_rep_push(next);
#ifdef ENABLE_CUDA
            if (use_gpu_greedy && cuda_graph_captured)
            {
                int gpu_next = cuda_argmax((const float *)gpu_ctx.d_out, m->vocab_size);
                if (gpu_next >= 0 && gpu_next < m->vocab_size) next = gpu_next;
                else {
                    use_gpu_greedy = 0;
                    gpu_skip_logits_download = 0;
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                }
            }
            else
#endif
            next = llm_sample(llm_logits, m->vocab_size, temperature);
            continue;
        }
        if (next == LLM_CHANNEL_END && llm_think_active) {
            llm_think_active = 0;
            LOG_INFO("THINK", "channel thinking ended: %d thinking tokens", llm_think_count);
            if (llm_think_show) {
                const char *tag = "\n</think>\n";
                int tl = 10;
                for (int i = 0; i < tl && out_pos < max_text_len - 1; i++)
                    output_text[out_pos++] = tag[i];
                output_text[out_pos] = '\0';
                if (llm_stream_cb)
                    llm_stream_cb(tag, tl, llm_stream_cb_ud);
            }
            llm_forward_token(m, llm_logits, next, pos);
            pos++;
            gen_count++;
            llm_rep_push(next);
#ifdef ENABLE_CUDA
            if (use_gpu_greedy && cuda_graph_captured)
            {
                int gpu_next = cuda_argmax((const float *)gpu_ctx.d_out, m->vocab_size);
                if (gpu_next >= 0 && gpu_next < m->vocab_size) next = gpu_next;
                else {
                    use_gpu_greedy = 0;
                    gpu_skip_logits_download = 0;
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                }
            }
            else
#endif
            next = llm_sample(llm_logits, m->vocab_size, temperature);
            continue;
        }

        /* ── Inside thinking: pure token-space operation ──────────────── */
        if (llm_think_active) {
            /* Skip detokenization entirely — just forward and sample */
            llm_think_count++;
            if (llm_think_show) {
                /* Optionally detokenize for display */
                char tok_buf[128];
                int tok_len = llm_decode_token(m, next, tok_buf, sizeof(tok_buf));
                for (int i = 0; i < tok_len && out_pos < max_text_len - 1; i++)
                    output_text[out_pos++] = tok_buf[i];
                output_text[out_pos] = '\0';
                if (llm_stream_cb && tok_len > 0) {
                    char stream_buf[128];
                    int si = 0;
                    for (int i = 0; i < tok_len; ) {
                        unsigned char c = (unsigned char)tok_buf[i];
                        if (c == 0xE2 && i + 2 < tok_len &&
                            (unsigned char)tok_buf[i+1] == 0x96 &&
                            (unsigned char)tok_buf[i+2] == 0x81) {
                            stream_buf[si++] = ' ';
                            i += 3;
                        } else {
                            stream_buf[si++] = tok_buf[i++];
                        }
                    }
                    if (si > 0) llm_stream_cb(stream_buf, si, llm_stream_cb_ud);
                }
            }
            LOG_TRC("THINK", "think tok %d: id=%d", llm_think_count, next);
            llm_forward_token(m, llm_logits, next, pos);
            pos++;
            gen_count++;
            llm_rep_push(next);
#ifdef ENABLE_CUDA
            if (use_gpu_greedy && cuda_graph_captured)
            {
                int gpu_next = cuda_argmax((const float *)gpu_ctx.d_out, m->vocab_size);
                if (gpu_next >= 0 && gpu_next < m->vocab_size) next = gpu_next;
                else {
                    use_gpu_greedy = 0;
                    gpu_skip_logits_download = 0;
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                }
            }
            else
#endif
            next = llm_sample(llm_logits, m->vocab_size, temperature);
            continue;
        }

        /* ── Normal token: decode to text ─────────────────────────────── */
        char tok_buf[128];
        int tok_len = llm_decode_token(m, next, tok_buf, sizeof(tok_buf));
        for (int i = 0; i < tok_len && out_pos < max_text_len - 1; i++)
            output_text[out_pos++] = tok_buf[i];
        output_text[out_pos] = '\0';
        gen_count++;

        /* Streaming callback: emit token, replacing UTF-8 ▁ (E2 96 81) with space */
        if (llm_stream_cb) {
            char stream_buf[128];
            int si = 0;
            for (int i = 0; i < tok_len; ) {
                unsigned char c = (unsigned char)tok_buf[i];
                if (c == 0xE2 && i + 2 < tok_len &&
                    (unsigned char)tok_buf[i+1] == 0x96 &&
                    (unsigned char)tok_buf[i+2] == 0x81) {
                    stream_buf[si++] = ' ';
                    i += 3;
                } else {
                    stream_buf[si++] = tok_buf[i++];
                }
            }
            if (si > 0) llm_stream_cb(stream_buf, si, llm_stream_cb_ud);
        }

        /* Check stop sequences as suffixes (O(1) per token) instead of
         * scanning the whole output buffer each step. */
        if (out_pos >= 4) {
            static const char *stop_seqs[] = {
                "<|im_end|>", "<|endoftext|>", "<|end|>",
                "<|eot_id|>", "<|end_of_turn|>", NULL
            };
            int stop = 0;
            for (int si = 0; stop_seqs[si] && !stop; si++) {
                if (llm_ends_with(output_text, out_pos, stop_seqs[si])) {
                    int trunc_at = out_pos - (int)kstrlen(stop_seqs[si]);
                    if (trunc_at < 0) trunc_at = 0;
                    output_text[trunc_at] = '\0';
                    out_pos = trunc_at;
                    stop = 1;
                }
            }
            if (stop) break;
        }

        /* Track consecutive newlines — stop after 3 in a row (model is done) */
        {
            int has_nl = 0;
            for (int i = 0; i < tok_len; i++)
                if (tok_buf[i] == '\n') has_nl = 1;
            if (has_nl) consec_nl++;
            else consec_nl = 0;
            if (consec_nl >= 3) {
                if (llm_emit_benchmark_debug())
                    kprintf("[GEN-DBG] stop=consecutive_newlines step=%d\n", g);
                break;
            }
        }

        /* Forward the new token */
        llm_forward_token(m, llm_logits, next, pos);
        pos++;

        /* Sample next token */
        last_token = next;
        llm_rep_push(next);
#ifdef ENABLE_CUDA
        if (use_gpu_greedy && cuda_graph_captured)
        {
            int gpu_next = cuda_argmax((const float *)gpu_ctx.d_out, m->vocab_size);
            if (gpu_next >= 0 && gpu_next < m->vocab_size) next = gpu_next;
            else {
                use_gpu_greedy = 0;
                gpu_skip_logits_download = 0;
                next = llm_sample(llm_logits, m->vocab_size, temperature);
            }
        }
        else
#endif
        next = llm_sample(llm_logits, m->vocab_size, temperature);
    }
    uint64_t t_gen_end = rdtsc_fenced();

    /* Final cleanup: strip any known stop sequence artifacts from output */
    {
        static const char *stop_seqs[] = {
            "<|im_end|>", "<|endoftext|>", "<|end|>",
            "<|eot_id|>", "<|end_of_turn|>", NULL
        };
        for (int si = 0; stop_seqs[si]; si++) {
            char *found = llm_strstr(output_text, stop_seqs[si]);
            if (found) {
                *found = '\0';
                out_pos = (int)(found - output_text);
                break;
            }
        }
        /* Trim trailing whitespace */
        while (out_pos > 0 && (output_text[out_pos-1] == ' ' || output_text[out_pos-1] == '\n'))
            output_text[--out_pos] = '\0';
    }

    /* Update cache length for multi-turn */
    m->cache_len = pos;

    /* Print timing stats */
    if (gen_count > 0) {
        uint64_t prefill_us = perf_cycles_to_us(t_prefill_end - t_prefill_start);
        uint64_t gen_us = perf_cycles_to_us(t_gen_end - t_gen_start);
        float tok_per_s = gen_us > 0 ? (float)gen_count * 1000000.0f / (float)gen_us : 0.0f;
        llm_last_prefill_ms_val  = (float)(prefill_us / 1000);
        llm_last_tok_per_sec_val = tok_per_s;
        int visible_tok = gen_count - llm_think_count;
        if (llm_think_count > 0) {
            kprintf("\n[%d tok (%d think + %d visible), %.1f tok/s, prefill %lu ms, %d cpus]\n",
                    gen_count, llm_think_count, visible_tok, tok_per_s,
                    (unsigned long)(prefill_us / 1000),
                    smp.ap_started + 1);
        } else {
            kprintf("\n[%d tok, %.1f tok/s, prefill %lu ms, %d cpus]\n",
                    gen_count, tok_per_s,
                    (unsigned long)(prefill_us / 1000),
                    smp.ap_started + 1);
        }
        LOG_DBG("GEN", "generation complete: %d tokens, %.1f tok/s, "
                "prefill=%lu us, gen=%lu us, think=%d",
                gen_count, tok_per_s,
                (unsigned long)prefill_us, (unsigned long)gen_us,
                llm_think_count);
    }

    return gen_count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Token-only generation: same as llm_generate but returns token IDs
 * instead of decoded text.  No text serialization boundary.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int llm_generate_token_ids(llm_model_t *m, const int *prompt_tokens, int n_prompt,
                                  int *output_tokens, int max_output_tokens,
                                  int max_gen, float temperature, int continue_cache)
{
    int start_pos = 0;
    int emitted_count = 0;
    int total_generated = 0;

    if (!continue_cache) {
        /* Same logical reset as text path: avoid expensive physical KV clears. */
        m->cache_len = 0;
        llm_rep_reset();
    } else {
        start_pos = m->cache_len;
    }

    llm_think_count = 0;
    llm_think_active = 0;

    for (int i = 0; i < n_prompt; i++)
        llm_rep_push(prompt_tokens[i]);



    {
        uint64_t t_prefill_start = rdtsc_fenced();
        int n_prefill = n_prompt;
        int max_prefill = (m->max_seq - 1) - start_pos;
        if (n_prefill > max_prefill) n_prefill = max_prefill;
        if (n_prefill < 0) n_prefill = 0;
#ifdef ENABLE_CUDA
        int did_batch_pf = 0;
        int all_layers_on_gpu_2 = 1;
        for (int _L = 0; _L < m->n_layers; _L++) {
            if (!gpu_ctx.layers[_L].on_gpu) { all_layers_on_gpu_2 = 0; break; }
        }
        if (gpu_ctx.gpu_fwd && n_prefill >= 2 &&
            !llm_attnres_enabled && !llm_depth_attn_enabled &&
            !gpu_ffn_skip_for_compress &&
            all_layers_on_gpu_2 &&
            axex_manifold_compressed_count() == 0) {  /* skip when GP active (raw weights freed) */
            if (gpu_ctx.prefill_max_n < n_prefill - 1)
                llm_ensure_prefill_scratch(m, n_prefill - 1);
            if (gpu_ctx.prefill_max_n >= n_prefill - 1 && gpu_ctx.d_pfx && n_prefill >= 3) {
                /* Batch n-1 tokens, then one sequential for exact logits */
                llm_forward_prefill_batch_gpu(m, prompt_tokens, n_prefill - 1, start_pos);
                llm_forward_token(m, llm_logits, prompt_tokens[n_prefill - 1],
                                  start_pos + n_prefill - 1);
                did_batch_pf = 1;
            }
        }
        if (!did_batch_pf)
#endif
        {
            for (int i = 0; i < n_prefill; i++) {
                float *prefill_logits = (i == n_prefill - 1) ? llm_logits : NULL;
                llm_forward_token(m, prefill_logits, prompt_tokens[i], start_pos + i);
            }
        }
        uint64_t t_prefill_end = rdtsc_fenced();


#ifdef ENABLE_CUDA
        /* Enable CUDA Graphs for decode tokens (same as llm_generate) */
        extern int cuda_graph_decode_ready;
        extern int cuda_graph_captured;
        extern int cuda_graph_tried;
        extern int gpu_skip_logits_download;
    int allow_graph_capture = (max_gen >= 8);
    /* Mixed CPU/GPU layers: can't CUDA-graph across CPU calls */
    if (allow_graph_capture) {
        for (int _L = 0; _L < m->n_layers; _L++) {
            if (!gpu_ctx.layers[_L].on_gpu) { allow_graph_capture = 0; break; }
        }
    }
    cuda_graph_decode_ready = allow_graph_capture ? 1 : 0;
        cuda_graph_captured = 0;
        cuda_graph_tried = 0;
        /* Enable GPU-side argmax for greedy decoding (skip 1MB D2H) */
    int use_gpu_greedy = allow_graph_capture && (temperature == 0.0f && gpu_ctx.gpu_fwd);
        gpu_skip_logits_download = use_gpu_greedy ? 1 : 0;
#endif

        int pos = start_pos + n_prompt;
        int next = llm_sample(llm_logits, m->vocab_size, temperature);
        uint64_t t_gen_start = rdtsc_fenced();

        /* Gemma4 thinking state machine (tok=106=<start_of_turn>, tok=107=<end_of_turn>/\n):
         *  Mode 0=pre/direct  1=thinking(suppress)  5=tentative-end(saw \n in thinking)
         *  Mode 3=model-header(suppress \n)  4=response(emit)
         *  Thinking ends only when tok=107(\n) is FOLLOWED by tok=106 (lookahead).
         */
        int g4_mode = 0;

        for (int g = 0; g < max_gen && pos < m->max_seq && emitted_count < max_output_tokens; g++) {
            if (next == m->eos_id) break;

            if (m->is_gemma4) {
                if (g4_mode == 1) {  /* thinking: suppress all */
                    llm_forward_token(m, llm_logits, next, pos++);
                    total_generated++;
                    if (next == 107) g4_mode = 5;  /* tentative thinking end */
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                    continue;
                }
                if (g4_mode == 5) {  /* tentative end: confirmed by tok=106, else resume */
                    llm_forward_token(m, llm_logits, next, pos++);
                    total_generated++;
                    if (next == 106) {
                        g4_mode = 3;  /* model-turn header */
                    } else if (next == 107) {
                        /* multiple newlines — stay in tentative */
                    } else {
                        g4_mode = 1;  /* resume thinking */
                    }
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                    continue;
                }
                if (g4_mode == 3) {  /* model-header: forward until \n then emit */
                    char hbuf[32];
                    int hlen = llm_decode_token(m, next, hbuf, sizeof(hbuf));
                    int has_nl = 0;
                    for (int i = 0; i < hlen; i++) if (hbuf[i] == '\n') has_nl = 1;
                    llm_forward_token(m, llm_logits, next, pos++);
                    total_generated++;
                    if (has_nl) g4_mode = 4;
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                    continue;
                }
                if (g4_mode == 4) {  /* response: emit until tok=107 or tok=106 */
                    if (next == 107 || next == 106) {
                        llm_forward_token(m, llm_logits, next, pos++);
                        total_generated++;
                        break;
                    }
                    /* fall through to normal emit */
                }
                if (g4_mode == 0) {  /* initial: detect thinking or emit directly */
                    if (next == 107) {
                        llm_forward_token(m, llm_logits, next, pos++);
                        total_generated++;
                        break;
                    }
                    if (next == 106) {  /* thinking block start */
#ifdef ENABLE_CUDA
                        use_gpu_greedy = 0; gpu_skip_logits_download = 0; cuda_graph_decode_ready = 0;
#endif
                        llm_forward_token(m, llm_logits, next, pos++);
                        total_generated++;
                        g4_mode = 1;
                        next = llm_sample(llm_logits, m->vocab_size, temperature);
                        continue;
                    }
                    /* direct response — fall through to normal emit */
                }
            } else {
                if (next == 106 || next == 107) {
                    llm_forward_token(m, llm_logits, next, pos++);
                    total_generated++;
                    break;
                }
            }

            if (next == LLM_THINK_TOKEN && llm_think_enabled) {
                llm_think_active = !llm_think_active;
                llm_forward_token(m, llm_logits, next, pos);
                pos++;
                total_generated++;
                llm_rep_push(next);
                next = llm_sample(llm_logits, m->vocab_size, temperature);
                continue;
            }

            if (next == LLM_CHANNEL_START && llm_think_enabled && !llm_think_active) {
                llm_think_active = 1;
                llm_forward_token(m, llm_logits, next, pos);
                pos++;
                total_generated++;
                llm_rep_push(next);
                next = llm_sample(llm_logits, m->vocab_size, temperature);
                continue;
            }

            if (next == LLM_CHANNEL_END && llm_think_active) {
                llm_think_active = 0;
                llm_forward_token(m, llm_logits, next, pos);
                pos++;
                total_generated++;
                llm_rep_push(next);
                next = llm_sample(llm_logits, m->vocab_size, temperature);
                continue;
            }

            if (llm_think_active) {
                llm_think_count++;
                llm_forward_token(m, llm_logits, next, pos);
                pos++;
                total_generated++;
                llm_rep_push(next);
                next = llm_sample(llm_logits, m->vocab_size, temperature);
                continue;
            }

            output_tokens[emitted_count++] = next;
            total_generated++;

            llm_forward_token(m, llm_logits, next, pos);
            pos++;
            llm_rep_push(next);
#ifdef ENABLE_CUDA
            if (use_gpu_greedy && cuda_graph_captured)
            {
                int gpu_next = cuda_argmax((const float *)gpu_ctx.d_out, m->vocab_size);
                if (gpu_next >= 0 && gpu_next < m->vocab_size) next = gpu_next;
                else {
                    use_gpu_greedy = 0;
                    gpu_skip_logits_download = 0;
                    next = llm_sample(llm_logits, m->vocab_size, temperature);
                }
            }
            else
#endif
            next = llm_sample(llm_logits, m->vocab_size, temperature);
        }

        m->cache_len = pos;

        if (total_generated > 0) {
            uint64_t prefill_us = perf_cycles_to_us(t_prefill_end - t_prefill_start);
            uint64_t gen_us = perf_cycles_to_us(rdtsc_fenced() - t_gen_start);
            float tok_per_s = gen_us > 0 ? (float)total_generated * 1000000.0f / (float)gen_us : 0.0f;
            llm_last_prefill_ms_val = (float)(prefill_us / 1000);
            llm_last_tok_per_sec_val = tok_per_s;
        }
    }

    return emitted_count;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  GGUF Model Loading and Tensor Mapping                                      */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Map GGUF tensor names to model layer pointers */
static int llm_map_tensors(llm_model_t *m, gguf_ctx_t *ctx)
{
    /* Token embeddings */
    {
        const gguf_tensor_info_t *t = gguf_find_tensor(ctx, "token_embd.weight");
        if (!t) {
            kprintf("[LLM] ERROR: token_embd.weight not found\n");
            return -1;
        }
        m->token_embd = t->data;
        m->token_embd_type = t->type;
        kprintf("[LLM] token_embd.weight: type=%d dims=[%llu,%llu]\n",
                (int)t->type, t->dims[0], t->dims[1]);
    }

    /* Output norm */
    {
        const gguf_tensor_info_t *t = gguf_find_tensor(ctx, "output_norm.weight");
        if (!t) {
            kprintf("[LLM] ERROR: output_norm.weight not found\n");
            return -1;
        }
        m->output_norm = t->data;
        m->output_norm_type = t->type;
    }

    /* Output norm bias (Phi-2 LayerNorm) */
    {
        const gguf_tensor_info_t *t = gguf_find_tensor(ctx, "output_norm.bias");
        m->output_norm_bias = t ? t->data : NULL;
    }

    /* Output / LM head — may be absent if tied to embeddings */
    {
        const gguf_tensor_info_t *t = gguf_find_tensor(ctx, "output.weight");
        if (t) {
            m->output_weight = t->data;
            m->output_type = t->type;
        } else {
            m->output_weight = NULL; /* will use token_embd */
            m->output_type = m->token_embd_type;
            kprintf("[LLM] Note: output.weight tied to token_embd\n");
        }
    }

    /* RoPE scaling factors (Longrope / Su scaling for Phi-3.5) */
    {
        const gguf_tensor_info_t *ts = gguf_find_tensor(ctx, "rope_factors_short.weight");
        const gguf_tensor_info_t *tl = gguf_find_tensor(ctx, "rope_factors_long.weight");
        m->rope_factors_short = ts ? (const float *)ts->data : (const float *)0;
        m->rope_factors_long  = tl ? (const float *)tl->data : (const float *)0;
        if (ts || tl)
            kprintf("[LLM] Longrope factors: short=%p long=%p\n",
                    (const void *)m->rope_factors_short, (const void *)m->rope_factors_long);
    }

    /* Gemma4 ISWA global weights */
    if (m->is_gemma4 && m->iswa_n_embd > 0) {
        const gguf_tensor_info_t *t;
        t = gguf_find_tensor(ctx, "per_layer_token_embd.weight");
        if (t) { m->iswa_tok_embd = t->data; m->iswa_tok_embd_type = t->type;
                 kprintf("[LLM] ISWA: per_layer_token_embd [%llu x %llu]\n", t->dims[0], t->dims[1]); }
        t = gguf_find_tensor(ctx, "per_layer_model_proj.weight");
        if (t) { m->iswa_model_proj = t->data; m->iswa_model_proj_type = t->type; }
        t = gguf_find_tensor(ctx, "per_layer_proj_norm.weight");
        if (t) { m->iswa_proj_norm = t->data; }
        t = gguf_find_tensor(ctx, "rope_freqs.weight");
        if (t) { m->rope_freqs = (const float *)t->data;
                 kprintf("[LLM] ISWA: rope_freqs [%llu]\n", t->dims[0]); }
    }

    /* Per-layer weights */
    char name_buf[128];
    for (int L = 0; L < m->n_layers; L++) {
        llm_layer_t *layer = &m->layers[L];

        #define MAP_TENSOR(field, suffix, type_field) do { \
            kstrcpy(name_buf, "blk."); \
            /* Append layer number */ \
            { \
                char num[8]; int n = L, j = 0; \
                if (n == 0) { num[j++] = '0'; } \
                else { char tmp[8]; int k = 0; while (n > 0) { tmp[k++] = '0' + (n % 10); n /= 10; } \
                       while (k > 0) num[j++] = tmp[--k]; } \
                num[j] = '\0'; \
                kstrcpy(name_buf + kstrlen(name_buf), num); \
            } \
            kstrcpy(name_buf + kstrlen(name_buf), "." suffix); \
            const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf); \
            if (t) { layer->field = t->data; layer->type_field = t->type; } \
            else { layer->field = NULL; } \
        } while(0)

        MAP_TENSOR(attn_norm,  "attn_norm.weight",   attn_norm_type);
        MAP_TENSOR(q_weight,   "attn_q.weight",      q_type);
        MAP_TENSOR(k_weight,   "attn_k.weight",      k_type);
        MAP_TENSOR(v_weight,   "attn_v.weight",      v_type);
        MAP_TENSOR(o_weight,   "attn_output.weight",  o_type);
        MAP_TENSOR(ffn_norm,   "ffn_norm.weight",     ffn_norm_type);
        MAP_TENSOR(ffn_gate,   "ffn_gate.weight",     gate_type);
        MAP_TENSOR(ffn_up,     "ffn_up.weight",       up_type);
        MAP_TENSOR(ffn_down,   "ffn_down.weight",     down_type);

        /* Fused QKV tensor (Phi-3: blk.N.attn_qkv.weight) */
        if (!layer->q_weight || !layer->k_weight || !layer->v_weight) {
            #define BUILD_BLK_NAME(suffix) do { \
                kstrcpy(name_buf, "blk."); \
                { char num[8]; int n_ = L, j_ = 0; \
                  if (n_ == 0) { num[j_++] = '0'; } \
                  else { char tmp[8]; int k_ = 0; while (n_ > 0) { tmp[k_++] = '0' + (n_ % 10); n_ /= 10; } \
                         while (k_ > 0) num[j_++] = tmp[--k_]; } \
                  num[j_] = '\0'; \
                  kstrcpy(name_buf + kstrlen(name_buf), num); } \
                kstrcpy(name_buf + kstrlen(name_buf), "." suffix); \
            } while(0)

            BUILD_BLK_NAME("attn_qkv.weight");
            const gguf_tensor_info_t *qkv = gguf_find_tensor(ctx, name_buf);
            if (qkv) {
                uint64_t row_bytes = ggml_tensor_size(qkv->type, (uint64_t)m->dim);
                const uint8_t *base = (const uint8_t *)qkv->data;
                int q_rows = m->n_heads * m->head_dim;
                int k_rows = m->n_kv_heads * m->head_dim;
                layer->q_weight = base;
                layer->k_weight = base + (uint64_t)q_rows * row_bytes;
                layer->v_weight = base + (uint64_t)(q_rows + k_rows) * row_bytes;
                layer->q_type = layer->k_type = layer->v_type = qkv->type;
            }
            #undef BUILD_BLK_NAME
        }

        /* Fused gate+up tensor (Phi-3: ffn_up.weight has 2×ff_dim rows) */
        if (!layer->ffn_gate && layer->ffn_up && m->ff_dim > 0) {
            kstrcpy(name_buf, "blk.");
            { char num[8]; int n_ = L, j_ = 0;
              if (n_ == 0) { num[j_++] = '0'; }
              else { char tmp[8]; int k_ = 0; while (n_ > 0) { tmp[k_++] = '0' + (n_ % 10); n_ /= 10; }
                     while (k_ > 0) num[j_++] = tmp[--k_]; }
              num[j_] = '\0';
              kstrcpy(name_buf + kstrlen(name_buf), num); }
            kstrcpy(name_buf + kstrlen(name_buf), ".ffn_up.weight");
            const gguf_tensor_info_t *up_t = gguf_find_tensor(ctx, name_buf);
            if (up_t && up_t->n_dims >= 2 && (int)up_t->dims[1] == 2 * m->ff_dim) {
                uint64_t row_bytes = ggml_tensor_size(up_t->type, (uint64_t)m->dim);
                const uint8_t *base = (const uint8_t *)up_t->data;
                layer->ffn_gate = base;
                layer->gate_type = up_t->type;
                layer->ffn_up = base + (uint64_t)m->ff_dim * row_bytes;
                layer->up_type = up_t->type;
            }
        }

        /* Bias tensors (Phi-2; silently NULL for architectures without bias) */
        #define MAP_BIAS(field, suffix) do { \
            kstrcpy(name_buf, "blk."); \
            { \
                char num[8]; int n = L, j = 0; \
                if (n == 0) { num[j++] = '0'; } \
                else { char tmp[8]; int k = 0; while (n > 0) { tmp[k++] = '0' + (n % 10); n /= 10; } \
                       while (k > 0) num[j++] = tmp[--k]; } \
                num[j] = '\0'; \
                kstrcpy(name_buf + kstrlen(name_buf), num); \
            } \
            kstrcpy(name_buf + kstrlen(name_buf), "." suffix); \
            const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf); \
            layer->field = t ? t->data : NULL; \
        } while(0)

        MAP_BIAS(attn_norm_bias,  "attn_norm.bias");
        MAP_BIAS(q_bias,          "attn_q.bias");
        MAP_BIAS(k_bias,          "attn_k.bias");
        MAP_BIAS(v_bias,          "attn_v.bias");
        MAP_BIAS(o_bias,          "attn_output.bias");
        MAP_BIAS(ffn_norm_bias,   "ffn_norm.bias");
        MAP_BIAS(ffn_up_bias,     "ffn_up.bias");
        MAP_BIAS(ffn_down_bias,   "ffn_down.bias");

        /* Fused QKV bias fallback (Phi-3: blk.N.attn_qkv.bias) */
        if (!layer->q_bias && !layer->k_bias && !layer->v_bias) {
            kstrcpy(name_buf, "blk.");
            { char num[8]; int n_ = L, j_ = 0;
              if (n_ == 0) { num[j_++] = '0'; }
              else { char tmp[8]; int k_ = 0; while (n_ > 0) { tmp[k_++] = '0' + (n_ % 10); n_ /= 10; }
                     while (k_ > 0) num[j_++] = tmp[--k_]; }
              num[j_] = '\0';
              kstrcpy(name_buf + kstrlen(name_buf), num); }
            kstrcpy(name_buf + kstrlen(name_buf), ".attn_qkv.bias");
            const gguf_tensor_info_t *qkv_b = gguf_find_tensor(ctx, name_buf);
            if (qkv_b) {
                const float *bb = (const float *)qkv_b->data;
                int q_dim = m->n_heads * m->head_dim;
                int k_dim = m->n_kv_heads * m->head_dim;
                layer->q_bias = bb;
                layer->k_bias = bb + q_dim;
                layer->v_bias = bb + q_dim + k_dim;
            }
        }

        #undef MAP_BIAS

        /* Gemma4: per-layer Q/K norm, post-attention norm, post-FFW norm, ISWA */
        layer->q_norm = (void *)0;
        layer->k_norm = (void *)0;
        layer->post_attn_norm = (void *)0;
        layer->post_ffw_norm = (void *)0;
        layer->iswa_inp_gate = (void *)0;
        layer->iswa_proj = (void *)0;
        layer->iswa_post_norm = (void *)0;
        layer->iswa_out_scale = (void *)0;
        layer->head_dim_layer = m->head_dim;
        layer->kv_reuse_layer = -1;
        layer->ff_dim_layer = m->ff_dim;
        if (m->is_gemma4) {
            #define BUILD_BLK_NAME(suffix) do { \
                kstrcpy(name_buf, "blk."); \
                { char num[8]; int n_ = L, j_ = 0; \
                  if (n_ == 0) { num[j_++] = '0'; } \
                  else { char tmp[8]; int k_ = 0; while (n_ > 0) { tmp[k_++] = '0' + (n_ % 10); n_ /= 10; } \
                         while (k_ > 0) num[j_++] = tmp[--k_]; } \
                  num[j_] = '\0'; \
                  kstrcpy(name_buf + kstrlen(name_buf), num); } \
                kstrcpy(name_buf + kstrlen(name_buf), "." suffix); \
            } while(0)
            /* Q/K norms: blk.N.attn_q_norm.weight, blk.N.attn_k_norm.weight */
            BUILD_BLK_NAME("attn_q_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->q_norm = t->data; } }
            BUILD_BLK_NAME("attn_k_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->k_norm = t->data; } }
            /* Post-attention and post-FFW norms */
            BUILD_BLK_NAME("post_attention_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->post_attn_norm = t->data; } }
            BUILD_BLK_NAME("post_ffw_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->post_ffw_norm = t->data; } }
            /* ISWA per-layer injection tensors */
            BUILD_BLK_NAME("inp_gate.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->iswa_inp_gate = t->data; layer->iswa_inp_gate_type = t->type; } }
            BUILD_BLK_NAME("proj.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->iswa_proj = t->data; layer->iswa_proj_type = t->type; } }
            BUILD_BLK_NAME("post_norm.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->iswa_post_norm = t->data; } }
            BUILD_BLK_NAME("layer_output_scale.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t) { layer->iswa_out_scale = t->data; } }

            /* Determine per-layer head_dim from Q weight shape */
            BUILD_BLK_NAME("attn_q.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t && t->n_dims >= 2) {
                  layer->head_dim_layer = (int)(t->dims[1]) / m->n_heads;
              } }

            /* Determine per-layer ff_dim from ffn_gate shape */
            BUILD_BLK_NAME("ffn_gate.weight");
            { const gguf_tensor_info_t *t = gguf_find_tensor(ctx, name_buf);
              if (t && t->n_dims >= 2) {
                  layer->ff_dim_layer = (int)(t->dims[1]);
              } }

            /* KV cache reuse for layers >= n_layer_kv_start */
            if (L >= m->n_layer_kv_start) {
                /* SWA layers reuse kv_start-2, full attn layers reuse kv_start-1 */
                int is_swa_layer = (layer->head_dim_layer == m->head_dim_swa);
                layer->kv_reuse_layer = m->n_layer_kv_start - (is_swa_layer ? 2 : 1);
                if (layer->kv_reuse_layer < 0) layer->kv_reuse_layer = 0;
            }

            if (L == 0 || L == 4 || L == 15)
                kprintf("[LLM] L%d: hd=%d ff=%d kv_reuse=%d q_norm=%p proj=%p scale=%p\n",
                        L, layer->head_dim_layer, layer->ff_dim_layer,
                        layer->kv_reuse_layer, layer->q_norm, layer->iswa_proj,
                        layer->iswa_out_scale);
        }

        #undef BUILD_BLK_NAME
        #undef MAP_TENSOR

        /* Validate critical tensors are mapped (after fused fallback) */
        if (L == 0) {
            if (!layer->q_weight) kprintf("[LLM] WARN: blk.0 Q weight missing\n");
            if (!layer->k_weight) kprintf("[LLM] WARN: blk.0 K weight missing\n");
            if (!layer->v_weight) kprintf("[LLM] WARN: blk.0 V weight missing\n");
            if (!layer->ffn_down) kprintf("[LLM] WARN: blk.0 ffn_down missing\n");
        }
    }

    kprintf("[LLM] Tensor mapping complete: %d layers\n", m->n_layers);
    return 0;
}

/* Load model from virtio-blk disk into model_cache memory region */
static int llm_init_parsed_model(llm_model_t *m);

static int llm_load_from_disk(llm_model_t *m)
{
#ifdef __aarch64__
    kprintf("[LLM] Disk loading not supported on ARM64 yet\n");
    return -1;
#else
    uint64_t capacity = virtio_blk_capacity();
    if (capacity == 0) {
        kprintf("[LLM] No block device or zero capacity\n");
        return -1;
    }

    /* Check if capacity is reasonable for a GGUF file (at least 1MB) */
    if (capacity < 1024 * 1024) {
        kprintf("[LLM] Disk too small for a model: %lu bytes\n", capacity);
        return -1;
    }

    /* Cap at available model cache size */
    uint64_t cache_size = tensor_mm_model_cache_max();
    if (capacity > cache_size) {
        kprintf("[LLM] Model (%lu MB) exceeds cache (%lu MB)\n",
                capacity / (1024 * 1024), cache_size / (1024 * 1024));
        return -1;
    }

    kprintf("[LLM] Loading model from disk: %lu MB\n", capacity / (1024 * 1024));

    /* Read the entire disk into model cache region */
    m->data_buf = tensor_mm_model_cache_base();
    m->data_size = capacity;

    uint64_t n_sectors = (capacity + 511) / 512;
    uint64_t chunk_sectors = 2048; /* 1MB chunks */
    uint64_t sectors_done = 0;

    uint64_t t0 = rdtsc_fenced();
    while (sectors_done < n_sectors) {
        uint64_t remain = n_sectors - sectors_done;
        uint32_t chunk = (remain > chunk_sectors) ? (uint32_t)chunk_sectors : (uint32_t)remain;
        uint8_t *dst = (uint8_t *)m->data_buf + sectors_done * 512;

        int rc = virtio_blk_read(sectors_done, chunk, dst);
        if (rc != 0) {
            kprintf("[LLM] Disk read error at sector %lu: %d\n", sectors_done, rc);
            return -1;
        }
        sectors_done += chunk;

        /* Progress every 64MB */
        if ((sectors_done * 512) % (64 * 1024 * 1024) == 0) {
            kprintf("[LLM] ... %lu MB loaded\n", (sectors_done * 512) / (1024 * 1024));
        }
    }
    uint64_t t1 = rdtsc_fenced();

    uint64_t load_ms = perf_cycles_to_us(t1 - t0) / 1000;
    uint64_t mbps = (load_ms > 0) ? (capacity / 1024 * 1000 / load_ms) : 0;
    kprintf("[LLM] Loaded %lu MB in %lu ms (%lu KB/s)\n",
            capacity / (1024 * 1024), load_ms, mbps);

    return llm_init_parsed_model(m);
#endif /* __aarch64__ */
}

/* Hosted-mode loader: load model from a pre-mapped memory buffer */
int llm_load_from_buffer(void *data, uint64_t size)
{
    if (!data || size < 1024) {
        kprintf("[LLM] Invalid buffer (ptr=%p, size=%lu)\n", data, (unsigned long)size);
        return -1;
    }

    model_format_t fmt = model_detect_format(data, size);
    if (fmt == MODEL_FMT_UNKNOWN) {
        kprintf("[LLM] Unrecognized model format\n");
        return -1;
    }
    if (fmt != MODEL_FMT_GGUF) {
        static const char *fmt_names[] = {"unknown","gguf","safetensors","onnx","pytorch"};
        kprintf("[LLM] Detected format: %s (not yet supported for inference, use GGUF)\n",
                fmt_names[fmt]);
        return -1;
    }

    kmemset(&llm_model, 0, sizeof(llm_model));
    llm_model.data_buf = data;
    llm_model.data_size = size;
    kprintf("[LLM] Loading model from buffer: %lu MB\n",
            (unsigned long)(size / (1024 * 1024)));
    return llm_init_parsed_model(&llm_model);
}

/* Shared model initialization after data is in memory */
static int llm_init_parsed_model(llm_model_t *m)
{
    /* Parse GGUF */
    kprintf("[LLM] Parsing GGUF...\n");
    int rc = gguf_parse(&llm_gguf_ctx, m->data_buf, m->data_size);
    if (rc != 0) {
        kprintf("[LLM] GGUF parse error: %d\n", rc);
        return -1;
    }
    m->gguf = &llm_gguf_ctx;

    /* Extract architecture parameters */
    kstrcpy(m->arch, llm_gguf_ctx.arch);
    m->dim = (int)llm_gguf_ctx.n_embd;
    m->n_layers = (int)llm_gguf_ctx.n_layers;
    m->n_heads = (int)llm_gguf_ctx.n_heads;
    m->n_kv_heads = (int)llm_gguf_ctx.n_kv_heads;
    m->ff_dim = (int)llm_gguf_ctx.n_ff;
    m->vocab_size = (int)llm_gguf_ctx.n_vocab;
    m->rope_base = llm_gguf_ctx.rope_freq_base;

    /* If vocab_size is 0, infer from token_embd dimensions or tokenizer */
    if (m->vocab_size == 0) {
        const gguf_tensor_info_t *embd = gguf_find_tensor(&llm_gguf_ctx, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            /* GGUF dims: [ne0=n_embd, ne1=n_vocab] for embedding matrix */
            m->vocab_size = (int)embd->dims[1];
        }
    }
    /* Cross-check with tokenizer array count (most reliable source) */
    {
        const gguf_kv_t *tok_kv = gguf_find_kv(&llm_gguf_ctx, "tokenizer.ggml.tokens");
        if (tok_kv && tok_kv->type == GGUF_TYPE_ARRAY && tok_kv->value.array.count > 0) {
            int tv = (int)tok_kv->value.array.count;
            if (tv > m->vocab_size) {
                kprintf("[LLM] vocab_size corrected: %d -> %d (from tokenizer)\n",
                        m->vocab_size, tv);
                m->vocab_size = tv;
            }
        }
    }

    /* Compute derived parameters */
    if (llm_gguf_ctx.n_embd_head_k > 0)
        m->head_dim = (int)llm_gguf_ctx.n_embd_head_k;
    else if (m->n_heads > 0)
        m->head_dim = m->dim / m->n_heads;
    else
        m->head_dim = 64;
    if (m->n_kv_heads == 0) m->n_kv_heads = m->n_heads;
    if (m->rope_base < 1.0f) m->rope_base = 10000.0f;

    /* Gemma-family detection (gemma, gemma2, gemma4, etc.) */
    int is_gemma = (kstrlen(m->arch) >= 5 &&
        m->arch[0] == 'g' && m->arch[1] == 'e' &&
        m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a');

    /* Embedding scaling: Gemma models scale embeddings by sqrt(dim) */
    m->embed_scale = is_gemma ? llm_sqrtf((float)m->dim) : 1.0f;

    /* GeGLU: Gemma models use GELU activation with gating (not SiLU) */
    m->use_geglu = is_gemma;

    /* Logit softcapping from GGUF metadata */
    m->logit_softcap = llm_gguf_ctx.final_logit_softcap;

    /* Gemma4 specific: per-layer head_dim, KV sharing, SWA RoPE */
    m->is_gemma4 = 0;
    m->head_dim_swa = 0;
    m->rope_base_swa = 0.0f;
    m->n_layer_kv_start = m->n_layers; /* default: all layers have own KV */
    m->iswa_tok_embd = (void *)0;
    m->iswa_model_proj = (void *)0;
    m->iswa_proj_norm = (void *)0;
    m->iswa_n_embd = 0;
    m->rope_freqs = (const float *)0;
    if (is_gemma && kstrlen(m->arch) >= 6 && m->arch[5] == '4') {
        m->is_gemma4 = 1;
        m->head_dim_swa = (int)llm_gguf_ctx.n_embd_head_k_swa;
        if (m->head_dim_swa == 0) m->head_dim_swa = m->head_dim; /* fallback */
        m->rope_base_swa = llm_gguf_ctx.rope_freq_base_swa;
        if (m->rope_base_swa < 1.0f) m->rope_base_swa = 10000.0f;
        /* KV sharing: n_layer_kv_start = n_layers - shared_kv_layers */
        if (llm_gguf_ctx.shared_kv_layers > 0)
            m->n_layer_kv_start = m->n_layers - (int)llm_gguf_ctx.shared_kv_layers;
        m->iswa_n_embd = (int)llm_gguf_ctx.n_embd_per_layer;
        kprintf("[LLM] Gemma4: head_dim_swa=%d rope_base_swa=%.0f kv_start=%d iswa_embd=%d\n",
                m->head_dim_swa, (double)m->rope_base_swa, m->n_layer_kv_start, m->iswa_n_embd);
    }

    /* Determine max sequence length from GGUF (already parsed by arch prefix) */
    int ctx_len = (int)llm_gguf_ctx.n_ctx;
    if (ctx_len < 128) ctx_len = 2048;  /* sensible default */
    /* Default: cap at 8192 tokens — enough for long conversations.
     * GPU KV cache for 8192 ctx on an 8B model ≈ 1 GB; the dynamic
     * memory check below reduces further if RAM is tight.
     * Users can call llm_set_max_ctx() before loading to override. */
    if (llm_max_ctx_override > 0)
        ctx_len = llm_max_ctx_override;
    else if (ctx_len > 8192)
        ctx_len = 8192;

    /* Dynamically reduce context length if KV cache would exceed available memory */
    {
        uint64_t avail = tensor_mm_free_bytes();
        /* KV cache per token: 2 (K+V) * n_layers * n_kv_heads * head_dim * 4 bytes */
        uint64_t kv_per_token = 2ULL * m->n_layers * m->n_kv_heads * m->head_dim * sizeof(float);
        /* Reserve 256 MB for scratch buffers (logits, FFN, etc.) */
        uint64_t reserve = 256ULL * 1024 * 1024;
        if (kv_per_token > 0 && avail > reserve) {
            int max_fit = (int)((avail - reserve) / kv_per_token);
            if (max_fit < ctx_len) {
                kprintf("[LLM] Capping context %d -> %d tokens (%lu MB heap free)\n",
                        ctx_len, max_fit, (unsigned long)(avail / (1024*1024)));
                ctx_len = max_fit;
            }
        }
        if (ctx_len < 128) ctx_len = 128;
    }
    m->max_seq = ctx_len;

    /* Detect Phi-2 architecture (LayerNorm + GELU + partial RoPE) */
    m->use_layernorm = 0;
    m->use_gelu = 0;
    m->rope_dim = 0; /* 0 = full head_dim */
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i') {
        if (m->arch[3] == '2' || (m->arch[3] == '\0')) {
            /* Phi-2: LayerNorm + GELU + partial RoPE */
            m->use_layernorm = 1;
            m->use_gelu = 1;
            /* Phi-2 partial RoPE: typically 32 out of 80 head_dim */
            char keybuf[128];
            kstrcpy(keybuf, m->arch);
            kstrcpy(keybuf + kstrlen(keybuf), ".rope.dimension_count");
            m->rope_dim = (int)gguf_get_u32(&llm_gguf_ctx, keybuf, 0);
            if (m->rope_dim == 0) m->rope_dim = m->head_dim; /* fallback */
            kprintf("[LLM] Phi-2 mode: LayerNorm + GELU, rope_dim=%d\n", m->rope_dim);
        } else {
            /* Phi-3/Phi-3.5: LLaMA-compatible (RMSNorm + SwiGLU + full RoPE) */
            kprintf("[LLM] Phi-3 mode: LLaMA-compatible architecture\n");
        }
    }

    /* Longrope: load original context length for factor selection */
    m->rope_orig_ctx = 0;
    {
        char keybuf[128];
        kstrcpy(keybuf, m->arch);
        kstrcpy(keybuf + kstrlen(keybuf), ".rope.scaling.original_context_length");
        m->rope_orig_ctx = (int)gguf_get_u32(&llm_gguf_ctx, keybuf, 0);
        if (m->rope_orig_ctx > 0)
            kprintf("[LLM] Longrope: original_context_length=%d\n", m->rope_orig_ctx);
    }

    /* RMSNorm epsilon from model metadata (default 1e-5) */
    {
        char keybuf[128];
        kstrcpy(keybuf, m->arch);
        kstrcpy(keybuf + kstrlen(keybuf), ".attention.layer_norm_rms_epsilon");
        m->rms_eps = gguf_get_f32(&llm_gguf_ctx, keybuf, 1e-5f);
        kprintf("[LLM] rms_epsilon=%f\n", (double)m->rms_eps);
    }

    /* Get model name */
    const char *name = gguf_get_str(&llm_gguf_ctx, "general.name");
    if (name) {
        /* Copy up to 127 chars (name is GGUF string, not null-terminated) */
        const gguf_kv_t *kv = gguf_find_kv(&llm_gguf_ctx, "general.name");
        if (kv && kv->type == GGUF_TYPE_STRING) {
            int n = (int)kv->value.str.len;
            if (n > 127) n = 127;
            kmemcpy(m->name, kv->value.str.data, n);
            m->name[n] = '\0';
        }
    } else {
        kstrcpy(m->name, m->arch);
    }

    /* Print model info (concise for boot) */
    kprintf("[LLM] Model: %s (%s)\n", m->name, m->arch);
    kprintf("[LLM] %d layers, %d-dim, %d vocab, %d heads (%d KV), head_dim=%d\n",
            m->n_layers, m->dim, m->vocab_size, m->n_heads, m->n_kv_heads, m->head_dim);
    if (is_gemma)
        kprintf("[LLM] Gemma mode: embed_scale=%.1f geglu=%d softcap=%.1f\n",
                (double)m->embed_scale, m->use_geglu, (double)m->logit_softcap);

    /* Validate basic sanity */
    if (m->dim <= 0 || m->n_layers <= 0 || m->n_heads <= 0) {
        kprintf("[LLM] ERROR: invalid model dims (dim=%d layers=%d heads=%d)\n",
                m->dim, m->n_layers, m->n_heads);
        return -1;
    }

    /* Allocate layers array */
    m->layers = (llm_layer_t *)tensor_alloc(
        (uint64_t)m->n_layers * sizeof(llm_layer_t));
    if (!m->layers) {
        kprintf("[LLM] ERROR: failed to allocate %d layer descriptors\n", m->n_layers);
        return -1;
    }
    kmemset(m->layers, 0, (uint64_t)m->n_layers * sizeof(llm_layer_t));
    m->n_layers_alloc = m->n_layers;

    /* Map tensors */
    kprintf("[LLM] About to map tensors...\n");
    rc = llm_map_tensors(m, &llm_gguf_ctx);
    if (rc != 0) return rc;

    /* Gemma4: feed_forward_length is stored as an array in GGUF, so n_ff=0.
       Derive m->ff_dim from the per-layer ff_dim_layer values set by tensor loading. */
    if (m->ff_dim == 0 && m->n_layers > 0) {
        int max_ff = 0;
        for (int i = 0; i < m->n_layers; i++) {
            if (m->layers[i].ff_dim_layer > max_ff)
                max_ff = m->layers[i].ff_dim_layer;
        }
        if (max_ff > 0) {
            m->ff_dim = max_ff;
            kprintf("[LLM] Derived ff_dim=%d from per-layer tensor shapes\n", max_ff);
        }
    }

    kprintf("[LLM] Tensors mapped, building vocab...\n");

    /* Build vocabulary */
    rc = llm_build_vocab(m, &llm_gguf_ctx);
    if (rc != 0) return rc;
    kprintf("[LLM] Vocab built (%d tokens), allocating scratch...\n", m->n_vocab);

    /* Final vocab_size sanity: prefer tokenizer count over GGUF metadata */
    if (m->n_vocab > 0 && m->n_vocab != m->vocab_size) {
        kprintf("[LLM] Vocab: metadata=%d, tokenizer=%d, using %d\n",
                m->vocab_size, m->n_vocab,
                m->n_vocab > m->vocab_size ? m->n_vocab : m->vocab_size);
        if (m->n_vocab > m->vocab_size)
            m->vocab_size = m->n_vocab;
    }

    /* Allocate inference scratch arena (KV cache + all buffers) */
    if (llm_alloc_scratch(m) != 0)
        return -1;

    m->k_cache = llm_kv_k;
    m->v_cache = llm_kv_v;
    m->cache_len = 0;

    kprintf("[LLM] Model loaded successfully! Ready for inference.\n");

    /* Initialize flash attention scratch buffers */
    flash_attn_init();

    /* Initialize backend registry (CPU always, CUDA/MLIR if compiled) */
    backend_init_all();

#ifdef ENABLE_CUDA
    /* Auto-select CUDA and upload weights to GPU if available */
    if (backend_get_by_id(BACKEND_CUDA) && backend_set(BACKEND_CUDA) == 0) {
        if (llm_gpu_init(m) == 0) {
            kprintf("[LLM] Backend: cuda (GPU-accelerated)\n");
        } else {
            /* GPU upload failed — fall back to CPU */
            backend_set(BACKEND_CPU);
            kprintf("[LLM] Backend: cpu (GPU upload failed, using CPU fallback)\n");
        }
    } else {
        kprintf("[LLM] Backend: %s\n", backend_get()->name);
    }
#else
    kprintf("[LLM] Backend: %s\n", backend_get()->name);
#endif

    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Custom float printing (bare-metal: kprintf doesn't support %f)             */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_print_float(float v)
{
    if (v < 0) { kprintf("-"); v = -v; }
    int integer = (int)v;
    int frac = (int)((v - (float)integer) * 100.0f + 0.5f);
    if (frac >= 100) { integer++; frac -= 100; }
    kprintf("%d.%02d", integer, frac);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Math Evaluation Suite                                                       */
/*                                                                             */
/*  Tests the loaded LLM on mathematical reasoning problems.                   */
/*  Categories: Arithmetic, Algebra, Sequences, Multi-step, Challenge          */
/* ─────────────────────────────────────────────────────────────────────────── */

typedef struct {
    const char *prompt;     /* Math problem */
    const char *expected;   /* Expected answer substring */
    int         category;   /* 0=arith, 1=algebra, 2=sequence, 3=multi, 4=challenge */
} math_problem_t;

static const math_problem_t math_problems[] = {
    /* Arithmetic (Category 0) */
    { "Calculate: 25 + 17 = ",                  "42",    0 },
    { "What is 144 / 12? ",                     "12",    0 },
    { "Solve: 7 * 8 = ",                        "56",    0 },

    /* Algebra (Category 1) */
    { "If x + 5 = 12, then x = ",              "7",     1 },
    { "If 2x = 18, then x = ",                 "9",     1 },

    /* Sequences (Category 2) */
    { "What is the next number: 2, 4, 6, 8, ", "10",    2 },

    /* Multi-step (Category 3) */
    { "What is 2^8? ",                          "256",   3 },

    /* Challenge (Category 4) */
    { "What is the square root of 144? ",       "12",    4 },
};
#define N_MATH_PROBLEMS (int)(sizeof(math_problems) / sizeof(math_problems[0]))

static const char *category_names[] = {
    "Arithmetic", "Algebra", "Sequences", "Multi-step", "Challenge"
};

/* Build a chat-formatted prompt based on model architecture */
static int llm_format_prompt(const llm_model_t *m, const char *question,
                             char *buf, int max_len)
{
    int pos = 0;
    /* Detect prompt format from architecture and model name */
    int is_chatml = 0;
    int is_phi3   = 0;

    /* Check for Qwen (ChatML native) */
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'q' && m->arch[1] == 'w' &&
        m->arch[2] == 'e' && m->arch[3] == 'n')
        is_chatml = 1;

    /* Check for Phi-3/3.5 */
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i' &&
        m->arch[3] != '2' && m->arch[3] != '\0')
        is_phi3 = 1;

    /* Check for SmolLM / smollm in model name — SmolLM uses ChatML but
     * the special tokens may not be in base vocab; use simple prompt */
    /* (ChatML disabled for SmolLM until vocab includes <|im_start|>) */

    if (is_phi3) {
        /* Phi-3 format */
        { const char *t = "<|user|>\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
        { int qlen = kstrlen(question); if (pos + qlen < max_len) { kmemcpy(buf + pos, question, qlen); pos += qlen; } }
        { const char *t = "<|end|>\n<|assistant|>\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
    } else if (is_chatml) {
        /* ChatML format (Qwen, SmolLM, Mistral-Instruct-v0.3+, etc.) */
        { const char *t = "<|im_start|>system\nYou are a helpful math assistant. Give only the numeric answer.<|im_end|>\n<|im_start|>user\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
        { int qlen = kstrlen(question); if (pos + qlen < max_len) { kmemcpy(buf + pos, question, qlen); pos += qlen; } }
        { const char *t = "<|im_end|>\n<|im_start|>assistant\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
    } else if (kstrlen(m->arch) >= 5 &&
               m->arch[0] == 'g' && m->arch[1] == 'e' &&
               m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a') {
        /* Gemma format */
        { const char *t = "<turn|>user\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
        { int qlen = kstrlen(question); if (pos + qlen < max_len) { kmemcpy(buf + pos, question, qlen); pos += qlen; } }
        { const char *t = "\n<turn|>model\n"; int tlen = kstrlen(t); if (pos + tlen < max_len) { kmemcpy(buf + pos, t, tlen); pos += tlen; } }
    } else {
        /* Generic: simple text prompt (works with any model) */
        { int qlen = kstrlen(question); if (pos + qlen < max_len) { kmemcpy(buf + pos, question, qlen); pos += qlen; } }
    }
    buf[pos < max_len ? pos : max_len - 1] = '\0';
    return pos;
}

/* Check if expected answer appears in the generated text */
static int llm_check_answer(const char *generated, const char *expected)
{
    int gen_len = kstrlen(generated);
    int exp_len = kstrlen(expected);
    if (exp_len > gen_len) return 0;

    for (int i = 0; i <= gen_len - exp_len; i++) {
        int match = 1;
        for (int j = 0; j < exp_len; j++) {
            if (generated[i + j] != expected[j]) { match = 0; break; }
        }
        if (match) return 1;
    }
    return 0;
}

/* Run full math evaluation on a loaded model */
static void llm_run_math_eval(llm_model_t *m)
{
    kprintf("\n");
    kprintf("==========================================================\n");
    kprintf("  MATH REASONING EVALUATION\n");
    kprintf("  Model: %s (%s)\n", m->name, m->arch);
    kprintf("  %d problems across 5 categories\n", N_MATH_PROBLEMS);
    kprintf("==========================================================\n\n");

    int total_correct = 0;
    int cat_correct[5] = {0, 0, 0, 0, 0};
    int cat_total[5]   = {0, 0, 0, 0, 0};

    char prompt_buf[512];
    char output_buf[256];

    for (int p = 0; p < N_MATH_PROBLEMS; p++) {
        const math_problem_t *prob = &math_problems[p];
        cat_total[prob->category]++;

        /* Format the prompt */
        llm_format_prompt(m, prob->prompt, prompt_buf, sizeof(prompt_buf));

        /* Tokenize */
        int n_tokens = llm_tokenize(m, prompt_buf, llm_tokens, llm_alloc_tokens - 32);

        kprintf("  [%d/%d] %s\n", p + 1, N_MATH_PROBLEMS, prob->prompt);
        kprintf("         Tokens: %d, generating...\n", n_tokens);

        /* Generate */
        uint64_t t0 = rdtsc_fenced();
        int n_gen = llm_generate(m, llm_tokens, n_tokens, output_buf, sizeof(output_buf),
                                 16, 0.0f, 0); /* greedy, max 16 tokens, fresh cache */
        uint64_t t1 = rdtsc_fenced();

        uint64_t gen_ms = perf_cycles_to_us(t1 - t0) / 1000;
        uint64_t ms_per_tok = (n_gen > 0) ? (gen_ms / (uint64_t)n_gen) : 0;

        /* Check answer */
        int correct = llm_check_answer(output_buf, prob->expected);
        if (correct) {
            total_correct++;
            cat_correct[prob->category]++;
        }

        kprintf("         Answer: %s\n", output_buf);
        kprintf("         Expected: %s | %s | %lu ms (%lu ms/tok)\n",
                prob->expected,
                correct ? "CORRECT" : "WRONG",
                gen_ms, ms_per_tok);
    }

    /* Summary */
    kprintf("\n  ─── Results ─────────────────────────────────────────\n");
    for (int c = 0; c < 5; c++) {
        if (cat_total[c] > 0) {
            kprintf("  %-12s: %d/%d", category_names[c], cat_correct[c], cat_total[c]);
            int pct = (cat_correct[c] * 100) / cat_total[c];
            kprintf(" (%d%%)\n", pct);
        }
    }
    kprintf("  ──────────────────────────────────────────────────────\n");
    kprintf("  TOTAL: %d/%d correct", total_correct, N_MATH_PROBLEMS);
    int total_pct = (total_correct * 100) / N_MATH_PROBLEMS;
    kprintf(" (%d%%)\n", total_pct);
    kprintf("==========================================================\n");
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Inference Speed Benchmark                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

static void llm_run_benchmark(llm_model_t *m)
{
    kprintf("\n  --- Inference Speed Benchmark ---\n");

    /* Reset KV cache */
    m->cache_len = 0;

    /* Time a single forward pass */
    uint64_t t0 = rdtsc_fenced();
    llm_forward_token(m, llm_logits, m->bos_id, 0);
    uint64_t t1 = rdtsc_fenced();

    uint64_t single_ms = perf_cycles_to_us(t1 - t0) / 1000;
    kprintf("  Single token forward: %lu ms\n", single_ms);

    /* Estimate tokens per second */
    if (single_ms > 0) {
        double tps = 1000.0 / (double)single_ms;
        kprintf("  Estimated throughput: ~%.1f tokens/s\n", tps);
    }

    /* Memory usage */
    int64_t kv_kbytes = ((int64_t)m->n_layers * m->max_seq * m->n_kv_heads * m->head_dim * 4 * 2) / 1024;
    kprintf("  KV cache: %lld KB (for %d seq len)\n", (long long)kv_kbytes, m->max_seq);
    kprintf("  Model data: %lu MB\n", m->data_size / (1024 * 1024));

    /* Compute total parameters */
    uint64_t total_params = m->gguf->total_param_count;
    kprintf("  Parameters: %lu M\n", total_params / 1000000);

    /* Compute FLOPS estimate per token */
    /* ~2 * params FLOPs per inference token */
    uint64_t flops_per_token = 2 * total_params;
    if (single_ms > 0) {
        double gf = (double)flops_per_token / ((double)single_ms * 1000000.0);
        kprintf("  Effective: ~%.2f GFLOPS\n", gf);
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Main Entry Point                                                           */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Boot-time model loader: load GGUF from disk into RAM, but don't run eval.
 * This is called during boot for fast startup. */
void llm_boot_load(void)
{
#ifdef __aarch64__
    kprintf("  [--] ARM64 disk loading not yet available\n");
    return;
#else
    uint64_t capacity = virtio_blk_capacity();
    if (capacity == 0) {
        kprintf("  [--] No model disk (use 'llm' in shell for instructions)\n");
        return;
    }

    /* Verify GGUF magic */
    {
        static uint8_t hdr_buf[512];
        int rc = virtio_blk_read(0, 1, hdr_buf);
        if (rc != 0) {
            kprintf("  [--] Failed to read disk\n");
            return;
        }
        uint32_t magic = (uint32_t)hdr_buf[0] | ((uint32_t)hdr_buf[1] << 8) |
                         ((uint32_t)hdr_buf[2] << 16) | ((uint32_t)hdr_buf[3] << 24);
        if (magic != GGUF_MAGIC) {
            kprintf("  [--] Disk is not a GGUF model file\n");
            return;
        }
    }

    kprintf("  Loading %lu MB model from disk...\n", capacity / (1024 * 1024));

    kmemset(&llm_model, 0, sizeof(llm_model));
    int rc = llm_load_from_disk(&llm_model);
    if (rc != 0) {
        kprintf("  [FAIL] Model load error: %d\n", rc);
        return;
    }

    kprintf("  [OK] %s (%d layers, %lu params)\n",
            llm_model.name, llm_model.n_layers,
            (uint64_t)llm_model.vocab_size * llm_model.dim);
#endif
}

/* Full eval (benchmarks + math) — callable from shell via 'demo llm' */
void llm_run_full_eval(void)
{
    if (!llm_is_loaded()) {
        kprintf("[LLM] No model loaded.\n");
        return;
    }
    llm_run_benchmark(&llm_model);
    llm_run_math_eval(&llm_model);
    kprintf("\n[LLM] Evaluation complete.\n");
}

void llm_run_eval(void)
{
    kprintf("\n============================================================\n");
    kprintf("  REAL LLM INFERENCE ENGINE\n");
    kprintf("  Bare-Metal Model Loading & Math Evaluation\n");
    kprintf("============================================================\n");
    kprintf("  Supported: Qwen, Gemma, LLaMA, SmolLM, Mistral, ...\n");
    kprintf("  Formats:   GGUF v2/v3 with Q4_0, Q8_0, F16, F32\n");
    kprintf("  Features:  GQA, RoPE, KV-Cache, SwiGLU, RMSNorm\n");
    kprintf("============================================================\n\n");

#ifdef __aarch64__
    kprintf("[LLM] ARM64 disk loading not implemented yet.\n");
    kprintf("[LLM] Use x86_64 QEMU with: -drive file=model.gguf,format=raw,if=virtio\n");
    return;
#else
    /* Check if virtio-blk is available with a model */
    uint64_t capacity = virtio_blk_capacity();
    if (capacity == 0) {
        kprintf("[LLM] No model disk detected.\n\n");
        kprintf("  To run real LLM inference, attach a GGUF model:\n\n");
        kprintf("  1. Download a model (e.g., Qwen2.5-0.5B or SmolLM2-135M):\n");
        kprintf("     .\\tools\\download_model.ps1 -Model qwen2.5-0.5b\n\n");
        kprintf("  2. Rebuild and run with the model attached:\n");
        kprintf("     .\\build.ps1 -Run\n");
        kprintf("     (the build script auto-detects models\\*.gguf)\n\n");
        kprintf("  Or manually:\n");
        kprintf("     qemu-system-x86_64 -kernel build\\tensoros.elf \\\n");
        kprintf("       -drive file=model.gguf,format=raw,if=virtio ...\n\n");
        kprintf("  Supported models:\n");
        kprintf("    - Qwen2.5-0.5B-Instruct  (Q4_0, ~350 MB)\n");
        kprintf("    - SmolLM2-135M-Instruct   (Q8_0, ~145 MB)\n");
        kprintf("    - TinyLlama-1.1B-Chat     (Q4_0, ~600 MB)\n");
        kprintf("    - Gemma-2-2B-IT           (Q4_0, ~1.2 GB)\n");
        kprintf("    - Any GGUF model with LLaMA-style architecture\n\n");
        return;
    }

    /* Verify it looks like a GGUF file (check magic at sector 0) */
    {
        static uint8_t hdr_buf[512];
        int rc = virtio_blk_read(0, 1, hdr_buf);
        if (rc != 0) {
            kprintf("[LLM] Failed to read first sector\n");
            return;
        }
        uint32_t magic = (uint32_t)hdr_buf[0] | ((uint32_t)hdr_buf[1] << 8) |
                         ((uint32_t)hdr_buf[2] << 16) | ((uint32_t)hdr_buf[3] << 24);
        if (magic != GGUF_MAGIC) {
            kprintf("[LLM] Disk does not contain a GGUF file (magic=%x, expected %x)\n",
                    magic, GGUF_MAGIC);
            kprintf("[LLM] Attach a .gguf file as raw virtio disk.\n");
            return;
        }
        kprintf("[LLM] GGUF file detected on virtio-blk (%lu MB)\n",
                capacity / (1024 * 1024));
    }

    /* Load the model */
    kmemset(&llm_model, 0, sizeof(llm_model));
    int rc = llm_load_from_disk(&llm_model);
    if (rc != 0) {
        kprintf("[LLM] Model loading failed: %d\n", rc);
        return;
    }

    /* Run benchmark */
    llm_run_benchmark(&llm_model);

    /* Run math evaluation */
    llm_run_math_eval(&llm_model);

    kprintf("\n[LLM] Evaluation complete.\n");
#endif
}

/* =============================================================================
 * Interactive LLM prompt — called from the shell
 * =============================================================================*/
int llm_is_loaded(void)
{
    return (llm_model.vocab_size > 0 && llm_model.data_buf != NULL);
}

const char *llm_model_name(void)
{
    if (!llm_is_loaded()) return "(none)";
    return llm_model.name;
}

const char *llm_model_arch(void)
{
    if (!llm_is_loaded()) return "(none)";
    return llm_model.arch;
}

int llm_model_layers(void)
{
    if (!llm_is_loaded()) return 0;
    return llm_model.n_layers;
}

int llm_model_dim(void)
{
    if (!llm_is_loaded()) return 0;
    return llm_model.dim;
}

int llm_model_vocab(void)
{
    if (!llm_is_loaded()) return 0;
    return llm_model.vocab_size;
}

uint64_t llm_param_count(void)
{
    if (!llm_is_loaded()) return 0;
    /* Approximate: dim² × 12 × n_layers (QKV+O+gate+up+down + norms + embd) */
    llm_model_t *m = &llm_model;
    uint64_t per_layer = (uint64_t)m->dim * (uint64_t)m->dim * 4  /* QKV+O */
                       + (uint64_t)m->dim * (uint64_t)m->ff_dim * 3; /* gate+up+down */
    uint64_t total = per_layer * (uint64_t)m->n_layers
                   + (uint64_t)m->vocab_size * (uint64_t)m->dim; /* embeddings */
    return total;
}

int llm_set_backend(llm_backend_t backend)
{
    switch (backend) {
    case LLM_BACKEND_CPU:
        llm_backend = backend;
        return 0;
    case LLM_BACKEND_CUDA:
    case LLM_BACKEND_MLIR:
        /* Backend scaffolding is in place; kernels/runtime are pending. */
        return -2;
    default:
        return -1;
    }
}

llm_backend_t llm_get_backend(void)
{
    return llm_backend;
}

int llm_last_vram_usage_mb(void)
{
    return llm_last_vram_mb;
}

float llm_kv_cache_hit_rate(void)
{
    if (g_kv_snap_lookups == 0) return 0.0f;
    return (float)g_kv_snap_hits / (float)g_kv_snap_lookups;
}

float llm_last_logit_entropy(void)
{
    return g_last_logit_entropy;
}

float llm_last_prefill_ms(void)
{
    return llm_last_prefill_ms_val;
}

float llm_last_tok_per_sec(void)
{
    return llm_last_tok_per_sec_val;
}

const char *llm_backend_name(void)
{
    switch (llm_backend) {
    case LLM_BACKEND_CPU:  return "cpu";
    case LLM_BACKEND_CUDA: return "cuda";
    case LLM_BACKEND_MLIR: return "mlir";
    default:               return "unknown";
    }
}

const llm_model_t *llm_get_model(void)
{
    if (!llm_is_loaded()) return (const llm_model_t *)0;
    return &llm_model;
}

int llm_get_embedding_vec(int token_id, float *out, int dim)
{
    if (!llm_is_loaded() || !out) return -1;
    if (token_id < 0 || token_id >= llm_model.vocab_size) return -1;
    if (dim != llm_model.dim) return -1;
    llm_embed(out, &llm_model, token_id);
    return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Manifold Exploitation API (axiom_exploit hooks)                            */
/* ─────────────────────────────────────────────────────────────────────────── */

const float *llm_get_kv_k(int layer, int head, int pos)
{
    if (!llm_is_loaded()) return NULL;
    llm_model_t *m = &llm_model;
    if (layer < 0 || layer >= m->n_layers)  return NULL;
    if (head  < 0 || head  >= m->n_kv_heads) return NULL;
    if (pos   < 0 || pos   >= m->cache_len)  return NULL;

    int hd      = m->head_dim;
    int kv_dim  = m->n_kv_heads * hd;
    int stride  = m->max_seq * kv_dim;   /* floats per layer in k_cache */

    /* k_cache layout: [n_layers][max_seq][n_kv_heads][head_dim] */
    return m->k_cache
           + (uint64_t)layer * stride
           + (uint64_t)pos   * kv_dim
           + (uint64_t)head  * hd;
}

/* Layer-granular GPU management — only meaningful when ENABLE_CUDA is set. */
int llm_gpu_layer_ensure(int layer)
{
#ifdef ENABLE_CUDA
    if (!llm_is_loaded()) return -1;
    if (layer < 0 || layer >= llm_model.n_layers) return -1;
    if (!gpu_ctx.active) return -1;
    /* If the layer's weight map entry doesn't exist, upload it now.
     * We piggyback on the existing gpu_upload_weight helper that is already
     * called per-weight at model load.  Here we re-check whether the layer's
     * key matrices are already on the GPU; if not we re-upload.
     * Full selective-upload is handled by axex_offload_apply() which calls
     * this function per layer — it's a no-op when the layer is already uploaded. */
    return 0;   /* weights already uploaded by llm_gpu_init — no-op for now */
#else
    (void)layer;
    return -1;  /* No GPU */
#endif
}

int llm_gpu_layer_release(int layer)
{
#ifdef ENABLE_CUDA
    if (!llm_is_loaded()) return -1;
    if (layer < 0 || layer >= llm_model.n_layers) return -1;
    if (!gpu_ctx.active) return -1;
    gpu_layer_t *gl = &gpu_ctx.layers[layer];
    const backend_t *be = backend_get_by_id(BACKEND_CUDA);
    if (!be) return -1;
    /* Free KV cache buffers (they can be reconstructed from CPU cache). */
    if (gl->d_k_cache) { be->mem.free(gl->d_k_cache); gl->d_k_cache = NULL; }
    if (gl->d_v_cache) { be->mem.free(gl->d_v_cache); gl->d_v_cache = NULL; }
    return 0;
#else
    (void)layer;
    return -1;
#endif
}

uint64_t llm_gpu_vram_available(void)
{
#ifdef ENABLE_CUDA
    const backend_t *be = backend_get_by_id(BACKEND_CUDA);
    if (be && be->get_free_memory)
        return be->get_free_memory(0);
    return 0;
#else
    return 0;
#endif
}

/*
 * Upload compressed weight matrices for all on_gpu layers to device memory.
 * Must be called after both llm_gpu_init() and axex_compress_model_ffn() complete.
 * Safe to call multiple times — skips layers already uploaded (d_Vt != NULL).
 */
void llm_gpu_upload_compressed_weights(void)
{
#ifdef ENABLE_CUDA
    if (!gpu_ctx.gpu_fwd) return;
    const backend_t *be = backend_get_by_id(BACKEND_CUDA);
    if (!be) return;

    const llm_model_t *m = llm_get_model();
    if (!m) return;

    int uploaded = 0;
    uint64_t phase1_freed_bytes = 0;
    int phase1_freed_entries = 0;
    for (int L = 0; L < m->n_layers; L++) {
        if (!gpu_ctx.layers[L].on_gpu) continue;   /* CPU layer — keep host only */
        for (int slot = 0; slot < AXEX_COMPRESS_WHICH_MAX; slot++) {
            axex_compressed_weight_t *cw =
                axex_get_compressed_layer_mut(L, slot);
            if (!cw || cw->d_Vt) continue;          /* not compressed or already on GPU */
            axex_upload_compressed_to_gpu(cw,
                (void *(*)(uint64_t))be->mem.alloc,
                (void (*)(void *, const void *, uint64_t))be->mem.upload,
                be->mem.sync);
            if (cw->d_Vt) {
                uploaded++;
                kprintf("[GPU] Layer %d slot %d: compressed weight uploaded to GPU "
                        "(%dx%d rank %d)\n", L, slot, cw->m, cw->n, cw->rank);
            }
        }
        /* Interleaved free: after uploading all slots for this layer, immediately
         * free the raw GPU FFN buffers if all three (down/gate/up) are now on GPU.
         * This recovers ~176 MB per layer DURING the loop, preventing VRAM
         * exhaustion that would otherwise limit uploads to ~7 layers. */
        {
            const llm_layer_t *ly = &m->layers[L];
            const axex_compressed_weight_t *cd = axex_get_compressed_layer(L, 0);
            const axex_compressed_weight_t *cg = axex_get_compressed_layer(L, 6);
            const axex_compressed_weight_t *cu = axex_get_compressed_layer(L, 5);
            int has_compressed_ffn = (cd && cd->d_Vt) &&
                                     (!ly->ffn_gate || (cg && cg->d_Vt)) &&
                                     (cu && cu->d_Vt);
            if (has_compressed_ffn) {
                const void *ffn_ptrs[3] = { ly->ffn_down, ly->ffn_gate, ly->ffn_up };
                for (int fp = 0; fp < 3; fp++) {
                    if (!ffn_ptrs[fp]) continue;
                    for (int wi = 0; wi < gpu_ctx.map_count; wi++) {
                        if (gpu_ctx.map[wi].host_ptr != ffn_ptrs[fp]) continue;
                        if (!gpu_ctx.map[wi].dev_ptr) continue;
                        phase1_freed_bytes += gpu_ctx.map[wi].size;
                        be->mem.free(gpu_ctx.map[wi].dev_ptr);
                        for (int wj = wi; wj < gpu_ctx.map_count - 1; wj++)
                            gpu_ctx.map[wj] = gpu_ctx.map[wj + 1];
                        gpu_ctx.map_count--;
                        phase1_freed_entries++;
                        break;
                    }
                }
            }

            /* Also free raw Q and O GPU buffers if both SVD-compressed slots are ready.
             * For 70B this recovers ~38 MB per layer (Q+O @ IQ2_XS 8192×8192 × 2). */
            const axex_compressed_weight_t *cq_s = axex_get_compressed_layer(L, 1);
            const axex_compressed_weight_t *co_s = axex_get_compressed_layer(L, 4);
            int has_compressed_qo = (cq_s && cq_s->d_Vt) && (co_s && co_s->d_Vt);
            if (has_compressed_qo) {
                const void *qo_ptrs[2] = { ly->q_weight, ly->o_weight };
                for (int fp = 0; fp < 2; fp++) {
                    if (!qo_ptrs[fp]) continue;
                    for (int wi = 0; wi < gpu_ctx.map_count; wi++) {
                        if (gpu_ctx.map[wi].host_ptr != qo_ptrs[fp]) continue;
                        if (!gpu_ctx.map[wi].dev_ptr) continue;
                        phase1_freed_bytes += gpu_ctx.map[wi].size;
                        be->mem.free(gpu_ctx.map[wi].dev_ptr);
                        for (int wj = wi; wj < gpu_ctx.map_count - 1; wj++)
                            gpu_ctx.map[wj] = gpu_ctx.map[wj + 1];
                        gpu_ctx.map_count--;
                        phase1_freed_entries++;
                        break;
                    }
                }
            }
        }
    }
    if (uploaded)
        kprintf("[GPU] %d compressed weight matrices uploaded to GPU\n", uploaded);
    if (phase1_freed_bytes > 0)
        kprintf("[GPU] Freed %llu MB raw FFN GPU buffers (%d tensors) during upload — "
                "using compressed path\n",
                (unsigned long long)(phase1_freed_bytes >> 20), phase1_freed_entries);

    /* ── Phase 1b: build cuBLAS batched GEMV pointer arrays for gate+up ── */
    {
        int batched = 0;
        for (int L = 0; L < m->n_layers; L++) {
            if (!gpu_ctx.layers[L].on_gpu) continue;
            axex_compressed_weight_t *cg = axex_get_compressed_layer_mut(L, 6); /* gate */
            axex_compressed_weight_t *cu = axex_get_compressed_layer_mut(L, 5); /* up   */
            if (!cg || !cg->d_Vt || !cu || !cu->d_Vt) continue;
            if (cg->d_Aarray_step1) continue; /* already prepared */
            axex_prepare_batched_ffn(cg, cu,
                gpu_ctx.d_xn, gpu_ctx.d_ffn_g, gpu_ctx.d_ffn_u,
                (void *(*)(uint64_t))be->mem.alloc,
                (void (*)(void *, const void *, uint64_t))be->mem.upload,
                be->mem.sync);
            if (cg->d_Aarray_step1) batched++;
        }
        if (batched)
            kprintf("[GPU] %d layers: cuBLAS batched gate+up GEMV ready\n", batched);
    }

    /* Upload GP manifold: Pt matrix once + all W_proj matrices */
    {
        int gp_uploaded = 0;
        axex_manifold_upload_gpu(
            (void *(*)(uint64_t))be->mem.alloc,
            (void (*)(void *, const void *, uint64_t))be->mem.upload,
            be->mem.sync);
        /* Count GP layers for logging */
        if (llm_get_model()) {
            for (int _L = 0; _L < llm_get_model()->n_layers; _L++)
                for (int _s = 1; _s <= 6; _s++) {
                    const axex_manifold_weight_t *_mw = axex_get_manifold_layer(_L, _s);
                    if (_mw && _mw->d_W_proj) gp_uploaded++;
                }
        }
        if (gp_uploaded)
            kprintf("[GPU] Manifold GP: Pt + %d W_proj matrices uploaded to GPU\n", gp_uploaded);
        /* Pin Pt F16 in GPU L2 persistent cache — after first token every Pt
         * access serves from L2 (~1800 GB/s) instead of HBM (~340 GB/s). */
#ifdef ENABLE_CUDA
        axex_manifold_l2_persist_pt_gpu(cuda_l2_persist);
#endif
    }

    if (!uploaded && !axex_manifold_k()) return; /* nothing new to do */

    /* ── Phase 2: free raw FFN GPU buffers for compressed layers ── */
    uint64_t freed_bytes = 0;
    int freed_entries = 0;

    for (int L = 0; L < m->n_layers; L++) {
        if (!gpu_ctx.layers[L].on_gpu) continue;
        const llm_layer_t *ly = &m->layers[L];
        /* Check if all three FFN projections are compressed on GPU */
        const axex_compressed_weight_t *cd = axex_get_compressed_layer(L, 0); /* down */
        const axex_compressed_weight_t *cg = axex_get_compressed_layer(L, 6); /* gate */
        const axex_compressed_weight_t *cu = axex_get_compressed_layer(L, 5); /* up   */
        int has_compressed_ffn = (cd && cd->d_Vt) &&
                                 (!ly->ffn_gate || (cg && cg->d_Vt)) &&
                                 (cu && cu->d_Vt);
        if (!has_compressed_ffn) continue;

        /* Free raw GPU tensors for ffn_down, ffn_gate, ffn_up */
        const void *ffn_ptrs[3] = { ly->ffn_down, ly->ffn_gate, ly->ffn_up };
        for (int fp = 0; fp < 3; fp++) {
            if (!ffn_ptrs[fp]) continue;
            for (int wi = 0; wi < gpu_ctx.map_count; wi++) {
                if (gpu_ctx.map[wi].host_ptr != ffn_ptrs[fp]) continue;
                if (!gpu_ctx.map[wi].dev_ptr) continue;
                freed_bytes += gpu_ctx.map[wi].size;
                be->mem.free(gpu_ctx.map[wi].dev_ptr);
                /* Remove from map by shifting */
                for (int wj = wi; wj < gpu_ctx.map_count - 1; wj++)
                    gpu_ctx.map[wj] = gpu_ctx.map[wj + 1];
                gpu_ctx.map_count--;
                freed_entries++;
                break; /* host_ptr is unique */
            }
        }
    }

    if (freed_bytes > 0)
        kprintf("[GPU] Freed %llu MB raw FFN GPU buffers (%d tensors) — "
                "using compressed path\n",
                (unsigned long long)(freed_bytes >> 20), freed_entries);

    /* ── Phase 2b: free raw attention GPU buffers for GP-manifold-compressed layers ──
     * When GP is active, Q/K/V weights are replaced by W_proj[m×k] — up to 64× smaller.
     * Freeing the original quantized tensors recovers the bulk of VRAM for 70B models. */
    {
        uint64_t gp_freed_bytes = 0;
        int gp_freed_entries = 0;
        int gp_k = axex_manifold_k();

        for (int L = 0; L < m->n_layers; L++) {
            if (!gpu_ctx.layers[L].on_gpu) continue;
            const llm_layer_t *ly = &m->layers[L];

            /* Q/K/V/O: free raw GPU tensor when GP manifold weight is uploaded.
             * IMPORTANT: the inference path enters the GP branch only when Q (slot 1)
             * is GP-compressed.  If Q is raw but K or V are compressed, we must NOT
             * free the raw K/V — the fallback else-branch needs them.
             * So: only free K (slot 2) and V (slot 3) if Q (slot 1) is also compressed
             * for this layer.  Q and O can always be freed independently. */
            const axex_manifold_weight_t *mw_q_check =
                axex_get_manifold_layer(L, 1);
            int q_gp_ready = (mw_q_check && mw_q_check->d_W_proj);

            struct { const void *host; int slot; } attn_map[] = {
                { ly->q_weight, 1 }, { ly->k_weight, 2 },
                { ly->v_weight, 3 }, { ly->o_weight, 4 }
            };
            for (int ap = 0; ap < 4; ap++) {
                if (!attn_map[ap].host) continue;
                int slot = attn_map[ap].slot;
                /* K (slot 2) and V (slot 3): only free if Q is also GP-compressed */
                if ((slot == 2 || slot == 3) && !q_gp_ready) continue;
                const axex_manifold_weight_t *mw =
                    axex_get_manifold_layer(L, slot);
                if (!mw || !mw->d_W_proj) continue; /* not GP-compressed on GPU */
                for (int wi = 0; wi < gpu_ctx.map_count; wi++) {
                    if (gpu_ctx.map[wi].host_ptr != attn_map[ap].host) continue;
                    if (!gpu_ctx.map[wi].dev_ptr) continue;
                    gp_freed_bytes += gpu_ctx.map[wi].size;
                    be->mem.free(gpu_ctx.map[wi].dev_ptr);
                    for (int wj = wi; wj < gpu_ctx.map_count - 1; wj++)
                        gpu_ctx.map[wj] = gpu_ctx.map[wj + 1];
                    gpu_ctx.map_count--;
                    gp_freed_entries++;
                    break;
                }
            }

            /* gate/up: free raw GPU tensor when GP W_proj is uploaded */
            struct { const void *host; int slot; } ffn_gp_map[] = {
                { ly->ffn_gate, 6 }, { ly->ffn_up, 5 }
            };
            for (int fp = 0; fp < 2; fp++) {
                if (!ffn_gp_map[fp].host) continue;
                const axex_manifold_weight_t *mw =
                    axex_get_manifold_layer(L, ffn_gp_map[fp].slot);
                if (!mw || !mw->d_W_proj) continue;
                for (int wi = 0; wi < gpu_ctx.map_count; wi++) {
                    if (gpu_ctx.map[wi].host_ptr != ffn_gp_map[fp].host) continue;
                    if (!gpu_ctx.map[wi].dev_ptr) continue;
                    gp_freed_bytes += gpu_ctx.map[wi].size;
                    be->mem.free(gpu_ctx.map[wi].dev_ptr);
                    for (int wj = wi; wj < gpu_ctx.map_count - 1; wj++)
                        gpu_ctx.map[wj] = gpu_ctx.map[wj + 1];
                    gpu_ctx.map_count--;
                    gp_freed_entries++;
                    break;
                }
            }
        }
        if (gp_freed_bytes > 0)
            kprintf("[GPU] Freed %llu MB raw Q/K/V/O/gate/up GPU buffers (%d tensors) — "
                    "GP manifold path active (k=%d)\n",
                    (unsigned long long)(gp_freed_bytes >> 20), gp_freed_entries, gp_k);
        freed_bytes += gp_freed_bytes;
        freed_entries += gp_freed_entries;
    }

    /* ── Phase 2c: fallback raw Q/O upload for attn-SVD layers where compression
     * did not cover every layer.  When --axex-attn-svd is set, raw Q/O were not
     * uploaded during llm_gpu_init.  If SVD or GP compression is NOT present for
     * a layer (max_err rejection, rank budget, etc.) the layer has no Q on GPU at
     * all, causing a "q_weight not on GPU" warning that silently skips the layer.
     * We fix that here: upload the raw Q/O for any layer that lacks a compressed
     * replacement, so every on-GPU layer always has a valid Q weight path. */
    if (gpu_attn_skip_for_compress) {
        int dim_fb  = gpu_ctx.dim;
        int hd_fb   = gpu_ctx.head_dim;
        int fb_uploaded = 0;
        for (int L = 0; L < m->n_layers; L++) {
            if (!gpu_ctx.layers[L].on_gpu) continue;
            const llm_layer_t *ly = &m->layers[L];
            int lhd_fb    = ly->head_dim_layer ? ly->head_dim_layer : hd_fb;
            int lq_dim_fb = m->n_heads * lhd_fb;

            /* Q: check if raw or compressed version is already on GPU */
            const axex_compressed_weight_t *cq_fb = axex_get_compressed_layer(L, 1);
            const axex_manifold_weight_t   *gq_fb = axex_get_manifold_layer(L, 1);
            int q_covered = (cq_fb && cq_fb->d_Vt) ||
                            (gq_fb && gq_fb->d_W_proj) ||
                            llm_gpu_lookup(ly->q_weight) != NULL;
            if (!q_covered && ly->q_weight) {
                llm_gpu_upload_weight_mat(ly->q_weight, lq_dim_fb, dim_fb, ly->q_type);
                fb_uploaded++;
            }

            /* O: same fallback */
            const axex_compressed_weight_t *co_fb = axex_get_compressed_layer(L, 4);
            const axex_manifold_weight_t   *go_fb = axex_get_manifold_layer(L, 4);
            int o_covered = (co_fb && co_fb->d_Vt) ||
                            (go_fb && go_fb->d_W_proj) ||
                            llm_gpu_lookup(ly->o_weight) != NULL;
            if (!o_covered && ly->o_weight) {
                llm_gpu_upload_weight_mat(ly->o_weight, dim_fb, lq_dim_fb, ly->o_type);
                fb_uploaded++;
            }
        }
        if (fb_uploaded)
            kprintf("[GPU] Fallback: uploaded %d raw Q/O matrices for layers "
                    "where attn-SVD compression was not applied\n", fb_uploaded);
    }

    /* ── Phase 2d: fallback raw FFN upload for ffn-compress layers where SVD
     * compression was not applied (degenerate matrix, OOM during compression,
     * or intentional rank-0 skip).  When --axex-ffn-compress is set, raw FFN
     * was never uploaded during llm_gpu_init.  Upload raw for any weight that
     * lacks a compressed replacement so the decode forward pass always has a
     * valid path and CUDA graph capture can succeed. */
    if (gpu_ffn_skip_for_compress) {
        int dim_2d   = gpu_ctx.dim;
        int fb2_uploaded = 0;
        for (int L = 0; L < m->n_layers; L++) {
            if (!gpu_ctx.layers[L].on_gpu) continue;
            const llm_layer_t *ly = &m->layers[L];
            /* Compute ff dim from layer (may differ per-layer in MoE models) */
            int lff = ly->ff_dim_layer > 0 ? ly->ff_dim_layer : gpu_ctx.max_ff;

            /* gate */
            if (ly->ffn_gate) {
                const axex_compressed_weight_t *cg2 = axex_get_compressed_layer(L, 6);
                if (!cg2 || !cg2->d_Vt) {
                    const axex_manifold_weight_t *mg2 = axex_get_manifold_layer(L, 6);
                    if (!(mg2 && mg2->d_W_proj) && !llm_gpu_lookup(ly->ffn_gate)) {
                        llm_gpu_upload_weight_mat(ly->ffn_gate, lff, dim_2d, ly->gate_type);
                        fb2_uploaded++;
                    }
                }
            }
            /* up */
            if (ly->ffn_up) {
                const axex_compressed_weight_t *cu2 = axex_get_compressed_layer(L, 5);
                if (!cu2 || !cu2->d_Vt) {
                    const axex_manifold_weight_t *mu2 = axex_get_manifold_layer(L, 5);
                    if (!(mu2 && mu2->d_W_proj) && !llm_gpu_lookup(ly->ffn_up)) {
                        llm_gpu_upload_weight_mat(ly->ffn_up, lff, dim_2d, ly->up_type);
                        fb2_uploaded++;
                    }
                }
            }
            /* down */
            if (ly->ffn_down) {
                const axex_compressed_weight_t *cd2b = axex_get_compressed_layer(L, 0);
                if (!cd2b || !cd2b->d_Vt) {
                    if (!llm_gpu_lookup(ly->ffn_down)) {
                        llm_gpu_upload_weight_mat(ly->ffn_down, dim_2d, lff, ly->down_type);
                        fb2_uploaded++;
                    }
                }
            }
        }
        if (fb2_uploaded)
            kprintf("[GPU] Fallback: uploaded %d raw FFN matrices for layers "
                    "where ffn-compress SVD was not applied\n", fb2_uploaded);
    }

    /* ── Phase 3: promote CPU layers to GPU using freed VRAM ── */
    int promoted = 0;
    const llm_model_t *m2 = m; /* alias for clarity */
    int dim     = gpu_ctx.dim;
    int hd      = gpu_ctx.head_dim;
    int n_kv    = gpu_ctx.n_kv;
    int max_seq = gpu_ctx.max_seq;

    for (int L = 0; L < m2->n_layers; L++) {
        gpu_layer_t *gl = &gpu_ctx.layers[L];
        if (gl->on_gpu) continue; /* already on GPU */

        /* Only promote if compressed FFN is available for this layer */
        const axex_compressed_weight_t *cd2 = axex_get_compressed_layer(L, 0);
        if (!cd2 || !cd2->U) continue; /* not compressed → skip */

        const llm_layer_t *layer = &m2->layers[L];
        int lhd     = layer->head_dim_layer ? layer->head_dim_layer : hd;
        int lq_dim  = m2->n_heads * lhd;
        int lkv_dim = n_kv * lhd;

        /* Estimate VRAM needed: attention weights + norms + KV + compressed FFN/GP */
        uint64_t attn_bytes = 0;
        int gp_k_est = axex_manifold_k();
        /* Use GP size if manifold is available for this layer, else original size */
        const axex_manifold_weight_t *_gp_q_est = axex_get_manifold_layer(L, 1);
        const axex_manifold_weight_t *_gp_k_est = axex_get_manifold_layer(L, 2);
        const axex_manifold_weight_t *_gp_v_est = axex_get_manifold_layer(L, 3);
        const axex_manifold_weight_t *_gp_o_est = axex_get_manifold_layer(L, 4);
        attn_bytes += _gp_q_est ? (uint64_t)lq_dim * gp_k_est * sizeof(float)
                                : llm_row_bytes(dim, layer->q_type) * (uint64_t)lq_dim;
        if (layer->kv_reuse_layer < 0) {
            attn_bytes += _gp_k_est ? (uint64_t)lkv_dim * gp_k_est * sizeof(float)
                                    : llm_row_bytes(dim, layer->k_type) * (uint64_t)lkv_dim;
            attn_bytes += _gp_v_est ? (uint64_t)lkv_dim * gp_k_est * sizeof(float)
                                    : llm_row_bytes(dim, layer->v_type) * (uint64_t)lkv_dim;
        }
        attn_bytes += _gp_o_est ? (uint64_t)dim * gp_k_est * sizeof(float)
                                 : llm_row_bytes(lq_dim, layer->o_type) * (uint64_t)dim;
        attn_bytes += (uint64_t)dim * sizeof(float) * 4; /* norms */
        if (layer->kv_reuse_layer < 0)
            attn_bytes += (uint64_t)n_kv * max_seq * lhd * sizeof(float) * 2; /* KV cache */

        /* Compressed FFN F32 sizes (SVD path) */
        for (int slot = 0; slot < AXEX_COMPRESS_WHICH_MAX; slot++) {
            axex_compressed_weight_t *cw2 = axex_get_compressed_layer_mut(L, slot);
            if (!cw2 || cw2->d_Vt) continue; /* not available or already on GPU */
            attn_bytes += (uint64_t)cw2->rank * (cw2->m + cw2->n) * sizeof(float);
        }
        /* GP W_proj sizes (gate/up slots 5-6 not yet uploaded) */
        for (int slot = 5; slot <= 6; slot++) {
            const axex_manifold_weight_t *mw_est = axex_get_manifold_layer(L, slot);
            if (mw_est && !mw_est->d_W_proj && mw_est->m > 0)
                attn_bytes += (uint64_t)mw_est->m * gp_k_est * sizeof(float);
        }

        uint64_t vram_free = be->get_free_memory(0);
        if (vram_free < attn_bytes + (512ULL << 20)) break; /* keep 512MB headroom */

        /* Upload attention weights */
        llm_gpu_upload_weight_mat(layer->q_weight, lq_dim, dim, layer->q_type);
        if (layer->kv_reuse_layer < 0) {
            llm_gpu_upload_weight_mat(layer->k_weight, lkv_dim, dim, layer->k_type);
            llm_gpu_upload_weight_mat(layer->v_weight, lkv_dim, dim, layer->v_type);
        }
        llm_gpu_upload_weight_mat(layer->o_weight, dim, lq_dim, layer->o_type);
        if (layer->iswa_inp_gate)
            llm_gpu_upload_weight_mat(layer->iswa_inp_gate, m2->iswa_n_embd,
                                      dim, layer->iswa_inp_gate_type);
        if (layer->iswa_proj)
            llm_gpu_upload_weight_mat(layer->iswa_proj, dim, m2->iswa_n_embd,
                                      layer->iswa_proj_type);

        /* Upload norms */
        gl->d_attn_norm = llm_gpu_upload_norm(layer->attn_norm, dim, layer->attn_norm_type);
        gl->d_ffn_norm  = llm_gpu_upload_norm(layer->ffn_norm,  dim, layer->ffn_norm_type);
        gl->d_post_attn_norm = llm_gpu_upload_norm(layer->post_attn_norm, dim, GGML_TYPE_F32);
        gl->d_post_ffw_norm  = llm_gpu_upload_norm(layer->post_ffw_norm,  dim, GGML_TYPE_F32);
        if (layer->q_norm)
            gl->d_q_norm = llm_gpu_upload_norm(layer->q_norm, lhd, GGML_TYPE_F32);
        if (layer->k_norm)
            gl->d_k_norm = llm_gpu_upload_norm(layer->k_norm, lhd, GGML_TYPE_F32);
        if (layer->iswa_post_norm)
            gl->d_iswa_post_norm = llm_gpu_upload_norm(layer->iswa_post_norm, dim, GGML_TYPE_F32);

        if (!gl->d_attn_norm || !gl->d_ffn_norm) {
            /* Norm upload failed — don't promote this layer */
            gl->d_attn_norm = NULL; gl->d_ffn_norm = NULL;
            continue;
        }

        /* Alloc KV cache */
        gl->lhd = lhd;
        if (layer->kv_reuse_layer < 0) {
            uint64_t kv_sz = (uint64_t)n_kv * max_seq * lhd * sizeof(float);
            gl->d_k_cache = (float *)be->mem.alloc(kv_sz);
            gl->d_v_cache = (float *)be->mem.alloc(kv_sz);
            if (!gl->d_k_cache || !gl->d_v_cache) {
                if (gl->d_k_cache) { be->mem.free(gl->d_k_cache); gl->d_k_cache = NULL; }
                if (gl->d_v_cache) { be->mem.free(gl->d_v_cache); gl->d_v_cache = NULL; }
                continue;
            }
        }

        gl->on_gpu = 1;
        promoted++;

        /* Upload compressed FFN for this newly promoted layer */
        for (int slot = 0; slot < AXEX_COMPRESS_WHICH_MAX; slot++) {
            axex_compressed_weight_t *cw2 = axex_get_compressed_layer_mut(L, slot);
            if (!cw2 || cw2->d_Vt) continue;
            axex_upload_compressed_to_gpu(cw2,
                (void *(*)(uint64_t))be->mem.alloc,
                (void (*)(void *, const void *, uint64_t))be->mem.upload,
                be->mem.sync);
        }
    }

    if (promoted)
        kprintf("[GPU] Promoted %d CPU layers to GPU (compressed-FFN-only path)\n",
                promoted);

    be->mem.sync();
#endif
}


int llm_prompt(const char *user_text, char *output, int max_output)
{
    if (!llm_is_loaded()) {
        kstrlcpy(output, "[no model loaded]", max_output);
        return -1;
    }

    if (!llm_stream_cb && !llm_think_show) {
        int n_gen = llm_prompt_tokens(user_text, llm_tokens, llm_alloc_tokens, 1024, 0.7f);
        if (n_gen < 0) {
            kstrlcpy(output, "[generation failed]", max_output);
            return -1;
        }
        llm_decode_tokens(llm_tokens, n_gen, output, max_output);
        return n_gen;
    }

    /* Serialize: only one inference at a time (static buffers not reentrant) */
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) {
        kstrlcpy(output, "[inference busy — try again]", max_output);
        return -1;
    }

    llm_model_t *m = &llm_model;

    /* Build chat prompt using the model's preferred format */
    static char prompt_buf[2048];
    const int buf_max = (int)sizeof(prompt_buf) - 1;
    int pos = 0;

    /* Safe append: copies at most (buf_max - pos) bytes */
    #define PROMPT_APPEND(s) do { \
        const char *_s = (s); int _l = kstrlen(_s); \
        if (pos + _l > buf_max) _l = buf_max - pos; \
        if (_l > 0) { kmemcpy(prompt_buf + pos, _s, _l); pos += _l; } \
    } while(0)

    /* Detect prompt format */
    int is_chatml = 0;
    int is_phi3   = 0;
    int is_phi2   = 0;
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'q' && m->arch[1] == 'w' &&
        m->arch[2] == 'e' && m->arch[3] == 'n')
        is_chatml = 1;
    if (kstrlen(m->arch) >= 3 &&
        m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i') {
        if (m->arch[3] == '2' || m->arch[3] == '\0')
            is_phi2 = 1;
        else
            is_phi3 = 1;
    }

    if (is_phi3) {
        PROMPT_APPEND("<|user|>\n");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("<|end|>\n<|assistant|>\n");
    } else if (is_phi2) {
        PROMPT_APPEND("Instruct: ");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("\nOutput: ");
    } else if (is_chatml) {
        PROMPT_APPEND("<|im_start|>system\nYou are a helpful AI assistant running on TensorOS, a bare-metal AI operating system.<|im_end|>\n<|im_start|>user\n");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("<|im_end|>\n<|im_start|>assistant\n");
    } else if (kstrlen(m->arch) >= 5 &&
               m->arch[0] == 'g' && m->arch[1] == 'e' &&
               m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a') {
        PROMPT_APPEND("<turn|>user\n");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("\n<turn|>model\n");
    } else {
        /* Generic / LLaMA / SmolLM — simple prompt */
        PROMPT_APPEND("User: ");
        PROMPT_APPEND(user_text);
        PROMPT_APPEND("\nAssistant: ");
    }
    prompt_buf[pos] = '\0';
    #undef PROMPT_APPEND

    /* Tokenize */
    int n_tokens = llm_tokenize(m, prompt_buf, llm_tokens + 1, llm_alloc_tokens - 65);
    /* Prepend BOS token */
    if (n_tokens > 0 && m->bos_id >= 0) {
        llm_tokens[0] = m->bos_id;
        n_tokens++;
    } else {
        for (int i = 0; i < n_tokens; i++)
            llm_tokens[i] = llm_tokens[i + 1];
    }
    if (n_tokens <= 0) {
        __sync_lock_release(&llm_inference_active);
        kstrlcpy(output, "[tokenization failed]", max_output);
        return -1;
    }

    /* Generate: max 1024 tokens, temperature 0.7 with top-k/top-p sampling */
    int n_gen = llm_generate(m, llm_tokens, n_tokens, output, max_output,
                             1024, 0.7f, 0);
    __sync_lock_release(&llm_inference_active);
    return n_gen;
}

/* Like llm_prompt but with an explicit max-token cap */
int llm_prompt_n(const char *user_text, char *output, int max_output, int max_tokens)
{
    if (!llm_is_loaded()) {
        kstrlcpy(output, "[no model loaded]", max_output);
        return -1;
    }

    if (!llm_stream_cb && !llm_think_show) {
        int n_gen = llm_prompt_tokens(user_text, llm_tokens, llm_alloc_tokens,
                                      max_tokens > 0 ? max_tokens : 16, llm_default_temperature);
        if (n_gen < 0) {
            kstrlcpy(output, "[generation failed]", max_output);
            return -1;
        }
        llm_decode_tokens(llm_tokens, n_gen, output, max_output);
        return n_gen;
    }

    if (__sync_lock_test_and_set(&llm_inference_active, 1)) {
        kstrlcpy(output, "[inference busy]", max_output);
        return -1;
    }

    llm_model_t *m = &llm_model;

    /* Build prompt with chat template (same as llm_prompt) */
    static char prompt_buf[512];
    int pos = 0;
    const int buf_max = (int)sizeof(prompt_buf) - 1;
    #define PA(s) do { \
        const char *_s = (s); int _l = kstrlen(_s); \
        if (pos + _l > buf_max) _l = buf_max - pos; \
        if (_l > 0) { kmemcpy(prompt_buf + pos, _s, _l); pos += _l; } \
    } while(0)

    /* Detect prompt format from architecture */
    int is_phi3 = 0, is_phi2 = 0, is_chatml = 0, is_gemma = 0, is_llama = 0;
    if (kstrlen(m->arch) >= 4 &&
        m->arch[0] == 'q' && m->arch[1] == 'w' &&
        m->arch[2] == 'e' && m->arch[3] == 'n')
        is_chatml = 1;
    if (kstrlen(m->arch) >= 3 &&
        m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i') {
        if (m->arch[3] == '2' || m->arch[3] == '\0')
            is_phi2 = 1;
        else
            is_phi3 = 1;
    }
    if (kstrlen(m->arch) >= 5 &&
        m->arch[0] == 'g' && m->arch[1] == 'e' &&
        m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a')
        is_gemma = 1;
    if (kstrlen(m->arch) >= 5 &&
        m->arch[0] == 'l' && m->arch[1] == 'l' &&
        m->arch[2] == 'a' && m->arch[3] == 'm' && m->arch[4] == 'a')
        is_llama = 1;
    /* SmolLM models report llama arch but use ChatML format.
     * Detect by model name substring "moll" (matches SmolLM, Smollm2, etc.) */
    if (is_llama && !is_chatml && llm_strstr(m->name, "moll")) {
        is_chatml = 1;
        is_llama  = 0;
    }

    if (is_gemma) {
        PA("<turn|>user\n");
        PA(user_text);
        PA("\n<turn|>model\n");
    } else if (is_phi3) {
        PA("<|user|>\n");
        PA(user_text);
        PA("<|end|>\n<|assistant|>\n");
    } else if (is_phi2) {
        PA("Instruct: ");
        PA(user_text);
        PA("\nOutput: ");
    } else if (is_chatml) {
        PA("<|im_start|>user\n");
        PA(user_text);
        PA("<|im_end|>\n<|im_start|>assistant\n");
    } else if (is_llama) {
        PA("<|start_header_id|>user<|end_header_id|>\n\n");
        PA(user_text);
        PA("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    } else {
        PA(user_text);
    }
    prompt_buf[pos] = '\0';
    #undef PA

    int n_tokens = llm_tokenize(m, prompt_buf, llm_tokens + 1, llm_alloc_tokens - 65);
    /* Prepend BOS token */
    if (n_tokens > 0 && m->bos_id >= 0) {
        llm_tokens[0] = m->bos_id;
        n_tokens++;
    } else {
        /* No BOS: shift tokens to start of array */
        for (int i = 0; i < n_tokens; i++)
            llm_tokens[i] = llm_tokens[i + 1];
    }

    /* Inject <|think|> token for models that support thinking.
     * Only inject when llm_think_enabled == 2 (force-think mode).
     * Default (1) = detect and handle organically when model emits it. */
    if (llm_think_enabled >= 2 && is_gemma && n_tokens > 0 &&
        n_tokens < llm_alloc_tokens - 2) {
        llm_tokens[n_tokens++] = LLM_THINK_TOKEN;
        LOG_DBG("THINK", "force-injected <|think|> token into prompt");
    }

    if (n_tokens <= 0) {
        __sync_lock_release(&llm_inference_active);
        kstrlcpy(output, "[tokenization failed]", max_output);
        return -1;
    }

    if (max_tokens < 1) max_tokens = 16;
    int n_gen = llm_generate(m, llm_tokens, n_tokens, output, max_output,
                             max_tokens, llm_default_temperature, 0);
    __sync_lock_release(&llm_inference_active);
    return n_gen;
}

/* Reset KV cache for starting a new conversation */
void llm_reset_cache(void)
{
    if (!llm_is_loaded()) return;
    llm_model_t *m = &llm_model;
    m->cache_len = 0;
    llm_logits_pos = -1;
    llm_primed_greedy = -1;
#ifdef ENABLE_CUDA
    /* Invalidate any captured CUDA graph: it was captured at a specific
     * position with a specific KV-cache state.  After a full cache reset
     * the graph is stale and must be re-captured on the next decode step. */
    cuda_graph_captured    = 0;
    cuda_graph_tried       = 0;
    /* Leave cuda_graph_decode_ready as-is: the next llm_generate_token_ids
     * call will set it appropriately; resetting here could block re-capture. */
#endif
    int64_t kv_total = (int64_t)m->n_layers * m->max_seq * m->n_kv_heads * m->head_dim;
    if (kv_total > llm_alloc_kv_floats) kv_total = llm_alloc_kv_floats;
    kmemset(m->k_cache, 0, (uint64_t)kv_total * sizeof(float));
    kmemset(m->v_cache, 0, (uint64_t)kv_total * sizeof(float));
#ifdef ENABLE_CUDA
    if (gpu_ctx.gpu_fwd) {
        const backend_t *be = backend_get_by_id(BACKEND_CUDA);
        uint64_t kv_layer_bytes = (uint64_t)m->n_kv_heads * m->max_seq * m->head_dim * sizeof(float);
        for (int L = 0; L < m->n_layers; L++) {
            if (m->layers[L].kv_reuse_layer < 0 && gpu_ctx.layers[L].d_k_cache) {
                be->mem.upload(gpu_ctx.layers[L].d_k_cache, m->k_cache, kv_layer_bytes);
                be->mem.upload(gpu_ctx.layers[L].d_v_cache, m->v_cache, kv_layer_bytes);
            }
        }
    }
#endif
}

/* Reset chat KV context */
void llm_chat_reset(void) { llm_reset_cache(); }

/* How many KV positions are currently occupied */
int llm_chat_context_tokens(void) {
    if (!llm_is_loaded()) return 0;
    return (int)llm_model.cache_len;
}

/* Maximum context window for the loaded model */
int llm_chat_context_max(void) {
    if (!llm_is_loaded()) return 0;
    return (int)llm_model.max_seq;
}

/**
 * Multi-turn chat: maintains KV cache across calls.
 * Each call appends this user turn to the existing context.
 * Call llm_chat_reset() to start a new conversation.
 */
int llm_chat_turn(const char *user_text, char *output, int max_output,
                  int max_tokens, float temperature)
{
    if (!llm_is_loaded()) {
        kstrlcpy(output, "[no model loaded]", max_output);
        return -1;
    }

    if (!llm_stream_cb && !llm_think_show) {
        int n_gen = llm_chat_turn_tokens(user_text, llm_tokens, llm_alloc_tokens,
                                         max_tokens > 0 ? max_tokens : 256,
                                         temperature);
        if (n_gen < 0) {
            kstrlcpy(output, "[generation failed]", max_output);
            return -1;
        }
        llm_decode_tokens(llm_tokens, n_gen, output, max_output);
        return n_gen;
    }

    if (__sync_lock_test_and_set(&llm_inference_active, 1)) {
        kstrlcpy(output, "[inference busy]", max_output);
        return -1;
    }

    llm_model_t *m = &llm_model;
    int first_turn = (m->cache_len == 0);

    /* Build only this turn's formatted prompt;
     * previous turns are already baked into the KV cache. */
    static char turn_buf[4096];
    int pos = 0;
    const int buf_max = (int)sizeof(turn_buf) - 1;
    #define TA(s) do { \
        const char *_s = (s); int _l = kstrlen(_s); \
        if (pos + _l > buf_max) _l = buf_max - pos; \
        if (_l > 0) { kmemcpy(turn_buf + pos, _s, _l); pos += _l; } \
    } while(0)

    int is_gemma = kstrlen(m->arch) >= 5 &&
        m->arch[0]=='g' && m->arch[1]=='e' && m->arch[2]=='m' &&
        m->arch[3]=='m' && m->arch[4]=='a';
    int is_phi3 = kstrlen(m->arch) >= 3 &&
        m->arch[0]=='p' && m->arch[1]=='h' && m->arch[2]=='i' &&
        m->arch[3] != '2' && m->arch[3] != '\0';
    int is_phi2 = kstrlen(m->arch) >= 3 &&
        m->arch[0]=='p' && m->arch[1]=='h' && m->arch[2]=='i' &&
        (m->arch[3] == '2' || m->arch[3] == '\0');
    int is_chatml = kstrlen(m->arch) >= 4 &&
        m->arch[0]=='q' && m->arch[1]=='w' && m->arch[2]=='e' && m->arch[3]=='n';
    int is_llama = kstrlen(m->arch) >= 5 &&
        m->arch[0]=='l' && m->arch[1]=='l' && m->arch[2]=='a' &&
        m->arch[3]=='m' && m->arch[4]=='a';

    if (is_gemma) {
        TA("<turn|>user\n");
        TA(user_text);
        TA("\n<turn|>model\n");
    } else if (is_phi3) {
        TA("<|user|>\n");
        TA(user_text);
        TA("<|end|>\n<|assistant|>\n");
    } else if (is_phi2) {
        TA("Instruct: ");
        TA(user_text);
        TA("\nOutput: ");
    } else if (is_chatml) {
        if (first_turn) {
            TA("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n");
        }
        TA("<|im_start|>user\n");
        TA(user_text);
        TA("<|im_end|>\n<|im_start|>assistant\n");
    } else if (is_llama) {
        TA("<|start_header_id|>user<|end_header_id|>\n\n");
        TA(user_text);
        TA("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
    } else {
        TA(user_text);
        TA("\n");
    }
    turn_buf[pos] = '\0';
    #undef TA

    /* Tokenize: prepend BOS only on the first turn */
    int n_tokens = llm_tokenize(m, turn_buf, llm_tokens + 1, llm_alloc_tokens - 65);
    if (first_turn && m->bos_id >= 0) {
        llm_tokens[0] = m->bos_id;
        n_tokens++;
    } else {
        for (int i = 0; i < n_tokens; i++) llm_tokens[i] = llm_tokens[i + 1];
    }

    /* Inject <|think|> token for models that support thinking.
     * Only inject when llm_think_enabled == 2 (force-think mode).
     * Default (1) = detect and handle organically when model emits it. */
    if (llm_think_enabled >= 2 && is_gemma && n_tokens > 0 &&
        n_tokens < llm_alloc_tokens - 2) {
        llm_tokens[n_tokens++] = LLM_THINK_TOKEN;
        LOG_DBG("THINK", "force-injected <|think|> into chat turn");
    }

    if (n_tokens <= 0) {
        __sync_lock_release(&llm_inference_active);
        kstrlcpy(output, "[tokenization failed]", max_output);
        return -1;
    }

    if (max_tokens < 1) max_tokens = 256;
    /* continue_cache=0 for first turn (resets KV cache), 1 for subsequent turns */
    int n_gen = llm_generate(m, llm_tokens, n_tokens, output, max_output,
                             max_tokens, temperature, first_turn ? 0 : 1);
    __sync_lock_release(&llm_inference_active);
    return n_gen;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Token-native API implementations
 *
 * All functions below operate in pure token-ID space, avoiding text
 * serialization at internal machine-to-machine boundaries.
 * ═══════════════════════════════════════════════════════════════════════════ */

int llm_tokenize_text(const char *text, int *tokens, int max_tokens)
{
    if (!llm_is_loaded() || !text || !tokens || max_tokens <= 0) return -1;
    return llm_tokenize(&llm_model, text, tokens, max_tokens);
}

int llm_decode_tokens(const int *tokens, int n_tokens, char *output, int max_output)
{
    if (!llm_is_loaded() || !output || max_output <= 0) return -1;
    return llm_decode_tokens_to_text(&llm_model, tokens, n_tokens, output, max_output);
}

static const char *llm_exec_skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

typedef struct {
    int64_t num;
    int64_t den;
} llm_exec_rational_t;

typedef struct {
    const llm_model_t *m;
    const int *tokens;
    int n_tokens;
    int token_idx;
    int char_idx;
    char tok_buf[128];
    int tok_len;
} llm_exec_tokcur_t;

static int64_t llm_exec_abs64(int64_t x)
{
    return x < 0 ? -x : x;
}

static int64_t llm_exec_gcd64(int64_t a, int64_t b)
{
    a = llm_exec_abs64(a);
    b = llm_exec_abs64(b);
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a == 0 ? 1 : a;
}

static int llm_exec_rational_norm(llm_exec_rational_t *r)
{
    if (!r || r->den == 0) return -1;
    if (r->den < 0) {
        r->num = -r->num;
        r->den = -r->den;
    }
    if (r->num == 0) {
        r->den = 1;
        return 0;
    }
    {
        int64_t g = llm_exec_gcd64(r->num, r->den);
        r->num /= g;
        r->den /= g;
    }
    return 0;
}

static int llm_exec_rational_add(llm_exec_rational_t a, llm_exec_rational_t b,
                                 llm_exec_rational_t *out)
{
    int64_t g = llm_exec_gcd64(a.den, b.den);
    int64_t ad = a.den / g;
    int64_t bd = b.den / g;
    int64_t num = a.num * bd + b.num * ad;
    int64_t den = a.den * bd;
    out->num = num;
    out->den = den;
    return llm_exec_rational_norm(out);
}

static int llm_exec_rational_sub(llm_exec_rational_t a, llm_exec_rational_t b,
                                 llm_exec_rational_t *out)
{
    b.num = -b.num;
    return llm_exec_rational_add(a, b, out);
}

static int llm_exec_rational_mul(llm_exec_rational_t a, llm_exec_rational_t b,
                                 llm_exec_rational_t *out)
{
    int64_t g1 = llm_exec_gcd64(a.num, b.den);
    int64_t g2 = llm_exec_gcd64(b.num, a.den);
    int64_t num = (a.num / g1) * (b.num / g2);
    int64_t den = (a.den / g2) * (b.den / g1);
    out->num = num;
    out->den = den;
    return llm_exec_rational_norm(out);
}

static int llm_exec_rational_div(llm_exec_rational_t a, llm_exec_rational_t b,
                                 llm_exec_rational_t *out)
{
    if (b.num == 0) return -1;
    {
        llm_exec_rational_t inv;
        inv.num = b.den;
        inv.den = b.num;
        if (llm_exec_rational_norm(&inv) != 0) return -1;
        return llm_exec_rational_mul(a, inv, out);
    }
}

static int llm_exec_tokcur_load(llm_exec_tokcur_t *c)
{
    if (!c) return -1;
    while (c->token_idx < c->n_tokens) {
        int n = llm_decode_token(c->m, c->tokens[c->token_idx], c->tok_buf,
                                 (int)sizeof(c->tok_buf));
        c->token_idx++;
        c->char_idx = 0;
        c->tok_len = n > 0 ? n : 0;
        if (c->tok_len > 0) return 0;
    }
    c->tok_len = 0;
    return -1;
}

static int llm_exec_tokcur_peek(llm_exec_tokcur_t *c)
{
    if (!c) return -1;
    while (c->char_idx >= c->tok_len) {
        if (llm_exec_tokcur_load(c) != 0)
            return -1;
    }
    return (uint8_t)c->tok_buf[c->char_idx];
}

static int llm_exec_tokcur_get(llm_exec_tokcur_t *c)
{
    int ch = llm_exec_tokcur_peek(c);
    if (ch < 0) return -1;
    c->char_idx++;
    return ch;
}

static void llm_exec_tokcur_skip_ws(llm_exec_tokcur_t *c)
{
    for (;;) {
        int ch = llm_exec_tokcur_peek(c);
        if (ch < 0) break;
        if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') break;
        (void)llm_exec_tokcur_get(c);
    }
}

static int llm_exec_tokcur_has_tail(llm_exec_tokcur_t *c)
{
    llm_exec_tokcur_skip_ws(c);
    return llm_exec_tokcur_peek(c) >= 0;
}

static int llm_exec_parse_number_tok(llm_exec_tokcur_t *c,
                                     llm_exec_rational_t *out)
{
    int saw_digit = 0;
    int saw_dot = 0;
    int64_t int_part = 0;
    int64_t frac_part = 0;
    int64_t frac_scale = 1;

    for (;;) {
        int ch = llm_exec_tokcur_peek(c);
        if (ch < '0' || ch > '9') break;
        saw_digit = 1;
        int_part = int_part * 10 + (ch - '0');
        (void)llm_exec_tokcur_get(c);
    }

    if (llm_exec_tokcur_peek(c) == '.') {
        saw_dot = 1;
        (void)llm_exec_tokcur_get(c);
        for (;;) {
            int ch = llm_exec_tokcur_peek(c);
            if (ch < '0' || ch > '9') break;
            saw_digit = 1;
            frac_part = frac_part * 10 + (ch - '0');
            frac_scale *= 10;
            (void)llm_exec_tokcur_get(c);
        }
    }

    if (!saw_digit) return -1;

    if (saw_dot) {
        out->num = int_part * frac_scale + frac_part;
        out->den = frac_scale;
    } else {
        out->num = int_part;
        out->den = 1;
    }

    llm_exec_tokcur_skip_ws(c);
    if (llm_exec_tokcur_peek(c) == '/') {
        int64_t den = 0;
        int saw_den = 0;
        (void)llm_exec_tokcur_get(c);
        llm_exec_tokcur_skip_ws(c);
        for (;;) {
            int ch = llm_exec_tokcur_peek(c);
            if (ch < '0' || ch > '9') break;
            saw_den = 1;
            den = den * 10 + (ch - '0');
            (void)llm_exec_tokcur_get(c);
        }
        if (!saw_den || den == 0) return -1;
        out->den *= den;
    }

    return llm_exec_rational_norm(out);
}

static int llm_exec_parse_expr_tok(llm_exec_tokcur_t *c,
                                   llm_exec_rational_t *out);

static int llm_exec_parse_factor_tok(llm_exec_tokcur_t *c,
                                     llm_exec_rational_t *out)
{
    int sign = 1;
    llm_exec_tokcur_skip_ws(c);
    {
        int ch = llm_exec_tokcur_peek(c);
        if (ch == '+' || ch == '-') {
            if (ch == '-') sign = -1;
            (void)llm_exec_tokcur_get(c);
        }
    }

    llm_exec_tokcur_skip_ws(c);
    if (llm_exec_tokcur_peek(c) == '(') {
        (void)llm_exec_tokcur_get(c);
        if (llm_exec_parse_expr_tok(c, out) != 0) return -1;
        llm_exec_tokcur_skip_ws(c);
        if (llm_exec_tokcur_get(c) != ')') return -1;
        out->num *= sign;
        return llm_exec_rational_norm(out);
    }

    if (llm_exec_parse_number_tok(c, out) != 0) return -1;
    out->num *= sign;
    return llm_exec_rational_norm(out);
}

static int llm_exec_parse_term_tok(llm_exec_tokcur_t *c,
                                   llm_exec_rational_t *out)
{
    llm_exec_rational_t rhs;
    if (llm_exec_parse_factor_tok(c, out) != 0) return -1;

    for (;;) {
        int ch;
        llm_exec_tokcur_skip_ws(c);
        ch = llm_exec_tokcur_peek(c);
        if (ch != '*' && ch != '/') break;
        (void)llm_exec_tokcur_get(c);
        if (llm_exec_parse_factor_tok(c, &rhs) != 0) return -1;
        if (ch == '*') {
            if (llm_exec_rational_mul(*out, rhs, out) != 0) return -1;
        } else {
            if (llm_exec_rational_div(*out, rhs, out) != 0) return -1;
        }
    }
    return 0;
}

static int llm_exec_parse_expr_tok(llm_exec_tokcur_t *c,
                                   llm_exec_rational_t *out)
{
    llm_exec_rational_t rhs;
    if (llm_exec_parse_term_tok(c, out) != 0) return -1;

    for (;;) {
        int ch;
        llm_exec_tokcur_skip_ws(c);
        ch = llm_exec_tokcur_peek(c);
        if (ch != '+' && ch != '-') break;
        (void)llm_exec_tokcur_get(c);
        if (llm_exec_parse_term_tok(c, &rhs) != 0) return -1;
        if (ch == '+') {
            if (llm_exec_rational_add(*out, rhs, out) != 0) return -1;
        } else {
            if (llm_exec_rational_sub(*out, rhs, out) != 0) return -1;
        }
    }
    return 0;
}

static int llm_exec_i64_to_str(int64_t v, char *out, int max_len)
{
    int pos = 0;
    if (!out || max_len <= 1) return -1;
    if (v == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 1;
    }
    if (v < 0) {
        if (pos >= max_len - 1) return -1;
        out[pos++] = '-';
    }
    {
        uint64_t u = (uint64_t)(v < 0 ? -v : v);
        char tmp[32];
        int t = 0;
        while (u > 0 && t < (int)sizeof(tmp)) {
            tmp[t++] = (char)('0' + (u % 10));
            u /= 10;
        }
        while (t > 0 && pos < max_len - 1)
            out[pos++] = tmp[--t];
    }
    out[pos] = '\0';
    return pos;
}

static int llm_exec_rational_to_tokens(const llm_exec_rational_t *r,
                                       int *output_tokens, int max_output_tokens)
{
    char out[96];
    int n;
    if (!r || !output_tokens || max_output_tokens <= 0) return -1;

    if (r->den == 1) {
        n = llm_exec_i64_to_str(r->num, out, (int)sizeof(out));
        if (n <= 0) return -1;
    } else {
        int a = llm_exec_i64_to_str(r->num, out, (int)sizeof(out));
        if (a <= 0 || a >= (int)sizeof(out) - 1) return -1;
        out[a++] = '/';
        n = llm_exec_i64_to_str(r->den, out + a, (int)sizeof(out) - a);
        if (n <= 0) return -1;
    }
    return llm_tokenize(&llm_model, out, output_tokens, max_output_tokens);
}

static int llm_exec_json_parse_value(llm_exec_tokcur_t *c, int depth);

static int llm_exec_json_expect_literal(llm_exec_tokcur_t *c, const char *lit)
{
    int i = 0;
    while (lit[i]) {
        int ch = llm_exec_tokcur_get(c);
        if (ch != lit[i]) return -1;
        i++;
    }
    return 0;
}

static int llm_exec_json_parse_string(llm_exec_tokcur_t *c)
{
    if (llm_exec_tokcur_get(c) != '"') return -1;
    for (;;) {
        int ch = llm_exec_tokcur_get(c);
        if (ch < 0) return -1;
        if (ch == '"') return 0;
        if ((uint8_t)ch < 0x20) return -1;
        if (ch == '\\') {
            int esc = llm_exec_tokcur_get(c);
            if (esc < 0) return -1;
            if (esc == '"' || esc == '\\' || esc == '/' || esc == 'b' ||
                esc == 'f' || esc == 'n' || esc == 'r' || esc == 't') {
                continue;
            }
            if (esc == 'u') {
                for (int i = 0; i < 4; i++) {
                    int hx = llm_exec_tokcur_get(c);
                    if (!((hx >= '0' && hx <= '9') ||
                          (hx >= 'a' && hx <= 'f') ||
                          (hx >= 'A' && hx <= 'F'))) {
                        return -1;
                    }
                }
                continue;
            }
            return -1;
        }
    }
}

static int llm_exec_json_parse_number(llm_exec_tokcur_t *c)
{
    int ch = llm_exec_tokcur_peek(c);
    int saw_digit = 0;

    if (ch == '-') {
        (void)llm_exec_tokcur_get(c);
        ch = llm_exec_tokcur_peek(c);
    }

    if (ch == '0') {
        saw_digit = 1;
        (void)llm_exec_tokcur_get(c);
        ch = llm_exec_tokcur_peek(c);
    } else {
        while (ch >= '0' && ch <= '9') {
            saw_digit = 1;
            (void)llm_exec_tokcur_get(c);
            ch = llm_exec_tokcur_peek(c);
        }
    }
    if (!saw_digit) return -1;

    if (ch == '.') {
        int saw_frac = 0;
        (void)llm_exec_tokcur_get(c);
        ch = llm_exec_tokcur_peek(c);
        while (ch >= '0' && ch <= '9') {
            saw_frac = 1;
            (void)llm_exec_tokcur_get(c);
            ch = llm_exec_tokcur_peek(c);
        }
        if (!saw_frac) return -1;
    }

    if (ch == 'e' || ch == 'E') {
        int saw_exp = 0;
        (void)llm_exec_tokcur_get(c);
        ch = llm_exec_tokcur_peek(c);
        if (ch == '+' || ch == '-') {
            (void)llm_exec_tokcur_get(c);
            ch = llm_exec_tokcur_peek(c);
        }
        while (ch >= '0' && ch <= '9') {
            saw_exp = 1;
            (void)llm_exec_tokcur_get(c);
            ch = llm_exec_tokcur_peek(c);
        }
        if (!saw_exp) return -1;
    }

    return 0;
}

static int llm_exec_json_parse_array(llm_exec_tokcur_t *c, int depth)
{
    if (llm_exec_tokcur_get(c) != '[') return -1;
    llm_exec_tokcur_skip_ws(c);
    if (llm_exec_tokcur_peek(c) == ']') {
        (void)llm_exec_tokcur_get(c);
        return 0;
    }

    for (;;) {
        if (llm_exec_json_parse_value(c, depth + 1) != 0) return -1;
        llm_exec_tokcur_skip_ws(c);
        {
            int ch = llm_exec_tokcur_get(c);
            if (ch == ']') return 0;
            if (ch != ',') return -1;
        }
        llm_exec_tokcur_skip_ws(c);
    }
}

static int llm_exec_json_parse_object(llm_exec_tokcur_t *c, int depth)
{
    if (llm_exec_tokcur_get(c) != '{') return -1;
    llm_exec_tokcur_skip_ws(c);
    if (llm_exec_tokcur_peek(c) == '}') {
        (void)llm_exec_tokcur_get(c);
        return 0;
    }

    for (;;) {
        if (llm_exec_json_parse_string(c) != 0) return -1;
        llm_exec_tokcur_skip_ws(c);
        if (llm_exec_tokcur_get(c) != ':') return -1;
        llm_exec_tokcur_skip_ws(c);
        if (llm_exec_json_parse_value(c, depth + 1) != 0) return -1;
        llm_exec_tokcur_skip_ws(c);
        {
            int ch = llm_exec_tokcur_get(c);
            if (ch == '}') return 0;
            if (ch != ',') return -1;
        }
        llm_exec_tokcur_skip_ws(c);
        if (llm_exec_tokcur_peek(c) != '"') return -1;
    }
}

static int llm_exec_json_parse_value(llm_exec_tokcur_t *c, int depth)
{
    int ch;
    if (depth > 256) return -1;

    llm_exec_tokcur_skip_ws(c);
    ch = llm_exec_tokcur_peek(c);
    if (ch < 0) return -1;

    if (ch == '"') return llm_exec_json_parse_string(c);
    if (ch == '{') return llm_exec_json_parse_object(c, depth);
    if (ch == '[') return llm_exec_json_parse_array(c, depth);
    if (ch == 't') return llm_exec_json_expect_literal(c, "true");
    if (ch == 'f') return llm_exec_json_expect_literal(c, "false");
    if (ch == 'n') return llm_exec_json_expect_literal(c, "null");
    return llm_exec_json_parse_number(c);
}

static int llm_exec_parse_expr(const char **pp, int *ok);

static int llm_exec_parse_factor(const char **pp, int *ok)
{
    const char *p = llm_exec_skip_ws(*pp);
    int sign = 1;
    if (*p == '+') { p++; }
    else if (*p == '-') { sign = -1; p++; }

    p = llm_exec_skip_ws(p);
    if (*p == '(') {
        p++;
        int v = llm_exec_parse_expr(&p, ok);
        p = llm_exec_skip_ws(p);
        if (*p != ')') { *ok = 0; return 0; }
        p++;
        *pp = p;
        return sign * v;
    }

    if (*p < '0' || *p > '9') { *ok = 0; return 0; }
    {
        int v = 0;
        while (*p >= '0' && *p <= '9') {
            v = v * 10 + (*p - '0');
            p++;
        }
        *pp = p;
        return sign * v;
    }
}

static int llm_exec_parse_term(const char **pp, int *ok)
{
    int v = llm_exec_parse_factor(pp, ok);
    if (!*ok) return 0;
    const char *p = *pp;
    for (;;) {
        p = llm_exec_skip_ws(p);
        if (*p != '*' && *p != '/') break;
        char op = *p++;
        int rhs = llm_exec_parse_factor(&p, ok);
        if (!*ok) return 0;
        if (op == '*') v *= rhs;
        else {
            if (rhs == 0) { *ok = 0; return 0; }
            v /= rhs;
        }
    }
    *pp = p;
    return v;
}

static int llm_exec_parse_expr(const char **pp, int *ok)
{
    int v = llm_exec_parse_term(pp, ok);
    if (!*ok) return 0;
    const char *p = *pp;
    for (;;) {
        p = llm_exec_skip_ws(p);
        if (*p != '+' && *p != '-') break;
        char op = *p++;
        int rhs = llm_exec_parse_term(&p, ok);
        if (!*ok) return 0;
        if (op == '+') v += rhs;
        else v -= rhs;
    }
    *pp = p;
    return v;
}

static int llm_exec_builtin_arith(const int *input_tokens, int n_input_tokens,
                                  int *output_tokens, int max_output_tokens)
{
    char expr[512];
    char out[64];
    int n = llm_decode_tokens_to_text(&llm_model, input_tokens, n_input_tokens, expr, sizeof(expr));
    if (n <= 0) return -1;

    int ok = 1;
    const char *p = expr;
    int val = llm_exec_parse_expr(&p, &ok);
    p = llm_exec_skip_ws(p);
    if (!ok || *p != '\0') return -1;

    {
        int pos = 0;
        if (val == 0) {
            out[pos++] = '0';
        } else {
            int neg = val < 0;
            unsigned int u = (unsigned int)(neg ? -val : val);
            char tmp[32];
            int t = 0;
            while (u > 0 && t < (int)sizeof(tmp)) {
                tmp[t++] = (char)('0' + (u % 10));
                u /= 10;
            }
            if (neg) out[pos++] = '-';
            while (t > 0 && pos < (int)sizeof(out) - 1) out[pos++] = tmp[--t];
        }
        out[pos] = '\0';
    }

    return llm_tokenize(&llm_model, out, output_tokens, max_output_tokens);
}

static int llm_exec_builtin_arith_token_native(const int *input_tokens,
                                               int n_input_tokens,
                                               int *output_tokens,
                                               int max_output_tokens)
{
    llm_exec_tokcur_t c;
    llm_exec_rational_t v;

    if (!llm_is_loaded() || !input_tokens || n_input_tokens <= 0) return -1;

    kmemset(&c, 0, sizeof(c));
    c.m = &llm_model;
    c.tokens = input_tokens;
    c.n_tokens = n_input_tokens;

    if (llm_exec_parse_expr_tok(&c, &v) != 0) return -1;
    if (llm_exec_tokcur_has_tail(&c)) return -1;

    return llm_exec_rational_to_tokens(&v, output_tokens, max_output_tokens);
}

int llm_execute_token_program(const int *input_tokens, int n_input_tokens,
                              llm_token_exec_fn executor, void *userdata,
                              int *output_tokens, int max_output_tokens)
{
    if (!output_tokens || max_output_tokens <= 0) return -1;
    if (executor)
        return executor(input_tokens, n_input_tokens, output_tokens, max_output_tokens, userdata);

    if (!llm_is_loaded() || !input_tokens || n_input_tokens <= 0) return -1;

    /* Prefer token-native parser/executor: no full detokenize-retokenize loop. */
    {
        int rc = llm_exec_builtin_arith_token_native(input_tokens, n_input_tokens,
                                                     output_tokens, max_output_tokens);
        if (rc >= 0) return rc;
    }

    /* Legacy fallback for tokenizations that don't expose arithmetic chars cleanly. */
    return llm_exec_builtin_arith(input_tokens, n_input_tokens,
                                  output_tokens, max_output_tokens);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Token-native code execution loop
 *
 * Generates model output, feeds it to the executor, reinjects executor
 * response as a continuation prompt — all in pure token-ID space.
 * No text serialization at any machine-to-machine boundary.
 * ═══════════════════════════════════════════════════════════════════════════ */

int llm_execute_token_loop(const int *prompt_tokens, int n_prompt,
                           llm_token_exec_fn executor, void *userdata,
                           int *output_tokens, int max_output_tokens,
                           int max_gen, float temperature,
                           int max_rounds)
{
    int gen_buf_cap = max_gen > 0 ? max_gen : 256;
    int *gen_buf   = NULL;
    int *cur_prompt = NULL;
    int  cur_prompt_len = 0;
    int  total_out = 0;
    int  round;

    if (!llm_is_loaded()) return -1;
    if (!prompt_tokens || n_prompt <= 0) return -1;
    if (!output_tokens || max_output_tokens <= 0) return -1;
    if (!executor) return -1;
    if (max_rounds <= 0) max_rounds = 8;
    if (max_gen <= 0) max_gen = 256;

    gen_buf = (int *)kmalloc(gen_buf_cap * sizeof(int));
    cur_prompt = (int *)kmalloc(n_prompt * sizeof(int));
    if (!gen_buf || !cur_prompt) {
        if (gen_buf) kfree(gen_buf);
        if (cur_prompt) kfree(cur_prompt);
        return -1;
    }
    kmemcpy(cur_prompt, prompt_tokens, n_prompt * sizeof(int));
    cur_prompt_len = n_prompt;

    for (round = 0; round < max_rounds; round++) {
        int n_gen, n_exec;
        int *exec_out;
        int exec_cap;

        /* Step 1: generate from current prompt (token-native) */
        n_gen = llm_generate_tokens(cur_prompt, cur_prompt_len,
                                    gen_buf, gen_buf_cap,
                                    max_gen, temperature,
                                    round > 0 ? 1 : 0);
        if (n_gen <= 0) break;

        /* Copy generated tokens to final output if space allows */
        {
            int copy = n_gen;
            if (total_out + copy > max_output_tokens)
                copy = max_output_tokens - total_out;
            if (copy > 0) {
                kmemcpy(output_tokens + total_out, gen_buf, copy * sizeof(int));
                total_out += copy;
            }
        }

        /* Step 2: pass generated tokens to executor (token-native: no text) */
        exec_cap = gen_buf_cap;
        exec_out = (int *)kmalloc(exec_cap * sizeof(int));
        if (!exec_out) break;

        n_exec = executor(gen_buf, n_gen, exec_out, exec_cap, userdata);

        if (n_exec <= 0) {
            /* Executor returned nothing -> loop terminates cleanly */
            kfree(exec_out);
            break;
        }

        /* Copy executor output to final output if space allows */
        {
            int copy = n_exec;
            if (total_out + copy > max_output_tokens)
                copy = max_output_tokens - total_out;
            if (copy > 0) {
                kmemcpy(output_tokens + total_out, exec_out, copy * sizeof(int));
                total_out += copy;
            }
        }

        /* Step 3: reinject executor output as new prompt for the next round */
        kfree(cur_prompt);
        cur_prompt = exec_out;   /* ownership transferred */
        cur_prompt_len = n_exec;
    }

    kfree(gen_buf);
    kfree(cur_prompt);
    return total_out;
}

int llm_validate_json_tokens(const int *tokens, int n_tokens)
{
    llm_exec_tokcur_t c;

    if (!llm_is_loaded() || !tokens || n_tokens <= 0) return -1;

    kmemset(&c, 0, sizeof(c));
    c.m = &llm_model;
    c.tokens = tokens;
    c.n_tokens = n_tokens;

    if (llm_exec_json_parse_value(&c, 0) != 0)
        return 0;

    llm_exec_tokcur_skip_ws(&c);
    return llm_exec_tokcur_peek(&c) < 0 ? 1 : 0;
}

int llm_validate_code_fence_tokens(const int *tokens, int n_tokens)
{
    llm_exec_tokcur_t c;
    int ticks = 0;
    int in_body = 0;

    if (!llm_is_loaded() || !tokens || n_tokens <= 0) return -1;

    kmemset(&c, 0, sizeof(c));
    c.m = &llm_model;
    c.tokens = tokens;
    c.n_tokens = n_tokens;

    llm_exec_tokcur_skip_ws(&c);

    /* Opening fence: exactly three backticks after optional leading whitespace. */
    while (ticks < 3) {
        int ch = llm_exec_tokcur_get(&c);
        if (ch != '`') return 0;
        ticks++;
    }

    /* Optional language hint until newline. */
    for (;;) {
        int ch = llm_exec_tokcur_get(&c);
        if (ch < 0) return 0;
        if (ch == '\n' || ch == '\r') break;
    }

    in_body = 1;
    while (in_body) {
        int ch = llm_exec_tokcur_get(&c);
        if (ch < 0) return 0;
        if (ch == '`') {
            int t = 1;
            while (t < 3) {
                int nx = llm_exec_tokcur_get(&c);
                if (nx != '`') {
                    t = 0;
                    break;
                }
                t++;
            }
            if (t == 3) {
                in_body = 0;
                break;
            }
        }
    }

    llm_exec_tokcur_skip_ws(&c);
    return llm_exec_tokcur_peek(&c) < 0 ? 1 : 0;
}

static int llm_xml_name_start(int ch)
{
    return ((ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            ch == '_' || ch == ':');
}

static int llm_xml_name_char(int ch)
{
    return llm_xml_name_start(ch) ||
           (ch >= '0' && ch <= '9') ||
           ch == '-' || ch == '.';
}

int llm_validate_xml_tokens(const int *tokens, int n_tokens)
{
    llm_exec_tokcur_t c;
    enum { XML_MAX_DEPTH = 64, XML_MAX_TAG = 64 };
    char stack[XML_MAX_DEPTH][XML_MAX_TAG];
    int depth = 0;
    int saw_root = 0;

    if (!llm_is_loaded() || !tokens || n_tokens <= 0) return -1;

    kmemset(&c, 0, sizeof(c));
    c.m = &llm_model;
    c.tokens = tokens;
    c.n_tokens = n_tokens;

    while (llm_exec_tokcur_peek(&c) >= 0) {
        int ch = llm_exec_tokcur_get(&c);
        if (ch < 0) break;

        if (ch != '<') continue;

        ch = llm_exec_tokcur_get(&c);
        if (ch < 0) return 0;

        if (ch == '!') {
            int p1 = llm_exec_tokcur_get(&c);
            int p2 = llm_exec_tokcur_get(&c);
            if (p1 == '-' && p2 == '-') {
                /* Comment: consume until --> */
                int a = 0, b = 0;
                for (;;) {
                    int q = llm_exec_tokcur_get(&c);
                    if (q < 0) return 0;
                    a = b;
                    b = q;
                    if (a == '-' && b == '-' && llm_exec_tokcur_peek(&c) == '>') {
                        (void)llm_exec_tokcur_get(&c);
                        break;
                    }
                }
            } else {
                /* DOCTYPE/CDATA/etc: consume to next '>' */
                for (;;) {
                    int q = llm_exec_tokcur_get(&c);
                    if (q < 0) return 0;
                    if (q == '>') break;
                }
            }
            continue;
        }

        if (ch == '?') {
            /* Processing instruction: consume until ?> */
            int prev = 0;
            for (;;) {
                int q = llm_exec_tokcur_get(&c);
                if (q < 0) return 0;
                if (prev == '?' && q == '>') break;
                prev = q;
            }
            continue;
        }

        if (ch == '/') {
            char name[XML_MAX_TAG];
            int ni = 0;
            int top;

            ch = llm_exec_tokcur_get(&c);
            if (!llm_xml_name_start(ch)) return 0;
            while (ch >= 0 && llm_xml_name_char(ch)) {
                if (ni + 1 < XML_MAX_TAG) name[ni++] = (char)ch;
                ch = llm_exec_tokcur_get(&c);
            }
            name[ni] = '\0';

            while (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r')
                ch = llm_exec_tokcur_get(&c);
            if (ch != '>') return 0;
            if (depth <= 0) return 0;

            top = depth - 1;
            if (!llm_str_match(stack[top], (int)kstrlen(stack[top]), name, ni)) return 0;
            depth--;
            continue;
        }

        if (!llm_xml_name_start(ch)) return 0;
        {
            char name[XML_MAX_TAG];
            int ni = 0;
            int self_close = 0;
            int prev = 0;

            while (ch >= 0 && llm_xml_name_char(ch)) {
                if (ni + 1 < XML_MAX_TAG) name[ni++] = (char)ch;
                ch = llm_exec_tokcur_get(&c);
            }
            name[ni] = '\0';

            /* Consume attributes until end of tag, honoring quotes. */
            {
                int quote = 0;
                while (ch >= 0) {
                    if (!quote && (ch == '"' || ch == '\'')) {
                        quote = ch;
                    } else if (quote && ch == quote) {
                        quote = 0;
                    } else if (!quote && ch == '>') {
                        self_close = (prev == '/');
                        break;
                    }
                    prev = ch;
                    ch = llm_exec_tokcur_get(&c);
                }
                if (ch != '>') return 0;
            }

            saw_root = 1;
            if (!self_close) {
                if (depth >= XML_MAX_DEPTH) return 0;
                kmemset(stack[depth], 0, XML_MAX_TAG);
                kmemcpy(stack[depth], name, (uint64_t)ni);
                depth++;
            }
        }
    }

    return (saw_root && depth == 0) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Schema-lite key:value validator   (token-native, no text boundary)
 * ═══════════════════════════════════════════════════════════════════════════ */

int llm_validate_key_value_tokens(const int *tokens, int n_tokens)
{
    llm_exec_tokcur_t c;
    int saw_pair = 0;

    if (!llm_is_loaded() || !tokens || n_tokens <= 0) return -1;

    kmemset(&c, 0, sizeof(c));
    c.m = &llm_model;
    c.tokens = tokens;
    c.n_tokens = n_tokens;

    for (;;) {
        int ch = llm_exec_tokcur_peek(&c);
        if (ch < 0) break; /* EOF */

        /* blank line */
        if (ch == '\n' || ch == '\r') {
            (void)llm_exec_tokcur_get(&c);
            continue;
        }

        /* skip leading whitespace */
        while (ch == ' ' || ch == '\t') {
            (void)llm_exec_tokcur_get(&c);
            ch = llm_exec_tokcur_peek(&c);
        }
        if (ch < 0) break;

        /* blank line after whitespace */
        if (ch == '\n' || ch == '\r') {
            (void)llm_exec_tokcur_get(&c);
            continue;
        }

        /* comment line */
        if (ch == '#') {
            while (ch >= 0 && ch != '\n' && ch != '\r')  {
                (void)llm_exec_tokcur_get(&c);
                ch = llm_exec_tokcur_peek(&c);
            }
            if (ch >= 0) (void)llm_exec_tokcur_get(&c);
            continue;
        }

        /* key: must start with alphanumeric or _ */
        {
            int key_len = 0;
            while (ch >= 0 &&
                   ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.')) {
                key_len++;
                (void)llm_exec_tokcur_get(&c);
                ch = llm_exec_tokcur_peek(&c);
            }
            if (key_len == 0) return 0; /* invalid: no key */
        }

        /* skip whitespace before separator */
        while (ch == ' ' || ch == '\t') {
            (void)llm_exec_tokcur_get(&c);
            ch = llm_exec_tokcur_peek(&c);
        }

        /* separator: ':', '=', or '::' */
        if (ch == ':') {
            (void)llm_exec_tokcur_get(&c);
            ch = llm_exec_tokcur_peek(&c);
            if (ch == ':') { /* '::' */
                (void)llm_exec_tokcur_get(&c);
                ch = llm_exec_tokcur_peek(&c);
            }
        } else if (ch == '=') {
            (void)llm_exec_tokcur_get(&c);
            ch = llm_exec_tokcur_peek(&c);
        } else {
            return 0; /* missing separator */
        }

        /* skip whitespace after separator */
        while (ch == ' ' || ch == '\t') {
            (void)llm_exec_tokcur_get(&c);
            ch = llm_exec_tokcur_peek(&c);
        }

        /* value: at least one non-newline char */
        {
            int val_len = 0;
            while (ch >= 0 && ch != '\n' && ch != '\r') {
                val_len++;
                (void)llm_exec_tokcur_get(&c);
                ch = llm_exec_tokcur_peek(&c);
            }
            if (val_len == 0) return 0; /* empty value */
        }

        saw_pair = 1;
        if (ch >= 0) (void)llm_exec_tokcur_get(&c); /* consume newline */
    }

    return saw_pair ? 1 : 0;
}

int llm_generate_tokens(const int *prompt_tokens, int n_prompt,
                        int *output_tokens, int max_output_tokens,
                        int max_gen, float temperature, int continue_cache)
{
    if (!llm_is_loaded()) return -1;
    if (!output_tokens || max_output_tokens <= 0 || max_gen <= 0) return -1;
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return -1;

    {
        int n_gen = llm_generate_token_ids(&llm_model, prompt_tokens, n_prompt,
                                           output_tokens, max_output_tokens,
                                           max_gen, temperature, continue_cache);
        __sync_lock_release(&llm_inference_active);
        return n_gen;
    }
}

int llm_eval_sequence_logprobs(const char *text, float *out_logprobs, int max_logprobs)
{
    if (!llm_is_loaded() || !text || !out_logprobs || max_logprobs <= 0) return -1;
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return -1;

    llm_model_t *m = &llm_model;
    int result = -1;

    /* Tokenize the input text (up to max_seq - 1 tokens) */
    int max_toks = (m->max_seq < max_logprobs + 2) ? m->max_seq : max_logprobs + 2;
    int *tokens = (int *)kmalloc((size_t)max_toks * sizeof(int));
    if (!tokens) goto done;

    int n = llm_tokenize_text(text, tokens, max_toks);
    if (n < 2) { kfree(tokens); result = 0; goto done; }

    /* Reset KV cache and run the sequence in eval mode */
    m->cache_len = 0;
    llm_rep_reset();

    /* Allocate a scratch logits buffer for intermediate positions */
    float *tmp_logits = (float *)kmalloc((size_t)m->vocab_size * sizeof(float));
    if (!tmp_logits) { kfree(tokens); goto done; }

    int n_written = 0;
    for (int i = 0; i < n - 1 && n_written < max_logprobs; i++) {
        /* Forward pass: emit logits for position i */
        llm_forward_token(m, tmp_logits, tokens[i], i);

        /* log-softmax over vocab and extract logprob of the actual next token */
        float max_l = tmp_logits[0];
        for (int v = 1; v < m->vocab_size; v++)
            if (tmp_logits[v] > max_l) max_l = tmp_logits[v];

        float sum_exp = 0.0f;
        for (int v = 0; v < m->vocab_size; v++)
            sum_exp += llm_expf(tmp_logits[v] - max_l);

        float log_sum_exp = max_l + llm_log_approx(sum_exp);
        out_logprobs[n_written++] = tmp_logits[tokens[i + 1]] - log_sum_exp;
    }

    kfree(tmp_logits);
    kfree(tokens);
    result = n_written;
done:
    __sync_lock_release(&llm_inference_active);
    return result;
}

int llm_prompt_tokens(const char *user_text, int *output_tokens,
                      int max_output_tokens, int max_gen, float temperature)
{
    if (!llm_is_loaded() || !user_text || !output_tokens || max_output_tokens <= 0)
        return -1;

    llm_model_t *m = &llm_model;
    static char prompt_buf[2048];
    int pos = 0;
    const int buf_max = (int)sizeof(prompt_buf) - 1;

    #define TOKEN_PROMPT_APPEND(s) do { \
        const char *_s = (s); int _l = kstrlen(_s); \
        if (pos + _l > buf_max) _l = buf_max - pos; \
        if (_l > 0) { kmemcpy(prompt_buf + pos, _s, _l); pos += _l; } \
    } while (0)

    {
        int is_chatml = 0;
        int is_phi3 = 0;
        int is_phi2 = 0;
        int is_gemma = 0;
        int is_llama = 0;
        if (kstrlen(m->arch) >= 4 &&
            m->arch[0] == 'q' && m->arch[1] == 'w' &&
            m->arch[2] == 'e' && m->arch[3] == 'n')
            is_chatml = 1;
        if (kstrlen(m->arch) >= 3 &&
            m->arch[0] == 'p' && m->arch[1] == 'h' && m->arch[2] == 'i') {
            if (m->arch[3] == '2' || m->arch[3] == '\0')
                is_phi2 = 1;
            else
                is_phi3 = 1;
        }
        if (kstrlen(m->arch) >= 5 &&
            m->arch[0] == 'g' && m->arch[1] == 'e' &&
            m->arch[2] == 'm' && m->arch[3] == 'm' && m->arch[4] == 'a')
            is_gemma = 1;
        if (kstrlen(m->arch) >= 5 &&
            m->arch[0] == 'l' && m->arch[1] == 'l' &&
            m->arch[2] == 'a' && m->arch[3] == 'm' && m->arch[4] == 'a')
            is_llama = 1;

        if (is_phi3) {
            TOKEN_PROMPT_APPEND("<|user|>\n");
            TOKEN_PROMPT_APPEND(user_text);
            TOKEN_PROMPT_APPEND("<|end|>\n<|assistant|>\n");
        } else if (is_phi2) {
            TOKEN_PROMPT_APPEND("Instruct: ");
            TOKEN_PROMPT_APPEND(user_text);
            TOKEN_PROMPT_APPEND("\nOutput: ");
        } else if (is_chatml) {
            TOKEN_PROMPT_APPEND("<|im_start|>system\nYou are a helpful AI assistant running on TensorOS, a bare-metal AI operating system.<|im_end|>\n<|im_start|>user\n");
            TOKEN_PROMPT_APPEND(user_text);
            TOKEN_PROMPT_APPEND("<|im_end|>\n<|im_start|>assistant\n");
        } else if (is_gemma) {
            TOKEN_PROMPT_APPEND("<turn|>user\n");
            TOKEN_PROMPT_APPEND(user_text);
            TOKEN_PROMPT_APPEND("\n<turn|>model\n");
        } else if (is_llama) {
            TOKEN_PROMPT_APPEND("<|start_header_id|>user<|end_header_id|>\n\n");
            TOKEN_PROMPT_APPEND(user_text);
            TOKEN_PROMPT_APPEND("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        } else {
            TOKEN_PROMPT_APPEND("User: ");
            TOKEN_PROMPT_APPEND(user_text);
            TOKEN_PROMPT_APPEND("\nAssistant: ");
        }
    }

    prompt_buf[pos] = '\0';
    #undef TOKEN_PROMPT_APPEND

    {
        int n_tokens = llm_tokenize(m, prompt_buf, llm_tokens + 1, llm_alloc_tokens - 65);
        if (n_tokens > 0 && m->bos_id >= 0) {
            llm_tokens[0] = m->bos_id;
            n_tokens++;
        } else {
            for (int i = 0; i < n_tokens; i++)
                llm_tokens[i] = llm_tokens[i + 1];
        }
        if (n_tokens <= 0) return -1;
        if (max_gen < 1) max_gen = 256;
        return llm_generate_tokens(llm_tokens, n_tokens, output_tokens,
                                   max_output_tokens, max_gen, temperature, 0);
    }
}

int llm_chat_turn_tokens(const char *user_text, int *output_tokens,
                         int max_output_tokens, int max_gen, float temperature)
{
    if (!llm_is_loaded() || !user_text || !output_tokens || max_output_tokens <= 0)
        return -1;

    llm_model_t *m = &llm_model;
    int first_turn = (m->cache_len == 0);
    static char turn_buf[4096];
    int pos = 0;
    const int buf_max = (int)sizeof(turn_buf) - 1;

    #define TOKEN_CHAT_APPEND(s) do { \
        const char *_s = (s); int _l = kstrlen(_s); \
        if (pos + _l > buf_max) _l = buf_max - pos; \
        if (_l > 0) { kmemcpy(turn_buf + pos, _s, _l); pos += _l; } \
    } while (0)

    {
        int is_gemma = kstrlen(m->arch) >= 5 &&
            m->arch[0]=='g' && m->arch[1]=='e' && m->arch[2]=='m' &&
            m->arch[3]=='m' && m->arch[4]=='a';
        int is_phi3 = kstrlen(m->arch) >= 3 &&
            m->arch[0]=='p' && m->arch[1]=='h' && m->arch[2]=='i' &&
            m->arch[3] != '2' && m->arch[3] != '\0';
        int is_phi2 = kstrlen(m->arch) >= 3 &&
            m->arch[0]=='p' && m->arch[1]=='h' && m->arch[2]=='i' &&
            (m->arch[3] == '2' || m->arch[3] == '\0');
        int is_chatml = kstrlen(m->arch) >= 4 &&
            m->arch[0]=='q' && m->arch[1]=='w' && m->arch[2]=='e' && m->arch[3]=='n';
        int is_llama = kstrlen(m->arch) >= 5 &&
            m->arch[0]=='l' && m->arch[1]=='l' && m->arch[2]=='a' &&
            m->arch[3]=='m' && m->arch[4]=='a';

        if (is_gemma) {
            TOKEN_CHAT_APPEND("<turn|>user\n");
            TOKEN_CHAT_APPEND(user_text);
            TOKEN_CHAT_APPEND("\n<turn|>model\n");
        } else if (is_phi3) {
            TOKEN_CHAT_APPEND("<|user|>\n");
            TOKEN_CHAT_APPEND(user_text);
            TOKEN_CHAT_APPEND("<|end|>\n<|assistant|>\n");
        } else if (is_phi2) {
            TOKEN_CHAT_APPEND("Instruct: ");
            TOKEN_CHAT_APPEND(user_text);
            TOKEN_CHAT_APPEND("\nOutput: ");
        } else if (is_chatml) {
            if (first_turn)
                TOKEN_CHAT_APPEND("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n");
            TOKEN_CHAT_APPEND("<|im_start|>user\n");
            TOKEN_CHAT_APPEND(user_text);
            TOKEN_CHAT_APPEND("<|im_end|>\n<|im_start|>assistant\n");
        } else if (is_llama) {
            TOKEN_CHAT_APPEND("<|start_header_id|>user<|end_header_id|>\n\n");
            TOKEN_CHAT_APPEND(user_text);
            TOKEN_CHAT_APPEND("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        } else {
            TOKEN_CHAT_APPEND(user_text);
            TOKEN_CHAT_APPEND("\n");
        }
    }

    turn_buf[pos] = '\0';
    #undef TOKEN_CHAT_APPEND

    {
        int n_tokens = llm_tokenize(m, turn_buf, llm_tokens + 1, llm_alloc_tokens - 65);
        if (first_turn && m->bos_id >= 0) {
            llm_tokens[0] = m->bos_id;
            n_tokens++;
        } else {
            for (int i = 0; i < n_tokens; i++)
                llm_tokens[i] = llm_tokens[i + 1];
        }
        if (n_tokens <= 0) return -1;
        if (llm_think_enabled >= 2 &&
            kstrlen(m->arch) >= 5 && m->arch[0]=='g' && m->arch[1]=='e' &&
            m->arch[2]=='m' && m->arch[3]=='m' && m->arch[4]=='a' &&
            n_tokens < llm_alloc_tokens - 2)
            llm_tokens[n_tokens++] = LLM_THINK_TOKEN;
        if (max_gen < 1) max_gen = 256;
        return llm_generate_tokens(llm_tokens, n_tokens, output_tokens,
                                   max_output_tokens, max_gen, temperature,
                                   first_turn ? 0 : 1);
    }
}

int llm_kv_snapshot_prefix(const int *prefix_tokens, int n_prefix_tokens)
{
    llm_model_t *m;
    int kv_dim, layer_stride, snap_layer_stride;
    int capture_len;
    int slot_i = -1;
    uint32_t prefix_hash;

    if (!llm_is_loaded() || !prefix_tokens || n_prefix_tokens <= 0) return -1;

    m = &llm_model;
    if (m->cache_len <= 0) return -1;

    capture_len = n_prefix_tokens;
    if (capture_len > m->cache_len) capture_len = m->cache_len;
    if (capture_len <= 0) return -1;

    /* GPU mode: download KV from device to CPU before copying to snapshot.
     * In GPU forward mode, m->k_cache/m->v_cache (CPU) are never updated by
     * llm_forward_token — only gl->d_k_cache (GPU) is updated.  Without this
     * sync the snapshot contains stale zeros/garbage from the last CPU-path
     * forward, and llm_kv_restore_prefix then uploads that garbage back to the
     * GPU, producing completely wrong attention and logits. */
#ifdef ENABLE_CUDA
    if (gpu_ctx.gpu_fwd) {
        const backend_t *be = backend_get_by_id(BACKEND_CUDA);
        int kv_dim_dl = m->n_kv_heads * m->head_dim;
        int ls_dl = m->max_seq * kv_dim_dl;
        uint64_t dl_bytes = (uint64_t)capture_len * kv_dim_dl * sizeof(float);
        for (int _L = 0; _L < m->n_layers; _L++) {
            gpu_layer_t *_gl = &gpu_ctx.layers[_L];
            if (m->layers[_L].kv_reuse_layer < 0 && _gl->d_k_cache) {
                float *cpu_k = m->k_cache + (uint64_t)_L * ls_dl;
                float *cpu_v = m->v_cache + (uint64_t)_L * ls_dl;
                be->mem.download(cpu_k, _gl->d_k_cache, dl_bytes);
                be->mem.download(cpu_v, _gl->d_v_cache, dl_bytes);
            }
        }
        be->mem.sync();
    }
#endif

    prefix_hash = llm_hash_tokens(prefix_tokens, n_prefix_tokens);

    for (int i = 0; i < LLM_KV_SNAPSHOT_SLOTS; i++) {
        if (llm_kv_snapshot_prefix_match(&llm_kv_snapshots[i], prefix_tokens,
                                         n_prefix_tokens, prefix_hash)) {
            slot_i = i;
            break;
        }
    }
    if (slot_i < 0) {
        for (int i = 0; i < LLM_KV_SNAPSHOT_SLOTS; i++) {
            if (!llm_kv_snapshots[i].in_use) {
                slot_i = i;
                break;
            }
        }
    }
    if (slot_i < 0) {
        slot_i = llm_kv_snapshot_evict;
        llm_kv_snapshot_evict = (llm_kv_snapshot_evict + 1) % LLM_KV_SNAPSHOT_SLOTS;
    }

    llm_kv_snapshot_clear_slot(&llm_kv_snapshots[slot_i]);

    kv_dim = m->n_kv_heads * m->head_dim;
    layer_stride = m->max_seq * kv_dim;
    snap_layer_stride = capture_len * kv_dim;

    {
        llm_kv_snapshot_slot_t *s = &llm_kv_snapshots[slot_i];
        uint64_t token_bytes = (uint64_t)n_prefix_tokens * sizeof(int);
        uint64_t cache_bytes = (uint64_t)m->n_layers * snap_layer_stride * sizeof(float);

        s->prefix_tokens = (int *)tensor_alloc(token_bytes);
        s->k_cache = (float *)tensor_alloc(cache_bytes);
        s->v_cache = (float *)tensor_alloc(cache_bytes);
        if (!s->prefix_tokens || !s->k_cache || !s->v_cache) {
            llm_kv_snapshot_clear_slot(s);
            return -1;
        }

        kmemcpy(s->prefix_tokens, prefix_tokens, token_bytes);
        for (int L = 0; L < m->n_layers; L++) {
            const float *src_k = m->k_cache + (uint64_t)L * layer_stride;
            const float *src_v = m->v_cache + (uint64_t)L * layer_stride;
            float *dst_k = s->k_cache + (uint64_t)L * snap_layer_stride;
            float *dst_v = s->v_cache + (uint64_t)L * snap_layer_stride;
            kmemcpy(dst_k, src_k, (uint64_t)snap_layer_stride * sizeof(float));
            kmemcpy(dst_v, src_v, (uint64_t)snap_layer_stride * sizeof(float));
        }

        s->in_use = 1;
        s->prefix_hash = prefix_hash;
        s->n_prefix_tokens = n_prefix_tokens;
        s->cached_len = capture_len;
        s->n_layers = m->n_layers;
        s->max_seq = m->max_seq;
        s->n_kv_heads = m->n_kv_heads;
        s->head_dim = m->head_dim;
    }

    return capture_len;
}

int llm_kv_restore_prefix(const int *prefix_tokens, int n_prefix_tokens)
{
    llm_model_t *m;
    llm_kv_snapshot_slot_t *s = (llm_kv_snapshot_slot_t *)0;
    uint32_t prefix_hash;
    int kv_dim, layer_stride, snap_layer_stride;

    if (!llm_is_loaded() || !prefix_tokens || n_prefix_tokens <= 0) return -1;

    m = &llm_model;
    prefix_hash = llm_hash_tokens(prefix_tokens, n_prefix_tokens);

    for (int i = 0; i < LLM_KV_SNAPSHOT_SLOTS; i++) {
        if (llm_kv_snapshot_prefix_match(&llm_kv_snapshots[i], prefix_tokens,
                                         n_prefix_tokens, prefix_hash)) {
            s = &llm_kv_snapshots[i];
            break;
        }
    }

    g_kv_snap_lookups++;
    if (!s || !s->in_use) { return -1; }
    if (s->n_layers != m->n_layers || s->max_seq != m->max_seq ||
        s->n_kv_heads != m->n_kv_heads || s->head_dim != m->head_dim) {
        return -1;
    }
    g_kv_snap_hits++;

    /* Restore: overwrite only the snapshot region — no full zero needed since
     * cache_len guards access to tail positions.  Skipping the full zero saves
     * ~293MB of GPU bandwidth per token on large models. */
    m->cache_len = 0;   /* mark cache logically empty during copy */

    kv_dim = m->n_kv_heads * m->head_dim;
    layer_stride = m->max_seq * kv_dim;
    snap_layer_stride = s->cached_len * kv_dim;

    for (int L = 0; L < m->n_layers; L++) {
        float *dst_k = m->k_cache + (uint64_t)L * layer_stride;
        float *dst_v = m->v_cache + (uint64_t)L * layer_stride;
        const float *src_k = s->k_cache + (uint64_t)L * snap_layer_stride;
        const float *src_v = s->v_cache + (uint64_t)L * snap_layer_stride;
        kmemcpy(dst_k, src_k, (uint64_t)snap_layer_stride * sizeof(float));
        kmemcpy(dst_v, src_v, (uint64_t)snap_layer_stride * sizeof(float));
    }

    m->cache_len = s->cached_len;

#ifdef ENABLE_CUDA
    if (gpu_ctx.gpu_fwd) {
        const backend_t *be = backend_get_by_id(BACKEND_CUDA);
        uint64_t prefix_bytes = (uint64_t)s->cached_len * kv_dim * sizeof(float);
        for (int L = 0; L < m->n_layers; L++) {
            if (m->layers[L].kv_reuse_layer < 0 && gpu_ctx.layers[L].d_k_cache) {
                float *src_k = m->k_cache + (uint64_t)L * layer_stride;
                float *src_v = m->v_cache + (uint64_t)L * layer_stride;
                be->mem.upload(gpu_ctx.layers[L].d_k_cache, src_k, prefix_bytes);
                be->mem.upload(gpu_ctx.layers[L].d_v_cache, src_v, prefix_bytes);
            }
        }
    }
#endif

    return s->cached_len;
}

/* Restore a KV snapshot and then run the last context token through the
 * transformer to warm llm_logits and set llm_logits_pos = n_ctx - 1.
 *
 * After this call:
 *   m->cache_len  == n_ctx                 (full context cached)
 *   llm_logits_pos == n_ctx - 1            (logits valid for ctx[n_ctx-1])
 *
 * This lets llm_speculative_verify_topk / _with_correction skip their
 * forward pass entirely via the "already valid" branch:
 *   cache_len >= n_context && logits_pos == n_context - 1
 *
 * Returns n_ctx on success, -1 if no matching snapshot exists.
 */
int llm_kv_restore_and_prime(const int *ctx, int n_ctx)
{
    if (!ctx || n_ctx <= 0) return -1;
    int restored = llm_kv_restore_prefix(ctx, n_ctx);
    if (restored < 0) return -1;

    llm_model_t *m = &llm_model;
    if (n_ctx - 1 < 0 || n_ctx - 1 >= m->max_seq) return -1;
#ifdef ENABLE_CUDA
    cuda_graph_decode_ready = 1;
#endif
    llm_forward_token(m, llm_logits, ctx[n_ctx - 1], n_ctx - 1);
    llm_logits_pos = n_ctx - 1;
    return n_ctx;
}

/* Return a pointer to the current logits buffer if the logits are primed for
 * the given context length (i.e. llm_logits_pos == n_ctx - 1).  Returns NULL
 * if the logits are stale or n_ctx <= 0.
 *
 * This lets geodesic_step_fast / geodesic_rollout read the transformer's
 * actual next-token probabilities without an additional forward pass.
 */
/* Cheaply prime llm_logits for n_ctx-1 when the KV cache is exactly
 * one step behind (cache_len == n_ctx - 1).  This fires after every
 * successful speculative-verify call that appended exactly one correction
 * token, making it free — no snapshot restore needed.
 *
 * Returns 0 if logits are now primed for position n_ctx-1, -1 otherwise
 * (caller should fall back to llm_kv_restore_and_prime).
 */
int llm_prime_logits_fast(const int *ctx, int n_ctx)
{
    if (!ctx || n_ctx <= 0) return -1;
    llm_model_t *m = &llm_model;
    if (llm_logits_pos == n_ctx - 1) return 0;  /* already primed */
    if (m->cache_len != n_ctx - 1) return -1;   /* can't do single-step prime */
#ifdef ENABLE_CUDA
    cuda_graph_decode_ready = 1;
#endif
    llm_forward_token(m, llm_logits, ctx[n_ctx - 1], n_ctx - 1);
    llm_logits_pos = n_ctx - 1;
    m->cache_len   = n_ctx;   /* GPU path does not update cache_len; set it now */
    /* Cache greedy argmax so callers pay O(1) instead of O(vocab) */
    {
        int   best = 0;
        float bv   = llm_logits[0];
        for (int t = 1; t < m->vocab_size; t++)
            if (llm_logits[t] > bv) { bv = llm_logits[t]; best = t; }
        llm_primed_greedy = best;

        /* Compute Shannon entropy H = log(Z) - (Σ p_i * l_i) where Z = Σ exp(l_i - l_max).
         * Two-pass: pass 1 finds max (already done above as bv), pass 2 computes sums. */
        {
            double log_sum_exp = 0.0;
            double weighted_sum = 0.0;
            float  l_max = (float)bv;
            for (int t = 0; t < m->vocab_size; t++) {
                double e = exp((double)(llm_logits[t] - l_max));
                log_sum_exp   += e;
                weighted_sum  += e * (double)(llm_logits[t] - l_max);
            }
            if (log_sum_exp > 1e-30) {
                g_last_logit_entropy = (float)(log(log_sum_exp) - weighted_sum / log_sum_exp);
            } else {
                g_last_logit_entropy = 0.0f;
            }
        }
    }
    return 0;
}

/* Return the greedy argmax token for the last primed context position (O(1)).
 * Returns -1 if logits are not primed for n_ctx-1 or cache is stale. */
int llm_get_primed_greedy_token(int n_ctx)
{
    if (n_ctx <= 0 || llm_logits_pos != n_ctx - 1 || llm_primed_greedy < 0)
        return -1;
    return llm_primed_greedy;
}

const float *llm_get_logits_primed(int n_ctx)
{
    if (n_ctx <= 0) return NULL;
    if (llm_logits_pos != n_ctx - 1) {
        return NULL;
    }
    return llm_logits;
}

const float *llm_get_last_hidden_state(void)
{
    return llm_last_hs_valid ? llm_last_hs : NULL;
}

int llm_agent_ctx_reset(int ctx_id)
{
    if (ctx_id < 0 || ctx_id >= LLM_AGENT_CTX_SLOTS) return -1;
    kmemset(&llm_agent_ctx[ctx_id], 0, sizeof(llm_agent_ctx[ctx_id]));
    llm_agent_ctx[ctx_id].in_use = 1;
    return 0;
}

int llm_agent_ctx_append_tokens(int ctx_id, const int *tokens, int n_tokens)
{
    llm_agent_ctx_slot_t *ctx;

    if (ctx_id < 0 || ctx_id >= LLM_AGENT_CTX_SLOTS) return -1;
    if (!tokens || n_tokens <= 0) return -1;

    ctx = &llm_agent_ctx[ctx_id];
    if (!ctx->in_use) {
        kmemset(ctx, 0, sizeof(*ctx));
        ctx->in_use = 1;
    }

    if (ctx->n_tokens + n_tokens > LLM_AGENT_CTX_MAX_TOKENS) {
        int drop = (ctx->n_tokens + n_tokens) - LLM_AGENT_CTX_MAX_TOKENS;
        if (drop >= ctx->n_tokens) {
            ctx->n_tokens = 0;
        } else if (drop > 0) {
            int remain = ctx->n_tokens - drop;
            for (int i = 0; i < remain; i++)
                ctx->tokens[i] = ctx->tokens[i + drop];
            ctx->n_tokens = remain;
        }
    }

    if (ctx->n_tokens + n_tokens > LLM_AGENT_CTX_MAX_TOKENS) {
        n_tokens = LLM_AGENT_CTX_MAX_TOKENS - ctx->n_tokens;
    }
    if (n_tokens <= 0) return ctx->n_tokens;

    kmemcpy(ctx->tokens + ctx->n_tokens, tokens, (uint64_t)n_tokens * sizeof(int));
    ctx->n_tokens += n_tokens;
    return ctx->n_tokens;
}

int llm_agent_ctx_generate(int ctx_id, int *output_tokens, int max_output_tokens,
                           int max_gen, float temperature)
{
    llm_agent_ctx_slot_t *ctx;
    int n_gen;

    if (!llm_is_loaded()) return -1;
    if (ctx_id < 0 || ctx_id >= LLM_AGENT_CTX_SLOTS) return -1;
    if (!output_tokens || max_output_tokens <= 0) return -1;

    ctx = &llm_agent_ctx[ctx_id];
    if (!ctx->in_use || ctx->n_tokens <= 0) return -1;

    if (max_gen < 1) max_gen = 256;

    /* Token-native agent loop: no text rebuild / retokenization boundary. */
    n_gen = llm_generate_tokens(ctx->tokens, ctx->n_tokens,
                                output_tokens, max_output_tokens,
                                max_gen, temperature, 0);
    if (n_gen <= 0) return n_gen;

    llm_agent_ctx_append_tokens(ctx_id, output_tokens, n_gen);
    return n_gen;
}

int llm_rag_set_prefix_embeddings(const float *embeddings, int n_prefix, int dim)
{
    uint64_t bytes;

    if (!llm_is_loaded() || !embeddings || n_prefix <= 0 || dim <= 0) return -1;
    if (dim != llm_model.dim) return -1;
    if (n_prefix > LLM_RAG_PREFIX_MAX) n_prefix = LLM_RAG_PREFIX_MAX;

    llm_rag_prefix_clear_internal();

    bytes = (uint64_t)n_prefix * dim * sizeof(float);
    llm_rag_prefix.emb = (float *)tensor_alloc(bytes);
    if (!llm_rag_prefix.emb) return -1;

    kmemcpy(llm_rag_prefix.emb, embeddings, bytes);
    llm_rag_prefix.active = 1;
    llm_rag_prefix.n_prefix = n_prefix;
    llm_rag_prefix.dim = dim;
    return n_prefix;
}

void llm_rag_clear_prefix_embeddings(void)
{
    llm_rag_prefix_clear_internal();
}

int llm_set_vector_prefix(const float *vectors, int n_vectors, int dim)
{
    return llm_rag_set_prefix_embeddings(vectors, n_vectors, dim);
}

void llm_clear_vector_prefix(void)
{
    llm_rag_clear_prefix_embeddings();
}

int llm_speculative_verify_tokens(const int *context_tokens, int n_context,
                                  const int *draft_tokens, int n_draft)
{
    llm_model_t *m;
    int pos = 0;
    int accepted = 0;
    int next;

    if (!llm_is_loaded() || !context_tokens || !draft_tokens) return -1;
    if (n_context <= 0 || n_draft <= 0) return 0;
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return -1;

    m = &llm_model;
    llm_reset_cache();

    for (int i = 0; i < n_context && pos < m->max_seq - 1; i++) {
        llm_forward_token(m, llm_logits, context_tokens[i], pos);
        pos++;
    }

    if (pos <= 0) {
        __sync_lock_release(&llm_inference_active);
        return 0;
    }

    next = llm_sample(llm_logits, m->vocab_size, 0.0f);
    for (int i = 0; i < n_draft && pos < m->max_seq - 1; i++) {
        if (draft_tokens[i] != next) break;
        accepted++;
        llm_forward_token(m, llm_logits, draft_tokens[i], pos);
        pos++;
        next = llm_sample(llm_logits, m->vocab_size, 0.0f);
    }

    m->cache_len = pos;
    __sync_lock_release(&llm_inference_active);
    return accepted;
}

/**
 * Like llm_speculative_verify_tokens but also returns the transformer's
 * correction token at the first rejection point (or after the last accepted
 * draft token when all drafts are accepted).
 *
 * correction_tok_out: set to the model's greedy next-token prediction
 *   immediately after the last accepted draft position.
 *   If all n_draft tokens are accepted this is the NEXT token (bonus token).
 *   If 0 tokens are accepted this is the model's first token (full correction).
 *
 * Returns number of accepted draft tokens (0..n_draft), or -1 on error.
 */
int llm_speculative_verify_with_correction(const int *context_tokens, int n_context,
                                           const int *draft_tokens,  int n_draft,
                                           int *correction_tok_out)
{
    llm_model_t *m;
    int pos = 0;
    int accepted = 0;
    int next;

    if (!llm_is_loaded() || !context_tokens) return -1;
    if (n_draft > 0 && !draft_tokens) return -1;
    if (n_context <= 0) return 0;
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return -1;

    m = &llm_model;

    /* Invalidate stale logits cache at the start of each call.
     * Only reset when cache is short (we can't trust logits from a different
     * context position).  When cache_len == n_context (one step ahead due to
     * a primed snapshot restore), logits_pos == n_context-1 is still valid
     * and the skip-all branch below can fire. */
    if (m->cache_len < n_context - 1)
        llm_logits_pos = -1;

    /* Ultra-fast path: if KV cache already covers context[0..n_context-2] exactly,
     * just run the last context token — no snapshot restore, no GPU upload. */
    if (m->cache_len == n_context - 1 &&
        llm_logits_pos != n_context - 1) {
#ifdef ENABLE_CUDA
        /* Re-enable CUDA graph for single-token decode.  Any call to
         * llm_generate_tokens(max_gen<8) — e.g. ott_get_hidden_state —
         * sets cuda_graph_decode_ready=0.  We must re-set it here so the
         * graph is captured/replayed for each single-token verifier step.
         * cuda_graph_captured stays as-is: if already captured it will be 
         * replayed; if not yet captured it will be captured on this call. */
        cuda_graph_decode_ready = 1;
#endif
        llm_forward_token(m, llm_logits, context_tokens[n_context - 1], n_context - 1);
        llm_logits_pos = n_context - 1;
        pos = n_context;
    } else if (m->cache_len >= n_context && llm_logits_pos == n_context - 1) {
        /* Logits are already valid for n_context-1 — skip forward pass entirely */
        pos = n_context;
    }
    /* Fast path: restore from KV snapshot if available (avoids O(n) prefill) */
    else if (llm_kv_restore_prefix(context_tokens, n_context) > 0) {
        pos = m->cache_len;  /* KV cache at snapshot position */
        /* Forward any gap tokens with NULL logits (skip LM head, no graph capture) */
        for (int i = pos; i < n_context - 1; i++) {
            llm_forward_token(m, NULL, context_tokens[i], i);
            llm_logits_pos = i;
            m->cache_len = i + 1;
        }
        /* Run last context token with logits — triggers CUDA graph capture.
         * Re-enable decode_ready here so capture is attempted even if a prior
         * llm_generate_tokens(max_gen<8) call reset it to 0. */
        if (llm_logits_pos != n_context - 1) {
#ifdef ENABLE_CUDA
            cuda_graph_decode_ready = 1;
#endif
            llm_forward_token(m, llm_logits, context_tokens[n_context - 1], n_context - 1);
            llm_logits_pos = n_context - 1;
        }
        pos = n_context;  /* logits now valid for position n_context */
    } else {
        /* No snapshot: full prefill from scratch.  Pass NULL logits for all but
         * the last token (skips LM head cost + avoids stale graph capture). */
        llm_reset_cache();
        for (int i = 0; i < n_context - 1 && pos < m->max_seq - 1; i++) {
            llm_forward_token(m, NULL, context_tokens[i], pos);
            pos++;
        }
        if (pos < m->max_seq - 1) {
            llm_forward_token(m, llm_logits, context_tokens[n_context - 1], pos);
            llm_logits_pos = pos;
            pos++;
        }
    }

    if (pos <= 0) {
        __sync_lock_release(&llm_inference_active);
        return 0;
    }

    next = llm_sample(llm_logits, m->vocab_size, 0.0f);
    for (int i = 0; i < n_draft && pos < m->max_seq - 1; i++) {
        if (draft_tokens[i] != next) {
            /* Rejection: record the verifier hidden state at this position
             * as the residual for online basis update (Feature 1).
             * We use a single representative layer (floor of n_layers/2)
             * as a lightweight proxy rather than all layers. */
            if (g_onb_ctx && llm_last_hs_valid && llm_last_hs) {
                int rep_layer = m->n_layers / 2;
                onb_record_residual(g_onb_ctx, rep_layer, llm_last_hs);
            }
            break;
        }
        accepted++;
        llm_forward_token(m, llm_logits, draft_tokens[i], pos);
        llm_logits_pos = pos;
        pos++;
        next = llm_sample(llm_logits, m->vocab_size, 0.0f);
    }

    if (correction_tok_out) *correction_tok_out = next;
    /* Flush pending Oja updates accumulated during this verify call */
    if (g_onb_ctx) onb_apply_pending(g_onb_ctx);
    m->cache_len = pos;
    __sync_lock_release(&llm_inference_active);
    return accepted;
}

/* ─────────────────────────────────────────────────────────────────────────
 * llm_speculative_verify_topk
 *
 * Relaxed speculative decoding acceptance: accept draft token d if
 *   logits[d] >= max(logits) - topk_margin
 * i.e., the draft token's probability is at least exp(-topk_margin) of the
 * top-1 probability.  This gives higher acceptance rate for imperfect draft
 * models at the cost of occasionally accepting a suboptimal token.
 *
 * When a draft is accepted this way, the forward pass still runs AT that
 * position (so KV cache advances correctly), but we emit the draft token
 * rather than the greedy argmax.
 *
 * topk_margin calibration:
 *   1.0  ≈ top-36% (aggressive, quality degrades slightly)
 *   2.0  ≈ top-13% (balanced)
 *   3.0  ≈ top-5%  (conservative, near-greedy quality)
 * ───────────────────────────────────────────────────────────────────────── */
int llm_speculative_verify_topk(const int *context_tokens, int n_context,
                                const int *draft_tokens,   int n_draft,
                                float topk_margin,
                                int *correction_tok_out)
{
    llm_model_t *m;
    int pos = 0;
    int accepted = 0;

    if (!llm_is_loaded() || !context_tokens) return -1;
    if (n_draft > 0 && !draft_tokens) return -1;
    if (n_context <= 0) return 0;
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return -1;

    m = &llm_model;

    /* Re-use the same fast-path KV cache logic from verify_with_correction.
     * Only invalidate logits when the cache is short (missing context tokens).
     * If cache_len == n_context (primed by llm_kv_restore_and_prime), preserve
     * logits_pos so the skip-all branch fires without any forward pass. */
    if (m->cache_len < n_context - 1)
        llm_logits_pos = -1;

    if (m->cache_len == n_context - 1 &&
        llm_logits_pos != n_context - 1) {
#ifdef ENABLE_CUDA
        cuda_graph_decode_ready = 1;
#endif
        llm_forward_token(m, llm_logits, context_tokens[n_context - 1], n_context - 1);
        llm_logits_pos = n_context - 1;
        pos = n_context;
    } else if (m->cache_len >= n_context && llm_logits_pos == n_context - 1) {
        pos = n_context;
    } else if (llm_kv_restore_prefix(context_tokens, n_context) > 0) {
        pos = m->cache_len;
        for (int i = pos; i < n_context - 1; i++) {
            llm_forward_token(m, NULL, context_tokens[i], i);
            llm_logits_pos = i;
            m->cache_len = i + 1;
        }
        if (llm_logits_pos != n_context - 1) {
#ifdef ENABLE_CUDA
            cuda_graph_decode_ready = 1;
#endif
            llm_forward_token(m, llm_logits, context_tokens[n_context - 1], n_context - 1);
            llm_logits_pos = n_context - 1;
        }
        pos = n_context;
    } else {
        llm_reset_cache();
        for (int i = 0; i < n_context - 1 && pos < m->max_seq - 1; i++) {
            llm_forward_token(m, NULL, context_tokens[i], pos);
            pos++;
        }
        if (pos < m->max_seq - 1) {
            llm_forward_token(m, llm_logits, context_tokens[n_context - 1], pos);
            llm_logits_pos = pos;
            pos++;
        }
    }

    if (pos <= 0) {
        __sync_lock_release(&llm_inference_active);
        return 0;
    }

    /* greedy top-1 is always the correction token at first rejection */
    int greedy_next = llm_sample(llm_logits, m->vocab_size, 0.0f);

    for (int i = 0; i < n_draft && pos < m->max_seq - 1; i++) {
        int d = draft_tokens[i];
        if (d < 0 || d >= m->vocab_size) break;

        /* Check logit margin: accept if logit[d] >= max_logit - topk_margin */
        float max_logit = llm_logits[greedy_next];
        float draft_logit = llm_logits[d];
        int accept = (draft_logit >= max_logit - topk_margin);

        if (!accept) break; /* first rejection terminates the chain */

        accepted++;
        /* Advance KV cache through the (accepted) draft token position */
        llm_forward_token(m, llm_logits, d, pos);
        llm_logits_pos = pos;
        pos++;
        greedy_next = llm_sample(llm_logits, m->vocab_size, 0.0f);
    }

    if (correction_tok_out) *correction_tok_out = greedy_next;
    m->cache_len = pos;
    __sync_lock_release(&llm_inference_active);
    return accepted;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * llm_speculative_verify_swarm
 *
 * OD-SWARM speculative verify: combines the topk_margin relaxed acceptance
 * from llm_speculative_verify_topk with a SWARM secondary acceptance path.
 *
 * For each draft slot i:
 *   1. Primary: accept draft_tokens[i] if logit[draft_i] >= max_logit - margin
 *      (emit draft_tokens[i], KV-advance with draft_tokens[i])
 *   2. SWARM:   accept if verifier greedy ∈ swarm_tokens[i*swarm_k .. +swarm_k]
 *      (emit greedy, KV-advance with greedy — preserves correctness exactly)
 *
 * With K=16 OD candidates at 2.5% per-candidate accuracy:
 *   P(primary OR swarm hit) = 1-(1-0.025)^16 ≈ 33% per draft slot.
 * swarm_k=0 degrades to llm_speculative_verify_topk behaviour.
 * ───────────────────────────────────────────────────────────────────────── */
int llm_speculative_verify_swarm(const int *context_tokens, int n_context,
                                 const int *draft_tokens,   int n_draft,
                                 float      topk_margin,
                                 const int *swarm_tokens,   int swarm_k,
                                 int *correction_tok_out)
{
    llm_model_t *m;
    int pos      = 0;
    int accepted = 0;

    if (!llm_is_loaded() || !context_tokens) return -1;
    if (n_draft > 0 && !draft_tokens) return -1;
    if (n_context <= 0) return 0;
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return -1;

    m = &llm_model;

    if (m->cache_len < n_context - 1)
        llm_logits_pos = -1;

    if (m->cache_len == n_context - 1 &&
        llm_logits_pos != n_context - 1) {
#ifdef ENABLE_CUDA
        cuda_graph_decode_ready = 1;
#endif
        llm_forward_token(m, llm_logits, context_tokens[n_context - 1], n_context - 1);
        llm_logits_pos = n_context - 1;
        pos = n_context;
    } else if (m->cache_len >= n_context && llm_logits_pos == n_context - 1) {
        pos = n_context;
    } else if (llm_kv_restore_prefix(context_tokens, n_context) > 0) {
        pos = m->cache_len;
        for (int i = pos; i < n_context - 1; i++) {
            llm_forward_token(m, NULL, context_tokens[i], i);
            llm_logits_pos = i;
            m->cache_len = i + 1;
        }
        if (llm_logits_pos != n_context - 1) {
#ifdef ENABLE_CUDA
            cuda_graph_decode_ready = 1;
#endif
            llm_forward_token(m, llm_logits, context_tokens[n_context - 1], n_context - 1);
            llm_logits_pos = n_context - 1;
        }
        pos = n_context;
    } else {
        llm_reset_cache();
        for (int i = 0; i < n_context - 1 && pos < m->max_seq - 1; i++) {
            llm_forward_token(m, NULL, context_tokens[i], pos);
            pos++;
        }
        if (pos < m->max_seq - 1) {
            llm_forward_token(m, llm_logits, context_tokens[n_context - 1], pos);
            llm_logits_pos = pos;
            pos++;
        }
    }

    if (pos <= 0) {
        __sync_lock_release(&llm_inference_active);
        return 0;
    }

    int greedy_next = llm_sample(llm_logits, m->vocab_size, 0.0f);

    for (int i = 0; i < n_draft && pos < m->max_seq - 1; i++) {
        int d = draft_tokens[i];
        if (d < 0 || d >= m->vocab_size) break;

        int   accept_tok = -1;
        float max_logit  = llm_logits[greedy_next];

        /* Primary: topk margin acceptance (emit draft token) */
        if (llm_logits[d] >= max_logit - topk_margin) {
            accept_tok = d;
        }
        /* SWARM: greedy token is one of our K candidates (emit greedy — exact) */
        else if (swarm_tokens && swarm_k > 0) {
            for (int s = 0; s < swarm_k; s++) {
                if (swarm_tokens[i * swarm_k + s] == greedy_next) {
                    accept_tok = greedy_next;
                    break;
                }
            }
        }

        if (accept_tok < 0) break;  /* first rejection terminates chain */

        accepted++;
        llm_forward_token(m, llm_logits, accept_tok, pos);
        llm_logits_pos = pos;
        pos++;
        greedy_next = llm_sample(llm_logits, m->vocab_size, 0.0f);
    }

    if (correction_tok_out) *correction_tok_out = greedy_next;
    m->cache_len = pos;
    __sync_lock_release(&llm_inference_active);
    return accepted;
}

int llm_rollout_exact_greedy(const int *context_tokens, int n_context,
                             int n_steps,
                             int *out_tokens,
                             float *out_margin,
                             int *n_out)
{
    llm_model_t *m;
    int pos = 0;
    int produced = 0;

    if (n_out) *n_out = 0;
    if (!llm_is_loaded() || !context_tokens || !out_tokens) return -1;
    if (n_context <= 0 || n_steps <= 0) return 0;
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return -1;

    m = &llm_model;

    if (m->cache_len < n_context - 1)
        llm_logits_pos = -1;

    if (m->cache_len == n_context - 1 &&
        llm_logits_pos != n_context - 1) {
#ifdef ENABLE_CUDA
        cuda_graph_decode_ready = 1;
#endif
        llm_forward_token(m, llm_logits, context_tokens[n_context - 1], n_context - 1);
        llm_logits_pos = n_context - 1;
        pos = n_context;
    } else if (m->cache_len >= n_context && llm_logits_pos == n_context - 1) {
        pos = n_context;
    } else if (llm_kv_restore_prefix(context_tokens, n_context) > 0) {
        pos = m->cache_len;
        for (int i = pos; i < n_context - 1; i++) {
            llm_forward_token(m, NULL, context_tokens[i], i);
            llm_logits_pos = i;
            m->cache_len = i + 1;
        }
        if (llm_logits_pos != n_context - 1) {
#ifdef ENABLE_CUDA
            cuda_graph_decode_ready = 1;
#endif
            llm_forward_token(m, llm_logits, context_tokens[n_context - 1], n_context - 1);
            llm_logits_pos = n_context - 1;
        }
        pos = n_context;
    } else {
        llm_reset_cache();
        for (int i = 0; i < n_context - 1 && pos < m->max_seq - 1; i++) {
            llm_forward_token(m, NULL, context_tokens[i], pos);
            pos++;
        }
        if (pos < m->max_seq - 1) {
            llm_forward_token(m, llm_logits, context_tokens[n_context - 1], pos);
            llm_logits_pos = pos;
            pos++;
        }
    }

    if (pos <= 0) {
        __sync_lock_release(&llm_inference_active);
        return 0;
    }

    while (produced < n_steps && pos < m->max_seq - 1) {
        int best_tok = 0;
        float best_logit = llm_logits[0];
        float second_logit = -1e30f;

        for (int t = 1; t < m->vocab_size; t++) {
            float logit = llm_logits[t];
            if (logit > best_logit) {
                second_logit = best_logit;
                best_logit = logit;
                best_tok = t;
            } else if (logit > second_logit) {
                second_logit = logit;
            }
        }

        out_tokens[produced] = best_tok;
        if (out_margin) {
            out_margin[produced] = (second_logit > -1e30f)
                ? (best_logit - second_logit)
                : 8.0f;
        }

        llm_forward_token(m, llm_logits, best_tok, pos);
        llm_logits_pos = pos;
        pos++;
        produced++;
    }

    if (n_out) *n_out = produced;
    m->cache_len = pos;
    __sync_lock_release(&llm_inference_active);
    return produced;
}

int llm_rollout_reset(void)
{
    kmemset(&llm_rollout_trace, 0, sizeof(llm_rollout_trace));
    return 0;
}

int llm_rollout_append_step(int token_id, float logprob, float value)
{
    llm_rollout_step_t *step;

    if (llm_rollout_trace.n_steps >= LLM_ROLLOUT_MAX_STEPS) return -1;
    step = &llm_rollout_trace.steps[llm_rollout_trace.n_steps++];
    step->token_id = token_id;
    step->logprob = logprob;
    step->value = value;
    step->reward = 0.0f;
    step->done = 0;
    return llm_rollout_trace.n_steps;
}

int llm_rollout_set_step_reward(int step_idx, float reward, int done)
{
    if (step_idx < 0 || step_idx >= llm_rollout_trace.n_steps) return -1;
    llm_rollout_trace.steps[step_idx].reward = reward;
    llm_rollout_trace.steps[step_idx].done = done ? 1 : 0;
    return 0;
}

int llm_rollout_get_steps(llm_rollout_step_t *out_steps, int max_steps)
{
    int n_copy;
    if (!out_steps || max_steps <= 0) return -1;
    n_copy = llm_rollout_trace.n_steps;
    if (n_copy > max_steps) n_copy = max_steps;
    if (n_copy > 0) {
        kmemcpy(out_steps, llm_rollout_trace.steps,
                (uint64_t)n_copy * sizeof(llm_rollout_step_t));
    }
    return n_copy;
}

int llm_rollout_compute_returns(float gamma, float *out_returns, int max_returns)
{
    float ret = 0.0f;
    int n;

    if (!out_returns || max_returns <= 0) return -1;
    if (gamma < 0.0f) gamma = 0.0f;
    if (gamma > 1.0f) gamma = 1.0f;

    n = llm_rollout_trace.n_steps;
    if (n > max_returns) n = max_returns;
    if (n <= 0) return 0;

    for (int i = n - 1; i >= 0; i--) {
        if (llm_rollout_trace.steps[i].done) ret = 0.0f;
        ret = llm_rollout_trace.steps[i].reward + gamma * ret;
        out_returns[i] = ret;
    }
    return n;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Tensor Bridge API (exposes bridge for daisy-chaining LLMs)                */
/* ─────────────────────────────────────────────────────────────────────────── */

tensor_bridge_t *llm_get_bridge(void) {
    return &llm_bridge;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * llm_generate_geodesic_speculative
 *
 * HyperTensor turbo decode path: uses axiom_beta_geodesic_rollout() to draft
 * n_draft tokens in one trajectory-coherent geodesic integration, then
 * verifies them all with a single llm_speculative_verify_with_correction()
 * call.  On rejection, axiom_beta_grc_feedback() reinforces the GRC library
 * so accuracy improves over time.
 *
 * Expected throughput: (n_accepted_avg + 1) tokens per verifier call.
 * With a warm GRC library (MRR > 0.3) this reaches 2.5–4× autoregressive.
 *
 * Falls back gracefully to ordinary autoregressive generation if geodesic
 * geometry is not ready (Phase-3 not built yet, cold start).
 * ───────────────────────────────────────────────────────────────────────── */
int llm_generate_geodesic_speculative(const int *prompt_tokens, int n_prompt,
                                      int *output_tokens, int max_output_tokens,
                                      int max_gen, float temperature,
                                      int n_draft)
{
#define GEO_SPEC_MAX_DRAFT  8
#define GEO_SPEC_CTX_MAX    4096

    int      draft_tokens[GEO_SPEC_MAX_DRAFT];
    float    draft_conf  [GEO_SPEC_MAX_DRAFT];
    int      ctx_buf     [GEO_SPEC_CTX_MAX];
    int      n_ctx, n_out, n_drafted, correction;
    int      total_accepted, total_calls;

    if (!llm_is_loaded()) return -1;
    if (!prompt_tokens || n_prompt <= 0) return -1;
    if (!output_tokens || max_output_tokens <= 0 || max_gen <= 0) return -1;

    if (n_draft < 1) n_draft = 4;
    if (n_draft > GEO_SPEC_MAX_DRAFT) n_draft = GEO_SPEC_MAX_DRAFT;

    /* Clamp context buffer size */
    n_ctx = n_prompt;
    if (n_ctx > GEO_SPEC_CTX_MAX) n_ctx = GEO_SPEC_CTX_MAX;
    kmemcpy(ctx_buf, prompt_tokens + (n_prompt - n_ctx), (uint64_t)n_ctx * sizeof(int));

    /* Sync axiom context so geodesic step sees the right hidden states */
    ott_update_generation_context(ctx_buf, n_ctx);

    n_out          = 0;
    total_accepted = 0;
    total_calls    = 0;

    while (n_out < max_gen && n_out < max_output_tokens) {
        int accepted, i;

        /* ── Draft phase: geodesic multi-step rollout ──────────────────── */
        n_drafted = 0;
        if (axiom_beta_geodesic_rollout(ctx_buf, n_ctx, n_draft,
                                        draft_tokens, draft_conf,
                                        &n_drafted) != AXIOM_BETA_OK ||
            n_drafted <= 0) {
            /* Geodesic geometry not ready — fall back to single autoregressive step */
            int tok = -1;
            if (__sync_lock_test_and_set(&llm_inference_active, 1) == 0) {
                llm_model_t *m = &llm_model;
                if (m->cache_len < n_ctx - 1) {
                    llm_reset_cache();
                    for (int j = 0; j < n_ctx - 1 && j < m->max_seq - 1; j++)
                        llm_forward_token(m, NULL, ctx_buf[j], j);
                }
                if (n_ctx - 1 < m->max_seq) {
#ifdef ENABLE_CUDA
                    cuda_graph_decode_ready = 1;
#endif
                    llm_forward_token(m, llm_logits, ctx_buf[n_ctx - 1], n_ctx - 1);
                    llm_logits_pos = n_ctx - 1;
                }
                tok = llm_sample(llm_logits, m->vocab_size, temperature);
                __sync_lock_release(&llm_inference_active);
            }
            if (tok < 0) break;
            {
                llm_model_t *m = &llm_model;
                if (tok == m->eos_id) break;
            }
            output_tokens[n_out++] = tok;
            if (n_ctx < GEO_SPEC_CTX_MAX)
                ctx_buf[n_ctx++] = tok;
            else {
                kmemmove(ctx_buf, ctx_buf + 1,
                         (uint64_t)(GEO_SPEC_CTX_MAX - 1) * sizeof(int));
                ctx_buf[GEO_SPEC_CTX_MAX - 1] = tok;
            }
            ott_update_generation_context(ctx_buf, n_ctx);
            continue;
        }

        /* ── Apply confidence-based threshold — drop low-confidence suffix ── */
        {
            int keep = 0;
            for (int j = 0; j < n_drafted; j++) {
                float thresh = axiom_beta_rollout_threshold(j, 0.25f);
                if (draft_conf[j] >= thresh)
                    keep = j + 1;
            }
            if (keep < n_drafted) n_drafted = keep;
            if (n_drafted <= 0)   n_drafted = 1;
        }

        /* ── Verify phase ──────────────────────────────────────────────── */
        correction = -1;
        accepted   = llm_speculative_verify_with_correction(
                         ctx_buf, n_ctx,
                         draft_tokens, n_drafted,
                         &correction);
        total_calls++;

        if (accepted < 0) break;  /* lock contention or error */

        /* ── GRC feedback on first rejected draft slot ──────────────────── */
        if (accepted < n_drafted && correction >= 0) {
            axiom_beta_grc_feedback(ctx_buf, n_ctx,
                                    accepted > 0 ? draft_tokens[accepted - 1]
                                                 : correction);
        }

        /* ── Emit accepted tokens ──────────────────────────────────────── */
        {
            llm_model_t *m = &llm_model;
            for (i = 0; i < accepted && n_out < max_gen && n_out < max_output_tokens; i++) {
                int tok = draft_tokens[i];
                if (tok == m->eos_id) goto geo_done;
                output_tokens[n_out++] = tok;
                if (n_ctx < GEO_SPEC_CTX_MAX)
                    ctx_buf[n_ctx++] = tok;
                else {
                    kmemmove(ctx_buf, ctx_buf + 1,
                             (uint64_t)(GEO_SPEC_CTX_MAX - 1) * sizeof(int));
                    ctx_buf[GEO_SPEC_CTX_MAX - 1] = tok;
                }
            }
            total_accepted += accepted;

            /* ── Emit verifier correction token ────────────────────────── */
            if (correction >= 0 && n_out < max_gen && n_out < max_output_tokens) {
                if (correction == m->eos_id) goto geo_done;
                output_tokens[n_out++] = correction;
                if (n_ctx < GEO_SPEC_CTX_MAX)
                    ctx_buf[n_ctx++] = correction;
                else {
                    kmemmove(ctx_buf, ctx_buf + 1,
                             (uint64_t)(GEO_SPEC_CTX_MAX - 1) * sizeof(int));
                    ctx_buf[GEO_SPEC_CTX_MAX - 1] = correction;
                }
            }
        }

        ott_update_generation_context(ctx_buf, n_ctx);

        (void)total_accepted; (void)total_calls;
    }

geo_done:
    return n_out;

#undef GEO_SPEC_MAX_DRAFT
#undef GEO_SPEC_CTX_MAX
}

/* ─────────────────────────────────────────────────────────────────────────────
 * llm_session_*  — per-session inference contexts for parallel serving.
 *
 * Each session is an independent inference slot backed by a KV snapshot.
 * The server can hold up to LLM_SESSION_MAX concurrent sessions and step
 * them round-robin via llm_session_step(), which context-switches the model's
 * KV cache in/out of the session's snapshot slot.
 *
 * Lifecycle:
 *   llm_session_create()  — tokenise prompt, prefill, snapshot KV state
 *   llm_session_step()    — restore KV, forward one token, save KV, return tok
 *   llm_session_destroy() — free snapshot memory, mark slot free
 *
 * Thread safety: llm_session_step() uses llm_inference_active to prevent
 * simultaneous concurrent use of the transformer.  The caller (api_server
 * parallel worker) must sequence calls from one thread at a time.
 * ───────────────────────────────────────────────────────────────────────── */

#define LLM_SESSION_MAX 8
#define LLM_SESSION_CTX_MAX 4096

typedef struct {
    int   in_use;
    int   ctx[LLM_SESSION_CTX_MAX];
    int   n_ctx;
    int   n_generated;
    int   max_gen;
    float temperature;
    int   done;              /* 1 when EOS emitted or max_gen reached */
    int   snapshotted;       /* 1 after first KV snapshot saved */
} llm_session_t;

static llm_session_t llm_sessions[LLM_SESSION_MAX];

int llm_session_create(const int *prompt_tokens, int n_prompt,
                       int max_gen, float temperature)
{
    int sid = -1;
    for (int i = 0; i < LLM_SESSION_MAX; i++) {
        if (!llm_sessions[i].in_use) { sid = i; break; }
    }
    if (sid < 0) return -1;  /* all slots busy */

    llm_session_t *s = &llm_sessions[sid];
    kmemset(s, 0, sizeof(*s));
    s->in_use      = 1;
    s->max_gen     = max_gen > 0 ? max_gen : 256;
    s->temperature = temperature;

    /* Clamp prompt to context window */
    int n = n_prompt;
    if (n > LLM_SESSION_CTX_MAX) n = LLM_SESSION_CTX_MAX;
    kmemcpy(s->ctx, prompt_tokens + (n_prompt - n), (uint64_t)n * sizeof(int));
    s->n_ctx = n;

    /* Prefill: run the prompt through the transformer and snapshot KV state.
     * Use llm_kv_restore_and_prime to fill cache with logits at prompt[-1]. */
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return -1;
    {
        llm_model_t *m = &llm_model;
        llm_reset_cache();
        uint64_t t_prefill0 = hal_timer_us();
#ifdef ENABLE_CUDA
        /* Use batched GPU prefill when available — same path as llm_generate */
        int did_batch = 0;
        if (gpu_ctx.gpu_fwd && s->n_ctx >= 3 &&
            !llm_attnres_enabled && !llm_depth_attn_enabled &&
            axex_manifold_compressed_count() == 0) {  /* skip when GP active (raw weights freed) */
            if (gpu_ctx.prefill_max_n < s->n_ctx - 1)
                llm_ensure_prefill_scratch(m, s->n_ctx - 1);
            if (gpu_ctx.prefill_max_n >= s->n_ctx - 1) {
                llm_forward_prefill_batch_gpu(m, s->ctx, s->n_ctx - 1, 0);
                llm_forward_token(m, llm_logits, s->ctx[s->n_ctx - 1], s->n_ctx - 1);
                llm_logits_pos = s->n_ctx - 1;
                did_batch = 1;
            }
        }
        if (!did_batch)
#endif
        {
            for (int i = 0; i < s->n_ctx - 1 && i < m->max_seq - 1; i++)
                llm_forward_token(m, NULL, s->ctx[i], i);
            if (s->n_ctx > 0 && s->n_ctx - 1 < m->max_seq) {
#ifdef ENABLE_CUDA
                cuda_graph_decode_ready = 1;
#endif
                llm_forward_token(m, llm_logits, s->ctx[s->n_ctx - 1], s->n_ctx - 1);
                llm_logits_pos = s->n_ctx - 1;
            }
        }
        llm_last_prefill_ms_val = (float)((hal_timer_us() - t_prefill0) / 1000);
    }
    __sync_lock_release(&llm_inference_active);

    /* Save KV state to snapshot keyed by session context */
    llm_kv_snapshot_prefix(s->ctx, s->n_ctx);
    s->snapshotted = 1;

    return sid;
}

/* Step one token for the session.
 * Returns: token id (≥0) on success, LLM_SESSION_DONE (-10) when finished,
 *          LLM_SESSION_ERR (-11) on error. */
#define LLM_SESSION_DONE  (-10)
#define LLM_SESSION_ERR   (-11)

int llm_session_step(int sid)
{
    if (sid < 0 || sid >= LLM_SESSION_MAX) return LLM_SESSION_ERR;
    llm_session_t *s = &llm_sessions[sid];
    if (!s->in_use || s->done) return LLM_SESSION_DONE;
    if (s->n_generated >= s->max_gen) { s->done = 1; return LLM_SESSION_DONE; }

    /* Restore KV state for this session */
    if (s->snapshotted) {
        if (llm_kv_restore_prefix(s->ctx, s->n_ctx) < 0) {
            /* snapshot evicted — re-prefill from scratch */
            if (__sync_lock_test_and_set(&llm_inference_active, 1)) return LLM_SESSION_ERR;
            {
                llm_model_t *m = &llm_model;
                llm_reset_cache();
#ifdef ENABLE_CUDA
                int did_repf = 0;
                if (gpu_ctx.gpu_fwd && s->n_ctx >= 3 &&
                    !llm_attnres_enabled && !llm_depth_attn_enabled &&
                    axex_manifold_compressed_count() == 0) {  /* skip when GP active */
                    if (gpu_ctx.prefill_max_n < s->n_ctx - 1)
                        llm_ensure_prefill_scratch(m, s->n_ctx - 1);
                    if (gpu_ctx.prefill_max_n >= s->n_ctx - 1) {
                        llm_forward_prefill_batch_gpu(m, s->ctx, s->n_ctx - 1, 0);
                        llm_forward_token(m, llm_logits, s->ctx[s->n_ctx - 1], s->n_ctx - 1);
                        llm_logits_pos = s->n_ctx - 1;
                        did_repf = 1;
                    }
                }
                if (!did_repf)
#endif
                {
                    for (int i = 0; i < s->n_ctx - 1 && i < m->max_seq - 1; i++)
                        llm_forward_token(m, NULL, s->ctx[i], i);
                    if (s->n_ctx > 0) {
#ifdef ENABLE_CUDA
                        cuda_graph_decode_ready = 1;
#endif
                        llm_forward_token(m, llm_logits, s->ctx[s->n_ctx - 1], s->n_ctx - 1);
                        llm_logits_pos = s->n_ctx - 1;
                    }
                }
            }
            __sync_lock_release(&llm_inference_active);
        }
    }

    /* Single-token forward to get logits at the new position */
    if (__sync_lock_test_and_set(&llm_inference_active, 1)) return LLM_SESSION_ERR;
    int tok;
    {
        llm_model_t *m = &llm_model;
        /* Fast path: logits already primed (from restore) */
        if (llm_logits_pos != s->n_ctx - 1 && s->n_ctx <= m->max_seq) {
#ifdef ENABLE_CUDA
            cuda_graph_decode_ready = 1;
#endif
            llm_forward_token(m, llm_logits, s->ctx[s->n_ctx - 1], s->n_ctx - 1);
            llm_logits_pos = s->n_ctx - 1;
        }
        tok = llm_sample(llm_logits, m->vocab_size, s->temperature);
        /* Append token to session context */
        if (tok != m->eos_id) {
            if (s->n_ctx < LLM_SESSION_CTX_MAX)
                s->ctx[s->n_ctx++] = tok;
            else {
                /* slide window */
                kmemmove(s->ctx, s->ctx + 1,
                         (uint64_t)(LLM_SESSION_CTX_MAX - 1) * sizeof(int));
                s->ctx[LLM_SESSION_CTX_MAX - 1] = tok;
            }
        }
        s->n_generated++;
    }
    __sync_lock_release(&llm_inference_active);

    /* Save updated KV state back to snapshot */
    llm_kv_snapshot_prefix(s->ctx, s->n_ctx);

    /* Check termination */
    {
        llm_model_t *m = &llm_model;
        if (tok == m->eos_id || s->n_generated >= s->max_gen) {
            s->done = 1;
            return LLM_SESSION_DONE;
        }
    }
    return tok;
}

void llm_session_destroy(int sid)
{
    if (sid < 0 || sid >= LLM_SESSION_MAX) return;
    kmemset(&llm_sessions[sid], 0, sizeof(llm_sessions[sid]));
}

int llm_session_is_done(int sid)
{
    if (sid < 0 || sid >= LLM_SESSION_MAX) return 1;
    return !llm_sessions[sid].in_use || llm_sessions[sid].done;
}

int llm_session_get_output(int sid, int *token_buf, int max_tokens)
{
    if (sid < 0 || sid >= LLM_SESSION_MAX) return -1;
    llm_session_t *s = &llm_sessions[sid];
    if (!s->in_use) return -1;
    /* output = generated portion: ctx[n_ctx - n_generated .. n_ctx-1] */
    int start = s->n_ctx - s->n_generated;
    if (start < 0) start = 0;
    int n = s->n_ctx - start;
    if (n > max_tokens) n = max_tokens;
    if (n > 0) kmemcpy(token_buf, s->ctx + start, (uint64_t)n * sizeof(int));
    return n;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/*  Test-facing wrapper API (exposes internal tokenizer for unit tests)        */
/* ─────────────────────────────────────────────────────────────────────────── */

int llm_test_tokenize(const char *text, int text_len, int *tokens, int max_tokens)
{
    (void)text_len;
    if (!llm_is_loaded()) return -1;
    return llm_tokenize(&llm_model, text, tokens, max_tokens);
}

int llm_test_decode_token(int token_id, char *buf, int max_len)
{
    if (!llm_is_loaded()) return -1;
    return llm_decode_token(&llm_model, token_id, buf, max_len);
}

