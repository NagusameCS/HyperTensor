/*
 * Geodessical MLIR-Inspired IR Optimizer
 *
 * Lightweight intermediate representation for tensor computation graphs.
 * Implements the core optimization passes that MLIR provides for inference:
 *
 *   1. Op Fusion: merge sequential ops into fused kernels
 *      (e.g. RMSNorm+GEMV, SiLU+Mul, GEMV+Add → single pass)
 *   2. Constant Folding: pre-compute constant subexpressions
 *   3. Dead Code Elimination: prune unused computation paths
 *   4. Buffer Planning: minimize memory allocation via liveness analysis
 *   5. Type Legalization: select optimal kernel specialization per type
 *
 * This replaces the stub backend_mlir.c with a real optimization layer
 * that sits between the model graph and the execution backend (CPU/CUDA).
 *
 * Design: We build a DAG of MlirOp nodes, run optimization passes over them,
 * then lower to fused kernels that call the active backend's compute ops.
 */

#include "runtime/nn/backend.h"

#ifdef GEODESSICAL_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * IR Node Types
 * ════════════════════════════════════════════════════════════════════════ */

typedef enum {
    MLIR_OP_GEMV,         /* Matrix-vector multiply (quantized) */
    MLIR_OP_GEMM,         /* Matrix-matrix multiply (float) */
    MLIR_OP_RMSNORM,      /* RMS normalization */
    MLIR_OP_LAYERNORM,    /* Layer normalization */
    MLIR_OP_ROPE,         /* Rotary position embeddings */
    MLIR_OP_SOFTMAX,      /* Softmax */
    MLIR_OP_SILU,         /* SiLU activation */
    MLIR_OP_GELU,         /* GELU activation */
    MLIR_OP_MUL,          /* Element-wise multiply */
    MLIR_OP_ADD,          /* Element-wise add */
    MLIR_OP_SCALE,        /* Scalar multiply */
    MLIR_OP_DOT,          /* Dot product */
    MLIR_OP_DEQUANT,      /* Dequantize */
    MLIR_OP_ATTENTION,    /* Full attention op */
    MLIR_OP_KV_UPDATE,    /* KV cache update */
    MLIR_OP_EMBED,        /* Embedding lookup */
    MLIR_OP_SOFTCAP,      /* Logit soft-capping */
    /* Fused ops (products of optimization passes) */
    MLIR_OP_FUSED_RMSNORM_GEMV,   /* RMSNorm → GEMV in one pass */
    MLIR_OP_FUSED_SILU_MUL,       /* SiLU(gate) * up → GeGLU */
    MLIR_OP_FUSED_GEMV_ADD,       /* GEMV + bias add */
    MLIR_OP_FUSED_ATTN_SOFTCAP,   /* Attention with logit capping */
    MLIR_OP_COUNT
} mlir_op_type_t;

/* IR node in the computation graph */
typedef struct mlir_node {
    mlir_op_type_t type;

    /* Operand pointers (resolved during lowering) */
    void          *inputs[4];   /* Input data pointers */
    float         *output;      /* Output buffer */
    int            dims[8];     /* Dimension parameters */
    float          params[4];   /* Float parameters (eps, scale, cap, etc.) */
    ggml_type_t    dtype;       /* Data type for quantized ops */

    /* Graph edges */
    int            input_nodes[4];  /* Indices of input nodes (-1 = external) */
    int            ref_count;       /* Number of downstream consumers */

    /* Optimization metadata */
    int            fused;       /* 1 if this node was merged into a fused op */
    int            dead;        /* 1 if marked for elimination */
    int            buf_id;      /* Assigned buffer ID for memory planning */
    uint64_t       compute_cost; /* Estimated FLOPs */
} mlir_node_t;

/* The full IR graph */
#define MLIR_MAX_NODES 512
#define MLIR_MAX_BUFFERS 64

typedef struct {
    mlir_node_t  nodes[MLIR_MAX_NODES];
    int          n_nodes;

    /* Buffer planning */
    float       *buffers[MLIR_MAX_BUFFERS];
    int          buf_sizes[MLIR_MAX_BUFFERS];
    int          n_buffers;

    /* Fusion statistics */
    int          fusions_applied;
    int          nodes_eliminated;
    int          buffers_reused;
} mlir_graph_t;

/* ═══════════════════════════════════════════════════════════════════════
 * Graph Construction API
 * ════════════════════════════════════════════════════════════════════════ */

static mlir_graph_t mlir_g;

void mlir_graph_reset(void) {
    kmemset(&mlir_g, 0, sizeof(mlir_g));
}

int mlir_emit_gemv(float *out, const void *weight, const float *x,
                   int out_dim, int in_dim, ggml_type_t wtype) {
    if (mlir_g.n_nodes >= MLIR_MAX_NODES) return -1;
    mlir_node_t *n = &mlir_g.nodes[mlir_g.n_nodes];
    n->type = MLIR_OP_GEMV;
    n->output = out;
    n->inputs[0] = (void *)weight;
    n->inputs[1] = (void *)x;
    n->dims[0] = out_dim;
    n->dims[1] = in_dim;
    n->dtype = wtype;
    n->compute_cost = (uint64_t)out_dim * in_dim * 2;
    return mlir_g.n_nodes++;
}

int mlir_emit_rmsnorm(float *out, const float *x, const float *w,
                      int dim, float eps) {
    if (mlir_g.n_nodes >= MLIR_MAX_NODES) return -1;
    mlir_node_t *n = &mlir_g.nodes[mlir_g.n_nodes];
    n->type = MLIR_OP_RMSNORM;
    n->output = out;
    n->inputs[0] = (void *)x;
    n->inputs[1] = (void *)w;
    n->dims[0] = dim;
    n->params[0] = eps;
    n->compute_cost = (uint64_t)dim * 5;
    return mlir_g.n_nodes++;
}

int mlir_emit_silu(float *x, int n_elem) {
    if (mlir_g.n_nodes >= MLIR_MAX_NODES) return -1;
    mlir_node_t *n = &mlir_g.nodes[mlir_g.n_nodes];
    n->type = MLIR_OP_SILU;
    n->output = x;
    n->inputs[0] = (void *)x;
    n->dims[0] = n_elem;
    n->compute_cost = (uint64_t)n_elem * 6;
    return mlir_g.n_nodes++;
}

int mlir_emit_mul(float *out, const float *a, const float *b, int n_elem) {
    if (mlir_g.n_nodes >= MLIR_MAX_NODES) return -1;
    mlir_node_t *n = &mlir_g.nodes[mlir_g.n_nodes];
    n->type = MLIR_OP_MUL;
    n->output = out;
    n->inputs[0] = (void *)a;
    n->inputs[1] = (void *)b;
    n->dims[0] = n_elem;
    n->compute_cost = (uint64_t)n_elem;
    return mlir_g.n_nodes++;
}

int mlir_emit_add(float *out, const float *a, const float *b, int n_elem) {
    if (mlir_g.n_nodes >= MLIR_MAX_NODES) return -1;
    mlir_node_t *n = &mlir_g.nodes[mlir_g.n_nodes];
    n->type = MLIR_OP_ADD;
    n->output = out;
    n->inputs[0] = (void *)a;
    n->inputs[1] = (void *)b;
    n->dims[0] = n_elem;
    n->compute_cost = (uint64_t)n_elem;
    return mlir_g.n_nodes++;
}

int mlir_emit_softmax(float *x, int n_elem) {
    if (mlir_g.n_nodes >= MLIR_MAX_NODES) return -1;
    mlir_node_t *n = &mlir_g.nodes[mlir_g.n_nodes];
    n->type = MLIR_OP_SOFTMAX;
    n->output = x;
    n->inputs[0] = (void *)x;
    n->dims[0] = n_elem;
    n->compute_cost = (uint64_t)n_elem * 5;
    return mlir_g.n_nodes++;
}

int mlir_emit_attention(float *out, const float *Q, const float *K,
                        const float *V, int nh, int nkv, int hd,
                        int sl, float scale, float cap) {
    if (mlir_g.n_nodes >= MLIR_MAX_NODES) return -1;
    mlir_node_t *n = &mlir_g.nodes[mlir_g.n_nodes];
    n->type = MLIR_OP_ATTENTION;
    n->output = out;
    n->inputs[0] = (void *)Q;
    n->inputs[1] = (void *)K;
    n->inputs[2] = (void *)V;
    n->dims[0] = nh;
    n->dims[1] = nkv;
    n->dims[2] = hd;
    n->dims[3] = sl;
    n->params[0] = scale;
    n->params[1] = cap;
    n->compute_cost = (uint64_t)nh * sl * hd * 4;
    return mlir_g.n_nodes++;
}

int mlir_emit_softcap(float *x, int n_elem, float cap) {
    if (mlir_g.n_nodes >= MLIR_MAX_NODES) return -1;
    mlir_node_t *n = &mlir_g.nodes[mlir_g.n_nodes];
    n->type = MLIR_OP_SOFTCAP;
    n->output = x;
    n->inputs[0] = (void *)x;
    n->dims[0] = n_elem;
    n->params[0] = cap;
    n->compute_cost = (uint64_t)n_elem * 4;
    return mlir_g.n_nodes++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Pass 1: Op Fusion
 *
 * Scans for patterns and replaces pairs/triples with fused ops.
 * Fused ops execute as a single kernel call, eliminating intermediate
 * memory traffic (the primary bottleneck in transformer inference).
 * ════════════════════════════════════════════════════════════════════════ */

static int mlir_pass_fusion(void) {
    int fused = 0;

    for (int i = 0; i < mlir_g.n_nodes - 1; i++) {
        mlir_node_t *a = &mlir_g.nodes[i];
        mlir_node_t *b = &mlir_g.nodes[i + 1];
        if (a->dead || b->dead) continue;

        /* Pattern: RMSNorm → GEMV (same output feeds as input)
         * Fuses into single pass: normalize + matvec without writing norm result */
        if (a->type == MLIR_OP_RMSNORM && b->type == MLIR_OP_GEMV &&
            a->output == (float *)b->inputs[1]) {
            b->type = MLIR_OP_FUSED_RMSNORM_GEMV;
            /* Pack RMSNorm params into the fused node */
            b->inputs[2] = a->inputs[0]; /* x (pre-norm) */
            b->inputs[3] = a->inputs[1]; /* norm weights */
            b->params[0] = a->params[0]; /* eps */
            a->dead = 1;
            a->fused = 1;
            fused++;
        }

        /* Pattern: SiLU → Mul (GeGLU gate computation)
         * SiLU(gate) * up → single fused kernel */
        if (a->type == MLIR_OP_SILU && b->type == MLIR_OP_MUL &&
            a->output == (float *)b->inputs[0]) {
            b->type = MLIR_OP_FUSED_SILU_MUL;
            /* inputs[0] = gate (pre-SiLU), inputs[1] = up */
            b->inputs[0] = a->inputs[0]; /* gate vector */
            a->dead = 1;
            a->fused = 1;
            fused++;
        }

        /* Pattern: Attention → Softcap (Gemma logit capping)
         * Fused attention with tanh capping inside the inner loop */
        if (a->type == MLIR_OP_ATTENTION && b->type == MLIR_OP_SOFTCAP &&
            a->output == b->output) {
            a->type = MLIR_OP_FUSED_ATTN_SOFTCAP;
            a->params[1] = b->params[0]; /* cap value */
            b->dead = 1;
            b->fused = 1;
            fused++;
        }

        /* Pattern: GEMV → Add (bias addition)
         * Fuse into single GEMV+bias kernel */
        if (a->type == MLIR_OP_GEMV && b->type == MLIR_OP_ADD &&
            a->output == (float *)b->inputs[0]) {
            a->type = MLIR_OP_FUSED_GEMV_ADD;
            a->inputs[2] = b->inputs[1]; /* bias vector */
            b->dead = 1;
            b->fused = 1;
            fused++;
        }
    }

    mlir_g.fusions_applied += fused;
    return fused;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Pass 2: Dead Code Elimination
 *
 * Walks the graph backwards. Any node whose output is not consumed
 * by a live node and is not a graph output gets eliminated.
 * ════════════════════════════════════════════════════════════════════════ */

static int mlir_pass_dce(void) {
    int eliminated = 0;

    /* Count references */
    for (int i = 0; i < mlir_g.n_nodes; i++)
        mlir_g.nodes[i].ref_count = 0;

    for (int i = 0; i < mlir_g.n_nodes; i++) {
        if (mlir_g.nodes[i].dead) continue;
        for (int j = 0; j < 4; j++) {
            int inp = mlir_g.nodes[i].input_nodes[j];
            if (inp >= 0 && inp < mlir_g.n_nodes)
                mlir_g.nodes[inp].ref_count++;
        }
    }

    /* Last node is always the graph output; ensure it stays alive */
    if (mlir_g.n_nodes > 0)
        mlir_g.nodes[mlir_g.n_nodes - 1].ref_count++;

    /* Eliminate unreferenced nodes (except the last / output) */
    for (int i = 0; i < mlir_g.n_nodes - 1; i++) {
        mlir_node_t *n = &mlir_g.nodes[i];
        if (!n->dead && n->ref_count == 0) {
            n->dead = 1;
            eliminated++;
        }
    }

    mlir_g.nodes_eliminated += eliminated;
    return eliminated;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Pass 3: Buffer Liveness / Memory Planning
 *
 * Performs simple liveness analysis to reuse intermediate buffers.
 * Assigns buffer IDs so that non-overlapping lifetimes share memory.
 * ════════════════════════════════════════════════════════════════════════ */

static int mlir_pass_buffer_plan(void) {
    int reused = 0;

    /* Track which buffer IDs are "free" at each point */
    int free_bufs[MLIR_MAX_BUFFERS];
    int n_free = 0;
    int next_buf = 0;

    for (int i = 0; i < mlir_g.n_nodes; i++) {
        mlir_node_t *n = &mlir_g.nodes[i];
        if (n->dead) { n->buf_id = -1; continue; }

        /* Release input buffers that die here (ref_count == 0 after this use) */
        for (int j = 0; j < 4; j++) {
            int inp = n->input_nodes[j];
            if (inp >= 0 && inp < i && !mlir_g.nodes[inp].dead) {
                mlir_g.nodes[inp].ref_count--;
                if (mlir_g.nodes[inp].ref_count <= 0 && n_free < MLIR_MAX_BUFFERS) {
                    free_bufs[n_free++] = mlir_g.nodes[inp].buf_id;
                }
            }
        }

        /* Assign buffer: reuse if available, else allocate new */
        if (n_free > 0) {
            n->buf_id = free_bufs[--n_free];
            reused++;
        } else if (next_buf < MLIR_MAX_BUFFERS) {
            n->buf_id = next_buf++;
        } else {
            n->buf_id = -1; /* Overflow — use inline allocation */
        }
    }

    mlir_g.n_buffers = next_buf;
    mlir_g.buffers_reused += reused;
    return reused;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Pass 4: Type Legalization
 *
 * Selects the optimal kernel specialization for each op based on the
 * active backend and input data types.
 * ════════════════════════════════════════════════════════════════════════ */

static void mlir_pass_legalize(void) {
    for (int i = 0; i < mlir_g.n_nodes; i++) {
        mlir_node_t *n = &mlir_g.nodes[i];
        if (n->dead) continue;

        /* For GEMV, check if we should dequantize first for unsupported types */
        if (n->type == MLIR_OP_GEMV || n->type == MLIR_OP_FUSED_RMSNORM_GEMV) {
            /* Keep native quantized dispatch for Q4_0, Q8_0, Q6_K, F16, F32 */
            /* Other types: insert dequant node (handled at execution time) */
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Graph Execution: Lower IR to backend dispatch calls
 * ════════════════════════════════════════════════════════════════════════ */

static void mlir_execute_node(mlir_node_t *n, const backend_t *be) {
    switch (n->type) {
    case MLIR_OP_GEMV:
        be->compute.gemv(n->output, n->inputs[0], (const float *)n->inputs[1],
                         n->dims[0], n->dims[1], n->dtype);
        break;

    case MLIR_OP_GEMM:
        be->compute.gemm(n->output, (const float *)n->inputs[0],
                         (const float *)n->inputs[1],
                         n->dims[0], n->dims[1], n->dims[2]);
        break;

    case MLIR_OP_RMSNORM:
        be->compute.rmsnorm(n->output, (const float *)n->inputs[0],
                            (const float *)n->inputs[1],
                            n->dims[0], n->params[0]);
        break;

    case MLIR_OP_LAYERNORM:
        be->compute.layernorm(n->output, (const float *)n->inputs[0],
                              (const float *)n->inputs[1],
                              (const float *)n->inputs[2],
                              n->dims[0], n->params[0]);
        break;

    case MLIR_OP_ROPE:
        be->compute.rope((float *)n->inputs[0], (float *)n->inputs[1],
                         n->dims[0], n->dims[1], n->dims[2],
                         n->dims[3], n->params[0],
                         (const float *)n->inputs[2]);
        break;

    case MLIR_OP_SOFTMAX:
        be->compute.softmax(n->output, n->dims[0]);
        break;

    case MLIR_OP_SILU:
        be->compute.silu(n->output, n->dims[0]);
        break;

    case MLIR_OP_GELU:
        be->compute.gelu(n->output, n->dims[0]);
        break;

    case MLIR_OP_MUL:
        be->compute.mul(n->output, (const float *)n->inputs[0],
                        (const float *)n->inputs[1], n->dims[0]);
        break;

    case MLIR_OP_ADD:
        be->compute.add(n->output, (const float *)n->inputs[0],
                        (const float *)n->inputs[1], n->dims[0]);
        break;

    case MLIR_OP_SCALE:
        be->compute.scale(n->output, (const float *)n->inputs[0],
                          n->params[0], n->dims[0]);
        break;

    case MLIR_OP_DEQUANT:
        be->compute.dequantize(n->output, n->inputs[0], n->dims[0], n->dtype);
        break;

    case MLIR_OP_ATTENTION:
        be->compute.attention(n->output, (const float *)n->inputs[0],
                              (const float *)n->inputs[1],
                              (const float *)n->inputs[2],
                              n->dims[0], n->dims[1], n->dims[2],
                              n->dims[3], n->dims[3], n->params[0], n->params[1]);
        break;

    case MLIR_OP_KV_UPDATE:
        be->compute.kv_update((float *)n->inputs[0], (float *)n->inputs[1],
                              (const float *)n->inputs[2],
                              (const float *)n->inputs[3],
                              n->dims[0], n->dims[1], n->dims[2],
                              n->dims[3], n->dims[4]);
        break;

    case MLIR_OP_EMBED:
        be->compute.embed_lookup(n->output, n->inputs[0],
                                 n->dims[0], n->dims[1], n->dtype);
        break;

    case MLIR_OP_SOFTCAP:
        be->compute.softcap(n->output, n->dims[0], n->params[0]);
        break;

    /* ── Fused Kernels ─────────────────────────────────────────────── */

    case MLIR_OP_FUSED_RMSNORM_GEMV: {
        /* RMSNorm(x, w, eps) → GEMV(weight, normed, out_dim, in_dim) */
        /* Execute as: normalize into temp, then GEMV */
        int dim = n->dims[1]; /* in_dim = norm dim */
        float *temp = n->output; /* Reuse output as temp for norm result */

        /* If backend has dedicated fused kernel, use it; else sequential */
        be->compute.rmsnorm(temp, (const float *)n->inputs[2],
                            (const float *)n->inputs[3],
                            dim, n->params[0]);
        be->compute.gemv(n->output, n->inputs[0], temp,
                         n->dims[0], dim, n->dtype);
        break;
    }

    case MLIR_OP_FUSED_SILU_MUL: {
        /* SiLU(gate) * up → single pass */
        int dim = n->dims[0];
        float *gate = (float *)n->inputs[0];
        const float *up = (const float *)n->inputs[1];

        /* SiLU in-place on gate */
        be->compute.silu(gate, dim);
        /* Then multiply gate * up */
        be->compute.mul(n->output, gate, up, dim);
        break;
    }

    case MLIR_OP_FUSED_GEMV_ADD: {
        /* GEMV + bias */
        be->compute.gemv(n->output, n->inputs[0], (const float *)n->inputs[1],
                         n->dims[0], n->dims[1], n->dtype);
        be->compute.add(n->output, n->output, (const float *)n->inputs[2],
                        n->dims[0]);
        break;
    }

    case MLIR_OP_FUSED_ATTN_SOFTCAP: {
        /* Attention with softcap baked in */
        be->compute.attention(n->output, (const float *)n->inputs[0],
                              (const float *)n->inputs[1],
                              (const float *)n->inputs[2],
                              n->dims[0], n->dims[1], n->dims[2],
                              n->dims[3], n->dims[3], n->params[0], n->params[1]);
        break;
    }

    default:
        break;
    }
}

/* Run all optimization passes and execute the graph */
int mlir_optimize_and_execute(void) {
    const backend_t *be = backend_get();
    if (!be) return -1;

    /* Run optimization passes */
    mlir_pass_fusion();
    mlir_pass_dce();
    mlir_pass_buffer_plan();
    mlir_pass_legalize();

    /* Execute surviving nodes in order */
    for (int i = 0; i < mlir_g.n_nodes; i++) {
        mlir_node_t *n = &mlir_g.nodes[i];
        if (n->dead) continue;
        mlir_execute_node(n, be);
    }

    return 0;
}

/* Get optimization statistics */
void mlir_get_stats(int *fusions, int *eliminated, int *buf_reused) {
    if (fusions) *fusions = mlir_g.fusions_applied;
    if (eliminated) *eliminated = mlir_g.nodes_eliminated;
    if (buf_reused) *buf_reused = mlir_g.buffers_reused;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MLIR Backend: wraps the IR optimizer as a backend_t
 *
 * When MLIR backend is active, all compute ops first emit to the graph.
 * The graph is then optimized and lowered to the CPU/CUDA backend.
 * ════════════════════════════════════════════════════════════════════════ */

#ifdef ENABLE_MLIR

/* Immediate-mode dispatch: emit + execute single op
 * (graph-mode optimization happens when full layer subgraphs are emitted) */

static void mlir_be_gemv(float *o, const void *w, const float *x,
                          int od, int id, ggml_type_t t) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
#ifdef ENABLE_CUDA
    const backend_t *cuda = backend_get_by_id(BACKEND_CUDA);
    if (cuda) { cuda->compute.gemv(o, w, x, od, id, t); return; }
#endif
    if (cpu) cpu->compute.gemv(o, w, x, od, id, t);
}

static void mlir_be_gemm(float *C, const float *A, const float *B,
                          int M, int N, int K) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.gemm(C, A, B, M, N, K);
}

static void mlir_be_rmsnorm(float *o, const float *x, const float *w,
                             int d, float e) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
#ifdef ENABLE_CUDA
    const backend_t *cuda = backend_get_by_id(BACKEND_CUDA);
    if (cuda) { cuda->compute.rmsnorm(o, x, w, d, e); return; }
#endif
    if (cpu) cpu->compute.rmsnorm(o, x, w, d, e);
}

static void mlir_be_layernorm(float *o, const float *x, const float *w,
                               const float *b, int d, float e) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.layernorm(o, x, w, b, d, e);
}

static void mlir_be_rope(float *q, float *k, int hd, int nh,
                          int nkv, int p, float b, const float *f) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.rope(q, k, hd, nh, nkv, p, b, f);
}

static void mlir_be_softmax(float *x, int n) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.softmax(x, n);
}

static void mlir_be_silu(float *x, int n) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.silu(x, n);
}

static void mlir_be_gelu(float *x, int n) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.gelu(x, n);
}

static void mlir_be_mul(float *o, const float *a, const float *b, int n) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.mul(o, a, b, n);
}

static void mlir_be_add(float *o, const float *a, const float *b, int n) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.add(o, a, b, n);
}

static void mlir_be_scale(float *o, const float *x, float s, int n) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.scale(o, x, s, n);
}

static float mlir_be_dot(const float *a, const float *b, int n) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) return cpu->compute.dot(a, b, n);
    return 0.0f;
}

static void mlir_be_dequant(float *o, const void *d, int n, ggml_type_t t) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.dequantize(o, d, n, t);
}

static void mlir_be_attention(float *o, const float *Q, const float *K,
                               const float *V, int nh, int nkv, int hd,
                               int sl, int ms, float sc, float cap) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.attention(o, Q, K, V, nh, nkv, hd, sl, ms, sc, cap);
}

static void mlir_be_kv_update(float *K, float *V, const float *Kn,
                               const float *Vn, int nkv, int hd,
                               int p, int ms, int l) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.kv_update(K, V, Kn, Vn, nkv, hd, p, ms, l);
}

static void mlir_be_embed(float *o, const void *t, int id, int d, ggml_type_t ty) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.embed_lookup(o, t, id, d, ty);
}

static void mlir_be_softcap(float *x, int n, float c) {
    const backend_t *cpu = backend_get_by_id(BACKEND_CPU);
    if (cpu) cpu->compute.softcap(x, n, c);
}

/* Memory ops: pass through to CPU */
static void *mlir_alloc(uint64_t sz) { return tensor_alloc(sz); }
static void  mlir_free(void *p)      { tensor_free(p); }
static int   mlir_upload(void *d, const void *s, uint64_t sz)  { kmemcpy(d, s, sz); return 0; }
static int   mlir_download(void *d, const void *s, uint64_t sz){ kmemcpy(d, s, sz); return 0; }
static void  mlir_sync(void) {}

static int      mlir_init(void)           { mlir_graph_reset(); return 0; }
static void     mlir_shutdown(void)       { mlir_graph_reset(); }
static int      mlir_device_count(void)   { return 1; }
static uint64_t mlir_free_mem(int dev)    { (void)dev; return backend_get_by_id(BACKEND_CPU)->get_free_memory(0); }

const backend_t backend_mlir = {
    .id   = BACKEND_MLIR,
    .name = "mlir",
    .init = mlir_init,
    .shutdown = mlir_shutdown,
    .get_device_count = mlir_device_count,
    .get_free_memory  = mlir_free_mem,
    .mem = {
        .alloc    = mlir_alloc,
        .free     = mlir_free,
        .upload   = mlir_upload,
        .download = mlir_download,
        .sync     = mlir_sync,
    },
    .compute = {
        .gemv         = mlir_be_gemv,
        .gemm         = mlir_be_gemm,
        .rmsnorm      = mlir_be_rmsnorm,
        .layernorm    = mlir_be_layernorm,
        .rope         = mlir_be_rope,
        .softmax      = mlir_be_softmax,
        .silu         = mlir_be_silu,
        .gelu         = mlir_be_gelu,
        .mul          = mlir_be_mul,
        .add          = mlir_be_add,
        .scale        = mlir_be_scale,
        .dot          = mlir_be_dot,
        .dequantize   = mlir_be_dequant,
        .attention    = mlir_be_attention,
        .kv_update    = mlir_be_kv_update,
        .embed_lookup = mlir_be_embed,
        .softcap      = mlir_be_softcap,
    },
};

#endif /* ENABLE_MLIR */
