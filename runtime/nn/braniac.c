/* =============================================================================
 * TensorOS — Braniac Implementation
 *
 * Brain-inspired predictive coding with JIT-compiled hot paths.
 * See braniac.h for architecture overview and SPEC.md for theory.
 * =============================================================================*/

#include "runtime/nn/braniac.h"
#include "runtime/tensor/tensor_cpu.h"
#include "runtime/jit/x86_jit.h"

/* =============================================================================
 * PRNG for Weight Initialization (xorshift64)
 * =============================================================================*/

static uint64_t braniac_rng = 0xB4A1A1AC5EED1234ULL;

static uint64_t xorshift64(void)
{
    uint64_t x = braniac_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    braniac_rng = x;
    return x;
}

static float rand_uniform(void)
{
    return (float)(xorshift64() >> 40) / (float)(1ULL << 24);
}

/* =============================================================================
 * Activation Helpers
 * =============================================================================*/

static void apply_activation(float *out, const float *in, int n, int act)
{
    switch (act) {
    case BRANIAC_ACT_RELU:
        tensor_cpu_relu(out, in, n);
        break;
    case BRANIAC_ACT_TANH:
        for (int i = 0; i < n; i++)
            out[i] = fast_tanhf(in[i]);
        break;
    case BRANIAC_ACT_GELU:
        tensor_cpu_gelu(out, in, n);
        break;
    default:
        tensor_cpu_relu(out, in, n);
        break;
    }
}

/* =============================================================================
 * Weight Initialization — Xavier Uniform
 *
 * W ~ U[-sqrt(6/(in+out)), sqrt(6/(in+out))]
 * Generative weights start smaller (predictions initially vague)
 * =============================================================================*/

static void xavier_init(float *W, int in_dim, int out_dim, float gain)
{
    float scale = gain * fast_sqrtf(6.0f / (float)(in_dim + out_dim));
    int total = in_dim * out_dim;
    for (int i = 0; i < total; i++)
        W[i] = (rand_uniform() * 2.0f - 1.0f) * scale;
}

static void zero_init(float *buf, int n)
{
    for (int i = 0; i < n; i++)
        buf[i] = 0.0f;
}

/* =============================================================================
 * Default Configuration
 * =============================================================================*/

braniac_config_t braniac_default_config(void)
{
    braniac_config_t c;
    kmemset(&c, 0, sizeof(c));

    c.layer_sizes[0] = 784;
    c.layer_sizes[1] = 256;
    c.layer_sizes[2] = 128;
    c.layer_sizes[3] = 64;
    c.layer_sizes[4] = 10;
    c.num_layers = 5;

    /* Phase 1 */
    c.pc_lr         = 0.05f;
    c.gen_lr        = 0.005f;
    c.precision_lr  = 0.0005f;
    c.pc_iterations = 5;

    /* Phase 2 */
    c.refine_lr       = 0.0005f;
    c.refine_schedule = BRANIAC_SCHED_COSINE;
    c.refine_warmup   = 100;

    /* Phase 3 */
    c.sparsity_k = 0.2f;

    /* Flags */
    c.activation     = BRANIAC_ACT_RELU;
    c.use_precision  = 1;
    c.use_lateral    = 1;
    c.use_refinement = 1;
    c.total_steps    = 10000;

    return c;
}

/* =============================================================================
 * Network Initialization
 * =============================================================================*/

int braniac_init(braniac_network_t *net, const braniac_config_t *config)
{
    kmemset(net, 0, sizeof(*net));

    int nc = config->num_layers - 1;
    if (nc < 1 || nc > BRANIAC_MAX_COLUMNS) return -1;

    /* Validate dimensions */
    for (int i = 0; i < config->num_layers; i++) {
        if (config->layer_sizes[i] < 1 || config->layer_sizes[i] > BRANIAC_MAX_DIM)
            return -1;
        net->layer_sizes[i] = config->layer_sizes[i];
    }

    net->num_layers  = config->num_layers;
    net->num_columns = nc;
    net->activation  = config->activation;
    net->jit_ready   = 0;

    /* Initialize each column */
    for (int i = 0; i < nc; i++) {
        braniac_column_t *col = &net->columns[i];
        int in_dim  = config->layer_sizes[i];
        int out_dim = config->layer_sizes[i + 1];
        col->in_dim  = in_dim;
        col->out_dim = out_dim;

        /* Forward weights: Xavier with gain 1.0 */
        xavier_init(col->W_forward, in_dim, out_dim, 1.0f);
        zero_init(col->W_bias, out_dim);

        /* Generative weights: Xavier with gain 0.5 (predictions start vague) */
        xavier_init(col->G, out_dim, in_dim, 0.5f);
        zero_init(col->G_bias, in_dim);

        /* Precision: start at 0 (exp(0) = 1, uniform confidence) */
        zero_init(col->log_precision, in_dim);
    }

    kprintf("[BRANIAC] Network initialized: %d layers [", config->num_layers);
    for (int i = 0; i < config->num_layers; i++) {
        kprintf("%d", config->layer_sizes[i]);
        if (i < config->num_layers - 1) kprintf("->");
    }
    kprintf("]\n");

    return 0;
}

/* =============================================================================
 * Trainer Initialization
 * =============================================================================*/

void braniac_trainer_init(braniac_trainer_t *trainer, braniac_network_t *net,
                          braniac_buffers_t *bufs, const braniac_config_t *config)
{
    kmemset(trainer, 0, sizeof(*trainer));
    trainer->network = net;
    trainer->config  = *config;
    trainer->buffers = bufs;
    trainer->step_count = 0;
    trainer->current_beta = 0.0f;
    kmemset(bufs, 0, sizeof(*bufs));
}

/* =============================================================================
 * Beta Schedule — Gradient refinement strength over training
 *
 * Brain analogue: neuromodulatory systems mature during development.
 * Early: dominated by local Hebbian learning (fast, robust)
 * Mid: global error signal gradually increases for fine-tuning
 * Late: signal decays as system becomes well-calibrated
 * =============================================================================*/

static float compute_beta(const braniac_config_t *config, int step)
{
    float base_lr = config->refine_lr;
    int warmup = config->refine_warmup;
    int total  = config->total_steps;

    if (step < warmup)
        return base_lr * ((float)step / (float)(warmup > 0 ? warmup : 1));

    float progress = (float)(step - warmup) / (float)(total - warmup > 0 ? total - warmup : 1);
    if (progress > 1.0f) progress = 1.0f;

    switch (config->refine_schedule) {
    case BRANIAC_SCHED_COSINE: {
        /* cos(π * progress) / 2 + 0.5 approximation using fast_tanhf */
        float x = progress * 3.14159265f;
        /* cos(x) ≈ 1 - x²/2 + x⁴/24 for small x; use identity for full range */
        float cos_approx = 1.0f - x * x * 0.5f + x * x * x * x * (1.0f / 24.0f);
        if (progress > 0.5f) {
            /* Better approximation for second half */
            float xr = (1.0f - progress) * 3.14159265f;
            cos_approx = -(1.0f - xr * xr * 0.5f + xr * xr * xr * xr * (1.0f / 24.0f));
        }
        return base_lr * 0.5f * (1.0f + cos_approx);
    }
    case BRANIAC_SCHED_LINEAR:
        return base_lr * (1.0f - progress);
    default:
        return base_lr;
    }
}

/* =============================================================================
 * Forward Pass — Bottom-up sensory sweep (no PC dynamics)
 *
 * Brain analogue: the initial feedforward sweep (~50ms after stimulus onset)
 * before recurrent dynamics kick in. Fast but coarse.
 * =============================================================================*/

void braniac_forward(const braniac_network_t *net, float *output,
                     const float *input, int batch_size)
{
    /* Use batch GEMV through each column */
    const float *h = input;
    /* Ping-pong buffer for intermediate activations (static — not reentrant) */
    static float fwd_buf[2][BRANIAC_MAX_BATCH * BRANIAC_MAX_DIM];
    int cur = 0;

    for (int i = 0; i < net->num_columns; i++) {
        const braniac_column_t *col = &net->columns[i];
        float *dst = (i == net->num_columns - 1) ? output : fwd_buf[cur];

        /* out[batch×out_dim] = in[batch×in_dim] × W_forward^T + bias */
        tensor_cpu_batch_gemv(dst, h, col->W_forward, col->W_bias,
                              batch_size, col->in_dim, col->out_dim,
                              (net->activation == BRANIAC_ACT_RELU) ? 1 : 0);

        /* Non-relu activations applied separately */
        if (net->activation != BRANIAC_ACT_RELU) {
            int n = batch_size * col->out_dim;
            apply_activation(dst, dst, n, net->activation);
        }

        h = dst;
        cur ^= 1;
    }
}

/* =============================================================================
 * PC Value Neuron Initialization — Forward sweep to seed representations
 * =============================================================================*/

static void init_value_neurons(braniac_network_t *net, braniac_buffers_t *bufs,
                               const float *input, int batch_size)
{
    int in_dim = net->layer_sizes[0];

    /* Layer 0 = clamped input */
    kmemcpy(bufs->values[0], input, (size_t)batch_size * in_dim * sizeof(float));

    /* Forward through each column */
    for (int i = 0; i < net->num_columns; i++) {
        braniac_column_t *col = &net->columns[i];
        tensor_cpu_batch_gemv(bufs->values[i + 1], bufs->values[i],
                              col->W_forward, col->W_bias,
                              batch_size, col->in_dim, col->out_dim,
                              (net->activation == BRANIAC_ACT_RELU) ? 1 : 0);
        if (net->activation != BRANIAC_ACT_RELU) {
            int n = batch_size * col->out_dim;
            apply_activation(bufs->values[i + 1], bufs->values[i + 1], n,
                             net->activation);
        }
    }

    /* Cache exp(log_precision) for each column */
    for (int i = 0; i < net->num_columns; i++) {
        int d = net->columns[i].in_dim;
        for (int j = 0; j < d; j++)
            bufs->precision[i][j] = fast_expf(net->columns[i].log_precision[j]);
    }
}

/* =============================================================================
 * PC Inference Step — One iteration of cortical settling
 *
 * For each column i:
 *   1. Compute prediction: pred = G_i × values[i+1] + G_bias
 *   2. Compute error: e_i = precision_i × (values[i] - pred)
 *   3. Compute feedback: fb = G_i^T × e_i (error projected to upper layer)
 *   4. Update: values[i+1] += pc_lr × (-e_{i+1} + fb_i)
 *
 * Brain analogue: recurrent dynamics between cortical layers (~20-80ms)
 * =============================================================================*/

static void pc_inference_step_c(braniac_network_t *net, braniac_buffers_t *bufs,
                                int batch_size, float pc_lr)
{
    int nc = net->num_columns;

    /* Step 1: Compute all prediction errors */
    for (int i = 0; i < nc; i++) {
        braniac_column_t *col = &net->columns[i];
        int in_dim  = col->in_dim;
        int out_dim = col->out_dim;

        for (int b = 0; b < batch_size; b++) {
            float *target = &bufs->values[i][b * in_dim];
            float *r      = &bufs->values[i + 1][b * out_dim];
            float *err    = &bufs->errors[i][b * in_dim];

            /* Prediction: G × r + G_bias */
            for (int j = 0; j < in_dim; j++) {
                float pred = col->G_bias[j];
                for (int k = 0; k < out_dim; k++)
                    pred += col->G[j * out_dim + k] * r[k];
                /* Precision-weighted error */
                err[j] = bufs->precision[i][j] * (target[j] - pred);
            }
        }
    }

    /* Step 2: Update value neurons at each non-input level */
    for (int layer = 1; layer < net->num_layers; layer++) {
        int dim = net->layer_sizes[layer];
        int col_below = layer - 1;  /* Column that predicts layer-1 from layer */

        for (int b = 0; b < batch_size; b++) {
            float *v = &bufs->values[layer][b * dim];

            /* Term 1: Negative error at this level */
            float neg_error[BRANIAC_MAX_DIM];
            if (layer < nc) {
                /* This layer is predicted by column[layer] from layer+1 */
                int ed = net->columns[layer].in_dim; /* == dim */
                for (int j = 0; j < ed; j++)
                    neg_error[j] = -bufs->errors[layer][b * ed + j];
            } else {
                for (int j = 0; j < dim; j++)
                    neg_error[j] = 0.0f;
            }

            /* Term 2: Feedback from error at level below via G^T */
            braniac_column_t *col = &net->columns[col_below];
            int in_dim  = col->in_dim;
            int out_dim = col->out_dim; /* == dim */
            float fb[BRANIAC_MAX_DIM];
            for (int k = 0; k < out_dim; k++) {
                float sum = 0.0f;
                for (int j = 0; j < in_dim; j++)
                    sum += bufs->errors[col_below][b * in_dim + j]
                           * col->G[j * out_dim + k];
                fb[k] = sum;
            }

            /* Update value neurons */
            for (int j = 0; j < dim; j++) {
                v[j] += pc_lr * (neg_error[j] + fb[j]);
            }

            /* Re-apply activation */
            apply_activation(v, v, dim, net->activation);
        }
    }
}

/* =============================================================================
 * JIT-Compiled PC Column Step
 *
 * Compiles a native x86_64 function that fuses:
 *   prediction GEMV + error computation + feedback GEMV^T
 * into a single SSE2-vectorized kernel for one column.
 *
 * Function signature:
 *   void col_step(float *error_out, float *feedback_out,
 *                 const float *target, const float *r,
 *                 const float *G, const float *G_bias,
 *                 const float *precision)
 *
 * ABI: rdi=error_out, rsi=feedback_out, rdx=target, rcx=r,
 *      r8=G, r9=G_bias, [rsp+8]=precision
 * =============================================================================*/

static braniac_jit_col_fn jit_compile_col_step(int in_dim, int out_dim)
{
    jit_buf_t *b = jit_create(8192);
    if (!b) return NULL;

    jit_prologue(b);

    /* Save callee-saved registers we'll use */
    jit_push(b, RBX);
    jit_push(b, R12);
    jit_push(b, R13);
    jit_push(b, R14);
    jit_push(b, R15);

    /* Map arguments:
     * rdi = error_out, rsi = feedback_out, rdx = target, rcx = r
     * r8 = G, r9 = G_bias
     * precision is at [rbp + 16] (first stack arg after return addr + saved rbp)
     */
    jit_mov_reg_reg(b, R12, RDI);   /* R12 = error_out */
    jit_mov_reg_reg(b, R13, RSI);   /* R13 = feedback_out */
    jit_mov_reg_reg(b, R14, RDX);   /* R14 = target */
    jit_mov_reg_reg(b, R15, RCX);   /* R15 = r */
    /* R8 = G, R9 = G_bias already in place */
    /* Load precision pointer from stack */
    jit_mov_reg_mem(b, RBX, RBP, 16); /* RBX = precision */

    /* Zero feedback_out[0..out_dim) */
    jit_mov_reg_reg(b, RDI, R13);
    jit_xor_reg_reg(b, RAX, RAX);
    int out_4 = out_dim & ~3;
    /* Zero 4 floats at a time using XMM */
    jit_xorps(b, XMM7, XMM7);
    for (int k = 0; k < out_4; k += 4) {
        jit_movups_store(b, R13, k * 4, XMM7);
    }
    /* Zero remaining */
    for (int k = out_4; k < out_dim; k++) {
        jit_movss_store(b, R13, k * 4, XMM7);
    }

    /* Main loop: for each row j of G */
    /* For j = 0 to in_dim:
     *   1. Compute prediction_j = dot(G[j,:], r) + G_bias[j]  (SSE2 dot product)
     *   2. Compute error_j = precision[j] * (target[j] - prediction_j)
     *   3. Scatter: feedback[:] += error_j * G[j,:] */
    for (int j = 0; j < in_dim; j++) {
        int g_row_off = j * out_dim * 4;

        /* === Dot product: G[j,:] · r  using SSE2 === */
        jit_xorps(b, XMM0, XMM0);  /* acc = 0 */
        for (int k = 0; k < out_4; k += 4) {
            jit_movups_load(b, XMM1, R8, g_row_off + k * 4);   /* G[j,k:k+4] */
            jit_movups_load(b, XMM2, R15, k * 4);               /* r[k:k+4] */
            jit_mulps(b, XMM1, XMM2);
            jit_addps(b, XMM0, XMM1);
        }
        /* Handle remainder */
        for (int k = out_4; k < out_dim; k++) {
            jit_movss_load(b, XMM1, R8, g_row_off + k * 4);
            jit_movss_load(b, XMM2, R15, k * 4);
            jit_mulss(b, XMM1, XMM2);
            jit_addss(b, XMM0, XMM1);
        }
        /* Horizontal sum of xmm0: [a,b,c,d] -> a+b+c+d */
        jit_movaps_reg(b, XMM1, XMM0);
        jit_shufps(b, XMM1, XMM1, 0x4E);  /* [c,d,a,b] */
        jit_addps(b, XMM0, XMM1);
        jit_movaps_reg(b, XMM1, XMM0);
        jit_shufps(b, XMM1, XMM1, 0xB1);  /* [b,a,d,c] */
        jit_addss(b, XMM0, XMM1);
        /* XMM0[0] = dot product */

        /* Add G_bias[j] */
        jit_addss_mem(b, XMM0, R9, j * 4);
        /* XMM0[0] = prediction_j */

        /* === Error: precision[j] * (target[j] - prediction_j) === */
        jit_movss_load(b, XMM3, R14, j * 4);    /* target[j] */
        /* subss xmm3, xmm0 → target - prediction */
        /* need subss: 0xF3 0x0F 0x5C /r */
        jit_emit8(b, 0xF3); jit_emit8(b, 0x0F); jit_emit8(b, 0x5C);
        jit_emit8(b, 0xC0 | (XMM3 << 3) | XMM0);
        /* XMM3[0] = target - prediction */
        jit_movss_load(b, XMM4, RBX, j * 4);    /* precision[j] */
        jit_mulss(b, XMM3, XMM4);
        /* XMM3[0] = error_j */
        jit_movss_store(b, R12, j * 4, XMM3);   /* store error[j] */

        /* === Scatter: feedback[:] += error_j * G[j,:] === */
        /* Broadcast error_j to all 4 lanes of XMM3 */
        jit_shufps(b, XMM3, XMM3, 0x00);
        for (int k = 0; k < out_4; k += 4) {
            jit_movups_load(b, XMM1, R8, g_row_off + k * 4);   /* G[j,k:k+4] */
            jit_mulps(b, XMM1, XMM3);                            /* error * G */
            jit_movups_load(b, XMM2, R13, k * 4);               /* feedback[k:k+4] */
            jit_addps(b, XMM2, XMM1);
            jit_movups_store(b, R13, k * 4, XMM2);
        }
        for (int k = out_4; k < out_dim; k++) {
            jit_movss_load(b, XMM1, R8, g_row_off + k * 4);
            jit_mulss(b, XMM1, XMM3);
            jit_movss_load(b, XMM2, R13, k * 4);
            jit_addss(b, XMM2, XMM1);
            jit_movss_store(b, R13, k * 4, XMM2);
        }
    }

    /* Restore callee-saved */
    jit_pop(b, R15);
    jit_pop(b, R14);
    jit_pop(b, R13);
    jit_pop(b, R12);
    jit_pop(b, RBX);

    jit_epilogue(b);

    braniac_jit_col_fn fn = (braniac_jit_col_fn)(uintptr_t)jit_get_fn(b);
    return fn;
}

/* =============================================================================
 * JIT-Compiled Hebbian Update Kernel
 *
 * Fuses: G[j,k] += lr * error[j] * r[k]  and  G_bias[j] += lr * error[j]
 * Using SSE2 vectorized outer product accumulation.
 *
 * ABI: rdi=G, rsi=G_bias, rdx=error, rcx=r, xmm0=lr
 * =============================================================================*/

static braniac_jit_hebb_fn jit_compile_hebb_update(int in_dim, int out_dim)
{
    jit_buf_t *b = jit_create(8192);
    if (!b) return NULL;

    jit_prologue(b);
    jit_push(b, RBX);
    jit_push(b, R12);

    /* rdi=G, rsi=G_bias, rdx=error, rcx=r, xmm0=lr */
    jit_mov_reg_reg(b, R12, RDI);    /* R12 = G */
    /* rsi = G_bias, rdx = error, rcx = r already in place */
    /* XMM0 = lr */

    /* Broadcast lr to all 4 lanes of XMM6 */
    jit_movaps_reg(b, XMM6, XMM0);
    jit_shufps(b, XMM6, XMM6, 0x00);

    int out_4 = out_dim & ~3;

    /* Outer product: for each j, update G[j,:] += lr * error[j] * r[:] */
    for (int j = 0; j < in_dim; j++) {
        int g_row_off = j * out_dim * 4;

        /* Load error[j], scale by lr, broadcast */
        jit_movss_load(b, XMM5, RDX, j * 4);     /* error[j] */
        jit_mulss(b, XMM5, XMM0);                  /* lr * error[j] */
        jit_shufps(b, XMM5, XMM5, 0x00);           /* broadcast */

        /* G[j,k] += (lr * error[j]) * r[k] for all k */
        for (int k = 0; k < out_4; k += 4) {
            jit_movups_load(b, XMM1, RCX, k * 4);            /* r[k:k+4] */
            jit_mulps(b, XMM1, XMM5);                         /* scaled_err * r */
            jit_movups_load(b, XMM2, R12, g_row_off + k * 4); /* G[j,k:k+4] */
            jit_addps(b, XMM2, XMM1);
            jit_movups_store(b, R12, g_row_off + k * 4, XMM2);
        }
        for (int k = out_4; k < out_dim; k++) {
            jit_movss_load(b, XMM1, RCX, k * 4);
            jit_mulss(b, XMM1, XMM5);
            jit_movss_load(b, XMM2, R12, g_row_off + k * 4);
            jit_addss(b, XMM2, XMM1);
            jit_movss_store(b, R12, g_row_off + k * 4, XMM2);
        }

        /* G_bias[j] += lr * error[j] */
        jit_movss_load(b, XMM1, RSI, j * 4);
        jit_movss_load(b, XMM2, RDX, j * 4);
        jit_mulss(b, XMM2, XMM0);  /* lr * error[j] */
        jit_addss(b, XMM1, XMM2);
        jit_movss_store(b, RSI, j * 4, XMM1);
    }

    jit_pop(b, R12);
    jit_pop(b, RBX);
    jit_epilogue(b);

    braniac_jit_hebb_fn fn = (braniac_jit_hebb_fn)(uintptr_t)jit_get_fn(b);
    return fn;
}

/* =============================================================================
 * Compile All JIT Kernels
 * =============================================================================*/

int braniac_jit_compile(braniac_network_t *net)
{
    kprintf("[BRANIAC] JIT compiling %d column kernels...\n", net->num_columns);

    for (int i = 0; i < net->num_columns; i++) {
        braniac_column_t *col = &net->columns[i];

        net->jit_col_step[i] = jit_compile_col_step(col->in_dim, col->out_dim);
        if (!net->jit_col_step[i]) {
            kprintf("[BRANIAC] JIT: column %d step kernel failed\n", i);
            return -1;
        }

        net->jit_hebb[i] = jit_compile_hebb_update(col->in_dim, col->out_dim);
        if (!net->jit_hebb[i]) {
            kprintf("[BRANIAC] JIT: column %d Hebbian kernel failed\n", i);
            return -1;
        }

        kprintf("[BRANIAC] JIT: column %d [%d->%d] compiled\n",
                i, col->in_dim, col->out_dim);
    }

    net->jit_ready = 1;
    kprintf("[BRANIAC] JIT compilation complete (%d kernels)\n",
            net->num_columns * 2);
    return 0;
}

/* =============================================================================
 * JIT PC Inference Step — Uses JIT-compiled column step kernels
 * =============================================================================*/

static void pc_inference_step_jit(braniac_network_t *net, braniac_buffers_t *bufs,
                                  int batch_size, float pc_lr)
{
    int nc = net->num_columns;

    /* Step 1: Compute errors and feedback via JIT kernels */
    for (int i = 0; i < nc; i++) {
        braniac_column_t *col = &net->columns[i];
        int in_dim  = col->in_dim;
        int out_dim = col->out_dim;

        for (int b = 0; b < batch_size; b++) {
            net->jit_col_step[i](
                &bufs->errors[i][b * in_dim],
                &bufs->feedback[i][b * out_dim],
                &bufs->values[i][b * in_dim],
                &bufs->values[i + 1][b * out_dim],
                col->G, col->G_bias,
                bufs->precision[i]
            );
        }
    }

    /* Step 2: Update value neurons */
    for (int layer = 1; layer < net->num_layers; layer++) {
        int dim = net->layer_sizes[layer];
        int col_below = layer - 1;

        for (int b = 0; b < batch_size; b++) {
            float *v = &bufs->values[layer][b * dim];

            /* Term 1: Negative error at this level */
            if (layer < nc) {
                int ed = net->columns[layer].in_dim;
                float *e = &bufs->errors[layer][b * ed];
                for (int j = 0; j < dim; j++)
                    v[j] += pc_lr * (-e[j] + bufs->feedback[col_below][b * dim + j]);
            } else {
                for (int j = 0; j < dim; j++)
                    v[j] += pc_lr * bufs->feedback[col_below][b * dim + j];
            }

            apply_activation(v, v, dim, net->activation);
        }
    }
}

/* =============================================================================
 * Full PC Dynamics — Run T_pc iterations of settling
 *
 * Brain analogue: the complete perceptual inference cycle from stimulus
 * onset to stable percept (~100-300ms)
 * =============================================================================*/

static void run_predictive_coding(braniac_trainer_t *trainer,
                                  const float *input, int batch_size)
{
    braniac_network_t *net = trainer->network;
    braniac_buffers_t *bufs = trainer->buffers;
    const braniac_config_t *cfg = &trainer->config;

    /* Initialize value neurons with forward sweep */
    init_value_neurons(net, bufs, input, batch_size);

    /* Run PC iterations */
    for (int t = 0; t < cfg->pc_iterations; t++) {
        if (net->jit_ready) {
            pc_inference_step_jit(net, bufs, batch_size, cfg->pc_lr);
        } else {
            pc_inference_step_c(net, bufs, batch_size, cfg->pc_lr);
        }
    }
}

/* =============================================================================
 * Phase 1: Local Hebbian Updates
 *
 * ΔG_i = α_g × error × r^T  (three-factor Hebbian rule)
 * Brain analogue: STDP modulated by local error signals
 * =============================================================================*/

static void local_hebbian_update(braniac_trainer_t *trainer, int batch_size)
{
    braniac_network_t *net = trainer->network;
    braniac_buffers_t *bufs = trainer->buffers;
    const braniac_config_t *cfg = &trainer->config;
    float lr = cfg->gen_lr;
    float inv_batch = 1.0f / (float)batch_size;

    for (int i = 0; i < net->num_columns; i++) {
        braniac_column_t *col = &net->columns[i];
        int in_dim  = col->in_dim;
        int out_dim = col->out_dim;

        if (net->jit_ready && net->jit_hebb[i]) {
            /* JIT path: accumulate over batch */
            for (int b = 0; b < batch_size; b++) {
                float scaled_lr = lr * inv_batch;
                /* The JIT function takes lr in xmm0 via the float argument */
                net->jit_hebb[i](
                    col->G, col->G_bias,
                    &bufs->errors[i][b * in_dim],
                    &bufs->values[i + 1][b * out_dim],
                    scaled_lr
                );
            }
        } else {
            /* C fallback: outer product update averaged over batch */
            for (int b = 0; b < batch_size; b++) {
                float *err = &bufs->errors[i][b * in_dim];
                float *r   = &bufs->values[i + 1][b * out_dim];

                for (int j = 0; j < in_dim; j++) {
                    float e_scaled = lr * inv_batch * err[j];
                    for (int k = 0; k < out_dim; k++)
                        col->G[j * out_dim + k] += e_scaled * r[k];
                    col->G_bias[j] += e_scaled;
                }
            }
        }

        /* Update precision weights */
        if (cfg->use_precision) {
            for (int j = 0; j < in_dim; j++) {
                float precision = bufs->precision[i][j];
                float mean_sq_err = 0.0f;
                for (int b = 0; b < batch_size; b++) {
                    float e = bufs->errors[i][b * in_dim + j];
                    mean_sq_err += e * e;
                }
                mean_sq_err *= inv_batch;
                float delta = 1.0f - precision * mean_sq_err;
                col->log_precision[j] += cfg->precision_lr * delta;
                /* Clamp log_precision to [-5, 5] */
                if (col->log_precision[j] > 5.0f) col->log_precision[j] = 5.0f;
                if (col->log_precision[j] < -5.0f) col->log_precision[j] = -5.0f;
            }
            /* Refresh precision cache */
            for (int j = 0; j < in_dim; j++)
                bufs->precision[i][j] = fast_expf(col->log_precision[j]);
        }
    }
}

/* =============================================================================
 * Compute Residual Loss — Total unexplained prediction error
 *
 * Brain analogue: the "surprise" signal that triggers neuromodulatory release
 * =============================================================================*/

static float compute_residual_loss(braniac_buffers_t *bufs, int nc,
                                   const int *layer_sizes, int batch_size)
{
    float total = 0.0f;
    for (int i = 0; i < nc; i++) {
        int dim = layer_sizes[i];
        for (int b = 0; b < batch_size; b++) {
            float *e = &bufs->errors[i][b * dim];
            for (int j = 0; j < dim; j++)
                total += e[j] * e[j];
        }
    }
    return total / (float)(nc * batch_size);
}

/* =============================================================================
 * Phase 2: Gradient Refinement — Lightweight backprop on residual errors
 *
 * Computes ∂L_residual/∂G_i and applies damped gradient updates.
 * The gradient is the same outer product form as Hebbian, but computed on
 * re-settled errors after Phase 1 updates.
 *
 * Additionally, if targets are provided, computes a task loss (MSE) gradient
 * at the output layer and backpropagates it through the column stack.
 *
 * Brain analogue: dopaminergic/noradrenergic broadcast signal that globally
 * modulates plasticity proportional to remaining unexplained variance.
 * =============================================================================*/

static float gradient_refinement(braniac_trainer_t *trainer,
                                 const float *targets, int batch_size,
                                 int output_dim)
{
    braniac_network_t *net = trainer->network;
    braniac_buffers_t *bufs = trainer->buffers;
    const braniac_config_t *cfg = &trainer->config;
    float beta = trainer->current_beta;
    float inv_batch = 1.0f / (float)batch_size;

    if (beta < 1e-10f) return 0.0f;

    float task_loss = 0.0f;

    /* Compute output-layer gradient from task loss (MSE) */
    float *delta = bufs->delta;
    int top = net->num_layers - 1;
    int top_dim = net->layer_sizes[top];

    if (targets && output_dim > 0 && output_dim <= top_dim) {
        for (int b = 0; b < batch_size; b++) {
            float *out = &bufs->values[top][b * top_dim];
            const float *tgt = &targets[b * output_dim];
            for (int j = 0; j < output_dim; j++) {
                float diff = out[j] - tgt[j];
                delta[b * top_dim + j] = diff;
                task_loss += diff * diff;
            }
            for (int j = output_dim; j < top_dim; j++)
                delta[b * top_dim + j] = 0.0f;
        }
        task_loss *= 0.5f * inv_batch;
    } else {
        kmemset(delta, 0, (size_t)batch_size * top_dim * sizeof(float));
    }

    /* Backpropagate delta through columns (top to bottom) */
    for (int i = net->num_columns - 1; i >= 0; i--) {
        braniac_column_t *col = &net->columns[i];
        int in_dim  = col->in_dim;
        int out_dim = col->out_dim;

        /* Weight gradient: ΔW_forward += beta * delta @ values[i]^T */
        for (int b = 0; b < batch_size; b++) {
            float *d = &delta[b * out_dim];
            float *v = &bufs->values[i][b * in_dim];
            float scale = beta * inv_batch;

            for (int n = 0; n < out_dim; n++) {
                float ds = scale * d[n];
                for (int k = 0; k < in_dim; k++)
                    col->W_forward[n * in_dim + k] -= ds * v[k];
                col->W_bias[n] -= ds;
            }
        }

        /* Also apply gradient to generative weights from residual errors */
        for (int b = 0; b < batch_size; b++) {
            float *err = &bufs->errors[i][b * in_dim];
            float *r   = &bufs->values[i + 1][b * out_dim];
            float scale = beta * inv_batch;

            for (int j = 0; j < in_dim; j++) {
                float es = scale * err[j];
                for (int k = 0; k < out_dim; k++)
                    col->G[j * out_dim + k] += es * r[k];
            }
        }

        /* Propagate delta backward: delta_below = W_forward^T @ delta */
        if (i > 0) {
            float *new_delta = bufs->delta_swap;
            for (int b = 0; b < batch_size; b++) {
                float *d = &delta[b * out_dim];
                float *nd = &new_delta[b * in_dim];
                for (int k = 0; k < in_dim; k++) {
                    float sum = 0.0f;
                    for (int n = 0; n < out_dim; n++)
                        sum += col->W_forward[n * in_dim + k] * d[n];
                    nd[k] = sum;
                }
            }
            /* Shift delta for next column */
            kmemcpy(delta, new_delta,
                    (size_t)batch_size * in_dim * sizeof(float));
        }
    }

    return task_loss;
}

/* =============================================================================
 * Phase 3: Lateral Inhibition — Top-k sparsification
 *
 * Brain analogue: GABAergic interneurons creating winner-take-all competition
 * within cortical columns. Prevents representation collapse.
 * =============================================================================*/

static float lateral_inhibition(braniac_network_t *net, braniac_buffers_t *bufs,
                                int batch_size, float sparsity_k)
{
    float total_active = 0.0f;
    int counted = 0;

    for (int layer = 1; layer < net->num_layers; layer++) {
        int dim = net->layer_sizes[layer];
        int k = (int)((float)dim * sparsity_k);
        if (k < 1) k = 1;
        if (k >= dim) continue;

        for (int b = 0; b < batch_size; b++) {
            float *v = &bufs->values[layer][b * dim];

            /* Find the k-th largest absolute value (partial selection) */
            /* Use a simple threshold approach: find k-th value by scanning */
            float abs_vals[BRANIAC_MAX_DIM];
            for (int j = 0; j < dim; j++)
                abs_vals[j] = (v[j] >= 0) ? v[j] : -v[j];

            /* Partial sort: find threshold by iterative elimination */
            float threshold = 0.0f;
            float lo = 0.0f, hi = 0.0f;
            for (int j = 0; j < dim; j++)
                if (abs_vals[j] > hi) hi = abs_vals[j];

            /* Binary search for the k-th percentile threshold */
            for (int iter = 0; iter < 20; iter++) {
                float mid = (lo + hi) * 0.5f;
                int count = 0;
                for (int j = 0; j < dim; j++)
                    if (abs_vals[j] >= mid) count++;
                if (count > k)
                    lo = mid;
                else
                    hi = mid;
            }
            threshold = lo;

            /* Zero neurons below threshold */
            int active = 0;
            for (int j = 0; j < dim; j++) {
                if (abs_vals[j] < threshold) {
                    v[j] = 0.0f;
                } else {
                    active++;
                }
            }
            total_active += (float)active / (float)dim;
            counted++;
        }
    }

    return counted > 0 ? total_active / (float)counted : 0.0f;
}

/* =============================================================================
 * Full PC Inference (evaluation path)
 * =============================================================================*/

void braniac_pc_infer(braniac_trainer_t *trainer, const float *input,
                      float *output, int batch_size)
{
    run_predictive_coding(trainer, input, batch_size);

    /* Extract top-level representation as output */
    int top = trainer->network->num_layers - 1;
    int top_dim = trainer->network->layer_sizes[top];
    kmemcpy(output, trainer->buffers->values[top],
            (size_t)batch_size * top_dim * sizeof(float));
}

/* =============================================================================
 * Training Step — All Three Phases
 * =============================================================================*/

int braniac_train_step(braniac_trainer_t *trainer,
                       const float *input, const float *targets,
                       int batch_size, int output_dim,
                       braniac_metrics_t *metrics)
{
    if (!trainer || !trainer->network || !trainer->buffers) return -1;
    if (batch_size > BRANIAC_MAX_BATCH) return -1;

    braniac_config_t *cfg = &trainer->config;
    braniac_metrics_t m;
    kmemset(&m, 0, sizeof(m));

    /* ================================================================
     * PHASE 1: Predictive Coding — Local Hebbian Learning
     * Brain analogue: cortical settling + STDP
     * ================================================================ */

    /* Run PC dynamics */
    run_predictive_coding(trainer, input, batch_size);

    /* Compute PC energy (total prediction error) */
    m.pc_energy = compute_residual_loss(trainer->buffers, trainer->network->num_columns,
                                        trainer->network->layer_sizes, batch_size);

    /* Apply local Hebbian weight updates */
    local_hebbian_update(trainer, batch_size);

    /* ================================================================
     * PHASE 2: Gradient Refinement — Neuromodulatory Fine-Tuning
     * Brain analogue: dopamine broadcast signal
     * ================================================================ */

    if (cfg->use_refinement) {
        trainer->current_beta = compute_beta(cfg, trainer->step_count);
        m.beta = trainer->current_beta;

        /* Re-run PC to get fresh errors after Hebbian update */
        run_predictive_coding(trainer, input, batch_size);

        m.residual_loss = compute_residual_loss(trainer->buffers,
                                                 trainer->network->num_columns,
                                                 trainer->network->layer_sizes,
                                                 batch_size);
        m.task_loss = gradient_refinement(trainer, targets, batch_size, output_dim);
        m.total_loss = m.residual_loss + m.task_loss;
    }

    /* ================================================================
     * PHASE 3: Lateral Inhibition — Sparsification
     * Brain analogue: GABAergic interneuron competition
     * ================================================================ */

    if (cfg->use_lateral) {
        m.avg_sparsity = lateral_inhibition(trainer->network, trainer->buffers,
                                            batch_size, cfg->sparsity_k);
    }

    m.step = trainer->step_count;
    trainer->step_count++;

    /* Cache metrics */
    trainer->last_pc_energy    = m.pc_energy;
    trainer->last_task_loss    = m.task_loss;
    trainer->last_residual_loss = m.residual_loss;
    trainer->last_sparsity     = m.avg_sparsity;

    if (metrics) *metrics = m;
    return 0;
}

/* =============================================================================
 * Evaluate — No weight updates
 * =============================================================================*/

void braniac_evaluate(braniac_trainer_t *trainer,
                      const float *input, const float *targets,
                      int batch_size, int output_dim,
                      braniac_metrics_t *metrics)
{
    braniac_metrics_t m;
    kmemset(&m, 0, sizeof(m));

    /* Fast forward inference (static buffer — not reentrant) */
    static float eval_out[BRANIAC_MAX_BATCH * BRANIAC_MAX_DIM];
    braniac_forward(trainer->network, eval_out, input, batch_size);

    /* Compute fast accuracy (argmax match for classification) */
    if (targets) {
        int top_dim = trainer->network->layer_sizes[trainer->network->num_layers - 1];
        float correct = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            /* Find predicted class */
            float *out = &eval_out[b * top_dim];
            const float *tgt = &targets[b * output_dim];
            int pred_class = 0, true_class = 0;
            float pred_max = out[0], true_max = tgt[0];
            for (int j = 1; j < output_dim && j < top_dim; j++) {
                if (out[j] > pred_max) { pred_max = out[j]; pred_class = j; }
                if (tgt[j] > true_max) { true_max = tgt[j]; true_class = j; }
            }
            if (pred_class == true_class) correct += 1.0f;
        }
        m.task_loss = correct / (float)batch_size; /* classification accuracy (0–1) */
    }

    /* Full PC inference */
    run_predictive_coding(trainer, input, batch_size);
    m.pc_energy = compute_residual_loss(trainer->buffers,
                                         trainer->network->num_columns,
                                         trainer->network->layer_sizes,
                                         batch_size);

    m.step = trainer->step_count;
    if (metrics) *metrics = m;
}

/* =============================================================================
 * Diagnostics
 * =============================================================================*/

void braniac_print_diagnostics(const braniac_trainer_t *trainer)
{
    const braniac_network_t *net = trainer->network;

    kprintf("[BRANIAC] === Diagnostics ===\n");
    kprintf("  Step: %d  Beta: %.6f\n", trainer->step_count,
            (double)trainer->current_beta);
    kprintf("  PC Energy: %.6f  Task Loss: %.6f  Residual: %.6f\n",
            (double)trainer->last_pc_energy,
            (double)trainer->last_task_loss,
            (double)trainer->last_residual_loss);
    kprintf("  Sparsity: %.2f%%\n", (double)(trainer->last_sparsity * 100.0f));
    kprintf("  JIT: %s\n", net->jit_ready ? "compiled" : "interpreted");

    for (int i = 0; i < net->num_columns; i++) {
        const braniac_column_t *col = &net->columns[i];
        /* Compute weight norms */
        float w_norm = 0.0f, g_norm = 0.0f;
        int total = col->in_dim * col->out_dim;
        for (int j = 0; j < total; j++) {
            w_norm += col->W_forward[j] * col->W_forward[j];
            g_norm += col->G[j] * col->G[j];
        }
        w_norm = fast_sqrtf(w_norm);
        g_norm = fast_sqrtf(g_norm);

        /* Precision stats */
        float p_mean = 0.0f, p_min = 100.0f, p_max = -100.0f;
        for (int j = 0; j < col->in_dim; j++) {
            float p = fast_expf(col->log_precision[j]);
            p_mean += p;
            if (p < p_min) p_min = p;
            if (p > p_max) p_max = p;
        }
        p_mean /= (float)col->in_dim;

        kprintf("  Col %d [%d->%d] W=%.3f G=%.3f prec=[%.3f, %.3f, %.3f]\n",
                i, col->in_dim, col->out_dim,
                (double)w_norm, (double)g_norm,
                (double)p_min, (double)p_mean, (double)p_max);
    }
}

/* =============================================================================
 * Self-Test — Verify convergence on a trivial XOR-like problem
 * =============================================================================*/

/* Global Braniac state for self-test and runtime use */
static braniac_network_t g_braniac_net  __attribute__((aligned(64)));
static braniac_buffers_t g_braniac_buf  __attribute__((aligned(64)));
static braniac_trainer_t g_braniac_trainer;

int braniac_selftest(void)
{
    kprintf("[BRANIAC] Running self-test...\n");

    /* Small [4->8->4->2] network for XOR-like classification */
    braniac_config_t cfg = braniac_default_config();
    cfg.layer_sizes[0] = 4;
    cfg.layer_sizes[1] = 8;
    cfg.layer_sizes[2] = 4;
    cfg.layer_sizes[3] = 2;
    cfg.num_layers = 4;
    cfg.pc_iterations = 3;
    cfg.total_steps = 200;
    cfg.use_refinement = 1;
    cfg.use_lateral = 0;   /* Disable for tiny network */

    if (braniac_init(&g_braniac_net, &cfg) != 0) {
        kprintf("[BRANIAC] FAIL: init\n");
        return -1;
    }

    braniac_trainer_init(&g_braniac_trainer, &g_braniac_net, &g_braniac_buf, &cfg);

    /* JIT compile hot-path kernels */
    braniac_jit_compile(&g_braniac_net);

    /* XOR-like training data: 4 inputs -> 2 outputs */
    float X[4][4] = {
        {0, 0, 0, 0},
        {1, 0, 1, 0},
        {0, 1, 0, 1},
        {1, 1, 1, 1},
    };
    float Y[4][2] = {
        {1, 0},  /* Class 0 */
        {0, 1},  /* Class 1 */
        {0, 1},  /* Class 1 */
        {1, 0},  /* Class 0 */
    };

    /* Train for a few steps */
    braniac_metrics_t m;
    float initial_loss = 0, final_loss = 0;
    for (int step = 0; step < 200; step++) {
        braniac_train_step(&g_braniac_trainer, (float *)X, (float *)Y,
                           4, 2, &m);
        if (step == 0) initial_loss = m.total_loss;
        if (step == 199) final_loss = m.total_loss;
    }

    int pass = (final_loss < initial_loss);
    kprintf("[BRANIAC] Self-test: initial_loss=%.4f final_loss=%.4f %s\n",
            (double)initial_loss, (double)final_loss,
            pass ? "PASS" : "FAIL");

    return pass ? 0 : -1;
}
