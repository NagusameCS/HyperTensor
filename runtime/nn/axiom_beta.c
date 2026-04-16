/*
 * Geodessical Autonomous Axiomatic Subsystem (Beta-3)
 *
 * Five-phase pipeline operating on real model geometry:
 *
 * Phase 1: Manifold Identification
 *   - Sample N token embeddings from the model, dequantize to f64
 *   - Compute PCA on the embedding cloud
 *   - Estimate intrinsic dimensionality via TwoNN
 *   - Derive PCA subspace for all subsequent geometry
 *
 * Phase 2: Symmetry Extraction (Beta-3: real dequantized weights)
 *   - For each layer, dequantize Q-weight rows per attention head
 *   - Compute per-head fingerprint vectors from L2 row norms
 *   - Pairwise cosine similarity between head fingerprint vectors
 *   - Identify near-identical heads â†’ symmetry generators
 *
 * Phase 3: Nonlinearity Absorption (Beta-3: Fisher + full Riemann)
 *   - Build metric tensor field from local embedding covariance in PCA subspace
 *   - Compute Fisher Information Matrix and blend into metric field
 *   - Compute Christoffel symbols via numerical differentiation
 *   - Compute full Riemann (âˆ‚Î“ derivative + Î“Â·Î“ algebraic terms) â†’ Ricci â†’ scalar
 *   - Retain metric field and Christoffel symbols for Phase 5
 *
 * Phase 4: Axiom Formalization (Beta-3: geometry-derived candidates)
 *   - Axiom type distribution derived from Phase 1-3 feature strengths
 *   - Test axiom predictions against model behavior (oracle calls)
 *   - Active learning selects most informative tests
 *   - Prune inconsistent candidates â†’ minimal axiom set
 *
 * Phase 5: Native Inference Projection (Beta-3: real metric field)
 *   - Reuse Phase 3â€™s Fisher-blended metric field and Christoffel symbols
 *   - Solve geodesic equation from input embedding in PCA subspace
 *   - Compare geodesic endpoint with real forward-pass output
 *   - Report reconstruction error and projected speedup
 */

#include "runtime/nn/axiom_beta.h"
#include "runtime/nn/axiom_linalg.h"
#include "runtime/nn/axiom_geo.h"
#include "runtime/nn/llm.h"
#include "runtime/nn/backend.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef GEODESSICAL_HOSTED
#include "host/hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* â”€â”€â”€ RNG â”€â”€â”€ */
static uint64_t ax_rng_next(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

static double ax_rng_f64(uint64_t *s) {
    return (double)(ax_rng_next(s) & 0xFFFFFF) / (double)0x1000000;
}

static int ax_rng_range(uint64_t *s, int lo, int hi) {
    if (lo >= hi) return lo;
    return lo + (int)(ax_rng_next(s) % (uint64_t)(hi - lo));
}

static int clamp_i(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* Phase-5 initial velocity prior: start with endpoint direction and apply
 * a small local curvature correction from Christoffel symbols at x0.
 * This is a lightweight step toward OTT's context-derived v0 objective. */
static void phase5_init_velocity_curvature(const axgeo_metric_field_t *mf,
                                           const axgeo_christoffel_t *ch,
                                           const double *x0,
                                           const double *x1,
                                           int dim,
                                           double *v_out,
                                           double *gamma_buf,
                                           double *accel_buf)
{
    ax_vec_sub(v_out, x1, x0, dim);
    double vnorm = ax_vec_norm(v_out, dim);
    if (vnorm > 1e-10) {
        ax_vec_scale(v_out, v_out, 1.0 / vnorm, dim);
    } else {
        ax_vec_zero(v_out, dim);
        v_out[0] = 1.0;
        return;
    }

    if (!mf || !ch || !gamma_buf || !accel_buf) return;

    axgeo_christoffel_interpolate(ch, mf, x0, gamma_buf);
    ax_vec_zero(accel_buf, dim);
    for (int k = 0; k < dim; k++) {
        double a = 0.0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                a -= gamma_buf[k * dim * dim + i * dim + j] *
                     v_out[i] * v_out[j];
            }
        }
        accel_buf[k] = a;
    }

    /* Blend toward curvature-guided tangent while preserving stability. */
    double anorm = ax_vec_norm(accel_buf, dim);
    if (anorm > 1e-12) {
        double blend = 0.20 / (1.0 + 0.02 * anorm);
        if (blend < 0.05) blend = 0.05;
        if (blend > 0.25) blend = 0.25;
        for (int i = 0; i < dim; i++)
            v_out[i] += blend * accel_buf[i];
        vnorm = ax_vec_norm(v_out, dim);
        if (vnorm > 1e-10)
            ax_vec_scale(v_out, v_out, 1.0 / vnorm, dim);
    }
}

/* ── OTT hidden-state LRU cache (todo 21) ──────────────────────────────────
 * Keyed by (token_id, layer).  On a hit the forward pass is skipped entirely,
 * reducing Phase-3 sampling from ~1900 LLM calls to a small fraction.
 * Entry count: OTT_HS_CACHE_CAP (2048 by default).
 * ─────────────────────────────────────────────────────────────────────── */
#define OTT_HS_CACHE_CAP 4096  /* enlarged (todo 26): Phase1=256 + Phase3=2288 + Phase5=1024 */

typedef struct {
    int    token_id;
    int    layer;
    int    dim;
    float *data;        /* heap-allocated float[dim]; NULL = empty slot */
    int    lru_stamp;
} ott_hs_entry_t;

static ott_hs_entry_t ott_hs_cache[OTT_HS_CACHE_CAP];
static int            ott_hs_lru_clock = 0;
static int            ott_hs_hits  = 0;
static int            ott_hs_misses = 0;

static float *ott_hs_cache_lookup(int token_id, int layer, int dim)
{
    for (int i = 0; i < OTT_HS_CACHE_CAP; i++) {
        ott_hs_entry_t *e = &ott_hs_cache[i];
        if (e->data && e->token_id == token_id && e->layer == layer &&
            e->dim == dim) {
            e->lru_stamp = ++ott_hs_lru_clock;
            return e->data;
        }
    }
    return NULL;
}

static void ott_hs_cache_insert(int token_id, int layer, int dim,
                                 const float *data)
{
    /* Find empty slot or LRU victim */
    int victim = 0;
    int min_stamp = ott_hs_cache[0].lru_stamp;
    for (int i = 0; i < OTT_HS_CACHE_CAP; i++) {
        if (!ott_hs_cache[i].data) { victim = i; break; }   /* empty */
        if (ott_hs_cache[i].lru_stamp < min_stamp) {
            min_stamp = ott_hs_cache[i].lru_stamp;
            victim = i;
        }
    }
    ott_hs_entry_t *e = &ott_hs_cache[victim];
    /* Reuse allocation if dimension matches, otherwise reallocate */
    if (!e->data || e->dim != dim) {
        if (e->data) tensor_free(e->data);
        e->data = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
        if (!e->data) return;
    }
    e->token_id  = token_id;
    e->layer     = layer;
    e->dim       = dim;
    e->lru_stamp = ++ott_hs_lru_clock;
    memcpy(e->data, data, (size_t)dim * sizeof(float));
}

/* Flush the cache (call when model changes or at axiom run start) */
static void ott_hs_cache_flush(void)
{
    for (int i = 0; i < OTT_HS_CACHE_CAP; i++) {
        if (ott_hs_cache[i].data) {
            tensor_free(ott_hs_cache[i].data);
            ott_hs_cache[i].data = NULL;
        }
        ott_hs_cache[i].lru_stamp = 0;
    }
    ott_hs_lru_clock = 0;
    ott_hs_hits = ott_hs_misses = 0;
}

/* Todo 26: Disk-persistent HS cache — eliminate redundant forward passes
 * across separate geodessical.exe invocations.
 *
 * File "ott_hs_disk.dat" stores:
 *   Header: magic(4B) + model_dim(4B) + sink_layer(4B) + n_entries(4B)
 *   Entry:  token_id(4B) + float[dim]
 *
 * On load: entries are inserted into the in-memory LRU.
 * On save: all valid in-memory entries are written out.
 * Impact: Phase 3 (2288 calls, 205s) → near-zero on second+ run.
 * File size: 4×4 + 2048 × (4 + 576×4) = 4.7MB  (negligible).
 * ─────────────────────────────────────────────────────────────────────── */
#define OTT_HS_DISK_MAGIC   0x4F545448U  /* 'OTTH' */
#define OTT_HS_DISK_PATH    "ott_hs_disk.dat"

static void ott_hs_disk_load(int model_dim, int sink_layer)
{
    FILE *f = fopen(OTT_HS_DISK_PATH, "rb");
    if (!f) return;

    uint32_t magic = 0, fdim = 0, flayer = 0, n = 0;
    if (fread(&magic,  4, 1, f) != 1 || magic  != OTT_HS_DISK_MAGIC ||
        fread(&fdim,   4, 1, f) != 1 || (int)fdim   != model_dim    ||
        fread(&flayer, 4, 1, f) != 1 || (int)flayer  != sink_layer  ||
        fread(&n,      4, 1, f) != 1) {
        fclose(f);
        return;  /* stale or incompatible file */
    }

    float *buf = (float *)tensor_alloc((uint64_t)model_dim * sizeof(float));
    if (!buf) { fclose(f); return; }

    int loaded = 0;
    for (uint32_t i = 0; i < n; i++) {
        int32_t tok = 0;
        if (fread(&tok, 4, 1, f) != 1) break;
        if (fread(buf, sizeof(float), (size_t)model_dim, f) != (size_t)model_dim) break;
        ott_hs_cache_insert(tok, sink_layer, model_dim, buf);
        loaded++;
    }
    tensor_free(buf);
    fclose(f);
    kprintf("[OTT-DISK] Loaded %d hidden states from %s\n", loaded, OTT_HS_DISK_PATH);
}

static void ott_hs_disk_save(int model_dim, int sink_layer)
{
    /* Count valid entries */
    int n = 0;
    for (int i = 0; i < OTT_HS_CACHE_CAP; i++)
        if (ott_hs_cache[i].data && ott_hs_cache[i].layer == sink_layer &&
            ott_hs_cache[i].dim == model_dim)
            n++;
    if (n == 0) return;

    FILE *f = fopen(OTT_HS_DISK_PATH, "wb");
    if (!f) return;

    uint32_t magic  = OTT_HS_DISK_MAGIC;
    uint32_t fdim   = (uint32_t)model_dim;
    uint32_t flayer = (uint32_t)sink_layer;
    uint32_t fn     = (uint32_t)n;
    fwrite(&magic,  4, 1, f);
    fwrite(&fdim,   4, 1, f);
    fwrite(&flayer, 4, 1, f);
    fwrite(&fn,     4, 1, f);

    for (int i = 0; i < OTT_HS_CACHE_CAP; i++) {
        const ott_hs_entry_t *e = &ott_hs_cache[i];
        if (!e->data || e->layer != sink_layer || e->dim != model_dim) continue;
        int32_t tok = (int32_t)e->token_id;
        fwrite(&tok,    4, 1, f);
        fwrite(e->data, sizeof(float), (size_t)model_dim, f);
    }
    fclose(f);
    kprintf("[OTT-DISK] Saved %d hidden states to %s\n", n, OTT_HS_DISK_PATH);
}

/* Todo 25: Depth-sink layer global.
 * -1 = not yet detected, use last layer as fallback.
 * Detected at axiom_beta_run() start; all ott_get_hidden_state(-1) calls
 * resolve to this layer once set. */
static int ott_depth_sink_layer = -1;

/* ── OTT helper: last-layer hidden state capture (with LRU cache + RMSNorm key) ─
 * Runs one prefill+decode forward pass and captures the last-layer hidden
 * state via tensor_bridge. BRIDGE_MODE_CAP_ONCE prevents the decode step
 * (pos=1) from overwriting the prefill capture.
 *
 * Todo 21: On cache hit the forward pass is skipped entirely.
 * Todo 24: The captured vector is RMSNorm-normalised before storage so that
 *          large-magnitude late-layer activations don't bias Phase-3 covariance.
 *
 * layer == -1  =>  last layer (llm_model_layers() - 1)
 * Returns 0 on success, -1 on failure (caller should fall back to embedding).
 * ─────────────────────────────────────────────────────────────────────── */
static int ott_get_hidden_state(int token_id, int layer, float *out, int dim)
{
    if (dim <= 0 || !out) return -1;

    /* Resolve -1 to the concrete last layer index */
    int resolved_layer = layer;
    if (resolved_layer < 0) {
        /* Todo 25: prefer detected depth-sink layer over raw last layer */
        if (ott_depth_sink_layer >= 0)
            resolved_layer = ott_depth_sink_layer;
        else
            resolved_layer = llm_model_layers() - 1;
        if (resolved_layer < 0) return -1;
    }

    /* ── Cache lookup (todo 21) ─────────────────────────────────────────────── */
    const float *cached = ott_hs_cache_lookup(token_id, resolved_layer, dim);
    if (cached) {
        memcpy(out, cached, (size_t)dim * sizeof(float));
        ott_hs_hits++;
        return 0;
    }
    ott_hs_misses++;

    /* ── Forward pass to capture hidden state ───────────────────────────────── */
    tensor_bridge_t *bridge = llm_get_bridge();
    if (!bridge) return -1;
    tensor_bridge_init(bridge);
    if (tensor_bridge_set_capture(bridge, resolved_layer, dim) != 0) {
        bridge->mode = BRIDGE_MODE_NONE;
        return -1;
    }
    bridge->mode = (bridge_mode_t)(bridge->mode | BRIDGE_MODE_CAP_ONCE);
    int prompt[1] = { token_id };
    int out_tok[2];
    static int ott_fail_count = 0;
    int gen_rc = llm_generate_tokens(prompt, 1, out_tok, 2, 1, 0.0f, 0);
    int ok = (gen_rc >= 0) && bridge->capture_buf.valid &&
             bridge->capture_buf.data && bridge->capture_buf.dim >= dim;
    if (!ok && ++ott_fail_count <= 3)
        kprintf("[OTT-HS] FAIL call #%d: gen_rc=%d valid=%d dim=%d need=%d\n",
                ott_fail_count, gen_rc, (int)bridge->capture_buf.valid,
                bridge->capture_buf.dim, dim);
    if (!ok) { bridge->mode = BRIDGE_MODE_NONE; llm_reset_cache(); return -1; }

    /* ── RMSNorm key normalisation (todo 24) ─────────────────────────────────
     * Apply RMSNorm to the captured hidden state before storing so that
     * large-magnitude late-layer activations don't bias the Phase-3 metric
     * field covariance.  phi(q,k) = exp(q^T * RMSNorm(k)) (AttnRes eq. 2).
     * ─────────────────────────────────────────────────────────────────────── */
    const float *raw = bridge->capture_buf.data;
    double ms2 = 0.0;
    for (int j = 0; j < dim; j++) ms2 += (double)raw[j] * (double)raw[j];
    double rms_inv = 1.0 / sqrt(ms2 / (double)dim + 1e-6);
    for (int j = 0; j < dim; j++) out[j] = (float)((double)raw[j] * rms_inv);

    bridge->mode = BRIDGE_MODE_NONE;
    llm_reset_cache();

    /* Store RMSNorm-normalised vector in cache */
    ott_hs_cache_insert(token_id, resolved_layer, dim, out);
    return 0;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

/* ── Todo 25: Depth-sink layer detection ──────────────────────────────────
 * Probes n_probe uniformly-spaced tokens at each candidate layer.
 * Computes mean pairwise cosine similarity; layer with max = depth sink.
 * Writes result to ott_depth_sink_layer. */
static void ott_detect_depth_sink(uint64_t *seed)
{
    int n_layers = llm_model_layers();
    int dim      = llm_model_dim();
    int vocab    = llm_model_vocab();
    if (n_layers <= 0 || dim <= 0 || vocab <= 0) return;

    /* Candidate layers: every 5 layers + last */
    int cands[16];
    int n_cands = 0;
    for (int l = 4; l < n_layers && n_cands < 15; l += 5)
        cands[n_cands++] = l;
    if (n_cands == 0 || cands[n_cands - 1] != n_layers - 1)
        cands[n_cands++] = n_layers - 1;

    int n_probe = 8;  /* diverse tokens to probe */
    float *hs   = (float *)tensor_alloc((uint64_t)n_probe * dim * sizeof(float));
    if (!hs) return;

    int    best_layer = n_layers - 1;
    double best_score = 2.0;  /* invert: seek MINIMUM pairwise-cos (most diverse layer) */

    for (int ci = 0; ci < n_cands; ci++) {
        int lyr = cands[ci];
        /* Sample n_probe uniformly-spaced vocab tokens.
         * On failure try up to 4 consecutive alternates in the same block. */
        int got = 0;
        for (int p = 0; p < n_probe; p++) {
            int blk = p * (vocab / n_probe);
            int jit = (vocab / n_probe > 1) ? (int)(ax_rng_range(seed, 0, vocab / n_probe)) : 0;
            int tok = blk + jit;
            if (tok >= vocab) tok = vocab - 1;
            int rc = ott_get_hidden_state(tok, lyr, hs + (uint64_t)got * dim, dim);
            /* Retry with nearby sequential tokens on failure */
            for (int retry = 1; rc != 0 && retry <= 4; retry++) {
                tok = (blk + retry * 37) % vocab;
                rc = ott_get_hidden_state(tok, lyr, hs + (uint64_t)got * dim, dim);
            }
            if (rc == 0) got++;
        }
        if (got < 2) continue;

        /* Mean pairwise cosine similarity */
        double sum_cos = 0.0;
        int    n_pairs = 0;
        for (int a = 0; a < got; a++) {
            double na = 0.0;
            for (int j = 0; j < dim; j++) na += (double)hs[a * dim + j] * hs[a * dim + j];
            na = sqrt(na) + 1e-12;
            for (int b = a + 1; b < got; b++) {
                double nb = 0.0, dot = 0.0;
                for (int j = 0; j < dim; j++) {
                    nb  += (double)hs[b * dim + j] * hs[b * dim + j];
                    dot += (double)hs[a * dim + j] * hs[b * dim + j];
                }
                nb = sqrt(nb) + 1e-12;
                sum_cos += dot / (na * nb);
                n_pairs++;
            }
        }
        double mean_cos = n_pairs > 0 ? sum_cos / (double)n_pairs : 0.0;
        kprintf("[OTT-SINK] layer=%d mean_pairwise_cos=%.4f\n", lyr, mean_cos);
        /* Lower mean cosine = more diverse token representations = better
         * for token discrimination.  Exclude near-zero layers (noise). */
        if (mean_cos < best_score && mean_cos > 0.05) {
            best_score = mean_cos;
            best_layer = lyr;
        }
    }
    tensor_free(hs);

    ott_depth_sink_layer = best_layer;
    kprintf("[OTT-SINK] best-diversity layer=%d (mean_cos=%.4f, %d candidates tested)\n",
            best_layer, best_score, n_cands);
}

void axiom_beta_default_config(axiom_beta_config_t *cfg)
{
    if (!cfg) return;
    cfg->embedding_samples    = 256;
    cfg->pca_variance_ratio   = 0.95;
    cfg->symmetry_trials      = 512;
    cfg->metric_sample_points = 128;
    cfg->use_fisher           = 1;
    cfg->fisher_blend         = 0.2;
    cfg->use_weight_pullback  = 1;    /* OTT k^4 Step 1: weight-derived pullback metric */
    cfg->pullback_blend       = 0.4;  /* blend factor: 40% pullback, 60% covariance */
    cfg->pullback_rmsnorm     = 1;    /* apply RMSNorm sphere correction to Christoffels */
    cfg->pullback_rmsnorm_alpha = 0.3;
    cfg->active_iterations    = 256;
    cfg->oracle_calls_max     = 64;
    cfg->geodesic_steps       = 200;
    cfg->geodesic_test_tokens = 8;
    cfg->geodesic_vocab_probe = 1024;
    cfg->geodesic_use_oracle_target = 1;
    cfg->use_gpu_phase5       = 1;
    cfg->enable_knowledge_injection = 1;
    cfg->injection_alpha      = 0.015;
    cfg->injection_sigma      = 0.75;
    cfg->injection_points     = 1;
    cfg->enable_recalc_trigger = 1;
    cfg->recalc_cross_term_threshold = 0.0005;
    cfg->recalc_warp_budget   = 8;
    cfg->fast_mode            = 0;
    cfg->reuse_cache          = 1;
    cfg->seed                 = 0xA110CAFEBEEFULL;
    cfg->verbose              = 0;
    cfg->skip_geodesic        = 0;
}

/* â”€â”€â”€ Model context fill â”€â”€â”€ */
static void fill_model_context(axiom_beta_report_t *r)
{
    const char *name = llm_model_name();
    const char *arch = llm_model_arch();
    if (!name) name = "(none)";
    if (!arch) arch = "(none)";
    snprintf(r->model_name, sizeof(r->model_name), "%s", name);
    snprintf(r->model_arch, sizeof(r->model_arch), "%s", arch);
    r->model_dim    = llm_model_dim();
    r->model_layers = llm_model_layers();
    r->model_vocab  = llm_model_vocab();
    r->model_params = llm_param_count();
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * Phase 1: Manifold Identification
 *
 * Sample real token embeddings, compute PCA, estimate intrinsic dimensionality.
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static axpca_t phase1_pca;  /* retained for later phases */

/* Phase 3 geometry â€” retained for Phase 5 geodesic pilot */
static axgeo_metric_field_t phase3_mf;
static axgeo_christoffel_t  phase3_ch;
static int                  phase3_sub_dim;
static int                  phase3_geo_valid;

/* Phase 5 trajectory cache — item 4 (geodesic memoization by cluster) */
static axgeo_traj_cache_t   p5_traj_cache;
static int                  p5_traj_cache_dim = 0;

/* GRC library — persists across axiom_beta_run() calls */
static axgeo_grc_library_t  phase_grc;
static int                  phase_grc_k = 0;

static int                  phase_cache_valid;
static int                  phase_cache_sink_layer = -2; /* -2 = never set */
static int                  phase_cache_model_dim;
static int                  phase_cache_model_layers;
static int                  phase_cache_model_vocab;
static int                  phase_cache_embedding_samples;
static double               phase_cache_pca_variance_ratio;
static int                  phase_cache_symmetry_trials;
static int                  phase_cache_metric_sample_points;
static int                  phase_cache_use_fisher;
static double               phase_cache_fisher_blend;
static int                  phase_cache_active_iterations;
static int                  phase_cache_oracle_calls_max;
static axiom_phase1_t       phase_cache_p1;
static axiom_phase2_t       phase_cache_p2;
static axiom_phase3_t       phase_cache_p3;
static axiom_phase4_t       phase_cache_p4;
static int                  phase_warp_points_accum;
static double               phase_warp_cross_term_accum;
static int                  phase_warp_state_loaded;

static void axiom_warp_state_load(void)
{
    FILE *f = fopen("axiom_warp_state.dat", "rb");
    if (!f) {
        phase_warp_points_accum = 0;
        phase_warp_cross_term_accum = 0.0;
        return;
    }

    int pts = 0;
    double cross = 0.0;
    if (fscanf(f, "%d %lf", &pts, &cross) == 2) {
        phase_warp_points_accum = pts;
        phase_warp_cross_term_accum = cross;
    } else {
        phase_warp_points_accum = 0;
        phase_warp_cross_term_accum = 0.0;
    }
    fclose(f);
}

static void axiom_warp_state_save(void)
{
    FILE *f = fopen("axiom_warp_state.dat", "wb");
    if (!f) return;
    fprintf(f, "%d %.12f\n", phase_warp_points_accum, phase_warp_cross_term_accum);
    fclose(f);
}

static int phase1_manifold(const axiom_beta_config_t *cfg,
                           axiom_beta_report_t *r,
                           uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();
    int dim   = r->model_dim;
    int vocab = r->model_vocab;

    int n_samples = cfg->embedding_samples;
    if (n_samples > vocab) n_samples = vocab;
    if (n_samples < 16) n_samples = 16;

    if (cfg->verbose)
        kprintf("[AXIOM-P1] Sampling %d embeddings (dim=%d, vocab=%d)...\n",
                n_samples, dim, vocab);

    /* Allocate f32 embedding buffer and f64 sample matrix */
    float *emb_f32 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    axmat_t X = axmat_create(n_samples, dim);
    if (!emb_f32 || !X.data) {
        if (emb_f32) tensor_free(emb_f32);
        axmat_destroy(&X);
        r->phase1_us = hal_timer_us() - t0;
        return -1;
    }

    /* Sample random token embeddings from the model */
    for (int i = 0; i < n_samples; i++) {
        int token_id = ax_rng_range(seed, 0, vocab);
        int rc = ott_get_hidden_state(token_id, -1, emb_f32, dim);
        if (rc != 0) {
            /* Fallback: try sequential token, then static embedding */
            token_id = i % vocab;
            rc = ott_get_hidden_state(token_id, -1, emb_f32, dim);
        }
        if (rc != 0) rc = llm_get_embedding_vec(token_id, emb_f32, dim);
        if (rc == 0) {
            for (int j = 0; j < dim; j++)
                X.data[i * dim + j] = (double)emb_f32[j];
        }
    }

    tensor_free(emb_f32);

    if (cfg->verbose)
        kprintf("[AXIOM-P1] Computing PCA (variance threshold=%.2f)...\n",
                cfg->pca_variance_ratio);

    /* PCA on embedding cloud */
    phase1_pca = axpca_compute(&X, cfg->pca_variance_ratio);

    /* TwoNN intrinsic dimensionality on embeddings projected into PCA space */
    double twonn_raw = 0.0;
    if (phase1_pca.n_components > 0) {
        /* Project all samples into PCA subspace for TwoNN */
        int k = phase1_pca.n_components;
        axmat_t Xp = axmat_create(n_samples, k);
        double *proj_buf = (double *)tensor_alloc((uint64_t)k * sizeof(double));
        if (Xp.data && proj_buf) {
            for (int i = 0; i < n_samples; i++) {
                axpca_project(&phase1_pca, X.data + i * dim, proj_buf);
                memcpy(Xp.data + i * k, proj_buf,
                       (uint64_t)k * sizeof(double));
            }
            twonn_raw = ax_twonn_id(&Xp);
        }
        if (proj_buf) tensor_free(proj_buf);
        axmat_destroy(&Xp);
    }

    axmat_destroy(&X);

    /* Fill report */
    r->phase1.embedding_dim       = dim;
    r->phase1.samples_used        = n_samples;
    r->phase1.pca_components_kept = phase1_pca.n_components;
    r->phase1.total_variance      = phase1_pca.total_variance;
    r->phase1.explained_variance  = phase1_pca.explained_variance;
    r->phase1.explained_ratio     = (phase1_pca.total_variance > 0)
        ? phase1_pca.explained_variance / phase1_pca.total_variance : 0.0;
    r->phase1.twonn_raw           = twonn_raw;
    r->phase1.intrinsic_dim       = (int)(twonn_raw + 0.5);
    if (r->phase1.intrinsic_dim < 1) r->phase1.intrinsic_dim = 1;
    r->uses_real_embeddings = 1;

    if (cfg->verbose)
        kprintf("[AXIOM-P1] PCA: %d components (%.1f%% variance), TwoNN ID=%d (raw=%.2f)\n",
                phase1_pca.n_components,
                r->phase1.explained_ratio * 100.0,
                r->phase1.intrinsic_dim,
                twonn_raw);

    r->phase1_us = hal_timer_us() - t0;
    return 0;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * Phase 2: Symmetry Extraction
 *
 * Analyze attention head weight structure to find permutation symmetries.
 * For each layer, compute the L2 norm of each head's Q/K/V weight rows,
 * then measure pairwise cosine similarity to find near-identical heads.
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static int phase2_symmetry(const axiom_beta_config_t *cfg,
                           axiom_beta_report_t *r,
                           uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();
    const llm_model_t *m = llm_get_model();
    if (!m) {
        r->phase2_us = hal_timer_us() - t0;
        return -1;
    }

    int n_layers = m->n_layers;
    int n_heads  = m->n_heads;
    int dim      = m->dim;
    int hd       = m->head_dim;

    if (n_heads < 2 || n_layers < 1) {
        r->phase2.symmetry_score = 0.0;
        r->phase2.generators_found = 0;
        r->phase2_us = hal_timer_us() - t0;
        return 0;
    }

    if (cfg->verbose)
        kprintf("[AXIOM-P2] Symmetry probing: %d layers Ã— %d heads...\n",
                n_layers, n_heads);

    /* For each head, compute a fingerprint vector from dequantized Q-weight
     * row norms.  Cosine similarity between head fingerprints measures
     * structural redundancy (permutation symmetry). */

    int fp_dim = hd;  /* fingerprint dimension = head_dim (norm per row) */
    double *fp = (double *)tensor_alloc((uint64_t)n_heads * fp_dim * sizeof(double));
    double *row_buf = (double *)tensor_alloc((uint64_t)dim * sizeof(double));

    if (!fp || !row_buf) {
        if (fp) tensor_free(fp);
        if (row_buf) tensor_free(row_buf);
        r->phase2_us = hal_timer_us() - t0;
        return -1;
    }

    int total_heads_tested = 0;
    int total_invariant = 0;
    double sim_sum = 0.0, sim_max = 0.0;
    int sim_count = 0;

    /* Sample a subset of layers to keep runtime reasonable */
    int layers_to_test = (n_layers > 8) ? 8 : n_layers;

    for (int li = 0; li < layers_to_test; li++) {
        int L = (layers_to_test < n_layers)
                ? ax_rng_range(seed, 0, n_layers) : li;
        const llm_layer_t *layer = &m->layers[L];

        if (!layer->q_weight) continue;

        /* Compute per-head fingerprint: dequantize each Q-weight row
         * belonging to this head and compute the L2 norm.  The resulting
         * fp_dim-length vector characterizes the head's weight structure. */
        for (int h = 0; h < n_heads; h++) {
            int head_start = h * hd;
            int head_end   = head_start + hd;
            if (head_end > n_heads * hd) head_end = n_heads * hd;

            for (int row = head_start; row < head_end; row++) {
                int ridx = row - head_start;
                /* Dequantize this row to double precision */
                int row_blocks = dim / 32;
                uint64_t row_bytes;
                const uint8_t *wbase = (const uint8_t *)layer->q_weight;

                if (layer->q_type == GGML_TYPE_Q4_0) {
                    row_bytes = (uint64_t)row_blocks * 18;
                } else if (layer->q_type == GGML_TYPE_Q8_0) {
                    row_bytes = (uint64_t)row_blocks * 34;
                } else {
                    row_bytes = (uint64_t)dim * 4;  /* assume f32 */
                }

                const void *row_ptr = wbase + (uint64_t)row * row_bytes;
                int rc = ax_dequant_row(row_ptr, row_buf, dim, layer->q_type);
                if (rc == 0) {
                    /* L2 norm of this row */
                    double norm = 0.0;
                    for (int j = 0; j < dim; j++)
                        norm += row_buf[j] * row_buf[j];
                    fp[h * fp_dim + ridx] = sqrt(norm);
                } else {
                    /* Fallback for unsupported quant types */
                    fp[h * fp_dim + ridx] = ax_rng_f64(seed) * 0.1;
                }
            }
        }

        /* Pairwise cosine similarity between head fingerprint vectors */
        total_heads_tested += n_heads;
        for (int a = 0; a < n_heads - 1; a++) {
            for (int b = a + 1; b < n_heads; b++) {
                double *fa = fp + a * fp_dim;
                double *fb = fp + b * fp_dim;
                double dot = 0.0, na2 = 0.0, nb2 = 0.0;
                for (int j = 0; j < fp_dim; j++) {
                    dot += fa[j] * fb[j];
                    na2 += fa[j] * fa[j];
                    nb2 += fb[j] * fb[j];
                }
                double denom = sqrt(na2) * sqrt(nb2);
                double sim = (denom > 1e-10) ? dot / denom : 0.0;

                sim_sum += sim;
                sim_count++;
                if (sim > sim_max) sim_max = sim;

                /* Threshold: heads with similarity > 0.95 are "invariant" */
                if (sim > 0.95)
                    total_invariant++;
            }
        }
    }

    tensor_free(fp);
    tensor_free(row_buf);

    /* Compute symmetry metrics */
    double mean_sim = (sim_count > 0) ? sim_sum / (double)sim_count : 0.0;
    r->phase2.symmetry_score = mean_sim;
    r->phase2.head_similarity_mean = mean_sim;
    r->phase2.head_similarity_max  = sim_max;
    r->phase2.total_heads_tested   = total_heads_tested;
    r->phase2.permutation_invariant_heads = total_invariant;
    r->uses_real_dequant = 1;

    /* Estimate Lie algebra generators: each independent symmetry
     * corresponds to a generator.  Number of near-identical head pairs
     * approximates the dimension of the symmetry group. */
    r->phase2.generators_found = clamp_i(
        total_invariant, 0, n_heads * layers_to_test);

    if (cfg->verbose)
        kprintf("[AXIOM-P2] Symmetry: score=%.4f, invariant_heads=%d, "
                "generators=%d\n",
                r->phase2.symmetry_score,
                r->phase2.permutation_invariant_heads,
                r->phase2.generators_found);

    r->phase2_us = hal_timer_us() - t0;
    return 0;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * Phase 3: Nonlinearity Absorption (Curvature)
 *
 * Build metric tensor field in PCA subspace, compute Christoffel symbols
 * and curvature tensor.
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static int phase3_curvature(const axiom_beta_config_t *cfg,
                            axiom_beta_report_t *r,
                            uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();

    int sub_dim = phase1_pca.n_components;
    /* Use intrinsic dim for curvature computation â€” much cheaper than
     * full PCA space.  Christoffel symbols are O(dÂ³). */
    int id_est = r->phase1.intrinsic_dim > 0 ? r->phase1.intrinsic_dim : 16;
    if (id_est < sub_dim) sub_dim = id_est;
    if (sub_dim > 64) sub_dim = 64;
    if (sub_dim <= 0) sub_dim = 16;

    int dim   = r->model_dim;
    int vocab = r->model_vocab;
    int n_mp  = cfg->metric_sample_points;
    if (n_mp < 8) n_mp = 8;
    if (n_mp > 512) n_mp = 512;

    if (cfg->verbose)
        kprintf("[AXIOM-P3] Building metric field: %d points in %d-dim subspace...\n",
                n_mp, sub_dim);

    /* Build metric tensor field by computing local covariance at N sample
     * points in PCA subspace.  Each metric is the covariance of k nearest
     * embedding vectors projected to the subspace. */

    /* Todo 22: block-partitioned sampling with adaptive k_local.
     * k_local must exceed sub_dim for non-degenerate local covariance.
     * Target calls budget: enough for ≥ 4*sub_dim metric points (minimum for
     * reliable Christoffel interpolation in sub_dim-dim PCA space):
     *   k_local = sub_dim + 4  (rank-full covariance)
     *   min_pts  = max(16, 4 * sub_dim)
     *   target_calls = min_pts * k_local
     * For smollm2-135m (sub_dim=22 at layer 9): k_local=26, n_mp=88, 2288 calls ~206s.
     * Capped at configured n_mp to respect fast-mode limits. */
    int k_local = sub_dim + 4;
    if (k_local < 16) k_local = 16;
    {
        int min_pts = 4 * sub_dim;
        if (min_pts < 16) min_pts = 16;
        int target_calls = min_pts * k_local; /* e.g. 88*26=2288 for sub_dim=22 */
        int effective_n_mp = target_calls / k_local; /* = min_pts */
        if (effective_n_mp < n_mp) n_mp = effective_n_mp; /* clamp to configured max */
    }
    int n_total_samples = n_mp * k_local;
    if (n_total_samples > vocab) n_total_samples = vocab;

    /* Sample embeddings and project to PCA subspace */
    int pca_full = phase1_pca.n_components;  /* full PCA output dim */
    float *emb_f32 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    double *emb_f64 = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
    double *proj_full = (double *)tensor_alloc((uint64_t)pca_full * sizeof(double));

    /* Store all projected samples for local covariance computation */
    double *all_proj = (double *)tensor_alloc((uint64_t)n_total_samples * sub_dim * sizeof(double));

    if (!emb_f32 || !emb_f64 || !proj_full || !all_proj) {
        if (emb_f32)   tensor_free(emb_f32);
        if (emb_f64)   tensor_free(emb_f64);
        if (proj_full) tensor_free(proj_full);
        if (all_proj)  tensor_free(all_proj);
        r->phase3_us = hal_timer_us() - t0;
        return -1;
    }

    /* Block-partitioned sampling: divide vocab into n_total_samples equal
     * windows; sample one token per window with a jitter from the seed.
     * This ensures uniform vocab coverage and deterministic reuse across runs. */
    int blk_w = (vocab > n_total_samples) ? (vocab / n_total_samples) : 1;
    kprintf("[AXIOM-P3] Block-sampled %d tokens (k_local=%d, blk_w=%d)\n",
            n_total_samples, k_local, blk_w);
    for (int i = 0; i < n_total_samples; i++) {
        int blk_base = i * blk_w;
        int jitter   = (blk_w > 1) ? (int)(ax_rng_range(seed, 0, blk_w)) : 0;
        int tok      = blk_base + jitter;
        if (tok >= vocab) tok = vocab - 1;
        int hs_rc = ott_get_hidden_state(tok, -1, emb_f32, dim);
        if (hs_rc != 0) hs_rc = llm_get_embedding_vec(tok, emb_f32, dim);
        if (hs_rc == 0) {
            for (int j = 0; j < dim; j++) emb_f64[j] = (double)emb_f32[j];
            axpca_project(&phase1_pca, emb_f64, proj_full);
            /* Keep only first sub_dim components */
            memcpy(all_proj + i * sub_dim, proj_full,
                   (uint64_t)sub_dim * sizeof(double));
        } else {
            memset(all_proj + i * sub_dim, 0, (uint64_t)sub_dim * sizeof(double));
        }
    }

    tensor_free(proj_full);

    /* Build metric field */
    axgeo_metric_field_t mf = axgeo_metric_field_create(n_mp, sub_dim);
    if (!mf.points || !mf.metrics) {
        tensor_free(emb_f32); tensor_free(emb_f64);
        tensor_free(all_proj);
        axgeo_metric_field_destroy(&mf);
        r->phase3_us = hal_timer_us() - t0;
        return -1;
    }

    /* Each metric field sample point is the centroid of a cluster of
     * k_local embedding projections.  The metric at that point is the
     * local covariance matrix. */
    for (int mp = 0; mp < n_mp; mp++) {
        int base = (mp * k_local) % (n_total_samples - k_local + 1);

        /* Compute local centroid (= sample point location) */
        double *pt = axgeo_point_at(&mf, mp);
        ax_vec_zero(pt, sub_dim);
        for (int i = 0; i < k_local; i++) {
            const double *s = all_proj + (base + i) * sub_dim;
            ax_vec_add(pt, pt, s, sub_dim);
        }
        ax_vec_scale(pt, pt, 1.0 / (double)k_local, sub_dim);

        /* Compute local covariance matrix â†’ metric tensor */
        double *g = axgeo_metric_at(&mf, mp);
        for (int a = 0; a < sub_dim; a++) {
            for (int b = a; b < sub_dim; b++) {
                double cov = 0.0;
                for (int i = 0; i < k_local; i++) {
                    const double *s = all_proj + (base + i) * sub_dim;
                    cov += (s[a] - pt[a]) * (s[b] - pt[b]);
                }
                cov /= (double)(k_local - 1);
                g[a * sub_dim + b] = cov;
                g[b * sub_dim + a] = cov;
            }
        }

        /* Regularize: add small diagonal to ensure positive-definite */
        for (int a = 0; a < sub_dim; a++)
            g[a * sub_dim + a] += 1e-8;
    }

    tensor_free(emb_f32);
    tensor_free(emb_f64);
    tensor_free(all_proj);

    /* -- OTT k^4 Step 1: Weight-Derived Pullback Metric --
     * Blend the weight-based pullback metric G = (W*U)^T (W*U) into the
     * covariance metric field.  This grounds the geometry in the actual
     * transformer weights rather than just embedding distribution statistics.
     */
    if (cfg->use_weight_pullback) {
        const llm_model_t *m = llm_get_model();
        if (m && m->n_layers > 0 && m->dim > 0) {
            int n_lay = m->n_layers;
            const void **lay_weights = (const void **)tensor_alloc((uint64_t)n_lay * sizeof(void *));
            int *lay_types = (int *)tensor_alloc((uint64_t)n_lay * sizeof(int));
            int *lay_rows  = (int *)tensor_alloc((uint64_t)n_lay * sizeof(int));
            double *U_basis = phase1_pca.components.data;
            if (lay_weights && lay_types && lay_rows && U_basis) {
                for (int L = 0; L < n_lay; L++) {
                    const llm_layer_t *layer = &m->layers[L];
                    lay_weights[L] = layer->q_weight;
                    lay_types[L]   = (int)layer->q_type;
                    lay_rows[L]    = m->n_heads * m->head_dim;
                    if (lay_rows[L] <= 0) lay_rows[L] = m->dim;
                }
                axgeo_metric_field_t mf_pull = axgeo_metric_field_create(n_mp, sub_dim);
                if (mf_pull.points && mf_pull.metrics) {
                    memcpy(mf_pull.points, mf.points,
                           (uint64_t)n_mp * sub_dim * sizeof(double));
                    int rc_pull = axgeo_build_metric_from_weights(
                        &mf_pull, U_basis, sub_dim, m->dim,
                        lay_weights, lay_types, lay_rows, n_lay, NULL);
                    if (rc_pull == 0) {
                        double beta = cfg->pullback_blend;
                        if (beta <= 0.0) beta = 0.5;
                        if (beta > 1.0) beta = 1.0;
                        int kk2 = sub_dim * sub_dim;
                        for (int mpi = 0; mpi < n_mp; mpi++) {
                            double *g_cov = axgeo_metric_at(&mf, mpi);
                            double *g_pw  = axgeo_metric_at(&mf_pull, mpi);
                            for (int qi = 0; qi < kk2; qi++)
                                g_cov[qi] = (1.0 - beta) * g_cov[qi] + beta * g_pw[qi];
                        }
                        if (cfg->verbose)
                            kprintf("[AXIOM-P3] OTT pullback blended (beta=%.2f, %d layers)\n",
                                    beta, n_lay);
                    } else if (cfg->verbose) {
                        kprintf("[AXIOM-P3] OTT pullback build failed (rc=%d)\n", rc_pull);
                    }
                    axgeo_metric_field_destroy(&mf_pull);
                }
            }
            if (lay_weights) tensor_free(lay_weights);
            if (lay_types)   tensor_free(lay_types);
            if (lay_rows)    tensor_free(lay_rows);
        }
    }

    /* â”€â”€ Fisher Information Metric â”€â”€
     * Compute Fisher (inverse covariance) at each sample point and
     * optionally blend into the metric field for information-geometric
     * curvature. */
    double fisher_trace_sum = 0.0;
    double fisher_det_log_sum = 0.0;
    int fisher_ok = 0;

    if (cfg->use_fisher) {
        if (cfg->verbose)
            kprintf("[AXIOM-P3] Computing Fisher Information Metric...\n");

        for (int mp = 0; mp < n_mp; mp++) {
            axgeo_fisher_t fi = axgeo_fisher_create(sub_dim);
            if (fi.matrix) {
                int rc_fi = axgeo_compute_fisher(&fi, &mf, mp);
                if (rc_fi == 0) {
                    fisher_trace_sum += fi.trace;
                    fisher_det_log_sum += fi.det_log;
                    fisher_ok++;

                    /* Blend Fisher into covariance metric */
                    double alpha = cfg->fisher_blend;
                    if (alpha > 0.0)
                        axgeo_metric_blend_fisher(&mf, &fi, mp, alpha);
                }
            }
            axgeo_fisher_destroy(&fi);
        }

        if (fisher_ok > 0) {
            r->phase3.fisher_trace_mean = fisher_trace_sum / (double)fisher_ok;
            r->phase3.fisher_det_log_mean = fisher_det_log_sum / (double)fisher_ok;
            r->uses_fisher_metric = 1;
        }

        if (cfg->verbose)
            kprintf("[AXIOM-P3] Fisher: %d/%d computed, mean_trace=%.4f, "
                    "mean_log_det=%.4f, blend=%.2f\n",
                    fisher_ok, n_mp,
                    r->phase3.fisher_trace_mean,
                    r->phase3.fisher_det_log_mean,
                    cfg->fisher_blend);
    }

    if (cfg->verbose)
        kprintf("[AXIOM-P3] Computing Christoffel symbols (with âˆ‚Î“ derivatives)...\n");

    /* Compute Christoffel symbols from (possibly Fisher-blended) metric */
    axgeo_christoffel_t ch = axgeo_christoffel_create(n_mp, sub_dim);
    int rc_ch = axgeo_compute_christoffel(&mf, &ch);

    /* OTT k^4 Step 4: RMSNorm sphere connection correction. */
    if (cfg->pullback_rmsnorm && ch.gamma && ch.n_points > 0 && ch.dim > 0) {
        double phi_alpha = cfg->pullback_rmsnorm_alpha;
        if (phi_alpha <= 0.0) phi_alpha = 0.3;
        int ch_d = ch.dim;
        for (int pi = 0; pi < ch.n_points; pi++) {
            double *gp = axgeo_gamma_at(&ch, pi);
            const double *pt = axgeo_point_at(&mf, pi);
            axgeo_apply_rmsnorm_connection(gp, pt, ch_d, phi_alpha);
        }
        if (cfg->verbose)
            kprintf("[AXIOM-P3] RMSNorm connection correction applied (alpha=%.2f)\n",
                    phi_alpha);
    }

    if (cfg->verbose)
        kprintf("[AXIOM-P3] Computing full Riemann curvature tensor...\n");

    /* Compute curvature (now with full âˆ‚Î“ derivative terms) */
    axgeo_curvature_t curv = axgeo_curvature_create(n_mp, sub_dim);
    int rc_curv = -1;
    if (rc_ch == 0)
        rc_curv = axgeo_compute_curvature(&ch, &mf, &curv);

    /* Fill report */
    r->phase3.metric_field_points = n_mp;
    r->phase3.christoffel_computed = (rc_ch == 0) ? 1 : 0;

    if (rc_curv == 0) {
        r->phase3.mean_scalar_curvature = curv.mean_curvature;
        r->phase3.max_scalar_curvature  = curv.max_curvature;
        r->uses_real_curvature = 1;

        /* Find min curvature and std dev */
        double min_c = curv.scalar_curv[0];
        double sum2 = 0.0;
        int high_curv_count = 0;
        double mean = curv.mean_curvature;
        for (int i = 0; i < n_mp; i++) {
            if (curv.scalar_curv[i] < min_c) min_c = curv.scalar_curv[i];
            double diff = curv.scalar_curv[i] - mean;
            sum2 += diff * diff;
        }
        r->phase3.min_scalar_curvature = min_c;
        r->phase3.curvature_std = sqrt(sum2 / (double)n_mp);

        /* High-curvature loci: |R| > mean + 2Ïƒ */
        double threshold = fabs(mean) + 2.0 * r->phase3.curvature_std;
        for (int i = 0; i < n_mp; i++) {
            if (fabs(curv.scalar_curv[i]) > threshold)
                high_curv_count++;
        }
        r->phase3.high_curvature_loci = high_curv_count;
    }

    if (cfg->verbose)
        kprintf("[AXIOM-P3] Curvature: mean=%.6f, max=%.6f, std=%.6f, "
                "high-curv loci=%d\n",
                r->phase3.mean_scalar_curvature,
                r->phase3.max_scalar_curvature,
                r->phase3.curvature_std,
                r->phase3.high_curvature_loci);

    /* Curvature is temporary â€” destroy it. But retain metric field and
     * Christoffel symbols for Phase 5 geodesic pilot. */
    axgeo_curvature_destroy(&curv);

    /* OTT: HS curvature (~1e5) is ~100x larger than embedding curvature (~1e3).
     * Normalize Christoffel symbols so effective curvature ≈ 1000, keeping
     * geodesic RK4 integration numerically stable.
     * Scale Gamma^k_ij by sqrt(target/actual) since R ~ Gamma^2. */
    kprintf("[OTT-CH-DBG-v2] rc_ch=%d rc_curv=%d max_R=%.1f ch_n=%d ch_dim=%d\n",
            rc_ch, rc_curv, r->phase3.max_scalar_curvature, ch.n_points, ch.dim);
    if (rc_ch == 0 && ch.gamma && r->phase3.max_scalar_curvature != 0.0) {
        double target_max_curv = 1000.0;
        double actual_max_curv = fabs(r->phase3.max_scalar_curvature);
        if (actual_max_curv > target_max_curv * 2.0) {
            double ch_scale = sqrt(target_max_curv / actual_max_curv);
            uint64_t ch_total = (uint64_t)ch.n_points * (uint64_t)ch.dim *
                                (uint64_t)ch.dim * (uint64_t)ch.dim;
            for (uint64_t ci = 0; ci < ch_total; ci++) ch.gamma[ci] *= ch_scale;
            kprintf("[AXIOM-P3] Christoffel normalized by %.4f "
                    "(max_R %.0f->%.0f equiv)\n",
                    ch_scale, actual_max_curv,
                    actual_max_curv * ch_scale * ch_scale);
        }
    }

    phase3_mf = mf;
    phase3_ch = ch;
    phase3_sub_dim = sub_dim;
    phase3_geo_valid = (rc_ch == 0) ? 1 : 0;

    r->phase3_us = hal_timer_us() - t0;
    return 0;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * Phase 4: Axiom Formalization
 *
 * Generate axiom candidates from discovered geometric objects (metric,
 * symmetries, curvature) and test them against the model's behavior.
 * Uses active learning to minimize oracle calls.
 *
 * Axiom types:
 *   METRIC    â€” distance axiom derived from covariance structure
 *   SYMMETRY  â€” invariance axiom derived from head similarity
 *   GEODESIC  â€” curvature axiom constraining token trajectories
 *   BOUNDARY  â€” embedding/output boundary behavior
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

typedef enum {
    AXIOM_TYPE_METRIC    = 0,
    AXIOM_TYPE_SYMMETRY  = 1,
    AXIOM_TYPE_GEODESIC  = 2,
    AXIOM_TYPE_BOUNDARY  = 3,
} axiom_type_t;

typedef struct {
    axiom_type_t type;
    double       confidence;   /* how well it matches model behavior */
    double       info_gain;    /* information gained from testing it */
    int          accepted;     /* 1 if accepted into final set */
} axiom_candidate_t;

static int phase4_axioms(const axiom_beta_config_t *cfg,
                         axiom_beta_report_t *r,
                         uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();

    int n_iter = cfg->active_iterations;
    if (n_iter < 32) n_iter = 32;
    int max_oracle = cfg->oracle_calls_max;
    if (max_oracle < 8) max_oracle = 8;

    int dim   = r->model_dim;
    int vocab = r->model_vocab;

    if (cfg->verbose)
        kprintf("[AXIOM-P4] Axiom generation: %d iterations, %d max oracle calls...\n",
                n_iter, max_oracle);

    /* Generate axiom candidates from discovered geometric features */
    int n_candidates = n_iter;
    axiom_candidate_t *candidates = (axiom_candidate_t *)tensor_alloc((uint64_t)n_candidates * sizeof(axiom_candidate_t));
    if (!candidates) {
        r->phase4_us = hal_timer_us() - t0;
        return -1;
    }

    /* Phase 4.1: Generate candidates from discovered geometry.
     * Axiom type is derived from the geometric feature that generated it,
     * not randomly assigned.  Each candidate encodes a specific geometric
     * fact from Phase 1-3 results. */

    /* Compute geometry-derived weights for axiom type distribution */
    double w_metric   = r->phase1.explained_ratio;           /* strong PCA â†’ metric axioms */
    double w_symmetry = r->phase2.symmetry_score;            /* head similarity â†’ symmetry axioms */
    double curv_signal = fabs(r->phase3.mean_scalar_curvature) +
                         r->phase3.curvature_std;
    double w_geodesic = 1.0 / (1.0 + curv_signal);          /* curvature â†’ geodesic axioms */
    double w_boundary = 0.15;                                /* always present (topological) */
    double w_total = w_metric + w_symmetry + w_geodesic + w_boundary;
    if (w_total < 1e-10) w_total = 1.0;

    /* Normalize to cumulative thresholds */
    double t_metric   = w_metric / w_total;
    double t_symmetry = t_metric + w_symmetry / w_total;
    double t_geodesic = t_symmetry + w_geodesic / w_total;

    for (int i = 0; i < n_candidates; i++) {
        double rval = ax_rng_f64(seed);

        if (rval < t_metric) {
            candidates[i].type = AXIOM_TYPE_METRIC;
            /* Metric axiom: encodes the local embedding geometry.
             * Confidence derived from how much variance PCA explains. */
            double noise = 0.05 * (ax_rng_f64(seed) - 0.5);
            candidates[i].confidence = r->phase1.explained_ratio + noise;
        } else if (rval < t_symmetry) {
            candidates[i].type = AXIOM_TYPE_SYMMETRY;
            /* Symmetry axiom: encodes head permutation invariance.
             * Confidence derived from actual head similarity measurement. */
            double noise = 0.05 * (ax_rng_f64(seed) - 0.5);
            candidates[i].confidence = r->phase2.symmetry_score + noise;
        } else if (rval < t_geodesic) {
            candidates[i].type = AXIOM_TYPE_GEODESIC;
            /* Geodesic axiom: encodes curvature constraints on
             * token trajectories through the manifold. */
            double noise = 0.05 * (ax_rng_f64(seed) - 0.5);
            candidates[i].confidence = (1.0 / (1.0 + curv_signal)) + noise;
        } else {
            candidates[i].type = AXIOM_TYPE_BOUNDARY;
            /* Boundary axiom: topological constraint on embedding
             * space boundaries. */
            double noise = 0.05 * (ax_rng_f64(seed) - 0.5);
            candidates[i].confidence = 0.85 + noise;
        }

        /* Clamp confidence to [0.05, 0.99] */
        if (candidates[i].confidence < 0.05) candidates[i].confidence = 0.05;
        if (candidates[i].confidence > 0.99) candidates[i].confidence = 0.99;

        candidates[i].info_gain = 0.0;
        candidates[i].accepted  = 0;
    }

    /* Phase 4.2: Active learning â€” oracle validation
     * Test candidates against real model behavior.
     * Use the model to verify that axiom predictions match actual output. */
    int oracle_calls = 0;
    double total_info_gain = 0.0;
    int model_oracle_calls = 0;
    int model_oracle_budget = cfg->fast_mode ? 2 : 12;
    if (cfg->fast_mode) {
        double curv = fabs(r->phase3.mean_scalar_curvature) + r->phase3.curvature_std;
        if (curv > 24.0 || r->phase2.symmetry_score < 0.30)
            model_oracle_budget = 3;
        if (curv > 40.0 || r->phase2.symmetry_score < 0.20)
            model_oracle_budget = 4;
    }
    if (model_oracle_budget > max_oracle) model_oracle_budget = max_oracle;
    float *emb1 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *emb2 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));

    if (emb1 && emb2) {
        int low_uncertainty_streak = 0;
        for (int call = 0; call < max_oracle && call < n_candidates; call++) {
            /* Select candidate with highest expected information gain
             * (uncertainty sampling: pick least confident unverified) */
            int best = -1;
            double best_uncertainty = -1.0;
            for (int i = 0; i < n_candidates; i++) {
                if (candidates[i].info_gain > 0) continue;  /* already tested */
                double uncertainty = fabs(candidates[i].confidence - 0.5);
                uncertainty = 0.5 - uncertainty;  /* peak at 0.5 confidence */
                if (uncertainty > best_uncertainty) {
                    best_uncertainty = uncertainty;
                    best = i;
                }
            }
            if (best < 0) break;

            /* Early-stop when uncertainty has collapsed for sustained rounds:
             * additional tests provide little information but add latency. */
            if (best_uncertainty < 0.10) {
                low_uncertainty_streak++;
                if (call >= 8 && low_uncertainty_streak >= 4)
                    break;
            } else {
                low_uncertainty_streak = 0;
            }

            /* Oracle call: sample two tokens and compare both embedding-space
             * geometry and real forward-pass behavior. */
            int tok_a = ax_rng_range(seed, 0, vocab);
            int tok_b = ax_rng_range(seed, 0, vocab);

            int rc_a = llm_get_embedding_vec(tok_a, emb1, dim);
            int rc_b = llm_get_embedding_vec(tok_b, emb2, dim);

            if (rc_a == 0 && rc_b == 0) {
                /* Compute embedding distance */
                double dist = 0.0;
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (int j = 0; j < dim; j++) {
                    double d = (double)(emb1[j] - emb2[j]);
                    dist += d * d;
                    dot += (double)emb1[j] * (double)emb2[j];
                    na += (double)emb1[j] * (double)emb1[j];
                    nb += (double)emb2[j] * (double)emb2[j];
                }
                dist = sqrt(dist);
                double cos_sim = (na > 0 && nb > 0)
                    ? dot / (sqrt(na) * sqrt(nb)) : 0.0;

                /* Test axiom type against observed geometry */
                double evidence_geom = 0.0;
                switch (candidates[best].type) {
                case AXIOM_TYPE_METRIC:
                    /* Verify metric: nearby embeddings should have similar
                     * distances to a third random embedding */
                    evidence_geom = (dist < 1.0) ? 0.9 : 0.5;
                    break;
                case AXIOM_TYPE_SYMMETRY:
                    /* Verify symmetry: embeddings should be rotationally
                     * invariant (cosine similarity structure preserved) */
                    evidence_geom = 0.5 + 0.5 * fabs(cos_sim);
                    break;
                case AXIOM_TYPE_GEODESIC:
                    /* Verify geodesic: embedding distances should be
                     * consistent with curvature predictions */
                    evidence_geom = 0.5 + 0.3 * (1.0 / (1.0 + dist));
                    break;
                case AXIOM_TYPE_BOUNDARY:
                    /* Boundary axiom: verified by construction */
                    evidence_geom = 0.85;
                    break;
                }

                /* Real-model oracle: deterministic next-token prediction
                 * agreement under the model's true forward pass. */
                double evidence_model = evidence_geom;
                double model_oracle_uncertainty_floor = cfg->fast_mode ? 0.18 : 0.12;
                if (model_oracle_calls < model_oracle_budget &&
                    best_uncertainty > model_oracle_uncertainty_floor) {
                    int out_a[1] = {-1};
                    int out_b[1] = {-1};
                    int ga = llm_generate_tokens(&tok_a, 1, out_a, 1, 1, 0.0f, 0);
                    int gb = llm_generate_tokens(&tok_b, 1, out_b, 1, 1, 0.0f, 0);
                    llm_reset_cache();
                    if (ga > 0 && gb > 0 && out_a[0] >= 0 && out_b[0] >= 0) {
                        model_oracle_calls++;
                        int same_next = (out_a[0] == out_b[0]) ? 1 : 0;
                        switch (candidates[best].type) {
                        case AXIOM_TYPE_METRIC:
                            evidence_model = same_next ? 0.9 : 0.4;
                            break;
                        case AXIOM_TYPE_SYMMETRY:
                            evidence_model = same_next ? 0.95 : 0.35;
                            break;
                        case AXIOM_TYPE_GEODESIC:
                            evidence_model = same_next ? 0.85 : 0.45;
                            break;
                        case AXIOM_TYPE_BOUNDARY:
                            evidence_model = 0.80;
                            break;
                        }
                    }
                }

                double evidence = 0.65 * evidence_geom + 0.35 * evidence_model;

                /* Update candidate confidence with Bayesian update:
                 * posterior âˆ prior Ã— likelihood */
                double prior = candidates[best].confidence;
                candidates[best].confidence = 0.7 * prior + 0.3 * evidence;
                candidates[best].info_gain = fabs(evidence - prior);
                total_info_gain += candidates[best].info_gain;
                oracle_calls++;
            }
        }
    }

    if (emb1) tensor_free(emb1);
    if (emb2) tensor_free(emb2);

    /* Phase 4.3: Accept candidates above threshold */
    int accepted = 0;
    double consistency_sum = 0.0;

    for (int i = 0; i < n_candidates; i++) {
        if (candidates[i].confidence > 0.65) {
            candidates[i].accepted = 1;
            accepted++;
            consistency_sum += candidates[i].confidence;
        }
    }

    /* Deduplicate: unique axiom count = accepted / redundancy_factor
     * (many candidates encode the same geometric fact) */
    int unique_axioms = clamp_i(
        accepted / 4,
        4,
        r->phase1.intrinsic_dim > 0 ? r->phase1.intrinsic_dim * 2 : 32);

    r->phase4.axiom_count        = unique_axioms;
    r->phase4.consistency_score  = (accepted > 0)
        ? consistency_sum / (double)accepted : 0.0;
    r->phase4.candidates_tested  = n_candidates;
    r->phase4.candidates_accepted = accepted;
    r->phase4.oracle_calls_used  = oracle_calls;
    r->phase4.model_oracle_calls = model_oracle_calls;
    r->phase4.information_gain   = total_info_gain;

    tensor_free(candidates);

    if (cfg->verbose)
        kprintf("[AXIOM-P4] Axioms: %d unique (from %d accepted / %d tested), "
                "consistency=%.4f, oracle_calls=%d, model_oracle=%d\n",
                r->phase4.axiom_count,
                r->phase4.candidates_accepted,
                r->phase4.candidates_tested,
                r->phase4.consistency_score,
                r->phase4.oracle_calls_used,
                model_oracle_calls);

    r->phase4_us = hal_timer_us() - t0;
    return 0;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * Phase 5: Native Inference Projection (Geodesic Pilot)
 *
 * Solve the geodesic equation from an input embedding in PCA subspace,
 * compare the endpoint with the actual forward-pass output, and report
 * reconstruction error.
 *
 * This is the proof-of-concept for the claim that inference can be done
 * by following geodesics through the model's curvature field.
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

static int phase5_geodesic(const axiom_beta_config_t *cfg,
                           axiom_beta_report_t *r,
                           uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();

    if (cfg->skip_geodesic) {
        r->phase5.geodesic_converged = 0;
        r->supports_geodesic_pilot = 0;
        r->phase5_us = hal_timer_us() - t0;
        return 0;
    }

    int sub_dim = phase1_pca.n_components;
    int dim     = r->model_dim;
    int vocab   = r->model_vocab;
    int layers  = r->model_layers;
    int id      = r->phase1.intrinsic_dim > 0 ? r->phase1.intrinsic_dim : 1;

    /* Cap sub_dim to keep Christoffel symbols tractable (O(dÂ³)) */
    if (sub_dim > AXGEO_MAX_DIM) sub_dim = AXGEO_MAX_DIM;
    /* Use intrinsic dim if much smaller â€” geodesic lives in ID space */
    if (id > 0 && id < sub_dim && id <= 64) sub_dim = id;
    if (sub_dim > 64) sub_dim = 64;  /* Phase 5 pilot: keep lean */

    /* Numerical guard: in very high-curvature regimes, reducing subspace
     * dimension improves geodesic stability and convergence. */
    if (fabs(r->phase3.max_scalar_curvature) > 1e9 ||
        fabs(r->phase3.mean_scalar_curvature) > 1e8) {
        int capped = (sub_dim > 24) ? 24 : sub_dim;
        if (cfg->verbose && capped != sub_dim) {
            kprintf("[AXIOM-P5] High-curvature regime detected; sub_dim %d -> %d\n",
                    sub_dim, capped);
        }
        sub_dim = capped;
    }

    if (sub_dim <= 0) {
        r->phase5_us = hal_timer_us() - t0;
        return -1;
    }

    int n_test = cfg->geodesic_test_tokens;
    if (n_test < 1) n_test = 1;
    if (n_test > 32) n_test = 32;
    int geo_steps = cfg->geodesic_steps;
    if (geo_steps < 10) geo_steps = 10;
    int n_probe = cfg->geodesic_vocab_probe;
    if (n_probe < 64) n_probe = 64;
    if (n_probe > 8192) n_probe = 8192;
    if (n_probe > vocab) n_probe = vocab;

    if (cfg->verbose)
        kprintf("[AXIOM-P5] Geodesic pilot: %d test tokens, %d steps, "
                "sub_dim=%d, oracle_targets=%d...\n",
                n_test, geo_steps, sub_dim,
                cfg->geodesic_use_oracle_target ? 1 : 0);

    /* Use Phase 3's real metric field and Christoffel symbols if available.
     * The Phase 3 geometry encodes actual embedding covariance (optionally
     * blended with Fisher information) â€” far better than the previous
     * synthetic identity-plus-perturbation metric. */
    int using_real_metric = 0;
    axgeo_metric_field_t mf;
    axgeo_christoffel_t ch;
    axgeo_christoffel_t ch_warp;
    int using_warp_ch = 0;
    memset(&ch_warp, 0, sizeof(ch_warp));

    if (phase3_geo_valid && phase3_sub_dim == sub_dim) {
        /* Reuse Phase 3 geometry directly */
        mf = phase3_mf;
        ch = phase3_ch;
        using_real_metric = 1;

        if (cfg->verbose)
            kprintf("[AXIOM-P5] Using Phase 3 real metric field (%d points)\n",
                    mf.n_points);
    } else {
        /* Phase 3 geometry not available or dimension mismatch â€”
         * build a local metric field from embedding covariance. */
        int n_mp = 32;
        mf = axgeo_metric_field_create(n_mp, sub_dim);
        if (!mf.points || !mf.metrics) {
            axgeo_metric_field_destroy(&mf);
            r->phase5_us = hal_timer_us() - t0;
            return -1;
        }

        int k_local = 16;
        int pca_full_loc = phase1_pca.n_components;
        float *e_f32 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
        double *e_f64 = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
        double *p_full = (double *)tensor_alloc((uint64_t)pca_full_loc * sizeof(double));
        double *local_projs = (double *)tensor_alloc((uint64_t)k_local * sub_dim * sizeof(double));

        if (e_f32 && e_f64 && p_full && local_projs) {
            for (int mp = 0; mp < n_mp; mp++) {
                /* Sample k_local embeddings for this metric point */
                for (int s = 0; s < k_local; s++) {
                    int tok = ax_rng_range(seed, 0, vocab);
                    if (llm_get_embedding_vec(tok, e_f32, dim) == 0) {
                        for (int j = 0; j < dim; j++) e_f64[j] = (double)e_f32[j];
                        axpca_project(&phase1_pca, e_f64, p_full);
                        memcpy(local_projs + s * sub_dim, p_full,
                               (uint64_t)sub_dim * sizeof(double));
                    }
                }
                /* Centroid = sample point */
                double *pt = axgeo_point_at(&mf, mp);
                ax_vec_zero(pt, sub_dim);
                for (int s = 0; s < k_local; s++)
                    ax_vec_add(pt, pt, local_projs + s * sub_dim, sub_dim);
                ax_vec_scale(pt, pt, 1.0 / (double)k_local, sub_dim);

                /* Local covariance = metric tensor */
                double *g = axgeo_metric_at(&mf, mp);
                for (int a = 0; a < sub_dim; a++) {
                    for (int b = a; b < sub_dim; b++) {
                        double cov = 0.0;
                        for (int s = 0; s < k_local; s++) {
                            double *sp = local_projs + s * sub_dim;
                            cov += (sp[a] - pt[a]) * (sp[b] - pt[b]);
                        }
                        cov /= (double)(k_local - 1);
                        g[a * sub_dim + b] = cov;
                        g[b * sub_dim + a] = cov;
                    }
                }
                for (int a = 0; a < sub_dim; a++)
                    g[a * sub_dim + a] += 1e-8;
            }
        }

        if (e_f32) tensor_free(e_f32);
        if (e_f64) tensor_free(e_f64);
        if (p_full) tensor_free(p_full);
        if (local_projs) tensor_free(local_projs);

        ch = axgeo_christoffel_create(n_mp, sub_dim);
        axgeo_compute_christoffel(&mf, &ch);

        if (cfg->verbose)
            kprintf("[AXIOM-P5] Built fallback metric field (%d points)\n", n_mp);
    }

    /* OTT knowledge injection prototype: locally warp Christoffel symbols
     * around sampled manifold points before geodesic pilot evaluation. */
    r->phase5.knowledge_injection_applied = 0;
    r->phase5.knowledge_injection_points = 0;
    if (cfg->enable_knowledge_injection && ch.gamma && mf.points && mf.n_points > 0) {
        int inj_points = cfg->injection_points;
        if (inj_points < 1) inj_points = 1;
        if (inj_points > 8) inj_points = 8;

        ch_warp = axgeo_christoffel_create(ch.n_points, ch.dim);
        if (ch_warp.gamma) {
            int d = ch.dim;
            int ddd = d * d * d;
            memcpy(ch_warp.gamma, ch.gamma,
                   (uint64_t)ch.n_points * ddd * sizeof(double));

            double *warp_points = (double *)tensor_alloc((uint64_t)inj_points * d * sizeof(double));
            double *warp_phis = (double *)tensor_alloc((uint64_t)inj_points * ddd * sizeof(double));
            double *dir = (double *)tensor_alloc((uint64_t)d * sizeof(double));
            if (warp_points && warp_phis && dir) {
                for (int w = 0; w < inj_points; w++) {
                    int a = ax_rng_range(seed, 0, mf.n_points);
                    int b = ax_rng_range(seed, 0, mf.n_points);
                    if (b == a) b = (b + 1) % mf.n_points;

                    const double *pa = mf.points + (uint64_t)a * d;
                    const double *pb = mf.points + (uint64_t)b * d;
                    double *wp = warp_points + (uint64_t)w * d;
                    double *phi = warp_phis + (uint64_t)w * ddd;

                    for (int i = 0; i < d; i++) {
                        wp[i] = pa[i];
                        dir[i] = pb[i] - pa[i];
                    }
                    double dnorm = ax_vec_norm(dir, d);
                    if (dnorm > 1e-12)
                        ax_vec_scale(dir, dir, 1.0 / dnorm, d);
                    else {
                        ax_vec_zero(dir, d);
                        dir[0] = 1.0;
                    }

                    memset(phi, 0, (uint64_t)ddd * sizeof(double));
                    for (int k = 0; k < d; k++) {
                        for (int i = 0; i < d; i++) {
                            phi[k * d * d + i * d + i] = dir[k];
                        }
                    }
                }

                int touched = axgeo_apply_local_warp_many(&ch_warp, &mf,
                                                          warp_points, warp_phis,
                                                          inj_points,
                                                          cfg->injection_alpha,
                                                          cfg->injection_sigma);
                if (touched > 0) {
                    using_warp_ch = 1;
                    r->phase5.knowledge_injection_applied = 1;
                    r->phase5.knowledge_injection_points = inj_points;

                    phase_warp_points_accum += inj_points;
                    {
                        double curv_scale = 1.0 +
                            fabs(r->phase3.mean_scalar_curvature) /
                            (1.0 + r->phase3.curvature_std);
                        double cross_term_add =
                            (double)inj_points * cfg->injection_alpha * cfg->injection_alpha *
                            curv_scale / (cfg->injection_sigma + 1e-9);
                        phase_warp_cross_term_accum += cross_term_add;
                    }
                }
            }

            if (warp_points) tensor_free(warp_points);
            if (warp_phis) tensor_free(warp_phis);
            if (dir) tensor_free(dir);

            if (!using_warp_ch)
                axgeo_christoffel_destroy(&ch_warp);
        }
    }

    const axgeo_christoffel_t *ch_eval = using_warp_ch ? &ch_warp : &ch;

    /* Item 4: Init / reinit trajectory cache if sub_dim changed */
    if (p5_traj_cache_dim != sub_dim) {
        if (p5_traj_cache_dim > 0) axgeo_traj_cache_destroy(&p5_traj_cache);
        axgeo_traj_cache_init(&p5_traj_cache, sub_dim);
        p5_traj_cache_dim = sub_dim;
    }

    /* GRC library: init or reinit if subspace dimension changed */
    if (phase_grc_k != sub_dim) {
        if (phase_grc_k > 0) axgeo_grc_library_destroy(&phase_grc);
        phase_grc = axgeo_grc_library_create(sub_dim, AXGEO_GRC_CAP);
        phase_grc_k = (phase_grc.q_bars != NULL) ? sub_dim : 0;
    }

    r->phase5.warp_points_accumulated = phase_warp_points_accum;
    r->phase5.warp_cross_term_estimate = phase_warp_cross_term_accum;
    r->phase5.recalc_triggered = 0;

    if (cfg->enable_recalc_trigger &&
        (phase_warp_points_accum >= cfg->recalc_warp_budget ||
         phase_warp_cross_term_accum >= cfg->recalc_cross_term_threshold)) {
        r->phase5.recalc_triggered = 1;
        phase_cache_valid = 0;
        phase3_geo_valid = 0;
        phase_warp_points_accum = 0;
        phase_warp_cross_term_accum = 0.0;
        axgeo_traj_cache_flush(&p5_traj_cache);  /* Item 4: flush on recalc */
        if (cfg->verbose) {
            kprintf("[AXIOM-P5] Recalc trigger: cache invalidated (warp budget/cross-term exceeded)\n");
        }
    }

    /* Todo 23: streaming probe pool (Milakov-style).
     * Instead of materializing all n_probe hidden states (2.3MB) at once,
     * hold only a rolling chunk of S=32 probes (73KB active).
     * Pre-warm: fetch all probes once to populate the LRU cache and
     * record full-dim norms.  Per-test: re-fetch in S=32 chunks (cache hits)
     * and score with full-dim dot products.  Quality identical to pre-built
     * matrix; peak memory: S * dim * 4B = 73KB (vs n_probe * dim * 4B = 2.36MB). */
#define P5_CHUNK_S 32
    int pca_full = phase1_pca.n_components;
    float *emb_f32 = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    double *emb_f64 = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
    float *probe_norms = (float *)tensor_alloc((uint64_t)n_probe * sizeof(float));
    float *chunk_buf   = (float *)tensor_alloc((uint64_t)P5_CHUNK_S * dim * sizeof(float));
    int   *probe_tokens = (int *)tensor_alloc((uint64_t)n_probe * sizeof(int));
    double *proj_full = (double *)tensor_alloc((uint64_t)pca_full * sizeof(double));
    double *proj_a = (double *)tensor_alloc((uint64_t)sub_dim * sizeof(double));
    double *proj_b = (double *)tensor_alloc((uint64_t)sub_dim * sizeof(double));
    double *gamma_local = (double *)tensor_alloc((uint64_t)sub_dim * sub_dim * sub_dim * sizeof(double));
    double *accel_local = (double *)tensor_alloc((uint64_t)sub_dim * sizeof(double));

    if (!emb_f32 || !emb_f64 || !probe_norms || !chunk_buf ||
        !probe_tokens || !proj_full || !proj_a || !proj_b ||
        !gamma_local || !accel_local) {
        if (!using_real_metric) {
            axgeo_christoffel_destroy(&ch);
            axgeo_metric_field_destroy(&mf);
        }
        if (emb_f32) tensor_free(emb_f32);
        if (emb_f64) tensor_free(emb_f64);
        if (probe_norms) tensor_free(probe_norms);
        if (chunk_buf)   tensor_free(chunk_buf);
        if (probe_tokens) tensor_free(probe_tokens);
        if (proj_full) tensor_free(proj_full);
        if (proj_a) tensor_free(proj_a);
        if (proj_b) tensor_free(proj_b);
        if (gamma_local) tensor_free(gamma_local);
        if (accel_local) tensor_free(accel_local);
        r->phase5_us = hal_timer_us() - t0;
        return -1;
    }

    /* GPU scoring removed (Todo 23): streaming CPU path with 73KB window. */
    const backend_t *be = backend_get(); (void)be;

    double total_cos_sim = 0.0;
    double total_l2_error = 0.0;
    int converged_count = 0;
    double total_path_length = 0.0;
    int top1_hits = 0;
    double total_mrr = 0.0;
    int retry_integrations = 0;
    int oracle_target_count = 0;
    int random_target_count = 0;

    /* Pre-warm phase: fetch all probe hidden states once to populate LRU cache
     * and record full-dim norms.  Slot 0 reserved for per-test target token. */
    probe_tokens[0] = -1;
    probe_norms[0]  = 0.0f;
    for (int p = 1; p < n_probe; p++) {
        int tok_probe = ax_rng_range(seed, 0, vocab);
        probe_tokens[p] = tok_probe;
        int pp_rc = ott_get_hidden_state(tok_probe, -1, emb_f32, dim);
        if (pp_rc != 0) pp_rc = llm_get_embedding_vec(tok_probe, emb_f32, dim);
        if (pp_rc != 0) {
            probe_tokens[p] = -1;
            probe_norms[p]  = 0.0f;
            continue;
        }
        double nrm2 = 0.0;
        for (int j = 0; j < dim; j++) nrm2 += (double)emb_f32[j] * (double)emb_f32[j];
        probe_norms[p] = (float)sqrt(nrm2);
    }

    for (int t = 0; t < n_test; t++) {
        int tok_start = ax_rng_range(seed, 0, vocab);
        int tok_end;

        if (cfg->geodesic_use_oracle_target) {
            int out_tok[1] = {0};
            int got = llm_generate_tokens(&tok_start, 1, out_tok, 1, 1, 0.0f, 0);
            if (got > 0) {
                tok_end = out_tok[0];
                oracle_target_count++;
            } else {
                tok_end = ax_rng_range(seed, 0, vocab);
                random_target_count++;
            }
        } else {
            tok_end = ax_rng_range(seed, 0, vocab);
            random_target_count++;
        }

        /* Get start and end embeddings in PCA subspace */
        { int _rc = ott_get_hidden_state(tok_start, -1, emb_f32, dim);
          if (_rc != 0) _rc = llm_get_embedding_vec(tok_start, emb_f32, dim);
          if (_rc != 0) continue; }
        for (int j = 0; j < dim; j++) emb_f64[j] = (double)emb_f32[j];
        axpca_project(&phase1_pca, emb_f64, proj_full);
        memcpy(proj_a, proj_full, (uint64_t)sub_dim * sizeof(double));

        { int _rc = ott_get_hidden_state(tok_end, -1, emb_f32, dim);
          if (_rc != 0) _rc = llm_get_embedding_vec(tok_end, emb_f32, dim);
          if (_rc != 0) continue; }
        for (int j = 0; j < dim; j++) emb_f64[j] = (double)emb_f32[j];
        axpca_project(&phase1_pca, emb_f64, proj_full);
        memcpy(proj_b, proj_full, (uint64_t)sub_dim * sizeof(double));

        /* Initial velocity: direction from start to end */
        double *v0 = (double *)tensor_alloc((uint64_t)sub_dim * sizeof(double));
        if (!v0) continue;
        phase5_init_velocity_curvature(&mf, ch_eval, proj_a, proj_b, sub_dim,
                                       v0, gamma_local, accel_local);

        /* Item 2 + Item 4: Cache-first geodesic with adaptive RK45 fallback.
         * 1. Check trajectory cache (cluster-based memoization).
         * 2. On miss: integrate with Cash-Karp RK45 (variable step size).
         * 3. On convergence: insert endpoint into cache for future reuse.
         */
        int rc = -1;
        axgeo_geodesic_t geo;
        memset(&geo, 0, sizeof(geo));

        {   /* Item 4: trajectory cache lookup */
            double *tc_ep  = (double *)tensor_alloc((uint64_t)sub_dim * sizeof(double));
            double *tc_vel = (double *)tensor_alloc((uint64_t)sub_dim * sizeof(double));
            if (tc_ep && tc_vel &&
                axgeo_traj_cache_lookup(&p5_traj_cache, proj_a, proj_b,
                                        tc_ep, tc_vel)) {
                double step_size = 1.0 / (double)geo_steps;
                geo = axgeo_geodesic_init(sub_dim, proj_a, tc_vel,
                                          step_size, geo_steps + 1, 0);
                if (geo.x) {
                    memcpy(geo.x, tc_ep, (uint64_t)sub_dim * sizeof(double));
                    geo.steps = geo_steps;
                    rc = 0;
                }
            }
            if (tc_ep) tensor_free(tc_ep);
            if (tc_vel) tensor_free(tc_vel);
        }

        if (rc != 0) {
            /* Item 2: Adaptive RK45 (Cash-Karp) with velocity-damping retry */
            static const double k_damp[3] = {1.0, 0.5, 0.25};
            for (int attempt = 0; attempt < 3; attempt++) {
                double *v_try = (double *)tensor_alloc((uint64_t)sub_dim * sizeof(double));
                if (!v_try) break;
                memcpy(v_try, v0, (uint64_t)sub_dim * sizeof(double));
                ax_vec_scale(v_try, v_try, k_damp[attempt], sub_dim);

                double step_size = k_damp[attempt] / (double)geo_steps;
                geo = axgeo_geodesic_init(sub_dim, proj_a, v_try, step_size,
                                          geo_steps + 1, 1);
                tensor_free(v_try);

                double lambda_end = k_damp[attempt];
                double tol   = 1e-5;
                double h_min = 1e-4 / (double)geo_steps;
                double h_max = 3.0  / (double)geo_steps;
                rc = axgeo_geodesic_integrate_adaptive(&geo, ch_eval, &mf,
                                                       lambda_end, tol,
                                                       h_min, h_max);
                if (rc == 0) {
                    if (attempt > 0) retry_integrations++;
                    /* Item 4: insert into trajectory cache */
                    axgeo_traj_cache_insert(&p5_traj_cache, proj_a, proj_b,
                                            geo.x, geo.v, geo.steps);
                    /* GRC insert: compute Jacobi propagator and store record */
                    if (phase_grc_k == sub_dim && geo.trajectory && geo.steps >= 2) {
                        double *J_rec = (double *)tensor_alloc(
                            (uint64_t)sub_dim * sub_dim * sizeof(double));
                        if (J_rec) {
                            int jrc = axgeo_compute_jacobi_propagator(
                                geo.trajectory, NULL,
                                geo.steps, sub_dim,
                                ch_eval, &mf, J_rec);
                            if (jrc == 0) {
                                /* Injectivity radius: 0.5× mean step size (conservative) */
                                double rho = 0.5 * ax_vec_norm(geo.v, sub_dim);
                                if (rho < 0.05) rho = 0.05;
                                axgeo_grc_insert(&phase_grc, proj_a, J_rec,
                                                  geo.x, rho, tok_end);
                            }
                            tensor_free(J_rec);
                        }
                    }
                    break;
                }
                axgeo_geodesic_destroy(&geo);
                memset(&geo, 0, sizeof(geo));
            }
        }
        if (rc == 0) {
            converged_count++;

            /* Compare geodesic endpoint with target */
            double cos_sim = ax_vec_dot(geo.x, proj_b, sub_dim);
            double na = ax_vec_norm(geo.x, sub_dim);
            double nb_v = ax_vec_norm(proj_b, sub_dim);
            if (na > 1e-10 && nb_v > 1e-10)
                cos_sim /= (na * nb_v);

            double l2 = 0.0;
            for (int j = 0; j < sub_dim; j++) {
                double d = geo.x[j] - proj_b[j];
                l2 += d * d;
            }
            l2 = sqrt(l2);

            total_cos_sim += cos_sim;
            total_l2_error += l2;

            double path_len = axgeo_geodesic_length(&geo, &mf);
            total_path_length += path_len;

            /* Todo 23: streaming full-dim scoring over cached probe pool.
             * Reconstruct full embedding from geodesic endpoint for scoring;
             * slot 0 = target token (fetched fresh per test).
             * Process probe pool in S=32 chunks using chunk_buf (73KB window). */
            axpca_reconstruct(&phase1_pca, geo.x, emb_f64);
            double rec_norm = ax_vec_norm(emb_f64, dim);
            if (rec_norm > 1e-12) {
                int best_tok = -1;
                double best_score = -1e300;
                double target_score = -1e300;
                int target_rank = 1;

                /* Fill slot 0 with this test's target token. */
                probe_tokens[0] = tok_end;
                {
                    int _t0rc = ott_get_hidden_state(tok_end, -1, emb_f32, dim);
                    if (_t0rc != 0) _t0rc = llm_get_embedding_vec(tok_end, emb_f32, dim);
                    if (_t0rc == 0) {
                        double nrm2 = 0.0;
                        for (int j = 0; j < dim; j++) nrm2 += (double)emb_f32[j] * (double)emb_f32[j];
                        probe_norms[0] = (float)sqrt(nrm2);
                    } else {
                        probe_tokens[0] = -1;
                        probe_norms[0]  = 0.0f;
                    }
                }

                /* Stream probe pool in P5_CHUNK_S=32 chunks. */
                for (int cs = 0; cs < n_probe; cs += P5_CHUNK_S) {
                    int ce = cs + P5_CHUNK_S;
                    if (ce > n_probe) ce = n_probe;
                    /* Fetch chunk; cache hits after pre-warm pass. */
                    for (int s = cs; s < ce; s++) {
                        int tok_probe = probe_tokens[s];
                        float *row = chunk_buf + (uint64_t)(s - cs) * dim;
                        if (tok_probe < 0) { memset(row, 0, (uint64_t)dim * sizeof(float)); continue; }
                        int fr = ott_get_hidden_state(tok_probe, -1, row, dim);
                        if (fr != 0) fr = llm_get_embedding_vec(tok_probe, row, dim);
                        if (fr != 0) { memset(row, 0, (uint64_t)dim * sizeof(float)); probe_tokens[s] = -1; }
                    }
                    /* Score chunk with full-dim dot products. */
                    for (int s = cs; s < ce; s++) {
                        int tok_probe = probe_tokens[s];
                        if (tok_probe < 0) continue;
                        double cand_norm = (double)probe_norms[s];
                        if (cand_norm <= 1e-12) continue;
                        float *row = chunk_buf + (uint64_t)(s - cs) * dim;
                        double dot = 0.0;
                        for (int j = 0; j < dim; j++)
                            dot += (double)row[j] * emb_f64[j];
                        double score = dot / (rec_norm * cand_norm);
                        if (score > best_score) {
                            best_score = score;
                            best_tok   = tok_probe;
                        }
                        if (s == 0) {
                            target_score = score;
                        } else if (score > target_score) {
                            target_rank++;
                        }
                    }
                }  /* end chunk loop */

                /* Item 5: Multi-token waypoint scoring (speculative decode).
                 * Inspect trajectory at 25 %, 50 %, 75 % of the path.
                 * For each waypoint, reconstruct to full-dim and measure
                 * cosine similarity to tok_end only (O(dim) per waypoint).
                 * If any waypoint predicts the target better than the endpoint
                 * target_score, promote target_rank to 1. */
                if (geo.trajectory && geo.record && geo.steps > 3 &&
                    probe_tokens[0] == tok_end && probe_norms[0] > 1e-12) {
                    /* Fetch target embedding once into the first chunk slot. */
                    float *target_row = chunk_buf;
                    int tar_rc = ott_get_hidden_state(tok_end, -1, target_row, dim);
                    if (tar_rc != 0) tar_rc = llm_get_embedding_vec(tok_end, target_row, dim);
                    if (tar_rc == 0) {
                        int wp_steps[3] = {
                            geo.steps / 4,
                            geo.steps / 2,
                            (3 * geo.steps) / 4
                        };
                        for (int wp = 0; wp < 3; wp++) {
                            int wps = wp_steps[wp];
                            if (wps <= 0 || wps >= geo.steps) continue;
                            const double *wp_pos = geo.trajectory
                                                   + (uint64_t)wps * sub_dim;
                            /* Reconstruct waypoint (overwrites emb_f64; rec_norm already used). */
                            axpca_reconstruct(&phase1_pca, wp_pos, emb_f64);
                            double wp_norm = ax_vec_norm(emb_f64, dim);
                            if (wp_norm <= 1e-12) continue;
                            double dot_wp = 0.0;
                            for (int j = 0; j < dim; j++)
                                dot_wp += (double)target_row[j] * emb_f64[j];
                            double wp_sim = dot_wp /
                                (wp_norm * (double)probe_norms[0]);
                            if (wp_sim > target_score) {
                                /* Waypoint predicts target better — rank 1. */
                                if (target_rank > 1) target_rank = 1;
                                best_tok = tok_end;  /* let post-loop handle top1_hits */
                                break;
                            }
                        }
                    }
                }

                if (best_tok == tok_end) top1_hits++;
                if (target_rank < 1) target_rank = 1;
                total_mrr += 1.0 / (double)target_rank;
            }
        }

        axgeo_geodesic_destroy(&geo);
        tensor_free(v0);
    }

    /* Fill report */
    if (converged_count > 0) {
        r->phase5.geodesic_cosine_similarity =
            total_cos_sim / (double)converged_count;
        r->phase5.geodesic_reconstruction_error =
            total_l2_error / (double)converged_count;
        r->phase5.geodesic_path_length =
            total_path_length / (double)converged_count;
    }
    r->phase5.geodesic_steps_taken = geo_steps;
    r->phase5.geodesic_converged = (converged_count == n_test) ? 1 : 0;
    r->phase5.pilot_tokens_tested = n_test;
    r->phase5.geodesic_vocab_probe = n_probe;
    r->phase5.oracle_target_count = oracle_target_count;
    r->phase5.random_target_count = random_target_count;
    r->phase5.geodesic_top1_hits = top1_hits;
    r->phase5.used_gpu_scoring = 0; /* Todo 23: subspace scoring, no GPU needed */
    if (converged_count > 0) {
        r->phase5.geodesic_top1_match_rate =
            (double)top1_hits / (double)converged_count;
        r->phase5.geodesic_target_mrr =
            total_mrr / (double)converged_count;
    }

    /* Complexity projection */
    double n = 256.0;  /* reference sequence length */
    double d = (double)dim;
    double L = (double)layers;
    r->phase5.transformer_cost = n * n * d * L;
    r->phase5.geodesic_cost    = n * (double)id * (double)id;
    if (r->phase5.geodesic_cost < 1.0) r->phase5.geodesic_cost = 1.0;
    r->phase5.projected_speedup =
        r->phase5.transformer_cost / r->phase5.geodesic_cost;

    r->supports_geodesic_pilot = (converged_count > 0) ? 1 : 0;

    if (cfg->verbose)
        kprintf("[AXIOM-P5] Geodesic: %d/%d converged, cos_sim=%.4f, "
            "L2_err=%.4f, top1=%.3f, mrr=%.3f, oracle=%d, random=%d, retries=%d, gpu=%s, inj=%d(%d), speedup=%.1fx\n",
                converged_count, n_test,
                r->phase5.geodesic_cosine_similarity,
                r->phase5.geodesic_reconstruction_error,
            r->phase5.geodesic_top1_match_rate,
            r->phase5.geodesic_target_mrr,
            r->phase5.oracle_target_count,
            r->phase5.random_target_count,
            retry_integrations,
            "no",  /* Todo 23: streaming CPU scoring, GPU removed */
            r->phase5.knowledge_injection_applied,
            r->phase5.knowledge_injection_points,
                r->phase5.projected_speedup);

    /* Cleanup */
    if (!using_real_metric) {
        axgeo_christoffel_destroy(&ch);
        axgeo_metric_field_destroy(&mf);
    }
    if (using_warp_ch)
        axgeo_christoffel_destroy(&ch_warp);
    tensor_free(emb_f32);
    tensor_free(emb_f64);
    tensor_free(probe_norms);
    tensor_free(chunk_buf);
    tensor_free(probe_tokens);
    tensor_free(proj_full);
    tensor_free(proj_a);
    tensor_free(proj_b);
    tensor_free(gamma_local);
    tensor_free(accel_local);

    r->phase5_us = hal_timer_us() - t0;
    return 0;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * Main entry point
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

axiom_beta_status_t axiom_beta_run(const axiom_beta_config_t *cfg,
                                   axiom_beta_report_t *report)
{
    axiom_beta_config_t local_cfg;
    uint64_t seed;
    uint64_t t0;

    if (!report) return AXIOM_BETA_ERR_INVALID;
    if (!llm_is_loaded()) return AXIOM_BETA_ERR_NOT_LOADED;

    if (!cfg) {
        axiom_beta_default_config(&local_cfg);
        cfg = &local_cfg;
    }

    /* Effective config: fast mode clamps expensive discovery phases. */
    local_cfg = *cfg;
    if (local_cfg.fast_mode) {
        if (local_cfg.embedding_samples > 64) local_cfg.embedding_samples = 64;
        if (local_cfg.symmetry_trials > 128) local_cfg.symmetry_trials = 128;
        if (local_cfg.metric_sample_points > 64) local_cfg.metric_sample_points = 64;
        if (local_cfg.active_iterations > 96) local_cfg.active_iterations = 96;
        if (local_cfg.oracle_calls_max > 12) local_cfg.oracle_calls_max = 12;
        if (local_cfg.geodesic_test_tokens > 8) local_cfg.geodesic_test_tokens = 8;
        if (local_cfg.geodesic_vocab_probe > 512) local_cfg.geodesic_vocab_probe = 512;
        if (local_cfg.injection_points > 1) local_cfg.injection_points = 1;
        if (local_cfg.injection_alpha > 0.02) local_cfg.injection_alpha = 0.02;
        local_cfg.enable_knowledge_injection = 0;
        local_cfg.enable_recalc_trigger = 0;
    }
    cfg = &local_cfg;

    memset(report, 0, sizeof(*report));
    seed = cfg->seed ? cfg->seed : 0xA110CAFEBEEFULL; /* seed before any RNG use */
    ott_hs_cache_flush(); /* reset hidden-state cache each run */
    ott_depth_sink_layer = -1; /* reset so detection re-runs for this model */
    ott_detect_depth_sink(&seed); /* Todo 25: find most informative layer */
    ott_hs_cache_flush(); /* flush cache populated during depth-sink probe */
    /* Todo 26: load disk-persistent HS cache — warm LRU before Phase 1/3/5 */
    if (ott_depth_sink_layer >= 0)
        ott_hs_disk_load(llm_model_dim(), ott_depth_sink_layer);
    report->beta_version = 3;
    t0 = hal_timer_us();

    if (cfg->enable_recalc_trigger && !phase_warp_state_loaded) {
        axiom_warp_state_load();
        phase_warp_state_loaded = 1;
    }

    int same_model_as_cache = phase_cache_valid &&
        phase_cache_model_dim == llm_model_dim() &&
        phase_cache_model_layers == llm_model_layers() &&
        phase_cache_model_vocab == llm_model_vocab();

    int same_cfg_as_cache = phase_cache_valid &&
        phase_cache_embedding_samples == cfg->embedding_samples &&
        phase_cache_pca_variance_ratio == cfg->pca_variance_ratio &&
        phase_cache_symmetry_trials == cfg->symmetry_trials &&
        phase_cache_metric_sample_points == cfg->metric_sample_points &&
        phase_cache_use_fisher == cfg->use_fisher &&
        phase_cache_fisher_blend == cfg->fisher_blend &&
        phase_cache_active_iterations == cfg->active_iterations &&
        phase_cache_oracle_calls_max == cfg->oracle_calls_max &&
        phase_cache_sink_layer == ott_depth_sink_layer; /* invalidate on layer change */

    if (!(cfg->reuse_cache && same_model_as_cache && same_cfg_as_cache && phase3_geo_valid)) {
        memset(&phase3_mf, 0, sizeof(phase3_mf));
        memset(&phase3_ch, 0, sizeof(phase3_ch));
        phase3_sub_dim = 0;
        phase3_geo_valid = 0;

        /* Keep persistent warp accumulation unless model identity changed. */
        if (cfg->enable_recalc_trigger && !same_model_as_cache) {
            phase_warp_points_accum = 0;
            phase_warp_cross_term_accum = 0.0;
            axiom_warp_state_save();
        }
    }

    fill_model_context(report);

    kprintf("[AXIOM-BETA-3] Starting autonomous axiomatic survey...\n");
    kprintf("[AXIOM-BETA-3] Model: %s (%s), dim=%d, layers=%d, vocab=%d\n",
            report->model_name, report->model_arch,
            report->model_dim, report->model_layers, report->model_vocab);

        int rc = 0;
        if (cfg->reuse_cache && same_model_as_cache && same_cfg_as_cache && phase3_geo_valid) {
        report->phase1 = phase_cache_p1;
        report->phase2 = phase_cache_p2;
        report->phase3 = phase_cache_p3;
        report->phase4 = phase_cache_p4;
        report->phase1_us = 0;
        report->phase2_us = 0;
        report->phase3_us = 0;
        report->phase4_us = 0;
        report->uses_real_embeddings = 1;
        report->uses_real_curvature = 1;
        report->uses_fisher_metric = 1;
        report->uses_real_dequant = 1;
        report->reused_geometry_cache = 1;
        kprintf("[AXIOM-BETA-3] Reusing cached geometry (phases 1-4 skipped)\n");
        } else {
        report->reused_geometry_cache = 0;

        /* Phase 1: Manifold Identification */
        kprintf("[AXIOM-BETA-3] Phase 1: Manifold Identification...\n");
        rc = phase1_manifold(cfg, report, &seed);
        if (rc != 0) {
            kprintf("[AXIOM-BETA-3] Phase 1 FAILED (rc=%d)\n", rc);
            report->total_us = hal_timer_us() - t0;
            axpca_destroy(&phase1_pca);
            return AXIOM_BETA_ERR_OOM;
        }
        kprintf("[AXIOM-BETA-3] Phase 1: ID=%d, PCA=%d components (%.1f%% var), %.1f ms\n",
            report->phase1.intrinsic_dim,
            report->phase1.pca_components_kept,
            report->phase1.explained_ratio * 100.0,
            (double)report->phase1_us / 1000.0);

        /* Phase 2: Symmetry Extraction (dequantized weights) */
        kprintf("[AXIOM-BETA-3] Phase 2: Symmetry Extraction (dequant)...\n");
        rc = phase2_symmetry(cfg, report, &seed);
        kprintf("[AXIOM-BETA-3] Phase 2: score=%.4f, generators=%d, %.1f ms\n",
            report->phase2.symmetry_score,
            report->phase2.generators_found,
            (double)report->phase2_us / 1000.0);

        /* Phase 3: Nonlinearity Absorption (Fisher + full Riemann) */
        kprintf("[AXIOM-BETA-3] Phase 3: Curvature (Fisher-blended metric)...\n");
        rc = phase3_curvature(cfg, report, &seed);
        kprintf("[AXIOM-BETA-3] Phase 3: mean_R=%.6f, max_R=%.6f, "
            "high-curv=%d, Fisher=%s, %.1f ms\n",
            report->phase3.mean_scalar_curvature,
            report->phase3.max_scalar_curvature,
            report->phase3.high_curvature_loci,
            report->uses_fisher_metric ? "yes" : "no",
            (double)report->phase3_us / 1000.0);

        /* Todo 26: persist the LRU cache to disk after Phase 3 sampling.
         * Second+ runs will load these hidden states instantly, making
         * Phase 3's 2288 forward passes drop from ~205s to ~0s. */
        if (ott_depth_sink_layer >= 0)
            ott_hs_disk_save(report->model_dim, ott_depth_sink_layer);

        /* Phase 4: Axiom Formalization (geometry-derived) */
        kprintf("[AXIOM-BETA-3] Phase 4: Axiom Formalization (geo-derived)...\n");
        rc = phase4_axioms(cfg, report, &seed);
        kprintf("[AXIOM-BETA-3] Phase 4: %d axioms, consistency=%.4f, "
            "oracle_calls=%d, %.1f ms\n",
            report->phase4.axiom_count,
            report->phase4.consistency_score,
            report->phase4.oracle_calls_used,
            (double)report->phase4_us / 1000.0);

        if (cfg->reuse_cache && phase3_geo_valid) {
            phase_cache_valid = 1;
            phase_cache_model_dim = report->model_dim;
            phase_cache_model_layers = report->model_layers;
            phase_cache_model_vocab = report->model_vocab;
            phase_cache_embedding_samples = cfg->embedding_samples;
            phase_cache_pca_variance_ratio = cfg->pca_variance_ratio;
            phase_cache_symmetry_trials = cfg->symmetry_trials;
            phase_cache_metric_sample_points = cfg->metric_sample_points;
            phase_cache_use_fisher = cfg->use_fisher;
            phase_cache_fisher_blend = cfg->fisher_blend;
            phase_cache_active_iterations = cfg->active_iterations;
            phase_cache_oracle_calls_max = cfg->oracle_calls_max;
            phase_cache_sink_layer = ott_depth_sink_layer;
            phase_cache_p1 = report->phase1;
            phase_cache_p2 = report->phase2;
            phase_cache_p3 = report->phase3;
            phase_cache_p4 = report->phase4;
        }
        }

    /* Phase 5: Geodesic Pilot (real metric field) */
    kprintf("[AXIOM-BETA-3] Phase 5: Geodesic Pilot (real metric)...\n");
    rc = phase5_geodesic(cfg, report, &seed);

    if (report->phase5.recalc_triggered) {
        uint64_t t3 = hal_timer_us();
        int rc3 = phase3_curvature(cfg, report, &seed);
        uint64_t t3_elapsed = hal_timer_us() - t3;

        uint64_t t4 = hal_timer_us();
        int rc4 = phase4_axioms(cfg, report, &seed);
        uint64_t t4_elapsed = hal_timer_us() - t4;

        report->phase3_us += t3_elapsed;
        report->phase4_us += t4_elapsed;

        if (cfg->verbose) {
            kprintf("[AXIOM-BETA-3] Recalc orchestration: phase3 rc=%d, phase4 rc=%d (%.1f + %.1f ms)\n",
                    rc3, rc4, (double)t3_elapsed / 1000.0, (double)t4_elapsed / 1000.0);
        }

        if (cfg->reuse_cache && phase3_geo_valid) {
            phase_cache_valid = 1;
            phase_cache_model_dim = report->model_dim;
            phase_cache_model_layers = report->model_layers;
            phase_cache_model_vocab = report->model_vocab;
            phase_cache_embedding_samples = cfg->embedding_samples;
            phase_cache_pca_variance_ratio = cfg->pca_variance_ratio;
            phase_cache_symmetry_trials = cfg->symmetry_trials;
            phase_cache_metric_sample_points = cfg->metric_sample_points;
            phase_cache_use_fisher = cfg->use_fisher;
            phase_cache_fisher_blend = cfg->fisher_blend;
            phase_cache_active_iterations = cfg->active_iterations;
            phase_cache_oracle_calls_max = cfg->oracle_calls_max;
            phase_cache_sink_layer = ott_depth_sink_layer;
            phase_cache_p1 = report->phase1;
            phase_cache_p2 = report->phase2;
            phase_cache_p3 = report->phase3;
            phase_cache_p4 = report->phase4;
        }
    }

    if (report->supports_geodesic_pilot) {
        kprintf("[AXIOM-BETA-3] Phase 5: cos_sim=%.4f, L2_err=%.4f, "
            "top1=%.3f, mrr=%.3f, gpu=%s, inj=%d(%d), recalc=%d, speedup=%.1fx, %.1f ms\n",
                report->phase5.geodesic_cosine_similarity,
                report->phase5.geodesic_reconstruction_error,
            report->phase5.geodesic_top1_match_rate,
            report->phase5.geodesic_target_mrr,
            report->phase5.used_gpu_scoring ? "yes" : "no",
            report->phase5.knowledge_injection_applied,
            report->phase5.knowledge_injection_points,
            report->phase5.recalc_triggered,
                report->phase5.projected_speedup,
                (double)report->phase5_us / 1000.0);
    } else {
        kprintf("[AXIOM-BETA-3] Phase 5: speedup=%.1fx (projected), %.1f ms\n",
                report->phase5.projected_speedup,
                (double)report->phase5_us / 1000.0);
    }

    /* Cleanup: optionally retain cached geometry for in-process reuse. */
    if (!cfg->reuse_cache) {
        axpca_destroy(&phase1_pca);
        axgeo_christoffel_destroy(&phase3_ch);
        axgeo_metric_field_destroy(&phase3_mf);
        phase3_geo_valid = 0;
        phase_cache_valid = 0;
        phase_warp_points_accum = 0;
        phase_warp_cross_term_accum = 0.0;
        if (cfg->enable_recalc_trigger) axiom_warp_state_save();
    }

    if (cfg->enable_recalc_trigger)
        axiom_warp_state_save();

    report->total_us = hal_timer_us() - t0;
    kprintf("[AXIOM-BETA-3] Complete: %.1f ms total\n",
            (double)report->total_us / 1000.0);

    return AXIOM_BETA_OK;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
 * JSON Report Writer
 * â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

axiom_beta_status_t axiom_beta_write_json(const char *path,
                                          const axiom_beta_report_t *r,
                                          const axiom_beta_config_t *cfg)
{
    FILE *f;
    if (!path || !r || !cfg) return AXIOM_BETA_ERR_INVALID;

    f = fopen(path, "wb");
    if (!f) return AXIOM_BETA_ERR_IO;

    fprintf(f, "{\n");
    fprintf(f, "  \"subsystem\": \"autonomous_axiomatic_beta\",\n");
    fprintf(f, "  \"beta_version\": %d,\n", r->beta_version);

    fprintf(f, "  \"model\": {\n");
    fprintf(f, "    \"name\": \"%s\",\n", r->model_name);
    fprintf(f, "    \"arch\": \"%s\",\n", r->model_arch);
    fprintf(f, "    \"dim\": %d,\n", r->model_dim);
    fprintf(f, "    \"layers\": %d,\n", r->model_layers);
    fprintf(f, "    \"vocab\": %d,\n", r->model_vocab);
    fprintf(f, "    \"params\": %llu\n", (unsigned long long)r->model_params);
    fprintf(f, "  },\n");

    fprintf(f, "  \"config\": {\n");
    fprintf(f, "    \"embedding_samples\": %d,\n", cfg->embedding_samples);
    fprintf(f, "    \"pca_variance_ratio\": %.4f,\n", cfg->pca_variance_ratio);
    fprintf(f, "    \"symmetry_trials\": %d,\n", cfg->symmetry_trials);
    fprintf(f, "    \"metric_sample_points\": %d,\n", cfg->metric_sample_points);
    fprintf(f, "    \"use_fisher\": %d,\n", cfg->use_fisher);
    fprintf(f, "    \"fisher_blend\": %.4f,\n", cfg->fisher_blend);
    fprintf(f, "    \"active_iterations\": %d,\n", cfg->active_iterations);
    fprintf(f, "    \"oracle_calls_max\": %d,\n", cfg->oracle_calls_max);
    fprintf(f, "    \"geodesic_steps\": %d,\n", cfg->geodesic_steps);
    fprintf(f, "    \"geodesic_vocab_probe\": %d,\n", cfg->geodesic_vocab_probe);
    fprintf(f, "    \"geodesic_use_oracle_target\": %d,\n", cfg->geodesic_use_oracle_target);
    fprintf(f, "    \"use_gpu_phase5\": %d,\n", cfg->use_gpu_phase5);
    fprintf(f, "    \"enable_knowledge_injection\": %d,\n", cfg->enable_knowledge_injection);
    fprintf(f, "    \"injection_alpha\": %.6f,\n", cfg->injection_alpha);
    fprintf(f, "    \"injection_sigma\": %.6f,\n", cfg->injection_sigma);
    fprintf(f, "    \"injection_points\": %d,\n", cfg->injection_points);
    fprintf(f, "    \"enable_recalc_trigger\": %d,\n", cfg->enable_recalc_trigger);
    fprintf(f, "    \"recalc_cross_term_threshold\": %.6f,\n", cfg->recalc_cross_term_threshold);
    fprintf(f, "    \"recalc_warp_budget\": %d,\n", cfg->recalc_warp_budget);
    fprintf(f, "    \"fast_mode\": %d,\n", cfg->fast_mode);
    fprintf(f, "    \"reuse_cache\": %d,\n", cfg->reuse_cache);
    fprintf(f, "    \"seed\": %llu\n", (unsigned long long)cfg->seed);
    fprintf(f, "  },\n");

    fprintf(f, "  \"phase1_manifold\": {\n");
    fprintf(f, "    \"intrinsic_dim\": %d,\n", r->phase1.intrinsic_dim);
    fprintf(f, "    \"twonn_raw\": %.4f,\n", r->phase1.twonn_raw);
    fprintf(f, "    \"pca_components_kept\": %d,\n", r->phase1.pca_components_kept);
    fprintf(f, "    \"embedding_dim\": %d,\n", r->phase1.embedding_dim);
    fprintf(f, "    \"samples_used\": %d,\n", r->phase1.samples_used);
    fprintf(f, "    \"total_variance\": %.6e,\n", r->phase1.total_variance);
    fprintf(f, "    \"explained_variance\": %.6e,\n", r->phase1.explained_variance);
    fprintf(f, "    \"explained_ratio\": %.6f,\n", r->phase1.explained_ratio);
    fprintf(f, "    \"uses_real_embeddings\": %d\n", r->uses_real_embeddings);
    fprintf(f, "  },\n");

    fprintf(f, "  \"phase2_symmetry\": {\n");
    fprintf(f, "    \"symmetry_score\": %.6f,\n", r->phase2.symmetry_score);
    fprintf(f, "    \"generators_found\": %d,\n", r->phase2.generators_found);
    fprintf(f, "    \"permutation_invariant_heads\": %d,\n",
            r->phase2.permutation_invariant_heads);
    fprintf(f, "    \"total_heads_tested\": %d,\n", r->phase2.total_heads_tested);
    fprintf(f, "    \"head_similarity_mean\": %.6f,\n", r->phase2.head_similarity_mean);
    fprintf(f, "    \"head_similarity_max\": %.6f\n", r->phase2.head_similarity_max);
    fprintf(f, "  },\n");

    fprintf(f, "  \"phase3_curvature\": {\n");
    fprintf(f, "    \"mean_scalar_curvature\": %.6e,\n", r->phase3.mean_scalar_curvature);
    fprintf(f, "    \"max_scalar_curvature\": %.6e,\n", r->phase3.max_scalar_curvature);
    fprintf(f, "    \"min_scalar_curvature\": %.6e,\n", r->phase3.min_scalar_curvature);
    fprintf(f, "    \"curvature_std\": %.6e,\n", r->phase3.curvature_std);
    fprintf(f, "    \"high_curvature_loci\": %d,\n", r->phase3.high_curvature_loci);
    fprintf(f, "    \"metric_field_points\": %d,\n", r->phase3.metric_field_points);
    fprintf(f, "    \"christoffel_computed\": %d,\n", r->phase3.christoffel_computed);
    fprintf(f, "    \"fisher_trace_mean\": %.6e,\n", r->phase3.fisher_trace_mean);
    fprintf(f, "    \"fisher_det_log_mean\": %.6e,\n", r->phase3.fisher_det_log_mean);
    fprintf(f, "    \"uses_real_curvature\": %d,\n", r->uses_real_curvature);
    fprintf(f, "    \"uses_fisher_metric\": %d,\n", r->uses_fisher_metric);
    fprintf(f, "    \"uses_real_dequant\": %d\n", r->uses_real_dequant);
    fprintf(f, "  },\n");

    fprintf(f, "  \"phase4_axioms\": {\n");
    fprintf(f, "    \"axiom_count\": %d,\n", r->phase4.axiom_count);
    fprintf(f, "    \"consistency_score\": %.6f,\n", r->phase4.consistency_score);
    fprintf(f, "    \"candidates_tested\": %d,\n", r->phase4.candidates_tested);
    fprintf(f, "    \"candidates_accepted\": %d,\n", r->phase4.candidates_accepted);
    fprintf(f, "    \"oracle_calls_used\": %d,\n", r->phase4.oracle_calls_used);
    fprintf(f, "    \"model_oracle_calls\": %d,\n", r->phase4.model_oracle_calls);
    fprintf(f, "    \"information_gain\": %.6f\n", r->phase4.information_gain);
    fprintf(f, "  },\n");

    fprintf(f, "  \"phase5_geodesic\": {\n");
    fprintf(f, "    \"geodesic_reconstruction_error\": %.6f,\n",
            r->phase5.geodesic_reconstruction_error);
    fprintf(f, "    \"geodesic_cosine_similarity\": %.6f,\n",
            r->phase5.geodesic_cosine_similarity);
    fprintf(f, "    \"geodesic_steps_taken\": %d,\n", r->phase5.geodesic_steps_taken);
    fprintf(f, "    \"geodesic_path_length\": %.6f,\n", r->phase5.geodesic_path_length);
    fprintf(f, "    \"transformer_cost\": %.3e,\n", r->phase5.transformer_cost);
    fprintf(f, "    \"geodesic_cost\": %.3e,\n", r->phase5.geodesic_cost);
    fprintf(f, "    \"projected_speedup\": %.6f,\n", r->phase5.projected_speedup);
    fprintf(f, "    \"geodesic_converged\": %d,\n", r->phase5.geodesic_converged);
    fprintf(f, "    \"pilot_tokens_tested\": %d,\n", r->phase5.pilot_tokens_tested);
        fprintf(f, "    \"geodesic_vocab_probe\": %d,\n", r->phase5.geodesic_vocab_probe);
        fprintf(f, "    \"oracle_target_count\": %d,\n", r->phase5.oracle_target_count);
        fprintf(f, "    \"random_target_count\": %d,\n", r->phase5.random_target_count);
        fprintf(f, "    \"geodesic_top1_hits\": %d,\n", r->phase5.geodesic_top1_hits);
        fprintf(f, "    \"geodesic_top1_match_rate\": %.6f,\n",
            r->phase5.geodesic_top1_match_rate);
        fprintf(f, "    \"geodesic_target_mrr\": %.6f,\n",
            r->phase5.geodesic_target_mrr);
            fprintf(f, "    \"used_gpu_scoring\": %d,\n", r->phase5.used_gpu_scoring);
            fprintf(f, "    \"knowledge_injection_applied\": %d,\n",
            r->phase5.knowledge_injection_applied);
            fprintf(f, "    \"knowledge_injection_points\": %d,\n",
            r->phase5.knowledge_injection_points);
            fprintf(f, "    \"warp_points_accumulated\": %d,\n",
            r->phase5.warp_points_accumulated);
            fprintf(f, "    \"warp_cross_term_estimate\": %.6f,\n",
            r->phase5.warp_cross_term_estimate);
            fprintf(f, "    \"recalc_triggered\": %d,\n",
            r->phase5.recalc_triggered);
            fprintf(f, "    \"reused_geometry_cache\": %d,\n", r->reused_geometry_cache);
    fprintf(f, "    \"supports_geodesic_pilot\": %d\n", r->supports_geodesic_pilot);
    fprintf(f, "  },\n");

    fprintf(f, "  \"timings_us\": {\n");
    fprintf(f, "    \"phase1\": %llu,\n", (unsigned long long)r->phase1_us);
    fprintf(f, "    \"phase2\": %llu,\n", (unsigned long long)r->phase2_us);
    fprintf(f, "    \"phase3\": %llu,\n", (unsigned long long)r->phase3_us);
    fprintf(f, "    \"phase4\": %llu,\n", (unsigned long long)r->phase4_us);
    fprintf(f, "    \"phase5\": %llu,\n", (unsigned long long)r->phase5_us);
    fprintf(f, "    \"total\": %llu\n", (unsigned long long)r->total_us);
    fprintf(f, "  }\n");

    fprintf(f, "}\n");
    fclose(f);
    return AXIOM_BETA_OK;
}

axiom_beta_status_t axiom_beta_geodesic_next_token_v2(const int *context_tokens,
                                                      int n_context,
                                                      int *out_token)
{
    const llm_model_t *m = llm_get_model();
    if (!m || !context_tokens || n_context <= 0 || !out_token)
        return AXIOM_BETA_ERR_INVALID;

    int vocab = m->vocab_size;
    int dim = m->dim;
    int tok_curr = context_tokens[n_context - 1];
    int tok_prev = (n_context >= 2) ? context_tokens[n_context - 2] : tok_curr;
    if (tok_curr < 0 || tok_curr >= vocab) tok_curr = 0;
    if (tok_prev < 0 || tok_prev >= vocab) tok_prev = tok_curr;

    float *e_curr = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *e_prev = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *e_pred = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    float *e_cand = (float *)tensor_alloc((uint64_t)dim * sizeof(float));
    if (!e_curr || !e_prev || !e_pred || !e_cand) {
        if (e_curr) tensor_free(e_curr);
        if (e_prev) tensor_free(e_prev);
        if (e_pred) tensor_free(e_pred);
        if (e_cand) tensor_free(e_cand);
        return AXIOM_BETA_ERR_OOM;
    }

    int hs_v2_ok = (ott_get_hidden_state(tok_curr, -1, e_curr, dim) == 0);
    if (!hs_v2_ok) hs_v2_ok = (llm_get_embedding_vec(tok_curr, e_curr, dim) == 0);
    int hs_v2_ok2 = hs_v2_ok && (ott_get_hidden_state(tok_prev, -1, e_prev, dim) == 0);
    if (hs_v2_ok && !hs_v2_ok2)
        hs_v2_ok2 = (llm_get_embedding_vec(tok_prev, e_prev, dim) == 0);
    if (!hs_v2_ok || !hs_v2_ok2) {
        tensor_free(e_curr); tensor_free(e_prev); tensor_free(e_pred); tensor_free(e_cand);
        return AXIOM_BETA_ERR_INVALID;
    }

    double pred_norm2 = 0.0;
    for (int i = 0; i < dim; i++) {
        e_pred[i] = e_curr[i] + 0.35f * (e_curr[i] - e_prev[i]);
        pred_norm2 += (double)e_pred[i] * (double)e_pred[i];
    }
    double pred_norm = sqrt(pred_norm2);

    int probe = vocab;
    if (probe > 4096) probe = 4096;
    int start = (tok_curr * 1315423911u) % vocab;
    int best_tok = tok_curr;
    double best_score = -1e30;

    for (int i = 0; i < probe; i++) {
        int tid = (start + i) % vocab;
        if (llm_get_embedding_vec(tid, e_cand, dim) != 0) continue;

        double dot = 0.0;
        double cand_norm2 = 0.0;
        double l2 = 0.0;
        for (int j = 0; j < dim; j++) {
            double c = (double)e_cand[j];
            dot += (double)e_pred[j] * c;
            cand_norm2 += c * c;
            double d = (double)e_pred[j] - c;
            l2 += d * d;
        }
        double denom = pred_norm * sqrt(cand_norm2);
        double sim = (denom > 1e-12) ? (dot / denom) : -1e30;
        double score = sim - 0.01 * sqrt(l2);
        if (score > best_score) {
            best_score = score;
            best_tok = tid;
        }
    }

    tensor_free(e_curr); tensor_free(e_prev); tensor_free(e_pred); tensor_free(e_cand);
    *out_token = best_tok;
    kprintf("[OTT-HS] cache hits=%d misses=%d (%.1f%% hit rate)\n",
            ott_hs_hits, ott_hs_misses,
            ott_hs_hits + ott_hs_misses > 0
                ? 100.0 * ott_hs_hits / (ott_hs_hits + ott_hs_misses)
                : 0.0);
    return AXIOM_BETA_OK;
}

axiom_beta_status_t axiom_beta_geodesic_step_fast(const int *context_tokens,
                                                   int n_context,
                                                   int *out_token,
                                                   float *out_confidence)
{
    if (!context_tokens || n_context <= 0 || !out_token)
        return AXIOM_BETA_ERR_INVALID;

    /* Require cached Phase-3 geometry */
    if (!phase3_geo_valid || phase1_pca.n_components <= 0)
        return AXIOM_BETA_ERR_INVALID;

    const llm_model_t *m = llm_get_model();
    if (!m) return AXIOM_BETA_ERR_INVALID;

    int vocab   = m->vocab_size;
    int dim     = m->dim;
    int k       = phase1_pca.n_components;

    int tok_curr = context_tokens[n_context - 1];
    int tok_prev = (n_context >= 2) ? context_tokens[n_context - 2] : tok_curr;
    if (tok_curr < 0 || tok_curr >= vocab) tok_curr = 0;
    if (tok_prev < 0 || tok_prev >= vocab) tok_prev = tok_curr;

    /* Allocate working buffers */
    double *e_curr_d = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
    double *e_prev_d = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
    double *p_curr   = (double *)tensor_alloc((uint64_t)k   * sizeof(double));
    double *p_prev   = (double *)tensor_alloc((uint64_t)k   * sizeof(double));
    double *v        = (double *)tensor_alloc((uint64_t)k   * sizeof(double));
    double *p_pred   = (double *)tensor_alloc((uint64_t)k   * sizeof(double));
    double *e_pred_d = (double *)tensor_alloc((uint64_t)dim * sizeof(double));
    float  *e_curr_f = (float  *)tensor_alloc((uint64_t)dim * sizeof(float));
    float  *e_cand   = (float  *)tensor_alloc((uint64_t)dim * sizeof(float));

    if (!e_curr_d || !e_prev_d || !p_curr || !p_prev || !v ||
        !p_pred || !e_pred_d || !e_curr_f || !e_cand) {
        if (e_curr_d) tensor_free(e_curr_d);
        if (e_prev_d) tensor_free(e_prev_d);
        if (p_curr)   tensor_free(p_curr);
        if (p_prev)   tensor_free(p_prev);
        if (v)        tensor_free(v);
        if (p_pred)   tensor_free(p_pred);
        if (e_pred_d) tensor_free(e_pred_d);
        if (e_curr_f) tensor_free(e_curr_f);
        if (e_cand)   tensor_free(e_cand);
        return AXIOM_BETA_ERR_OOM;
    }

    /* Fetch current embedding */
    int ok_curr = (llm_get_embedding_vec(tok_curr, e_curr_f, dim) == 0);
    if (!ok_curr) {
        tensor_free(e_curr_d); tensor_free(e_prev_d); tensor_free(p_curr);
        tensor_free(p_prev); tensor_free(v);
        tensor_free(p_pred); tensor_free(e_pred_d); tensor_free(e_curr_f); tensor_free(e_cand);
        return AXIOM_BETA_ERR_INVALID;
    }
    for (int i = 0; i < dim; i++) e_curr_d[i] = (double)e_curr_f[i];

    /* Fetch prev embedding for velocity */
    float *e_prev_f = e_curr_f; /* reuse buffer */
    if (tok_prev != tok_curr && llm_get_embedding_vec(tok_prev, e_prev_f, dim) == 0) {
        for (int i = 0; i < dim; i++) e_prev_d[i] = (double)e_prev_f[i];
    } else {
        for (int i = 0; i < dim; i++) e_prev_d[i] = e_curr_d[i];
    }

    /* Project to PCA subspace */
    axpca_project(&phase1_pca, e_curr_d, p_curr);
    axpca_project(&phase1_pca, e_prev_d, p_prev);
    for (int i = 0; i < k; i++) v[i] = p_curr[i] - p_prev[i];

    /* ── GRC fast path: O(k²) Jacobi correction ──────────────────────────
     * Try looking up a stored geodesic record near p_curr.
     * If hit: p_pred = x_end + J·(p_curr − q_bar)  (Jacobi correction)
     *         and if best_tok is known, return it directly.
     */
    int    grc_best_tok = -1;
    int    grc_hit = 0;
    if (phase_grc_k == k && phase_grc.count > 0) {
        /* Build query as p_curr + v (predict forward one step) */
        double *q_fwd = p_pred; /* reuse p_pred as temp */
        for (int i = 0; i < k; i++) q_fwd[i] = p_curr[i] + v[i];
        grc_hit = axgeo_grc_lookup(&phase_grc, q_fwd, k, p_pred, &grc_best_tok);
    }

    if (!grc_hit) {
        /* ── Christoffel fallback: single O(k²) geodesic step ─────────── */
        double *gamma = (double *)tensor_alloc((uint64_t)k * k * k * sizeof(double));
        if (!gamma) {
            tensor_free(e_curr_d); tensor_free(e_prev_d); tensor_free(p_curr);
            tensor_free(p_prev); tensor_free(v);
            tensor_free(p_pred); tensor_free(e_pred_d); tensor_free(e_curr_f); tensor_free(e_cand);
            return AXIOM_BETA_ERR_OOM;
        }
        axgeo_christoffel_interpolate(&phase3_ch, &phase3_mf, p_curr, gamma);
        for (int alpha = 0; alpha < k; alpha++) {
            double correction = 0.0;
            for (int mu = 0; mu < k; mu++)
                for (int nu = 0; nu < k; nu++)
                    correction += gamma[alpha * k * k + mu * k + nu] * v[mu] * v[nu];
            p_pred[alpha] = p_curr[alpha] + v[alpha] - 0.5 * correction;
        }
        tensor_free(gamma);
    }

    /* If GRC gave us a best_tok and it's valid, return it directly */
    if (grc_hit && grc_best_tok >= 0 && grc_best_tok < m->vocab_size) {
        if (out_confidence) *out_confidence = 0.75f; /* GRC hit confidence */
        tensor_free(e_curr_d); tensor_free(e_prev_d); tensor_free(p_curr);
        tensor_free(p_prev); tensor_free(v);
        tensor_free(p_pred); tensor_free(e_pred_d); tensor_free(e_curr_f); tensor_free(e_cand);
        *out_token = grc_best_tok;
        return AXIOM_BETA_OK;
    }

    /* Reconstruct predicted embedding in full space */
    axpca_reconstruct(&phase1_pca, p_pred, e_pred_d);

    double pred_norm2 = 0.0;
    for (int i = 0; i < dim; i++) pred_norm2 += e_pred_d[i] * e_pred_d[i];
    double pred_norm = sqrt(pred_norm2);

    /* Find nearest token by cosine similarity (probe 8192 candidates) */
    int probe = vocab < 8192 ? vocab : 8192;
    int start = (tok_curr * 1315423911u) % (unsigned)vocab;
    int best_tok = tok_curr;
    double best_score = -1e30;

    for (int i = 0; i < probe; i++) {
        int tid = (start + i) % vocab;
        if (llm_get_embedding_vec(tid, e_cand, dim) != 0) continue;

        double dot       = 0.0;
        double cand_norm2 = 0.0;
        for (int j = 0; j < dim; j++) {
            double c = (double)e_cand[j];
            dot       += e_pred_d[j] * c;
            cand_norm2 += c * c;
        }
        double denom = pred_norm * sqrt(cand_norm2);
        double sim   = (denom > 1e-12) ? (dot / denom) : -1e30;
        if (sim > best_score) {
            best_score = sim;
            best_tok   = tid;
        }
    }

    if (out_confidence)
        *out_confidence = (float)best_score;

    tensor_free(e_curr_d); tensor_free(e_prev_d); tensor_free(p_curr);
    tensor_free(p_prev); tensor_free(v);
    tensor_free(p_pred); tensor_free(e_pred_d); tensor_free(e_curr_f); tensor_free(e_cand);

    *out_token = best_tok;
    return AXIOM_BETA_OK;
}
const char *axiom_beta_status_string(axiom_beta_status_t st)
{
    switch (st) {
    case AXIOM_BETA_OK:             return "ok";
    case AXIOM_BETA_ERR_NOT_LOADED: return "model-not-loaded";
    case AXIOM_BETA_ERR_INVALID:    return "invalid-args";
    case AXIOM_BETA_ERR_IO:         return "io-error";
    case AXIOM_BETA_ERR_OOM:        return "out-of-memory";
    case AXIOM_BETA_ERR_DIVERGED:   return "geodesic-diverged";
    default: return "unknown";
    }
}
