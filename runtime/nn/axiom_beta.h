/*
 * Geodessical Autonomous Axiomatic Subsystem (Beta-3)
 *
 * Treats a trained model's weight tensor as the constitution of a
 * mathematical reality.  Derives the model's intrinsic geometry — metric
 * tensor, symmetry group, curvature field, minimal axiom set — and
 * projects toward native geodesic inference.
 *
 * Five-phase pipeline:
 *  1) Manifold identification — PCA + TwoNN on real embedding geometry
 *  2) Symmetry extraction — dequantized weight analysis + invariance probes
 *  3) Nonlinearity absorption — curvature tensor from Fisher-blended metric
 *  4) Axiom formalization — geometry-derived axiom generation + oracle tests
 *  5) Native inference projection — geodesic pilot on real metric field
 *
 * Layers below:
 *   axiom_linalg.h  — dense matrix ops, PCA, TwoNN, GGUF dequant
 *   axiom_geo.h     — metric tensors, Christoffel symbols, curvature,
 *                      geodesic RK4 integrator, Fisher Information
 */

#ifndef GEODESSICAL_AXIOM_BETA_H
#define GEODESSICAL_AXIOM_BETA_H

#include <stdint.h>
#include "runtime/nn/axiom_linalg.h"

/* ─── Status codes ─── */
typedef enum {
    AXIOM_BETA_OK             =  0,
    AXIOM_BETA_ERR_NOT_LOADED = -1,
    AXIOM_BETA_ERR_INVALID    = -2,
    AXIOM_BETA_ERR_IO         = -3,
    AXIOM_BETA_ERR_OOM        = -4,
    AXIOM_BETA_ERR_DIVERGED   = -5,
} axiom_beta_status_t;

/* ─── Configuration ─── */
typedef struct {
    /* Phase 1: manifold identification */
    int      embedding_samples;  /* Number of token embeddings to sample */
    double   pca_variance_ratio; /* Explained variance threshold (0.95 = keep 95%) */

    /* Phase 2: symmetry extraction */
    int      symmetry_trials;    /* Random invariance probes per layer */

    /* Phase 3: curvature computation */
    int      metric_sample_points; /* Sample points for metric field */
    int      use_fisher;           /* 1 = compute & blend Fisher metric */
    double   fisher_blend;         /* Fisher blend factor α ∈ [0,1] (0.2 default) */
    int      use_weight_pullback;  /* 1 = blend weight-derived pullback metric (OTT k^4 Step 1) */
    double   pullback_blend;       /* Pullback blend factor β ∈ [0,1] (0.5 default) */
    int      pullback_rmsnorm;     /* 1 = apply RMSNorm sphere connection correction */
    double   pullback_rmsnorm_alpha; /* Connection correction blend [0,1] (0.3 default) */

    /* Phase 4: axiom formalization */
    int      active_iterations;  /* Axiom candidate iterations */
    int      oracle_calls_max;   /* Max model oracle calls */

    /* Phase 5: geodesic pilot */
    int      geodesic_steps;     /* RK4 integration steps */
    int      geodesic_test_tokens; /* Tokens for geodesic vs forward-pass comparison */
    int      geodesic_vocab_probe; /* Candidate tokens for endpoint->token projection */
    int      geodesic_use_oracle_target; /* 1 = target token from model next-token oracle */
    int      use_gpu_phase5;      /* 1 = enable CUDA-backed Phase-5 scoring when available */
    int      enable_knowledge_injection; /* 1 = apply local curvature warps before Phase-5 pilot */
    double   injection_alpha;     /* Warp strength alpha */
    double   injection_sigma;     /* Warp radius sigma (in intrinsic coordinates) */
    int      injection_points;    /* Number of local warp points to superpose */
    int      enable_recalc_trigger;      /* 1 = enable warp accumulation trigger */
    double   recalc_cross_term_threshold;/* Trigger threshold for accumulated cross-terms */
    int      recalc_warp_budget;         /* Trigger threshold for accumulated warp points */
    int      fast_mode;           /* 1 = reduced-cost settings for faster E2E runs */
    int      reuse_cache;         /* 1 = allow in-process reuse of prior Phase 1-4 geometry */

    /* OneDecode: amortised one-time bake → instant per-step decode */
    int      one_decode_mode;           /* 1 = use precomputed one-shot decode map  */
    int      one_decode_vocab_coverage; /* Vocab tokens to bake (0 = auto → 2048) */

    /* General */
    uint64_t seed;               /* Deterministic seed (0 = default) */
    int      verbose;            /* Print per-phase diagnostics */
    int      skip_geodesic;      /* 1 = skip phase 5 (expensive) */
} axiom_beta_config_t;

/* ─── Phase 1 results ─── */
typedef struct {
    int    intrinsic_dim;        /* TwoNN estimate */
    int    pca_components_kept;  /* Components above variance threshold */
    double total_variance;       /* Total variance of embedding cloud */
    double explained_variance;   /* Variance explained by kept components */
    double explained_ratio;      /* explained / total */
    double twonn_raw;            /* Raw TwoNN estimate (may be fractional) */
    int    embedding_dim;        /* Full embedding dimension */
    int    samples_used;         /* Actual embeddings sampled */
} axiom_phase1_t;

/* ─── Phase 2 results ─── */
typedef struct {
    double symmetry_score;       /* Mean invariance score [0,1] */
    int    generators_found;     /* Estimated Lie algebra generators */
    int    permutation_invariant_heads; /* Near-identical attention heads */
    int    total_heads_tested;
    double head_similarity_mean; /* Mean pairwise head cosine similarity */
    double head_similarity_max;  /* Maximum pairwise head similarity */
} axiom_phase2_t;

/* ─── Phase 3 results ─── */
typedef struct {
    double mean_scalar_curvature;
    double max_scalar_curvature;
    double min_scalar_curvature;
    int    high_curvature_loci;  /* Points with |R| > 2σ */
    int    metric_field_points;  /* Points in the metric field */
    int    christoffel_computed; /* 1 if Christoffel symbols computed */
    double curvature_std;        /* Standard deviation of scalar curvature */
    double fisher_trace_mean;    /* Mean Fisher trace across metric points */
    double fisher_det_log_mean;  /* Mean log-det(Fisher) across points */
    int    boundary_map_built;   /* 1 if intrinsic-space boundary map was built */
    int    boundary_anchor_points; /* Extreme-point anchors retained in boundary map */
    double boundary_shell_radius;  /* 95th percentile normalized shell radius */
    double boundary_mean_clearance; /* Mean normalized interior clearance */
} axiom_phase3_t;

/* ─── Phase 4 results ─── */
typedef struct {
    int    axiom_count;          /* Minimal axiom set size */
    double consistency_score;    /* Axiom-model agreement [0,1] */
    int    candidates_tested;    /* Total axiom candidates evaluated */
    int    candidates_accepted;  /* Accepted into final set */
    int    oracle_calls_used;    /* Model queries for axiom validation */
    int    model_oracle_calls;   /* Forward-pass oracle calls (token-level) */
    double information_gain;     /* Cumulative active-learning info gain */
} axiom_phase4_t;

/* ─── Phase 5 results ─── */
typedef struct {
    double geodesic_reconstruction_error; /* L2 error vs forward pass */
    double geodesic_cosine_similarity;    /* Cosine sim with forward pass output */
    int    geodesic_steps_taken;
    double geodesic_path_length;
    double transformer_cost;     /* O(n²·d·L) */
    double geodesic_cost;        /* O(n·ID²) */
    double projected_speedup;
    int    geodesic_converged;   /* 1 if RK4 didn't diverge */
    int    pilot_tokens_tested;
    int    geodesic_vocab_probe;
    int    oracle_target_count;
    int    random_target_count;
    int    geodesic_top1_hits;
    double geodesic_top1_match_rate;
    double geodesic_target_mrr;
    int    used_gpu_scoring;
    int    knowledge_injection_applied;
    int    knowledge_injection_points;
    int    warp_points_accumulated;
    double warp_cross_term_estimate;
    int    recalc_triggered;
    int    threshold_profile_steps;
    double threshold_quality;
    double threshold_step_growth;
} axiom_phase5_t;

/* ─── Full report ─── */
typedef struct {
    /* Model context */
    char     model_name[128];
    char     model_arch[64];
    int      model_dim;
    int      model_layers;
    int      model_vocab;
    uint64_t model_params;

    /* Phase results */
    axiom_phase1_t phase1;
    axiom_phase2_t phase2;
    axiom_phase3_t phase3;
    axiom_phase4_t phase4;
    axiom_phase5_t phase5;

    /* Timings (microseconds) */
    uint64_t phase1_us;
    uint64_t phase2_us;
    uint64_t phase3_us;
    uint64_t phase4_us;
    uint64_t phase5_us;
    uint64_t total_us;

    /* Status flags */
    int  uses_real_embeddings;          /* 1 = real model data, 0 = surrogate */
    int  uses_real_curvature;           /* 1 = computed from metric field */
    int  uses_fisher_metric;            /* 1 = Fisher blended into metric */
    int  uses_real_dequant;             /* 1 = dequantized weights for symmetry */
    int  supports_geodesic_pilot;       /* 1 = phase 5 produced results */
    int  reused_geometry_cache;         /* 1 = phases 1-4 reused from in-process cache */
    int  beta_version;                  /* 3 = this version */

    /* OneDecode results */
    int      one_decode_ready;          /* 1 = bake table loaded and usable */
    int      one_decode_table_entries;  /* Vocab entries in the bake table */
    uint64_t one_decode_bake_us;        /* Wall-clock time to build the table (µs) */
} axiom_beta_report_t;

/* ─── API ─── */

void axiom_beta_default_config(axiom_beta_config_t *cfg);

axiom_beta_status_t axiom_beta_run(const axiom_beta_config_t *cfg,
                                   axiom_beta_report_t *report);

/**
 * Return a read-only pointer to the Phase-1 PCA computed by the last
 * axiom_beta_run() call.  Returns NULL if Phase 1 has not completed.
 * Used by axiom_exploit for KV-cache compression projection.
 */
const axpca_t *axiom_beta_get_pca(void);

/**
 * Probe the hidden state of `token_id` at transformer layer `layer`
 * (0 = after embedding, n_layers-1 = last layer output; -1 = sink layer).
 * Writes `dim` floats to `out`.  Returns 0 on success, non-zero on error.
 * Used by axex_compress_model_manifold_layerwise for per-layer PCA.
 */
int axiom_beta_probe_layer_state(int token_id, int layer, float *out, int dim);
/* Probe all layers in a single forward pass. out_per_layer = float[n_layers*dim],
 * out_valid = int[n_layers]. Returns 0 on success, -1 on error. */
int axiom_beta_probe_all_layer_states(int token_id, float *out_per_layer,
                                      int *out_valid, int n_layers, int dim);

/* Probe FFN intermediate activations (SiLU(gate)⊙up) at every layer in one
 * forward pass.  out_per_layer = float[n_layers * ff_dim], out_valid = int[n_layers].
 * ff_dim must be model->ff_dim.  Returns 0 on success, -1 on error. */
int axiom_beta_probe_all_ffn_states(int token_id, float *out_per_layer,
                                    int *out_valid, int n_layers, int ff_dim);

axiom_beta_status_t axiom_beta_write_json(const char *path,
                                          const axiom_beta_report_t *report,
                                          const axiom_beta_config_t *cfg);

axiom_beta_status_t axiom_beta_geodesic_next_token(const int *context_tokens,
                                                   int n_context,
                                                   int *out_token);

axiom_beta_status_t axiom_beta_geodesic_next_token_v2(const int *context_tokens,
                                                      int n_context,
                                                      int *out_token);

const char *axiom_beta_status_string(axiom_beta_status_t st);

/*
 * Fast single geodesic step using cached Phase-3 geometry (computed by a
 * prior axiom_beta_run() call).  Projects the last two context embeddings
 * into the PCA subspace, integrates one geodesic step using the cached
 * Christoffel symbols, reconstructs the predicted embedding, and returns
 * the nearest-token ID and cosine-similarity confidence.
 *
 * Returns AXIOM_BETA_OK on success, AXIOM_BETA_ERR_INVALID when the
 * geometry cache is not ready (caller should fall back to heuristic).
 */
axiom_beta_status_t axiom_beta_geodesic_step_fast(const int *context_tokens,
                                                   int n_context,
                                                   int *out_token,
                                                   float *out_confidence);

/**
 * GRC Online Feedback — call after the transformer rejects a geodesic draft.
 * Inserts a direction hint record from the current context embedding toward
 * the correct token into the GRC library, improving future draft accuracy.
 * Requires a valid Phase-3 geometry cache (from a prior axiom_beta_run()).
 */
axiom_beta_status_t axiom_beta_grc_feedback(const int *context_tokens,
                                             int n_context,
                                             int correct_tok);
int axiom_beta_grc_count(void);

/**
 * Multi-step geodesic rollout — generate up to `n_steps` draft tokens in a
 * single trajectory-coherent integration.  The geodesic velocity is carried
 * forward between steps, so Christoffel corrections accumulate correctly and
 * the draft path follows the true manifold curvature rather than making N
 * independent 1-step corrections.
 *
 * out_tokens[0..n_out-1] and out_conf[0..n_out-1] are filled on success.
 * n_out <= n_steps.  Returns AXIOM_BETA_OK on success.
 * Requires a valid Phase-3 geometry cache (from axiom_beta_run or geometry_load).
 */
axiom_beta_status_t axiom_beta_geodesic_rollout(const int *context_tokens,
                                                 int n_context,
                                                 int n_steps,
                                                 int *out_tokens,
                                                 float *out_conf,
                                                 int *n_out);

/* Return the calibrated acceptance threshold for rollout position `step_index`.
 * Falls back to `base_thresh` when no Phase-3/5 calibration profile is ready. */
float axiom_beta_rollout_threshold(int step_index, float base_thresh);

/* Decode hot-path geodesic hook.
 * axiom_beta_hotpath_ready() reports whether Phase-3 pullback geometry and
 * per-layer Christoffels are available. axiom_beta_hotpath_apply() projects
 * the current residual stream into PCA space, applies a layer-specific
 * Christoffel curvature bias in place, and returns 1 when a correction was
 * applied, 0 otherwise. */
int axiom_beta_hotpath_ready(void);
int axiom_beta_hotpath_apply(float *hidden, int dim, int layer_idx, int token_pos);

/**
 * Save Phase-3 geometry (PCA, metric field, Christoffel symbols) to a binary
 * file.  Eliminates the ~200s Phase-3 recomputation on subsequent runs.
 * Returns AXIOM_BETA_OK on success.
 */
axiom_beta_status_t axiom_beta_geometry_save(const char *path);

/**
 * Load Phase-3 geometry from a file saved by axiom_beta_geometry_save().
 * On success, phase3_geo_valid is set and geodesic_step_fast() / the GRC
 * library work immediately without running axiom_beta_run().
 * Validates that the file was produced for the same model dimension.
 * Returns AXIOM_BETA_OK on success, AXIOM_BETA_ERR_IO if not found/stale.
 */
axiom_beta_status_t axiom_beta_geometry_load(const char *path);

/* ── Todo 27: Context-conditioned hidden state warmup ──────────────────────
 * Call before geodesic generation starts with the full prompt token sequence.
 * Prefills the context through the transformer, snapshots the KV state, and
 * eagerly captures hidden states for the last 8 prompt tokens.
 * Subsequent ott_get_hidden_state() calls will use this snapshot for
 * context-conditioned activations instead of single-token pos=0 probing.
 * Clears automatically at the end of axiom_beta_run(); caller may also clear
 * by calling ott_set_generation_context(NULL, 0).
 */
void ott_set_generation_context(const int *ctx, int n_ctx);

/* Lightweight context sync — called each decode iteration to keep OTT ctx
 * current without a full prefill.  Updates ott_gen_ctx_tokens/ott_gen_ctx_n. */
void ott_update_generation_context(const int *ctx, int n_ctx);

/* ── OneDecode ────────────────────────────────────────────────────────────────
 *
 * Amortises ALL geodesic Christoffel math into a one-time "bake" pass so
 * that every subsequent decode step is O(coverage × pca_dim) — projection
 * plus a nearest-neighbour table lookup, with no RK4 or Christoffel work
 * at decode time.
 *
 * Tradeoff: one-time bake cost (seconds, runs once and caches to disk) in
 * exchange for near-instant per-token prediction during generation.
 *
 * Workflow:
 *   1. axiom_beta_run()            — build Phase-3 geometry
 *   2. axiom_beta_one_decode_load()— try loading ott_one_decode.bin
 *   3. axiom_beta_one_decode_bake()— if (2) failed, bake from scratch
 *   4. axiom_beta_one_decode_save()— persist so (2) succeeds next run
 *   5. axiom_beta_one_decode_next()— fast per-step decode (replaces step_fast)
 */

/* Returns 1 when the bake table is ready (loaded or freshly baked). */
int axiom_beta_one_decode_ready(void);

/**
 * Build the decode map.  For each of `vocab_coverage` vocab tokens (0 =
 * auto-select 2048), project its embedding into PCA space, integrate one
 * Christoffel-corrected geodesic step, and record the nearest vocab token
 * to the predicted endpoint.  Requires valid Phase-3 geometry.
 */
axiom_beta_status_t axiom_beta_one_decode_bake(int vocab_coverage);

/**
 * Serialise the bake table.  Stamped with model identity so a stale file
 * is rejected on load.
 */
axiom_beta_status_t axiom_beta_one_decode_save(const char *path);

/**
 * Load a previously saved bake table.  Returns AXIOM_BETA_ERR_IO when the
 * file is absent or stale (wrong model / PCA dim).
 */
axiom_beta_status_t axiom_beta_one_decode_load(const char *path);

/**
 * Single decode step using the bake table.  Projects the last context
 * token's embedding to PCA space, finds the nearest bake entry by L2
 * distance, and returns its pre-computed predicted token + confidence.
 * Returns AXIOM_BETA_ERR_INVALID when the table is not ready.
 */
axiom_beta_status_t axiom_beta_one_decode_next(const int *context_tokens,
                                                int n_context,
                                                int *out_token,
                                                float *out_confidence);

/** Maximum SWARM fan-out K for axiom_beta_one_decode_topk(). */
#define OD_SWARM_MAX 64

/**
 * OD-SWARM: returns the top k_out nearest-neighbour predicted tokens
 * sorted best-first by cosine similarity (or by logit when primed).
 * k_out is clamped to [1, OD_SWARM_MAX].  out_confidences may be NULL.
 */
axiom_beta_status_t axiom_beta_one_decode_topk(const int *context_tokens,
                                                int n_context,
                                                int *out_tokens,
                                                float *out_confidences,
                                                int k_out);

#endif /* GEODESSICAL_AXIOM_BETA_H */
