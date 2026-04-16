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
} axiom_beta_report_t;

/* ─── API ─── */

void axiom_beta_default_config(axiom_beta_config_t *cfg);

axiom_beta_status_t axiom_beta_run(const axiom_beta_config_t *cfg,
                                   axiom_beta_report_t *report);

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

#endif /* GEODESSICAL_AXIOM_BETA_H */
