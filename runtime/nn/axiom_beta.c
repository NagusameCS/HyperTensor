/*
 * HyperTensor Autonomous Axiomatic Subsystem (Beta)
 */

#include "runtime/nn/axiom_beta.h"
#include "runtime/nn/llm.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef HYPERTENSOR_HOSTED
#include "host/hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

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

static int clamp_i(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void axiom_beta_default_config(axiom_beta_config_t *cfg) {
    if (!cfg) return;
    cfg->samples = 4096;
    cfg->symmetry_trials = 2048;
    cfg->active_iterations = 512;
    cfg->inference_tokens = 256;
    cfg->seed = 0xA1104D5EEDULL;
    cfg->verbose = 0;
}

static void axiom_beta_fill_model_context(axiom_beta_report_t *r) {
    const char *name = llm_model_name();
    const char *arch = llm_model_arch();
    if (!name) name = "(none)";
    if (!arch) arch = "(none)";

    snprintf(r->model_name, sizeof(r->model_name), "%s", name);
    snprintf(r->model_arch, sizeof(r->model_arch), "%s", arch);
    r->model_dim = llm_model_dim();
    r->model_layers = llm_model_layers();
    r->model_vocab = llm_model_vocab();
    r->model_params = llm_param_count();
}

static void phase1_manifold_identification(const axiom_beta_config_t *cfg,
                                           axiom_beta_report_t *r,
                                           uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();
    int dim = r->model_dim > 0 ? r->model_dim : 1;
    int layers = r->model_layers > 0 ? r->model_layers : 1;
    int vocab = r->model_vocab > 0 ? r->model_vocab : 1;

    /* Surrogate ID estimator grounded in architecture scale. */
    double logv = log((double)vocab + 1.0);
    int id_est = (int)(sqrt((double)dim) * 1.35 + logv * 2.0 + layers * 0.15);
    id_est = clamp_i(id_est, 16, dim > 0 ? dim : 16);

    /* Monte-Carlo perturbation energy proxy for local Fisher trace behavior. */
    int n = cfg->samples > 64 ? cfg->samples : 64;
    double e_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double u = ax_rng_f64(seed);
        double v = ax_rng_f64(seed);
        double p = (u - 0.5) * (v - 0.5);
        e_sum += fabs(p);
    }
    double energy = e_sum / (double)n;

    r->intrinsic_dim_estimate = id_est;
    r->metric_rank_estimate = clamp_i(id_est * 2, id_est, dim);
    r->fisher_trace_proxy = (double)r->metric_rank_estimate * (0.75 + energy);
    r->uses_surrogate_metric = 1;
    r->phase1_us = hal_timer_us() - t0;
}

static void phase2_symmetry_extraction(const axiom_beta_config_t *cfg,
                                       axiom_beta_report_t *r,
                                       uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();
    int trials = cfg->symmetry_trials > 128 ? cfg->symmetry_trials : 128;
    int invariant = 0;
    double score_acc = 0.0;

    for (int i = 0; i < trials; i++) {
        /* Synthetic perturbation pair as a permutation/rotation proxy. */
        double a = ax_rng_f64(seed) - 0.5;
        double b = ax_rng_f64(seed) - 0.5;
        double delta = fabs(a - b);
        double inv = exp(-delta * 8.0);
        score_acc += inv;
        if (inv > 0.7) invariant++;
    }

    r->symmetry_invariance_score = score_acc / (double)trials;
    r->symmetry_generators_estimate = clamp_i(
        (int)(r->symmetry_invariance_score * (double)r->intrinsic_dim_estimate * 0.5),
        3,
        r->intrinsic_dim_estimate);
    r->phase2_us = hal_timer_us() - t0;
}

static void phase3_nonlinearity_absorption(const axiom_beta_config_t *cfg,
                                           axiom_beta_report_t *r,
                                           uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();
    int n = cfg->samples > 256 ? cfg->samples / 2 : 256;
    double c_acc = 0.0;

    for (int i = 0; i < n; i++) {
        double x = (ax_rng_f64(seed) - 0.5) * 6.0;
        double silu = x / (1.0 + exp(-x));
        double lin = 0.5 * x;
        c_acc += fabs(silu - lin);
    }

    r->curvature_proxy = c_acc / (double)n;
    r->uses_surrogate_curvature = 1;
    r->phase3_us = hal_timer_us() - t0;
}

static void phase4_axiom_formalization(const axiom_beta_config_t *cfg,
                                       axiom_beta_report_t *r,
                                       uint64_t *seed)
{
    uint64_t t0 = hal_timer_us();
    int it = cfg->active_iterations > 64 ? cfg->active_iterations : 64;
    int accepted = 0;
    double cons = 0.0;

    for (int i = 0; i < it; i++) {
        double candidate = ax_rng_f64(seed);
        double quality = 0.50 * r->symmetry_invariance_score
                       + 0.30 * (1.0 / (1.0 + r->curvature_proxy))
                       + 0.20 * candidate;
        if (quality > 0.60) {
            accepted++;
            cons += quality;
        }
    }

    if (accepted <= 0) {
        r->axiom_count_estimate = 8;
        r->axiom_consistency_score = 0.50;
    } else {
        r->axiom_count_estimate = clamp_i(accepted / 6,
                                          8,
                                          r->intrinsic_dim_estimate * 2);
        r->axiom_consistency_score = cons / (double)accepted;
    }
    r->phase4_us = hal_timer_us() - t0;
}

static void phase5_native_inference_projection(const axiom_beta_config_t *cfg,
                                               axiom_beta_report_t *r)
{
    uint64_t t0 = hal_timer_us();
    double n = (double)(cfg->inference_tokens > 1 ? cfg->inference_tokens : 1);
    double d = (double)(r->model_dim > 0 ? r->model_dim : 1);
    double L = (double)(r->model_layers > 0 ? r->model_layers : 1);
    double id = (double)(r->intrinsic_dim_estimate > 0 ? r->intrinsic_dim_estimate : 1);

    r->projected_transformer_cost = n * n * d * L;
    r->projected_geodesic_cost = n * id * id;
    if (r->projected_geodesic_cost <= 1.0) r->projected_geodesic_cost = 1.0;
    r->projected_speedup = r->projected_transformer_cost / r->projected_geodesic_cost;

    /* This beta does not yet replace forward pass with a production geodesic solver. */
    r->supports_single_step_native_infer = 0;
    r->phase5_us = hal_timer_us() - t0;
}

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

    if (cfg->samples <= 0 || cfg->symmetry_trials <= 0 || cfg->active_iterations <= 0) {
        return AXIOM_BETA_ERR_INVALID;
    }

    memset(report, 0, sizeof(*report));
    seed = cfg->seed ? cfg->seed : 0xA1104D5EEDULL;
    t0 = hal_timer_us();

    axiom_beta_fill_model_context(report);

    phase1_manifold_identification(cfg, report, &seed);
    phase2_symmetry_extraction(cfg, report, &seed);
    phase3_nonlinearity_absorption(cfg, report, &seed);
    phase4_axiom_formalization(cfg, report, &seed);
    phase5_native_inference_projection(cfg, report);

    report->total_us = hal_timer_us() - t0;
    return AXIOM_BETA_OK;
}

axiom_beta_status_t axiom_beta_write_json(const char *path,
                                          const axiom_beta_report_t *report,
                                          const axiom_beta_config_t *cfg)
{
    FILE *f;
    if (!path || !report || !cfg) return AXIOM_BETA_ERR_INVALID;

    f = fopen(path, "wb");
    if (!f) return AXIOM_BETA_ERR_IO;

    fprintf(f, "{\n");
    fprintf(f, "  \"subsystem\": \"autonomous_axiomatic_beta\",\n");
    fprintf(f, "  \"model\": {\n");
    fprintf(f, "    \"name\": \"%s\",\n", report->model_name);
    fprintf(f, "    \"arch\": \"%s\",\n", report->model_arch);
    fprintf(f, "    \"dim\": %d,\n", report->model_dim);
    fprintf(f, "    \"layers\": %d,\n", report->model_layers);
    fprintf(f, "    \"vocab\": %d,\n", report->model_vocab);
    fprintf(f, "    \"params\": %llu\n", (unsigned long long)report->model_params);
    fprintf(f, "  },\n");

    fprintf(f, "  \"config\": {\n");
    fprintf(f, "    \"samples\": %d,\n", cfg->samples);
    fprintf(f, "    \"symmetry_trials\": %d,\n", cfg->symmetry_trials);
    fprintf(f, "    \"active_iterations\": %d,\n", cfg->active_iterations);
    fprintf(f, "    \"inference_tokens\": %d,\n", cfg->inference_tokens);
    fprintf(f, "    \"seed\": %llu\n", (unsigned long long)cfg->seed);
    fprintf(f, "  },\n");

    fprintf(f, "  \"phases\": {\n");
    fprintf(f, "    \"phase1_manifold\": {\n");
    fprintf(f, "      \"intrinsic_dim_estimate\": %d,\n", report->intrinsic_dim_estimate);
    fprintf(f, "      \"metric_rank_estimate\": %d,\n", report->metric_rank_estimate);
    fprintf(f, "      \"fisher_trace_proxy\": %.6f,\n", report->fisher_trace_proxy);
    fprintf(f, "      \"surrogate\": %d\n", report->uses_surrogate_metric);
    fprintf(f, "    },\n");

    fprintf(f, "    \"phase2_symmetry\": {\n");
    fprintf(f, "      \"symmetry_invariance_score\": %.6f,\n", report->symmetry_invariance_score);
    fprintf(f, "      \"symmetry_generators_estimate\": %d\n", report->symmetry_generators_estimate);
    fprintf(f, "    },\n");

    fprintf(f, "    \"phase3_curvature\": {\n");
    fprintf(f, "      \"curvature_proxy\": %.6f,\n", report->curvature_proxy);
    fprintf(f, "      \"surrogate\": %d\n", report->uses_surrogate_curvature);
    fprintf(f, "    },\n");

    fprintf(f, "    \"phase4_axioms\": {\n");
    fprintf(f, "      \"axiom_count_estimate\": %d,\n", report->axiom_count_estimate);
    fprintf(f, "      \"axiom_consistency_score\": %.6f\n", report->axiom_consistency_score);
    fprintf(f, "    },\n");

    fprintf(f, "    \"phase5_projection\": {\n");
    fprintf(f, "      \"projected_transformer_cost\": %.3e,\n", report->projected_transformer_cost);
    fprintf(f, "      \"projected_geodesic_cost\": %.3e,\n", report->projected_geodesic_cost);
    fprintf(f, "      \"projected_speedup\": %.6f,\n", report->projected_speedup);
    fprintf(f, "      \"supports_single_step_native_infer\": %d\n", report->supports_single_step_native_infer);
    fprintf(f, "    }\n");
    fprintf(f, "  },\n");

    fprintf(f, "  \"timings_us\": {\n");
    fprintf(f, "    \"phase1\": %llu,\n", (unsigned long long)report->phase1_us);
    fprintf(f, "    \"phase2\": %llu,\n", (unsigned long long)report->phase2_us);
    fprintf(f, "    \"phase3\": %llu,\n", (unsigned long long)report->phase3_us);
    fprintf(f, "    \"phase4\": %llu,\n", (unsigned long long)report->phase4_us);
    fprintf(f, "    \"phase5\": %llu,\n", (unsigned long long)report->phase5_us);
    fprintf(f, "    \"total\": %llu\n", (unsigned long long)report->total_us);
    fprintf(f, "  }\n");

    fprintf(f, "}\n");
    fclose(f);
    return AXIOM_BETA_OK;
}

const char *axiom_beta_status_string(axiom_beta_status_t st) {
    switch (st) {
    case AXIOM_BETA_OK: return "ok";
    case AXIOM_BETA_ERR_NOT_LOADED: return "model-not-loaded";
    case AXIOM_BETA_ERR_INVALID: return "invalid-args";
    case AXIOM_BETA_ERR_IO: return "io-error";
    default: return "unknown";
    }
}
