/*
 * HyperTensor Autonomous Axiomatic Subsystem (Beta)
 *
 * This subsystem builds a model-centric geometric report in five phases:
 *  1) manifold identification
 *  2) symmetry extraction
 *  3) nonlinearity/curvature absorption proxy
 *  4) axiom-set formalization
 *  5) native-inference complexity projection
 *
 * The current implementation is intentionally conservative and reports
 * validated surrogate metrics while hidden-state and Jacobian probes are
 * integrated incrementally.
 */

#ifndef HYPERTENSOR_AXIOM_BETA_H
#define HYPERTENSOR_AXIOM_BETA_H

#include <stdint.h>

typedef enum {
    AXIOM_BETA_OK = 0,
    AXIOM_BETA_ERR_NOT_LOADED = -1,
    AXIOM_BETA_ERR_INVALID = -2,
    AXIOM_BETA_ERR_IO = -3,
} axiom_beta_status_t;

typedef struct {
    int      samples;            /* phase-1 survey samples */
    int      symmetry_trials;    /* phase-2 random invariance probes */
    int      active_iterations;  /* phase-4 axiom candidate iterations */
    int      inference_tokens;   /* phase-5 cost projection token count */
    uint64_t seed;               /* deterministic seed */
    int      verbose;            /* print phase diagnostics */
} axiom_beta_config_t;

typedef struct {
    /* model context */
    char model_name[128];
    char model_arch[64];
    int  model_dim;
    int  model_layers;
    int  model_vocab;
    uint64_t model_params;

    /* phase outputs */
    int   intrinsic_dim_estimate;
    int   metric_rank_estimate;
    double fisher_trace_proxy;
    double curvature_proxy;
    double symmetry_invariance_score;
    int   symmetry_generators_estimate;
    int   axiom_count_estimate;
    double axiom_consistency_score;

    /* complexity projection */
    double projected_transformer_cost;
    double projected_geodesic_cost;
    double projected_speedup;

    /* timings */
    uint64_t phase1_us;
    uint64_t phase2_us;
    uint64_t phase3_us;
    uint64_t phase4_us;
    uint64_t phase5_us;
    uint64_t total_us;

    /* notes */
    int uses_surrogate_metric;
    int uses_surrogate_curvature;
    int supports_single_step_native_infer;
} axiom_beta_report_t;

void axiom_beta_default_config(axiom_beta_config_t *cfg);

axiom_beta_status_t axiom_beta_run(const axiom_beta_config_t *cfg,
                                   axiom_beta_report_t *report);

axiom_beta_status_t axiom_beta_write_json(const char *path,
                                          const axiom_beta_report_t *report,
                                          const axiom_beta_config_t *cfg);

const char *axiom_beta_status_string(axiom_beta_status_t st);

#endif /* HYPERTENSOR_AXIOM_BETA_H */
