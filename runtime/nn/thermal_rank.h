/*
 * thermal_rank.h - Thermal-budget-adaptive compression rank (Feature 3)
 *                  Tokens-per-joule as first-class objective  (Feature 4)
 *
 * Feature 3: thermal-adaptive rank
 *   throttLL'eM (HPCA 2025) scales GPU *clock frequency* to save energy but
 *   causes a throughput cliff when the device thermal-throttles.  This module
 *   takes a different approach: keep the clock constant and scale the manifold
 *   *compression rank* according to thermal headroom.  Higher rank means more
 *   FLOPs, more heat, lower throughput; lower rank means less.  Laptops and
 *   edge devices thermal-throttle within ~30 s of sustained load; an adaptive-
 *   rank runtime absorbs the thermal budget without the cliff.
 *
 *   Implementation: polls NVIDIA NVML (loaded dynamically at run-time) for
 *   current GPU temperature and power draw, then linearly interpolates rank
 *   between [rank_min, rank_max] over [temp_low, temp_high].  Falls back
 *   gracefully if NVML is unavailable.
 *
 * Feature 4: tokens-per-joule as diffplan objective
 *   Current diffplan (geo_research.h) minimises reconstruction error + L1 rank.
 *   That optimises accuracy-at-fixed-rank, not accuracy-at-fixed-joules.
 *   This module adds a measured-joules gradient term to the diffplan theta
 *   update so that rank allocation is explicitly penalised for energy cost.
 *   The joule estimate comes from NVML power readings divided by measured
 *   tokens/s.  The gradient is a policy-gradient update on top of the
 *   softmax-parameterised rank plan.
 */

#ifndef GEODESSICAL_THERMAL_RANK_H
#define GEODESSICAL_THERMAL_RANK_H

#include "runtime/nn/geo_research.h"  /* diffplan_t, DIFFPLAN_MAX_LAYERS, DIFFPLAN_N_LEVELS */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Feature 3: Thermal-adaptive rank
 * ========================================================================= */

typedef struct {
    /* ── configuration ─────────────────────────────────────────── */
    float   temp_low_C;       /* below this, use rank_max   (default 65 °C) */
    float   temp_high_C;      /* above this, use rank_min   (default 85 °C) */
    float   power_budget_W;   /* target power cap in watts; 0 = no limit    */
    int     rank_min;         /* minimum compression rank allowed            */
    int     rank_max;         /* maximum compression rank allowed            */
    int     poll_interval_ms; /* re-poll GPU every this many ms (default 500)*/

    /* ── current readings (updated by thermal_poll) ─────────────── */
    float   current_temp_C;
    float   current_power_W;
    int     nvml_ok;          /* 1 = NVML is available and initialised       */

    /* ── internal (opaque) ──────────────────────────────────────── */
    uint64_t last_poll_us;    /* timestamp of last successful poll           */
    void    *_nvml_lib;       /* platform DLL / SO handle                    */
    void    *_nvml_device;    /* nvmlDevice_t handle                         */
} thermal_ctx_t;

/*
 * Initialise the thermal context.  Attempts to load NVML dynamically.
 * If NVML is unavailable, ctx->nvml_ok is set to 0 and thermal_get_rank()
 * always returns base_rank unchanged.
 *
 * temp_low_C / temp_high_C : thermal thresholds in °C
 * power_budget_W           : power cap in watts (0 = no limit)
 * rank_min / rank_max      : compression rank bounds for scaling
 */
int  thermal_init(thermal_ctx_t *ctx,
                  float temp_low_C, float temp_high_C,
                  float power_budget_W,
                  int rank_min, int rank_max);

/* Release NVML handles and DLL. */
void thermal_destroy(thermal_ctx_t *ctx);

/*
 * Poll GPU temperature and power draw.  Respects poll_interval_ms cooldown.
 * Returns 0 if readings were refreshed, 1 if still within cooldown, -1 on error.
 */
int  thermal_poll(thermal_ctx_t *ctx);

/*
 * Return a rank scaled by current temperature.
 * Linear interpolation: temp_low → rank_max, temp_high → rank_min.
 * Also considers power_budget_W if set.
 * Returns base_rank unchanged when nvml_ok == 0.
 */
int  thermal_get_rank(thermal_ctx_t *ctx, int base_rank);

void thermal_print(const thermal_ctx_t *ctx);

/* =========================================================================
 * Feature 4: Tokens-per-joule objective
 * ========================================================================= */

#define TPJ_HISTORY_LEN 256

typedef struct {
    thermal_ctx_t *thermal;                  /* borrowed (must outlive tpj_ctx_t) */
    float          lambda;                   /* energy regularisation weight       */
    float          rank_coeff;               /* estimated joules / unit-rank / tok */
    float          joules_history[TPJ_HISTORY_LEN];
    int            n_history;
    double         cumulative_joules;
    int            cumulative_tokens;
} tpj_ctx_t;

/*
 * Initialise the TPJ context.
 * thermal: pointer to an already-initialised thermal_ctx_t.  May be NULL
 *          (all joule measurements will return 0, gradient is a no-op).
 * lambda:  energy regularisation weight (e.g. 0.005).
 */
int   tpj_init(tpj_ctx_t *ctx, thermal_ctx_t *thermal, float lambda);

/*
 * Record one observation at the given tokens/s rate.
 * Converts current_power_W / tps → joules/token, stores in history, and
 * updates the rolling rank_coeff estimate.
 * Returns the estimated joules/token for this observation.
 */
float tpj_record(tpj_ctx_t *ctx, float tokens_per_second);

/*
 * Compute and ADD the energy-penalty gradient to grads[][].
 *
 * The contribution is:
 *   grads[l][r] += lambda * rank_coeff * p[l][r] * (RANK_LEVELS[r] - rank_soft[l])
 *
 * This is the policy-gradient term that pushes the diffplan softmax towards
 * lower-rank configurations when rank_coeff (joules/rank) is high.
 *
 * grads[DIFFPLAN_MAX_LAYERS][DIFFPLAN_N_LEVELS]: accumulator (caller may initialise
 *   to zero or pass an already-populated gradient for accumulation)
 * dp:       current diffplan (reads p[][], rank_soft[], rank_hard[])
 * n_layers: number of active layers
 */
void  tpj_gradient(const tpj_ctx_t *ctx,
                   float grads[][DIFFPLAN_N_LEVELS],
                   int n_layers,
                   const diffplan_t *dp);

/*
 * Pre-calibrate rank_coeff from a rough tokens/s estimate and current GPU
 * power draw.  Intended for use right after tpj_init(), before any observed
 * data, so that tpj_gradient() produces a non-trivial regularisation signal
 * from the very first diffplan step.
 *
 * The estimate: rank_coeff ≈ (power_W / tps_estimate) / RANK_LEVELS[mid]
 *
 * If NVML is unavailable or power reading is zero the call is a no-op.
 */
void  tpj_bootstrap(tpj_ctx_t *ctx, float tps_estimate);

void  tpj_print(const tpj_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GEODESSICAL_THERMAL_RANK_H */
