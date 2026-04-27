/*
 * mcr_compress.c - Mix-Compress-Refine phase detection & attention-sink bypass
 *
 * See mcr_compress.h for design documentation.
 */

#include "runtime/nn/mcr_compress.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#ifdef GEODESSICAL_HOSTED
#  include "host/hal.h"
#  define MCR_PRINTF kprintf
#else
#  include "kernel/core/kernel.h"
#  define MCR_PRINTF kprintf
#endif

/* =========================================================================
 * Feature 1: MCR phase detection
 * ========================================================================= */

int mcr_detect_phases(mcr_result_t *r,
                      const mcr_layer_stats_t *stats,
                      int n_layers,
                      float compress_thr)
{
    if (!r || !stats || n_layers < 2) return -1;
    if (n_layers > MCR_MAX_LAYERS) n_layers = MCR_MAX_LAYERS;
    if (compress_thr < 1.0f) compress_thr = 1.0f;

    memset(r, 0, sizeof(*r));
    r->n_layers = n_layers;

    /* 3-tap moving-average of activation variance */
    for (int l = 0; l < n_layers; l++) {
        float prev = (l > 0)          ? stats[l - 1].act_variance : stats[l].act_variance;
        float curr =                    stats[l].act_variance;
        float next = (l < n_layers-1) ? stats[l + 1].act_variance : stats[l].act_variance;
        r->smoothed_var[l] = (prev + curr + next) * (1.0f / 3.0f);
    }

    /* Find global minimum and maximum */
    r->var_min = FLT_MAX;
    r->var_max = -FLT_MAX;
    int l_min = 0;
    for (int l = 0; l < n_layers; l++) {
        if (r->smoothed_var[l] < r->var_min) {
            r->var_min = r->smoothed_var[l];
            l_min = l;
        }
        if (r->smoothed_var[l] > r->var_max)
            r->var_max = r->smoothed_var[l];
    }

    /*
     * If the profile is too flat (ratio < 1.15×) there are no meaningful
     * phases — treat all layers as Mix and recommend uniform rank.
     */
    float ratio = (r->var_min > 1e-15f) ? (r->var_max / r->var_min) : 1.0f;
    if (ratio < 1.15f) {
        for (int l = 0; l < n_layers; l++) r->phase[l] = MCR_PHASE_MIX;
        r->mix_end        = n_layers - 1;
        r->compress_start = n_layers;
        r->compress_end   = n_layers - 1;
        r->refine_start   = n_layers;
        r->phases_valid   = 0;
        return 0;
    }
    r->phases_valid = 1;

    /*
     * Expand the Compress zone outward from the minimum-variance layer,
     * including all layers within compress_thr × var_min.
     */
    float threshold = r->var_min * compress_thr;
    int cs = l_min, ce = l_min;
    while (cs > 0          && r->smoothed_var[cs - 1] <= threshold) cs--;
    while (ce < n_layers-1 && r->smoothed_var[ce + 1] <= threshold) ce++;

    r->compress_start = cs;
    r->compress_end   = ce;
    r->mix_end        = cs - 1;
    r->refine_start   = ce + 1;

    /* Assign phase labels */
    for (int l = 0; l < n_layers; l++) {
        if      (l < cs)  r->phase[l] = MCR_PHASE_MIX;
        else if (l <= ce) r->phase[l] = MCR_PHASE_COMPRESS;
        else              r->phase[l] = MCR_PHASE_REFINE;
    }
    return 0;
}

void mcr_rank_budget(const mcr_result_t *r,
                     int total_budget,
                     int min_rank, int max_rank,
                     float mix_scale, float compress_scale, float refine_scale,
                     int *ranks_out)
{
    if (!r || !ranks_out || r->n_layers < 1) return;
    int n = r->n_layers;

    if (!r->phases_valid || total_budget <= 0) {
        int base = (total_budget > 0) ? (total_budget / n) : min_rank;
        if (base < min_rank) base = min_rank;
        if (base > max_rank) base = max_rank;
        for (int l = 0; l < n; l++) ranks_out[l] = base;
        return;
    }

    float base  = (float)total_budget / (float)n;
    float raw[MCR_MAX_LAYERS];
    float sum   = 0.0f;

    for (int l = 0; l < n; l++) {
        float scale;
        switch (r->phase[l]) {
            case MCR_PHASE_MIX:      scale = mix_scale;      break;
            case MCR_PHASE_COMPRESS: scale = compress_scale; break;
            case MCR_PHASE_REFINE:   scale = refine_scale;   break;
            default:                 scale = 1.0f;           break;
        }
        raw[l] = base * scale;
        sum   += raw[l];
    }

    /* Normalize so the sum stays at total_budget, then clamp */
    float nf = (sum > 0.0f) ? ((float)total_budget / sum) : 1.0f;
    for (int l = 0; l < n; l++) {
        int rv = (int)(raw[l] * nf + 0.5f);
        if (rv < min_rank) rv = min_rank;
        if (rv > max_rank) rv = max_rank;
        ranks_out[l] = rv;
    }
}

void mcr_print(const mcr_result_t *r)
{
    if (!r) return;
    MCR_PRINTF("[MCR] n_layers=%d  phases_valid=%d  var_ratio=%.2f\n",
               r->n_layers, r->phases_valid,
               (double)((r->var_min > 1e-15f) ? r->var_max / r->var_min : 1.0f));
    if (r->phases_valid) {
        MCR_PRINTF("[MCR] Mix:      layers   0 .. %3d  (%d layers)\n",
                   r->mix_end,
                   (r->mix_end >= 0) ? (r->mix_end + 1) : 0);
        MCR_PRINTF("[MCR] Compress: layers %3d .. %3d  (%d layers, var_min=%.5f)\n",
                   r->compress_start, r->compress_end,
                   r->compress_end - r->compress_start + 1,
                   (double)r->var_min);
        MCR_PRINTF("[MCR] Refine:   layers %3d .. %3d  (%d layers)\n",
                   r->refine_start, r->n_layers - 1,
                   r->n_layers - r->refine_start);
    } else {
        MCR_PRINTF("[MCR] Flat activation-variance profile — uniform rank recommended\n");
    }
}

/* =========================================================================
 * Feature 2: Attention-sink bypass
 * ========================================================================= */

int sink_detect(sink_ctx_t *ctx, const float *norms, int n, float sigma_threshold)
{
    if (!ctx || !norms || n < 2) return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->sigma_threshold = sigma_threshold;

    /* Compute mean and variance */
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; i++) {
        sum  += (double)norms[i];
        sum2 += (double)norms[i] * (double)norms[i];
    }
    ctx->norm_mean = (float)(sum / n);
    float var      = (float)(sum2 / n - (sum / n) * (sum / n));
    ctx->norm_std  = (var > 0.0f) ? sqrtf(var) : 0.0f;

    float hi_thr = ctx->norm_mean + sigma_threshold * ctx->norm_std;
    float lo_thr = ctx->norm_mean + 1.5f * ctx->norm_std;

    for (int i = 0; i < n && ctx->n_sinks < SINK_MAX_ITEMS; i++) {
        int is_bos = (i == 0 && norms[0] >= lo_thr);
        if (norms[i] >= hi_thr || is_bos) {
            ctx->indices[ctx->n_sinks] = i;
            ctx->norms[ctx->n_sinks]   = norms[i];
            ctx->n_sinks++;
        }
    }
    ctx->valid = 1;
    return 0;
}

int sink_check_basis_coverage(const sink_ctx_t *ctx,
                               const float  *mean_sink_hs,
                               const double *basis,
                               int n_comp, int dim,
                               float cos_threshold,
                               float *extra_dir)
{
    if (!ctx || !mean_sink_hs || !basis || n_comp < 1 || dim < 1) return -1;
    if (!ctx->valid || ctx->n_sinks == 0) return 0;

    /* Compute L2 norm of mean sink hidden state */
    double norm_sq = 0.0;
    for (int i = 0; i < dim; i++)
        norm_sq += (double)mean_sink_hs[i] * (double)mean_sink_hs[i];
    if (norm_sq < 1e-20) return 0;
    double inv_norm = 1.0 / sqrt(norm_sq);

    /* For each PCA basis vector, compute |d · b_i| where d is the normalised
     * mean sink direction */
    float max_cos = 0.0f;
    for (int c = 0; c < n_comp; c++) {
        const double *b = basis + (size_t)c * dim;
        double dot = 0.0;
        for (int i = 0; i < dim; i++)
            dot += (double)mean_sink_hs[i] * inv_norm * b[i];
        float cos_val = (float)fabs(dot);
        if (cos_val > max_cos) max_cos = cos_val;
    }

    if (max_cos >= cos_threshold) return 0; /* adequately covered */

    /* Fill extra_dir with the normalised sink direction */
    if (extra_dir) {
        float inv_f = (float)inv_norm;
        for (int i = 0; i < dim; i++)
            extra_dir[i] = mean_sink_hs[i] * inv_f;
    }
    return 1; /* sink direction missing from basis */
}

void sink_print(const sink_ctx_t *ctx)
{
    if (!ctx || !ctx->valid) return;
    MCR_PRINTF("[SINK] mean_norm=%.3f  std=%.3f  sigma_thr=%.1f  n_sinks=%d\n",
               (double)ctx->norm_mean, (double)ctx->norm_std,
               (double)ctx->sigma_threshold, ctx->n_sinks);
    int show = ctx->n_sinks < 8 ? ctx->n_sinks : 8;
    for (int i = 0; i < show; i++)
        MCR_PRINTF("[SINK]   idx=%d  norm=%.3f\n",
                   ctx->indices[i], (double)ctx->norms[i]);
    if (ctx->n_sinks > 8)
        MCR_PRINTF("[SINK]   ... (%d more)\n", ctx->n_sinks - 8);
}
