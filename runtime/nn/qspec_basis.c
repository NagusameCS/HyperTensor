/*
 * qspec_basis.c - Cross-quantization shared basis and failure-mode rank allocation.
 *
 * See qspec_basis.h for design documentation.
 */

#include "runtime/nn/qspec_basis.h"
#include "runtime/nn/axiom_exploit.h"
#include "runtime/nn/geo_research.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#ifdef GEODESSICAL_HOSTED
#  include "host/hal.h"
#  define QB_PRINTF kprintf
#else
#  include "kernel/core/kernel.h"
#  define QB_PRINTF kprintf
#endif

/* =========================================================================
 * Feature 5: Cross-quantization shared basis test
 * ========================================================================= */

int qspec_test_shared_basis(qspec_result_t *result, int n_layers, float share_threshold)
{
    if (!result || n_layers <= 0) return -1;
    if (n_layers > QSPEC_MAX_LAYERS) n_layers = QSPEC_MAX_LAYERS;

    memset(result, 0, sizeof(*result));
    result->share_threshold = share_threshold;
    result->min_alignment   = FLT_MAX;

    double sum_align = 0.0;
    int    n_valid   = 0;

    /*
     * Iterate every (layer, slot) pair.
     * Slot numbering from axiom_exploit.h:
     *   0 = FFN_down,  1 = Q,  2 = K,  3 = V,  4 = O,
     *   5 = FFN_up,    6 = FFN_gate
     * We skip slot 0 (FFN_down vanilla SVD) as it rarely has a manifold
     * counterpart; slots 1-6 cover the attention and FFN_up/gate pairs.
     */
    for (int l = 0; l < n_layers; l++) {
        for (int s = 1; s < QSPEC_MAX_SLOTS; s++) {
            const axex_compressed_weight_t *cw = axex_get_compressed_layer(l, s);
            const axex_manifold_weight_t   *mw = axex_get_manifold_layer(l, s);
            if (!cw || !mw) continue;

            float frob = cw->frobenius_err;
            float pe   = mw->proj_energy;

            /* Clamp frob to [0, 1] to guard against un-initialised slots */
            if (frob < 0.0f) frob = 0.0f;
            if (frob > 1.0f) frob = 1.0f;

            float svd_expl = 1.0f - frob * frob;
            if (svd_expl < 1e-6f) svd_expl = 1e-6f; /* avoid divide-by-zero */

            /*
             * Alignment: how similar are the two energy fractions?
             * Use min/max ratio ∈ [0, 1] so the metric is symmetric and
             * bounded.  1.0 = both fractions identical; 0.0 = one is zero
             * while the other is non-zero.
             */
            float lo = (pe < svd_expl) ? pe : svd_expl;
            float hi = (pe > svd_expl) ? pe : svd_expl;
            float align = (hi > 1e-6f) ? (lo / hi) : 0.0f;

            if (result->n_entries >= QSPEC_MAX_LAYERS * QSPEC_MAX_SLOTS) break;
            qspec_entry_t *e = &result->entries[result->n_entries++];
            e->layer        = l;
            e->slot         = s;
            e->proj_energy  = pe;
            e->svd_explained = svd_expl;
            e->alignment    = align;
            e->shared_ok    = (align >= share_threshold) ? 1 : 0;

            sum_align += (double)align;
            n_valid++;
            if (e->alignment < result->min_alignment) result->min_alignment = e->alignment;
            if (e->shared_ok) result->n_shared_ok++;
        }
    }

    result->mean_alignment = (n_valid > 0) ? (float)(sum_align / n_valid) : 0.0f;
    if (result->min_alignment == FLT_MAX) result->min_alignment = 0.0f;
    return n_valid;
}

void qspec_print(const qspec_result_t *result)
{
    if (!result) return;
    QB_PRINTF("[QSPEC] n_evaluated=%d  shared_ok=%d  mean_align=%.3f  min_align=%.3f\n",
              result->n_entries, result->n_shared_ok,
              (double)result->mean_alignment, (double)result->min_alignment);
    QB_PRINTF("[QSPEC] threshold=%.2f  => %s\n",
              (double)result->share_threshold,
              (result->n_entries > 0 &&
               (float)result->n_shared_ok / (float)result->n_entries >= 0.8f)
                  ? "basis likely transferable across quant levels"
                  : "basis may need recomputation per quant level");

    /* Show worst layers */
    int worst_l = -1, worst_s = -1;
    float worst_a = FLT_MAX;
    for (int i = 0; i < result->n_entries; i++) {
        if (result->entries[i].alignment < worst_a) {
            worst_a = result->entries[i].alignment;
            worst_l = result->entries[i].layer;
            worst_s = result->entries[i].slot;
        }
    }
    if (worst_l >= 0)
        QB_PRINTF("[QSPEC] worst: layer=%d slot=%d align=%.3f\n",
                  worst_l, worst_s, (double)worst_a);
}

/* =========================================================================
 * Feature 6: Failure-mode-targeted rank allocation
 * ========================================================================= */

static const char *fmode_name(fail_mode_t m)
{
    switch (m) {
        case FMODE_FACTUAL:   return "FACTUAL (early layers)";
        case FMODE_REASONING: return "REASONING (mid layers)";
        case FMODE_COHERENCE: return "COHERENCE (late layers)";
        case FMODE_CONTEXT:   return "CONTEXT (all layers)";
        default:              return "NONE";
    }
}

int frank_build(frank_result_t *r,
                const float *frob_err, int n_layers,
                float dominant_boost, float decay)
{
    if (!r || !frob_err || n_layers < 2) return -1;
    if (n_layers > DIFFPLAN_MAX_LAYERS) n_layers = DIFFPLAN_MAX_LAYERS;
    if (dominant_boost < 1.0f) dominant_boost = 1.8f;
    if (decay < 0.0f || decay > 1.0f) decay = 0.6f;

    memset(r, 0, sizeof(*r));
    r->n_layers = n_layers;

    /* Partition into three zones: [0, n/3), [n/3, 2n/3), [2n/3, n) */
    int b1 = n_layers / 3;
    int b2 = 2 * n_layers / 3;

    double e_sum = 0.0, m_sum = 0.0, l_sum = 0.0, g_sum = 0.0;
    int    e_cnt = 0,   m_cnt = 0,   l_cnt = 0;

    for (int l = 0; l < n_layers; l++) {
        float e = frob_err[l];
        if (e < 0.0f) e = 0.0f;
        g_sum += e;
        if      (l < b1) { e_sum += e; e_cnt++; }
        else if (l < b2) { m_sum += e; m_cnt++; }
        else             { l_sum += e; l_cnt++; }
    }

    r->early_err  = e_cnt ? (float)(e_sum / e_cnt) : 0.0f;
    r->mid_err    = m_cnt ? (float)(m_sum / m_cnt) : 0.0f;
    r->late_err   = l_cnt ? (float)(l_sum / l_cnt) : 0.0f;
    r->global_err = (float)(g_sum / n_layers);

    /*
     * Determine dominant failure mode.
     * "Uniform" = all three zone means within 20% of the global mean.
     */
    float tol = r->global_err * 0.20f;
    int uniform = (fabsf(r->early_err - r->global_err) < tol) &&
                  (fabsf(r->mid_err   - r->global_err) < tol) &&
                  (fabsf(r->late_err  - r->global_err) < tol);

    if (uniform && r->global_err > 0.05f) {
        r->dominant_mode = FMODE_CONTEXT;
    } else if (r->early_err >= r->mid_err && r->early_err >= r->late_err) {
        r->dominant_mode = FMODE_FACTUAL;
    } else if (r->mid_err >= r->early_err && r->mid_err >= r->late_err) {
        r->dominant_mode = FMODE_REASONING;
    } else {
        r->dominant_mode = FMODE_COHERENCE;
    }

    /* Assign per-layer rank scales */
    for (int l = 0; l < n_layers; l++) {
        int in_zone;
        switch (r->dominant_mode) {
            case FMODE_FACTUAL:   in_zone = (l < b1);           break;
            case FMODE_REASONING: in_zone = (l >= b1 && l < b2); break;
            case FMODE_COHERENCE: in_zone = (l >= b2);           break;
            case FMODE_CONTEXT:   in_zone = 1;                   break;
            default:              in_zone = 0;                   break;
        }

        if (in_zone) {
            r->rank_scale[l] = dominant_boost;
        } else {
            /*
             * Adjacent zones get a partial boost to avoid abrupt rank cliffs.
             * We decay by `decay` per zone boundary away from the dominant zone.
             */
            int zone_of_l = (l < b1) ? 0 : (l < b2) ? 1 : 2;
            int dom_zone;
            switch (r->dominant_mode) {
                case FMODE_FACTUAL:   dom_zone = 0; break;
                case FMODE_REASONING: dom_zone = 1; break;
                default:              dom_zone = 2; break;
            }
            int dist = abs(zone_of_l - dom_zone);
            float s  = dominant_boost;
            for (int d = 0; d < dist; d++) s = 1.0f + (s - 1.0f) * decay;
            r->rank_scale[l] = (s < 1.0f) ? 1.0f : s;
        }
    }

    r->valid = 1;
    return 0;
}

void frank_apply(const frank_result_t *r, int *ranks, int n_layers,
                 int min_rank, int max_rank)
{
    if (!r || !ranks || !r->valid) return;
    int n = n_layers < r->n_layers ? n_layers : r->n_layers;
    for (int l = 0; l < n; l++) {
        int rv = (int)((float)ranks[l] * r->rank_scale[l] + 0.5f);
        if (rv < min_rank) rv = min_rank;
        if (rv > max_rank) rv = max_rank;
        ranks[l] = rv;
    }
}

void frank_print(const frank_result_t *r)
{
    if (!r || !r->valid) return;
    QB_PRINTF("[FRANK] n_layers=%d  dominant_mode=%s\n",
              r->n_layers, fmode_name(r->dominant_mode));
    QB_PRINTF("[FRANK] frob_err: early=%.4f  mid=%.4f  late=%.4f  global=%.4f\n",
              (double)r->early_err, (double)r->mid_err,
              (double)r->late_err,  (double)r->global_err);

    /* Show rank_scale range */
    float smin = FLT_MAX, smax = -FLT_MAX;
    for (int l = 0; l < r->n_layers; l++) {
        if (r->rank_scale[l] < smin) smin = r->rank_scale[l];
        if (r->rank_scale[l] > smax) smax = r->rank_scale[l];
    }
    QB_PRINTF("[FRANK] rank_scale: min=%.2f  max=%.2f\n",
              (double)smin, (double)smax);
    QB_PRINTF("[FRANK] => concentrate rank budget on %s to suppress failure\n",
              fmode_name(r->dominant_mode));
}
