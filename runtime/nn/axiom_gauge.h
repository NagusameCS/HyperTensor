/*
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
 * ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
 * ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
 * ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
 * ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
 * ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
 * :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
 * :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
 * ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
 * :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
 * ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
 * ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
 * :::::::::................................:@@@@@@@@@@%:...............................::::::
 * ::::::::..................................*@@@@@@@@@-................................::::::::
 * ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
 * :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
 * :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
 * :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
 * :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
 * :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
 * :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
 * :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
 * :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
 * :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
 * ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
 * ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
 * :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
 * ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
 * :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
 * :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
 * ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
 * ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
 * :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
 * ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
 * ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
 * :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */

/*
 * Gauge-Optimal Residual Stream Compression
 *
 * Theory
 * 
 * Every transformer residual stream has an exact gauge symmetry: any
 * invertible G ∈ GL(d) can be absorbed into the weights with no change
 * to model outputs:
 *
 *   x  ↦  G · x
 *   W_read  ↦  W_read  · G⁻¹     (all attention Q/K/V, FFN gate/up)
 *   W_write ↦  G       · W_write  (attention output O, FFN down)
 *
 * This freedom has a direct bearing on SVD compression: the singular
 * value spectrum of W_read · G⁻¹ (and G · W_write) is NOT the same as
 * that of W unless G is orthogonal.  For a general diagonal gauge
 * G = diag(g), we can jointly minimise the SVD tail energy across ALL
 * weight matrices by choosing the right g.
 *
 * Objective
 * 
 *   min_{g > 0}  Σ_{l, W∈reads_l}  tail_r(W · diag(1/g))
 *              + Σ_{l, W∈writes_l} tail_r(diag(g) · W)
 *
 * where tail_r(M) = ||M||²_F − ||trunc_r(M)||²_F  is the energy not
 * captured by the top-r SVD.
 *
 * Gradient (in log space:  λ = log g,  g = exp(λ))
 * 
 * For a READ matrix W (m  d), with X = W · diag(e^{-λ}):
 *
 *   ∂ tail_r(X) / ∂ λ_i  =  −2  ( ‖X[:,i]‖²  −  Σ_{k≤r} S_k² Vt[k,i]² )
 *                         =  −2  ‖X_tail[:,i]‖²
 *
 * For a WRITE matrix W (d  n), with X = diag(e^{λ}) · W:
 *
 *   ∂ tail_r(X) / ∂ λ_i  =  +2  ( ‖X[i,:]‖²  −  Σ_{k≤r} S_k² U[i,k]² )
 *                         =  +2  ‖X_tail[i,:]‖²
 *
 * Zero-overhead inference
 * 
 * Once the optimal g is found, it is baked into the compressed weight
 * factors before GPU upload — no runtime overhead:
 *
 *   READ  (W→W·diag(1/g)):  d_Vt[k,i] ← S_k · Vt[k,i] · g[i]
 *   WRITE (W→diag(g)·W):    d_U[i,k]  ← U[i,k] / g[i]
 *
 * The GPU two-GEMV kernel (tmp = d_Vt·x; out = d_U·tmp) is unchanged.
 */

#ifndef GEODESSICAL_AXIOM_GAUGE_H
#define GEODESSICAL_AXIOM_GAUGE_H

#include "runtime/nn/llm.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*  Calibration result  */

/*
 * Diagonal gauge vector g[dim].
 *
 * Allocated by axex_gauge_optimize(); caller owns and must free with
 * tensor_free() (or the system free() if tensor_alloc wraps malloc).
 * All entries are strictly positive.  The vector is normalised so that
 * the geometric mean equals 1.0 (no net scale change on the residual
 * stream).
 */
typedef struct {
    float   *g;          /* diagonal gauge: g[0..dim-1] > 0             */
    int      dim;        /* residual stream dimension                    */
    double   tail_before; /* total tail energy before optimisation       */
    double   tail_after;  /* total tail energy after  optimisation       */
    float    gain_pct;   /* percentage improvement in tail energy        */
} axex_gauge_t;

void axex_gauge_free(axex_gauge_t *gauge);

/*  Main optimisation entry point  */

/*
 * Find the optimal diagonal gauge for compressing the model's residual
 * stream at the given SVD rank budget.
 *
 * model:   loaded LLM model (weights must be dequantisable via
 *          ax_dequant_row_f32; may be any quantisation).
 * rank:    target SVD rank per weight matrix (same as --axex-compress-rank).
 * n_iter:  number of gradient-descent iterations in log-space (10–30
 *          is typically sufficient; set 0 for the free analytic step).
 *
 * The function processes ONLY the matrices involved in SVD compression:
 *   READ  (compress as W · diag(1/g)): Q, K, V, FFN_gate, FFN_up
 *   WRITE (compress as diag(g) · W):   O, FFN_down
 *
 * Matrices that are NOT compressed (e.g. embedding) are ignored.
 *
 * Returns a freshly-allocated axex_gauge_t on success, NULL on failure.
 */
axex_gauge_t *axex_gauge_optimize(const llm_model_t *model,
                                   int rank, int n_iter);

/*  Per-matrix gauge application (called inside the compression loop)  */

/*
 * Pre-multiply columns of a READ matrix W [m  dim] by 1/g:
 *   W[:,i] *= 1.0f / g[i]
 * In-place.  Call this before axex_weight_compress() for Q,K,V,gate,up.
 */
void axex_gauge_apply_read(float *W, int m, int dim, const float *g);

/*
 * Pre-multiply rows of a WRITE matrix W [dim  n] by g:
 *   W[i,:] *= g[i]
 * In-place.  Call this before axex_weight_compress() for O,down.
 */
void axex_gauge_apply_write(float *W, int dim, int n, const float *g);

/*
 * Post-SVD: bake g into the right singular factor of a READ matrix.
 * After compressing X = W·diag(1/g) we have Vt [rank  dim].
 * To reconstruct W from (U,S,Vt_baked) we need:
 *   W · x = U · S · Vt · diag(g) · x
 * So scale Vt column-wise:  Vt[k,i] *= g[i]
 * Call this BEFORE axex_upload_compressed_to_gpu().
 * (axex_upload_compressed_to_gpu will then further bake S, yielding
 *  d_Vt[k,i] = S[k] * Vt[k,i] * g[i].)
 */
void axex_gauge_bake_vt(float *Vt, int rank, int dim, const float *g);

/*
 * Post-SVD: bake 1/g into the left singular factor of a WRITE matrix.
 * After compressing X = diag(g)·W we have U [dim  rank].
 * To reconstruct W from (U_baked,S,Vt) we need:
 *   W · x = diag(1/g) · U · S · Vt · x
 * So scale U row-wise:  U[i,k] /= g[i]
 * Call this BEFORE axex_upload_compressed_to_gpu().
 */
void axex_gauge_bake_u(float *U, int dim, int rank, const float *g);

#ifdef __cplusplus
}
#endif

#endif /* GEODESSICAL_AXIOM_GAUGE_H */
