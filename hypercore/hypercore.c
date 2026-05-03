/**
 * hypercore.c — HyperTensor Core Algorithms Implementation
 * 
 * Pure C, no external dependencies beyond libc and LAPACK (for SVD).
 * Designed for embedded inference runtime integration.
 * 
 * Compile: gcc -O3 -shared -o libhypercore.so hypercore.c -llapacke -lopenblas
 *     or: cl /O2 /LD hypercore.c /link openblas.lib
 */

#include "hypercore.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Use LAPACK via LAPACKE for SVD */
#include <lapacke.h>

/* ========================================================================
 * Internal BLAS-level helpers
 * ======================================================================== */

/* Compute column L2 norms: norms[j] = ||A[:,j]||_2 */
static void col_norms(const float *A, int32_t rows, int32_t cols,
                      float *norms) {
    for (int32_t j = 0; j < cols; j++) {
        float sum = 0.0f;
        for (int32_t i = 0; i < rows; i++) {
            float a = A[i + j * rows];  /* Column-major */
            sum += a * a;
        }
        norms[j] = sqrtf(sum + 1e-10f);
    }
}

/* Matrix multiply C = A @ B where A(mk), B(kn), C(mn) row-major */
static void matmul(const float *A, const float *B, float *C,
                   int32_t m, int32_t k, int32_t n) {
    /* C = A * B, all row-major: C[i][j] = sum_l A[i][l] * B[l][j] */
    memset(C, 0, m * n * sizeof(float));
    for (int32_t i = 0; i < m; i++) {
        for (int32_t l = 0; l < k; l++) {
            float ail = A[i * k + l];
            for (int32_t j = 0; j < n; j++) {
                C[i * n + j] += ail * B[l * n + j];
            }
        }
    }
}

/* Frobenius norm */
static float frob_norm(const float *A, int32_t rows, int32_t cols) {
    float sum = 0.0f;
    for (int32_t i = 0; i < rows * cols; i++)
        sum += A[i] * A[i];
    return sqrtf(sum);
}

/* ========================================================================
 * GRC: Build Shared Basis (Paper I)
 * ======================================================================== */

float hc_grc_build_basis(const float *Wq, const float *Wk, const float *Wv,
                         int32_t d, int32_t k, float *P, float *S_out,
                         float *work) {
    /* Step 1: Concatenate Q, K, V into M = [Wq; Wk; Wv] (3d  d, col-major for LAPACKE) */
    int32_t m = 3 * d;  /* rows of M */
    int32_t n = d;       /* cols of M */
    
    /* Copy weights into column-major M */
    /* Wq: dd row-major → col-major: M[i + j*m] = Wq[i*d + j] */
    float *M = work;  /* 3d * d floats */
    float *U_svd = M + m * n;  /* 3d * d floats */
    float *S = U_svd + m * n;  /* min(m,n) floats = d floats */
    float *Vt = S + n;          /* d * d floats */
    float *superb = Vt + n * n; /* min(m,n)-1 floats */
    
    for (int32_t j = 0; j < d; j++) {
        for (int32_t i = 0; i < d; i++) {
            M[i + j * m]           = Wq[i * d + j];  /* Q weights */
            M[i + d + j * m]       = Wk[i * d + j];  /* K weights */
            M[i + 2*d + j * m]     = Wv[i * d + j];  /* V weights */
        }
    }
    
    /* Step 2: SVD via LAPACKE (dgesvd) */
    lapack_int info = LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'S', 'S',
                                      m, n, M, m, S, U_svd, m, Vt, n, superb);
    
    if (info != 0) {
        /* SVD failed — return identity projection */
        memset(P, 0, d * k * sizeof(float));
        for (int32_t i = 0; i < k && i < d; i++) P[i * d + i] = 1.0f;
        if (S_out) memset(S_out, 0, k * sizeof(float));
        return 0.0f;
    }
    
    /* Step 3: Extract top-k right singular vectors as projection basis */
    /* Vt is (d  d) in row-major from LAPACKE's perspective (col-major storage) */
    /* Vt[l + j*d] = V[j][l] in row-major => V[l][j] in column-major */
    /* We want P = V[:, 0:k] = first k rows of Vt (as stored) */
    
    for (int32_t j = 0; j < d; j++) {
        for (int32_t i = 0; i < k; i++) {
            P[j + i * d] = Vt[i + j * d];  /* Row-major: P[j][i] = Vt[i][j] */
        }
    }
    
    /* Step 4: Signal preservation */
    float total_signal = 0.0f;
    for (int32_t i = 0; i < d; i++) total_signal += S[i] * S[i];
    float kept_signal = 0.0f;
    for (int32_t i = 0; i < k && i < d; i++) kept_signal += S[i] * S[i];
    
    if (S_out) memcpy(S_out, S, (k < d ? k : d) * sizeof(float));
    
    return total_signal > 0.0f ? kept_signal / total_signal : 1.0f;
}

/* ========================================================================
 * GRC: Project Weight Matrix
 * ======================================================================== */

void hc_grc_project(const float *W, const float *P, int32_t d, int32_t k,
                    float *W_out, float *work) {
    /* W_proj = W @ P @ P^T */
    float *temp = work;  /* d * k */
    
    /* temp = W @ P  (dd @ dk → dk) */
    matmul(W, P, temp, d, d, k);
    
    /* W_out = temp @ P^T (dk @ kd → dd) */
    /* P^T is (kd): P^T[i][j] = P[j][i] */
    float *Pt = work + d * k;
    for (int32_t i = 0; i < k; i++)
        for (int32_t j = 0; j < d; j++)
            Pt[i + j * k] = P[j + i * d];
    
    matmul(temp, Pt, W_out, d, k, d);
}

/* ========================================================================
 * GRC: Sink-Protected Projection
 * ======================================================================== */

float hc_grc_project_sink_protected(
    const float *Wq, const float *Wk, const float *Wv,
    float *P, int32_t d, int32_t k, int32_t sink_T,
    float *Wq_out, float *Wk_out, float *Wv_out,
    float *work) {
    
    /* Build basis on sink-removed weights */
    float *Wq_sinkless = work;
    float *Wk_sinkless = Wq_sinkless + d * d;
    float *Wv_sinkless = Wk_sinkless + d * d;
    
    memcpy(Wq_sinkless, Wq, d * d * sizeof(float));
    memcpy(Wk_sinkless, Wk, d * d * sizeof(float));
    memcpy(Wv_sinkless, Wv, d * d * sizeof(float));
    
    /* Find and zero out sink channels */
    int32_t *sinks = (int32_t *)(Wv_sinkless + d * d);
    hc_find_sinks(Wq, Wk, Wv, d, sink_T, sinks);
    
    /* Zero sink columns in copied weights (column-major: column j = A[:,j]) */
    /* Row-major: A[i][j] = A[i*d + j], column j = all i: A[i*d + j] */
    for (int32_t t = 0; t < sink_T; t++) {
        int32_t j = sinks[t];
        for (int32_t i = 0; i < d; i++) {
            Wq_sinkless[i * d + j] = 0.0f;
            Wk_sinkless[i * d + j] = 0.0f;
            Wv_sinkless[i * d + j] = 0.0f;
        }
    }
    
    /* Build basis from sinkless weights */
    float *basis_work = (float *)(sinks + sink_T);
    float *P_sinkless = basis_work + hc_grc_workspace_bytes(d) / sizeof(float);
    float signal = hc_grc_build_basis(Wq_sinkless, Wk_sinkless, Wv_sinkless,
                                       d, k, P_sinkless, NULL, basis_work);
    
    /* Apply projection, then restore sink channels */
    float *proj_work = basis_work;  /* Reuse workspace */
    hc_grc_project(Wq, P_sinkless, d, k, Wq_out, proj_work);
    hc_grc_project(Wk, P_sinkless, d, k, Wk_out, proj_work);
    hc_grc_project(Wv, P_sinkless, d, k, Wv_out, proj_work);
    
    /* Restore sink channels from original weights */
    for (int32_t t = 0; t < sink_T; t++) {
        int32_t j = sinks[t];
        for (int32_t i = 0; i < d; i++) {
            Wq_out[i * d + j] = Wq[i * d + j];
            Wk_out[i * d + j] = Wk[i * d + j];
            Wv_out[i * d + j] = Wv[i * d + j];
        }
    }
    
    return signal;
}

/* ========================================================================
 * Sink Detection (Paper IV)
 * ======================================================================== */

void hc_find_sinks(const float *Wq, const float *Wk, const float *Wv,
                   int32_t d, int32_t T, int32_t *sink_indices) {
    float *norms = (float *)malloc(d * sizeof(float));
    memset(norms, 0, d * sizeof(float));
    
    /* Combined column norm across Q, K, V */
    for (int32_t j = 0; j < d; j++) {
        float nq = 0.0f, nk = 0.0f, nv = 0.0f;
        for (int32_t i = 0; i < d; i++) {
            nq += Wq[i * d + j] * Wq[i * d + j];
            nk += Wk[i * d + j] * Wk[i * d + j];
            nv += Wv[i * d + j] * Wv[i * d + j];
        }
        norms[j] = nq + nk + nv;
    }
    
    /* Top-T selection (simple partial sort) */
    for (int32_t t = 0; t < T && t < d; t++) {
        float max_norm = -1.0f;
        int32_t max_idx = 0;
        for (int32_t j = 0; j < d; j++) {
            if (norms[j] > max_norm) {
                max_norm = norms[j];
                max_idx = j;
            }
        }
        sink_indices[t] = max_idx;
        norms[max_idx] = -1.0f;  /* Mark as taken */
    }
    
    free(norms);
}

/* ========================================================================
 * Grassmann Distance (Paper X)
 * ======================================================================== */

float hc_grassmann_distance(const float *U, const float *V,
                            int32_t d, int32_t k, float *work) {
    /* GD = ||U U^T - V V^T||_F / sqrt(2k) */
    
    float *UU = work;       /* d  d */
    float *VV = UU + d * d; /* d  d */
    
    /* UU = U @ U^T */
    matmul(U, U, UU, d, k, d);  /* Wait, U is dk, U^T is kd */
    /* Actually: U (dk), U^T (kd) → we need a temporary */
    float *Ut = (float *)malloc(d * k * sizeof(float));
    for (int32_t i = 0; i < k; i++)
        for (int32_t j = 0; j < d; j++)
            Ut[i + j * k] = U[j + i * d];
    matmul(U, Ut, UU, d, k, d);
    
    /* VV = V @ V^T */
    for (int32_t i = 0; i < k; i++)
        for (int32_t j = 0; j < d; j++)
            Ut[i + j * k] = V[j + i * d];
    matmul(V, Ut, VV, d, k, d);
    free(Ut);
    
    /* Diff = UU - VV */
    float diff_norm = 0.0f;
    for (int32_t i = 0; i < d * d; i++) {
        float diff = UU[i] - VV[i];
        diff_norm += diff * diff;
    }
    
    return sqrtf(diff_norm) / sqrtf(2.0f * k);
}

float hc_subspace_overlap(const float *U, const float *V,
                          int32_t d, int32_t k, float *work) {
    /* overlap = ||V V^T U||_F^2 / ||U||_F^2 */
    float *Vt = (float *)malloc(d * k * sizeof(float));
    for (int32_t i = 0; i < k; i++)
        for (int32_t j = 0; j < d; j++)
            Vt[i + j * k] = V[j + i * d];
    
    float *temp = (float *)malloc(d * k * sizeof(float));
    matmul(V, Vt, temp, d, k, d);  /* V @ V^T (dd) */
    
    /* temp2 = (V @ V^T) @ U (dk) */
    float *temp2 = (float *)malloc(d * k * sizeof(float));
    matmul(temp, U, temp2, d, d, k);
    
    float proj_norm = frob_norm(temp2, d, k);
    float u_norm = frob_norm(U, d, k);
    
    free(Vt); free(temp); free(temp2);
    
    return (u_norm > 1e-10f) ? (proj_norm * proj_norm) / (u_norm * u_norm) : 0.0f;
}

/* ========================================================================
 * Full CECI Splice Measurement
 * ======================================================================== */

HCSpliceResult hc_ceci_splice_measure(
    const float *Wq_a, const float *Wk_a, const float *Wv_a,
    const float *Wq_b, const float *Wk_b, const float *Wv_b,
    int32_t d, int32_t k, int32_t sink_T, bool apply_gauge,
    float *work) {
    
    HCSpliceResult result = {0};
    
    /* Build bases */
    float *Pa = work;
    float *Pb = Pa + d * k;
    float *grc_work = Pb + d * k;
    
    hc_grc_build_basis(Wq_a, Wk_a, Wv_a, d, k, Pa, NULL, grc_work);
    hc_grc_build_basis(Wq_b, Wk_b, Wv_b, d, k, Pb, NULL, grc_work);
    
    /* Pre-gauge Grassmann distance */
    float *gd_work = grc_work;
    result.gd_pre = hc_grassmann_distance(Pa, Pb, d, k, gd_work);
    result.overlap = hc_subspace_overlap(Pa, Pb, d, k, gd_work);
    
    /* Post-gauge (simplified: no actual gauge — just measure) */
    result.gd_post = result.gd_pre;
    
    /* Splice residual measurement */
    float *Wq_proj = (float *)malloc(d * d * sizeof(float));
    float *Wk_proj = (float *)malloc(d * d * sizeof(float));
    float *Wv_proj = (float *)malloc(d * d * sizeof(float));
    float *proj_work = (float *)malloc(d * k * 2 * sizeof(float));
    
    hc_grc_project(Wq_b, Pa, d, k, Wq_proj, proj_work);
    hc_grc_project(Wk_b, Pa, d, k, Wk_proj, proj_work);
    hc_grc_project(Wv_b, Pa, d, k, Wv_proj, proj_work);
    
    /* Relative errors */
    float norm_q = frob_norm(Wq_b, d, d);
    float norm_k = frob_norm(Wk_b, d, d);
    float norm_v = frob_norm(Wv_b, d, d);
    
    for (int32_t i = 0; i < d * d; i++) {
        result.q_err += (Wq_proj[i] - Wq_b[i]) * (Wq_proj[i] - Wq_b[i]);
        result.k_err += (Wk_proj[i] - Wk_b[i]) * (Wk_proj[i] - Wk_b[i]);
        result.v_err += (Wv_proj[i] - Wv_b[i]) * (Wv_proj[i] - Wv_b[i]);
    }
    
    result.q_err = sqrtf(result.q_err) / (norm_q + 1e-10f);
    result.k_err = sqrtf(result.k_err) / (norm_k + 1e-10f);
    result.v_err = sqrtf(result.v_err) / (norm_v + 1e-10f);
    
    /* Simplified rho_ceci (no actual LoRA SVD — placeholder) */
    float mean_err = (result.q_err + result.k_err + result.v_err) / 3.0f;
    result.rho_ceci = 1.0f / (1.0f + mean_err);
    
    /* Viability: GD < 0.90 AND rho > 0.30 */
    result.viable = (result.gd_pre < 0.90f) && (result.rho_ceci > 0.30f);
    result.signal_kept = 1.0f / (1.0f + mean_err);
    
    free(Wq_proj); free(Wk_proj); free(Wv_proj); free(proj_work);
    
    return result;
}

/* ========================================================================
 * GTC Cache (Paper VIII)
 * ======================================================================== */

HCGTCCache *hc_gtc_cache_create(int32_t capacity, int32_t d, int32_t vocab_size) {
    HCGTCCache *cache = (HCGTCCache *)malloc(sizeof(HCGTCCache));
    if (!cache) return NULL;
    
    cache->capacity = capacity;
    cache->count = 0;
    cache->semantic_radius = 0.05f;
    cache->entries = (HCTrajectory *)calloc(capacity, sizeof(HCTrajectory));
    
    if (!cache->entries) { free(cache); return NULL; }
    
    /* Pre-allocate all embedding/logit storage together */
    float *embedding_pool = (float *)calloc(capacity * d, sizeof(float));
    float *logits_pool = (float *)calloc(capacity * vocab_size, sizeof(float));
    int32_t *tokens_pool = (int32_t *)calloc(capacity * 32, sizeof(int32_t));  /* Up to 32 tokens per trajectory */
    
    for (int32_t i = 0; i < capacity; i++) {
        cache->entries[i].embedding = embedding_pool + i * d;
        cache->entries[i].logits = logits_pool + i * vocab_size;
        cache->entries[i].tokens = tokens_pool + i * 32;
        cache->entries[i].vocab_size = vocab_size;
    }
    
    return cache;
}

void hc_gtc_cache_free(HCGTCCache *cache) {
    if (!cache) return;
    if (cache->entries) {
        if (cache->entries[0].embedding) free(cache->entries[0].embedding);
        if (cache->entries[0].logits) free(cache->entries[0].logits);
        if (cache->entries[0].tokens) free(cache->entries[0].tokens);
        free(cache->entries);
    }
    free(cache);
}

int32_t hc_gtc_query(const HCGTCCache *cache, const float *query_embedding,
                     float *similarity_out) {
    float best_sim = -1.0f;
    int32_t best_idx = -1;
    
    for (int32_t i = 0; i < cache->count; i++) {
        /* Cosine similarity */
        float dot = 0.0f, norm_q = 0.0f, norm_e = 0.0f;
        const float *emb = cache->entries[i].embedding;
        for (int32_t j = 0; j < 32; j++) {  /* Simplified: first 32 dims */
            dot += query_embedding[j] * emb[j];
            norm_q += query_embedding[j] * query_embedding[j];
        }
        /* Assume normalized embeddings for speed */
        float sim = dot;  /* Normalized assumption */
        if (sim > best_sim) {
            best_sim = sim;
            best_idx = i;
        }
    }
    
    if (similarity_out) *similarity_out = best_sim;
    
    /* Hit if within semantic radius: 1-cos < radius */
    return (1.0f - best_sim < cache->semantic_radius) ? best_idx : -1;
}

bool hc_gtc_insert(HCGTCCache *cache, const float *embedding,
                   const float *logits, const int32_t *tokens,
                   int32_t token_count, float radius) {
    if (cache->count >= cache->capacity) return false;
    
    int32_t i = cache->count;
    memcpy(cache->entries[i].embedding, embedding, 32 * sizeof(float));
    memcpy(cache->entries[i].logits, logits, cache->entries[i].vocab_size * sizeof(float));
    memcpy(cache->entries[i].tokens, tokens, token_count * sizeof(int32_t));
    cache->entries[i].token_count = token_count;
    cache->entries[i].radius = radius;
    cache->count++;
    
    return true;
}

/* ========================================================================
 * FFN Cluster Compression (Paper VII)
 * ======================================================================== */

float hc_ffn_cluster_compress(const float *gate, const float *up, const float *down,
                              int32_t d, int32_t d_ffn,
                              int32_t clusters, float k_frac,
                              float *gate_out, float *up_out, float *down_out,
                              float *work) {
    /* Simplified: apply global SVD with k=k_frac * min(d, d_ffn) */
    /* Full cluster-SVD requires per-column clustering — see Python implementation */
    
    int32_t k = (int32_t)(k_frac * (d < d_ffn ? d : d_ffn));
    if (k < 1) k = 1;
    
    /* Apply to gate matrix (d  d_ffn) */
    int32_t m = d, n = d_ffn;
    float *M = work;  /* Copy for LAPACK */
    memcpy(M, gate, d * d_ffn * sizeof(float));
    float *U = M + m * n;
    float *S = U + m * n;
    float *Vt = S + (m < n ? m : n);
    float *superb = Vt + n * n;
    
    LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, M, m, S, U, m, Vt, n, superb);
    
    /* Reconstruct gate with rank k */
    memset(gate_out, 0, d * d_ffn * sizeof(float));
    for (int32_t i = 0; i < m; i++)
        for (int32_t l = 0; l < k; l++)
            for (int32_t j = 0; j < n; j++)
                gate_out[i * n + j] += U[i + l * m] * S[l] * Vt[l + j * n];
    
    /* Same for up */
    memcpy(M, up, d * d_ffn * sizeof(float));
    LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'S', 'S', m, n, M, m, S, U, m, Vt, n, superb);
    memset(up_out, 0, d * d_ffn * sizeof(float));
    for (int32_t i = 0; i < m; i++)
        for (int32_t l = 0; l < k; l++)
            for (int32_t j = 0; j < n; j++)
                up_out[i * n + j] += U[i + l * m] * S[l] * Vt[l + j * n];
    
    /* Down: transposed (d_ffn  d) */
    memcpy(M, down, d_ffn * d * sizeof(float));
    LAPACKE_sgesvd(LAPACK_COL_MAJOR, 'S', 'S', d_ffn, d, M, d_ffn, U, d_ffn, Vt, d, superb);
    memset(down_out, 0, d_ffn * d * sizeof(float));
    int32_t m_dn = d_ffn, n_dn = d;
    for (int32_t i = 0; i < m_dn; i++)
        for (int32_t l = 0; l < k; l++)
            for (int32_t j = 0; j < n_dn; j++)
                down_out[i * n_dn + j] += U[i + l * m_dn] * S[l] * Vt[l + j * n_dn];
    
    /* Signal preservation */
    float total = 0.0f, kept = 0.0f;
    for (int32_t i = 0; i < (m < n ? m : n); i++) total += S[i] * S[i];
    for (int32_t i = 0; i < k; i++) kept += S[i] * S[i];
    
    return total > 0.0f ? kept / total : 1.0f;
}
