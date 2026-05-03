/**
 * hypercore.h — HyperTensor Core Algorithms in C
 * 
 * Production-ready C implementations of proven geometric compression
 * algorithms from the HyperTensor research framework. Designed for
 * direct integration into the geodessical2 inference runtime.
 * 
 * Modules:
 *   - GRC Projection (Paper I): Attention weight compression via joint SVD
 *   - Sink Detection (Paper IV): High-norm channel identification
 *   - Grassmann Distance (Paper X): Subspace similarity metric
 *   - GTC Caching (Paper VIII): Geodesic trajectory lookup
 *   - Cluster Compression (Paper VII): FFN per-cluster SVD
 *
 * Author: NagusameCS / HyperTensor
 * License: MIT
 */

#ifndef HYPERTENSOR_HYPERCORE_H
#define HYPERTENSOR_HYPERCORE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Configuration Types
 * ======================================================================== */

/** Model configuration for dimension-aware algorithms */
typedef struct {
    int32_t d;           /* Hidden dimension (e.g., 576 for SmolLM2-135M) */
    int32_t d_kv;        /* KV dimension (for GQA: d_kv < d) */
    int32_t d_ffn;       /* FFN intermediate dimension */
    int32_t n_layers;    /* Number of transformer layers */
    int32_t n_heads;     /* Number of attention heads */
    int32_t n_kv_heads;  /* Number of KV heads */
    float   eps;         /* Numerical stability epsilon */
} HCModelConfig;

/** Compression parameters */
typedef struct {
    int32_t k;           /* Target rank */
    int32_t sink_T;      /* Number of sink channels to protect */
    int32_t n_clusters;  /* Number of FFN clusters */
    float   k_frac;      /* Fraction of rank to keep per cluster */
    bool    use_gauge;    /* Apply gauge alignment */
    bool    protect_sinks;/* Exempt sink channels from projection */
} HCCompressParams;

/** Splice result metrics */
typedef struct {
    float gd_pre;        /* Grassmann distance before gauge */
    float gd_post;       /* Grassmann distance after gauge */
    float overlap;       /* Subspace overlap fraction */
    float rho_ceci;      /* CECI LoRA recoverability */
    float q_err;         /* Q projection relative error */
    float k_err;         /* K projection relative error */
    float v_err;         /* V projection relative error */
    float signal_kept;   /* Fraction of signal preserved */
    bool  viable;        /* Meets viability threshold */
    int32_t sinks_shared;/* Number of shared sink channels */
} HCSpliceResult;

/** GTC trajectory record */
typedef struct {
    float  *embedding;   /* Trajectory embedding (d floats) */
    float  *logits;      /* Terminal logits (vocab_size floats) */
    float   radius;      /* Geodesic validity radius */
    int32_t token_count; /* Number of tokens in trajectory */
    int32_t *tokens;     /* Token IDs */
    int32_t vocab_size;  /* Vocabulary size */
} HCTrajectory;

/** GTC cache */
typedef struct {
    HCTrajectory *entries;
    int32_t       capacity;
    int32_t       count;
    float         semantic_radius;  /* Max cosine distance for hit */
} HCGTC cache;

/* ========================================================================
 * GRC Projection (Paper I)
 * ======================================================================== */

/**
 * Build shared GRC basis from concatenated Q, K, V weights.
 * 
 * @param Wq    Q weight matrix (d  d, row-major)
 * @param Wk    K weight matrix (d  d, row-major)
 * @param Wv    V weight matrix (d  d, row-major)
 * @param d     Hidden dimension
 * @param k     Target rank (output)
 * @param P     Output: projection basis (d  k, row-major, column-orthonormal)
 * @param S_out Output: singular values (k floats)
 * @param work  Workspace buffer (size: d*d*3 + d*3 floats)
 * @return      Fraction of total signal preserved (0.0-1.0)
 */
float hc_grc_build_basis(const float *Wq, const float *Wk, const float *Wv,
                         int32_t d, int32_t k, float *P, float *S_out,
                         float *work);

/**
 * Apply GRC projection to a weight matrix.
 * W_proj = W @ P @ P^T
 * 
 * @param W       Weight matrix (d  d, row-major)
 * @param P       Projection basis (d  k, column-orthonormal)
 * @param d       Hidden dimension
 * @param k       Projection rank
 * @param W_out   Output: projected weight (d  d, row-major)
 * @param work    Workspace (d*k + k*k floats)
 */
void hc_grc_project(const float *W, const float *P, int32_t d, int32_t k,
                    float *W_out, float *work);

/**
 * Apply GRC with sink protection.
 * Sink channels (top-T highest-norm) are EXEMPT from projection.
 * 
 * @return Signal preservation fraction
 */
float hc_grc_project_sink_protected(const float *Wq, const float *Wk,
                                     const float *Wv, float *P, int32_t d,
                                     int32_t k, int32_t sink_T,
                                     float *Wq_out, float *Wk_out, float *Wv_out,
                                     float *work);

/* ========================================================================
 * Sink Detection (Paper IV)
 * ======================================================================== */

/**
 * Identify top-T sink channels by combined Q+K+V column norms.
 * Sinks are channels with unusually high L2 norm that encode 
 * positional/default information — these should be exempt from projection.
 * 
 * @param Wq, Wk, Wv  Weight matrices
 * @param d            Dimension
 * @param T            Number of sinks to identify
 * @param sink_indices Output: indices of sink channels (size T)
 */
void hc_find_sinks(const float *Wq, const float *Wk, const float *Wv,
                   int32_t d, int32_t T, int32_t *sink_indices);

/* ========================================================================
 * Grassmann Distance (Paper X)
 * ======================================================================== */

/**
 * Compute Grassmann distance between two k-dimensional subspaces.
 * GD(U, V) = ||U U^T - V V^T||_F / sqrt(2k)
 * 
 * 0.0 = identical subspaces, 1.0 = completely orthogonal.
 * 
 * @param U    Basis A (d  k, column-orthonormal)
 * @param V    Basis B (d  k, column-orthonormal)
 * @param d    Ambient dimension
 * @param k    Subspace dimension
 * @param work Workspace (k*k + d*d floats)
 * @return     Grassmann distance [0.0, 1.0]
 */
float hc_grassmann_distance(const float *U, const float *V,
                            int32_t d, int32_t k, float *work);

/**
 * Subspace overlap: fraction of U's variance captured by V's subspace.
 * overlap = ||V V^T U||_F^2 / ||U||_F^2
 */
float hc_subspace_overlap(const float *U, const float *V,
                          int32_t d, int32_t k, float *work);

/**
 * Full CECI splice measurement.
 * Computes all splice metrics between two sets of attention weights.
 */
HCSpliceResult hc_ceci_splice_measure(
    const float *Wq_a, const float *Wk_a, const float *Wv_a,
    const float *Wq_b, const float *Wk_b, const float *Wv_b,
    int32_t d, int32_t k, int32_t sink_T, bool apply_gauge,
    float *work);

/* ========================================================================
 * GTC Caching (Paper VIII)
 * ======================================================================== */

/**
 * Initialize GTC cache.
 */
HCGTCCache *hc_gtc_cache_create(int32_t capacity, int32_t d, int32_t vocab_size);

/**
 * Free GTC cache.
 */
void hc_gtc_cache_free(HCGTCCache *cache);

/**
 * Query GTC cache for a matching trajectory.
 * Uses brute-force cosine similarity (FAISS integration planned).
 * 
 * @return Index of best match, or -1 if no hit within semantic_radius.
 */
int32_t hc_gtc_query(const HCGTCCache *cache, const float *query_embedding,
                     float *similarity_out);

/**
 * Insert trajectory into GTC cache.
 * Returns true if inserted (doesn't exceed capacity).
 */
bool hc_gtc_insert(HCGTCCache *cache, const float *embedding,
                   const float *logits, const int32_t *tokens,
                   int32_t token_count, float radius);

/* ========================================================================
 * FFN Cluster Compression (Paper VII)
 * ======================================================================== */

/**
 * Cluster FFN columns by L2 similarity and compress via per-cluster SVD.
 * 
 * @param gate     Gate projection (d  d_ffn, row-major)
 * @param up       Up projection (d  d_ffn, row-major)
 * @param down     Down projection (d_ffn  d, row-major)
 * @param clusters Number of clusters
 * @param k_frac   Fraction of rank to preserve per cluster
 * @param gate_out, up_out, down_out  Output compressed weights
 * @param work     Workspace (large: ~d*d_ffn*2 + d*d floats)
 * @return         Average signal preservation across clusters
 */
float hc_ffn_cluster_compress(const float *gate, const float *up, const float *down,
                              int32_t d, int32_t d_ffn,
                              int32_t clusters, float k_frac,
                              float *gate_out, float *up_out, float *down_out,
                              float *work);

/* ========================================================================
 * Workspace Size Helpers
 * ======================================================================== */

/** Required workspace for hc_grc_build_basis */
static inline size_t hc_grc_workspace_bytes(int32_t d) {
    return (size_t)(d * d * 3 + d * 3) * sizeof(float);
}

/** Required workspace for hc_grassmann_distance */
static inline size_t hc_gd_workspace_bytes(int32_t d, int32_t k) {
    return (size_t)(k * k + d * d) * sizeof(float);
}

/** Required workspace for hc_ceci_splice_measure */
static inline size_t hc_ceci_workspace_bytes(int32_t d, int32_t k) {
    return (size_t)(d * d * 6 + d * k * 4 + k * k * 2) * sizeof(float);
}

/** Required workspace for hc_ffn_cluster_compress */
static inline size_t hc_ffn_workspace_bytes(int32_t d, int32_t d_ffn) {
    return (size_t)(d * d_ffn * 3 + d * d * 2) * sizeof(float);
}

#ifdef __cplusplus
}
#endif

#endif /* HYPERTENSOR_HYPERCORE_H */
