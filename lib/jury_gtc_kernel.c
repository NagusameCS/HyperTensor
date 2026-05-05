/* jury_gtc_kernel.c — C implementation of Jury-GTC search.
 *
 * COMPILE:
 *   Linux:   gcc -O3 -march=native -shared -fPIC -o libjury_gtc.so jury_gtc_kernel.c -lm
 *   Windows: gcc -O3 -shared -o jury_gtc.dll jury_gtc_kernel.c -lm
 *   macOS:   gcc -O3 -shared -o libjury_gtc.dylib jury_gtc_kernel.c -lm
 *
 * This implements the TWO critical bottlenecks:
 *   1. jury_search — Two-stage jury routing + domain search
 *   2. batch_similarity — Batched cosine similarity (BLAS replacement)
 *
 * For maximum performance, link against OpenBLAS or MKL:
 *   gcc -O3 -shared -fPIC -o libjury_gtc.so jury_gtc_kernel.c -lopenblas -lm
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── Utility: L2 normalize in-place ── */
static void normalize(float *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)v[i] * v[i];
    float norm = (float)(1.0 / sqrt(sum + 1e-12));
    for (int i = 0; i < n; i++) v[i] *= norm;
}

/* ── Utility: cosine similarity ── */
static float cosine_sim(const float *a, const float *b, int k) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < k; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(na * nb) + 1e-12));
}

/* ── Utility: softmax in-place ── */
static void softmax(float *v, int n, float temp) {
    float max_val = v[0];
    for (int i = 1; i < n; i++) if (v[i] > max_val) max_val = v[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        v[i] = expf((v[i] - max_val) * temp);
        sum += v[i];
    }
    float inv_sum = (float)(1.0 / (sum + 1e-12));
    for (int i = 0; i < n; i++) v[i] *= inv_sum;
}

/* ═══════════════════════════════════════════════════════════════════
 * KERNEL 1: jury_search — Two-stage Jury-GTC lookup
 *
 * Parameters:
 *   pool       [in]  float[N * K]  — trajectory pool (row-major)
 *   domains    [in]  int[N]        — domain labels per trajectory
 *   query      [in]  float[K]      — query vector (normalized)
 *   N          [in]  int           — pool size
 *   K          [in]  int           — embedding dimension
 *   sample_n   [in]  int           — jury sample size (e.g., 20)
 *   temperature[in]  float         — softmax temperature
 *   threshold  [in]  float         — hit threshold
 *   best_idx   [out] int*          — best trajectory index
 *   best_sim   [out] float*        — best similarity score
 *   comparisons[out] int*          — number of comparisons made
 *   dominant   [out] int*          — dominant domain index
 *
 * Returns: 1 if hit (best_sim >= threshold), 0 if miss
 * ═══════════════════════════════════════════════════════════════════ */
int jury_search(
    const float *pool, const int *domains,
    const float *query, int N, int K,
    int sample_n, float temperature, float threshold,
    int *best_idx, float *best_sim, int *comparisons, int *dominant)
{
    *best_idx = -1;
    *best_sim = -1.0f;
    *comparisons = 0;
    *dominant = -1;
    
    if (N <= 0 || K <= 0) return 0;
    
    /* ── Stage 1: Jury routing on random sample ── */
    int actual_sample = (sample_n < N) ? sample_n : N;
    
    /* Use deterministic stride sampling instead of random for speed */
    int stride = N / actual_sample;
    if (stride < 1) stride = 1;
    
    /* Allocate temp storage for sample similarities and weights */
    float *sample_sims = (float*)malloc(actual_sample * sizeof(float));
    if (!sample_sims) return 0;
    
    /* Compute similarities to sampled trajectories */
    for (int i = 0; i < actual_sample; i++) {
        int idx = i * stride;
        if (idx >= N) idx = N - 1;
        sample_sims[i] = cosine_sim(query, pool + idx * K, K);
    }
    *comparisons += actual_sample;
    
    /* Softmax to get domain weights */
    softmax(sample_sims, actual_sample, temperature);
    
    /* Aggregate weights by domain (max 64 unique domains) */
    float domain_w[64] = {0};
    int domain_ids[64];
    int n_domains = 0;
    
    for (int i = 0; i < actual_sample; i++) {
        int idx = i * stride;
        if (idx >= N) idx = N - 1;
        int d = domains[idx];
        
        /* Find or add domain */
        int found = 0;
        for (int j = 0; j < n_domains; j++) {
            if (domain_ids[j] == d) {
                domain_w[j] += sample_sims[i];
                found = 1;
                break;
            }
        }
        if (!found && n_domains < 64) {
            domain_ids[n_domains] = d;
            domain_w[n_domains] = sample_sims[i];
            n_domains++;
        }
    }
    free(sample_sims);
    
    /* Find top-2 domains */
    int top_d1 = 0, top_d2 = -1;
    float top_w1 = -1.0f, top_w2 = -1.0f;
    for (int i = 0; i < n_domains; i++) {
        if (domain_w[i] > top_w1) {
            top_w2 = top_w1; top_d2 = top_d1;
            top_w1 = domain_w[i]; top_d1 = i;
        } else if (domain_w[i] > top_w2) {
            top_w2 = domain_w[i]; top_d2 = i;
        }
    }
    
    int search_domains[2];
    search_domains[0] = domain_ids[top_d1];
    search_domains[1] = (top_d2 >= 0) ? domain_ids[top_d2] : search_domains[0];
    *dominant = search_domains[0];
    
    /* ── Stage 2: Domain search with early exit ── */
    for (int di = 0; di < 2; di++) {
        int target_domain = search_domains[di];
        
        for (int i = 0; i < N; i++) {
            if (domains[i] != target_domain) continue;
            
            float sim = cosine_sim(query, pool + i * K, K);
            (*comparisons)++;
            
            if (sim > *best_sim) {
                *best_sim = sim;
                *best_idx = i;
            }
            
            /* Early exit: near-perfect match */
            if (*best_sim >= 0.995f) {
                return (*best_sim >= threshold) ? 1 : 0;
            }
        }
        
        /* Early exit: found good enough match in first domain */
        if (*best_sim >= threshold && di == 0) {
            return 1;
        }
    }
    
    return (*best_sim >= threshold) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * KERNEL 2: batch_cosine — Batched cosine similarity
 *
 * Computes similarities between M queries and N vectors.
 *   queries  [in]  float[M * K]  — M query vectors (row-major)
 *   pool     [in]  float[N * K]  — N reference vectors
 *   results  [out] float[M * N]  — similarity matrix (row-major)
 *   M        [in]  int           — number of queries
 *   N        [in]  int           — number of references
 *   K        [in]  int           — dimension
 *
 * This replaces PyTorch's matrix multiply for CPU-only deployment.
 * With OpenBLAS, this compiles down to a single SGEMM call.
 * ═══════════════════════════════════════════════════════════════════ */
void batch_cosine(
    const float *queries, const float *pool,
    float *results, int M, int N, int K)
{
    /* queries: M×K, pool^T: K×N → results: M×N */
    /* result[i][j] = dot(queries[i], pool[j]) / (|queries[i]| * |pool[j]|) */
    
    /* Pre-compute pool norms */
    float *pool_norms = (float*)malloc(N * sizeof(float));
    for (int j = 0; j < N; j++) {
        double n2 = 0.0;
        for (int d = 0; d < K; d++) {
            float v = pool[j * K + d];
            n2 += (double)v * v;
        }
        pool_norms[j] = (float)(1.0 / sqrt(n2 + 1e-12));
    }
    
    for (int i = 0; i < M; i++) {
        const float *q = queries + i * K;
        
        /* Query norm */
        double qn2 = 0.0;
        for (int d = 0; d < K; d++) {
            qn2 += (double)q[d] * q[d];
        }
        float qnorm = (float)(1.0 / sqrt(qn2 + 1e-12));
        
        for (int j = 0; j < N; j++) {
            double dot = 0.0;
            for (int d = 0; d < K; d++) {
                dot += (double)q[d] * pool[j * K + d];
            }
            results[i * N + j] = (float)(dot * qnorm * pool_norms[j]);
        }
    }
    
    free(pool_norms);
}

/* ═══════════════════════════════════════════════════════════════════
 * KERNEL 3: jury_batch_search — Jury-GTC for batched queries
 *
 * Processes M queries through jury two-stage search.
 * This is the production entry point for ISAGI integration.
 *
 *   pool       [in]  float[N * K]
 *   domains    [in]  int[N]
 *   queries    [in]  float[M * K]
 *   N, K, M          dimensions
 *   sample_n, temp, threshold  — jury params
 *   best_indices[out] int[M]    — best trajectory per query
 *   best_sims   [out] float[M]  — similarity per query
 *   hits        [out] int[M]    — 1 if hit, 0 if miss
 *   comparisons [out] int[M]    — comparisons per query
 *   domains_out [out] int[M]    — detected domain per query
 * ═══════════════════════════════════════════════════════════════════ */
void jury_batch_search(
    const float *pool, const int *domains,
    const float *queries, int N, int K, int M,
    int sample_n, float temperature, float threshold,
    int *best_indices, float *best_sims, int *hits,
    int *comparisons, int *domains_out)
{
    for (int i = 0; i < M; i++) {
        const float *q = queries + i * K;
        hits[i] = jury_search(
            pool, domains, q, N, K,
            sample_n, temperature, threshold,
            &best_indices[i], &best_sims[i],
            &comparisons[i], &domains_out[i]
        );
    }
}
