/*
 * Geodessical — Axiomatic Linear Algebra Primitives
 *
 * Dense matrix operations, eigenvalue decomposition, PCA, TwoNN,
 * and GGUF dequantization for the axiomatic manifold pipeline.
 */

#include "runtime/nn/axiom_linalg.h"
#include "runtime/nn/gguf.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef GEODESSICAL_HOSTED
#include "host/hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* Q6_K block struct (256 values per super-block) */
typedef struct {
    uint8_t ql[128];    /* lower 4 bits of quantized values */
    uint8_t qh[64];     /* upper 2 bits of quantized values */
    int8_t  scales[16]; /* scales per sub-block */
    uint16_t d;         /* super-block scale (f16) */
} ax_q6k_block_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix Create / Destroy
 * ═══════════════════════════════════════════════════════════════════════════ */

axmat_t axmat_create(int rows, int cols)
{
    axmat_t m;
    m.rows = rows;
    m.cols = cols;
    m.data = (double *)tensor_alloc((uint64_t)rows * cols * sizeof(double));
    if (m.data) memset(m.data, 0, (uint64_t)rows * cols * sizeof(double));
    return m;
}

void axmat_destroy(axmat_t *m)
{
    if (m && m->data) {
        tensor_free(m->data);
        m->data = NULL;
        m->rows = m->cols = 0;
    }
}

void axmat_zero(axmat_t *m)
{
    if (m && m->data)
        memset(m->data, 0, (uint64_t)m->rows * m->cols * sizeof(double));
}

void axmat_identity(axmat_t *m)
{
    if (!m || !m->data || m->rows != m->cols) return;
    axmat_zero(m);
    for (int i = 0; i < m->rows; i++)
        m->data[i * m->cols + i] = 1.0;
}

axmat_t axmat_clone(const axmat_t *m)
{
    axmat_t c = axmat_create(m->rows, m->cols);
    if (c.data)
        memcpy(c.data, m->data, (uint64_t)m->rows * m->cols * sizeof(double));
    return c;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix Arithmetic
 * ═══════════════════════════════════════════════════════════════════════════ */

axmat_t axmat_mul(const axmat_t *A, const axmat_t *B)
{
    axmat_t C = axmat_create(A->rows, B->cols);
    if (!C.data) return C;

    for (int i = 0; i < A->rows; i++) {
        for (int k = 0; k < A->cols; k++) {
            double a_ik = A->data[i * A->cols + k];
            if (a_ik == 0.0) continue;
            for (int j = 0; j < B->cols; j++) {
                C.data[i * C.cols + j] += a_ik * B->data[k * B->cols + j];
            }
        }
    }
    return C;
}

axmat_t axmat_transpose(const axmat_t *A)
{
    axmat_t T = axmat_create(A->cols, A->rows);
    if (!T.data) return T;

    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            T.data[j * T.cols + i] = A->data[i * A->cols + j];
    return T;
}

void axmat_add_inplace(axmat_t *A, const axmat_t *B)
{
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] += B->data[i];
}

void axmat_scale_inplace(axmat_t *A, double s)
{
    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        A->data[i] *= s;
}

void axmat_add_outer(axmat_t *A, const double *u, const double *v,
                     int n, double alpha)
{
    for (int i = 0; i < n; i++) {
        double au = alpha * u[i];
        for (int j = 0; j < n; j++)
            A->data[i * A->cols + j] += au * v[j];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Symmetric Eigenvalue Decomposition (Jacobi)
 *
 * Classical Jacobi iteration for real symmetric matrices.  Stable and
 * reliable for moderate dimensions (n ≤ 512).  O(n³) per sweep, typically
 * converges in 5-10 sweeps.
 * ═══════════════════════════════════════════════════════════════════════════ */

int axmat_symeig(const axmat_t *A, double *eigenvalues, axmat_t *eigenvectors)
{
    int n = A->rows;
    if (n != A->cols || n <= 0) return -1;

    /* Work copy of A — we destroy it during iteration */
    axmat_t S = axmat_clone(A);
    if (!S.data) return -1;

    /* Pre-scale by Frobenius norm so absolute tolerance works regardless
     * of the magnitude of the input matrix entries.  We undo this by
     * multiplying eigenvalues by the scale factor after iteration. */
    double frob_sq = 0.0;
    for (int i = 0; i < n * n; i++) frob_sq += S.data[i] * S.data[i];
    double scale_factor = 1.0;
    if (frob_sq > 1e-30) {
        scale_factor = sqrt(frob_sq);
        double inv_scale = 1.0 / scale_factor;
        for (int i = 0; i < n * n; i++) S.data[i] *= inv_scale;
    }

    /* V starts as identity (accumulates rotations) */
    axmat_identity(eigenvectors);

    int max_sweeps = 100;
    double tol = 1e-12;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        /* Compute off-diagonal frobenius norm */
        double off_norm = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                double v = S.data[i * n + j];
                off_norm += v * v;
            }

        if (off_norm < tol) break;

        /* Sweep over all upper-triangular pairs */
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = S.data[p * n + q];
                if (fabs(apq) < tol * 0.01) continue;

                double app = S.data[p * n + p];
                double aqq = S.data[q * n + q];
                double tau = (aqq - app) / (2.0 * apq);
                double t;
                if (fabs(tau) > 1e15)
                    t = 1.0 / (2.0 * tau);
                else
                    t = (tau >= 0 ? 1.0 : -1.0) /
                        (fabs(tau) + sqrt(1.0 + tau * tau));

                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                /* Update S: Givens rotation in (p,q) plane */
                S.data[p * n + p] = app - t * apq;
                S.data[q * n + q] = aqq + t * apq;
                S.data[p * n + q] = 0.0;
                S.data[q * n + p] = 0.0;

                for (int r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    double srp = S.data[r * n + p];
                    double srq = S.data[r * n + q];
                    S.data[r * n + p] = c * srp - s * srq;
                    S.data[p * n + r] = S.data[r * n + p];
                    S.data[r * n + q] = s * srp + c * srq;
                    S.data[q * n + r] = S.data[r * n + q];
                }

                /* Accumulate rotation into eigenvector matrix */
                for (int r = 0; r < n; r++) {
                    double vrp = eigenvectors->data[r * n + p];
                    double vrq = eigenvectors->data[r * n + q];
                    eigenvectors->data[r * n + p] = c * vrp - s * vrq;
                    eigenvectors->data[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }

    /* Extract eigenvalues from diagonal and undo pre-scaling */
    for (int i = 0; i < n; i++)
        eigenvalues[i] = S.data[i * n + i] * scale_factor;

    /* Sort eigenvalues descending, reorder eigenvectors */
    for (int i = 0; i < n - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < n; j++)
            if (eigenvalues[j] > eigenvalues[max_idx])
                max_idx = j;

        if (max_idx != i) {
            double tmp = eigenvalues[i];
            eigenvalues[i] = eigenvalues[max_idx];
            eigenvalues[max_idx] = tmp;

            /* Swap eigenvector columns */
            for (int r = 0; r < n; r++) {
                double t = eigenvectors->data[r * n + i];
                eigenvectors->data[r * n + i] = eigenvectors->data[r * n + max_idx];
                eigenvectors->data[r * n + max_idx] = t;
            }
        }
    }

    axmat_destroy(&S);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PCA
 *
 * Given X (n_samples × dim), compute PCA by:
 * 1. Center X (subtract mean)
 * 2. If n_samples < dim: compute n×n Gram matrix X X^T, eigen-decompose,
 *    recover components via X^T eigvecs.  This is the "economy" approach.
 * 3. If n_samples >= dim: compute dim×dim covariance matrix, eigen-decompose.
 * ═══════════════════════════════════════════════════════════════════════════ */

axpca_t axpca_compute(const axmat_t *X, double min_explained_ratio)
{
    axpca_t pca;
    memset(&pca, 0, sizeof(pca));

    int n = X->rows;
    int d = X->cols;
    pca.dim = d;

    /* 1. Compute mean */
    pca.mean = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    if (!pca.mean) return pca;

    memset(pca.mean, 0, (uint64_t)d * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            pca.mean[j] += X->data[i * d + j];
    for (int j = 0; j < d; j++)
        pca.mean[j] /= (double)n;

    /* 2. Center the data */
    axmat_t Xc = axmat_clone(X);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            Xc.data[i * d + j] -= pca.mean[j];

    /* 3. Choose economy or standard mode */
    int m = (n < d) ? n : d;  /* effective rank bound */

    /* We always use the Gram matrix approach X*X^T (n×n) when n < d,
     * or the covariance X^T*X (d×d) otherwise */
    double *evals = (double *)tensor_alloc((uint64_t)m * sizeof(double));
    if (!evals) { axmat_destroy(&Xc); return pca; }

    if (n <= d) {
        /* Gram matrix: G = Xc × Xc^T  (n × n) */
        axmat_t G = axmat_create(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double dot = 0.0;
                for (int k = 0; k < d; k++)
                    dot += Xc.data[i * d + k] * Xc.data[j * d + k];
                dot /= (double)(n - 1);
                G.data[i * n + j] = dot;
                G.data[j * n + i] = dot;
            }
        }

        axmat_t Vg = axmat_create(n, n);
        axmat_symeig(&G, evals, &Vg);

        /* Recover principal components: PC_k = X^T v_k / sqrt(lambda_k * (n-1)) */
        axmat_t comps = axmat_create(n, d);
        for (int k = 0; k < n; k++) {
            double scale = (evals[k] > 1e-14)
                           ? 1.0 / sqrt(evals[k] * (double)(n - 1))
                           : 0.0;
            for (int j = 0; j < d; j++) {
                double sum = 0.0;
                for (int i = 0; i < n; i++)
                    sum += Xc.data[i * d + j] * Vg.data[i * n + k];
                comps.data[k * d + j] = sum * scale;
            }
        }

        axmat_destroy(&G);
        axmat_destroy(&Vg);
        pca.components = comps;
    } else {
        /* Covariance matrix: C = X^T X / (n-1)  (d × d) */
        axmat_t C = axmat_create(d, d);
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                double dot = 0.0;
                for (int k = 0; k < n; k++)
                    dot += Xc.data[k * d + i] * Xc.data[k * d + j];
                dot /= (double)(n - 1);
                C.data[i * d + j] = dot;
                C.data[j * d + i] = dot;
            }
        }

        axmat_t Vc = axmat_create(d, d);
        axmat_symeig(&C, evals, &Vc);

        /* Components are the eigenvector rows (transpose of column-eigvecs) */
        pca.components = axmat_transpose(&Vc);
        axmat_destroy(&C);
        axmat_destroy(&Vc);
        m = d;
    }

    axmat_destroy(&Xc);

    /* 4. Determine number of components to keep */
    pca.total_variance = 0.0;
    for (int i = 0; i < m; i++)
        pca.total_variance += (evals[i] > 0 ? evals[i] : 0);

    double cum = 0.0;
    int keep = 0;
    for (int i = 0; i < m; i++) {
        if (evals[i] <= 0) break;
        cum += evals[i];
        keep++;
        if (cum / pca.total_variance >= min_explained_ratio)
            break;
    }
    if (keep < 1) keep = 1;

    pca.n_components = keep;
    pca.explained_variance = cum;
    pca.eigenvalues = (double *)tensor_alloc((uint64_t)keep * sizeof(double));
    if (pca.eigenvalues)
        memcpy(pca.eigenvalues, evals, (uint64_t)keep * sizeof(double));

    /* Trim component matrix to [keep × d] */
    if (pca.components.rows > keep) {
        axmat_t trimmed = axmat_create(keep, d);
        memcpy(trimmed.data, pca.components.data,
               (uint64_t)keep * d * sizeof(double));
        axmat_destroy(&pca.components);
        pca.components = trimmed;
    }

    tensor_free(evals);
    return pca;
}

void axpca_destroy(axpca_t *pca)
{
    if (!pca) return;
    if (pca->eigenvalues) { tensor_free(pca->eigenvalues); pca->eigenvalues = NULL; }
    if (pca->mean) { tensor_free(pca->mean); pca->mean = NULL; }
    axmat_destroy(&pca->components);
    pca->n_components = 0;
}

void axpca_project(const axpca_t *pca, const double *x_full, double *x_sub)
{
    int d = pca->dim;
    int k = pca->n_components;
    for (int i = 0; i < k; i++) {
        double dot = 0.0;
        for (int j = 0; j < d; j++)
            dot += pca->components.data[i * d + j] * (x_full[j] - pca->mean[j]);
        x_sub[i] = dot;
    }
}

void axpca_reconstruct(const axpca_t *pca, const double *x_sub, double *x_full)
{
    int d = pca->dim;
    int k = pca->n_components;
    for (int j = 0; j < d; j++) {
        double sum = pca->mean[j];
        for (int i = 0; i < k; i++)
            sum += x_sub[i] * pca->components.data[i * d + j];
        x_full[j] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TwoNN Intrinsic Dimensionality Estimator
 *
 * Facco et al. "Estimating the intrinsic dimension of datasets by a
 * minimal neighborhood information" (2017).
 *
 * For each point, find the two nearest neighbors.  The ratio μ = r2/r1
 * follows a power law whose exponent is the intrinsic dimension.
 * ID = n / Σ ln(μ_i)
 * ═══════════════════════════════════════════════════════════════════════════ */

double ax_twonn_id(const axmat_t *X)
{
    int n = X->rows;
    int d = X->cols;
    if (n < 4) return 1.0;

    double log_mu_sum = 0.0;
    int valid = 0;

    for (int i = 0; i < n; i++) {
        double d1 = DBL_MAX, d2 = DBL_MAX;

        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double dist = 0.0;
            for (int k = 0; k < d; k++) {
                double diff = X->data[i * d + k] - X->data[j * d + k];
                dist += diff * diff;
            }
            if (dist < d1) {
                d2 = d1;
                d1 = dist;
            } else if (dist < d2) {
                d2 = dist;
            }
        }

        if (d1 > 0.0 && d2 > 0.0) {
            double mu = sqrt(d2 / d1);
            if (mu > 1.0) {
                log_mu_sum += log(mu);
                valid++;
            }
        }
    }

    if (valid < 2) return 1.0;
    return (double)valid / log_mu_sum;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Vector Operations
 * ═══════════════════════════════════════════════════════════════════════════ */

double ax_vec_dot(const double *a, const double *b, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

double ax_vec_norm(const double *a, int n)
{
    return sqrt(ax_vec_dot(a, a, n));
}

void ax_vec_scale(double *dst, const double *src, double s, int n)
{
    for (int i = 0; i < n; i++) dst[i] = src[i] * s;
}

void ax_vec_add(double *dst, const double *a, const double *b, int n)
{
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

void ax_vec_sub(double *dst, const double *a, const double *b, int n)
{
    for (int i = 0; i < n; i++) dst[i] = a[i] - b[i];
}

void ax_vec_zero(double *dst, int n)
{
    memset(dst, 0, (uint64_t)n * sizeof(double));
}

void ax_vec_copy(double *dst, const double *src, int n)
{
    memcpy(dst, src, (uint64_t)n * sizeof(double));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Sorting
 * ═══════════════════════════════════════════════════════════════════════════ */

void ax_sort_ascending(double *arr, int n)
{
    /* Simple insertion sort — n ≤ a few thousand for our use cases */
    for (int i = 1; i < n; i++) {
        double key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void ax_argsort_ascending(const double *arr, int *indices, int n)
{
    for (int i = 0; i < n; i++) indices[i] = i;
    for (int i = 1; i < n; i++) {
        int ki = indices[i];
        double key = arr[ki];
        int j = i - 1;
        while (j >= 0 && arr[indices[j]] > key) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = ki;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GGUF Dequantization
 *
 * Convert quantized weight rows to double (for geometric analysis).
 * ═══════════════════════════════════════════════════════════════════════════ */

static float ax_fp16_to_f32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (expo == 0) {
        if (mant == 0) { f = sign; }
        else {
            expo = 1;
            while (!(mant & 0x400)) { mant <<= 1; expo--; }
            mant &= 0x3FF;
            f = sign | ((uint32_t)(expo + 127 - 15) << 23) | ((uint32_t)mant << 13);
        }
    } else if (expo == 31) {
        f = sign | 0x7F800000u | ((uint32_t)mant << 13);
    } else {
        f = sign | ((uint32_t)(expo + 127 - 15) << 23) | ((uint32_t)mant << 13);
    }

    float r;
    memcpy(&r, &f, 4);
    return r;
}

static float ax_bf16_to_f32(uint16_t h)
{
    uint32_t f = (uint32_t)h << 16;
    float r;
    memcpy(&r, &f, 4);
    return r;
}

int ax_dequant_row_f32(const void *src, float *dst, int n, ggml_type_t type)
{
    if (!src || !dst || n <= 0) return -1;

    switch (type) {
    case GGML_TYPE_F32: {
        memcpy(dst, src, (uint64_t)n * sizeof(float));
        return 0;
    }
    case GGML_TYPE_F16: {
        const uint16_t *s = (const uint16_t *)src;
        for (int i = 0; i < n; i++) dst[i] = ax_fp16_to_f32(s[i]);
        return 0;
    }
    case GGML_TYPE_BF16: {
        const uint16_t *s = (const uint16_t *)src;
        for (int i = 0; i < n; i++) dst[i] = ax_bf16_to_f32(s[i]);
        return 0;
    }
    case GGML_TYPE_Q4_0: {
        /* Each block: 32 values, 1 float16 scale + 16 bytes nibbles */
        const uint8_t *p = (const uint8_t *)src;
        int nblocks = n / 32;
        for (int b = 0; b < nblocks; b++) {
            uint16_t dh;
            memcpy(&dh, p, 2);
            float d = ax_fp16_to_f32(dh);
            p += 2;
            for (int j = 0; j < 16; j++) {
                uint8_t byte = p[j];
                int lo = (int)(byte & 0xF) - 8;
                int hi = (int)(byte >> 4) - 8;
                dst[b * 32 + j]      = d * (float)lo;
                dst[b * 32 + j + 16] = d * (float)hi;
            }
            p += 16;
        }
        return 0;
    }
    case GGML_TYPE_Q8_0: {
        /* Each block: 32 values, 1 float16 scale + 32 int8 quants */
        const uint8_t *p = (const uint8_t *)src;
        int nblocks = n / 32;
        for (int b = 0; b < nblocks; b++) {
            uint16_t dh;
            memcpy(&dh, p, 2);
            float d = ax_fp16_to_f32(dh);
            p += 2;
            const int8_t *qs = (const int8_t *)p;
            for (int j = 0; j < 32; j++)
                dst[b * 32 + j] = d * (float)qs[j];
            p += 32;
        }
        return 0;
    }
    case GGML_TYPE_Q6_K: {
        /* Q6_K: 256 values per super-block */
        const ax_q6k_block_t *blocks = (const ax_q6k_block_t *)src;
        int nsb = n / 256;
        for (int sb = 0; sb < nsb; sb++) {
            const ax_q6k_block_t *b = &blocks[sb];
            float d_val = ax_fp16_to_f32(b->d);
            float *o = dst + sb * 256;
            for (int half = 0; half < 2; half++) {
                const uint8_t *ql_h = b->ql + half * 64;
                const uint8_t *qh_h = b->qh + half * 32;
                const int8_t *sc_h = b->scales + half * 8;
                for (int l = 0; l < 32; l++) {
                    int q0 = (int)(ql_h[l] & 0xF)      | (int)(((qh_h[l] >> 0) & 3) << 4);
                    int q1 = (int)(ql_h[l + 32] & 0xF)  | (int)(((qh_h[l] >> 2) & 3) << 4);
                    int q2 = (int)(ql_h[l] >> 4)         | (int)(((qh_h[l] >> 4) & 3) << 4);
                    int q3 = (int)(ql_h[l + 32] >> 4)    | (int)(((qh_h[l] >> 6) & 3) << 4);
                    int si = l / 16;
                    o[half*128 + l]      = d_val * (float)sc_h[0 + si] * (float)(q0 - 32);
                    o[half*128 + l + 32] = d_val * (float)sc_h[2 + si] * (float)(q1 - 32);
                    o[half*128 + l + 64] = d_val * (float)sc_h[4 + si] * (float)(q2 - 32);
                    o[half*128 + l + 96] = d_val * (float)sc_h[6 + si] * (float)(q3 - 32);
                }
            }
        }
        return 0;
    }
    default:
        return -1;
    }
}

int ax_dequant_row(const void *src, double *dst, int n, ggml_type_t type)
{
    /* Two-step: dequant to f32, then promote to f64.
     * Avoids duplicating all type-specific logic. */
    float *tmp = (float *)tensor_alloc((uint64_t)n * sizeof(float));
    if (!tmp) return -1;

    int rc = ax_dequant_row_f32(src, tmp, n, type);
    if (rc == 0) {
        for (int i = 0; i < n; i++)
            dst[i] = (double)tmp[i];
    }

    tensor_free(tmp);
    return rc;
}
