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

#if defined(__AVX2__)
#include <immintrin.h>
#endif

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

/* 
 * Matrix Create / Destroy
 *  */

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

/* 
 * Matrix Arithmetic
 *  */

axmat_t axmat_mul(const axmat_t *A, const axmat_t *B)
{
    axmat_t C = axmat_create(A->rows, B->cols);
    if (!C.data) return C;

    int M = A->rows, K = A->cols, N = B->cols;

#if defined(__AVX2__) && defined(__FMA__)
    /* AVX2 row-panel: for each row of A, accumulate into the corresponding
     * row of C using FMA over the B rows. */
    for (int i = 0; i < M; i++) {
        const double *A_row = A->data + (uint64_t)i * K;
        double       *C_row = C.data  + (uint64_t)i * N;
        for (int k = 0; k < K; k++) {
            double a_ik = A_row[k];
            if (a_ik == 0.0) continue;
            const double *B_row = B->data + (uint64_t)k * N;
            __m256d va = _mm256_set1_pd(a_ik);
            int j = 0;
            for (; j + 4 <= N; j += 4) {
                __m256d vc = _mm256_loadu_pd(C_row + j);
                __m256d vb = _mm256_loadu_pd(B_row + j);
                vc = _mm256_fmadd_pd(va, vb, vc);
                _mm256_storeu_pd(C_row + j, vc);
            }
            for (; j < N; j++) C_row[j] += a_ik * B_row[j];
        }
    }
#else
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            double a_ik = A->data[i * K + k];
            if (a_ik == 0.0) continue;
            for (int j = 0; j < N; j++)
                C.data[i * N + j] += a_ik * B->data[k * N + j];
        }
    }
#endif
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

/* 
 * Symmetric Eigenvalue Decomposition (Jacobi)
 *
 * Classical Jacobi iteration for real symmetric matrices.  Stable and
 * reliable for moderate dimensions (n ≤ 512).  O(n³) per sweep, typically
 * converges in 5-10 sweeps.
 *  */

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

/* 
 * PCA
 *
 * Given X (n_samples  dim), compute PCA by:
 * 1. Center X (subtract mean)
 * 2. If n_samples < dim: compute nn Gram matrix X X^T, eigen-decompose,
 *    recover components via X^T eigvecs.  This is the "economy" approach.
 * 3. If n_samples >= dim: compute dimdim covariance matrix, eigen-decompose.
 *  */

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

    /* We always use the Gram matrix approach X*X^T (nn) when n < d,
     * or the covariance X^T*X (dd) otherwise */
    double *evals = (double *)tensor_alloc((uint64_t)m * sizeof(double));
    if (!evals) { axmat_destroy(&Xc); return pca; }

    if (n <= d) {
        /* Gram matrix: G = Xc  Xc^T  (n  n) */
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
        /* Covariance matrix: C = X^T X / (n-1)  (d  d) */
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

    /* Trim component matrix to [keep  d] */
    if (pca.components.rows > keep) {
        axmat_t trimmed = axmat_create(keep, d);
        memcpy(trimmed.data, pca.components.data,
               (uint64_t)keep * d * sizeof(double));
        axmat_destroy(&pca.components);
        pca.components = trimmed;
    }

    tensor_free(evals);

    /* 5. Precompute mean projection: mean_proj[i] = dot(component_i, mean).
     * This amortizes O(k*d) work that would otherwise repeat on every
     * axpca_project call, replacing k full dot products with k scalar lookups. */
    pca.mean_proj = (double *)tensor_alloc((uint64_t)pca.n_components * sizeof(double));
    if (pca.mean_proj) {
        for (int i = 0; i < pca.n_components; i++) {
            const double *row = pca.components.data + (uint64_t)i * d;
            pca.mean_proj[i] = ax_vec_dot(row, pca.mean, d);
        }
    }

    return pca;
}

void axpca_destroy(axpca_t *pca)
{
    if (!pca) return;
    if (pca->eigenvalues) { tensor_free(pca->eigenvalues); pca->eigenvalues = NULL; }
    if (pca->mean)        { tensor_free(pca->mean);        pca->mean        = NULL; }
    if (pca->mean_proj)   { tensor_free(pca->mean_proj);   pca->mean_proj   = NULL; }
    axmat_destroy(&pca->components);
    pca->n_components = 0;
}

/* 
 * Fast Truncated PCA — Randomized SVD (Halko, Martinsson, Tropp 2011)
 *  */
axpca_t axpca_compute_topk(const axmat_t *X, int k_max)
{
    if (!X || !X->data) {
        axpca_t empty; memset(&empty, 0, sizeof(empty)); return empty;
    }
    int n = X->rows, d = X->cols;
    if (k_max <= 0) k_max = (n < d ? n : d);
    /* Fall back to full Jacobi for near-full-rank */
    if ((double)k_max >= 0.65 * (double)(n < d ? n : d))
        return axpca_compute(X, 0.0);

    axpca_t pca; memset(&pca, 0, sizeof(pca));
    pca.dim = d;
    pca.mean = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    if (!pca.mean) return pca;
    memset(pca.mean, 0, (uint64_t)d * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            pca.mean[j] += X->data[(uint64_t)i * d + j];
    for (int j = 0; j < d; j++) pca.mean[j] /= (double)n;

    axmat_t Xc = axmat_clone(X);
    if (!Xc.data) { axpca_destroy(&pca); return pca; }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            Xc.data[(uint64_t)i * d + j] -= pca.mean[j];

    int over = 10, k_eff = k_max + over;
    if (k_eff > d) k_eff = d;

    double *Omega = (double *)tensor_alloc((uint64_t)d * k_eff * sizeof(double));
    if (!Omega) { axmat_destroy(&Xc); axpca_destroy(&pca); return pca; }
    uint64_t seed = 0xB1C2D3E4F5061728ULL;
    for (int i = 0; i < d * k_eff; i += 2) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        double u1 = (double)((seed >> 11) + 1) * (1.0 / 9007199254740992.0);
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        double u2 = (double)((seed >> 11) + 1) * (1.0 / 9007199254740992.0);
        double r = sqrt(-2.0 * log(u1)), th = 6.283185307179586476925 * u2;
        Omega[i] = r * cos(th);
        if (i + 1 < d * k_eff) Omega[i + 1] = r * sin(th);
    }
    axmat_t Y; memset(&Y, 0, sizeof(Y));
    Y.data = Omega; Y.rows = d; Y.cols = k_eff;

    axmat_t XcT = axmat_transpose(&Xc);
    for (int pi = 0; pi < 3; pi++) {
        axmat_t T = axmat_mul(&Xc, &Y);
        tensor_free(Y.data);
        Y.data = NULL;
        axmat_t Y2 = axmat_mul(&XcT, &T);
        axmat_destroy(&T);
        Y = Y2;
    }
    axmat_destroy(&XcT);

    /* QR via Gram-Schmidt */
    axmat_t Q = axmat_create(d, k_eff);
    if (!Q.data) { tensor_free(Y.data); axmat_destroy(&Xc); axpca_destroy(&pca); return pca; }
    memcpy(Q.data, Y.data, (uint64_t)d * k_eff * sizeof(double));
    tensor_free(Y.data); Y.data = NULL;
    for (int j = 0; j < k_eff; j++) {
        for (int jj = 0; jj < j; jj++) {
            double dot = 0.0;
            for (int i = 0; i < d; i++)
                dot += Q.data[(uint64_t)i * k_eff + jj] * Q.data[(uint64_t)i * k_eff + j];
            for (int i = 0; i < d; i++)
                Q.data[(uint64_t)i * k_eff + j] -= dot * Q.data[(uint64_t)i * k_eff + jj];
        }
        double norm = 0.0;
        for (int i = 0; i < d; i++) { double v = Q.data[(uint64_t)i * k_eff + j]; norm += v * v; }
        if (norm > 1e-28) { double inv = 1.0 / sqrt(norm);
            for (int i = 0; i < d; i++) Q.data[(uint64_t)i * k_eff + j] *= inv; }
    }

    /* B = Q^T Xc^T Xc Q */
    axmat_t XcQ = axmat_mul(&Xc, &Q);
    axmat_destroy(&Xc);
    axmat_t Qt = axmat_transpose(&Q);
    axmat_t B = axmat_mul(&Qt, &XcQ);
    axmat_destroy(&Qt); axmat_destroy(&XcQ);
    for (int i = 0; i < k_eff; i++)
        for (int j = i + 1; j < k_eff; j++) {
            double avg = 0.5 * (B.data[(uint64_t)i * k_eff + j] + B.data[(uint64_t)j * k_eff + i]);
            B.data[(uint64_t)i * k_eff + j] = avg; B.data[(uint64_t)j * k_eff + i] = avg;
        }
    double *evals = (double *)tensor_alloc((uint64_t)k_eff * sizeof(double));
    axmat_t U = axmat_create(k_eff, k_eff);
    if (!evals || !U.data) { if (evals) tensor_free(evals); axmat_destroy(&U);
        axmat_destroy(&B); axmat_destroy(&Q); axpca_destroy(&pca); return pca; }
    axmat_symeig(&B, evals, &U); axmat_destroy(&B);
    int keep = (k_max < k_eff) ? k_max : k_eff;
    axmat_t Vt = axmat_create(keep, d);
    if (!Vt.data) { tensor_free(evals); axmat_destroy(&U); axmat_destroy(&Q);
        axpca_destroy(&pca); return pca; }
    for (int i = 0; i < keep; i++)
        for (int j = 0; j < d; j++) {
            double val = 0.0;
            for (int l = 0; l < k_eff; l++) val += Q.data[(uint64_t)j * k_eff + l] * U.data[(uint64_t)l * k_eff + i];
            Vt.data[(uint64_t)i * d + j] = val;
        }
    axmat_destroy(&Q); axmat_destroy(&U);
    pca.components = Vt; pca.eigenvalues = evals; pca.n_components = keep;
    pca.explained_variance = 0.0;
    for (int i = 0; i < keep; i++) pca.explained_variance += evals[i];
    pca.mean_proj = (double *)tensor_alloc((uint64_t)keep * sizeof(double));
    if (pca.mean_proj)
        for (int i = 0; i < keep; i++) {
            double dot = 0.0;
            for (int j = 0; j < d; j++) dot += pca.components.data[(uint64_t)i * d + j] * pca.mean[j];
            pca.mean_proj[i] = dot;
        }
    return pca;
}

/* 
 * Fast Weighted PCA — eigenvectors of Xc^T Xc K (data  weight gram)
 *  */
axpca_t axpca_compute_topk_weighted(
    const axmat_t *X,
    matvec_fn      K_apply,
    void          *K_ctx,
    int            k_max)
{
    if (!X || !X->data) {
        axpca_t empty; memset(&empty, 0, sizeof(empty)); return empty;
    }
    int n = X->rows, d = X->cols;
    if (k_max <= 0) k_max = (n < d ? n : d);

    axpca_t pca; memset(&pca, 0, sizeof(pca)); pca.dim = d;
    pca.mean = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    if (!pca.mean) return pca;
    memset(pca.mean, 0, (uint64_t)d * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            pca.mean[j] += X->data[(uint64_t)i * d + j];
    for (int j = 0; j < d; j++) pca.mean[j] /= (double)n;

    axmat_t Xc = axmat_clone(X);
    if (!Xc.data) { axpca_destroy(&pca); return pca; }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            Xc.data[(uint64_t)i * d + j] -= pca.mean[j];

    int oversampling = 10, k_eff = k_max + oversampling;
    if (k_eff > d) k_eff = d;

    double *Omega_data = (double *)tensor_alloc((uint64_t)d * k_eff * sizeof(double));
    if (!Omega_data) { axmat_destroy(&Xc); axpca_destroy(&pca); return pca; }
    uint64_t seed = 0xA1B2C3D4E5F60718ULL;
    for (int i = 0; i < d * k_eff; i += 2) {
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        double u1 = (double)((seed >> 11) + 1) * (1.0 / 9007199254740992.0);
        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
        double u2 = (double)((seed >> 11) + 1) * (1.0 / 9007199254740992.0);
        double r = sqrt(-2.0 * log(u1)), th = 6.283185307179586476925 * u2;
        Omega_data[i] = r * cos(th);
        if (i + 1 < d * k_eff) Omega_data[i + 1] = r * sin(th);
    }
    axmat_t Y; memset(&Y, 0, sizeof(Y));
    Y.data = Omega_data; Y.rows = d; Y.cols = k_eff;

    float *x_f32 = (float *)tensor_alloc((uint64_t)d * sizeof(float));
    float *y_f32 = (float *)tensor_alloc((uint64_t)d * sizeof(float));
    if (!x_f32 || !y_f32) {
        if (x_f32) tensor_free(x_f32); if (y_f32) tensor_free(y_f32);
        tensor_free(Y.data); axmat_destroy(&Xc); axpca_destroy(&pca); return pca;
    }
    axmat_t XcT = axmat_transpose(&Xc);

    /* Power iterations: Y <- Xc^T Xc K Y */
    for (int pi = 0; pi < 3; pi++) {
        axmat_t Z = axmat_create(d, k_eff);
        if (!Z.data) break;
        for (int j = 0; j < k_eff; j++) {
            for (int i = 0; i < d; i++) x_f32[i] = (float)Y.data[(uint64_t)i * k_eff + j];
            K_apply(x_f32, y_f32, d, K_ctx);
            for (int i = 0; i < d; i++) Z.data[(uint64_t)i * k_eff + j] = (double)y_f32[i];
        }
        axmat_t T = axmat_mul(&Xc, &Z); axmat_destroy(&Z);
        tensor_free(Y.data); Y.data = NULL;
        axmat_t Y2 = axmat_mul(&XcT, &T); axmat_destroy(&T);
        Y = Y2;
    }
    tensor_free(x_f32); tensor_free(y_f32);

    /* QR via Gram-Schmidt Q[d  k_eff] */
    axmat_t Q = axmat_create(d, k_eff);
    if (!Q.data) { tensor_free(Y.data); axmat_destroy(&Xc); axmat_destroy(&XcT); axpca_destroy(&pca); return pca; }
    memcpy(Q.data, Y.data, (uint64_t)d * k_eff * sizeof(double));
    tensor_free(Y.data); Y.data = NULL;
    for (int j = 0; j < k_eff; j++) {
        for (int jj = 0; jj < j; jj++) {
            double dot = 0.0;
            for (int i = 0; i < d; i++)
                dot += Q.data[(uint64_t)i * k_eff + jj] * Q.data[(uint64_t)i * k_eff + j];
            for (int i = 0; i < d; i++)
                Q.data[(uint64_t)i * k_eff + j] -= dot * Q.data[(uint64_t)i * k_eff + jj];
        }
        double norm = 0.0;
        for (int i = 0; i < d; i++) { double v = Q.data[(uint64_t)i * k_eff + j]; norm += v * v; }
        if (norm > 1e-28) { double inv = 1.0 / sqrt(norm);
            for (int i = 0; i < d; i++) Q.data[(uint64_t)i * k_eff + j] *= inv; }
    }
    axmat_destroy(&Y);

    /* B = Q^T Xc^T Xc K Q */
    x_f32 = (float *)tensor_alloc((uint64_t)d * sizeof(float));
    y_f32 = (float *)tensor_alloc((uint64_t)d * sizeof(float));
    axmat_t KQ = axmat_create(d, k_eff);
    if (!x_f32 || !y_f32 || !KQ.data) {
        if (x_f32) tensor_free(x_f32); if (y_f32) tensor_free(y_f32);
        axmat_destroy(&KQ); axmat_destroy(&Q); axmat_destroy(&Xc); axmat_destroy(&XcT);
        axpca_destroy(&pca); return pca;
    }
    for (int j = 0; j < k_eff; j++) {
        for (int i = 0; i < d; i++) x_f32[i] = (float)Q.data[(uint64_t)i * k_eff + j];
        K_apply(x_f32, y_f32, d, K_ctx);
        for (int i = 0; i < d; i++) KQ.data[(uint64_t)i * k_eff + j] = (double)y_f32[i];
    }
    tensor_free(x_f32); tensor_free(y_f32);
    axmat_t T = axmat_mul(&Xc, &KQ); axmat_t XcT_T = axmat_mul(&XcT, &T);
    axmat_destroy(&T); axmat_destroy(&XcT); axmat_destroy(&KQ); axmat_destroy(&Xc);
    axmat_t Qt = axmat_transpose(&Q);
    axmat_t B = axmat_mul(&Qt, &XcT_T);
    axmat_destroy(&Qt); axmat_destroy(&XcT_T);
    for (int i = 0; i < k_eff; i++)
        for (int j = i + 1; j < k_eff; j++) {
            double avg = 0.5 * (B.data[(uint64_t)i * k_eff + j] + B.data[(uint64_t)j * k_eff + i]);
            B.data[(uint64_t)i * k_eff + j] = avg; B.data[(uint64_t)j * k_eff + i] = avg;
        }
    double *evals = (double *)tensor_alloc((uint64_t)k_eff * sizeof(double));
    axmat_t U = axmat_create(k_eff, k_eff);
    if (!evals || !U.data) { if (evals) tensor_free(evals); axmat_destroy(&U);
        axmat_destroy(&B); axmat_destroy(&Q); axpca_destroy(&pca); return pca; }
    axmat_symeig(&B, evals, &U); axmat_destroy(&B);
    int keep = (k_max < k_eff) ? k_max : k_eff;
    axmat_t Vt = axmat_create(keep, d);
    if (!Vt.data) { tensor_free(evals); axmat_destroy(&U); axmat_destroy(&Q);
        axpca_destroy(&pca); return pca; }
    for (int i = 0; i < keep; i++)
        for (int j = 0; j < d; j++) {
            double val = 0.0;
            for (int l = 0; l < k_eff; l++) val += Q.data[(uint64_t)j * k_eff + l] * U.data[(uint64_t)l * k_eff + i];
            Vt.data[(uint64_t)i * d + j] = val;
        }
    axmat_destroy(&Q); axmat_destroy(&U);
    pca.components = Vt; pca.eigenvalues = evals; pca.n_components = keep;
    pca.explained_variance = 0.0;
    for (int i = 0; i < keep; i++) pca.explained_variance += evals[i];
    pca.mean_proj = (double *)tensor_alloc((uint64_t)keep * sizeof(double));
    if (pca.mean_proj)
        for (int i = 0; i < keep; i++) {
            double dot = 0.0;
            for (int j = 0; j < d; j++) dot += pca.components.data[(uint64_t)i * d + j] * pca.mean[j];
            pca.mean_proj[i] = dot;
        }
    return pca;
}

/* 
 * Weight-only eigenvector basis (no calibration data needed)
 *
 * Finds top-k eigenvectors of K = Σ Wᵢ^T Wᵢ using matrix-free power
 * iteration via K_apply.  This maximises sum_i ||Wᵢ Ptᵀ||_F^2 exactly,
 * unlike axpca_compute_topk_weighted which finds eigenvectors of Xc^T Xc K
 * (data-dominated when hidden-state covariance has large eigenvalue spread).
 *  */
axpca_t axpca_compute_weight_topk(
    matvec_fn  K_apply,
    void      *K_ctx,
    int        d,
    int        k_max)
{
    axpca_t pca; memset(&pca, 0, sizeof(pca));
    if (!K_apply || d <= 0 || k_max <= 0) return pca;
    pca.dim = d;
    pca.mean = (double *)tensor_alloc((uint64_t)d * sizeof(double));
    if (!pca.mean) return pca;
    memset(pca.mean, 0, (uint64_t)d * sizeof(double));

    int oversampling = 10, k_eff = k_max + oversampling;
    if (k_eff > d) k_eff = d;

    double *Y_data = (double *)tensor_alloc((uint64_t)d * k_eff * sizeof(double));
    if (!Y_data) { axpca_destroy(&pca); return pca; }
    {
        uint64_t seed = 0xC3D4E5F6A1B20718ULL;
        for (int i = 0; i < d * k_eff; i += 2) {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            double u1 = (double)((seed >> 11) + 1) * (1.0 / 9007199254740992.0);
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            double u2 = (double)((seed >> 11) + 1) * (1.0 / 9007199254740992.0);
            double r = sqrt(-2.0 * log(u1)), th = 6.283185307179586476925 * u2;
            Y_data[i] = r * cos(th);
            if (i + 1 < d * k_eff) Y_data[i + 1] = r * sin(th);
        }
    }
    axmat_t Y; memset(&Y, 0, sizeof(Y));
    Y.data = Y_data; Y.rows = d; Y.cols = k_eff;

    float *x_f32 = (float *)tensor_alloc((uint64_t)d * sizeof(float));
    float *y_f32 = (float *)tensor_alloc((uint64_t)d * sizeof(float));
    if (!x_f32 || !y_f32) {
        if (x_f32) tensor_free(x_f32); if (y_f32) tensor_free(y_f32);
        tensor_free(Y.data); axpca_destroy(&pca); return pca;
    }

    /* Power iterations: Y <- K @ Y */
    for (int pi = 0; pi < 3; pi++) {
        axmat_t Z = axmat_create(d, k_eff);
        if (!Z.data) break;
        for (int j = 0; j < k_eff; j++) {
            for (int i = 0; i < d; i++) x_f32[i] = (float)Y.data[(uint64_t)i * k_eff + j];
            K_apply(x_f32, y_f32, d, K_ctx);
            for (int i = 0; i < d; i++) Z.data[(uint64_t)i * k_eff + j] = (double)y_f32[i];
        }
        tensor_free(Y.data); Y.data = Z.data; Y.rows = Z.rows; Y.cols = Z.cols;
    }
    tensor_free(x_f32); x_f32 = NULL;
    tensor_free(y_f32); y_f32 = NULL;

    /* QR via Gram-Schmidt */
    axmat_t Q = axmat_create(d, k_eff);
    if (!Q.data) { tensor_free(Y.data); axpca_destroy(&pca); return pca; }
    memcpy(Q.data, Y.data, (uint64_t)d * k_eff * sizeof(double));
    tensor_free(Y.data); Y.data = NULL;
    for (int j = 0; j < k_eff; j++) {
        for (int jj = 0; jj < j; jj++) {
            double dot = 0.0;
            for (int i = 0; i < d; i++)
                dot += Q.data[(uint64_t)i * k_eff + jj] * Q.data[(uint64_t)i * k_eff + j];
            for (int i = 0; i < d; i++)
                Q.data[(uint64_t)i * k_eff + j] -= dot * Q.data[(uint64_t)i * k_eff + jj];
        }
        double norm = 0.0;
        for (int i = 0; i < d; i++) { double v = Q.data[(uint64_t)i * k_eff + j]; norm += v * v; }
        if (norm > 1e-28) { double inv = 1.0 / sqrt(norm);
            for (int i = 0; i < d; i++) Q.data[(uint64_t)i * k_eff + j] *= inv; }
    }

    /* B = Q^T K Q */
    x_f32 = (float *)tensor_alloc((uint64_t)d * sizeof(float));
    y_f32 = (float *)tensor_alloc((uint64_t)d * sizeof(float));
    axmat_t KQ = axmat_create(d, k_eff);
    if (!x_f32 || !y_f32 || !KQ.data) {
        if (x_f32) tensor_free(x_f32); if (y_f32) tensor_free(y_f32);
        axmat_destroy(&KQ); axmat_destroy(&Q); axpca_destroy(&pca); return pca;
    }
    for (int j = 0; j < k_eff; j++) {
        for (int i = 0; i < d; i++) x_f32[i] = (float)Q.data[(uint64_t)i * k_eff + j];
        K_apply(x_f32, y_f32, d, K_ctx);
        for (int i = 0; i < d; i++) KQ.data[(uint64_t)i * k_eff + j] = (double)y_f32[i];
    }
    tensor_free(x_f32); tensor_free(y_f32);
    axmat_t Qt = axmat_transpose(&Q);
    axmat_t B = axmat_mul(&Qt, &KQ);
    axmat_destroy(&Qt); axmat_destroy(&KQ);
    for (int i = 0; i < k_eff; i++)
        for (int j = i + 1; j < k_eff; j++) {
            double avg = 0.5 * (B.data[(uint64_t)i * k_eff + j] + B.data[(uint64_t)j * k_eff + i]);
            B.data[(uint64_t)i * k_eff + j] = avg; B.data[(uint64_t)j * k_eff + i] = avg;
        }
    double *evals = (double *)tensor_alloc((uint64_t)k_eff * sizeof(double));
    axmat_t U = axmat_create(k_eff, k_eff);
    if (!evals || !U.data) { if (evals) tensor_free(evals); axmat_destroy(&U);
        axmat_destroy(&B); axmat_destroy(&Q); axpca_destroy(&pca); return pca; }
    axmat_symeig(&B, evals, &U); axmat_destroy(&B);
    int keep = (k_max < k_eff) ? k_max : k_eff;
    axmat_t Vt = axmat_create(keep, d);
    if (!Vt.data) { tensor_free(evals); axmat_destroy(&U); axmat_destroy(&Q);
        axpca_destroy(&pca); return pca; }
    for (int i = 0; i < keep; i++)
        for (int j = 0; j < d; j++) {
            double val = 0.0;
            for (int l = 0; l < k_eff; l++) val += Q.data[(uint64_t)j * k_eff + l] * U.data[(uint64_t)l * k_eff + i];
            Vt.data[(uint64_t)i * d + j] = val;
        }
    axmat_destroy(&Q); axmat_destroy(&U);
    pca.components = Vt; pca.eigenvalues = evals; pca.n_components = keep;
    pca.explained_variance = 0.0;
    for (int i = 0; i < keep; i++) pca.explained_variance += evals[i];
    pca.mean_proj = (double *)tensor_alloc((uint64_t)keep * sizeof(double));
    if (pca.mean_proj) memset(pca.mean_proj, 0, (uint64_t)keep * sizeof(double));
    return pca;
}

void axpca_project(const axpca_t *pca, const double *x_full, double *x_sub)
{
    int d = pca->dim;
    int k = pca->n_components;
    for (int i = 0; i < k; i++) {
        const double *row = pca->components.data + (uint64_t)i * d;
        double mp = pca->mean_proj ? pca->mean_proj[i]
                                   : ax_vec_dot(row, pca->mean, d);
        x_sub[i] = ax_vec_dot(row, x_full, d) - mp;
    }
}

/* Float-input variant: avoids a separate f32→f64 widening pass.
 * Uses ax_f32_f64_dot (AVX2+FMA) to project a float embedding directly. */
void axpca_project_f32(const axpca_t *pca, const float *x_full_f32, double *x_sub)
{
    int d = pca->dim;
    int k = pca->n_components;
    for (int i = 0; i < k; i++) {
        const double *row = pca->components.data + (uint64_t)i * d;
        double mp = pca->mean_proj ? pca->mean_proj[i]
                                   : ax_vec_dot(row, pca->mean, d);
        x_sub[i] = ax_f32_f64_dot(x_full_f32, row, d) - mp;
    }
}

void axpca_reconstruct(const axpca_t *pca, const double *x_sub, double *x_full)
{
    int d = pca->dim;
    int k = pca->n_components;
    /* Loop order: k-outer, d-inner (row-major component access).
     * The original d-outer / k-inner layout caused stride-d column accesses
     * into the components matrix, thrashing cache when d >= 512. */
    memcpy(x_full, pca->mean, (uint64_t)d * sizeof(double));
    for (int i = 0; i < k; i++) {
        double c = x_sub[i];
        if (c == 0.0) continue;
        const double *row = pca->components.data + (uint64_t)i * d;
#if defined(__AVX2__) && defined(__FMA__)
        {
            __m256d vc = _mm256_set1_pd(c);
            int j = 0;
            for (; j + 4 <= d; j += 4) {
                __m256d vx = _mm256_loadu_pd(x_full + j);
                __m256d vr = _mm256_loadu_pd(row    + j);
                vx = _mm256_fmadd_pd(vr, vc, vx);
                _mm256_storeu_pd(x_full + j, vx);
            }
            for (; j < d; j++) x_full[j] += c * row[j];
        }
#else
        for (int j = 0; j < d; j++) x_full[j] += c * row[j];
#endif
    }
}

/* 
 * TwoNN Intrinsic Dimensionality Estimator
 *
 * Facco et al. "Estimating the intrinsic dimension of datasets by a
 * minimal neighborhood information" (2017).
 *
 * For each point, find the two nearest neighbors.  The ratio μ = r2/r1
 * follows a power law whose exponent is the intrinsic dimension.
 * ID = n / Σ ln(μ_i)
 *  */

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

/* 
 * Vector Operations
 *  */

double ax_vec_dot(const double *a, const double *b, int n)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        acc = _mm256_fmadd_pd(va, vb, acc);
    }
    /* Horizontal sum */
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double sum = _mm_cvtsd_f64(lo);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

double ax_vec_norm(const double *a, int n)
{
    return sqrt(ax_vec_dot(a, a, n));
}

/* Mixed-precision dot product: float row  double query.
 * Used in Phase 5 probe-pool scoring; AVX2 path converts float→double
 * in-register (256-bit = 4 doubles at a time, loaded from 4 floats). */
double ax_f32_f64_dot(const float *a, const double *b, int n)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        /* Load 4 floats and widen to 4 doubles */
        __m128 va_f = _mm_loadu_ps(a + i);
        __m256d va  = _mm256_cvtps_pd(va_f);
        __m256d vb  = _mm256_loadu_pd(b + i);
        acc = _mm256_fmadd_pd(va, vb, acc);
    }
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double sum = _mm_cvtsd_f64(lo);
    for (; i < n; i++) sum += (double)a[i] * b[i];
    return sum;
#else
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)a[i] * b[i];
    return sum;
#endif
}

/* Squared L2 norm of a float vector — returns sum(a[i]^2) in double precision.
 * Despite the float input, the accumulator is double (__m256d): each group of
 * 8 floats is split into two __m128 halves, widened to __m256d via cvtps_pd,
 * then accumulated with FMA.  This is identical in structure to ax_f32_f64_dot
 * and avoids the ~3 precision loss that a float accumulator would introduce
 * for large vectors (dim=5120, values ≈ 0.01 → sum ≈ 0.512, but relative error
 * from float accumulation over 640 additions per lane ≈ 810⁻⁵). */
double ax_f32_norm_sq(const float *a, int n)
{
#if defined(__AVX2__) && defined(__FMA__)
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(a + i);
        /* Widen lower 4 floats to double, FMA into accumulator */
        __m128 vlo = _mm256_castps256_ps128(v);
        __m256d vd_lo = _mm256_cvtps_pd(vlo);
        acc = _mm256_fmadd_pd(vd_lo, vd_lo, acc);
        /* Widen upper 4 floats to double, FMA into accumulator */
        __m128 vhi = _mm256_extractf128_ps(v, 1);
        __m256d vd_hi = _mm256_cvtps_pd(vhi);
        acc = _mm256_fmadd_pd(vd_hi, vd_hi, acc);
    }
    /* Horizontal sum of 4-wide double accumulator */
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    double sum = _mm_cvtsd_f64(lo);
    for (; i < n; i++) sum += (double)a[i] * (double)a[i];
    return sum;
#else
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)a[i] * (double)a[i];
    return sum;
#endif
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

/* 
 * Sorting
 *  */

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

/* 
 * GGUF Dequantization
 *
 * Convert quantized weight rows to double (for geometric analysis).
 *  */

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
    case GGML_TYPE_Q4_K: {
        /* Q4_K: 256 values per super-block, 144 bytes.
         * Layout: d(fp16) + dmin(fp16) + scales[12] + qs[128].
         * value = d * scale_j * nibble - dmin * min_j, 8 sub-blocks of 32 */
        const uint8_t *p = (const uint8_t *)src;
        int nsb = n / 256;
        for (int sb = 0; sb < nsb; sb++) {
            uint16_t dh, dmh;
            memcpy(&dh,  p,     2);
            memcpy(&dmh, p + 2, 2);
            float d    = ax_fp16_to_f32(dh);
            float dmin = ax_fp16_to_f32(dmh);
            const uint8_t *scales = p + 4;
            const uint8_t *qs     = p + 16;
            p += 144;

            float *o = dst + sb * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                if (is < 4) {
                    sc0 = scales[is] & 63;      m0 = scales[is + 4] & 63;
                } else {
                    sc0 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                    m0  = (scales[is + 4] >> 4)  | ((scales[is + 0] >> 6) << 4);
                }
                int is1 = is + 1;
                if (is1 < 4) {
                    sc1 = scales[is1] & 63;     m1 = scales[is1 + 4] & 63;
                } else {
                    sc1 = (scales[is1 + 4] & 0xF) | ((scales[is1 - 4] >> 6) << 4);
                    m1  = (scales[is1 + 4] >> 4)  | ((scales[is1 + 0] >> 6) << 4);
                }
                float d1 = d * (float)sc0;  float m1f = dmin * (float)m0;
                float d2 = d * (float)sc1;  float m2f = dmin * (float)m1;

                const uint8_t *qj = qs + (j / 2);
                for (int l = 0; l < 32; l++)
                    o[j + l]      = d1 * (float)(qj[l] & 0xF) - m1f;
                for (int l = 0; l < 32; l++)
                    o[j + 32 + l] = d2 * (float)(qj[l] >> 4)  - m2f;
                is += 2;
            }
        }
        return 0;
    }
    case GGML_TYPE_IQ4_NL: {
        /* IQ4_NL: 32 elements/block, 18 bytes. d(fp16) + qs[16]. */
        static const int8_t tbl[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
        };
        const uint8_t *p = (const uint8_t *)src;
        int nb = n / 32;
        for (int b = 0; b < nb; b++) {
            uint16_t dh;
            memcpy(&dh, p, 2);
            float d = ax_fp16_to_f32(dh);
            const uint8_t *qs = p + 2;
            float *o = dst + b * 32;
            for (int l = 0; l < 16; l++) {
                o[2*l + 0] = d * (float)tbl[qs[l] & 0xF];
                o[2*l + 1] = d * (float)tbl[qs[l] >> 4];
            }
            p += 18;
        }
        return 0;
    }
    case GGML_TYPE_IQ4_XS: {
        /* IQ4_XS: 256 elements/super-block, 136 bytes.
         * d(fp16) + scales_h(u16) + scales_l[4] + qs[128]. */
        static const int8_t tbl[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
        };
        const uint8_t *p = (const uint8_t *)src;
        int nsb = n / 256;
        for (int sb = 0; sb < nsb; sb++) {
            uint16_t dh, scales_h_u;
            memcpy(&dh,         p,     2);
            memcpy(&scales_h_u, p + 2, 2);
            float d = ax_fp16_to_f32(dh);
            const uint8_t *scales_l = p + 4;
            const uint8_t *qs       = p + 8;
            p += 136;
            float *o = dst + sb * 256;
            for (int ib = 0; ib < 8; ib++) {
                uint8_t ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF)
                             | (((scales_h_u >> (2 * ib)) & 3) << 4);
                float dl = d * (float)((int)ls - 32);
                const uint8_t *qb = qs + ib * 16;
                float *ob = o + ib * 32;
                for (int j = 0; j < 16; j++) {
                    ob[2*j + 0] = dl * (float)tbl[qb[j] & 0xF];
                    ob[2*j + 1] = dl * (float)tbl[qb[j] >> 4];
                }
            }
        }
        return 0;
    }
    case GGML_TYPE_Q2_K: {
        /* Q2_K: 256 elements/super-block, 84 bytes.
         * Layout: scales[16](nibble-packed) + qs[64] + d(fp16) + dmin(fp16).
         * Each scales[i]: low nibble = d_scale, high nibble = min_scale. */
        const uint8_t *p = (const uint8_t *)src;
        int nsb = n / 256;
        for (int sb = 0; sb < nsb; sb++) {
            const uint8_t *scales = p;
            const uint8_t *qs     = p + 16;
            uint16_t dh, dmh;
            memcpy(&dh,  p + 80, 2);
            memcpy(&dmh, p + 82, 2);
            float d    = ax_fp16_to_f32(dh);
            float dmin = ax_fp16_to_f32(dmh);
            p += 84;
            float *o = dst + sb * 256;
            int is = 0;
            const uint8_t *qb = qs;
            for (int half = 0; half < 2; half++) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0xF);
                    float ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        *o++ = dl * (float)((qb[l] >> shift) & 3) - ml;
                    sc = scales[is++];
                    dl = d * (float)(sc & 0xF);
                    ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        *o++ = dl * (float)((qb[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                qb += 32;
            }
        }
        return 0;
    }
    case GGML_TYPE_Q3_K: {
        /* Q3_K: 256 elements/super-block, 110 bytes.
         * Layout: hmask[32] + qs[64] + scales[12] + d(fp16). */
        const uint8_t *p = (const uint8_t *)src;
        int nsb = n / 256;
        for (int sb = 0; sb < nsb; sb++) {
            const uint8_t *hmask  = p;
            const uint8_t *qs     = p + 32;
            const uint8_t *scales = p + 96;
            uint16_t dh;
            memcpy(&dh, p + 108, 2);
            float d = ax_fp16_to_f32(dh);
            p += 110;
            float *o = dst + sb * 256;
            for (int sub = 0; sub < 8; sub++) {
                int is = sub;
                uint8_t sc;
                if (is < 4)
                    sc = (scales[is] & 0xF) | ((scales[is + 8] & 3) << 4);
                else
                    sc = (scales[is + 4] >> 4) | ((scales[is + 4] & 3) << 4);
                float d_sc = d * (float)((int)sc - 32);
                int base = sub * 32;
                for (int l = 0; l < 32; l++) {
                    int idx = base + l;
                    int qlo = (qs[idx >> 2] >> (2 * (idx & 3))) & 3;
                    int qhi = (hmask[idx >> 3] >> (idx & 7)) & 1;
                    o[idx] = d_sc * (float)((qlo | (qhi << 2)) - 4);
                }
            }
        }
        return 0;
    }
    case GGML_TYPE_Q5_K: {
        /* Q5_K: 256 elements/super-block, 176 bytes.
         * Layout: d(fp16) + dmin(fp16) + scales[12] + qh[32] + qs[128]. */
        const uint8_t *p = (const uint8_t *)src;
        int nsb = n / 256;
        for (int sb = 0; sb < nsb; sb++) {
            uint16_t dh, dmh;
            memcpy(&dh,  p,     2);
            memcpy(&dmh, p + 2, 2);
            float d    = ax_fp16_to_f32(dh);
            float dmin = ax_fp16_to_f32(dmh);
            const uint8_t *scales = p + 4;
            const uint8_t *qh     = p + 16;
            const uint8_t *qs     = p + 48;
            p += 176;
            float *o = dst + sb * 256;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t sc0, m0, sc1, m1;
                if (is < 4) {
                    sc0 = scales[is] & 63;      m0 = scales[is + 4] & 63;
                } else {
                    sc0 = (scales[is + 4] & 0xF) | ((scales[is - 4] >> 6) << 4);
                    m0  = (scales[is + 4] >> 4)  | ((scales[is + 0] >> 6) << 4);
                }
                int is1 = is + 1;
                if (is1 < 4) {
                    sc1 = scales[is1] & 63;     m1 = scales[is1 + 4] & 63;
                } else {
                    sc1 = (scales[is1 + 4] & 0xF) | ((scales[is1 - 4] >> 6) << 4);
                    m1  = (scales[is1 + 4] >> 4)  | ((scales[is1 + 0] >> 6) << 4);
                }
                float d1 = d * (float)sc0;  float m1f = dmin * (float)m0;
                float d2 = d * (float)sc1;  float m2f = dmin * (float)m1;
                const uint8_t *qj = qs + (j / 2);
                for (int l = 0; l < 32; l++) {
                    int pos = j + l;
                    int hi  = (qh[pos >> 3] >> ((pos >> 5) * 4 + (pos & 7) / 8)) & 1;
                    o[pos]  = d1 * (float)((qj[l] & 0xF) | (hi << 4)) - m1f;
                }
                for (int l = 0; l < 32; l++) {
                    int pos = j + 32 + l;
                    int hi  = (qh[pos >> 3] >> ((pos >> 5) * 4 + (pos & 7) / 8)) & 1;
                    o[pos]  = d2 * (float)((qj[l] >> 4) | (hi << 4)) - m2f;
                }
                is += 2;
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
