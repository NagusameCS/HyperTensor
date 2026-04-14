/*
 * Geodessical — Axiomatic Linear Algebra Primitives
 *
 * Small-to-medium dense matrix operations for geometric analysis of neural
 * manifolds.  All matrices are row-major, heap-allocated via hal_alloc.
 * Designed for subspace dimensions (d ≤ 512) — not for full hidden-dim
 * matmuls; use the inference engine for those.
 *
 * Provides:
 *   - Dense matrix create/destroy/multiply/transpose
 *   - Symmetric eigenvalue decomposition (Jacobi)
 *   - PCA via covariance eigendecomposition
 *   - TwoNN intrinsic dimensionality estimator
 *   - Basic vector operations
 *   - Dequantization of GGUF quantized rows
 */

#ifndef GEODESSICAL_AXIOM_LINALG_H
#define GEODESSICAL_AXIOM_LINALG_H

#include <stdint.h>
#include "runtime/nn/gguf.h"

/* ─── Dense matrix (row-major, heap) ─── */
typedef struct {
    double *data;   /* row-major [rows * cols] */
    int     rows;
    int     cols;
} axmat_t;

/* Create/destroy */
axmat_t  axmat_create(int rows, int cols);
void     axmat_destroy(axmat_t *m);
void     axmat_zero(axmat_t *m);
void     axmat_identity(axmat_t *m);  /* must be square */
axmat_t  axmat_clone(const axmat_t *m);

/* Element access */
static inline double axmat_get(const axmat_t *m, int r, int c) {
    return m->data[r * m->cols + c];
}
static inline void axmat_set(axmat_t *m, int r, int c, double v) {
    m->data[r * m->cols + c] = v;
}

/* Arithmetic */
axmat_t  axmat_mul(const axmat_t *A, const axmat_t *B);        /* C = A × B */
axmat_t  axmat_transpose(const axmat_t *A);                    /* A^T */
void     axmat_add_inplace(axmat_t *A, const axmat_t *B);      /* A += B */
void     axmat_scale_inplace(axmat_t *A, double s);            /* A *= s */
void     axmat_add_outer(axmat_t *A, const double *u,
                         const double *v, int n, double alpha); /* A += alpha * u v^T */

/* ─── Symmetric eigenvalue decomposition ─── */

/*
 * Jacobi eigenvalue decomposition for real symmetric matrices.
 * Input:  A (n×n symmetric, overwritten with eigenvalues on diagonal)
 * Output: eigenvalues[n] (descending), eigenvectors (n×n, columns are eigvecs)
 * Returns 0 on success.
 */
int axmat_symeig(const axmat_t *A, double *eigenvalues,
                 axmat_t *eigenvectors);

/* ─── PCA ─── */

/*
 * Compute PCA from a sample matrix X (n_samples × dim).
 * Returns the number of principal components with eigenvalue > threshold.
 *
 * Outputs:
 *   eigenvalues[min(n_samples,dim)]  — sorted descending
 *   components (min(n_samples,dim) × dim) — principal component rows
 *   mean[dim] — sample mean vector
 */
typedef struct {
    double *eigenvalues;   /* [n_components] */
    axmat_t components;    /* [n_components × dim] */
    double *mean;          /* [dim] */
    int     n_components;
    int     dim;
    double  total_variance;
    double  explained_variance; /* sum of kept eigenvalues */
} axpca_t;

axpca_t axpca_compute(const axmat_t *X, double min_explained_ratio);
void    axpca_destroy(axpca_t *pca);

/* Project a vector from full space to PCA subspace. */
void axpca_project(const axpca_t *pca, const double *x_full, double *x_sub);

/* Reconstruct from subspace to full space. */
void axpca_reconstruct(const axpca_t *pca, const double *x_sub, double *x_full);

/* ─── TwoNN intrinsic dimensionality estimator ─── */

/*
 * Facco et al. (2017) Two-Nearest-Neighbor estimator.
 * Input:  X (n_samples × dim)
 * Returns estimated intrinsic dimensionality.
 */
double ax_twonn_id(const axmat_t *X);

/* ─── Vector operations ─── */

double ax_vec_dot(const double *a, const double *b, int n);
double ax_vec_norm(const double *a, int n);
void   ax_vec_scale(double *dst, const double *src, double s, int n);
void   ax_vec_add(double *dst, const double *a, const double *b, int n);
void   ax_vec_sub(double *dst, const double *a, const double *b, int n);
void   ax_vec_zero(double *dst, int n);
void   ax_vec_copy(double *dst, const double *src, int n);

/* ─── GGUF dequantization ─── */

/*
 * Dequantize one row of a quantized weight tensor to float64.
 * Handles F32, F16, BF16, Q4_0, Q8_0, Q6_K.
 * Returns 0 on success, -1 on unsupported type.
 */
int ax_dequant_row(const void *src, double *dst, int n, ggml_type_t type);

/*
 * Dequantize one row to float32 (faster, for bulk operations).
 */
int ax_dequant_row_f32(const void *src, float *dst, int n, ggml_type_t type);

/* ─── Sorting ─── */

void ax_sort_ascending(double *arr, int n);
void ax_argsort_ascending(const double *arr, int *indices, int n);

#endif /* GEODESSICAL_AXIOM_LINALG_H */
