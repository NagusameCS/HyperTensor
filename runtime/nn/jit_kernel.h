#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*jit_gemm_f64_t)(const double *A, const double *B, double *C, int M, int N, int K);

// Returns a JIT-compiled kernel for the given (M,N,K), or NULL if not supported.
jit_gemm_f64_t jit_gemm_f64_create(int M, int N, int K);
// Frees a JIT kernel.
void jit_gemm_f64_destroy(jit_gemm_f64_t kernel);

#ifdef __cplusplus
}
#endif
