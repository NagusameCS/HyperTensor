#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*jit_gemm_f32_t)(const float *A, const float *B, float *C, int M, int N, int K);

jit_gemm_f32_t jit_gemm_f32_create(int M, int N, int K);
void jit_gemm_f32_destroy(jit_gemm_f32_t kernel);

#ifdef __cplusplus
}
#endif
