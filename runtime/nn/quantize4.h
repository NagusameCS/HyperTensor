/* =============================================================================
 * TensorOS - INT4 Block Quantization Engine (Q4_0 Format)
 *
 * Sub-byte quantization for maximum memory efficiency. Same technology
 * behind GGML/llama.cpp that makes LLM inference possible on consumer
 * hardware. We run it on bare metal with zero OS overhead.
 *
 * Format: Q4_0 (GGML-compatible block quantization)
 *   - Block size: 32 elements
 *   - Storage: 1 float32 scale + 16 bytes packed nibbles = 20 bytes/block
 *   - Compression: 6.4x vs FP32, 3.2x vs INT16
 *   - Range: [-8, 7] per nibble (4-bit signed via unsigned offset)
 *
 * SSE2 optimized: dequantize-and-dot using 4-wide SIMD accumulation.
 * Each dot product: unpack nibbles → float, multiply with input, accumulate.
 * =============================================================================*/

#ifndef TENSOROS_NN_QUANTIZE4_H
#define TENSOROS_NN_QUANTIZE4_H

#include <stdint.h>
#include "runtime/nn/inference.h"

/* Q4_0 block: 32 elements packed into 20 bytes.
 * Total compression: 128 bytes FP32 → 20 bytes Q4_0 = 6.4x
 * Encoding: q[i] = round(clamp(x[i] / scale, -8, 7)) + 8
 *           stored as nibble pairs, low nibble first per byte */
#define Q4_BLOCK_SIZE 32

typedef struct {
    float    scale;     /* absmax / 7.0f  */
    uint8_t  data[16];  /* 32 packed 4-bit values (2 per byte) */
} __attribute__((packed)) q4_block_t;

/* Quantized layer (Q4 weights, FP32 bias) */
typedef struct {
    q4_block_t *w_blocks;   /* Weight blocks: ceil(out_dim * in_dim / 32) */
    float      *bias;       /* FP32 bias (shared with original model) */
    int         in_dim;
    int         out_dim;
    int         activation; /* NN_ACT_* */
    int         num_blocks; /* Total weight blocks */
} q4_layer_t;

#define Q4_MAX_LAYERS 16

typedef struct {
    q4_layer_t layers[Q4_MAX_LAYERS];
    int num_layers;
    int max_dim;
} q4_model_t;

/* =============================================================================
 * API
 * =============================================================================*/

/* Quantize a single block of 32 floats to Q4_0 */
void q4_quantize_block(q4_block_t *block, const float *values);

/* Dequantize a Q4_0 block back to 32 floats */
void q4_dequantize_block(float *out, const q4_block_t *block);

/* SSE2-optimized dot product: Q4 block · FP32 vector (32 elements) */
float q4_dot_block(const q4_block_t *block, const float *x);

/* Q4 GEMV: multiply Q4 weight matrix by FP32 input vector */
void q4_gemv(float *out, const q4_block_t *weights, const float *input,
             int out_dim, int in_dim);

/* Reset the Q4 block pool (call before quantizing a new model) */
void q4_reset_pool(void);

/* Quantize an FP32 model to Q4_0. Returns 0 on success. */
int q4_quantize_model(q4_model_t *qm, const nn_model_t *fm);

/* Q4 forward pass: Q4 weights with FP32 activations */
void q4_forward(q4_model_t *model, float *output, const float *input);

/* Run INT4 quantization benchmarks and demos */
void q4_run_demos(void);

#endif /* TENSOROS_NN_QUANTIZE4_H */
