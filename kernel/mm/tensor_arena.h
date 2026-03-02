/* =============================================================================
 * TensorOS - Tensor Memory Arena (Zero-Fragmentation Bump Allocator)
 *
 * AI workloads have a unique memory pattern: they allocate many temporary
 * tensors of varying sizes during inference, then free them all at once.
 * Traditional malloc/free causes fragmentation; a bump allocator is ideal.
 *
 * The arena provides:
 *   - O(1) allocation (just bump a pointer)
 *   - Zero fragmentation (contiguous memory)
 *   - Checkpoint/restore (save/rollback allocation state)
 *   - Alignment guaranteed (16-byte for SSE2 SIMD)
 *   - Zero-overhead deallocation (reset the entire arena)
 *
 * This is the same strategy used by ONNX Runtime, TensorFlow's arena
 * allocator, and WASM linear memory — but at the OS kernel level.
 * =============================================================================*/

#ifndef TENSOROS_TENSOR_ARENA_H
#define TENSOROS_TENSOR_ARENA_H

#include <stdint.h>

/* Arena configuration */
#define ARENA_SIZE       (2 * 1024 * 1024)  /* 2 MB main arena */
#define ARENA_ALIGN      16                  /* SSE2 alignment */
#define ARENA_MAX_CHECKPOINTS 16

/* Arena handle. Usage:
 *   arena_reset(&arena);
 *   float *a = arena_alloc(&arena, 1024 * sizeof(float));
 *   float *b = arena_alloc(&arena, 512 * sizeof(float));
 *   // ... compute ...
 *   arena_reset(&arena);  // free everything instantly */
typedef struct {
    uint8_t  *base;          /* Base of arena memory */
    uint64_t  size;          /* Total arena size */
    uint64_t  used;          /* Current offset (bumped on alloc) */
    uint64_t  peak;          /* High-water mark */
    uint64_t  alloc_count;   /* Number of allocations since last reset */
    uint64_t  total_allocs;  /* Lifetime allocation count */
    uint64_t  total_resets;  /* Lifetime reset count */
    /* Checkpoint stack for nested scopes */
    uint64_t  checkpoints[ARENA_MAX_CHECKPOINTS];
    int       checkpoint_depth;
} tensor_arena_t;

/* =============================================================================
 * API
 * =============================================================================*/

/* Initialize the arena with its static backing store */
void arena_init(tensor_arena_t *arena);

/* Allocate `size` bytes (16-byte aligned). Returns NULL if full. */
void *arena_alloc(tensor_arena_t *arena, uint64_t size);

/* Reset the arena: free all allocations in O(1) */
void arena_reset(tensor_arena_t *arena);

/* Save current allocation state (checkpoint).
 * All allocations after this can be rolled back with arena_restore(). */
int arena_checkpoint(tensor_arena_t *arena);

/* Restore to the most recent checkpoint. Returns 0 on success. */
int arena_restore(tensor_arena_t *arena);

/* Get arena statistics */
uint64_t arena_used(const tensor_arena_t *arena);
uint64_t arena_peak(const tensor_arena_t *arena);
uint64_t arena_remaining(const tensor_arena_t *arena);

/* Run arena demos and benchmarks */
void arena_run_demos(void);

#endif /* TENSOROS_TENSOR_ARENA_H */
