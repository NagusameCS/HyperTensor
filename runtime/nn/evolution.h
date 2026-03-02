/* =============================================================================
 * TensorOS - Neural Architecture Search via Neuroevolution
 *
 * The most revolutionary feature: the OS discovers optimal neural network
 * architectures for its own hardware during boot. A population of candidate
 * networks competes for fitness (accuracy × speed), with the best surviving
 * and reproducing via mutation. Within seconds, evolution discovers networks
 * that solve problems humans designed by hand.
 *
 * This is what Google Brain's NAS and OpenAI's evolution strategies do —
 * but running at the OS kernel level, with JIT-compiled candidates.
 * =============================================================================*/

#ifndef TENSOROS_NN_EVOLUTION_H
#define TENSOROS_NN_EVOLUTION_H

#include <stdint.h>

/* Evolution parameters */
#define EVO_MAX_LAYERS   4    /* Max hidden layers per genome */
#define EVO_POP_SIZE     16   /* Population size */
#define EVO_MAX_WEIGHTS  2048 /* Max weights per genome */
#define EVO_ELITES       4    /* Top survivors per generation */

/* A genome encodes both architecture and weights */
typedef struct {
    int num_layers;                         /* Number of layers (1-3) */
    int dims[EVO_MAX_LAYERS + 2];           /* [input, hidden..., output] */
    int activations[EVO_MAX_LAYERS + 1];    /* Per-layer activation */
    int num_weights;                        /* Actual weight count */
    float fitness;                          /* Fitness score */
    float accuracy;                         /* XOR accuracy (0-1) */
    float mse;                              /* Mean squared error */
    uint64_t latency_ns;                    /* Inference latency */
    float weights[EVO_MAX_WEIGHTS] __attribute__((aligned(16))); /* 16-byte aligned for SSE2 */
} evo_genome_t;

/* =============================================================================
 * API
 * =============================================================================*/

/* Run neuroevolution demo: evolve XOR solver from scratch */
void nn_evolve_demos(void);

#endif /* TENSOROS_NN_EVOLUTION_H */
