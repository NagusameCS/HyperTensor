/* distributed.h - Distributed Training (Multi-Node) */
#ifndef DISTRIBUTED_H
#define DISTRIBUTED_H

#include <stdint.h>

/* Distributed operation types */
typedef enum {
    DIST_OP_ALLREDUCE_SUM  = 0,
    DIST_OP_ALLREDUCE_AVG  = 1,
    DIST_OP_BROADCAST      = 2,
    DIST_OP_SCATTER        = 3,
    DIST_OP_GATHER         = 4,
    DIST_OP_ALLGATHER      = 5,
    DIST_OP_REDUCE_SCATTER = 6,
} dist_op_t;

/* Node info */
typedef struct {
    uint32_t ip;
    uint16_t port;
    int      rank;
    int      alive;           /* Heartbeat status */
    uint64_t last_heartbeat;
} dist_node_t;

#define DIST_MAX_NODES  32

/* Distributed context */
typedef struct {
    dist_node_t nodes[DIST_MAX_NODES];
    int         n_nodes;
    int         my_rank;
    int         world_size;
    uint16_t    base_port;
    int         initialized;

    /* Ring topology for ring-allreduce */
    int         ring_left;    /* Rank of left neighbor */
    int         ring_right;   /* Rank of right neighbor */

    /* Stats */
    uint64_t    bytes_sent;
    uint64_t    bytes_received;
    uint64_t    allreduce_count;
} dist_ctx_t;

int  dist_init(dist_ctx_t *ctx, int rank, int world_size, uint16_t base_port);
int  dist_add_node(dist_ctx_t *ctx, uint32_t ip, uint16_t port, int rank);
int  dist_allreduce(dist_ctx_t *ctx, float *data, int count, dist_op_t op);
int  dist_broadcast(dist_ctx_t *ctx, float *data, int count, int root);
int  dist_barrier(dist_ctx_t *ctx);
int  dist_scatter(dist_ctx_t *ctx, const float *send, float *recv, int count, int root);
int  dist_gather(dist_ctx_t *ctx, const float *send, float *recv, int count, int root);

/* High-level training API */
int  dist_sync_gradients(dist_ctx_t *ctx, float *gradients, int count);
int  dist_sync_model(dist_ctx_t *ctx, float *params, int count, int root);
void dist_print_stats(const dist_ctx_t *ctx);

#endif /* DISTRIBUTED_H */
