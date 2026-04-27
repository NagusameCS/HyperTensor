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

/* =============================================================================
 * TensorOS - Distributed Training Engine
 *
 * Multi-node gradient synchronization using ring-allreduce:
 *
 * Ring-AllReduce Algorithm:
 *   Given N nodes each holding a gradient vector of size S:
 *   1. SCATTER-REDUCE: Each node sends 1/N of its data to the right neighbor
 *      and receives + accumulates 1/N from the left. After N-1 steps, each
 *      node holds the fully-reduced version of its 1/N chunk.
 *   2. ALL-GATHER: Each node sends its reduced chunk right and receives the
 *      next chunk from left. After N-1 steps, all nodes have the full result.
 *
 *   Total data transferred per node: 2 * S * (N-1)/N  (bandwidth-optimal)
 *
 * Network: Uses TensorOS netstack UDP for inter-node communication.
 *          Each node listens on base_port + rank.
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/core/perf.h"
#include "kernel/net/distributed.h"

/* External netstack UDP API */
extern int netstack_send_udp(const uint8_t dst_ip[4], uint16_t src_port,
                             uint16_t dst_port, const void *data, uint32_t len);

/* Receive buffer — filled by the netstack UDP callback */
#define DIST_RX_BUFSZ  2048
static volatile uint8_t g_dist_rx_buf[DIST_RX_BUFSZ];
static volatile int     g_dist_rx_len;
static volatile int     g_dist_rx_ready;

static void dist_udp_callback(const uint8_t src_ip[4], uint16_t src_port,
                               const uint8_t *data, uint32_t len)
{
    (void)src_ip; (void)src_port;
    if (len > DIST_RX_BUFSZ) len = DIST_RX_BUFSZ;
    kmemcpy((void *)g_dist_rx_buf, data, len);
    g_dist_rx_len = (int)len;
    g_dist_rx_ready = 1;
}

/* Register our UDP handler — called from dist_init */
typedef void (*udp_handler_t)(const uint8_t src_ip[4], uint16_t src_port,
                               const uint8_t *data, uint32_t len);
extern void netstack_register_udp(uint16_t port, udp_handler_t handler);

/* =============================================================================
 * Initialization
 * =============================================================================*/

int dist_init(dist_ctx_t *ctx, int rank, int world_size, uint16_t base_port)
{
    if (!ctx || rank < 0 || rank >= world_size || world_size > DIST_MAX_NODES) return -1;

    kmemset(ctx, 0, sizeof(*ctx));
    ctx->my_rank    = rank;
    ctx->world_size = world_size;
    ctx->base_port  = base_port;
    ctx->n_nodes    = 0;

    /* Set up ring topology */
    ctx->ring_left  = (rank - 1 + world_size) % world_size;
    ctx->ring_right = (rank + 1) % world_size;

    ctx->initialized = 1;

    /* Register UDP listener for receiving distributed messages */
    netstack_register_udp(base_port + rank, dist_udp_callback);

    kprintf("[DIST] Initialized: rank=%d/%d, ring: left=%d right=%d, port=%d\n",
            rank, world_size, ctx->ring_left, ctx->ring_right, base_port + rank);
    return 0;
}

int dist_add_node(dist_ctx_t *ctx, uint32_t ip, uint16_t port, int rank)
{
    if (!ctx || ctx->n_nodes >= DIST_MAX_NODES || rank < 0) return -1;

    dist_node_t *n = &ctx->nodes[ctx->n_nodes++];
    n->ip    = ip;
    n->port  = port;
    n->rank  = rank;
    n->alive = 1;
    return 0;
}

/* =============================================================================
 * Ring-AllReduce Implementation
 *
 * Phase 1 (Scatter-Reduce): N-1 rounds
 *   Round r: send chunk[(rank-r) % N] to right, recv chunk[(rank-r-1) % N]
 *            from left, accumulate received into local buffer
 *
 * Phase 2 (AllGather): N-1 rounds
 *   Round r: send chunk[(rank-r+1) % N] to right, recv chunk[(rank-r) % N]
 *            from left, overwrite local with received
 * =============================================================================*/

/* Max chunk size per UDP packet (stay under MTU) */
#define DIST_CHUNK_MTU  1400

/* Message header for distributed ops */
typedef struct {
    uint32_t magic;      /* 0xD15T0001 */
    uint32_t op;
    int32_t  src_rank;
    int32_t  round;
    int32_t  chunk_idx;
    int32_t  n_floats;
} dist_msg_hdr_t;

#define DIST_MAGIC 0xD1500001

static int send_chunk(dist_ctx_t *ctx, int dst_rank, int round, int chunk_idx,
                      const float *data, int n_floats)
{
    if (dst_rank < 0 || dst_rank >= ctx->n_nodes) return -1;
    dist_node_t *dst = NULL;
    for (int i = 0; i < ctx->n_nodes; i++) {
        if (ctx->nodes[i].rank == dst_rank) { dst = &ctx->nodes[i]; break; }
    }
    if (!dst) return -1;

    /* Send in MTU-sized packets */
    int offset = 0;
    while (offset < n_floats) {
        int this_chunk = n_floats - offset;
        int max_floats = (DIST_CHUNK_MTU - (int)sizeof(dist_msg_hdr_t)) / (int)sizeof(float);
        if (this_chunk > max_floats) this_chunk = max_floats;

        uint8_t pkt[DIST_CHUNK_MTU];
        dist_msg_hdr_t *hdr = (dist_msg_hdr_t *)pkt;
        hdr->magic     = DIST_MAGIC;
        hdr->op        = DIST_OP_ALLREDUCE_SUM;
        hdr->src_rank  = ctx->my_rank;
        hdr->round     = round;
        hdr->chunk_idx = chunk_idx;
        hdr->n_floats  = this_chunk;
        kmemcpy(pkt + sizeof(dist_msg_hdr_t), data + offset, this_chunk * sizeof(float));

        uint16_t pkt_len = (uint16_t)(sizeof(dist_msg_hdr_t) + this_chunk * sizeof(float));
        /* Convert IP to byte array for netstack */
        uint8_t dst_ip[4] = {
            (uint8_t)(dst->ip >> 24), (uint8_t)(dst->ip >> 16),
            (uint8_t)(dst->ip >> 8),  (uint8_t)(dst->ip)
        };
        netstack_send_udp(dst_ip, ctx->base_port + ctx->my_rank, dst->port, pkt, pkt_len);

        ctx->bytes_sent += pkt_len;
        offset += this_chunk;
    }
    return 0;
}

static int recv_chunk(dist_ctx_t *ctx, float *data, int max_floats,
                      int *out_round, int *out_chunk, int *out_src)
{
    (void)ctx;
    /* Spin-wait for the UDP callback to deliver a packet */
    uint64_t deadline = rdtsc() + 5000000000ULL; /* ~few seconds */
    while (!g_dist_rx_ready) {
        __asm__ volatile("pause");
        if (rdtsc() > deadline) return -1;
    }
    g_dist_rx_ready = 0;

    int len = g_dist_rx_len;
    if (len < (int)sizeof(dist_msg_hdr_t)) return -1;

    dist_msg_hdr_t *hdr = (dist_msg_hdr_t *)(void *)g_dist_rx_buf;
    if (hdr->magic != DIST_MAGIC) return -1;

    int n = hdr->n_floats;
    if (n > max_floats) n = max_floats;
    kmemcpy(data, (const uint8_t *)g_dist_rx_buf + sizeof(dist_msg_hdr_t), n * sizeof(float));

    *out_round = hdr->round;
    *out_chunk = hdr->chunk_idx;
    *out_src   = hdr->src_rank;
    return n;
}

int dist_allreduce(dist_ctx_t *ctx, float *data, int count, dist_op_t op)
{
    if (!ctx || !ctx->initialized || !data || count <= 0) return -1;
    if (ctx->world_size == 1) return 0;  /* Single node — nothing to do */

    int N = ctx->world_size;
    int chunk_size = (count + N - 1) / N;

    /* Allocate receive buffer */
    float *recv_buf = (float *)kmalloc(chunk_size * sizeof(float));
    if (!recv_buf) return -1;

    uint64_t start = rdtsc();

    /* Phase 1: Scatter-Reduce */
    for (int r = 0; r < N - 1; r++) {
        int send_chunk_idx = (ctx->my_rank - r + N) % N;
        int recv_chunk_idx = (ctx->my_rank - r - 1 + N) % N;

        /* Calculate offsets */
        int send_off = send_chunk_idx * chunk_size;
        int recv_off = recv_chunk_idx * chunk_size;
        int send_len = chunk_size;
        int recv_len = chunk_size;
        if (send_off + send_len > count) send_len = count - send_off;
        if (recv_off + recv_len > count) recv_len = count - recv_off;
        if (send_len <= 0 || recv_len <= 0) continue;

        /* Send our chunk to right neighbor */
        send_chunk(ctx, ctx->ring_right, r, send_chunk_idx,
                   data + send_off, send_len);

        /* Receive chunk from left neighbor and accumulate */
        int got_round, got_chunk, got_src;
        int got = recv_chunk(ctx, recv_buf, recv_len, &got_round, &got_chunk, &got_src);
        if (got > 0) {
            for (int i = 0; i < got && (recv_off + i) < count; i++) {
                if (op == DIST_OP_ALLREDUCE_SUM || op == DIST_OP_ALLREDUCE_AVG)
                    data[recv_off + i] += recv_buf[i];
            }
        }
    }

    /* Phase 2: AllGather */
    for (int r = 0; r < N - 1; r++) {
        int send_chunk_idx = (ctx->my_rank - r + 1 + N) % N;
        int recv_chunk_idx = (ctx->my_rank - r + N) % N;

        int send_off = send_chunk_idx * chunk_size;
        int recv_off = recv_chunk_idx * chunk_size;
        int send_len = chunk_size;
        int recv_len = chunk_size;
        if (send_off + send_len > count) send_len = count - send_off;
        if (recv_off + recv_len > count) recv_len = count - recv_off;
        if (send_len <= 0 || recv_len <= 0) continue;

        send_chunk(ctx, ctx->ring_right, r + N, send_chunk_idx,
                   data + send_off, send_len);

        int got_round, got_chunk, got_src;
        int got = recv_chunk(ctx, recv_buf, recv_len, &got_round, &got_chunk, &got_src);
        if (got > 0) {
            kmemcpy(data + recv_off, recv_buf, got * sizeof(float));
        }
    }

    /* Average if requested */
    if (op == DIST_OP_ALLREDUCE_AVG) {
        float inv_n = 1.0f / (float)N;
        for (int i = 0; i < count; i++) data[i] *= inv_n;
    }

    uint64_t elapsed = rdtsc() - start;
    ctx->allreduce_count++;
    kfree(recv_buf);

    kprintf("[DIST] AllReduce: %d floats across %d nodes in %lu cycles\n",
            count, N, (unsigned long)elapsed);
    return 0;
}

/* =============================================================================
 * Broadcast: root sends data to all other nodes
 * =============================================================================*/

int dist_broadcast(dist_ctx_t *ctx, float *data, int count, int root)
{
    if (!ctx || !ctx->initialized || !data) return -1;
    if (ctx->world_size == 1) return 0;

    if (ctx->my_rank == root) {
        /* Send to all other nodes */
        for (int r = 0; r < ctx->world_size; r++) {
            if (r == root) continue;
            send_chunk(ctx, r, 0, 0, data, count);
        }
    } else {
        /* Receive from root */
        int got_round, got_chunk, got_src;
        recv_chunk(ctx, data, count, &got_round, &got_chunk, &got_src);
    }
    return 0;
}

/* =============================================================================
 * Barrier: synchronize all nodes
 * =============================================================================*/

int dist_barrier(dist_ctx_t *ctx)
{
    if (!ctx || !ctx->initialized) return -1;
    /* Simple barrier: allreduce a single float */
    float dummy = 1.0f;
    return dist_allreduce(ctx, &dummy, 1, DIST_OP_ALLREDUCE_SUM);
}

/* =============================================================================
 * High-Level Training API
 * =============================================================================*/

int dist_sync_gradients(dist_ctx_t *ctx, float *gradients, int count)
{
    /* Average gradients across all nodes (standard data-parallel training) */
    return dist_allreduce(ctx, gradients, count, DIST_OP_ALLREDUCE_AVG);
}

int dist_sync_model(dist_ctx_t *ctx, float *params, int count, int root)
{
    return dist_broadcast(ctx, params, count, root);
}

int dist_scatter(dist_ctx_t *ctx, const float *send, float *recv, int count, int root)
{
    if (!ctx || !ctx->initialized) return -1;
    int chunk = count / ctx->world_size;

    if (ctx->my_rank == root) {
        for (int r = 0; r < ctx->world_size; r++) {
            if (r == root) {
                kmemcpy(recv, send + r * chunk, chunk * sizeof(float));
            } else {
                send_chunk(ctx, r, 0, r, send + r * chunk, chunk);
            }
        }
    } else {
        int got_round, got_chunk, got_src;
        recv_chunk(ctx, recv, chunk, &got_round, &got_chunk, &got_src);
    }
    return 0;
}

int dist_gather(dist_ctx_t *ctx, const float *send, float *recv, int count, int root)
{
    if (!ctx || !ctx->initialized) return -1;
    int chunk = count / ctx->world_size;

    if (ctx->my_rank == root) {
        kmemcpy(recv + root * chunk, send, chunk * sizeof(float));
        for (int r = 0; r < ctx->world_size; r++) {
            if (r == root) continue;
            int got_round, got_chunk, got_src;
            recv_chunk(ctx, recv + r * chunk, chunk, &got_round, &got_chunk, &got_src);
        }
    } else {
        send_chunk(ctx, root, 0, ctx->my_rank, send, chunk);
    }
    return 0;
}

void dist_print_stats(const dist_ctx_t *ctx)
{
    if (!ctx) return;
    kprintf("[DIST] Stats: rank=%d/%d, allreduce_count=%lu\n",
            ctx->my_rank, ctx->world_size, (unsigned long)ctx->allreduce_count);
    kprintf("[DIST]   sent=%lu bytes, received=%lu bytes\n",
            (unsigned long)ctx->bytes_sent, (unsigned long)ctx->bytes_received);
}
