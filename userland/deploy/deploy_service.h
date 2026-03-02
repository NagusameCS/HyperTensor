/* =============================================================================
 * TensorOS - Model Deployment Service
 * =============================================================================
 * Provides:
 *   - One-command model serving (HTTP-like tensor endpoint)
 *   - Auto-scaling: duplicate MEUs under load
 *   - Health checks and automatic restart on MEU failure
 *   - Request batching for throughput optimization
 *   - A/B testing with traffic splitting
 *   - Canary deployments with automatic rollback
 * =============================================================================*/

#ifndef DEPLOY_SERVICE_H
#define DEPLOY_SERVICE_H

#include "kernel/core/kernel.h"
#include "kernel/sched/tensor_sched.h"

#define DEPLOY_MAX_SERVICES    32
#define DEPLOY_MAX_REPLICAS    16
#define DEPLOY_MAX_QUEUE       256

typedef enum {
    DEPLOY_STATE_STOPPED = 0,
    DEPLOY_STATE_STARTING,
    DEPLOY_STATE_RUNNING,
    DEPLOY_STATE_SCALING,
    DEPLOY_STATE_DRAINING,
    DEPLOY_STATE_FAILED,
} deploy_state_t;

typedef struct {
    uint32_t meu_id;
    uint64_t requests_served;
    uint64_t total_latency_us;
    float    avg_latency_ms;
    bool     healthy;
} deploy_replica_t;

typedef struct {
    /* Request queue (ring buffer) */
    struct {
        tensor_desc_t input;
        tensor_desc_t output;
        uint64_t      submitted_at;
        uint64_t      completed_at;
        bool          completed;
    } queue[DEPLOY_MAX_QUEUE];
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_count;
} deploy_request_queue_t;

typedef struct {
    char             name[64];
    char             model_name[64];
    uint16_t         port;
    deploy_state_t   state;

    /* Replicas */
    deploy_replica_t replicas[DEPLOY_MAX_REPLICAS];
    uint32_t         replica_count;
    uint32_t         target_replicas;

    /* Auto-scaling */
    bool             autoscale_enabled;
    uint32_t         min_replicas;
    uint32_t         max_replicas;
    float            scale_up_threshold;   /* avg latency ms */
    float            scale_down_threshold;
    uint32_t         scale_cooldown_ticks;
    uint64_t         last_scale_tick;

    /* Statistics */
    uint64_t         total_requests;
    uint64_t         total_errors;
    float            p50_latency_ms;
    float            p99_latency_ms;
    uint64_t         uptime_ticks;

    /* Request batching */
    uint32_t         batch_size;
    uint32_t         batch_timeout_ms;

    /* A/B testing */
    bool             ab_enabled;
    char             ab_model_b[64];
    float            ab_traffic_split; /* 0.0-1.0, fraction to model B */

    deploy_request_queue_t requests;
} deploy_service_t;

/* API */
int  deploy_init(void);
int  deploy_create(const char *name, const char *model, uint16_t port);
int  deploy_start(const char *name);
int  deploy_stop(const char *name);
int  deploy_scale(const char *name, uint32_t replicas);
int  deploy_enable_autoscale(const char *name, uint32_t min, uint32_t max);
int  deploy_submit_request(const char *name, const tensor_desc_t *input,
                            tensor_desc_t *output);
void deploy_health_check(void);
void deploy_print_status(void);
void deploy_daemon_main(void);

#endif /* DEPLOY_SERVICE_H */
